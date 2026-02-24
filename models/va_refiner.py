"""
models/va_refiner.py
====================
Visual-Absence (VA) Refiner for EmberNet.

Detects visually ungrounded (hallucinated) tokens during autoregressive
decoding and suppresses them in the logit space before sampling.

Key components
--------------
VARefinerConfig  – lightweight dataclass with all knobs
VATemporalMemory – sliding-window burst detector
VARefiner        – main class: hooks, neuron detector, logit discrepancy,
                   refine_logits(), calibration prefix
is_visual_sensitive_token  – token-type classifier
calibrate_answer_prefix    – post-generation uncertainty prefixer

Architecture notes
------------------
Neuron-based signal:
  Hooks are placed on the ``down_proj`` layer inside ``shared_expert`` for
  each monitored decoder layer (default: layers 6-11).  The hook captures
  the intermediate SwiGLU hidden state (shape [tokens, intermediate_size])
  just before down-projection, giving us the richest per-token feature.
  A small 2-layer MLP (total_neurons → 64 → 1, sigmoid) scores each token.

Logit-discrepancy signal:
  At the START of every generate() call, one extra decoder forward pass is
  performed with the visual token positions zeroed out.  This yields a
  ``null_baseline`` logit vector for the last context position.  At each
  decoding step the L1 distance between the live logits and this baseline is
  converted to a [0,1] score via  p = 1 - tanh(dist / scale).
  If the model "doesn't care" about the image, the two passes produce
  similar logits → high distance → high p_VA. Wait, let me think again:
  if model relies on image, the change when masking image = large distance
  → the model IS visually grounded → p_VA should be LOW.
  So: large distance = grounded = low p_VA.
  p_VA_logit = 1 - tanh(dist / scale)   ← correct direction.

Combined score:
  p_VA = alpha * p_neuron + (1 - alpha) * p_VA_logit

Intervention:
  normal: subtract va_soft_penalty from the top flagged candidate logit
  burst : decay whole distribution + hard-block top candidate

All of this is strictly inference-time. Hooks are cleaned up after each
generate() call. No gradient or optimizer interaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Visual keyword set for token-type-aware thresholding
# ---------------------------------------------------------------------------
VISUAL_KEYWORDS: frozenset = frozenset({
    # Colours
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "black", "white", "gray", "grey", "cyan", "magenta", "teal", "violet",
    "gold", "silver", "beige", "maroon", "navy", "olive", "coral", "salmon",
    # Objects (common VQA targets)
    "dog", "cat", "car", "bus", "truck", "bicycle", "person", "man", "woman",
    "child", "boy", "girl", "tree", "flower", "building", "house", "bridge",
    "road", "sky", "cloud", "mountain", "water", "ocean", "lake", "river",
    "table", "chair", "cup", "bottle", "book", "phone", "keyboard", "ball",
    "ball", "bird", "fish", "horse", "cow", "elephant", "bear", "zebra",
    "pizza", "cake", "fruit", "apple", "banana", "orange", "bowl", "plate",
    "traffic", "sign", "window", "door", "wall", "floor", "ceiling", "lamp",
    # Spatial / positional terms
    "left", "right", "above", "below", "top", "bottom", "front", "back",
    "behind", "next", "beside", "between", "near", "far", "inside", "outside",
    "center", "middle", "corner", "edge", "side",
    # Numbers / quantities (as words)
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "zero", "first", "second", "third", "several", "many", "few",
    # Counting / size
    "large", "small", "big", "tiny", "tall", "short", "wide", "narrow",
    "long", "thick", "thin",
})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class VARefinerConfig:
    """All knobs for the VA Refiner.  Defaults are conservative / off."""

    use_va_refiner: bool = False

    # Combination weight: alpha * neuron + (1-alpha) * logit
    va_alpha: float = 0.5

    # Base thresholds (per token type)
    va_tau_visual: float = 0.60      # stricter for colours/objects/numbers
    va_tau_non_visual: float = 0.75  # looser for generic tokens

    # Convenience alias used in CLI args (maps to tau_visual)
    va_p_threshold: float = 0.70

    # Which decoder layers to attach hooks on (mid layers for EmberNet 16L)
    va_layer_indices: List[int] = field(
        default_factory=lambda: [6, 7, 8, 9, 10, 11]
    )

    # Total neurons to use across all monitored layers
    va_neuron_k: int = 128

    # Burst-mode detection
    va_window_size: int = 8
    va_burst_threshold: float = 0.70

    # Intervention strengths
    va_soft_penalty: float = 5.0   # logit decrement in normal mode
    va_decay_factor: float = 0.20  # multiplicative decay in burst mode

    # Enable/disable individual detectors
    va_use_neuron: bool = True
    va_use_logit: bool = True

    # Scale for logit discrepancy normalisation (L1 mean over vocab)
    va_logit_scale: float = 2.0

    def __post_init__(self):
        # Sync the convenience alias into the typed thresholds if caller
        # only set va_p_threshold (e.g. from CLI)
        if self.va_p_threshold != 0.70:
            self.va_tau_visual = self.va_p_threshold * 0.85
            self.va_tau_non_visual = self.va_p_threshold


# ---------------------------------------------------------------------------
# Temporal memory / burst detector
# ---------------------------------------------------------------------------
class VATemporalMemory:
    """Sliding-window average of recent p_VA scores; enters burst mode
    when the window mean exceeds va_burst_threshold."""

    def __init__(self, window_size: int = 8, burst_threshold: float = 0.70):
        self.window_size = window_size
        self.burst_threshold = burst_threshold
        self._buffer: List[float] = []
        self._burst: bool = False

    def update(self, p_va: float) -> None:
        self._buffer.append(p_va)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)
        avg = sum(self._buffer) / len(self._buffer)
        self._burst = avg >= self.burst_threshold

    def in_burst_mode(self) -> bool:
        return self._burst

    def window_mean(self) -> float:
        if not self._buffer:
            return 0.0
        return sum(self._buffer) / len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()
        self._burst = False


# ---------------------------------------------------------------------------
# Token-type helper
# ---------------------------------------------------------------------------
def is_visual_sensitive_token(
    token_id: int,
    tokenizer,
) -> bool:
    """Return True if the token is likely to be visually grounded."""
    if tokenizer is None:
        return False
    try:
        text = tokenizer.decode([token_id], skip_special_tokens=True).strip().lower()
        # Digit check
        if any(ch.isdigit() for ch in text):
            return True
        # Keyword check (substring match handles word-pieces like "Ġdog")
        for kw in VISUAL_KEYWORDS:
            if kw in text:
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Main VARefiner class
# ---------------------------------------------------------------------------
class VARefiner:
    """
    Inference-time hallucination mitigation for EmberNet.

    Usage inside generate():
        refiner.reset()
        refiner.register_hooks()
        null_logits = refiner.compute_null_baseline(inputs_embeds, attn_mask)
        refiner.set_null_baseline(null_logits)
        ...
        # inside token loop:
        next_token_logits = refiner.refine_logits(
            next_token_logits, generated_ids, tokenizer
        )
        ...
        refiner.remove_hooks()
        answer = calibrate_answer_prefix(raw_answer, refiner.get_va_scores(), tokenizer)
    """

    def __init__(
        self,
        model: "EmberNetVLM",
        config: VARefinerConfig,
        tokenizer=None,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer

        # Hook management
        self._hooks: List = []
        self._activation_buffer: Dict[int, torch.Tensor] = {}

        # Null logit baseline (computed once per generate() call)
        self._null_baseline: Optional[torch.Tensor] = None

        # Per-token va scores accumulated during generation
        self._va_scores: List[float] = []

        # Burst detector
        self.memory = VATemporalMemory(
            window_size=config.va_window_size,
            burst_threshold=config.va_burst_threshold,
        )

        # Derive per-layer neuron count
        n_layers = max(1, len(config.va_layer_indices))
        self.neurons_per_layer = max(1, config.va_neuron_k // n_layers)
        total_neurons = self.neurons_per_layer * n_layers

        # Lightweight 2-layer MLP classifier
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.classifier = nn.Sequential(
            nn.Linear(total_neurons, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        ).to(device=device, dtype=torch.float32)

        # Small initial weights → classifier starts near 0.5
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

        # Disable gradient for classifier (inference-only)
        for p in self.classifier.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------
    def register_hooks(self) -> None:
        """Attach hooks to shared_expert.down_proj in each monitored layer."""
        for layer_idx in self.config.va_layer_indices:
            try:
                target = self.model.decoder.layers[layer_idx].moe.shared_expert.down_proj
            except (IndexError, AttributeError):
                continue

            def make_hook(lidx: int, nperlayer: int):
                def hook_fn(module, inp, out):
                    # inp[0]: [total_tokens, intermediate_size]
                    act = inp[0].detach().float()
                    # Take the LAST token position, first nperlayer neurons
                    self._activation_buffer[lidx] = act[-1, :nperlayer].cpu()
                return hook_fn

            h = target.register_forward_hook(make_hook(layer_idx, self.neurons_per_layer))
            self._hooks.append(h)

    def remove_hooks(self) -> None:
        """Remove all registered hooks cleanly."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Neuron-based score
    # ------------------------------------------------------------------
    def compute_neuron_va_score(self) -> float:
        """Return p_VA^neuron ∈ [0,1] from the current activation buffer."""
        if not self._activation_buffer:
            return 0.5  # no signal → neutral

        feats = []
        for lidx in self.config.va_layer_indices:
            if lidx in self._activation_buffer:
                feats.append(self._activation_buffer[lidx])
            else:
                feats.append(torch.zeros(self.neurons_per_layer))

        feature_vec = torch.cat(feats, dim=0).unsqueeze(0).float()  # [1, total]
        device = next(self.model.parameters()).device
        feature_vec = feature_vec.to(device)

        with torch.no_grad():
            score = self.classifier(feature_vec).item()
        return float(score)

    # ------------------------------------------------------------------
    # Logit-discrepancy score
    # ------------------------------------------------------------------
    def compute_null_baseline(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        One extra decoder pass with visual tokens zeroed out.

        Returns the null logit vector at the last position: [1, vocab_size].
        """
        num_image_tokens = getattr(self.model.config, "num_image_tokens", 64)
        null_embeds = inputs_embeds.clone()
        # Zero out visual token positions
        null_embeds[:, :num_image_tokens, :] = 0.0

        with torch.no_grad():
            outputs = self.model.decoder(
                inputs_embeds=null_embeds,
                attention_mask=attention_mask,
            )
        return outputs[0][:, -1, :].detach()  # [1, vocab_size]

    def set_null_baseline(self, null_logits: torch.Tensor) -> None:
        self._null_baseline = null_logits

    def compute_logit_va_score(self, next_token_logits: torch.Tensor) -> float:
        """
        p_VA^logit ∈ [0,1].

        Large distance → model heavily uses image → visually GROUNDED → p_VA LOW.
        Small distance → model doesn't leverage image → visually ABSENT → p_VA HIGH.
        """
        if self._null_baseline is None:
            return 0.5  # no baseline → neutral

        diff = (next_token_logits.float() - self._null_baseline.float()).abs().mean()
        # tanh(large) ≈ 1 → 1 - tanh(large) ≈ 0 (grounded)
        p_va_logit = 1.0 - torch.tanh(diff / self.config.va_logit_scale).item()
        return float(max(0.0, min(1.0, p_va_logit)))

    # ------------------------------------------------------------------
    # Combined score
    # ------------------------------------------------------------------
    def _combine(self, p_neuron: float, p_logit: float) -> float:
        alpha = self.config.va_alpha
        if not self.config.va_use_neuron:
            return p_logit
        if not self.config.va_use_logit:
            return p_neuron
        return alpha * p_neuron + (1.0 - alpha) * p_logit

    # ------------------------------------------------------------------
    # Main intervention
    # ------------------------------------------------------------------
    def refine_logits(
        self,
        next_token_logits: torch.Tensor,
        generated_ids: List[int],
        tokenizer=None,
    ) -> torch.Tensor:
        """
        Call this at each decoding step BEFORE sampling.

        Mutates (or re-creates) next_token_logits if VA is triggered.
        """
        tokenizer = tokenizer or self.tokenizer

        # --- Score ---
        p_neuron = self.compute_neuron_va_score() if self.config.va_use_neuron else 0.5
        p_logit = self.compute_logit_va_score(next_token_logits) if self.config.va_use_logit else 0.5
        p_va = self._combine(p_neuron, p_logit)

        # --- Update memory ---
        self.memory.update(p_va)
        self._va_scores.append(p_va)

        # --- Decide threshold ---
        top_token_id = int(next_token_logits.argmax(dim=-1).item())
        sensitive = is_visual_sensitive_token(top_token_id, tokenizer)
        tau = self.config.va_tau_visual if sensitive else self.config.va_tau_non_visual

        if p_va < tau:
            return next_token_logits  # below threshold – no intervention

        burst_mode = self.memory.in_burst_mode()
        logits = next_token_logits.clone()

        if burst_mode:
            # Hard block + whole-distribution decay
            logits = logits * (1.0 - self.config.va_decay_factor)
            logits[..., top_token_id] = -1e9
        else:
            # Soft penalty on the top flagged candidate
            logits[..., top_token_id] -= self.config.va_soft_penalty

        return logits

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Call at the start of each generate() invocation."""
        self.memory.reset()
        self._va_scores.clear()
        self._activation_buffer.clear()
        self._null_baseline = None

    def get_va_scores(self) -> List[float]:
        """Per-generated-token p_VA scores from the last generate() call."""
        return list(self._va_scores)

    def get_window_mean(self) -> float:
        return self.memory.window_mean()


# ---------------------------------------------------------------------------
# Post-generation calibration
# ---------------------------------------------------------------------------
def calibrate_answer_prefix(
    answer_text: str,
    va_scores: List[float],
    tokenizer=None,
    config: Optional[VARefinerConfig] = None,
) -> str:
    """
    Prepend a short natural-language disclaimer based on average p_VA
    over visually sensitive tokens.

    Thresholds:
        avg < 0.40  → return raw answer (confident)
        0.40–0.70  → "I might be wrong, but …"
        >= 0.70    → "I cannot see that clearly in the image, but …"
    """
    if not va_scores:
        return answer_text

    # Prefer to average only over visually sensitive tokens
    sensitive_scores = []
    if tokenizer is not None:
        try:
            token_ids = tokenizer.encode(answer_text, add_special_tokens=False)
            paired = list(zip(token_ids, va_scores[: len(token_ids)]))
            for tid, score in paired:
                if is_visual_sensitive_token(tid, tokenizer):
                    sensitive_scores.append(score)
        except Exception:
            pass

    scores_to_use = sensitive_scores if sensitive_scores else va_scores
    avg = sum(scores_to_use) / len(scores_to_use)

    if avg < 0.40:
        return answer_text
    elif avg < 0.70:
        return "I might be wrong, but " + answer_text
    else:
        return "I cannot see that clearly in the image, but " + answer_text
