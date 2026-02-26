"""
hub_utils.py
============
Utilities for pushing EmberNet checkpoints to the HuggingFace Hub.

Called automatically after each epoch when --push-to-hub is set.
Each push *overwrites* the same files in a single commit — the repo
always reflects the latest epoch rather than accumulating duplicate files.

Pushed files
------------
  pytorch_model.bin   – full model state dict
  config.json         – EmberNet + BitNetMoE config
  tokenizer_config.json / tokenizer.json / vocab.json / merges.txt /
  special_tokens_map.json / added_tokens.json  – tokenizer artefacts
  README.md           – model card (auto-generated, updated every epoch)
"""

import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Model card template
# ---------------------------------------------------------------------------

_CARD_TEMPLATE = """\
---
language: en
license: mit
tags:
  - vision-language-model
  - bitnet
  - mixture-of-experts
  - vlm
  - multimodal
  - edge-ai
pipeline_tag: image-text-to-text
---

# EmberNet{trial_suffix} — BitNet b1.58 MoE VLM

> **Status:** {status_line}

EmberNet is a tiny but capable Vision-Language Model built for edge deployment
and domain-expert reasoning.  It combines a frozen **SigLIP** vision backbone
with a **BitNet b1.58 ternary-quantized Mixture-of-Experts** language decoder,
achieving ~3× memory reduction over a full-precision equivalent while
preserving strong visual understanding across 8 specialised domains.

---

## Model Details

| Property | Value |
|---|---|
| **Model type** | Vision-Language Model (VLM) |
| **Quantisation** | BitNet b1.58 (ternary weights: −1, 0, +1) |
| **Total parameters** | {total_params_str} |
| **Trainable parameters** | {trainable_params_str} |
| **Active parameters / forward** | ~{active_params_str} (top-2 routing) |
| **Carbon footprint** | {carbon_str} |
| **Training stage** | Stage {stage}/2 — {stage_name} |
| **Epoch** | {epoch}/{total_epochs} |
| **Best loss** | {avg_loss:.4f} |
| **Last updated** | {timestamp} |

---

## Architecture

{arch_block}

### Expert Domains

| ID | Expert | Trained on |
|----|--------|-----------|
| 0 | `vision_ocr` | TextVQA, DocVQA, OCR-VQA, InfoVQA |
| 1 | `vision_diagram` | AI2D, InfoVQA diagrams |
| 2 | `code_math_chart` | ChartQA, PlotQA, FigureQA, DVQA |
| 3 | `code_math_formula` | MathVista, math formula datasets |
| 4 | `spatial_scene` | VQAv2, GQA, Visual Genome |
| 5 | `spatial_reasoning` | RefCOCO, GQA spatial splits |
| 6 | `agentic_knowledge` | OK-VQA, A-OKVQA |
| 7 | `agentic_reasoning` | ScienceQA, CLEVR |
| — | `shared` | All domains (always active) |

---

## Training

### Configuration

```yaml
stage_1_projector_alignment:
  epochs: 3
  batch_size: 8  (effective: 32 with grad-accum 4)
  learning_rate: 1e-4
  trainable: vision projector + compressor + pooler only

stage_2_expert_sft:
  epochs: 10
  batch_size: 4  (effective: 16 with grad-accum 4)
  learning_rate: 3e-4
  trainable: router + all 8 expert FFNs + shared expert
  expert_supervision_weight: 0.1
```

### Optimiser

- **BitNetStableOptimizer** — custom Adam with FP32 master weights  
- Two-phase LR: full LR for 60 % of training, then 0.1 × LR  
- Warmup: 100 steps  
- Weight clamp: [−3, 3] (maps cleanly to −1 / 0 / +1 at inference)

---

## Usage

```python
import torch
from PIL import Image
from transformers import AutoTokenizer

# Clone the repo and add it to your Python path, then:
from models import EmberNetVLM
from models.vlm import EmberNetConfig

# Load
config = EmberNetConfig()
model = EmberNetVLM(config)
ckpt = torch.load("pytorch_model.bin", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Inference
image = Image.open("scene.jpg").convert("RGB")
prompt = "<image>\\nDescribe what you see."

response = model.generate(
    image=image,
    prompt=prompt,
    tokenizer=tokenizer,
    max_new_tokens=256,
)
print(response)
```

---

## Intended Uses

- **Edge & embedded deployment** — ternary weights run efficiently on CPUs and NPUs  
- **Domain-aware visual reasoning** — dedicated experts for OCR, charts, math, spatial, and agentic tasks  
- **Robotic / agentic pipelines** — `agentic_knowledge` + `agentic_reasoning` experts support multi-step planning  
- **Fine-tuning base** — swap in domain datasets to specialise any of the 8 experts independently  

## Limitations

- Optimised for efficiency; maximum single-task accuracy is lower than full-precision models of similar size  
- Image resolution fixed at 224 × 224; very fine-grained OCR may degrade  
- Expert routing is learned; novel domains may activate sub-optimal experts until fine-tuned  
- Tokeniser vocabulary (32 002) is Phi-2 derived; non-English performance is limited  

---

## Citation

```bibtex
@software{{embernet_vlm,
  title  = {{EmberNet: Tiny BitNet b1.58 MoE Vision-Language Model}},
  author = {{Aman Euh}},
  year   = {{2026}},
  url    = {{https://huggingface.co/euhidaman/EmberNet{trial_suffix_nobrace}}}
}}
```
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(n: int) -> str:
    """Format large integer as human-readable string."""
    if n >= 1e9:
        return f"{n/1e9:.2f} B"
    if n >= 1e6:
        return f"{n/1e6:.1f} M"
    if n >= 1e3:
        return f"{n/1e3:.1f} K"
    return str(n)


def _estimate_carbon(training_seconds: float, gpu_count: int = 1) -> str:
    """
    Rough carbon estimate using median GPU TDP of 300 W at 50 % utilisation,
    US average grid intensity 0.386 kg CO2/kWh (EPA 2024).
    """
    kwh = (300 * 0.5 * gpu_count * training_seconds / 3600) / 1000
    co2 = kwh * 0.386
    if co2 < 0.001:
        return f"{co2*1000:.4f} g CO₂eq"
    return f"{co2:.4f} kg CO₂eq"


def _extract_param_breakdown(model) -> dict:
    """Extract per-component parameter counts from EmberNetVLM."""
    pb = {}
    ve = {}

    # Vision encoder components
    vision = getattr(model, "vision_encoder", None)
    if vision is not None:
        for attr in ("encoder", "compressor", "pooler", "projector"):
            sub = getattr(vision, attr, None)
            if sub is not None:
                ve[attr] = sum(p.numel() for p in sub.parameters())
    pb["vision_encoder"] = ve

    # Decoder
    decoder = getattr(model, "decoder", None)
    if decoder is not None:
        pb["decoder"] = sum(p.numel() for p in decoder.parameters())

        # Sub-components
        embed = getattr(decoder, "embed_tokens", None)
        if embed is not None:
            pb["decoder_embed"] = sum(p.numel() for p in embed.parameters())

        # Attention params across all layers
        layers = getattr(decoder, "layers", None)
        if layers is not None:
            attn_total = 0
            for layer in layers:
                attn = getattr(layer, "self_attn", None)
                if attn:
                    attn_total += sum(p.numel() for p in attn.parameters())
            pb["decoder_attn"] = attn_total

        # Router, shared expert, domain experts
        if layers is not None:
            router_total = expert_total = shared_total = 0
            n_experts = 0
            for layer in layers:
                moe = getattr(layer, "moe", None) or getattr(layer, "mlp", None)
                if moe is None:
                    continue
                router = getattr(moe, "router", None) or getattr(moe, "gate", None)
                if router:
                    router_total += sum(p.numel() for p in router.parameters())
                shared = getattr(moe, "shared_expert", None)
                if shared:
                    shared_total += sum(p.numel() for p in shared.parameters())
                experts = getattr(moe, "experts", None)
                if experts:
                    expert_total += sum(p.numel() for p in experts.parameters())
                    if n_experts == 0:
                        n_experts = len(experts)
            pb["decoder_router"] = router_total
            pb["decoder_shared"] = shared_total
            pb["decoder_experts"] = expert_total
            pb["n_experts"] = n_experts

    return pb


def _build_model_card(
    total_params: int,
    trainable_params: int,
    stage: int,
    epoch: int,
    total_epochs: int,
    avg_loss: float,
    is_trial: bool,
    training_seconds: float = 0.0,
    param_breakdown: Optional[dict] = None,
) -> str:
    trial_suffix = "-Trial" if is_trial else ""
    stage_name = "Projector Alignment" if stage == 1 else "Expert SFT"
    active_params = max(1, int(total_params * 0.28))
    status_line = (
        f"Trial run — Stage {stage}/2, Epoch {epoch}/{total_epochs}, Loss {avg_loss:.4f}"
        if is_trial
        else f"Stage {stage}/2, Epoch {epoch}/{total_epochs}, Loss {avg_loss:.4f}"
    )

    # Build architecture block with real values when available
    pb = param_breakdown or {}
    ve = pb.get("vision_encoder", {})
    enc_str  = _fmt(ve.get("encoder", 92_884_224))
    comp_str = _fmt(ve.get("compressor", 2_363_904))
    pool_str = _fmt(ve.get("pooler", 2_412_288))
    proj_str = _fmt(ve.get("projector", 10_088_448))
    dec_total = pb.get("decoder", 733_055_360)
    dec_str  = _fmt(dec_total)

    # Compute decoder sub-components if model was passed
    embed_str  = _fmt(pb.get("decoder_embed", 0))
    attn_str   = _fmt(pb.get("decoder_attn", 0))
    router_str = _fmt(pb.get("decoder_router", 0))
    shared_str = _fmt(pb.get("decoder_shared", 0))
    expert_str = _fmt(pb.get("decoder_experts", 0))
    n_experts  = pb.get("n_experts", 8)
    per_exp    = _fmt(pb.get("decoder_experts", 0) // max(n_experts, 1))

    arch_block = f"""\
```
EmberNet VLM
├── Vision Encoder  (frozen)
│   ├── SigLIP-base-patch16-224       {enc_str} params
│   ├── Token Compressor              {comp_str} params
│   ├── Spatial Pooler                {pool_str} params
│   └── BitLinear Projector           {proj_str} params
│
└── BitNet b1.58 MoE Decoder          {dec_str} params total
    ├── Layers: 16   Hidden: 768   Heads: 12 (GQA kv=6)
    ├── Experts: {n_experts} domain + 1 shared (always active)
    ├── Routing: Top-2 per token
    └── Quantisation: ternary weights, 4-bit activations
```"""

    # Add decoder breakdown table if we have real numbers
    if pb.get("decoder_embed"):
        arch_block += f"""

| Decoder Component | Parameters |
|---|---|
| Embeddings | {embed_str} |
| Attention (all layers) | {attn_str} |
| Router (all layers) | {router_str} |
| Shared Expert | {shared_str} |
| Domain Experts ({n_experts}×) | {expert_str} ({per_exp}/expert) |"""

    return _CARD_TEMPLATE.format(
        trial_suffix=trial_suffix,
        trial_suffix_nobrace=trial_suffix,
        status_line=status_line,
        total_params_str=_fmt(total_params),
        trainable_params_str=_fmt(trainable_params),
        active_params_str=_fmt(active_params),
        carbon_str=_estimate_carbon(training_seconds),
        stage=stage,
        stage_name=stage_name,
        epoch=epoch,
        total_epochs=total_epochs,
        avg_loss=avg_loss,
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        arch_block=arch_block,
    )


def _build_config_json(vlm_config, model_config, param_breakdown: Optional[dict] = None) -> dict:
    """Serialise EmberNetConfig + BitNetMoEConfig to a JSON-serialisable dict."""
    cfg = {
        "model_type": "embernet_vlm",
        "architecture": "BitNet b1.58 MoE VLM",
        "vision_encoder": {
            "model_name": getattr(vlm_config, "vision_model_name", "google/siglip-base-patch16-224"),
            "num_image_tokens": getattr(vlm_config, "num_image_tokens", 64),
            "freeze_vision": getattr(vlm_config, "freeze_vision", True),
        },
        "language_decoder": {
            "vocab_size":             getattr(model_config, "vocab_size", 32002),
            "hidden_size":            getattr(model_config, "hidden_size", 768),
            "intermediate_size":      getattr(model_config, "intermediate_size", 2048),
            "num_layers":             getattr(model_config, "num_layers", 16),
            "num_attention_heads":    getattr(model_config, "num_attention_heads", 12),
            "num_kv_heads":           getattr(model_config, "num_kv_heads", 6),
            "max_position_embeddings":getattr(model_config, "max_position_embeddings", 4096),
            "num_experts":            getattr(model_config, "num_experts", 8),
            "num_experts_per_tok":    getattr(model_config, "num_experts_per_tok", 2),
            "use_shared_expert":      getattr(model_config, "use_shared_expert", True),
            "expert_domains":         list(getattr(model_config, "expert_domains", [])),
            "quantisation":           "BitNet b1.58 (ternary)",
            "activation_bits":        4,
        },
        "torch_dtype": "bfloat16",
        "transformers_version": ">=4.36.0",
    }
    if param_breakdown:
        ve = param_breakdown.get("vision_encoder", {})
        cfg["parameter_counts"] = {
            "vision_encoder": sum(ve.values()),
            "vision_encoder_breakdown": ve,
            "decoder_total": param_breakdown.get("decoder", 0),
            "decoder_embeddings": param_breakdown.get("decoder_embed", 0),
            "decoder_attention": param_breakdown.get("decoder_attn", 0),
            "decoder_router": param_breakdown.get("decoder_router", 0),
            "decoder_shared_expert": param_breakdown.get("decoder_shared", 0),
            "decoder_domain_experts": param_breakdown.get("decoder_experts", 0),
            "num_domain_experts": param_breakdown.get("n_experts", 8),
        }
    return cfg


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def push_to_hub(
    model: "torch.nn.Module",
    vlm_config,
    training_config,
    stage: int,
    epoch: int,
    total_epochs: int,
    avg_loss: float,
    global_step: int,
    training_seconds: float = 0.0,
    repo_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    is_trial: bool = True,
):
    """
    Save model artefacts to a temp directory then push them to the Hub.

    All files are overwritten in a single commit so the repo always shows
    the latest epoch – no accumulation of versioned filenames.

    Parameters
    ----------
    model          : The EmberNetVLM module (on any device).
    vlm_config     : EmberNetConfig instance.
    training_config: TrainingConfig instance.
    stage          : Current training stage (1 or 2).
    epoch          : Current epoch number (1-indexed).
    total_epochs   : Total planned epochs for this stage.
    avg_loss       : Epoch average training loss.
    global_step    : Global optimizer step count.
    training_seconds: Elapsed wall-clock time in seconds.
    repo_id        : HF Hub repo e.g. "euhidaman/EmberNet-Trial".
                     Defaults to "euhidaman/EmberNet-Trial" for trial,
                     "euhidaman/EmberNet" for main.
    hf_token       : HuggingFace write token. Falls back to HF_TOKEN env var.
    is_trial       : True if this is a --trial run.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("  [hub] huggingface_hub not installed — skipping Hub push.")
        return

    # get_token moved to top-level in huggingface_hub >= 0.20; fall back gracefully
    try:
        from huggingface_hub import get_token as _hf_get_token
    except ImportError:
        try:
            from huggingface_hub.utils import get_token as _hf_get_token  # type: ignore[no-redef]
        except ImportError:
            _hf_get_token = lambda: None

    # Priority: explicit arg → env vars → token cached by `hf auth login`
    try:
        _cached_token = _hf_get_token()
    except Exception:
        _cached_token = None
    token = (
        hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or _cached_token
    )
    if not token:
        print("  [hub] No HF token found — run `huggingface-cli login` or set HF_TOKEN env var.")
        return
    print(f"  [hub] Token resolved (source: {'arg' if hf_token else 'env/cache'})")

    if repo_id is None:
        repo_id = "euhidaman/EmberNet-Trial" if is_trial else "euhidaman/EmberNet"

    # ------------------------------------------------------------------
    # Count parameters
    # ------------------------------------------------------------------
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Extract per-component breakdown for model card
    param_breakdown = _extract_param_breakdown(model)

    # ------------------------------------------------------------------
    # Ensure repo exists
    # ------------------------------------------------------------------
    api = HfApi(token=token)
    try:
        create_repo(repo_id=repo_id, token=token, exist_ok=True, repo_type="model", private=False)
    except Exception as e:
        print(f"  [hub] Could not create/verify repo {repo_id}: {e}")
        return

    # ------------------------------------------------------------------
    # Build artefacts in a temp directory
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Model weights
        print(f"  [hub] Saving model weights …")
        ckpt = {
            "model_state_dict": model.state_dict(),
            "global_step": global_step,
            "stage": stage,
            "epoch": epoch,
            "avg_loss": avg_loss,
        }
        torch.save(ckpt, tmp / "pytorch_model.bin")

        # 2. Config
        decoder_config = getattr(model, "decoder", None)
        decoder_cfg_obj = getattr(decoder_config, "config", None) if decoder_config else None
        cfg_dict = _build_config_json(vlm_config, decoder_cfg_obj or vlm_config, param_breakdown)
        (tmp / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

        # 3. README / model card
        card = _build_model_card(
            total_params=total_params,
            trainable_params=trainable_params,
            stage=stage,
            epoch=epoch,
            total_epochs=total_epochs,
            avg_loss=avg_loss,
            is_trial=is_trial,
            training_seconds=training_seconds,
            param_breakdown=param_breakdown,
        )
        (tmp / "README.md").write_text(card, encoding="utf-8")

        # 4. Tokenizer files  (saved from the tokenizer used in training)
        _save_tokenizer(tmp, training_config)

        # ------------------------------------------------------------------
        # Push — single commit that overwrites all files
        # ------------------------------------------------------------------
        commit_msg = (
            f"Update EmberNet{'-Trial' if is_trial else ''} "
            f"Stage {stage} Epoch {epoch}/{total_epochs} "
            f"| loss {avg_loss:.4f} | step {global_step}"
        )
        print(f"  [hub] Pushing to {repo_id} …")
        try:
            api.upload_folder(
                folder_path=str(tmp),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_msg,
                # upload_folder overwrites same-named files by default;
                # delete_patterns omitted for broad huggingface_hub version compatibility
            )
            print(f"  [hub] ✓ Pushed → https://huggingface.co/{repo_id}")
        except Exception as e:
            import traceback
            print(f"  [hub] Upload failed: {e}")
            traceback.print_exc()


def _save_tokenizer(dest: Path, training_config) -> None:
    """Save tokenizer files to dest directory."""
    tokenizer_name = "microsoft/phi-2"
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        tok.save_pretrained(str(dest))
        print(f"  [hub] Tokenizer ({tokenizer_name}) saved.")
    except Exception as e:
        print(f"  [hub] Could not save tokenizer: {e}")
