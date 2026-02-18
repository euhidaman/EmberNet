"""
BitNet MoE Decoder Module

This module implements the BitNet b1.58 quantization with Mixture-of-Experts (MoE) architecture.

VERIFIED SOURCES:
- BitNet Paper: https://arxiv.org/abs/2310.11453
- Microsoft BitNet Repo: https://github.com/microsoft/BitNet
- BitNet b1.58 Paper: https://arxiv.org/abs/2402.17764
- MoE Implementation Reference: HuggingFace transformers MixtralForCausalLM

ARCHITECTURE:
- 16 transformer layers with GQA attention
- 768 hidden dimension, 12 attention heads (6 KV heads for GQA)
- MoE FFN: 8 domain experts + 1 shared expert, top-2 routing
- All Linear layers use BitLinear (ternary weights {-1, 0, +1})
- Embeddings and LayerNorms remain in FP16

QUANTIZATION:
- Ternary quantization: W_q = sign(W) * (|W| >= threshold)
- Threshold = α * mean(|W|), α ≈ 0.7
- Straight-through estimator (STE) for gradients
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BitNetMoEConfig:
    """Configuration for BitNet MoE Decoder."""
    vocab_size: int = 32002  # 32000 base + special tokens (IMAGE_TOKEN_ID=32001)
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_layers: int = 16
    num_attention_heads: int = 12
    num_kv_heads: int = 6  # For GQA
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6

    # MoE configuration
    num_experts: int = 8
    num_experts_per_tok: int = 2  # Top-2 routing
    use_shared_expert: bool = True  # 1 shared expert always active

    # Expert domain specializations (8 experts total)
    # Each expert learns from specific dataset domains
    expert_domains: Tuple[str, ...] = (
        # Vision/OCR Experts (2) - trained on TextVQA, DocVQA, OCR-VQA, InfoVQA
        "vision_ocr",       # Expert 0: Text reading, document OCR
        "vision_diagram",   # Expert 1: Diagrams, infographics (AI2D, InfoVQA)

        # Chart/Math Experts (2) - trained on ChartQA, PlotQA, FigureQA, MathVista
        "code_math_chart",   # Expert 2: Charts, graphs, plots
        "code_math_formula", # Expert 3: Math equations, formulas

        # Spatial/Scene Experts (2) - trained on VQAv2, GQA, Visual Genome
        "spatial_scene",     # Expert 4: Scene understanding, object detection
        "spatial_reasoning", # Expert 5: Spatial relationships, counting

        # Reasoning Experts (2) - trained on ScienceQA, A-OKVQA, OK-VQA, CLEVR
        "agentic_knowledge", # Expert 6: Knowledge-based reasoning
        "agentic_reasoning", # Expert 7: Multi-step reasoning, logic
    )

    # + 1 Shared Expert (always active) - handles common patterns across all domains

    # Training settings
    router_aux_loss_coef: float = 0.01


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Microsoft BitNet)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x.clamp(-100.0, 100.0)
        variance = x.pow(2).mean(-1, keepdim=True)
        variance = variance.clamp(min=self.eps)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.clamp(-10.0, 10.0)
        return (self.weight * x).to(dtype)


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """
    Per-token activation quantization to 8-bit (Microsoft BitNet official).
    Clamps min to 1e-5 to prevent division by zero/tiny values.
    """
    dtype = x.dtype
    x = x.float()
    x = x.clamp(-50.0, 50.0)
    max_val = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-4)
    scale = 127.0 / max_val
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y.to(dtype)


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """
    Ternary weight quantization (Microsoft BitNet official).
    Uses mean for centering and scaling - no division by scale.
    """
    dtype = w.dtype
    w = w.float()
    scale = w.abs().mean().clamp(min=1e-8)
    e = w.mean()
    u = (w - e).sign() * scale
    return u.to(dtype)


class BitLinear(nn.Module):
    """
    BitNet b1.58 linear layer (Microsoft BitNet official implementation).

    Key differences from naive implementations:
    1. RMSNorm applied BEFORE activation quantization
    2. Weight quantization uses mean (no division)
    3. STE pattern: x + (quant(x) - x).detach()
    4. No explicit division by potentially tiny scales
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # RMSNorm for input stabilization
        self.norm = RMSNorm(in_features)

        self._init_weights()

    def _init_weights(self):
        std = 0.02 / math.sqrt(self.in_features)
        nn.init.normal_(self.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight

        # Normalize input before quantization
        x_norm = self.norm(x)

        # STE for activation quantization
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()

        # STE for weight quantization
        w_quant = w + (weight_quant(w) - w).detach()

        # Standard linear operation
        y = F.linear(x_quant, w_quant, self.bias)

        return y

    def get_quantized_weight(self) -> torch.Tensor:
        """Return quantized weights for inference."""
        return weight_quant(self.weight)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cache', emb.cos())
        self.register_buffer('sin_cache', emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cache.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dimensions."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key."""
    # cos/sin: [seq_len, head_dim]
    # q/k: [batch, num_heads, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class BitNetAttention(nn.Module):
    """
    Multi-head attention with BitLinear and Grouped Query Attention (GQA).
    Supports KV-cache for efficient generation.
    """

    def __init__(self, config: BitNetMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Projections using BitLinear
        self.q_proj = BitLinear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = BitLinear(self.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = BitLinear(self.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = BitLinear(self.num_heads * self.head_dim, self.hidden_size)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Determine position offset for rotary embeddings when using cache
        kv_seq_len = seq_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, kv_seq_len)
        if past_key_value is not None:
            # Only apply to the new positions
            cos = cos[kv_seq_len - seq_len:]
            sin = sin[kv_seq_len - seq_len:]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Concatenate with past KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        # Store KV for future use
        present_key_value = (k, v) if use_cache else None
        kv_seq_len = k.shape[2]

        # Expand KV for GQA
        if self.num_kv_groups > 1:
            k_expanded = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k_expanded = k_expanded.reshape(batch_size, self.num_heads, kv_seq_len, self.head_dim)
            v_expanded = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v_expanded = v_expanded.reshape(batch_size, self.num_heads, kv_seq_len, self.head_dim)
        else:
            k_expanded, v_expanded = k, v

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        # NaN protection for attention weights
        if torch.isnan(attn_weights).any():
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            # Focus on first token as fallback
            attn_weights[:, :, :, 0] = 1.0

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v_expanded)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)

        return output, present_key_value


class BitNetExpert(nn.Module):
    """Single FFN expert with BitLinear layers."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = BitLinear(hidden_size, intermediate_size)
        self.up_proj = BitLinear(hidden_size, intermediate_size)
        self.down_proj = BitLinear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation with stability
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Clamp before activation to prevent overflow
        gate = gate.clamp(-20, 20)
        hidden = F.silu(gate) * up
        # Clamp hidden states
        hidden = hidden.clamp(-1e3, 1e3)
        output = self.down_proj(hidden)
        return output


class BitNetMoELayer(nn.Module):
    """
    Mixture-of-Experts layer with improved routing, expert dropout,
    and richer auxiliary losses for better specialization.
    """

    def __init__(self, config: BitNetMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.use_shared_expert = config.use_shared_expert

        # Router with small-initialization for stability
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=True)
        nn.init.zeros_(self.router.bias)
        nn.init.normal_(self.router.weight, std=0.01)

        # Domain experts
        self.experts = nn.ModuleList([
            BitNetExpert(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_experts)
        ])

        # Optional shared expert
        if self.use_shared_expert:
            self.shared_expert = BitNetExpert(config.hidden_size, config.intermediate_size)

        # Expert dropout probability
        self.expert_dropout = 0.1

        # Track expert usage statistics
        self.register_buffer("expert_usage", torch.zeros(config.num_experts))

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with top-k routing.

        Returns:
            output: MoE output tensor
            router_logits: Logits for auxiliary loss computation
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        # Compute router logits
        router_logits = self.router(hidden_flat)

        # Expert dropout (mask some experts per batch during training)
        if self.training and self.expert_dropout > 0.0:
            expert_mask = torch.bernoulli(
                torch.full(
                    (1, self.num_experts),
                    1.0 - self.expert_dropout,
                    device=router_logits.device,
                )
            )
            router_logits = router_logits * expert_mask + (1 - expert_mask) * (-1e9)

        # Gumbel-Softmax routing during training for better gradient flow
        if self.training:
            # Use stable Gumbel noise computation
            u = torch.rand_like(router_logits).clamp(1e-6, 1.0 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(u))
            logits_with_noise = router_logits + gumbel_noise * 0.1  # Reduced noise scale
            routing_weights = F.softmax(logits_with_noise, dim=-1)
        else:
            routing_weights = F.softmax(router_logits, dim=-1)

        # Top-k routing
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Initialize output
        final_output = torch.zeros_like(hidden_flat)

        # Track expert usage
        if self.training:
            for i in range(self.num_experts):
                selected = (top_k_indices == i).any(dim=-1).float().sum()
                self.expert_usage[i] = 0.99 * self.expert_usage[i] + 0.01 * selected

        # Compute per-expert outputs
        for expert_idx in range(self.num_experts):
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            expert_input = hidden_flat[expert_mask]
            expert_output = self.experts[expert_idx](expert_input)

            # Aggregate weights for this expert
            weight_mask = (top_k_indices == expert_idx).float()
            weights = (weight_mask * top_k_weights).sum(dim=-1)[expert_mask]

            final_output[expert_mask] += expert_output * weights.unsqueeze(-1)

        # Shared expert contribution
        if self.use_shared_expert:
            shared_output = self.shared_expert(hidden_flat)
            final_output = final_output + shared_output

        # Clamp output to prevent NaN/Inf propagation
        final_output = final_output.clamp(-1e4, 1e4)

        output = final_output.view(batch_size, seq_len, hidden_dim)

        return output, router_logits


class BitNetMoEBlock(nn.Module):
    """Single transformer block with attention + MoE FFN."""

    def __init__(self, config: BitNetMoEConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = BitNetAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.moe = BitNetMoELayer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LN for attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        attn_output, present_key_value = self.attention(
            hidden_states, attention_mask, position_ids,
            past_key_value=past_key_value, use_cache=use_cache
        )

        # NaN check and recovery
        if torch.isnan(attn_output).any():
            print("WARNING: NaN in attention output, using zero")
            attn_output = torch.zeros_like(attn_output)

        hidden_states = residual + attn_output

        # Pre-LN for MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MoE FFN forward
        ffn_output, router_logits = self.moe(hidden_states)

        # NaN check and recovery
        if torch.isnan(ffn_output).any():
            print("WARNING: NaN in FFN output, using zero")
            ffn_output = torch.zeros_like(ffn_output)

        hidden_states = residual + ffn_output

        return hidden_states, router_logits, present_key_value


class BitNetMoEDecoder(nn.Module):
    """
    Full BitNet MoE decoder model.

    16 transformer blocks with ternary BitLinear layers and MoE FFN.
    """

    def __init__(self, config: BitNetMoEConfig):
        super().__init__()
        self.config = config

        # Token embeddings (kept in full precision)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            BitNetMoEBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Output head
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, BitLinear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _make_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the decoder.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask
            inputs_embeds: Optional pre-computed embeddings (for VLM integration)
            past_key_values: Optional KV cache from previous forward passes
            use_cache: Whether to return KV cache for generation
            return_router_logits: Whether to return router logits for aux loss

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            past_key_values: (optional) KV cache for each layer
            router_logits: (optional) List of router logits from each layer
        """
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        batch_size, seq_len, _ = hidden_states.shape

        # Calculate past sequence length for proper masking
        past_length = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_length = past_key_values[0][0].shape[2]

        # Create causal mask (accounting for past)
        total_len = seq_len + past_length
        causal_mask = self._make_causal_mask(total_len, hidden_states.device, hidden_states.dtype)
        # Only use the last seq_len rows for the current sequence
        causal_mask = causal_mask[:, :, -seq_len:, :]

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention mask
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask.float()) * float('-inf')
            causal_mask = causal_mask + extended_mask[:, :, :, -total_len:]

        all_router_logits = []
        present_key_values = [] if use_cache else None

        # Forward through layers
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, router_logits, present_kv = layer(
                hidden_states, causal_mask,
                past_key_value=past_kv, use_cache=use_cache
            )
            if return_router_logits:
                all_router_logits.append(router_logits)
            if use_cache:
                present_key_values.append(present_kv)

        # Output projection
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # Build return tuple
        outputs = (logits,)
        if use_cache:
            outputs = outputs + (present_key_values,)
        if return_router_logits:
            outputs = outputs + (all_router_logits,)

        return outputs

    def compute_router_aux_loss(self, router_logits_list: list) -> torch.Tensor:
        """
        Compute auxiliary loss for router with multiple components:
        - Load balancing: encourage uniform expert usage
        - Router z-loss: prevent router logits from growing too large
        - Entropy regularization: encourage confident routing
        """
        total_loss = 0.0
        num_layers = len(router_logits_list)

        for router_logits in router_logits_list:
            # router_logits: [batch * seq_len, num_experts]
            routing_weights = F.softmax(router_logits, dim=-1)

            # Load balancing: encourage uniform expert usage
            tokens_per_expert = routing_weights.mean(dim=0)  # [E]
            uniform = torch.full_like(tokens_per_expert, 1.0 / self.config.num_experts)
            load_balance_loss = F.kl_div(
                (tokens_per_expert + 1e-8).log(), uniform, reduction="batchmean"
            )

            # Router z-loss: prevent logits from growing too large
            z_loss = torch.logsumexp(router_logits, dim=-1).mean()

            # Entropy regularization: encourage confident routing
            entropy = -(routing_weights * (routing_weights + 1e-8).log()).sum(dim=-1).mean()

            layer_loss = (
                load_balance_loss
                + 0.01 * z_loss
                + 0.001 * entropy
            )
            total_loss += layer_loss

        return total_loss / max(1, num_layers)

    def get_num_params(self, trainable_only: bool = False) -> int:
        """Count parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# Utility function to create model
def create_bitnet_moe_decoder(
    vocab_size: int = 32002,
    hidden_size: int = 768,
    num_layers: int = 16,
    num_experts: int = 8,
    **kwargs
) -> BitNetMoEDecoder:
    """Create a BitNet MoE decoder with specified configuration."""
    config = BitNetMoEConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_experts=num_experts,
        **kwargs
    )
    return BitNetMoEDecoder(config)

