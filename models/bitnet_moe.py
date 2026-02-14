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
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BitNetMoEConfig:
    """Configuration for BitNet MoE Decoder."""
    vocab_size: int = 32000
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
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def ternary_quantize(weight: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    """
    Quantize weights to ternary {-1, 0, +1}.

    Uses absmean scaling: threshold = α * mean(|W|)
    Values below threshold become 0, others become sign(W).
    """
    threshold = alpha * weight.abs().mean()
    quantized = torch.sign(weight) * (weight.abs() >= threshold).float()
    return quantized


class STEQuantize(torch.autograd.Function):
    """Straight-Through Estimator for ternary quantization."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
        return ternary_quantize(weight, alpha)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-through: pass gradient unchanged
        return grad_output, None


class BitLinear(nn.Module):
    """
    Linear layer with BitNet b1.58 ternary quantization.

    Weights are stored in FP16 during training but quantized to {-1, 0, +1}
    during forward pass. Gradients flow through via straight-through estimator.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight stored in full precision for training
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Scaling factor for quantized weights
        self.register_buffer('weight_scale', torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self):
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute scaling factor (absmean)
        weight_scale = self.weight.abs().mean()

        # Quantize weights with STE
        if self.training:
            quantized_weight = STEQuantize.apply(self.weight)
        else:
            quantized_weight = ternary_quantize(self.weight)

        # Scale output to compensate for quantization
        output = F.linear(x, quantized_weight, self.bias)
        output = output * weight_scale

        return output

    def get_quantized_weight(self) -> torch.Tensor:
        """Return the quantized ternary weights for inference."""
        return ternary_quantize(self.weight)


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
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV for GQA
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)

        return output


class BitNetExpert(nn.Module):
    """Single FFN expert with BitLinear layers."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = BitLinear(hidden_size, intermediate_size)
        self.up_proj = BitLinear(hidden_size, intermediate_size)
        self.down_proj = BitLinear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class BitNetMoELayer(nn.Module):
    """
    Mixture-of-Experts layer with BitLinear experts.

    Uses top-k routing with optional shared expert for common knowledge.
    """

    def __init__(self, config: BitNetMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.use_shared_expert = config.use_shared_expert

        # Router (stays in full precision)
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # Domain experts
        self.experts = nn.ModuleList([
            BitNetExpert(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_experts)
        ])

        # Optional shared expert
        if self.use_shared_expert:
            self.shared_expert = BitNetExpert(config.hidden_size, config.intermediate_size)

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

        # Top-k routing
        routing_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )

        # Normalize weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        final_output = torch.zeros_like(hidden_flat)

        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            # Get tokens for this expert
            expert_input = hidden_flat[expert_mask]
            expert_output = self.experts[expert_idx](expert_input)

            # Get weights for this expert
            weight_mask = (top_k_indices == expert_idx).float()
            weights = (weight_mask * top_k_weights).sum(dim=-1)[expert_mask]

            # Weighted sum
            final_output[expert_mask] += expert_output * weights.unsqueeze(-1)

        # Add shared expert output
        if self.use_shared_expert:
            shared_output = self.shared_expert(hidden_flat)
            final_output = final_output + shared_output

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        # MoE FFN with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, router_logits


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
        return_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the decoder.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask
            inputs_embeds: Optional pre-computed embeddings (for VLM integration)
            return_router_logits: Whether to return router logits for aux loss

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            router_logits: (optional) List of router logits from each layer
        """
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        batch_size, seq_len, _ = hidden_states.shape

        # Create causal mask
        causal_mask = self._make_causal_mask(seq_len, hidden_states.device, hidden_states.dtype)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention mask
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask.float()) * float('-inf')
            causal_mask = causal_mask + extended_mask

        all_router_logits = []

        # Forward through layers
        for layer in self.layers:
            hidden_states, router_logits = layer(hidden_states, causal_mask)
            if return_router_logits:
                all_router_logits.append(router_logits)

        # Output projection
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if return_router_logits:
            return logits, all_router_logits
        return logits,

    def compute_router_aux_loss(self, router_logits_list: list) -> torch.Tensor:
        """
        Compute auxiliary loss to encourage load balancing across experts.

        Uses the simplified load balancing loss from Switch Transformer.
        """
        total_loss = 0.0
        num_layers = len(router_logits_list)

        for router_logits in router_logits_list:
            # router_logits: [batch * seq_len, num_experts]
            routing_weights = F.softmax(router_logits, dim=-1)

            # Fraction of tokens routed to each expert
            tokens_per_expert = routing_weights.mean(dim=0)

            # Average routing probability per expert
            routing_prob = routing_weights.mean(dim=0)

            # Load balancing loss
            loss = (tokens_per_expert * routing_prob).sum() * self.config.num_experts
            total_loss += loss

        return total_loss / num_layers

    def get_num_params(self, trainable_only: bool = False) -> int:
        """Count parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# Utility function to create model
def create_bitnet_moe_decoder(
    vocab_size: int = 32000,
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

