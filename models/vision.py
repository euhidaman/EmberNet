"""
Vision Encoder Module

This module wraps a SigLIP vision encoder with aggressive token compression
for efficient VLM integration.

VERIFIED SOURCES:
- SigLIP Model: google/siglip-base-patch16-224 (HuggingFace)
  URL: https://huggingface.co/google/siglip-base-patch16-224
  Params: ~86M total, vision encoder ~85M
- SigLIP Paper: https://arxiv.org/abs/2303.15343
- SmolVLM Architecture: https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct

TOKEN COMPRESSION:
- SigLIP-base outputs 196 tokens (14x14 grid) at 768 dimensions
- We apply 4x4 pixel shuffle: 14x14 -> 3x3 (effectively 9 tokens)
- Additional pooling to get 32-64 tokens if needed
- Final output: 32-64 tokens at projection_dim (768 for VLM)

DESIGN NOTES:
- Vision encoder is frozen by default (only projector trains)
- Pixel shuffle reduces spatial dimension while preserving info
- Simple MLP projector with BitLinear for efficiency
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import SiglipVisionModel, SiglipImageProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# Import BitLinear and RMSNorm for the projector
from .bitnet_moe import BitLinear, RMSNorm


class PixelShuffleCompressor(nn.Module):
    """
    Compress visual tokens using pixel shuffle operation.

    Takes a grid of visual tokens and merges them spatially.
    4x4 shuffle means 16 neighboring tokens become 1 token with 16x channels.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        shuffle_factor: int = 4,
        min_tokens: int = 9
    ):
        super().__init__()
        self.shuffle_factor = shuffle_factor
        self.min_tokens = min_tokens

        # After shuffle, we have shuffle_factor^2 times more channels
        expanded_dim = input_dim * (shuffle_factor ** 2)

        # Use RMSNorm instead of LayerNorm for gradient stability (Microsoft BitNet approach)
        self.pre_norm = RMSNorm(expanded_dim, eps=1e-5)
        # Initialize norm weights conservatively for stability
        nn.init.constant_(self.pre_norm.weight, 0.1)

        # Project back down with very small initialization
        self.proj = nn.Linear(expanded_dim, output_dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01 / math.sqrt(expanded_dim))
        nn.init.zeros_(self.proj.bias)

        # Post-projection normalization using RMSNorm
        self.post_norm = RMSNorm(output_dim, eps=1e-5)
        nn.init.constant_(self.post_norm.weight, 0.1)

    def forward(self, x: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: Visual tokens [batch, num_tokens, hidden_dim]
            spatial_size: Original grid size (H, W) before flattening

        Returns:
            Compressed tokens [batch, new_num_tokens, output_dim]
        """
        batch_size, num_tokens, hidden_dim = x.shape
        H, W = spatial_size

        # Input stabilization - replace NaN/Inf and clamp to reasonable range
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        x = torch.clamp(x, min=-10.0, max=10.0)

        # Reshape to spatial grid
        x = x.view(batch_size, H, W, hidden_dim)

        # Pad if needed to make divisible by shuffle_factor
        pad_h = (self.shuffle_factor - H % self.shuffle_factor) % self.shuffle_factor
        pad_w = (self.shuffle_factor - W % self.shuffle_factor) % self.shuffle_factor
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w

        # Pixel unshuffle (merge spatial neighbors into channels)
        # [B, H, W, C] -> [B, H/f, W/f, C*f*f]
        x = x.view(
            batch_size,
            H // self.shuffle_factor, self.shuffle_factor,
            W // self.shuffle_factor, self.shuffle_factor,
            hidden_dim
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(
            batch_size,
            H // self.shuffle_factor,
            W // self.shuffle_factor,
            -1
        )

        # Flatten spatial dimensions
        new_H, new_W = H // self.shuffle_factor, W // self.shuffle_factor
        x = x.view(batch_size, new_H * new_W, -1)

        # Normalize BEFORE projection (critical for gradient stability)
        x = self.pre_norm(x)

        # Project to output dimension
        x = self.proj(x)

        # Post-normalization for stable output distribution
        x = self.post_norm(x)

        # Final output stabilization
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))

        return x


class AdaptivePooler(nn.Module):
    """Pool visual tokens to a fixed number of tokens."""

    def __init__(self, num_output_tokens: int = 64, hidden_dim: int = 768):
        super().__init__()
        self.num_output_tokens = num_output_tokens

        # Learnable query tokens for pooling
        self.queries = nn.Parameter(torch.randn(1, num_output_tokens, hidden_dim) * 0.02)

        # Simple cross-attention for pooling
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        self.norm = RMSNorm(hidden_dim, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool variable number of tokens to fixed size.

        Args:
            x: Visual tokens [batch, num_tokens, hidden_dim]

        Returns:
            Pooled tokens [batch, num_output_tokens, hidden_dim]
        """
        # Input stabilization
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))

        batch_size = x.shape[0]
        queries = self.queries.expand(batch_size, -1, -1)

        # Cross-attention: queries attend to visual tokens
        pooled, _ = self.attn(queries, x, x)
        pooled = self.norm(pooled + queries)

        # Output stabilization
        pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

        return pooled


class VisionProjector(nn.Module):
    """
    Project vision embeddings to language model dimension.

    Uses BitLinear layers for efficiency and RMSNorm for stability.
    """

    def __init__(
        self,
        vision_dim: int,
        lm_dim: int,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        hidden_dim = hidden_dim or lm_dim

        # Two-layer MLP with BitLinear
        self.fc1 = BitLinear(vision_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = BitLinear(hidden_dim, lm_dim)
        self.norm = RMSNorm(lm_dim, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input stability - replace NaN/Inf and clamp
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        x = torch.clamp(x, min=-10.0, max=10.0)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm(x)

        # Output stability - replace any remaining NaN/Inf
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))

        return x


class CrossModalProjector(nn.Module):
    """
    Enhanced projector with cross-attention for better vision-language alignment.
    Uses BitLinear for efficiency while adding semantic richness.
    """

    def __init__(
        self,
        vision_dim: int,
        lm_dim: int,
        num_query_tokens: int = 64,
        num_cross_attn_layers: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens

        # Learnable query embeddings (Q-Former style)
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, lm_dim) * 0.02
        )

        # Project vision features to LM dimension using BitLinear
        self.vision_proj = BitLinear(vision_dim, lm_dim)

        # Cross-attention stack
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=lm_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_cross_attn_layers)
        ])

        # Feed-forward with BitLinear
        self.ffn = nn.Sequential(
            BitLinear(lm_dim, lm_dim * 4),
            nn.GELU(),
            BitLinear(lm_dim * 4, lm_dim),
        )

        # RMSNorms for gradient stability
        self.ln_q = RMSNorm(lm_dim, eps=1e-5)
        self.ln_ff = RMSNorm(lm_dim, eps=1e-5)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [B, N_vision, vision_dim]

        Returns:
            aligned_features: [B, num_query_tokens, lm_dim]
        """
        B = vision_features.shape[0]

        # Project vision features into LM space
        vision_proj = self.vision_proj(vision_features)  # [B, N, lm_dim]

        # Expand query tokens
        queries = self.query_tokens.expand(B, -1, -1)  # [B, Q, lm_dim]

        # Apply stacked cross-attention
        for cross_attn in self.cross_attn_layers:
            attn_out, _ = cross_attn(
                query=queries,
                key=vision_proj,
                value=vision_proj,
            )
            queries = self.ln_q(queries + attn_out)

        # Feed-forward
        ffn_out = self.ffn(queries)
        output = self.ln_ff(queries + ffn_out)

        return output


class VisionEncoder(nn.Module):
    """
    Complete vision encoder with SigLIP backbone and token compression.

    Pipeline:
    1. SigLIP extracts 196 tokens (14x14) at 768 dims
    2. Pixel shuffle compresses to ~9-16 tokens
    3. Optional adaptive pooling to fixed token count
    4. Projector maps to LM dimension

    Total trainable params: ~10-20M (projector + compression)
    Frozen params: ~85M (SigLIP backbone)
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        lm_hidden_size: int = 768,
        num_output_tokens: int = 64,
        use_pixel_shuffle: bool = True,
        shuffle_factor: int = 2,  # 2x2 gives 49 tokens from 196
        freeze_encoder: bool = True,
        use_pooling: bool = True,
        use_cross_modal_projector: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.lm_hidden_size = lm_hidden_size
        self.num_output_tokens = num_output_tokens
        self.use_pixel_shuffle = use_pixel_shuffle
        self.use_pooling = use_pooling
        self.use_cross_modal_projector = use_cross_modal_projector

        # Load SigLIP vision encoder
        if HAS_TRANSFORMERS:
            self.encoder = SiglipVisionModel.from_pretrained(model_name)
            self.processor = SiglipImageProcessor.from_pretrained(model_name)
            vision_dim = self.encoder.config.hidden_size  # 768 for base
            self.image_size = self.encoder.config.image_size  # 224
            self.patch_size = self.encoder.config.patch_size  # 16
        else:
            # Fallback for testing without transformers
            vision_dim = 768
            self.encoder = None
            self.processor = None
            self.image_size = 224
            self.patch_size = 16

        # Calculate spatial dimensions
        self.grid_size = self.image_size // self.patch_size  # 14 for base

        # Token compression
        if use_pixel_shuffle:
            self.compressor = PixelShuffleCompressor(
                input_dim=vision_dim,
                output_dim=vision_dim,
                shuffle_factor=shuffle_factor
            )
        else:
            self.compressor = None

        # Optional pooling to fixed size
        if use_pooling:
            self.pooler = AdaptivePooler(num_output_tokens, vision_dim)
        else:
            self.pooler = None

        # Projector: use cross-modal or simple projector
        if use_cross_modal_projector:
            self.projector = CrossModalProjector(
                vision_dim=vision_dim,
                lm_dim=lm_hidden_size,
                num_query_tokens=num_output_tokens,
            )
        else:
            self.projector = VisionProjector(vision_dim, lm_hidden_size)

        # Freeze encoder if requested
        if freeze_encoder and self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Initialize trainable components
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for trainable components."""
        # Initialize pooler attention
        if self.pooler is not None:
            for name, param in self.pooler.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=0.1)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def get_image_features(
        self,
        pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract visual features from images.

        Args:
            pixel_values: Preprocessed images [batch, 3, H, W]

        Returns:
            Visual features [batch, num_tokens, hidden_dim]
        """
        if self.encoder is None:
            # Dummy output for testing
            batch_size = pixel_values.shape[0]
            return torch.randn(batch_size, self.num_output_tokens, self.lm_hidden_size,
                             device=pixel_values.device, dtype=pixel_values.dtype)

        # Get encoder outputs
        outputs = self.encoder(pixel_values)
        hidden_states = outputs.last_hidden_state  # [B, 197, 768] with CLS token

        # Remove CLS token if present (SigLIP may not have it)
        if hidden_states.shape[1] == self.grid_size ** 2 + 1:
            hidden_states = hidden_states[:, 1:]

        return hidden_states

    def forward(
        self,
        pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Full forward pass: encode -> compress -> project.

        Args:
            pixel_values: Preprocessed images [batch, 3, H, W]

        Returns:
            Visual tokens for LM [batch, num_tokens, lm_hidden_size]
        """
        # Get visual features
        hidden_states = self.get_image_features(pixel_values)

        # Stability - clamp instead of nan_to_num to preserve gradients
        hidden_states = torch.clamp(hidden_states, min=-10.0, max=10.0)

        # Apply pixel shuffle compression
        if self.compressor is not None:
            hidden_states = self.compressor(
                hidden_states,
                (self.grid_size, self.grid_size)
            )

        # Pool to fixed number of tokens
        if self.pooler is not None:
            hidden_states = self.pooler(hidden_states)

        # Project to LM dimension
        hidden_states = self.projector(hidden_states)

        # Final stability - clamp to prevent extreme values
        hidden_states = torch.clamp(hidden_states, min=-10.0, max=10.0)

        return hidden_states

    def preprocess_images(
        self,
        images: list,
        return_tensors: str = "pt"
    ) -> torch.Tensor:
        """
        Preprocess images using SigLIP processor.

        Args:
            images: List of PIL images or numpy arrays
            return_tensors: Output format ("pt" for PyTorch)

        Returns:
            Preprocessed pixel values
        """
        if self.processor is None:
            # Fallback preprocessing
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            if isinstance(images, list):
                return torch.stack([transform(img) for img in images])
            return transform(images).unsqueeze(0)

        return self.processor(images, return_tensors=return_tensors)["pixel_values"]

    def get_num_params(self, trainable_only: bool = False) -> dict:
        """Count parameters in each component."""
        def count_params(module, trainable=False):
            if trainable:
                return sum(p.numel() for p in module.parameters() if p.requires_grad)
            return sum(p.numel() for p in module.parameters())

        params = {
            "encoder": count_params(self.encoder, trainable_only) if self.encoder else 0,
            "compressor": count_params(self.compressor, trainable_only) if self.compressor else 0,
            "pooler": count_params(self.pooler, trainable_only) if self.pooler else 0,
            "projector": count_params(self.projector, trainable_only),
        }
        params["total"] = sum(params.values())
        return params


# Convenience function to create vision encoder
def create_vision_encoder(
    model_name: str = "google/siglip-base-patch16-224",
    lm_hidden_size: int = 768,
    num_output_tokens: int = 64,
    freeze_encoder: bool = True,
    use_cross_modal_projector: bool = True,
) -> VisionEncoder:
    """Create a vision encoder with default settings."""
    return VisionEncoder(
        model_name=model_name,
        lm_hidden_size=lm_hidden_size,
        num_output_tokens=num_output_tokens,
        freeze_encoder=freeze_encoder,
        use_cross_modal_projector=use_cross_modal_projector,
    )

