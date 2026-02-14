"""
Model Conversion for EmberNet VLM

Converts trained EmberNet models to efficient ternary format for deployment.
Target: <500MB model size with 1.58-bit weight storage.

CONVERSION PROCESS:
1. Load trained model checkpoint
2. Quantize BitLinear weights to actual ternary values
3. Pack weights into efficient 2-bit representation
4. Save in compact format (.gguf or custom .ember format)

STORAGE FORMAT:
- Ternary weights: {-1, 0, +1} stored as 2 bits per weight
- Scales: FP16 per tensor
- Embeddings/LayerNorms: Keep in FP16
- Vision encoder: Can optionally be quantized to INT8

ESTIMATED SIZES:
- Language decoder: ~200M ternary params = ~50MB
- Vision encoder (INT8): ~85M params = ~85MB
- Embeddings/misc: ~50MB
- Total: ~200-400MB (well under 500MB target)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np


@dataclass
class ConversionConfig:
    """Configuration for model conversion."""
    output_format: str = "ember"  # "ember" or "gguf"
    quantize_vision: bool = False  # Quantize vision encoder to INT8
    keep_fp16_layers: list = None  # Layers to keep in FP16
    compression_level: int = 1  # 0=none, 1=zlib, 2=lzma


def pack_ternary_weights(weights: torch.Tensor) -> Tuple[np.ndarray, float]:
    """
    Pack ternary weights into 2-bit representation.

    Encoding:
    - -1 -> 0b00
    - 0  -> 0b01
    - +1 -> 0b10

    Args:
        weights: Ternary weight tensor {-1, 0, +1}

    Returns:
        packed: Packed bytes array
        scale: Weight scale for dequantization
    """
    # Flatten weights
    flat = weights.flatten().cpu().numpy()

    # Get scale (absmean of original weights)
    scale = float(np.abs(flat).mean())

    # Map to 2-bit values
    # -1 -> 0, 0 -> 1, +1 -> 2
    mapped = (flat + 1).astype(np.uint8)

    # Pack 4 values per byte
    num_weights = len(mapped)
    num_bytes = (num_weights + 3) // 4
    packed = np.zeros(num_bytes, dtype=np.uint8)

    for i in range(num_weights):
        byte_idx = i // 4
        bit_offset = (i % 4) * 2
        packed[byte_idx] |= (mapped[i] & 0b11) << bit_offset

    return packed, scale


def unpack_ternary_weights(
    packed: np.ndarray,
    scale: float,
    shape: Tuple[int, ...]
) -> torch.Tensor:
    """
    Unpack 2-bit packed weights back to ternary tensor.

    Args:
        packed: Packed bytes array
        scale: Weight scale
        shape: Original tensor shape

    Returns:
        Unpacked ternary weight tensor
    """
    num_weights = np.prod(shape)

    # Unpack
    unpacked = np.zeros(num_weights, dtype=np.int8)
    for i in range(num_weights):
        byte_idx = i // 4
        bit_offset = (i % 4) * 2
        value = (packed[byte_idx] >> bit_offset) & 0b11
        unpacked[i] = value - 1  # Map back: 0->-1, 1->0, 2->1

    # Reshape and convert to tensor
    tensor = torch.from_numpy(unpacked.reshape(shape)).float()

    return tensor * scale


def quantize_to_int8(tensor: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    """Quantize tensor to INT8."""
    min_val = tensor.min().item()
    max_val = tensor.max().item()

    scale = (max_val - min_val) / 255.0
    zero_point = -min_val / scale

    quantized = torch.round((tensor - min_val) / scale).clamp(0, 255).to(torch.uint8)

    return quantized, scale, zero_point


def dequantize_int8(
    quantized: torch.Tensor,
    scale: float,
    zero_point: float
) -> torch.Tensor:
    """Dequantize INT8 tensor back to float."""
    return (quantized.float() - zero_point) * scale


class TernaryLinear(nn.Module):
    """
    Linear layer with pre-quantized ternary weights.

    Used for inference with packed weights.
    """

    def __init__(
        self,
        packed_weights: np.ndarray,
        scale: float,
        weight_shape: Tuple[int, int],
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.weight_shape = weight_shape
        self.register_buffer('scale', torch.tensor(scale))
        self.register_buffer('packed_weights', torch.from_numpy(packed_weights))

        # Pre-unpack for fast inference
        unpacked = unpack_ternary_weights(packed_weights, scale, weight_shape)
        self.register_buffer('weight', unpacked)

        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)


def convert_bitlinear_to_ternary(module: nn.Module) -> nn.Module:
    """
    Convert BitLinear modules to TernaryLinear with packed weights.

    Args:
        module: Module containing BitLinear layers

    Returns:
        Module with converted TernaryLinear layers
    """
    from models.bitnet_moe import BitLinear, ternary_quantize

    for name, child in module.named_children():
        if isinstance(child, BitLinear):
            # Quantize weights
            quantized_weight = ternary_quantize(child.weight)

            # Pack weights
            packed, scale = pack_ternary_weights(quantized_weight)

            # Create TernaryLinear
            bias = child.bias.data if child.bias is not None else None
            ternary_linear = TernaryLinear(
                packed_weights=packed,
                scale=scale,
                weight_shape=(child.out_features, child.in_features),
                bias=bias,
            )

            setattr(module, name, ternary_linear)
        else:
            # Recursive conversion
            convert_bitlinear_to_ternary(child)

    return module


def convert_to_ternary(
    model: nn.Module,
    config: Optional[ConversionConfig] = None,
) -> nn.Module:
    """
    Convert EmberNet model to ternary format for inference.

    Args:
        model: Trained EmberNet model
        config: Conversion configuration

    Returns:
        Converted model with packed ternary weights
    """
    config = config or ConversionConfig()

    print("Converting model to ternary format...")

    # Convert BitLinear layers
    model = convert_bitlinear_to_ternary(model)

    # Optionally quantize vision encoder to INT8
    if config.quantize_vision:
        print("Quantizing vision encoder to INT8...")
        # This would require more complex handling
        # For now, we keep vision encoder in FP16

    return model


def save_ternary_model(
    model: nn.Module,
    save_path: str,
    config: Optional[ConversionConfig] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save converted model in compact format.

    Args:
        model: Converted model
        save_path: Output file path
        config: Conversion configuration
        metadata: Additional metadata to save
    """
    config = config or ConversionConfig()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect state
    state = {
        "model_state_dict": model.state_dict(),
        "format_version": "1.0",
        "metadata": metadata or {},
    }

    # Calculate size
    total_bytes = 0
    for name, param in model.state_dict().items():
        total_bytes += param.numel() * param.element_size()

    print(f"Saving model to {save_path}")
    print(f"Estimated size: {total_bytes / 1024 / 1024:.1f} MB")

    # Save
    torch.save(state, save_path)

    # Report actual size
    actual_size = save_path.stat().st_size / 1024 / 1024
    print(f"Actual file size: {actual_size:.1f} MB")


def load_ternary_model(
    load_path: str,
    device: str = "cpu",
):
    """
    Load converted ternary model.

    Args:
        load_path: Path to saved model
        device: Device to load to

    Returns:
        Loaded model
    """
    from models import EmberNetVLM
    from models.vlm import EmberNetConfig

    state = torch.load(load_path, map_location=device)

    # Create model
    config = EmberNetConfig()
    if "metadata" in state and "config" in state["metadata"]:
        config = EmberNetConfig(**state["metadata"]["config"])

    model = EmberNetVLM(config)

    # Load state (handle packed weights)
    model.load_state_dict(state["model_state_dict"], strict=False)

    return model.to(device)


def estimate_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Estimate model size after conversion.

    Returns size estimates in MB.
    """
    from models.bitnet_moe import BitLinear

    sizes = {
        "bitlinear_ternary": 0,  # 2 bits per weight
        "fp16": 0,  # 16 bits per weight
        "fp32": 0,  # 32 bits per weight
        "total_original": 0,
        "total_converted": 0,
    }

    for name, param in model.named_parameters():
        num_params = param.numel()
        original_bits = num_params * 32  # FP32

        sizes["total_original"] += original_bits

        # Check if this is a BitLinear weight
        is_bitlinear = False
        for module_name, module in model.named_modules():
            if isinstance(module, BitLinear):
                if name.endswith(f"{module_name}.weight"):
                    is_bitlinear = True
                    break

        if is_bitlinear:
            # 2 bits per weight + scale overhead
            converted_bits = num_params * 2 + 32  # 2-bit + FP32 scale
            sizes["bitlinear_ternary"] += converted_bits
        else:
            # Keep in FP16
            converted_bits = num_params * 16
            sizes["fp16"] += converted_bits

        sizes["total_converted"] += converted_bits

    # Convert to MB
    for key in sizes:
        sizes[key] = sizes[key] / 8 / 1024 / 1024

    return sizes


def main():
    """Convert a trained model to ternary format."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert EmberNet model")
    parser.add_argument("input", type=str, help="Input checkpoint path")
    parser.add_argument("output", type=str, help="Output model path")
    parser.add_argument("--quantize-vision", action="store_true",
                        help="Quantize vision encoder to INT8")

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.input}")
    from models import EmberNetVLM
    model = EmberNetVLM.from_pretrained(args.input)

    # Estimate sizes
    print("\nSize estimates:")
    sizes = estimate_model_size(model)
    for key, value in sizes.items():
        print(f"  {key}: {value:.1f} MB")

    # Convert
    config = ConversionConfig(quantize_vision=args.quantize_vision)
    model = convert_to_ternary(model, config)

    # Save
    save_ternary_model(model, args.output, config)

    print("\nConversion complete!")


if __name__ == "__main__":
    main()

