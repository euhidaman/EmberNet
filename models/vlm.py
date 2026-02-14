"""
EmberNet VLM - Complete Vision-Language Model

This module integrates the vision encoder, projector, and BitNet MoE decoder
into a complete VLM that can perform visual question answering, OCR,
document understanding, and multi-turn conversations about images.

VERIFIED SOURCES:
- SmolVLM Architecture: https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct
- LLaVA Integration Pattern: https://github.com/haotian-liu/LLaVA
- Vision Token Embedding: Qwen-VL, InternVL patterns

ARCHITECTURE OVERVIEW:
- Vision Encoder: SigLIP-base (frozen, ~85M params)
- Token Compression: Pixel shuffle + adaptive pooling (64 tokens)
- Projector: BitLinear MLP (~5M params)
- Decoder: BitNet MoE (16 layers, 8 experts, ~200M params)
- Total: ~300M params (~50M active per forward)

USAGE:
    model = EmberNetVLM()

    # Single-turn Q&A
    response = model.generate(image, "What is in this image?")

    # Multi-turn conversation
    response = model.chat(image, "Describe this image")
    response = model.chat(prompt="What colors do you see?")
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Dict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision import VisionEncoder, create_vision_encoder
from .bitnet_moe import BitNetMoEDecoder, BitNetMoEConfig


# Special tokens for image embedding
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 32001  # Assuming vocab_size = 32000, add special tokens after
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 0


@dataclass
class EmberNetConfig:
    """Configuration for EmberNet VLM."""

    # Vision encoder settings
    vision_model_name: str = "google/siglip-base-patch16-224"
    num_image_tokens: int = 64
    freeze_vision: bool = True

    # Language model settings
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_layers: int = 16
    num_attention_heads: int = 12
    num_kv_heads: int = 6
    max_position_embeddings: int = 4096

    # MoE settings
    num_experts: int = 8
    num_experts_per_tok: int = 2
    use_shared_expert: bool = True
    router_aux_loss_coef: float = 0.01

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    # Training settings
    pad_token_id: int = PAD_TOKEN_ID
    bos_token_id: int = BOS_TOKEN_ID
    eos_token_id: int = EOS_TOKEN_ID
    image_token_id: int = IMAGE_TOKEN_ID


class EmberNetVLM(nn.Module):
    """
    EmberNet Vision-Language Model.

    A tiny but capable VLM using BitNet quantization and MoE for efficiency.
    Designed for edge deployment with <500MB model size.

    Key Features:
    - Multi-image support
    - Multi-turn conversation
    - Efficient ternary quantization
    - Sparse MoE with domain experts
    """

    def __init__(self, config: Optional[EmberNetConfig] = None):
        super().__init__()

        self.config = config or EmberNetConfig()

        # Build vision encoder
        self.vision_encoder = create_vision_encoder(
            model_name=self.config.vision_model_name,
            lm_hidden_size=self.config.hidden_size,
            num_output_tokens=self.config.num_image_tokens,
            freeze_encoder=self.config.freeze_vision,
        )

        # Build language decoder
        decoder_config = BitNetMoEConfig(
            vocab_size=self.config.vocab_size + 10,  # Extra tokens for special tokens
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_layers=self.config.num_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_kv_heads,
            max_position_embeddings=self.config.max_position_embeddings,
            num_experts=self.config.num_experts,
            num_experts_per_tok=self.config.num_experts_per_tok,
            use_shared_expert=self.config.use_shared_expert,
            router_aux_loss_coef=self.config.router_aux_loss_coef,
        )
        self.decoder = BitNetMoEDecoder(decoder_config)

        # Image token embedding placeholder
        self.image_newline = nn.Parameter(
            torch.randn(self.config.hidden_size) * 0.02
        )

        # Conversation history for multi-turn
        self._conversation_history: List[Dict] = []
        self._cached_image_embeds: Optional[torch.Tensor] = None

        # Tokenizer (loaded lazily)
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                # Use a small, fast tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/phi-2",  # Good small tokenizer
                    trust_remote_code=True,
                )
                # Add special tokens
                self._tokenizer.add_special_tokens({
                    "additional_special_tokens": [IMAGE_TOKEN]
                })
                self._tokenizer.pad_token = self._tokenizer.eos_token
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")
                self._tokenizer = None
        return self._tokenizer

    def _merge_image_text_embeds(
        self,
        input_ids: torch.Tensor,
        image_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge image embeddings with text embeddings at IMAGE_TOKEN positions.

        Args:
            input_ids: Text token IDs [batch, text_seq_len]
            image_embeds: Image embeddings [batch, num_images * num_image_tokens, hidden]
            attention_mask: Attention mask for text

        Returns:
            merged_embeds: Combined embeddings [batch, total_seq_len, hidden]
            merged_attention_mask: Updated attention mask
        """
        batch_size = input_ids.shape[0]
        text_embeds = self.decoder.embed_tokens(input_ids)

        # Find image token positions
        image_token_mask = (input_ids == self.config.image_token_id)

        if not image_token_mask.any():
            # No image tokens, prepend image embeddings
            merged_embeds = torch.cat([image_embeds, text_embeds], dim=1)
            if attention_mask is not None:
                image_mask = torch.ones(
                    batch_size, image_embeds.shape[1],
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                merged_attention_mask = torch.cat([image_mask, attention_mask], dim=1)
            else:
                merged_attention_mask = None
        else:
            # Replace image tokens with image embeddings
            # For simplicity, assume one <image> token that gets expanded
            merged_embeds_list = []
            merged_mask_list = []

            for b in range(batch_size):
                positions = torch.where(image_token_mask[b])[0]
                if len(positions) == 0:
                    merged_embeds_list.append(
                        torch.cat([image_embeds[b:b+1], text_embeds[b:b+1]], dim=1)
                    )
                else:
                    # Insert image embeddings at first <image> position
                    pos = positions[0].item()
                    before = text_embeds[b, :pos]
                    after = text_embeds[b, pos+1:]
                    merged = torch.cat([
                        before.unsqueeze(0),
                        image_embeds[b:b+1],
                        after.unsqueeze(0).reshape(1, -1, text_embeds.shape[-1])
                    ], dim=1)
                    merged_embeds_list.append(merged)

            # Pad to same length
            max_len = max(m.shape[1] for m in merged_embeds_list)
            padded_embeds = []
            for m in merged_embeds_list:
                if m.shape[1] < max_len:
                    pad = torch.zeros(
                        1, max_len - m.shape[1], m.shape[2],
                        device=m.device, dtype=m.dtype
                    )
                    m = torch.cat([m, pad], dim=1)
                padded_embeds.append(m)

            merged_embeds = torch.cat(padded_embeds, dim=0)
            merged_attention_mask = torch.ones(
                batch_size, merged_embeds.shape[1],
                device=input_ids.device,
                dtype=torch.long
            )

        return merged_embeds, merged_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            input_ids: Text token IDs [batch, seq_len]
            pixel_values: Images [batch, num_images, 3, H, W] or [batch, 3, H, W]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target token IDs for loss computation
            return_dict: Whether to return dict output

        Returns:
            Dictionary with logits, loss (if labels provided), router_logits
        """
        # Encode images if provided
        if pixel_values is not None:
            if pixel_values.dim() == 4:
                # Single image per sample: [B, 3, H, W]
                image_embeds = self.vision_encoder(pixel_values)
            else:
                # Multiple images: [B, N, 3, H, W]
                batch_size, num_images = pixel_values.shape[:2]
                flat_images = pixel_values.view(-1, *pixel_values.shape[2:])
                flat_embeds = self.vision_encoder(flat_images)
                image_embeds = flat_embeds.view(
                    batch_size, num_images * flat_embeds.shape[1], -1
                )
        else:
            image_embeds = None

        # Merge image and text embeddings
        if image_embeds is not None and input_ids is not None:
            inputs_embeds, attention_mask = self._merge_image_text_embeds(
                input_ids, image_embeds, attention_mask
            )
            # Forward through decoder with embeddings
            outputs = self.decoder(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_router_logits=True,
            )
        elif input_ids is not None:
            # Text-only forward
            outputs = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_router_logits=True,
            )
        else:
            raise ValueError("Either input_ids or pixel_values must be provided")

        logits = outputs[0]
        router_logits = outputs[1] if len(outputs) > 1 else None

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Add router auxiliary loss
            if router_logits is not None:
                aux_loss = self.decoder.compute_router_aux_loss(router_logits)
                loss = loss + self.config.router_aux_loss_coef * aux_loss

        if return_dict:
            return {
                "logits": logits,
                "loss": loss,
                "router_logits": router_logits,
            }

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        image: Optional[Union[str, torch.Tensor, "Image.Image"]] = None,
        prompt: str = "Describe this image.",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: bool = True,
    ) -> str:
        """
        Generate response for a single image and prompt.

        Args:
            image: Image path, tensor, or PIL Image
            prompt: Text prompt/question about the image
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            do_sample: Whether to use sampling (vs greedy)

        Returns:
            Generated text response
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Load and preprocess image
        if image is not None:
            pixel_values = self._prepare_image(image, device, dtype)
            image_embeds = self.vision_encoder(pixel_values)
        else:
            image_embeds = self._cached_image_embeds

        if image_embeds is None:
            raise ValueError("No image provided and no cached image available")

        # Tokenize prompt
        if self.tokenizer is not None:
            formatted_prompt = f"User: {IMAGE_TOKEN}\n{prompt}\nAssistant:"
            input_ids = self.tokenizer.encode(
                formatted_prompt,
                return_tensors="pt",
                add_special_tokens=True
            ).to(device)
        else:
            # Fallback: use dummy tokens
            input_ids = torch.tensor([[1, 2, 3]], device=device)

        # Merge embeddings
        inputs_embeds, attention_mask = self._merge_image_text_embeds(
            input_ids, image_embeds
        )

        # Generate tokens autoregressively
        generated_ids = []
        current_embeds = inputs_embeds

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.decoder(
                inputs_embeds=current_embeds,
                attention_mask=attention_mask,
            )
            logits = outputs[0]
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if temperature > 0 and do_sample:
                next_token_logits = next_token_logits / temperature

                # Apply top-k
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k
                    )[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Apply top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_ids.append(next_token.item())

            # Check for EOS
            if next_token.item() == self.config.eos_token_id:
                break

            # Update embeddings for next iteration
            next_embed = self.decoder.embed_tokens(next_token)
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, 1, device=device, dtype=attention_mask.dtype)
            ], dim=1)

        # Decode generated tokens
        if self.tokenizer is not None:
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            response = f"[Generated {len(generated_ids)} tokens]"

        return response.strip()

    def chat(
        self,
        image: Optional[Union[str, torch.Tensor, "Image.Image"]] = None,
        prompt: str = "",
        reset_history: bool = False,
        **generate_kwargs,
    ) -> str:
        """
        Multi-turn conversation interface.

        Args:
            image: Optional new image (uses cached if not provided)
            prompt: User message/question
            reset_history: Clear conversation history
            **generate_kwargs: Additional generation parameters

        Returns:
            Assistant response

        Example:
            >>> model.chat(image="photo.jpg", prompt="What do you see?")
            "I see a beautiful sunset over the ocean..."
            >>> model.chat(prompt="What colors are present?")
            "The image contains vibrant oranges, reds, and purples..."
        """
        if reset_history:
            self._conversation_history = []
            self._cached_image_embeds = None

        # Update cached image if new one provided
        if image is not None:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            pixel_values = self._prepare_image(image, device, dtype)
            self._cached_image_embeds = self.vision_encoder(pixel_values)

            # Add image to conversation history
            self._conversation_history.append({
                "role": "image",
                "content": "[Image provided]"
            })

        # Add user message
        self._conversation_history.append({
            "role": "user",
            "content": prompt
        })

        # Generate response
        response = self.generate(
            image=None,  # Use cached
            prompt=prompt,
            **generate_kwargs
        )

        # Add assistant response to history
        self._conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def _prepare_image(
        self,
        image: Union[str, torch.Tensor, "Image.Image"],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Load and preprocess an image."""
        if isinstance(image, torch.Tensor):
            return image.to(device=device, dtype=dtype)

        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image).convert("RGB")

        # Use vision encoder's preprocessing
        pixel_values = self.vision_encoder.preprocess_images([image])

        if isinstance(pixel_values, torch.Tensor):
            return pixel_values.to(device=device, dtype=dtype)

        return pixel_values

    def get_conversation_history(self) -> List[Dict]:
        """Get current conversation history."""
        return self._conversation_history.copy()

    def clear_history(self):
        """Clear conversation history and cached images."""
        self._conversation_history = []
        self._cached_image_embeds = None

    def save_pretrained(self, save_path: str):
        """Save model to directory."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save config
        import json
        config_dict = {k: v for k, v in vars(self.config).items()}
        with open(save_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save model weights
        torch.save(self.state_dict(), save_path / "model.pt")
        print(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cpu") -> "EmberNetVLM":
        """Load model from directory."""
        load_path = Path(load_path)

        # Load config
        import json
        with open(load_path / "config.json", "r") as f:
            config_dict = json.load(f)
        config = EmberNetConfig(**config_dict)

        # Create model
        model = cls(config)

        # Load weights
        state_dict = torch.load(load_path / "model.pt", map_location=device)
        model.load_state_dict(state_dict)

        return model.to(device)

    def get_model_size(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        vision_params = self.vision_encoder.get_num_params()
        decoder_params = self.decoder.get_num_params()

        return {
            "vision_encoder": vision_params,
            "decoder": decoder_params,
            "total": vision_params["total"] + decoder_params,
            "trainable": (
                self.vision_encoder.get_num_params(trainable_only=True)["total"] +
                self.decoder.get_num_params(trainable_only=True)
            ),
        }

    def print_model_summary(self):
        """Print a summary of the model architecture."""
        sizes = self.get_model_size()
        print("\n" + "="*50)
        print("EmberNet VLM Model Summary")
        print("="*50)
        print(f"\nVision Encoder ({self.config.vision_model_name}):")
        for k, v in sizes["vision_encoder"].items():
            print(f"  {k}: {v:,} params")
        print(f"\nLanguage Decoder (BitNet MoE):")
        print(f"  Layers: {self.config.num_layers}")
        print(f"  Hidden size: {self.config.hidden_size}")
        print(f"  Experts: {self.config.num_experts} (top-{self.config.num_experts_per_tok})")
        print(f"  Total params: {sizes['decoder']:,}")
        print(f"\nTotal Parameters: {sizes['total']:,}")
        print(f"Trainable Parameters: {sizes['trainable']:,}")
        print(f"\nImage tokens: {self.config.num_image_tokens}")
        print(f"Max sequence length: {self.config.max_position_embeddings}")
        print("="*50 + "\n")


# Convenience function to create model
def create_embernet_vlm(
    vision_model: str = "google/siglip-base-patch16-224",
    hidden_size: int = 768,
    num_layers: int = 16,
    num_experts: int = 8,
    **kwargs,
) -> EmberNetVLM:
    """Create an EmberNet VLM with specified configuration."""
    config = EmberNetConfig(
        vision_model_name=vision_model,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_experts=num_experts,
        **kwargs,
    )
    return EmberNetVLM(config)

