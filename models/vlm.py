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
    vocab_size: int = 32002  # 32000 base + special tokens (IMAGE_TOKEN_ID=32001)
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

        # Initialize decoder embeddings with small values for stability
        if hasattr(self.decoder, 'embed_tokens'):
            nn.init.normal_(self.decoder.embed_tokens.weight, mean=0.0, std=0.02)

        # Image token embedding placeholder
        self.image_newline = nn.Parameter(
            torch.randn(self.config.hidden_size) * 0.02
        )

        # Initialize decoder weights for stability
        self._initialize_decoder()

        # Conversation history for multi-turn
        self._conversation_history: List[Dict] = []
        self._cached_image_embeds: Optional[torch.Tensor] = None

        # Tokenizer (loaded lazily)
        self._tokenizer = None

    def _initialize_decoder(self):
        """Initialize decoder with small weights to prevent NaN during Stage 1."""
        from .bitnet_moe import BitLinear, RMSNorm

        for name, module in self.decoder.named_modules():
            # Skip BitLinear - it has its own initialization
            if isinstance(module, BitLinear):
                continue
            # Skip RMSNorm - it has its own initialization
            if isinstance(module, RMSNorm):
                continue

            if isinstance(module, nn.Linear):
                if module.weight is not None and module.weight.dim() >= 2:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        print("✓ Decoder initialized with small weights for stability")

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                # Use TinyLlama tokenizer - vocab_size=32000 matches our model
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
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
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Merge image embeddings with text embeddings by prepending image embeddings.

        Args:
            input_ids: Text token IDs [batch, text_seq_len]
            image_embeds: Image embeddings [batch, num_image_tokens, hidden]
            attention_mask: Attention mask for text

        Returns:
            merged_embeds: Combined embeddings [batch, total_seq_len, hidden]
            merged_attention_mask: Updated attention mask
            label_offsets: List of offsets where text labels should start for each batch item
        """
        batch_size = input_ids.shape[0]
        num_image_tokens = image_embeds.shape[1]

        # Replace any special image tokens with pad for embedding lookup
        safe_input_ids = input_ids.clone()
        safe_input_ids[safe_input_ids == self.config.image_token_id] = self.config.pad_token_id
        # Also clamp any out-of-range token IDs
        safe_input_ids = safe_input_ids.clamp(0, self.config.vocab_size - 1)

        # Embed text tokens
        text_embeds = self.decoder.embed_tokens(safe_input_ids)

        # Always prepend image embeddings to text embeddings
        merged_embeds = torch.cat([image_embeds, text_embeds], dim=1)


        # Update attention mask
        if attention_mask is not None:
            image_mask = torch.ones(
                batch_size, num_image_tokens,
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            merged_attention_mask = torch.cat([image_mask, attention_mask], dim=1)
        else:
            merged_attention_mask = torch.ones(
                batch_size, merged_embeds.shape[1],
                device=input_ids.device,
                dtype=torch.long
            )

        # Text labels start after image tokens
        label_offsets = [num_image_tokens] * batch_size

        return merged_embeds, merged_attention_mask, label_offsets

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
        adjusted_labels = labels
        if image_embeds is not None and input_ids is not None:
            inputs_embeds, attention_mask, label_offsets = self._merge_image_text_embeds(
                input_ids, image_embeds, attention_mask
            )
            target_len = inputs_embeds.shape[1]

            # Adjust labels to match merged sequence length
            if labels is not None:
                # Ensure labels is a tensor
                labels_tensor = labels
                if not isinstance(labels_tensor, torch.Tensor):
                    labels_tensor = torch.tensor(labels_tensor, device=inputs_embeds.device)

                batch_size = labels_tensor.shape[0]

                new_adjusted_labels = []
                for b in range(batch_size):
                    offset = label_offsets[b] if b < len(label_offsets) else label_offsets[0]
                    # Create new labels with ignore tokens for image positions
                    row_labels = torch.full(
                        (target_len,), -100,
                        dtype=labels_tensor.dtype, device=labels_tensor.device
                    )
                    # Copy original text labels starting after the offset
                    text_labels = labels_tensor[b]
                    copy_len = min(len(text_labels), target_len - offset)
                    if copy_len > 0:
                        row_labels[offset:offset + copy_len] = text_labels[:copy_len]
                    new_adjusted_labels.append(row_labels)

                adjusted_labels = torch.stack(new_adjusted_labels, dim=0)

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

        # Clamp logits to reasonable range to prevent extreme values while preserving gradients
        logits = torch.clamp(logits, min=-100.0, max=100.0)

        # Compute loss if labels provided
        loss = None
        if adjusted_labels is not None:
            # Ensure batch sizes match
            if logits.shape[0] != adjusted_labels.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: logits {logits.shape[0]} vs labels {adjusted_labels.shape[0]}"
                )

            # Align sequence lengths (may differ due to image token injection)
            if logits.shape[1] != adjusted_labels.shape[1]:
                min_len = min(logits.shape[1], adjusted_labels.shape[1])
                logits = logits[:, :min_len, :]
                adjusted_labels = adjusted_labels[:, :min_len]

            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = adjusted_labels[..., 1:].contiguous()

            # Ensure exact same sequence length after shift
            if shift_logits.shape[1] != shift_labels.shape[1]:
                min_len = min(shift_logits.shape[1], shift_labels.shape[1])
                shift_logits = shift_logits[:, :min_len, :]
                shift_labels = shift_labels[:, :min_len]

            # Final shape verification - ensure exact match before flattening
            batch_size = shift_logits.shape[0]
            logits_seq_len = shift_logits.shape[1]
            labels_seq_len = shift_labels.shape[1]
            vocab_size = shift_logits.shape[2]

            # Force exact match
            if logits_seq_len != labels_seq_len:
                min_len = min(logits_seq_len, labels_seq_len)
                shift_logits = shift_logits[:, :min_len, :]
                shift_labels = shift_labels[:, :min_len]
                logits_seq_len = min_len

            # Flatten for cross-entropy
            flat_logits = shift_logits.contiguous().view(-1, vocab_size)
            flat_labels = shift_labels.contiguous().view(-1)

            # Final sanity check - they must be same size
            if flat_logits.shape[0] != flat_labels.shape[0]:
                print(f"WARNING: Size mismatch before loss - logits: {flat_logits.shape}, labels: {flat_labels.shape}")
                print(f"  Original shapes - shift_logits: {shift_logits.shape}, shift_labels: {shift_labels.shape}")
                min_size = min(flat_logits.shape[0], flat_labels.shape[0])
                flat_logits = flat_logits[:min_size]
                flat_labels = flat_labels[:min_size]
                print(f"  Adjusted to size: {min_size}")

            # Verify sizes match
            assert flat_logits.shape[0] == flat_labels.shape[0], \
                f"Size mismatch after adjustment: logits {flat_logits.shape[0]} vs labels {flat_labels.shape[0]}"

            # Clamp logits to prevent overflow in cross-entropy
            flat_logits = flat_logits.clamp(-100.0, 100.0)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
            loss = loss_fct(flat_logits, flat_labels)

            # Check for NaN loss - fail fast if it occurs (indicates upstream problem)
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError("NaN/Inf loss detected - indicates upstream numerical instability")

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
        use_cache: bool = True,
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
            use_cache: Whether to use KV-cache for faster generation

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

        # Generate tokens autoregressively with KV-cache
        generated_ids = []
        past_key_values = None

        # First forward pass with full context
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
        logits = outputs[0]
        if use_cache:
            past_key_values = outputs[1]

        for step in range(max_new_tokens):
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

            # Next forward pass - only process the new token using cache
            if use_cache and past_key_values is not None:
                next_embed = self.decoder.embed_tokens(next_token)
                new_attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(1, 1, device=device, dtype=attention_mask.dtype)
                ], dim=1)
                outputs = self.decoder(
                    inputs_embeds=next_embed,
                    attention_mask=new_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs[0]
                past_key_values = outputs[1]
                attention_mask = new_attention_mask
            else:
                # No cache - recompute everything (slower)
                next_embed = self.decoder.embed_tokens(next_token)
                inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(1, 1, device=device, dtype=attention_mask.dtype)
                ], dim=1)
                outputs = self.decoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
                logits = outputs[0]

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
        state_dict = torch.load(load_path / "model.pt", map_location=device, weights_only=False)
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

