"""
User-Friendly Inference Interface for EmberNet VLM

This module provides a simple, high-level interface for interacting with
the EmberNet vision-language model. Users can ask questions about images,
have multi-turn conversations, and analyze various types of visual content.

USAGE EXAMPLES:

    # Basic image question
    >>> model = EmberVLM()
    >>> response = model.chat(image="photo.jpg", prompt="What's in this image?")
    >>> print(response)
    "I see a beautiful landscape with mountains and a lake..."

    # OCR / Text extraction
    >>> response = model.chat(image="document.pdf", prompt="Extract the text")
    >>> print(response)
    "The document contains: ..."

    # Multi-turn conversation
    >>> response = model.chat(image="chart.png", prompt="What does this chart show?")
    >>> print(response)  # First response about the chart
    >>> response = model.chat(prompt="What's the highest value?")
    >>> print(response)  # Follow-up using same image context

    # Analysis
    >>> response = model.chat(image="scene.jpg", prompt="Analyze this image")
    >>> print(response)
    "This image depicts..."

SUPPORTED IMAGE TYPES:
- Photos (JPEG, PNG, WebP)
- Documents (images of PDFs, scanned pages)
- Charts and graphs
- Screenshots
- Diagrams and technical drawings
- Mathematical figures
"""

import sys
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import torch

try:
    from codecarbon import EmissionsTracker as _EmissionsTracker
    _HAS_CODECARBON = True
except ImportError:
    _HAS_CODECARBON = False

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class EmberVLM:
    """
    High-level inference interface for EmberNet VLM.

    Provides an easy-to-use API for vision-language tasks:
    - Image understanding and description
    - Visual question answering
    - OCR and text extraction
    - Chart and diagram analysis
    - Multi-turn conversations

    Example:
        >>> model = EmberVLM("checkpoints/embernet.pt")
        >>> response = model.chat(image="photo.jpg", prompt="What do you see?")
        >>> print(response)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        use_va_refiner: bool = False,
        va_threshold: float = 0.7,
        va_burst_threshold: float = 0.7,
        va_soft_penalty: float = 5.0,
        va_alpha: float = 0.5,
    ):
        """
        Initialize the EmberVLM inference wrapper.

        Args:
            model_path: Path to model checkpoint (None for random init)
            device: Device to run on ("cuda", "cpu", or None for auto)
            torch_dtype: Data type (torch.float16, torch.float32, etc.)
            use_va_refiner: Enable VA Refiner hallucination mitigation
            va_threshold: VA score threshold for non-visual tokens
            va_burst_threshold: Burst detection threshold
            va_soft_penalty: Log-prob penalty applied during bursts
            va_alpha: Blend weight between neuron score and logit discrepancy
        """
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Determine dtype
        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.dtype = torch_dtype

        # Load or create model
        self._load_model(model_path)

        # Attach VA Refiner if requested
        if use_va_refiner:
            try:
                from models.va_refiner import VARefiner, VARefinerConfig
                va_cfg = VARefinerConfig(
                    use_va_refiner=True,
                    va_tau_non_visual=va_threshold,
                    va_burst_threshold=va_burst_threshold,
                    va_soft_penalty=va_soft_penalty,
                    va_alpha=va_alpha,
                )
                refiner = VARefiner(self.model, va_cfg, self.model.tokenizer)
                self.model.set_va_refiner(refiner)
                print("VA Refiner enabled.")
            except Exception as _va_err:
                print(f"[warning] VA Refiner could not be loaded: {_va_err}")

        # State
        self._current_image = None
        self._conversation_history: List[Dict[str, str]] = []

    def _load_model(self, model_path: Optional[str]):
        """Load model from checkpoint or create new."""
        from models import EmberNetVLM
        from models.vlm import EmberNetConfig

        if model_path is not None and Path(model_path).exists():
            print(f"Loading model from {model_path}...")
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                config = EmberNetConfig()
                self.model = EmberNetVLM(config)
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model = EmberNetVLM.from_pretrained(model_path)
        else:
            print("Initializing new model (no checkpoint loaded)...")
            config = EmberNetConfig()
            self.model = EmberNetVLM(config)

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        print(f"Model loaded on {self.device}")

    def chat(
        self,
        image: Optional[Union[str, "Image.Image", torch.Tensor]] = None,
        prompt: str = "",
        reset: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """
        Have a conversation about an image.

        Args:
            image: Image path, PIL Image, or tensor (uses cached if None)
            prompt: User question or instruction
            reset: Reset conversation history
            max_tokens: Maximum response length
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling threshold
            **kwargs: Additional generation parameters

        Returns:
            Model's text response

        Examples:
            # Ask about an image
            >>> response = model.chat(image="photo.jpg", prompt="What's in this image?")

            # Follow-up question (uses same image)
            >>> response = model.chat(prompt="What colors are visible?")

            # Reset conversation
            >>> response = model.chat(image="new_image.jpg", prompt="Describe this", reset=True)
        """
        if reset:
            self._conversation_history = []
            self._current_image = None

        # Update current image if new one provided
        if image is not None:
            self._current_image = self._load_image(image)
            self._conversation_history.append({
                "role": "system",
                "content": "[New image provided]"
            })

        if self._current_image is None and image is None:
            raise ValueError(
                "No image available. Provide an image or continue a conversation "
                "that already has an image."
            )

        # Add user message to history
        self._conversation_history.append({
            "role": "user",
            "content": prompt
        })

        # Generate response
        with torch.no_grad():
            response = self.model.generate(
                image=self._current_image if image is not None else None,
                prompt=self._build_prompt(),
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

        # Add assistant response to history
        self._conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def _build_prompt(self) -> str:
        """Build prompt from conversation history."""
        # Use only the last user message for now
        for msg in reversed(self._conversation_history):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    def _load_image(
        self,
        image: Union[str, "Image.Image", torch.Tensor]
    ) -> torch.Tensor:
        """Load and preprocess an image."""
        if isinstance(image, torch.Tensor):
            return image.to(device=self.device, dtype=self.dtype)

        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image).convert("RGB")

        # Use model's preprocessing
        pixel_values = self.model.vision_encoder.preprocess_images([image])

        return pixel_values.to(device=self.device, dtype=self.dtype)

    def describe(self, image: Union[str, "Image.Image", torch.Tensor]) -> str:
        """
        Get a detailed description of an image.

        Args:
            image: Image to describe

        Returns:
            Detailed description of the image
        """
        return self.chat(
            image=image,
            prompt="Please describe this image in detail, including all visible objects, "
                   "people, text, colors, and the overall scene.",
            reset=True,
        )

    def ocr(self, image: Union[str, "Image.Image", torch.Tensor]) -> str:
        """
        Extract text from an image (OCR).

        Args:
            image: Image containing text

        Returns:
            Extracted text from the image
        """
        return self.chat(
            image=image,
            prompt="Extract and transcribe all text visible in this image. "
                   "Maintain the original formatting and structure as much as possible.",
            reset=True,
        )

    def analyze_chart(self, image: Union[str, "Image.Image", torch.Tensor]) -> str:
        """
        Analyze a chart or graph.

        Args:
            image: Chart/graph image

        Returns:
            Analysis of the chart including key data points and trends
        """
        return self.chat(
            image=image,
            prompt="Analyze this chart/graph. Describe what it shows, identify key data points, "
                   "trends, and any notable patterns. If there are labels or values, mention them.",
            reset=True,
        )

    def answer(
        self,
        image: Union[str, "Image.Image", torch.Tensor],
        question: str,
    ) -> str:
        """
        Answer a specific question about an image.

        Args:
            image: Image to analyze
            question: Question about the image

        Returns:
            Answer to the question
        """
        return self.chat(image=image, prompt=question, reset=True)

    def get_expert_routing(self) -> Dict[str, Any]:
        """
        Get information about which experts were activated for the last query.

        Useful for understanding model behavior and debugging.
        """
        # This would require tracking router outputs during generation
        # For now, return placeholder
        return {
            "last_query": "N/A",
            "activated_experts": [],
            "routing_weights": [],
        }

    def clear_history(self):
        """Clear conversation history and cached image."""
        self._conversation_history = []
        self._current_image = None

    def get_history(self) -> List[Dict[str, str]]:
        """Get current conversation history."""
        return self._conversation_history.copy()


def load_model(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> EmberVLM:
    """
    Convenience function to load EmberVLM.

    Args:
        model_path: Path to model checkpoint
        device: Device to use (None for auto)

    Returns:
        Loaded EmberVLM instance
    """
    return EmberVLM(model_path=model_path, device=device)


def demo():
    """Run a demo of EmberVLM capabilities."""
    print("=" * 60)
    print("EmberNet VLM Demo")
    print("=" * 60)

    # Initialize model
    print("\nInitializing model...")
    model = EmberVLM()

    # Demo 1: Basic image description (with dummy image)
    print("\n--- Demo 1: Image Description ---")
    print("Creating dummy image for demonstration...")

    # Create a dummy colored image
    import numpy as np
    dummy_image = torch.randn(1, 3, 224, 224)

    print("Prompt: 'What do you see in this image?'")
    try:
        response = model.chat(image=dummy_image, prompt="What do you see in this image?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Note: Full generation requires a trained model. Error: {e}")

    # Demo 2: Multi-turn conversation
    print("\n--- Demo 2: Multi-turn Conversation ---")
    print("Prompt 1: 'Describe the colors in this image'")
    try:
        response = model.chat(prompt="Describe the colors in this image")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Note: {e}")

    print("Prompt 2: 'What is the dominant color?'")
    try:
        response = model.chat(prompt="What is the dominant color?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Note: {e}")

    # Model info
    print("\n--- Model Information ---")
    model.model.print_model_summary()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def interactive_cli():
    """Run an interactive command-line interface."""
    print("=" * 60)
    print("EmberNet VLM - Interactive Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  /load <path>  - Load an image")
    print("  /clear        - Clear conversation")
    print("  /describe     - Describe current image")
    print("  /ocr          - Extract text from image")
    print("  /chart        - Analyze chart/graph")
    print("  /quit         - Exit")
    print("\nOr just type a question about the loaded image.\n")

    model = EmberVLM()
    current_image_path = None

    _cc_tracker = None
    if _HAS_CODECARBON:
        try:
            _cc_tracker = _EmissionsTracker(
                project_name="EmberNet_inference",
                log_level="error",
                save_to_file=False,
            )
            _cc_tracker.start()
        except Exception:
            _cc_tracker = None

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

                if cmd == "/quit":
                    print("Goodbye!")
                    break
                elif cmd == "/clear":
                    model.clear_history()
                    current_image_path = None
                    print("Conversation cleared.")
                elif cmd == "/load":
                    if arg and Path(arg).exists():
                        current_image_path = arg
                        print(f"Loaded: {arg}")
                    else:
                        print(f"File not found: {arg}")
                elif cmd == "/describe":
                    if current_image_path:
                        response = model.describe(current_image_path)
                        print(f"Assistant: {response}")
                    else:
                        print("No image loaded. Use /load <path> first.")
                elif cmd == "/ocr":
                    if current_image_path:
                        response = model.ocr(current_image_path)
                        print(f"Assistant: {response}")
                    else:
                        print("No image loaded. Use /load <path> first.")
                elif cmd == "/chart":
                    if current_image_path:
                        response = model.analyze_chart(current_image_path)
                        print(f"Assistant: {response}")
                    else:
                        print("No image loaded. Use /load <path> first.")
                else:
                    print(f"Unknown command: {cmd}")
            else:
                # Regular chat
                try:
                    if current_image_path and model._current_image is None:
                        response = model.chat(image=current_image_path, prompt=user_input)
                    else:
                        response = model.chat(prompt=user_input)
                    print(f"Assistant: {response}")
                except ValueError as e:
                    print(f"Error: {e}")
                except Exception as e:
                    print(f"Error during generation: {e}")
    finally:
        if _cc_tracker is not None:
            try:
                _emissions_kg = _cc_tracker.stop()
                _kwh = float(_cc_tracker.final_emissions_data.energy_consumed)
                print(f"\n[energy] Session: {_kwh:.4f} kWh  |  CO\u2082: {_emissions_kg:.6f} kg")
            except Exception:
                pass


def main():
    """Main entry point for inference scripts."""
    import argparse

    parser = argparse.ArgumentParser(description="EmberNet VLM Inference")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image file")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt/question about the image")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--use-va-refiner", action="store_true",
                        help="Enable VA Refiner hallucination mitigation")
    parser.add_argument("--va-threshold", type=float, default=0.7,
                        help="VA score threshold for non-visual tokens (default: 0.7)")
    parser.add_argument("--va-burst-threshold", type=float, default=0.7,
                        help="Burst detection threshold (default: 0.7)")
    parser.add_argument("--va-soft-penalty", type=float, default=5.0,
                        help="Log-prob penalty during hallucination bursts (default: 5.0)")
    parser.add_argument("--va-alpha", type=float, default=0.5,
                        help="Blend weight: neuron score vs logit discrepancy (default: 0.5)")

    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.interactive:
        interactive_cli()
    elif args.image and args.prompt:
        # Single inference with optional energy tracking
        _cc = None
        if _HAS_CODECARBON:
            try:
                _cc = _EmissionsTracker(project_name="EmberNet_inference", log_level="error", save_to_file=False)
                _cc.start()
            except Exception:
                _cc = None
        model = EmberVLM(
            model_path=args.model,
            device=args.device,
            use_va_refiner=args.use_va_refiner,
            va_threshold=args.va_threshold,
            va_burst_threshold=args.va_burst_threshold,
            va_soft_penalty=args.va_soft_penalty,
            va_alpha=args.va_alpha,
        )
        response = model.chat(image=args.image, prompt=args.prompt)
        print(f"Response: {response}")
        if _cc is not None:
            try:
                _em = _cc.stop()
                _kw = float(_cc.final_emissions_data.energy_consumed)
                print(f"[energy] {_kw:.4f} kWh  |  CO\u2082: {_em:.6f} kg")
            except Exception:
                pass
    else:
        # Default to interactive
        interactive_cli()


if __name__ == "__main__":
    main()

