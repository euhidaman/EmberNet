"""
Training Script for EmberNet VLM

Two-stage training pipeline:
1. Stage 1 - Projector Alignment: Train only the vision projector while
   keeping the vision encoder and most of the LM frozen
2. Stage 2 - Expert SFT: Train the router and experts on domain-specific data

TRAINING STRATEGY:
- Stage 1: ~1-3 epochs on general VLM data, LR=1e-3
- Stage 2: ~5-10 epochs on mixed domain data, LR=1e-4
- AdamW optimizer with cosine schedule
- Gradient clipping for stable BitNet training

USAGE:
    # Stage 1: Projector alignment
    python training/train.py --stage 1 --epochs 3 --batch-size 8

    # Stage 2: Expert SFT
    python training/train.py --stage 2 --epochs 10 --batch-size 4
"""

import sys
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Weights & Biases for experiment tracking
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Install with 'pip install wandb' for experiment tracking.")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import EmberNetVLM
from models.vlm import EmberNetConfig
from training.data import DataConfig, create_dataloaders


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Training stage
    stage: int = 1  # 1=projector alignment, 2=expert SFT

    # Paths
    output_dir: str = "./checkpoints"
    data_dir: str = "./data"

    # Experiment tracking
    use_wandb: bool = True
    wandb_project: str = "EmberNet"
    wandb_run_name: Optional[str] = None
    resume_from: Optional[str] = None

    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-3  # Higher for stage 1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # Stage-specific settings
    freeze_vision: bool = True
    freeze_lm_layers: bool = True  # For stage 1
    train_router: bool = False  # Enable for stage 2
    train_experts: bool = False  # Enable for stage 2

    # Curriculum learning (Stage 2 only)
    use_curriculum: bool = True

    # EMA (exponential moving average of model weights)
    use_ema: bool = True
    ema_decay: float = 0.999

    # Adaptive gradient clipping
    use_adaptive_grad_clip: bool = True
    grad_clip_percentile: float = 0.95

    # Token masking (future use)
    token_masking_prob: float = 0.0

    # Expert supervision (Stage 2)
    use_expert_supervision: bool = True
    expert_supervision_weight: float = 0.1

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    num_workers: int = 4

    # Model config override
    model_config: Optional[EmberNetConfig] = None

    def __post_init__(self):
        # Adjust settings based on stage
        if self.stage == 1:
            self.learning_rate = 1e-3
            self.freeze_lm_layers = True
            self.train_router = False
            self.train_experts = False
        elif self.stage == 2:
            self.learning_rate = 1e-4
            self.freeze_lm_layers = False
            self.train_router = True
            self.train_experts = True


class Trainer:
    """
    Trainer for EmberNet VLM.

    Handles two-stage training with proper parameter freezing
    and logging.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize model
        model_config = config.model_config or EmberNetConfig()
        self.model = EmberNetVLM(model_config)

        # Resume from checkpoint if specified
        if config.resume_from:
            self._load_checkpoint(config.resume_from)

        self.model = self.model.to(self.device)

        # EMA setup
        self.ema_model = None
        if config.use_ema:
            from copy import deepcopy
            self.ema_model = deepcopy(self.model)
            for p in self.ema_model.parameters():
                p.requires_grad = False
            print("✓ EMA model initialized")

        # Apply parameter freezing based on stage
        self._setup_parameter_freezing()

        # Print model summary
        self.model.print_model_summary()

        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = None  # Created after dataloader is ready

        # Mixed precision
        self.scaler = None
        if config.mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()

        # Weights & Biases initialization
        self.use_wandb = config.use_wandb and HAS_WANDB
        if self.use_wandb:
            run_name = config.wandb_run_name or f"stage{config.stage}_bs{config.batch_size}"
            wandb.init(
                project=config.wandb_project,
                name=run_name,
                config={
                    "stage": config.stage,
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "model_config": {
                        "hidden_size": model_config.hidden_size,
                        "num_layers": model_config.num_layers,
                        "num_experts": model_config.num_experts,
                        "num_experts_per_tok": model_config.num_experts_per_tok,
                    }
                }
            )
            # Watch model gradients and parameters
            wandb.watch(self.model, log="all", log_freq=100)
            print(f"✓ Initialized Weights & Biases: {config.wandb_project}/{run_name}")

        # Logging
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_log: List[Dict] = []

    def _setup_parameter_freezing(self):
        """Freeze parameters based on training stage."""
        config = self.config

        # Always freeze vision encoder (unless fine-tuning)
        if config.freeze_vision:
            for param in self.model.vision_encoder.encoder.parameters():
                param.requires_grad = False

        if config.stage == 1:
            # Stage 1: Only train projector
            for param in self.model.decoder.parameters():
                param.requires_grad = False

            # Unfreeze projector
            for param in self.model.vision_encoder.projector.parameters():
                param.requires_grad = True

            if self.model.vision_encoder.pooler is not None:
                for param in self.model.vision_encoder.pooler.parameters():
                    param.requires_grad = True

            if self.model.vision_encoder.compressor is not None:
                for param in self.model.vision_encoder.compressor.parameters():
                    param.requires_grad = True

        elif config.stage == 2:
            # Stage 2: Train router and experts
            # Keep embeddings and some layers frozen for stability
            for name, param in self.model.decoder.named_parameters():
                if "embed_tokens" in name:
                    param.requires_grad = False
                elif "router" in name and config.train_router:
                    param.requires_grad = True
                elif "experts" in name and config.train_experts:
                    param.requires_grad = True
                elif "shared_expert" in name and config.train_experts:
                    param.requires_grad = True
                else:
                    param.requires_grad = config.train_experts

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"\nStage {config.stage} Training:")
        print(f"  Trainable parameters: {trainable:,}")
        print(f"  Total parameters: {total:,}")
        print(f"  Trainable ratio: {trainable/total*100:.2f}%\n")

    def _create_optimizer(self) -> AdamW:
        """Create optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return AdamW(optimizer_groups, lr=self.config.learning_rate)

    def _create_scheduler(self, num_training_steps: int):
        """Create cosine annealing scheduler with warmup."""
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - self.config.warmup_steps,
            eta_min=self.config.learning_rate * 0.1,
        )

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.global_step = checkpoint.get("global_step", 0)
        else:
            self.model.load_state_dict(checkpoint)

    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }

        save_path = output_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")

        if is_best:
            best_path = output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def _train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step."""
        # Move batch to device
        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"]
        else:
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"]

        # Optional expert supervision (Stage 2 only)
        if (
            self.config.stage == 2
            and self.config.use_expert_supervision
            and "expert_targets" in batch
            and outputs.get("router_logits", None) is not None
        ):
            router_logits_list = outputs["router_logits"]
            router_logits_last = router_logits_list[-1]
            expert_targets = batch["expert_targets"].to(router_logits_last.device)
            seq_len = input_ids.shape[1]
            expanded_targets = expert_targets.unsqueeze(1).expand(-1, seq_len).reshape(-1)

            expert_supervision_loss = torch.nn.functional.cross_entropy(
                router_logits_last, expanded_targets, reduction="mean"
            )
            loss = loss + self.config.expert_supervision_weight * expert_supervision_loss

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
        }

    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # Adaptive or fixed gradient clipping
        if self.config.use_adaptive_grad_clip:
            grads = []
            for p in self.model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().abs().view(-1))
            if grads:
                all_grads = torch.cat(grads)
                clip_val = torch.quantile(all_grads, self.config.grad_clip_percentile)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val.item())
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

        # Scheduler step
        if self.scheduler is not None and self.global_step >= self.config.warmup_steps:
            self.scheduler.step()

        # EMA update
        self._update_ema()

    def _update_ema(self):
        """Update EMA model weights."""
        if self.ema_model is None:
            return
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.config.ema_decay).add_(
                    param.data, alpha=1.0 - self.config.ema_decay
                )

    def _warmup_lr(self):
        """Linear warmup for learning rate."""
        if self.global_step < self.config.warmup_steps:
            warmup_factor = self.global_step / self.config.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.config.learning_rate * warmup_factor

    def _calculate_token_statistics(self, train_loader):
        """Calculate token statistics from a sample batch."""
        try:
            # Get first batch
            sample_batch = next(iter(train_loader))

            # Get dimensions
            batch_size = sample_batch["input_ids"].shape[0]
            seq_length = sample_batch["input_ids"].shape[1]

            # Check if we have images (pixel_values present)
            has_images = "pixel_values" in sample_batch and sample_batch["pixel_values"] is not None

            if has_images:
                # Vision encoder outputs 64 tokens (from VisionEncoder config)
                num_image_tokens = self.model.vision_encoder.num_output_tokens if hasattr(self.model, 'vision_encoder') else 64
                num_text_tokens = seq_length - num_image_tokens
            else:
                # No images, all text
                num_image_tokens = 0
                num_text_tokens = seq_length

            total_tokens = seq_length

            return {
                "total_tokens_per_sample": total_tokens,
                "image_tokens_per_sample": num_image_tokens,
                "text_tokens_per_sample": num_text_tokens,
                "batch_size": batch_size,
            }
        except Exception as e:
            print(f"Warning: Could not calculate token statistics: {e}")
            return None

    def train(
        self,
        train_loader=None,
        val_loader=None,
    ):
        """
        Main training loop.

        Args:
            train_loader: Training dataloader (created if not provided)
            val_loader: Validation dataloader (optional)
        """
        # Create dataloaders if not provided
        if train_loader is None:
            data_config = DataConfig(data_dir=self.config.data_dir)
            train_loader, val_loader = create_dataloaders(
                data_config,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                stage=self.config.stage,
                use_curriculum=self.config.use_curriculum,
            )

        # Calculate training steps
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.config.epochs

        # Create scheduler
        self._create_scheduler(total_steps)

        # Calculate token statistics
        token_stats = self._calculate_token_statistics(train_loader)

        print(f"\n{'='*70}")
        print(f"Starting Stage {self.config.stage} Training")
        print(f"{'='*70}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total steps: {total_steps}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Device: {self.device}")

        # Print token statistics
        if token_stats:
            print(f"\n--- Token Statistics (per sample) ---")
            print(f"Total tokens:  {token_stats['total_tokens_per_sample']:,}")
            print(f"  ├─ Image tokens: {token_stats['image_tokens_per_sample']:,}")
            print(f"  └─ Text tokens:  {token_stats['text_tokens_per_sample']:,}")
            print(f"\n--- Token Statistics (per batch) ---")
            total_batch = token_stats['total_tokens_per_sample'] * token_stats['batch_size']
            image_batch = token_stats['image_tokens_per_sample'] * token_stats['batch_size']
            text_batch = token_stats['text_tokens_per_sample'] * token_stats['batch_size']
            print(f"Total tokens:  {total_batch:,}")
            print(f"  ├─ Image tokens: {image_batch:,}")
            print(f"  └─ Text tokens:  {text_batch:,}")
            print(f"\n--- Total Training Tokens (all epochs) ---")
            total_training = total_batch * total_steps
            image_training = image_batch * total_steps
            text_training = text_batch * total_steps
            print(f"Total tokens:  {total_training:,}")
            print(f"  ├─ Image tokens: {image_training:,}")
            print(f"  └─ Text tokens:  {text_training:,}")

        print(f"{'='*70}\n")

        self.model.train()
        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            for step, batch in enumerate(train_loader):
                # Warmup
                self._warmup_lr()

                # Training step
                metrics = self._train_step(batch)
                epoch_loss += metrics["loss"]
                epoch_steps += 1

                # Optimizer step (after accumulation)
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self._optimizer_step()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.log_interval == 0:
                        avg_loss = epoch_loss / epoch_steps
                        lr = self.optimizer.param_groups[0]["lr"]
                        elapsed = time.time() - start_time

                        print(
                            f"Epoch {epoch+1}/{self.config.epochs} | "
                            f"Step {self.global_step} | "
                            f"Loss: {metrics['loss']:.4f} | "
                            f"Avg Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Time: {elapsed:.1f}s"
                        )

                        log_dict = {
                            "train/loss": metrics["loss"],
                            "train/avg_loss": avg_loss,
                            "train/lr": lr,
                            "train/epoch": epoch + 1,
                            "train/step": self.global_step,
                        }

                        self.training_log.append({
                            "step": self.global_step,
                            "loss": metrics["loss"],
                            "avg_loss": avg_loss,
                            "lr": lr,
                        })

                        # Log to wandb
                        if self.use_wandb:
                            wandb.log(log_dict, step=self.global_step)

                    # Evaluation
                    if val_loader is not None and self.global_step % self.config.eval_interval == 0:
                        val_loss = self.evaluate(val_loader)
                        print(f"Validation Loss: {val_loss:.4f}")

                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            self._save_checkpoint("best_model.pt", is_best=True)

                        # Log validation to wandb
                        if self.use_wandb:
                            wandb.log({
                                "val/loss": val_loss,
                                "val/best_loss": self.best_loss,
                            }, step=self.global_step)

                        self.model.train()

                    # Save checkpoint
                    if self.global_step % self.config.save_interval == 0:
                        self._save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

            # End of epoch
            epoch_avg_loss = epoch_loss / epoch_steps
            print(f"\nEpoch {epoch+1} Complete | Average Loss: {epoch_avg_loss:.4f}\n")

            # Log epoch metrics to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch/avg_loss": epoch_avg_loss,
                    "epoch/number": epoch + 1,
                }, step=self.global_step)

            # Save epoch checkpoint
            self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

        # Final save
        self._save_checkpoint("final_model.pt")

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_loss:.4f}")

        # Finish wandb run
        if self.use_wandb:
            wandb.finish()
            print("✓ Weights & Biases run completed")

    @torch.no_grad()
    def evaluate(self, val_loader) -> float:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs["loss"].item()
            num_batches += 1

        return total_loss / num_batches


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train EmberNet VLM")

    # Training settings
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Training stage (1=projector, 2=expert SFT)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (auto-set based on stage)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")

    # Paths
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Data directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

    # Hardware
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda/cpu/auto)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")

    # New training features
    parser.add_argument("--no-ema", action="store_true",
                        help="Disable EMA model")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning")
    parser.add_argument("--no-expert-supervision", action="store_true",
                        help="Disable expert supervision loss")
    parser.add_argument("--no-adaptive-clip", action="store_true",
                        help="Disable adaptive gradient clipping")

    # Experiment tracking
    parser.add_argument("--wandb", action="store_true", default=True,
                        help="Use Weights & Biases for logging (default: True)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="EmberNet",
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (default: auto-generated)")

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Create config
    config = TrainingConfig(
        stage=args.stage,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        resume_from=args.resume,
        device=device,
        mixed_precision=not args.no_amp,
        num_workers=args.num_workers,
        use_wandb=args.wandb and not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        use_ema=not args.no_ema,
        use_curriculum=not args.no_curriculum,
        use_expert_supervision=not args.no_expert_supervision,
        use_adaptive_grad_clip=not args.no_adaptive_clip,
    )

    if args.lr is not None:
        config.learning_rate = args.lr

    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

