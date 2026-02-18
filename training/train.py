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
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Iterator

import torch
import torch.nn as nn
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


# =============================================================================
# BitNet b1.58 Training Stability Classes (Based on Microsoft Research)
# =============================================================================

class BitNetStableOptimizer:
    """
    Specialized optimizer for BitNet b1.58 ternary quantization.
    Implements Microsoft's two-phase LR schedule and FP32 gradient handling.
    """
    def __init__(
        self,
        parameters: Iterator[nn.Parameter],
        lr: float = 6e-4,
        phase1_steps: Optional[int] = None,
        phase2_lr_factor: float = 0.1,
        warmup_steps: int = 2000,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8
    ):
        self.param_groups = [{'params': list(parameters)}]
        self.lr = lr
        self.phase1_steps = phase1_steps
        self.phase2_lr_factor = phase2_lr_factor
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.betas = betas
        self.eps = eps

        self.state = {}
        for param in self.param_groups[0]['params']:
            self.state[param] = {
                'step': 0,
                'exp_avg': torch.zeros_like(param.data, dtype=torch.float32),
                'exp_avg_sq': torch.zeros_like(param.data, dtype=torch.float32)
            }
        self.global_step = 0

    def _get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr * (step / max(1, self.warmup_steps))
        if self.phase1_steps is None or step < self.phase1_steps:
            return self.lr
        return self.lr * self.phase2_lr_factor

    def zero_grad(self):
        for param in self.param_groups[0]['params']:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def step(self):
        self.global_step += 1
        current_lr = self._get_lr(self.global_step)

        for param in self.param_groups[0]['params']:
            if param.grad is None:
                continue

            # Check for NaN/Inf gradients - clamp to safe range instead of zeroing
            if not torch.isfinite(param.grad).all():
                # Replace NaN/Inf with clamped finite values
                grad_finite = torch.where(
                    torch.isfinite(param.grad),
                    param.grad,
                    torch.zeros_like(param.grad)
                )
                # Clamp to prevent extreme updates
                param.grad = torch.clamp(grad_finite, min=-10.0, max=10.0)

            grad = param.grad.data
            state = self.state[param]

            if self.grad_clip > 0:
                grad_norm = grad.norm()
                if grad_norm > self.grad_clip:
                    grad = grad * (self.grad_clip / (grad_norm + 1e-6))

            grad = grad.to(torch.float32)
            state['step'] += 1
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = self.betas

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            step_size = current_lr / bias_correction1
            bias_correction2_sqrt = math.sqrt(bias_correction2)

            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(self.eps)
            update = exp_avg / denom

            if self.weight_decay > 0:
                update.add_(param.data.to(torch.float32), alpha=self.weight_decay)

            param.data.add_(update.to(param.dtype), alpha=-step_size)

    def state_dict(self):
        """Return optimizer state for checkpointing."""
        return {
            'global_step': self.global_step,
            'lr': self.lr,
            'phase1_steps': self.phase1_steps,
            'phase2_lr_factor': self.phase2_lr_factor,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'grad_clip': self.grad_clip,
            'betas': self.betas,
            'eps': self.eps,
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state from checkpoint."""
        self.global_step = state_dict.get('global_step', 0)
        self.lr = state_dict.get('lr', self.lr)
        self.phase1_steps = state_dict.get('phase1_steps', self.phase1_steps)
        self.phase2_lr_factor = state_dict.get('phase2_lr_factor', self.phase2_lr_factor)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.weight_decay = state_dict.get('weight_decay', self.weight_decay)
        self.grad_clip = state_dict.get('grad_clip', self.grad_clip)
        self.betas = state_dict.get('betas', self.betas)
        self.eps = state_dict.get('eps', self.eps)


class BitNetGradientScaler:
    """Custom gradient scaler for BitNet b1.58 to prevent underflow."""
    def __init__(self, init_scale: float = 2**10):
        self.scale = init_scale
        self.growth_factor = 2.0
        self.backoff_factor = 0.5
        self.growth_interval = 2000
        self._growth_tracker = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale

    def unscale_gradients(self, optimizer):
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.div_(self.scale)

    def update(self, found_inf: bool):
        if found_inf:
            self.scale *= self.backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self._growth_tracker = 0


def check_gradients_finite(model: nn.Module) -> Tuple[bool, Optional[str]]:
    """Check if all gradients are finite."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                return False, name
    return True, None


def emergency_gradient_fix(model: nn.Module):
    """Emergency gradient clipping if NaN detected."""
    for param in model.parameters():
        if param.grad is not None:
            param.grad = torch.where(
                torch.isfinite(param.grad),
                param.grad,
                torch.zeros_like(param.grad)
            )
            param.grad.clamp_(-0.1, 0.1)


def initialize_bitnet_weights(model: nn.Module):
    """Proper initialization for BitNet b1.58 ternary weights."""
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if module.weight.dim() >= 2:
                fan_in = module.weight.size(1)
                std = math.sqrt(2.0 / fan_in) * 0.5
                nn.init.normal_(module.weight, mean=0.0, std=std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)


# =============================================================================
# Training Configuration
# =============================================================================

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
    learning_rate: float = 1e-4  # Lower for BitNet stability
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5  # Tighter clipping for stability
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

    # Data limiting (for trial runs)
    max_samples_per_dataset: Optional[int] = None  # None = use all samples

    # Model config override
    model_config: Optional[EmberNetConfig] = None

    # BitNet-specific settings
    use_bitnet_optimizer: bool = True  # Use BitNet stable optimizer
    bitnet_phase1_ratio: float = 0.6  # 60% of training in phase 1
    bitnet_phase2_lr_factor: float = 0.1  # Drop LR by 10x in phase 2

    def __post_init__(self):
        # Adjust settings based on stage - using BitNet methodology
        if self.stage == 1:
            # Lower LR for vision encoder components (they're not BitNet quantized)
            self.learning_rate = 1e-4  # Conservative LR for vision projector/compressor
            self.warmup_steps = 100  # Shorter warmup
            self.freeze_lm_layers = True
            self.train_router = False
            self.train_experts = False
            self.max_grad_norm = 1.0  # BitNet uses 1.0 clipping
            self.use_adaptive_grad_clip = False  # Disable adaptive, use fixed
        elif self.stage == 2:
            # BitNet decoder can handle higher LR
            self.learning_rate = 3e-4  # Higher for BitNet decoder
            self.warmup_steps = 100
            self.freeze_lm_layers = False
            self.train_router = True
            self.train_experts = True
            self.max_grad_norm = 1.0


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
        else:
            # Apply BitNet-specific weight initialization for fresh training
            print("Applying BitNet weight initialization...")
            initialize_bitnet_weights(self.model)
            print("✓ BitNet weights initialized")

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

        # Initialize optimizer - use BitNet stable optimizer if enabled
        self.use_bitnet_optimizer = config.use_bitnet_optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = None  # Created after dataloader is ready

        # BitNet gradient scaler (replaces standard AMP scaler)
        self.bitnet_scaler = None
        self.scaler = None
        if config.use_bitnet_optimizer:
            # Use BitNet-specific gradient scaler
            self.bitnet_scaler = BitNetGradientScaler(init_scale=1024)
            print("✓ BitNet gradient scaler initialized")
        elif config.mixed_precision and self.device.type == "cuda":
            # Use standard AMP scaler
            try:
                self.scaler = torch.amp.GradScaler('cuda')
            except (TypeError, AttributeError):
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
                    "use_bitnet_optimizer": config.use_bitnet_optimizer,
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

    def _create_optimizer(self):
        """Create optimizer - uses BitNetStableOptimizer if enabled."""
        # Collect trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if self.config.use_bitnet_optimizer:
            # Use BitNet-specific optimizer with two-phase LR schedule
            print(f"Using BitNetStableOptimizer with LR={self.config.learning_rate}")
            return BitNetStableOptimizer(
                iter(trainable_params),
                lr=self.config.learning_rate,
                phase1_steps=None,  # Set later when we know total steps
                phase2_lr_factor=self.config.bitnet_phase2_lr_factor,
                warmup_steps=self.config.warmup_steps,
                weight_decay=self.config.weight_decay,
                grad_clip=self.config.max_grad_norm,
            )
        else:
            # Use standard AdamW
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

        # PyTorch 2.6+ changed weights_only default to True
        # We need weights_only=False for checkpoints with config objects
        # Add safe_globals to allow TrainingConfig
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Error loading checkpoint with weights_only=False: {e}")
            # Fallback: try with weights_only=True if checkpoint is simple
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except Exception as e2:
                print(f"Error loading checkpoint with weights_only=True: {e2}")
                raise e

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
            "stage": self.config.stage,
            "learning_rate": self.config.learning_rate,
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
        """Training step with BitNet-specific gradient handling."""
        # Move batch to device
        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs["loss"]

        # Check logits for NaN - FAIL FAST instead of skipping
        if "logits" in outputs and outputs["logits"] is not None:
            logits = outputs["logits"]
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("\n" + "="*70)
                print("TRAINING ERROR: NaN/Inf detected in logits!")
                print("This indicates an upstream problem, likely:")
                print("  1. Bad learning rate (too high)")
                print("  2. Exploding activations in vision encoder or projector")
                print("  3. Corrupted input data")
                print("="*70 + "\n")
                raise ValueError("NaN/Inf in logits - see above for likely causes")

        # Check for None, NaN/Inf loss - FAIL FAST
        if loss is None:
            raise ValueError("Loss is None - model forward() returned None loss")

        if torch.isnan(loss) or torch.isinf(loss):
            print("\n" + "="*70)
            print("TRAINING ERROR: NaN/Inf detected in loss!")
            print("This indicates an upstream problem, likely:")
            print("  1. Bad learning rate (too high)")
            print("  2. Corrupted targets or labels")
            print("  3. Division by zero in loss calculation")
            print(f"Loss value: {loss}")
            print("="*70 + "\n")
            raise ValueError("NaN/Inf in loss - see above for likely causes")

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

            total_tokens = router_logits_last.shape[0]
            batch_size = expert_targets.shape[0]
            router_seq_len = total_tokens // batch_size

            router_logits_flat = router_logits_last.reshape(-1, router_logits_last.shape[-1])
            expanded_targets = expert_targets.unsqueeze(1).expand(-1, router_seq_len).reshape(-1)

            if router_logits_flat.shape[0] == expanded_targets.shape[0]:
                expert_supervision_loss = torch.nn.functional.cross_entropy(
                    router_logits_flat, expanded_targets, reduction="mean"
                )
                loss = loss + self.config.expert_supervision_weight * expert_supervision_loss

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass with BitNet gradient scaling
        if self.bitnet_scaler is not None:
            # Use BitNet gradient scaler
            scaled_loss = self.bitnet_scaler.scale_loss(loss)
            scaled_loss.backward()
            self.bitnet_scaler.unscale_gradients(self.optimizer)
        elif self.scaler is not None:
            # Use standard AMP scaler
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Check gradients for NaN/Inf - FAIL FAST
        is_finite, bad_param = check_gradients_finite(self.model)
        if not is_finite:
            print("\n" + "="*70)
            print(f"TRAINING ERROR: NaN/Inf gradient in parameter: {bad_param}")
            print("This indicates an upstream problem, likely:")
            print("  1. Learning rate too high")
            print("  2. Numerical instability in model (e.g., division by zero)")
            print("  3. Exploding activations")
            print("\nSuggested fixes:")
            print("  - Lower learning rate (try 1e-5 or 1e-6)")
            print("  - Check input data for NaN/Inf")
            print("  - Add gradient clipping (already enabled)")
            print("="*70 + "\n")
            raise ValueError(f"NaN/Inf gradient in {bad_param} - see above for fixes")

        if self.bitnet_scaler is not None:
            self.bitnet_scaler.update(found_inf=False)

        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
        }

    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.use_bitnet_optimizer:
            # BitNet optimizer handles its own gradient clipping internally
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            # Standard optimizer flow
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

            # Scheduler step (only for non-BitNet optimizer)
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
        """Linear warmup for learning rate (skipped for BitNet optimizer)."""
        if self.use_bitnet_optimizer:
            # BitNet optimizer handles its own warmup internally
            return
        if self.global_step < self.config.warmup_steps:
            warmup_factor = self.global_step / max(1, self.config.warmup_steps)
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
                max_samples_per_dataset=self.config.max_samples_per_dataset,
            )

        # Calculate training steps
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.config.epochs

        # Set phase1_steps for BitNet optimizer (60% of total training)
        if self.use_bitnet_optimizer and hasattr(self.optimizer, 'phase1_steps'):
            self.optimizer.phase1_steps = int(total_steps * self.config.bitnet_phase1_ratio)
            print(f"✓ BitNet optimizer phase1_steps set to {self.optimizer.phase1_steps}")

        # Create scheduler (only for non-BitNet optimizer)
        if not self.use_bitnet_optimizer:
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
        print(f"Using BitNet optimizer: {self.use_bitnet_optimizer}")
        if self.use_bitnet_optimizer:
            print(f"BitNet phase1 steps: {self.optimizer.phase1_steps}")
            print(f"BitNet phase2 LR factor: {self.config.bitnet_phase2_lr_factor}")
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
                        # Get LR from BitNet optimizer or standard optimizer
                        if self.use_bitnet_optimizer:
                            lr = self.optimizer._get_lr(self.optimizer.global_step)
                        else:
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

            # End of epoch - handle case where all batches were skipped
            if epoch_steps == 0:
                print(f"\n{'='*70}")
                print(f"ERROR: Epoch {epoch+1} - ALL batches produced NaN/Inf!")
                print(f"Model stability issue detected.")
                print(f"{'='*70}\n")
                self._save_checkpoint(f"emergency_epoch_{epoch+1}.pt")
                raise RuntimeError("Training failed: All batches produced NaN/Inf losses")

            epoch_avg_loss = epoch_loss / epoch_steps
            print(f"\nEpoch {epoch+1} Complete | Average Loss: {epoch_avg_loss:.4f}")
            print(f"  Valid batches: {epoch_steps} / {len(train_loader)}\n")

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


def run_training(args, stage: int, resume_from: Optional[str] = None):
    """Run training for a single stage. Returns the final checkpoint path."""

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # =========================================================================
    # TRIAL vs MAIN mode presets
    # =========================================================================
    if args.trial:
        epochs = args.epochs if args.epochs is not None else 1
        batch_size = args.batch_size if args.batch_size is not None else 1
        grad_accum = args.grad_accum if args.grad_accum is not None else 4
        max_samples = args.max_samples_per_dataset if args.max_samples_per_dataset is not None else 50
        use_wandb = False
        log_interval = 5
        eval_interval = 20
        save_interval = 50
        output_dir = args.output_dir if args.output_dir != "./checkpoints" else "./checkpoints/trial"
        # Disable AMP in trial mode for stability unless explicitly enabled
        if not hasattr(args, '_amp_explicitly_set'):
            args.no_amp = True
    elif args.main:
        epochs = args.epochs if args.epochs is not None else (3 if stage == 1 else 10)
        batch_size = args.batch_size if args.batch_size is not None else (8 if stage == 1 else 4)
        grad_accum = args.grad_accum if args.grad_accum is not None else 4
        max_samples = args.max_samples_per_dataset
        use_wandb = args.wandb and not args.no_wandb
        log_interval = 10
        eval_interval = 500
        save_interval = 1000
        output_dir = args.output_dir
    else:
        epochs = args.epochs if args.epochs is not None else 3
        batch_size = args.batch_size if args.batch_size is not None else 4
        grad_accum = args.grad_accum if args.grad_accum is not None else 4
        max_samples = args.max_samples_per_dataset
        use_wandb = args.wandb and not args.no_wandb
        log_interval = 10
        eval_interval = 500
        save_interval = 1000
        output_dir = args.output_dir

    # Create stage-specific output directory
    stage_output_dir = f"{output_dir}/stage{stage}"

    # Create config
    config = TrainingConfig(
        stage=stage,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        output_dir=stage_output_dir,
        data_dir=args.data_dir,
        resume_from=resume_from,
        device=device,
        mixed_precision=not args.no_amp,
        num_workers=args.num_workers,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=f"{args.wandb_run_name or 'embernet'}_stage{stage}" if args.wandb_run_name or use_wandb else None,
        use_ema=not args.no_ema,
        use_curriculum=not args.no_curriculum,
        use_expert_supervision=not args.no_expert_supervision,
        use_adaptive_grad_clip=not args.no_adaptive_clip,
        log_interval=log_interval,
        eval_interval=eval_interval,
        save_interval=save_interval,
    )
    config.max_samples_per_dataset = max_samples

    if args.lr is not None:
        config.learning_rate = args.lr

    # Print configuration
    mode_name = "TRIAL" if args.trial else ("MAIN" if args.main else "DEFAULT")
    print(f"\n{'='*70}")
    print(f"STAGE {stage} TRAINING ({mode_name} MODE)")
    print(f"{'='*70}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Max samples per dataset: {max_samples if max_samples else 'ALL'}")
    print(f"Output directory: {config.output_dir}")
    print(f"Resume from: {resume_from or 'None'}")
    print(f"W&B logging: {config.use_wandb}")
    print(f"{'='*70}\n")

    # Create trainer and run training
    trainer = Trainer(config)
    trainer.train()

    # Return path to final checkpoint
    return f"{stage_output_dir}/final_model.pt"


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train EmberNet VLM")

    # Training mode presets
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--trial", action="store_true",
                        help="Trial run: minimal data, 1 epoch, quick validation of pipeline")
    mode_group.add_argument("--main", action="store_true",
                        help="Main run: full datasets, full epochs, production training")

    # Stage selection
    parser.add_argument("--stage", type=int, default=None, choices=[1, 2],
                        help="Training stage (1=projector, 2=expert SFT). If not specified with --trial/--main, runs both stages.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (auto-set by --trial/--main)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Training batch size (auto-set by --trial/--main)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (auto-set based on stage)")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Gradient accumulation steps (auto-set by --trial/--main)")
    parser.add_argument("--max-samples-per-dataset", type=int, default=None,
                        help="Max samples per dataset (auto-set by --trial/--main)")

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

    # =========================================================================
    # Determine which stages to run
    # =========================================================================
    if args.stage is not None:
        # Specific stage requested
        stages_to_run = [args.stage]
        print(f"\n{'='*70}")
        print(f"Running Stage {args.stage} only")
        print(f"{'='*70}")
    elif args.trial or args.main:
        # --trial or --main without --stage runs both stages
        stages_to_run = [1, 2]
        mode = "TRIAL" if args.trial else "MAIN"
        print(f"\n{'='*70}")
        print(f"{mode} MODE: Running BOTH Stage 1 and Stage 2 sequentially")
        print(f"{'='*70}")
    else:
        # Default: just stage 1
        stages_to_run = [1]
        print(f"\n{'='*70}")
        print(f"Running Stage 1 (use --trial or --main for full pipeline)")
        print(f"{'='*70}")

    # =========================================================================
    # Run training stages
    # =========================================================================
    checkpoint_path = args.resume  # Initial checkpoint (if any)

    for stage in stages_to_run:
        print(f"\n{'#'*70}")
        print(f"# STARTING STAGE {stage}")
        print(f"{'#'*70}\n")

        checkpoint_path = run_training(args, stage=stage, resume_from=checkpoint_path)

        print(f"\n{'#'*70}")
        print(f"# STAGE {stage} COMPLETE - Checkpoint: {checkpoint_path}")
        print(f"{'#'*70}\n")

    # Final summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Stages completed: {stages_to_run}")
    print(f"Final checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

