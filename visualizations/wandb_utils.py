"""
W&B logging utilities for EmberNet visualizations.

Wraps wandb calls with graceful fallback when W&B is unavailable,
handles image uploads and metric grouping.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    warnings.warn("wandb not installed – W&B logging will be skipped.", stacklevel=2)


class WandBLogger:
    """
    Thin wrapper around wandb that handles missing keys gracefully,
    uploads plot images, and organises metrics by group.

    Usage::

        logger = WandBLogger(run=wandb.init(...))
        logger.log({"train/loss": 0.5, "train/step": 100})
        logger.log_image("plots/training_dynamics/...", "plots/loss_curve")
    """

    def __init__(self, run=None, project: str = "EmberNet", disabled: bool = False):
        self.disabled = disabled or (not HAS_WANDB)
        self._run = run

        if not self.disabled and self._run is None:
            try:
                self._run = wandb.run  # use active run if any
            except Exception:
                self._run = None

    # ------------------------------------------------------------------
    # Core log
    # ------------------------------------------------------------------
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log scalar metrics to W&B."""
        if self.disabled or self._run is None:
            return
        try:
            kwargs = {"step": step} if step is not None else {}
            wandb.log(metrics, **kwargs)
        except Exception as e:
            warnings.warn(f"W&B log failed: {e}")

    # ------------------------------------------------------------------
    # Image upload
    # ------------------------------------------------------------------
    def log_image(
        self,
        local_path: str | Path,
        wb_key: str,
        caption: str = "",
        step: Optional[int] = None,
    ):
        """Upload a saved plot image to W&B under *wb_key*."""
        if self.disabled or self._run is None:
            return
        try:
            img = wandb.Image(str(local_path), caption=caption)
            kwargs = {"step": step} if step is not None else {}
            wandb.log({wb_key: img}, **kwargs)
        except Exception as e:
            warnings.warn(f"W&B image log failed for {wb_key}: {e}")

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------
    def log_histogram(
        self,
        values,
        wb_key: str,
        step: Optional[int] = None,
        num_bins: int = 64,
    ):
        """Log a histogram to W&B."""
        if self.disabled or self._run is None:
            return
        try:
            hist = wandb.Histogram(values, num_bins=num_bins)
            kwargs = {"step": step} if step is not None else {}
            wandb.log({wb_key: hist}, **kwargs)
        except Exception as e:
            warnings.warn(f"W&B histogram log failed for {wb_key}: {e}")

    # ------------------------------------------------------------------
    # Bulk plot upload (after epoch)
    # ------------------------------------------------------------------
    def upload_plots(self, paths: List[Path], prefix: str = "plots", step: Optional[int] = None):
        """Upload a list of local plot files to W&B."""
        if self.disabled or self._run is None:
            return
        for p in paths:
            if p.exists():
                # Derive wb_key from file path relative to "plots/"
                try:
                    rel = p.relative_to("plots")
                    key = f"{prefix}/{rel.with_suffix('').as_posix()}"
                except ValueError:
                    key = f"{prefix}/{p.stem}"
                self.log_image(p, key, step=step)

    # ------------------------------------------------------------------
    # Expert metrics helpers
    # ------------------------------------------------------------------
    def log_expert_metrics(
        self,
        expert_name: str,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log per-expert metrics with correct key hierarchy."""
        prefixed = {f"expert/{expert_name}/{k}": v for k, v in metrics.items()}
        self.log(prefixed, step=step)

    def log_dataset_metrics(
        self,
        dataset_name: str,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log per-dataset metrics."""
        prefixed = {f"dataset/{dataset_name}/{k}": v for k, v in metrics.items()}
        self.log(prefixed, step=step)
