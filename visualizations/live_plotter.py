"""
live_plotter.py
===============
Streams real training data into the visualizations package every epoch.
Plots are saved to {output_dir}/plots/ — directly next to checkpoints.

Wired into training/train.py:
    init:      Trainer.__init__   → self.live_plotter = LivePlotter(...)
    per step:  inner train loop   → self.live_plotter.record_step(...)
    per epoch: end of epoch loop  → self.live_plotter.plot_epoch(...)
    at end:    after all epochs   → self.live_plotter.plot_final(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class LivePlotter:
    """
    Accumulates per-step data and generates plots into {output_dir}/plots/
    at the end of every epoch and at end of training.

    All plot calls are individually guarded — a broken plot never crashes
    the training loop.
    """

    def __init__(
        self,
        output_dir: str,
        stage: int,
        logger=None,            # WandBLogger instance, optional
        disabled: bool = False, # force-disable (e.g. headless env without matplotlib)
    ):
        self.output_dir = Path(output_dir)
        self.stage = stage
        self.disabled = disabled

        # Redirect every plotter's file output to output_dir/plots/
        self.plots_dir = self.output_dir / "plots"

        if not disabled:
            try:
                import matplotlib
                matplotlib.use("Agg")   # non-interactive backend (safe on servers)

                # must import AFTER setting backend
                from visualizations.config import set_plots_root, ensure_plot_dirs
                set_plots_root(self.plots_dir)
                ensure_plot_dirs()

                from visualizations import TrainingDynamicsPlotter, ExpertAnalysisPlotter
                from visualizations.wandb_utils import WandBLogger as _WBL
                _log = logger or _WBL(disabled=True)
                self._dyn = TrainingDynamicsPlotter(_log, save_dir=self.plots_dir)
                self._exp = ExpertAnalysisPlotter(_log)
                print(f"  [viz] Plots directory: {self.plots_dir.resolve()}")
            except ImportError as e:
                print(f"  [viz] Disabled — missing dependency: {e}")
                self.disabled = True
            except Exception as e:
                print(f"  [viz] Disabled — init error: {e}")
                self.disabled = True

        # ---- per-step accumulators ----
        self._steps:        List[int]         = []
        self._losses:       List[float]        = []
        self._lrs:          List[float]        = []
        self._grad_norms:   List[float]        = []
        self._clipped:      List[float]        = []   # 1.0 if step was clipped
        self._expert_probs: List[np.ndarray]   = []   # shape (8,) per step when available

        # index into per-step lists at which each epoch ended
        self._epoch_ends:   List[int]          = []

    # ------------------------------------------------------------------
    # Called every optimizer step from the training loop
    # ------------------------------------------------------------------

    def record_step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float = 0.0,
        clipped: bool = False,
        expert_probs=None,      # iterable of 8 floats (Stage 2)
    ):
        if self.disabled:
            return
        self._steps.append(step)
        self._losses.append(float(loss))
        self._lrs.append(float(lr))
        self._grad_norms.append(float(grad_norm))
        self._clipped.append(1.0 if clipped else 0.0)
        if expert_probs is not None:
            arr = np.asarray(expert_probs, dtype=float)
            if arr.shape == (8,):
                self._expert_probs.append(arr)

    # ------------------------------------------------------------------
    # Called at the end of each epoch
    # ------------------------------------------------------------------

    def plot_epoch(self, epoch: int, stage: int):
        """Regenerate key plots with all data collected so far."""
        if self.disabled or not self._steps:
            return

        self._epoch_ends.append(len(self._steps))   # mark epoch boundary

        cur_step = self._steps[-1]
        steps    = np.array(self._steps)
        losses   = np.array(self._losses)
        lrs      = np.array(self._lrs)
        gnorms   = np.array(self._grad_norms)

        # 1 ── loss curve
        other = 2 if stage == 1 else 1
        loss_data: Dict = {
            f"stage{stage}_steps":       steps,
            f"stage{stage}_train_loss":  losses,
            f"stage{other}_steps":       np.array([cur_step]),
            f"stage{other}_train_loss":  np.array([losses[-1]]),
        }
        self._safe(self._dyn.plot_multistage_loss, loss_data, cur_step, tag="loss curve")

        # 2 ── gradient norms
        if gnorms.any():
            self._safe(
                self._dyn.plot_gradient_norms,
                {"steps": steps, "global_norm": gnorms, "clip_threshold": 1.0},
                cur_step, tag="grad norms",
            )

        # 3 ── LR schedule
        if lrs.any():
            warmup     = 100
            phase1_end = max(warmup + 1, int(len(steps) * 0.6))
            self._safe(
                self._dyn.plot_bitnet_lr_schedule,
                {
                    "steps":      steps,
                    "actual_lr":  lrs,
                    "warmup_end": warmup,
                    "phase1_end": phase1_end,
                    "max_lr":     float(lrs.max()),
                    "phase2_lr":  float(lrs[-1]) if len(lrs) > 0 else float(lrs.max()) * 0.1,
                },
                cur_step, tag="lr schedule",
            )

        # 4 ── expert selection heatmap (available once Stage 2 data arrives)
        if self._expert_probs:
            freq_matrix, ckpt_labels = self._build_freq_matrix()
            self._safe(
                self._exp.plot_expert_selection_heatmap,
                {"freq": freq_matrix, "ckpt_labels": ckpt_labels},
                cur_step, tag="expert heatmap",
            )

            # 5 ── load balancing over training steps
            imb_steps = np.array(self._steps[-len(self._expert_probs):])
            imb_vals  = np.array([
                float(np.std(p) / max(float(np.mean(p)), 1e-8))
                for p in self._expert_probs
            ])
            self._safe(
                self._exp.plot_load_balancing,
                {"steps": imb_steps, "imbalance": imb_vals, "target": 0.1},
                cur_step, tag="load balancing",
            )

        self._print_summary(label=f"Epoch {epoch}")

    # ------------------------------------------------------------------
    # Called once at the very end of training
    # ------------------------------------------------------------------

    def plot_final(self, stage: int):
        """Generate heavier summary plots after all epochs complete."""
        if self.disabled or not self._steps:
            return

        cur_step = self._steps[-1]
        steps    = np.array(self._steps)
        losses   = np.array(self._losses)

        # Convergence rate
        self._safe(
            self._dyn.plot_convergence_rate,
            {"steps": steps, "losses": losses},
            cur_step, tag="convergence",
        )

        # Per-epoch clipping frequency
        if self._epoch_ends:
            epochs_arr     = np.arange(1, len(self._epoch_ends) + 1)
            clip_per_epoch = []
            prev = 0
            for end in self._epoch_ends:
                window = self._clipped[prev:end]
                clip_per_epoch.append(float(np.mean(window)) * 100.0 if window else 0.0)
                prev = end
            self._safe(
                self._dyn.plot_gradient_clipping_frequency,
                {
                    "epochs":   epochs_arr,
                    "adaptive": np.array(clip_per_epoch),
                    "fixed":    np.array(clip_per_epoch),
                },
                cur_step, tag="clip frequency",
            )

        # Expert summary plots (Stage 2)
        if self._expert_probs:
            self._safe(self._exp.plot_cooccurrence_matrix, step=cur_step, tag="co-occurrence")
            self._safe(self._exp.plot_specialization_index, step=cur_step, tag="specialization")
            self._safe(self._exp.plot_usage_violin, step=cur_step, tag="usage violin")

        self._print_summary(label="Final", final=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_freq_matrix(self):
        """Return (8 × n_epochs) frequency matrix + epoch labels."""
        n_epochs  = max(1, len(self._epoch_ends))
        n_experts = 8
        matrix    = np.zeros((n_experts, n_epochs))
        labels    = []
        prev = 0
        for i, end in enumerate(self._epoch_ends):
            slice_ = self._expert_probs[prev:end]
            if slice_:
                matrix[:, i] = np.mean(slice_, axis=0)
            labels.append(f"Ep{i + 1}")
            prev = end
        # fill final in-progress epoch if epoch_ends hasn't been appended yet
        if len(self._expert_probs) > prev:
            matrix[:, -1] = np.mean(self._expert_probs[prev:], axis=0)
        return matrix, labels

    def _safe(self, fn, data=None, step=None, *, tag="plot"):
        """Call a plotter method, swallowing any exception."""
        try:
            if data is not None:
                fn(data, step=step)
            else:
                fn(step=step)
        except Exception as e:
            print(f"  [viz] {tag} error (non-fatal): {e}")

    def _print_summary(self, label: str = "", final: bool = False):
        try:
            n = sum(1 for f in self.plots_dir.rglob("*.png") if f.is_file())
        except Exception:
            n = 0
        tag = "saved" if final else "updated"
        print(f"  [viz] {label} plots {tag} — {n} PNG files → {self.plots_dir.resolve()}")
