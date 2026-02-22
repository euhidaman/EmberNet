"""
Training Dynamics Visualizations

Covers:
  1.1  Loss Curves (multi-stage, components, per-dataset heatmap)
  1.2  Learning Rate Schedules (two-phase BitNet, per-group)
  1.3  Gradient Statistics (norms, flow heatmap, clipping frequency)
  1.4  Convergence Analysis (rate, efficiency)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns

from visualizations.config import (
    VIZ_CONFIG, PLOT_DIRS, STAGE_COLORS, EXPERT_NAMES,
    ALL_DATASETS, DATASET_DOMAINS, DOMAIN_COLORS,
    apply_mpl_style, plot_filename, log_plot_error,
)
from visualizations.wandb_utils import WandBLogger

apply_mpl_style()


class TrainingDynamicsPlotter:
    """
    Generates all §1 training-dynamics plots.

    All methods accept optional *data* dicts of NumPy arrays / lists.
    When data is None or a key is missing the method falls back to
    plausible synthetic / placeholder data so the plot is always rendered
    (annotated as "Incomplete" in that case).
    """

    def __init__(self, logger: Optional[WandBLogger] = None, save_dir: Optional[Path] = None):
        self.logger = logger or WandBLogger(disabled=True)
        self._generated: List[Path] = []

    # ==================================================================
    # 1.1 Loss Curves
    # ==================================================================

    def plot_multistage_loss(
        self,
        data: Optional[Dict] = None,
        step: Optional[int] = None,
    ) -> Path:
        """Plot 1.1.1 – Multi-Stage Loss Progression."""
        key = "multi_stage_loss_progression"
        out = PLOT_DIRS["loss_curves"] / plot_filename("training_dynamics", "loss_curves", key)
        try:
            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])

            incomplete = data is None
            if data is None:
                data = _synthetic_loss_data()

            s1_steps = data.get("stage1_steps", np.arange(0, 2000))
            s2_steps = data.get("stage2_steps", np.arange(2000, 5000))
            s1_train  = data.get("stage1_train_loss",   _mock_loss(len(s1_steps), start=4.0))
            s1_val    = data.get("stage1_val_loss",     _mock_loss(len(s1_steps), start=3.8))
            s2_train  = data.get("stage2_train_loss",   _mock_loss(len(s2_steps), start=2.0))
            s2_val    = data.get("stage2_val_loss",     _mock_loss(len(s2_steps), start=1.9))

            c1, c2 = STAGE_COLORS[1], STAGE_COLORS[2]
            lw = VIZ_CONFIG["lw_main"]

            # EMA helper
            def ema(x, alpha=0.1):
                s = np.array(x, dtype=float)
                for i in range(1, len(s)):
                    s[i] = alpha * x[i] + (1 - alpha) * s[i - 1]
                return s

            ax.semilogy(s1_steps, s1_train, color=c1, lw=lw, label="Stage 1 train")
            ax.semilogy(s1_steps, s1_val,   color=c1, lw=VIZ_CONFIG["lw_dashed"], ls="--", label="Stage 1 val")
            ax.semilogy(s1_steps, ema(s1_train), color=c1, alpha=0.4, lw=1.0)
            ax.semilogy(s1_steps, ema(s1_val),   color=c1, alpha=0.4, lw=1.0)

            ax.semilogy(s2_steps, s2_train, color=c2, lw=lw, label="Stage 2 train")
            ax.semilogy(s2_steps, s2_val,   color=c2, lw=VIZ_CONFIG["lw_dashed"], ls="--", label="Stage 2 val")
            ax.semilogy(s2_steps, ema(s2_train), color=c2, alpha=0.4, lw=1.0)
            ax.semilogy(s2_steps, ema(s2_val),   color=c2, alpha=0.4, lw=1.0)

            # Stage separator
            sep = s1_steps[-1] if len(s1_steps) else 2000
            ax.axvline(sep, color="black", lw=1.2, ls=":", label="Stage boundary")
            ax.text(sep + 20, ax.get_ylim()[1] * 0.9, "Stage 2 →", fontsize=9, va="top")

            # Annotations: best val
            best_s1 = float(np.min(s1_val))
            best_s2 = float(np.min(s2_val))
            ax.annotate(
                f"best {best_s1:.3f}",
                xy=(s1_steps[np.argmin(s1_val)], best_s1),
                xytext=(10, 15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8,
            )
            ax.annotate(
                f"best {best_s2:.3f}",
                xy=(s2_steps[np.argmin(s2_val)], best_s2),
                xytext=(10, 15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8,
            )

            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Loss (log scale)")
            title = "EmberNet – Multi-Stage Loss Progression"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend(loc="upper right")
            _save_and_log(fig, out, self.logger, "plots/training_dynamics/loss_curves/multi_stage_loss", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e)
            plt.close("all")
            return out

    def plot_loss_components(
        self,
        data: Optional[Dict] = None,
        step: Optional[int] = None,
    ) -> Path:
        """Plot 1.1.2 – Loss Components Breakdown (Stage 2)."""
        key = "loss_components_breakdown"
        out = PLOT_DIRS["loss_curves"] / plot_filename("training_dynamics", "loss_curves", key)
        try:
            incomplete = data is None
            if data is None:
                n = 3000
                steps = np.arange(n)
                data = {
                    "steps":       steps,
                    "lm_loss":     _mock_loss(n, start=2.0, noise=0.05),
                    "aux_loss":    np.abs(np.random.normal(0.05, 0.01, n)),
                    "router_z":    np.abs(np.random.normal(0.02, 0.005, n)),
                    "entropy":     np.abs(np.random.normal(0.01, 0.003, n)),
                    "expert_sup":  np.abs(np.random.normal(0.03, 0.008, n)),
                }

            steps    = np.asarray(data["steps"])
            lm       = np.asarray(data["lm_loss"])
            aux      = np.asarray(data["aux_loss"])
            router_z = np.asarray(data["router_z"])
            entropy  = np.asarray(data["entropy"])
            exp_sup  = np.asarray(data["expert_sup"])

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # Stacked area chart (absolute)
            colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c"]
            labels = ["LM loss", "MoE aux loss", "Router z-loss", "Entropy reg", "Expert supervision"]
            data_stacked = [lm, aux, router_z, entropy, exp_sup]
            ax1.stackplot(steps, data_stacked, labels=labels, colors=colors, alpha=0.8)
            ax1.set_ylabel("Loss (absolute)")
            ax1.legend(loc="upper right", fontsize=8)
            ax1.set_title("Stage 2 Loss Components – Absolute", fontweight="bold")

            # Percentage breakdown
            total = lm + aux + router_z + entropy + exp_sup
            pct = [d / (total + 1e-12) * 100 for d in data_stacked]
            ax2.stackplot(steps, pct, labels=labels, colors=colors, alpha=0.8)
            ax2.set_ylabel("Loss Contribution (%)")
            ax2.set_xlabel("Training Steps")
            ax2.set_ylim(0, 100)
            title2 = "Stage 2 Loss Components – Percentage"
            if incomplete:
                title2 += "  [Incomplete – placeholder data]"
            ax2.set_title(title2, fontweight="bold")

            _save_and_log(fig, out, self.logger, "plots/training_dynamics/loss_curves/components", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e)
            plt.close("all")
            return out

    def plot_per_dataset_loss_heatmap(
        self,
        data: Optional[Dict] = None,
        step: Optional[int] = None,
    ) -> Path:
        """Plot 1.1.3 – Per-Dataset Loss Heatmap."""
        key = "per_dataset_loss_heatmap"
        out = PLOT_DIRS["loss_curves"] / plot_filename("training_dynamics", "loss_curves", key)
        try:
            incomplete = data is None
            datasets = ALL_DATASETS
            n_ckpts = 10
            if data is None:
                np.random.seed(42)
                matrix = np.random.uniform(0.5, 4.0, (len(datasets), n_ckpts))
                # Simulate decreasing loss over checkpoints
                for i in range(len(datasets)):
                    matrix[i] = np.linspace(matrix[i, 0], matrix[i, 0] * 0.3, n_ckpts) + \
                                 np.random.normal(0, 0.05, n_ckpts)
                ckpt_labels = [f"ckpt\n{(i+1)*500}" for i in range(n_ckpts)]
            else:
                matrix = np.asarray(data["matrix"])
                ckpt_labels = data.get("ckpt_labels", [str(i) for i in range(matrix.shape[1])])

            fig, ax = plt.subplots(figsize=(14, 8))
            im = sns.heatmap(
                matrix, ax=ax,
                xticklabels=ckpt_labels,
                yticklabels=datasets,
                cmap="RdYlGn_r",
                annot=True, fmt=".2f",
                linewidths=0.3,
                cbar_kws={"label": "Loss"},
            )
            ax.set_xlabel("Training Checkpoint")
            ax.set_ylabel("Dataset")
            title = "Per-Dataset Loss Heatmap"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")

            # Domain separators
            cum = 0
            for domain, ds_list in DATASET_DOMAINS.items():
                cum += len(ds_list)
                if cum < len(datasets):
                    ax.axhline(cum, color="white", lw=2)

            _save_and_log(fig, out, self.logger, "plots/training_dynamics/loss_curves/per_dataset_heatmap", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e)
            plt.close("all")
            return out

    # ==================================================================
    # 1.2 Learning Rate Schedules
    # ==================================================================

    def plot_bitnet_lr_schedule(
        self,
        data: Optional[Dict] = None,
        step: Optional[int] = None,
    ) -> Path:
        """Plot 1.2.1 – BitNet Two-Phase LR Schedule."""
        key = "bitnet_two_phase_lr_schedule"
        out = PLOT_DIRS["learning_rates"] / plot_filename("training_dynamics", "learning_rates", key)
        try:
            incomplete = data is None
            if data is None:
                total = 5000
                warmup = 500
                phase1_end = int(total * 0.6)
                max_lr = 3e-4
                phase2_lr = max_lr * 0.1
                steps = np.arange(total)

                def _lr(s):
                    if s < warmup:
                        return max_lr * s / warmup
                    elif s < phase1_end:
                        return max_lr
                    else:
                        return phase2_lr

                actual_lr = np.array([_lr(s) for s in steps])
                data = {
                    "steps": steps,
                    "actual_lr": actual_lr,
                    "warmup_end": warmup,
                    "phase1_end": phase1_end,
                    "max_lr": max_lr,
                    "phase2_lr": phase2_lr,
                }

            steps     = np.asarray(data["steps"])
            actual_lr = np.asarray(data["actual_lr"])
            warmup_end  = data.get("warmup_end", 500)
            phase1_end  = data.get("phase1_end", 3000)
            max_lr      = data.get("max_lr", 3e-4)
            phase2_lr   = data.get("phase2_lr", 3e-5)

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            ax.semilogy(steps, actual_lr, color="#1f77b4", lw=VIZ_CONFIG["lw_main"], label="Actual LR")
            ax.axhline(max_lr,    color="#2ca02c", lw=VIZ_CONFIG["lw_dashed"], ls="--", label="Phase 1 target LR")
            ax.axhline(phase2_lr, color="#ff7f0e", lw=VIZ_CONFIG["lw_dashed"], ls="--", label="Phase 2 target LR")

            # Shaded regions
            ymin, ymax = actual_lr.min() * 0.5, actual_lr.max() * 2
            ax.set_ylim(ymin, ymax)
            ax.axvspan(0, warmup_end, alpha=0.07, color="green",  label="Warmup")
            ax.axvspan(warmup_end, phase1_end, alpha=0.07, color="blue",  label="Phase 1")
            ax.axvspan(phase1_end, steps[-1],  alpha=0.07, color="orange", label="Phase 2")

            ax.axvline(warmup_end,  color="black", lw=1.0, ls=":")
            ax.axvline(phase1_end,  color="black", lw=1.0, ls=":")

            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Learning Rate (log scale)")
            title = "BitNet b1.58 – Two-Phase LR Schedule"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend(ncol=2, fontsize=9)

            _save_and_log(fig, out, self.logger, "plots/training_dynamics/learning_rates/bitnet_two_phase_lr", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e)
            plt.close("all")
            return out

    def plot_per_group_lr(
        self,
        data: Optional[Dict] = None,
        step: Optional[int] = None,
    ) -> Path:
        """Plot 1.2.2 – Per-Parameter-Group LR."""
        key = "per_param_group_lr"
        out = PLOT_DIRS["learning_rates"] / plot_filename("training_dynamics", "learning_rates", key)
        try:
            incomplete = data is None
            groups = ["projector", "router", "experts", "shared_expert", "norm_layers"]
            group_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
            n = 5000
            if data is None:
                np.random.seed(0)
                steps = np.arange(n)
                scales = [1.0, 0.8, 0.9, 0.7, 0.5]
                lrs = {}
                for g, sc in zip(groups, scales):
                    base = sc * 3e-4
                    warmup = np.linspace(0, base, 500)
                    flat   = np.full(2500, base)
                    decay  = np.linspace(base, base * 0.1, 2000)
                    lrs[g] = np.concatenate([warmup, flat, decay])
                data = {"steps": steps, "lrs": lrs}

            steps = np.asarray(data["steps"])
            lrs   = data["lrs"]

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            for g, c in zip(groups, group_colors):
                if g in lrs:
                    ax.semilogy(steps, lrs[g], color=c, lw=VIZ_CONFIG["lw_main"], label=g)

            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Learning Rate (log scale)")
            title = "Per-Parameter-Group Learning Rate"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            _save_and_log(fig, out, self.logger, "plots/training_dynamics/learning_rates/per_group_lr", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e)
            plt.close("all")
            return out

    # ==================================================================
    # 1.3 Gradient Statistics
    # ==================================================================

    def plot_gradient_norms(
        self,
        data: Optional[Dict] = None,
        step: Optional[int] = None,
    ) -> Path:
        """Plot 1.3.1 – Gradient Norms Over Time."""
        key = "gradient_norms_over_time"
        out = PLOT_DIRS["gradient_stats"] / plot_filename("training_dynamics", "gradient_stats", key)
        try:
            incomplete = data is None
            if data is None:
                n = 5000
                np.random.seed(1)
                steps = np.arange(n)
                global_norm = np.abs(np.random.normal(0.5, 0.3, n)).clip(0.01, 5)
                # Spike simulation
                spike_positions = np.random.choice(n, size=20, replace=False)
                global_norm[spike_positions] *= 4
                data = {
                    "steps": steps,
                    "global_norm": global_norm,
                    "clip_threshold": 1.0,
                }

            steps = np.asarray(data["steps"])
            g_norm = np.asarray(data["global_norm"])
            clip_th = data.get("clip_threshold", 1.0)
            layer_norms = data.get("layer_norms", {})

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            ax.semilogy(steps, g_norm, color="black", lw=VIZ_CONFIG["lw_main"], label="Global grad norm", alpha=0.9)

            cmap = plt.cm.tab10
            for idx, (layer, vals) in enumerate(layer_norms.items()):
                ax.semilogy(steps, vals, color=cmap(idx % 10), lw=0.8, alpha=0.6, label=layer)

            ax.axhline(clip_th, color="red", lw=VIZ_CONFIG["lw_dashed"], ls="--", label=f"Clip threshold ({clip_th})")

            # Highlight spikes (norm > 3× median)
            med_norm = float(np.median(g_norm))
            spikes = steps[g_norm > 3 * med_norm]
            if len(spikes):
                ax.scatter(spikes, g_norm[g_norm > 3 * med_norm], color="red", s=10, zorder=5, label="Spikes")

            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Gradient Norm (log scale)")
            title = "Gradient Norms Over Time"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=8, ncol=2)

            _save_and_log(fig, out, self.logger, "plots/training_dynamics/gradient_stats/grad_norms", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e)
            plt.close("all")
            return out

    def plot_gradient_flow_heatmap(
        self,
        data: Optional[Dict] = None,
        step: Optional[int] = None,
    ) -> Path:
        """Plot 1.3.2 – Gradient Flow Heatmap."""
        key = "gradient_flow_heatmap"
        out = PLOT_DIRS["gradient_stats"] / plot_filename("training_dynamics", "gradient_stats", key)
        try:
            incomplete = data is None
            layer_names = (
                ["embed_tokens"]
                + [f"layer_{i}" for i in range(16)]
                + ["lm_head"]
            )
            n_ckpts = 10
            if data is None:
                np.random.seed(2)
                matrix = np.random.uniform(1e-4, 1e-2, (len(layer_names), n_ckpts))
                # Simulate vanishing gradient at lower layers
                for i, _ in enumerate(layer_names[:3]):
                    matrix[i] *= 0.01
                ckpt_labels = [f"{(i+1)*500}" for i in range(n_ckpts)]
            else:
                matrix = np.asarray(data["matrix"])
                ckpt_labels = data.get("ckpt_labels", [str(i) for i in range(matrix.shape[1])])

            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(
                np.log10(matrix + 1e-10), ax=ax,
                xticklabels=ckpt_labels,
                yticklabels=layer_names,
                cmap="viridis",
                cbar_kws={"label": "log₁₀(avg |grad|)"},
            )
            ax.set_xlabel("Training Step (Checkpoint)")
            ax.set_ylabel("Model Layer")
            title = "Gradient Flow Heatmap (log₁₀ scale)"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")

            _save_and_log(fig, out, self.logger, "plots/training_dynamics/gradient_stats/grad_flow_heatmap", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e)
            plt.close("all")
            return out

    def plot_gradient_clipping_frequency(
        self,
        data: Optional[Dict] = None,
        step: Optional[int] = None,
    ) -> Path:
        """Plot 1.3.3 – Gradient Clipping Frequency per Epoch."""
        key = "gradient_clipping_frequency"
        out = PLOT_DIRS["gradient_stats"] / plot_filename("training_dynamics", "gradient_stats", key)
        try:
            incomplete = data is None
            n_epochs = 10
            if data is None:
                np.random.seed(3)
                epochs = np.arange(1, n_epochs + 1)
                adaptive_freq = np.random.uniform(5, 30, n_epochs)
                fixed_freq    = np.random.uniform(10, 50, n_epochs)
                data = {"epochs": epochs, "adaptive": adaptive_freq, "fixed": fixed_freq}

            epochs   = np.asarray(data["epochs"])
            adaptive = np.asarray(data.get("adaptive", np.zeros(len(epochs))))
            fixed    = np.asarray(data.get("fixed",    np.zeros(len(epochs))))
            x = np.arange(len(epochs))
            w = 0.35

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            ax.bar(x - w / 2, adaptive, w, label="Adaptive clipping", color=STAGE_COLORS[2], alpha=0.85)
            ax.bar(x + w / 2, fixed,    w, label="Fixed clipping",     color=STAGE_COLORS[1], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([f"Ep {int(e)}" for e in epochs])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Clipping Frequency (%)")
            title = "Gradient Clipping Frequency per Epoch"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            _save_and_log(fig, out, self.logger, "plots/training_dynamics/gradient_stats/clip_frequency", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e)
            plt.close("all")
            return out

    # ==================================================================
    # 1.4 Convergence Analysis
    # ==================================================================

    def plot_convergence_rate(
        self,
        data: Optional[Dict] = None,
        step: Optional[int] = None,
    ) -> Path:
        """Plot 1.4.1 – Loss Convergence Rate with power-law fit."""
        key = "loss_convergence_rate"
        out = PLOT_DIRS["convergence"] / plot_filename("training_dynamics", "convergence", key)
        try:
            incomplete = data is None
            if data is None:
                n = 3000
                s1_steps = np.arange(1, 2000)
                s2_steps = np.arange(1, 1001)
                s1_val = 3.0 * (s1_steps ** -0.35) + np.random.normal(0, 0.02, len(s1_steps))
                s2_val = 1.8 * (s2_steps ** -0.45) + np.random.normal(0, 0.02, len(s2_steps))
                data = {"s1_steps": s1_steps, "s1_val": s1_val, "s2_steps": s2_steps, "s2_val": s2_val}

            # Use .get() per key so a partial data dict (e.g. only Stage 1
            # data available mid-training) falls back to synthetic placeholders
            # rather than crashing with KeyError.
            _syn = {}
            if any(k not in data for k in ("s1_steps", "s1_val", "s2_steps", "s2_val")):
                incomplete = True   # mark as incomplete when any key is missing
                _rng = np.random.default_rng(seed=0)
                _s1 = np.arange(1, 500)
                _s2 = np.arange(1, 500)
                _syn = {
                    "s1_steps": _s1,
                    "s1_val":   3.0 * (_s1 ** -0.35) + _rng.normal(0, 0.02, len(_s1)),
                    "s2_steps": _s2,
                    "s2_val":   1.8 * (_s2 ** -0.45) + _rng.normal(0, 0.02, len(_s2)),
                }

            s1_steps = np.asarray(data.get("s1_steps", _syn.get("s1_steps", np.arange(1, 500))))
            s1_val   = np.asarray(data.get("s1_val",   _syn.get("s1_val",   3.0 * (s1_steps ** -0.35))))
            s2_steps = np.asarray(data.get("s2_steps", _syn.get("s2_steps", np.arange(1, 500))))
            s2_val   = np.asarray(data.get("s2_val",   _syn.get("s2_val",   1.8 * (s2_steps ** -0.45))))

            def powerlaw_fit(steps, vals):
                log_s = np.log(steps + 1)
                log_v = np.log(np.clip(vals, 1e-6, None))
                coeffs = np.polyfit(log_s, log_v, 1)
                alpha = -coeffs[0]
                A = np.exp(coeffs[1])
                fitted = A * (steps ** -alpha)
                return alpha, A, fitted

            alpha1, A1, fit1 = powerlaw_fit(s1_steps, s1_val)
            alpha2, A2, fit2 = powerlaw_fit(s2_steps, s2_val)

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            ax.loglog(s1_steps, s1_val, color=STAGE_COLORS[1], alpha=0.5, lw=1.0, label="Stage 1 val loss")
            ax.loglog(s1_steps, fit1,   color=STAGE_COLORS[1], lw=VIZ_CONFIG["lw_main"],
                      label=rf"Stage 1 fit: A×steps^(-{alpha1:.3f})")
            ax.loglog(s2_steps, s2_val, color=STAGE_COLORS[2], alpha=0.5, lw=1.0, label="Stage 2 val loss")
            ax.loglog(s2_steps, fit2,   color=STAGE_COLORS[2], lw=VIZ_CONFIG["lw_main"],
                      label=rf"Stage 2 fit: A×steps^(-{alpha2:.3f})")

            ax.set_xlabel("Training Steps (log scale)")
            ax.set_ylabel("Validation Loss (log scale)")
            title = "Loss Convergence Rate (Power-Law Fit)"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            _save_and_log(fig, out, self.logger, "plots/training_dynamics/convergence/convergence_rate", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e)
            plt.close("all")
            return out

    def plot_training_efficiency(
        self,
        data: Optional[Dict] = None,
        step: Optional[int] = None,
    ) -> Path:
        """Plot 1.4.2 – Training Efficiency (loss vs tokens vs wall-clock)."""
        key = "training_efficiency_metrics"
        out = PLOT_DIRS["convergence"] / plot_filename("training_dynamics", "convergence", key)
        try:
            incomplete = data is None
            if data is None:
                n = 5000
                time_h = np.linspace(0, 12, n)
                steps  = np.arange(n)
                train_loss = _mock_loss(n, start=4.0)
                val_loss   = _mock_loss(n, start=3.8)
                cum_tokens = np.cumsum(np.full(n, 4096))         # tokens per step
                data = {"time_h": time_h, "steps": steps,
                        "train_loss": train_loss, "val_loss": val_loss,
                        "cum_tokens": cum_tokens}

            time_h     = np.asarray(data["time_h"])
            train_loss = np.asarray(data["train_loss"])
            val_loss   = np.asarray(data["val_loss"])
            cum_tokens = np.asarray(data["cum_tokens"])

            fig, ax1 = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            ax2 = ax1.twinx()

            ax1.semilogy(time_h, train_loss, color=STAGE_COLORS[1], lw=VIZ_CONFIG["lw_main"], label="Train loss")
            ax1.semilogy(time_h, val_loss,   color=STAGE_COLORS[2], lw=VIZ_CONFIG["lw_main"], ls="--", label="Val loss")
            ax2.plot(time_h, cum_tokens / 1e9, color="gray", lw=1.2, alpha=0.7, label="Cumulative tokens (B)")

            ax1.set_xlabel("Wall-clock Time (hours)")
            ax1.set_ylabel("Loss (log scale)")
            ax2.set_ylabel("Cumulative Tokens (Billions)")
            title = "Training Efficiency: Loss vs Wall-Clock Time vs Tokens"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax1.set_title(title, fontweight="bold")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            _save_and_log(fig, out, self.logger, "plots/training_dynamics/convergence/training_efficiency", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e)
            plt.close("all")
            return out

    # ------------------------------------------------------------------
    # Convenience: generate all plots in §1
    # ------------------------------------------------------------------
    def generate_all(self, data: Optional[Dict] = None, step: Optional[int] = None) -> List[Path]:
        """Generate every §1 plot and return list of saved paths."""
        d = data or {}
        methods = [
            (self.plot_multistage_loss,           d.get("loss", None)),
            (self.plot_loss_components,           d.get("loss_components", None)),
            (self.plot_per_dataset_loss_heatmap,  d.get("per_dataset_loss", None)),
            (self.plot_bitnet_lr_schedule,        d.get("lr_schedule", None)),
            (self.plot_per_group_lr,              d.get("per_group_lr", None)),
            (self.plot_gradient_norms,            d.get("grad_norms", None)),
            (self.plot_gradient_flow_heatmap,     d.get("grad_flow", None)),
            (self.plot_gradient_clipping_frequency, d.get("grad_clip_freq", None)),
            (self.plot_convergence_rate,          d.get("convergence", None)),
            (self.plot_training_efficiency,       d.get("efficiency", None)),
        ]
        paths = []
        for fn, dat in methods:
            try:
                p = fn(dat, step=step)
                paths.append(p)
                print(f"  ✓ {p.name}")
            except Exception as e:
                print(f"  ✗ {fn.__name__}: {e}")
        return paths


# ===========================================================================
# Private helpers
# ===========================================================================

def _mock_loss(n: int, start: float = 3.0, noise: float = 0.1) -> np.ndarray:
    """Exponentially decaying mock loss with noise."""
    steps = np.arange(n)
    loss  = start * np.exp(-steps / (n * 0.4)) + 0.1
    loss += np.random.normal(0, noise, n)
    return np.clip(loss, 0.01, None)


def _synthetic_loss_data() -> Dict:
    np.random.seed(42)
    s1 = np.arange(0, 2000)
    s2 = np.arange(2000, 5000)
    return {
        "stage1_steps":      s1,
        "stage2_steps":      s2,
        "stage1_train_loss": _mock_loss(len(s1), start=4.0),
        "stage1_val_loss":   _mock_loss(len(s1), start=3.8),
        "stage2_train_loss": _mock_loss(len(s2), start=2.0),
        "stage2_val_loss":   _mock_loss(len(s2), start=1.9),
    }


def _save_and_log(
    fig: plt.Figure,
    out_path: Path,
    logger: WandBLogger,
    wb_key: str,
    step: Optional[int],
):
    """Save figure to disk, log to W&B, and close."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=VIZ_CONFIG["dpi"])
    plt.close(fig)
    logger.log_image(out_path, wb_key, step=step)
