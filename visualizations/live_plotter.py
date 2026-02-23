"""
live_plotter.py
===============
Streams real training data into reliable matplotlib plots every N steps.
Plots are saved to {output_dir}/plots/ — directly next to checkpoints.

Uses direct matplotlib only (no dependency on the visualizations package)
so plots are guaranteed to appear even if other viz helpers fail.

Wired into training/train.py:
    init:      Trainer.__init__   → self.live_plotter = LivePlotter(...)
    per step:  inner train loop   → self.live_plotter.record_step(...)
    per epoch: end of epoch loop  → self.live_plotter.plot_epoch(...)
    at end:    after all epochs   → self.live_plotter.plot_final(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

# Expert domain labels (8 EmberNet experts)
_EXPERT_LABELS = [
    "OCR",
    "Diagram",
    "Chart/Math",
    "Formula",
    "Scene",
    "Spatial",
    "Knowledge",
    "Reasoning",
]

# Per-expert accent colours (matches EmberNet palette)
_EXPERT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def _smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Exponential moving average smoothing."""
    if len(values) < 2:
        return values.copy()
    out = np.empty_like(values, dtype=float)
    alpha = 2.0 / (window + 1)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


class LivePlotter:
    """
    Accumulates per-step data and generates plots at regular step intervals,
    at the end of every epoch, and at end of training.

    All plot calls are individually guarded — a broken plot never crashes
    the training loop.

    Parameters
    ----------
    output_dir   : training stage output dir; plots go to {output_dir}/plots/
    stage        : 1 or 2
    plot_interval: generate plots every this many optimizer steps
                   (default 50 for TRIAL, 200 for MAIN)
    is_trial     : True when running in --trial mode  (adds label, higher freq)
    """

    def __init__(
        self,
        output_dir: str,
        stage: int,
        plot_interval: int = 200,
        disabled: bool = False,
        is_trial: bool = False,
    ):
        self.output_dir    = Path(output_dir)
        self.stage         = stage
        self.is_trial      = is_trial
        # In trial mode, default to a tighter interval (every 50 steps max)
        self.plot_interval = max(1, min(plot_interval, 50) if is_trial else plot_interval)
        self.disabled      = disabled

        self.plots_dir = self.output_dir / "plots"

        if not disabled:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as _plt  # noqa: F401 — verify import works
                _make_dirs(self.plots_dir)
                print(f"  [viz] Plots directory : {self.plots_dir.resolve()}")
                print(f"  [viz] Plot interval   : every {self.plot_interval} steps")
                print(f"  [viz] Trial mode      : {is_trial}")
            except Exception as e:
                print(f"  [viz] Disabled — {e}")
                self.disabled = True

        # ---- per-step accumulators ----
        self._steps:        List[int]   = []
        self._losses:       List[float] = []
        self._lrs:          List[float] = []
        self._grad_norms:   List[float] = []
        self._clipped:      List[float] = []          # 1.0 if step was clipped
        self._expert_probs: List[np.ndarray] = []     # shape (8,) per step when available
        self._routing_entropy: List[float] = []       # Shannon entropy per step
        self._energy_kwh:   List[float] = []          # cumulative kWh per step (if tracked)
        self._co2_kg:       List[float] = []          # cumulative kg CO₂ per step (if tracked)

        # index into per-step lists at which each epoch ended
        self._epoch_ends: List[int] = []

    @property
    def _cur_step(self) -> int:
        """Most-recently recorded global step (0 if nothing recorded yet)."""
        return self._steps[-1] if self._steps else 0

    # ------------------------------------------------------------------
    # Public API called from train.py
    # ------------------------------------------------------------------

    def record_step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float = 0.0,
        clipped: bool = False,
        expert_probs=None,
        routing_entropy: float = 0.0,
        energy_kwh: float = 0.0,
        co2_kg: float = 0.0,
    ):
        """Accumulate metrics; trigger periodic plots every plot_interval steps."""
        if self.disabled:
            return

        self._steps.append(step)
        self._losses.append(float(loss))
        self._lrs.append(float(lr))
        self._grad_norms.append(float(grad_norm))
        self._clipped.append(1.0 if clipped else 0.0)
        self._routing_entropy.append(float(routing_entropy))
        self._energy_kwh.append(float(energy_kwh))
        self._co2_kg.append(float(co2_kg))
        if expert_probs is not None:
            arr = np.asarray(expert_probs, dtype=float)
            if arr.shape == (8,):
                self._expert_probs.append(arr)

        # Periodic mid-training plots
        if len(self._steps) % self.plot_interval == 0:
            self._plot_core(label=f"step {step}")

    def plot_epoch(self, epoch: int, stage: int):
        """Regenerate all plots at epoch end."""
        if self.disabled or not self._steps:
            return
        self._epoch_ends.append(len(self._steps))
        self._plot_core(label=f"epoch {epoch}")
        if self._expert_probs:
            self._plot_expert_heatmap()
        if len(self._routing_entropy) >= 2:
            self._plot_routing_entropy()
        if self._expert_probs:
            self._plot_expert_stacked_area()
        self._print_summary(f"Epoch {epoch}")

    def plot_final(self, stage: int):
        """Generate final summary plots after all epochs."""
        if self.disabled or not self._steps:
            return
        self._plot_core(label="final")
        self._plot_loss_distribution()
        self._plot_clipping_rate()
        if self._expert_probs:
            self._plot_expert_heatmap()
            self._plot_expert_specialisation()
            self._plot_expert_stacked_area()
        if len(self._routing_entropy) >= 2:
            self._plot_routing_entropy()
        if any(e > 0 for e in self._energy_kwh):
            self._plot_energy_curve()
        self._print_summary("Final", final=True)

    # ------------------------------------------------------------------
    # Core chart generation (direct matplotlib — always reliable)
    # ------------------------------------------------------------------

    def _savefig(self, fig, plot_dir: "Path", semantic_key: str, dpi: int = 150):
        """Write step-specific archive + latest snapshot, then close the figure."""
        import matplotlib.pyplot as plt
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        # Step-specific archive
        step_path = plot_dir / f"plot-step-{self._cur_step}-{semantic_key}.png"
        fig.savefig(step_path, dpi=dpi, bbox_inches="tight")
        # Always-latest convenience copy
        latest_path = plot_dir / f"{semantic_key}-latest.png"
        fig.savefig(latest_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def _mode_suffix(self) -> str:
        return "  (TRIAL MODE)" if self.is_trial else ""

    def _plot_core(self, label: str = ""):
        """Plot loss curve, LR schedule, and gradient norms."""
        steps  = np.array(self._steps,     dtype=float)
        losses = np.array(self._losses,    dtype=float)
        lrs    = np.array(self._lrs,       dtype=float)
        gnorms = np.array(self._grad_norms, dtype=float)

        self._safe(self._save_loss_curve,   steps, losses, label=label)
        self._safe(self._save_lr_curve,     steps, lrs,    label=label)
        self._safe(self._save_grad_norms,   steps, gnorms, label=label)
        if self._expert_probs:
            self._safe(self._save_expert_bar)

    # ---- individual plot functions ----

    def _save_loss_curve(self, steps: np.ndarray, losses: np.ndarray, label: str = ""):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, losses, color="#aec7e8", linewidth=0.8, alpha=0.6, label="raw")
        if len(losses) >= 5:
            ax.plot(steps, _smooth(losses, window=20), color="#1f77b4",
                    linewidth=2.0, label="smoothed (EMA-20)")
            # Shaded confidence band (sliding EMA std)
            smooth_vals = _smooth(losses, window=20)
            noise  = np.abs(losses - smooth_vals)
            std_band = _smooth(noise, window=20)
            ax.fill_between(steps,
                            np.clip(smooth_vals - std_band, 0, None),
                            smooth_vals + std_band,
                            color="#1f77b4", alpha=0.12, label="±1σ band")
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss — Stage {self.stage}  ({label}){self._mode_suffix()}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        _add_epoch_markers(ax, steps, self._epoch_ends)
        out = self.plots_dir / "training_dynamics" / "loss_curves"
        self._savefig(fig, out, "loss_curve")

    def _save_lr_curve(self, steps: np.ndarray, lrs: np.ndarray, label: str = ""):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.semilogy(steps, np.clip(lrs, 1e-10, None), color="#ff7f0e", linewidth=1.5)
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Learning Rate (log)")
        ax.set_title(f"LR Schedule — Stage {self.stage}  ({label}){self._mode_suffix()}")
        ax.grid(True, alpha=0.3)
        _add_epoch_markers(ax, steps, self._epoch_ends)
        out = self.plots_dir / "training_dynamics" / "learning_rates"
        self._savefig(fig, out, "lr_schedule")

    def _save_grad_norms(self, steps: np.ndarray, gnorms: np.ndarray, label: str = ""):
        if not gnorms.any():
            return
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.semilogy(steps, np.clip(gnorms, 1e-8, None), color="#9467bd",
                    linewidth=0.8, alpha=0.6, label="raw")
        if len(gnorms) >= 5:
            ax.semilogy(steps, np.clip(_smooth(gnorms, window=10), 1e-8, None),
                        color="#7b2d8b", linewidth=1.8, label="smoothed")
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0, label="clip=1.0")
        # Mark spikes
        med = float(np.median(gnorms))
        spk = gnorms > 3 * med
        if spk.any():
            ax.scatter(steps[spk], gnorms[spk], color="red", s=15, zorder=5)
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Grad Norm (log)")
        ax.set_title(f"Gradient Norms — Stage {self.stage}  ({label}){self._mode_suffix()}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        _add_epoch_markers(ax, steps, self._epoch_ends)
        out = self.plots_dir / "training_dynamics" / "gradient_stats"
        self._savefig(fig, out, "grad_norms")

    def _save_expert_bar(self):
        """Mean expert utilization bar chart (latest snapshot)."""
        import matplotlib.pyplot as plt
        probs = np.mean(self._expert_probs, axis=0)  # shape (8,)
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(_EXPERT_LABELS, probs, color=_EXPERT_COLORS, edgecolor="white")
        ax.axhline(1.0 / 8, color="black", linestyle="--", linewidth=1.0, label="uniform (1/8)")
        for bar, v in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Mean Routing Probability")
        ax.set_title(f"Expert Utilization — Stage {self.stage}{self._mode_suffix()}")
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(probs.max() * 1.25, 0.2))
        ax.grid(True, axis="y", alpha=0.3)
        out = self.plots_dir / "expert_analysis" / "expert_utilization"
        self._savefig(fig, out, "expert_utilization")

    def _plot_expert_heatmap(self):
        """Expert usage heatmap: 8 experts × N epochs."""
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
            _have_sns = True
        except ImportError:
            _have_sns = False

        if not self._epoch_ends:
            return

        n_epochs  = len(self._epoch_ends)
        matrix    = np.zeros((8, n_epochs))
        prev = 0
        for i, end in enumerate(self._epoch_ends):
            slice_ = self._expert_probs[prev:end]
            if slice_:
                matrix[:, i] = np.mean(slice_, axis=0)
            prev = end
        epoch_labels = [f"Ep{i+1}" for i in range(n_epochs)]

        fig, ax = plt.subplots(figsize=(max(6, n_epochs * 1.2), 5))
        if _have_sns:
            sns.heatmap(
                matrix, ax=ax,
                xticklabels=epoch_labels, yticklabels=_EXPERT_LABELS,
                cmap="YlOrRd", annot=True, fmt=".3f",
                linewidths=0.5,
            )
        else:
            im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(n_epochs)); ax.set_xticklabels(epoch_labels)
            ax.set_yticks(range(8));       ax.set_yticklabels(_EXPERT_LABELS)
            fig.colorbar(im, ax=ax)
            for r in range(8):
                for c in range(n_epochs):
                    ax.text(c, r, f"{matrix[r,c]:.3f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"Expert Selection Heatmap — Stage {self.stage}{self._mode_suffix()}")
        fig.tight_layout()
        out = self.plots_dir / "expert_analysis" / "routing_patterns"
        self._savefig(fig, out, "expert_heatmap")

    def _plot_loss_distribution(self):
        """Histogram of loss values across the entire training run."""
        import matplotlib.pyplot as plt
        losses = np.array(self._losses)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(losses, bins=40, color="#1f77b4", edgecolor="white", alpha=0.8)
        ax.axvline(losses.mean(), color="red",    linestyle="--", label=f"mean={losses.mean():.3f}")
        ax.axvline(np.median(losses), color="orange", linestyle="--",
                   label=f"median={np.median(losses):.3f}")
        ax.set_xlabel("Loss")
        ax.set_ylabel("Count")
        ax.set_title(f"Loss Distribution — Stage {self.stage}{self._mode_suffix()}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        out = self.plots_dir / "training_dynamics" / "convergence"
        self._savefig(fig, out, "loss_distribution")

    def _plot_clipping_rate(self):
        """Gradient clipping frequency per epoch."""
        if not self._epoch_ends:
            return
        import matplotlib.pyplot as plt
        clip_rates = []
        prev = 0
        for end in self._epoch_ends:
            window = self._clipped[prev:end]
            clip_rates.append(float(np.mean(window)) * 100.0 if window else 0.0)
            prev = end
        epochs = np.arange(1, len(clip_rates) + 1)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(epochs, clip_rates, color="#d62728", lw=2.0, marker="o", ms=5,
                label="Clipping rate")
        ax.fill_between(epochs,
                        np.array(clip_rates) * 0.85,
                        np.array(clip_rates) * 1.15,
                        color="#d62728", alpha=0.15)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Clipping Rate (%)")
        ax.set_title(f"Gradient Clipping Frequency — Stage {self.stage}{self._mode_suffix()}")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        out = self.plots_dir / "training_dynamics" / "gradient_stats"
        self._savefig(fig, out, "clipping_rate")

    def _plot_expert_specialisation(self):
        """Per-expert coefficient-of-variation across time (shows specialisation)."""
        import matplotlib.pyplot as plt
        matrix = np.array(self._expert_probs)  # (T, 8)
        cv = np.std(matrix, axis=0) / (np.mean(matrix, axis=0) + 1e-8)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(_EXPERT_LABELS, cv, color=_EXPERT_COLORS, edgecolor="white")
        ax.set_ylabel("Coefficient of Variation")
        ax.set_title(f"Expert Specialisation Index — Stage {self.stage}{self._mode_suffix()}")
        ax.grid(True, axis="y", alpha=0.3)
        out = self.plots_dir / "expert_analysis" / "specialization_metrics"
        self._savefig(fig, out, "expert_specialisation")

    # ------------------------------------------------------------------
    # New: routing entropy, stacked-area expert usage, energy curve
    # ------------------------------------------------------------------

    def _plot_routing_entropy(self):
        """Routing entropy over training steps (line + shaded band)."""
        import matplotlib.pyplot as plt
        if len(self._routing_entropy) < 2:
            return
        steps   = np.array(self._steps[:len(self._routing_entropy)], dtype=float)
        entropy = np.array(self._routing_entropy, dtype=float)
        window  = max(5, len(steps) // 20)
        roll_s  = np.array([
            entropy[max(0, i - window // 2): i + window // 2].std() + 0.01
            for i in range(len(entropy))
        ])
        import math
        max_ent = math.log(8)   # log(N_EXPERTS)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(steps, entropy, color="#e377c2", lw=1.8, label="Routing entropy")
        ax.fill_between(steps, np.clip(entropy - roll_s, 0, None), entropy + roll_s,
                        color="#e377c2", alpha=0.20)
        ax.axhline(max_ent, color="gray", lw=1.0, ls="--",
                   label=f"Max entropy ({max_ent:.2f})")
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Entropy (nats)")
        ax.set_ylim(0, max_ent * 1.1)
        ax.set_title(f"Router Entropy — Stage {self.stage}{self._mode_suffix()}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        out = self.plots_dir / "expert_analysis" / "routing_patterns"
        self._savefig(fig, out, "routing_entropy")

    def _plot_expert_stacked_area(self):
        """Stacked area chart of expert token-routing shares over time."""
        import matplotlib.pyplot as plt
        if not self._expert_probs:
            return
        steps  = np.array(self._steps[:len(self._expert_probs)], dtype=float)
        matrix = np.array(self._expert_probs)   # (T, 8)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.stackplot(
            steps,
            [matrix[:, i] * 100 for i in range(8)],
            labels=_EXPERT_LABELS,
            colors=_EXPERT_COLORS,
            alpha=0.85,
        )
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("% Tokens Routed")
        ax.set_ylim(0, 100)
        ax.set_title(f"Expert Token Routing (Stacked) — Stage {self.stage}{self._mode_suffix()}")
        ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(True, axis="x", alpha=0.2)
        out = self.plots_dir / "expert_analysis" / "expert_utilization"
        self._savefig(fig, out, "expert_stacked_area")

    def _plot_energy_curve(self):
        """Cumulative energy and CO₂ over training steps (dual-axis line)."""
        import matplotlib.pyplot as plt
        if not any(e > 0 for e in self._energy_kwh):
            return
        steps   = np.array(self._steps[:len(self._energy_kwh)], dtype=float)
        energy  = np.array(self._energy_kwh, dtype=float)
        co2     = np.array(self._co2_kg[:len(steps)], dtype=float)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(steps, energy, color="#2ca02c", lw=2.0, label="Cumul. energy (kWh)")
        ax.fill_between(steps, 0, energy, color="#2ca02c", alpha=0.15)
        ax2 = ax.twinx()
        if co2.any():
            ax2.plot(steps, co2, color="#555555", lw=1.2, ls="--", label="Cumul. CO₂ (kg)")
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Cumulative Energy (kWh)", color="#2ca02c")
        ax2.set_ylabel("Cumulative CO₂ (kg)", color="#555555")
        ax.set_title(f"Training Energy — Stage {self.stage}{self._mode_suffix()}")
        l1, ll1 = ax.get_legend_handles_labels()
        l2, ll2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, ll1 + ll2, fontsize=9)
        ax.grid(True, alpha=0.3)
        out = self.plots_dir / "efficiency"
        self._savefig(fig, out, "energy_curve")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _safe(self, fn, *args, **kwargs):
        """Call a plot function — print full traceback on error, never crash training."""
        try:
            fn(*args, **kwargs)
        except Exception as e:
            import traceback
            print(f"  [viz] ERROR in {fn.__name__}: {e}")
            traceback.print_exc()

    def _print_summary(self, label: str = "", final: bool = False):
        try:
            n = sum(1 for f in self.plots_dir.rglob("*.png") if f.is_file())
        except Exception:
            n = 0
        tag = "saved" if final else "updated"
        print(f"  [viz] {label} plots {tag} — {n} PNG files → {self.plots_dir.resolve()}")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_dirs(plots_dir: Path):
    """Pre-create the standard subfolder tree."""
    for sub in [
        "training_dynamics/loss_curves",
        "training_dynamics/learning_rates",
        "training_dynamics/gradient_stats",
        "training_dynamics/convergence",
        "expert_analysis/routing_patterns",
        "expert_analysis/expert_utilization",
        "expert_analysis/specialization_metrics",
        "efficiency/energy_metrics",
        "efficiency/co2_metrics",
        "efficiency/efficiency_tradeoffs",
    ]:
        (plots_dir / sub).mkdir(parents=True, exist_ok=True)


def _add_epoch_markers(ax, steps: np.ndarray, epoch_ends: List[int]):
    """Draw vertical grey dashed lines at epoch boundaries."""
    if not epoch_ends or len(steps) == 0:
        return
    for end_idx in epoch_ends:
        if 0 < end_idx <= len(steps):
            x = steps[end_idx - 1]
            ax.axvline(x, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
