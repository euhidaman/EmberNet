"""
Stage Comparison Visualizations

Covers:
  7.1  Stage 1 vs Stage 2   – loss side-by-side, parameter update magnitudes,
                               expert routing before/after stage 2
  7.2  Ablation Studies     – number of experts, routing strategy, bitwidth impact
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from visualizations.config import (
    VIZ_CONFIG, PLOT_DIRS, STAGE_COLORS, EXPERT_NAMES, EXPERT_COLORS, EXPERT_LABELS,
    ALL_DATASETS, DATASET_DOMAINS, DOMAIN_COLORS,
    apply_mpl_style, plot_filename, log_plot_error,
)
from visualizations.training_dynamics import _save_and_log, _mock_loss
from visualizations.wandb_utils import WandBLogger

apply_mpl_style()


class StageComparisonPlotter:
    """Generates all §7 stage-comparison and ablation plots."""

    def __init__(self, logger: Optional[WandBLogger] = None):
        self.logger = logger or WandBLogger(disabled=True)
        self._generated: List[Path] = []

    # ==================================================================
    # 7.1 Stage 1 vs Stage 2
    # ==================================================================

    def plot_loss_side_by_side(self, data=None, step=None) -> Path:
        """Plot 7.1.1 – Loss Comparison: Stage 1 vs Stage 2 (side-by-side)."""
        key = "loss_stage1_vs_stage2"
        out = PLOT_DIRS["stage1_vs_stage2"] / plot_filename("stage_comparison", "stage_vs_stage", key)
        try:
            incomplete = data is None
            if data is None:
                np.random.seed(150)
                s1_n = 2000
                s2_n = 3000
                data = {
                    "s1_steps": np.arange(s1_n),
                    "s1_train":  _mock_loss(s1_n, start=4.0),
                    "s1_val":    _mock_loss(s1_n, start=3.8),
                    "s2_steps": np.arange(s2_n),
                    "s2_train":  _mock_loss(s2_n, start=2.0),
                    "s2_val":    _mock_loss(s2_n, start=1.9),
                }

            s1_steps = np.asarray(data["s1_steps"])
            s1_train = np.asarray(data["s1_train"])
            s1_val   = np.asarray(data["s1_val"])
            s2_steps = np.asarray(data["s2_steps"])
            s2_train = np.asarray(data["s2_train"])
            s2_val   = np.asarray(data["s2_val"])

            y_max = max(s1_train.max(), s2_train.max(), s1_val.max(), s2_val.max())
            y_min = min(s1_train.min(), s2_train.min(), s1_val.min(), s2_val.min()) * 0.8

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ_CONFIG["figsize_dual"], sharey=True)
            for ax, steps, train_l, val_l, stage in [
                (ax1, s1_steps, s1_train, s1_val, 1),
                (ax2, s2_steps, s2_train, s2_val, 2),
            ]:
                ax.semilogy(steps, train_l, color=STAGE_COLORS[stage], lw=VIZ_CONFIG["lw_main"],
                            label="Train loss")
                ax.semilogy(steps, val_l,   color=STAGE_COLORS[stage], lw=VIZ_CONFIG["lw_dashed"],
                            ls="--", label="Val loss")
                ax.set_xlabel("Steps")
                ax.set_ylabel("Loss (log)")
                ax.set_title(f"Stage {stage}", fontweight="bold")
                ax.legend()
                ax.set_ylim(y_min, y_max * 1.2)

            title = "Stage 1 vs Stage 2 – Loss Curves"
            if incomplete:
            fig.suptitle(title, fontsize=VIZ_CONFIG["font_title"], fontweight="bold")

            out = _save_and_log(fig, out, self.logger, "plots/stage_comparison/stage1_vs_stage2/loss_comparison", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_parameter_update_magnitude(self, data=None, step=None) -> Path:
        """Plot 7.1.2 – Parameter Update Magnitude Heatmap."""
        key = "parameter_update_magnitude"
        out = PLOT_DIRS["stage1_vs_stage2"] / plot_filename("stage_comparison", "stage_vs_stage", key)
        try:
            incomplete = data is None
            param_groups = [
                "embed_tokens", "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
                "router", "expert.gate", "expert.up", "expert.down",
                "shared_expert", "norm", "lm_head", "projector",
            ]
            n_ckpts = 8
            if data is None:
                np.random.seed(151)
                s1_updates = np.random.uniform(0, 0.1, (len(param_groups), n_ckpts))
                s2_updates = np.random.uniform(0, 0.2, (len(param_groups), n_ckpts))
                # Projector updates high in stage 1
                proj_idx = param_groups.index("projector")
                s1_updates[proj_idx] *= 3
                # Router / expert updates high in stage 2
                router_idx = param_groups.index("router")
                s2_updates[router_idx] *= 2.5
                ckpt_labels = [f"ckpt {i+1}" for i in range(n_ckpts)]
                data = {
                    "param_groups": param_groups,
                    "s1_updates": s1_updates,
                    "s2_updates": s2_updates,
                    "ckpt_labels": ckpt_labels,
                }

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
            ckpt_labels = data.get("ckpt_labels", [str(i) for i in range(data["s1_updates"].shape[1])])
            kw = dict(yticklabels=data["param_groups"], cmap="YlOrRd", linewidths=0.3)
            sns.heatmap(data["s1_updates"], ax=ax1, xticklabels=ckpt_labels,
                        cbar_kws={"label": "Update magnitude"}, annot=True, fmt=".3f",
                        annot_kws={"size": 7}, **kw)
            ax1.set_title("Stage 1 – Parameter Update Magnitude", fontweight="bold")
            ax1.tick_params(axis="x", rotation=30)

            sns.heatmap(data["s2_updates"], ax=ax2, xticklabels=ckpt_labels,
                        cbar_kws={"label": "Update magnitude"}, annot=True, fmt=".3f",
                        annot_kws={"size": 7}, **kw)
            ax2.set_title("Stage 2 – Parameter Update Magnitude", fontweight="bold")
            ax2.tick_params(axis="x", rotation=30)

            title = "Parameter Update Magnitudes by Stage"
            if incomplete:
            fig.suptitle(title, fontsize=VIZ_CONFIG["font_title"], fontweight="bold")

            out = _save_and_log(fig, out, self.logger, "plots/stage_comparison/stage1_vs_stage2/param_updates", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_routing_before_after_stage2(self, data=None, step=None) -> Path:
        """Plot 7.1.3 – Expert Routing Before/After Stage 2 (dual spider)."""
        key = "routing_before_after_stage2"
        out = PLOT_DIRS["stage1_vs_stage2"] / plot_filename("stage_comparison", "stage_vs_stage", key)
        try:
            incomplete = data is None
            N = len(EXPERT_NAMES)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

            if data is None:
                np.random.seed(152)
                # Before: near-uniform
                before = {name: np.random.uniform(0.08, 0.18, N) for name in EXPERT_NAMES}
                # After: specialized
                after  = {}
                for i, name in enumerate(EXPERT_NAMES):
                    scores = np.random.uniform(0.05, 0.25, N)
                    scores[i] = np.random.uniform(0.55, 0.85)
                    after[name] = np.clip(scores, 0, 1)
                # Ideal
                ideal = {}
                for i, name in enumerate(EXPERT_NAMES):
                    s = np.full(N, 0.05)
                    s[i] = 1.0
                    ideal[name] = s
                data = {"before": before, "after": after, "ideal": ideal}

            def draw_spider_ax(ax, perf_dict, ideal_dict=None, title=""):
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_xticks(angles)
                ax.set_xticklabels([e.replace("_", "\n") for e in EXPERT_NAMES], fontsize=6)
                ax.set_yticks([0.25, 0.5, 0.75, 1.0])
                ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=5)
                ax.set_ylim(0, 1)
                for name in EXPERT_NAMES:
                    vals = np.asarray(perf_dict.get(name, np.full(N, 0.1)))
                    vals_loop = np.concatenate([vals, [vals[0]]])
                    angs_loop = np.concatenate([angles, [angles[0]]])
                    ax.plot(angs_loop, vals_loop, color=EXPERT_COLORS[name], lw=1.5)
                    ax.fill(angs_loop, vals_loop, color=EXPERT_COLORS[name], alpha=0.15)
                if ideal_dict:
                    for name in EXPERT_NAMES:
                        ideal_v = np.asarray(ideal_dict.get(name, np.full(N, 0.1)))
                        iv_loop = np.concatenate([ideal_v, [ideal_v[0]]])
                        ax.plot(angs_loop, iv_loop, color=EXPERT_COLORS[name], lw=0.7, ls=":")
                ax.set_title(title, fontsize=9, fontweight="bold", pad=10)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                           subplot_kw=dict(polar=True))
            draw_spider_ax(ax1, data["before"], data.get("ideal"), title="End of Stage 1 (near-uniform)")
            draw_spider_ax(ax2, data["after"],  data.get("ideal"), title="End of Stage 2 (specialized)")

            title = "Expert Routing: Before vs After Stage 2"
            if incomplete:
            fig.suptitle(title, fontsize=VIZ_CONFIG["font_title"], fontweight="bold")

            out = _save_and_log(fig, out, self.logger, "plots/stage_comparison/stage1_vs_stage2/routing_before_after", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    # ==================================================================
    # 7.2 Ablation Studies
    # ==================================================================

    def plot_ablation_num_experts(self, data=None, step=None) -> Path:
        """Plot 7.2.1 – Ablation: Number of Experts."""
        key = "ablation_num_experts"
        out = PLOT_DIRS["ablation_studies"] / plot_filename("stage_comparison", "ablation", key)
        try:
            incomplete = data is None
            num_experts_list = [2, 4, 8, 16]
            if data is None:
                np.random.seed(160)
                acc_by_ds = {}
                for ds in ALL_DATASETS[:5]:
                    acc_by_ds[ds] = np.random.uniform(55, 80, len(num_experts_list)) + \
                                    np.array([0, 3, 5, 4.5])
                data = {"num_experts": num_experts_list, "acc_by_ds": acc_by_ds}

            x = np.asarray(data["num_experts"])
            acc_by_ds = data["acc_by_ds"]

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            for i, (ds, vals) in enumerate(acc_by_ds.items()):
                domain = _dataset_domain(ds)
                color  = DOMAIN_COLORS.get(domain, "#aaaaaa")
                ax.plot(x, vals, color=color, lw=VIZ_CONFIG["lw_main"],
                        marker="o", ms=6, label=ds)

            ax.set_xticks(x)
            ax.set_xticklabels([str(n) for n in x])
            ax.set_xlabel("Number of Experts")
            ax.set_ylabel("Validation Accuracy (%)")
            title = "Ablation: Number of Experts vs Accuracy"
            if incomplete:
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=8)

            out = _save_and_log(fig, out, self.logger, "plots/stage_comparison/ablation/num_experts", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_ablation_routing_strategy(self, data=None, step=None) -> Path:
        """Plot 7.2.2 – Ablation: Expert Routing Strategy."""
        key = "ablation_routing_strategy"
        out = PLOT_DIRS["ablation_studies"] / plot_filename("stage_comparison", "ablation", key)
        try:
            incomplete = data is None
            strategies = ["Top-1", "Top-2", "Top-4", "Soft\nRouting"]
            if data is None:
                np.random.seed(161)
                accuracy = np.array([64.5, 68.2, 67.8, 66.1])
                size_mb  = np.array([100.0, 135.0, 200.0, 180.0])
                speed    = np.array([320.0, 280.0, 180.0, 210.0])
                data = {"strategies": strategies, "accuracy": accuracy,
                        "size_mb": size_mb, "speed": speed}

            strategies = data.get("strategies", strategies)
            accuracy   = np.asarray(data["accuracy"])
            size_mb    = np.asarray(data["size_mb"])
            speed      = np.asarray(data["speed"])
            x = np.arange(len(strategies))
            w = 0.25

            fig, ax1 = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            ax2 = ax1.twinx()
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("axes", 1.12))

            bars1 = ax1.bar(x - w, accuracy, w, label="Accuracy (%)", color="#1f77b4", alpha=0.85)
            bars2 = ax2.bar(x,     size_mb,  w, label="Model size (MB)", color="#ff7f0e", alpha=0.85)
            bars3 = ax3.bar(x + w, speed,    w, label="Speed (tok/s)", color="#2ca02c", alpha=0.85)

            ax1.set_xticks(x)
            ax1.set_xticklabels(strategies, fontsize=9)
            ax1.set_ylabel("Accuracy (%)",        color="#1f77b4")
            ax2.set_ylabel("Model Size (MB)",     color="#ff7f0e")
            ax3.set_ylabel("Inference Speed (tok/s)", color="#2ca02c")
            title = "Ablation: Routing Strategy Comparison"
            if incomplete:
            ax1.set_title(title, fontweight="bold")

            lines = [bars1, bars2, bars3]
            ax1.legend(
                [b.patches[0] for b in lines],
                ["Accuracy", "Model size", "Speed"],
                loc="lower right",
            )

            out = _save_and_log(fig, out, self.logger, "plots/stage_comparison/ablation/routing_strategy", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_ablation_quantization(self, data=None, step=None) -> Path:
        """Plot 7.2.3 – Ablation: BitNet Quantization Impact (heatmap table)."""
        key = "ablation_quantization_impact"
        out = PLOT_DIRS["ablation_studies"] / plot_filename("stage_comparison", "ablation", key)
        try:
            incomplete = data is None
            levels  = ["FP32", "FP16", "Ternary", "Binary"]
            metrics = ["Accuracy (%)", "Model Size (MB)", "Inference Speed\n(tok/s)", "Peak Memory\n(MB)"]
            if data is None:
                table = np.array([
                    [75.0, 1540.0, 120.0, 3200.0],  # FP32
                    [74.5,  770.0, 230.0, 1600.0],  # FP16
                    [72.0,  135.0, 280.0,  350.0],  # Ternary  (EmberNet)
                    [65.0,  100.0, 310.0,  260.0],  # Binary
                ])
                data = {"table": table}

            table = np.asarray(data["table"])
            # Normalise per column: 0=worst 1=best
            normed = np.zeros_like(table)
            # Accuracy: higher is better
            normed[:, 0] = (table[:, 0] - table[:, 0].min()) / ((table[:, 0].max() - table[:, 0].min()) + 1e-8)
            # Size: lower is better
            normed[:, 1] = 1 - (table[:, 1] - table[:, 1].min()) / ((table[:, 1].max() - table[:, 1].min()) + 1e-8)
            # Speed: higher is better
            normed[:, 2] = (table[:, 2] - table[:, 2].min()) / ((table[:, 2].max() - table[:, 2].min()) + 1e-8)
            # Memory: lower is better
            normed[:, 3] = 1 - (table[:, 3] - table[:, 3].min()) / ((table[:, 3].max() - table[:, 3].min()) + 1e-8)

            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.imshow(normed, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

            # Annotate with raw values
            for i in range(len(levels)):
                for j in range(len(metrics)):
                    raw = table[i, j]
                    fmt = f"{raw:.0f}" if raw > 10 else f"{raw:.1f}"
                    ax.text(j, i, fmt, ha="center", va="center", fontsize=9, fontweight="bold")

            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(metrics, fontsize=9)
            ax.set_yticks(range(len(levels)))
            ax.set_yticklabels(levels)
            fig.colorbar(im, ax=ax, label="0=worst → 1=best (per-column)")
            title = "Ablation: Quantization Impact (green=better)"
            if incomplete:
            ax.set_title(title, fontweight="bold")

            out = _save_and_log(fig, out, self.logger, "plots/stage_comparison/ablation/quantization_impact", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def generate_all(self, data=None, step=None) -> List[Path]:
        d = data or {}
        methods = [
            (self.plot_loss_side_by_side,            d.get("loss_sbs")),
            (self.plot_parameter_update_magnitude,   d.get("param_updates")),
            (self.plot_routing_before_after_stage2,  d.get("routing_ba")),
            (self.plot_ablation_num_experts,         d.get("abl_experts")),
            (self.plot_ablation_routing_strategy,    d.get("abl_routing")),
            (self.plot_ablation_quantization,        d.get("abl_quant")),
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


def _dataset_domain(ds: str) -> str:
    from visualizations.config import DATASET_DOMAINS
    for domain, ds_list in DATASET_DOMAINS.items():
        if ds in ds_list:
            return domain
    return "alignment"
