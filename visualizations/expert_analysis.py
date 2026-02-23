"""
Expert Analysis Visualizations

Covers:
  2.1  Routing Patterns (heatmap, co-occurrence, by-dataset bar, Sankey snapshots)
  2.2  Specialization Metrics (specialization index, weight sparsity, output variance)
  2.3  Expert Utilization (load balancing, violin, dead-expert detection)
  2.4  Spider Charts (per-expert, comparative, temporal evolution)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns

from visualizations.config import (
    VIZ_CONFIG, PLOT_DIRS, EXPERT_NAMES, EXPERT_COLORS, EXPERT_LABELS,
    ALL_DATASETS, DATASET_DOMAINS, DOMAIN_COLORS,
    apply_mpl_style, plot_filename, log_plot_error,
)
from visualizations.training_dynamics import _save_and_log
from visualizations.wandb_utils import WandBLogger

apply_mpl_style()

N_EXPERTS = len(EXPERT_NAMES)
_COLORS   = [EXPERT_COLORS[e] for e in EXPERT_NAMES]


class ExpertAnalysisPlotter:
    """Generates all §2 expert-analysis plots."""

    def __init__(self, logger: Optional[WandBLogger] = None):
        self.logger = logger or WandBLogger(disabled=True)
        self._generated: List[Path] = []

    # ==================================================================
    # 2.1 Routing Patterns
    # ==================================================================

    def plot_expert_selection_heatmap(self, data=None, step=None) -> Path:
        """Plot 2.1.1 – Expert Selection Frequency over Checkpoints."""
        key = "expert_selection_heatmap"
        out = PLOT_DIRS["routing_patterns"] / plot_filename("expert_analysis", "routing_patterns", key)
        try:
            n_ckpts = 10
            if data is None:
                np.random.seed(10)
                # Start near uniform, end near specialized
                freq = np.random.dirichlet(np.ones(N_EXPERTS) * 2, size=n_ckpts).T
                ckpt_labels = [f"step\n{(i+1)*500}" for i in range(n_ckpts)]
            else:
                freq = np.asarray(data["freq"])
                ckpt_labels = data.get("ckpt_labels", [str(i) for i in range(freq.shape[1])])

            fig, ax = plt.subplots(figsize=(12, 6))
            im = sns.heatmap(
                freq, ax=ax,
                xticklabels=ckpt_labels,
                yticklabels=[EXPERT_LABELS[e] for e in EXPERT_NAMES],
                cmap="Blues", vmin=0, vmax=1,
                annot=True, fmt=".2f",
                cbar_kws={"label": "Selection Frequency (0-1)"},
            )
            ax.set_xlabel("Training Checkpoint")
            ax.set_ylabel("Expert")
            title = "Expert Selection Frequency over Training"
            ax.set_title(title, fontweight="bold")

            out = _save_and_log(fig, out, self.logger, "plots/expert_analysis/routing_patterns/selection_heatmap", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_cooccurrence_matrix(self, data=None, step=None) -> Path:
        """Plot 2.1.2 – Top-2 Expert Co-occurrence Matrix."""
        key = "expert_cooccurrence_matrix"
        out = PLOT_DIRS["routing_patterns"] / plot_filename("expert_analysis", "routing_patterns", key)
        try:
            if data is None:
                np.random.seed(11)
                mat = np.random.uniform(0.01, 0.3, (N_EXPERTS, N_EXPERTS))
                mat = (mat + mat.T) / 2  # symmetric
                np.fill_diagonal(mat, np.random.uniform(0.05, 0.4, N_EXPERTS))
            else:
                mat = np.asarray(data["matrix"])

            labels = [EXPERT_LABELS[e] for e in EXPERT_NAMES]
            fig, ax = plt.subplots(figsize=(9, 8))
            mask = np.zeros_like(mat, dtype=bool)  # show full matrix (symmetric)
            sns.heatmap(
                mat, ax=ax, mask=mask,
                xticklabels=labels, yticklabels=labels,
                cmap="YlOrRd", vmin=0, vmax=mat.max(),
                annot=True, fmt=".2f",
                cbar_kws={"label": "Co-occurrence Probability"},
            )
            ax.set_title(
                "Top-2 Expert Co-occurrence Matrix",
                fontweight="bold",
            )
            plt.xticks(rotation=45, ha="right")

            out = _save_and_log(fig, out, self.logger, "plots/expert_analysis/routing_patterns/cooccurrence", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_routing_by_dataset(self, data=None, step=None) -> Path:
        """Plot 2.1.3 – Expert Routing by Dataset Domain (stacked bar)."""
        key = "expert_routing_by_dataset"
        out = PLOT_DIRS["routing_patterns"] / plot_filename("expert_analysis", "routing_patterns", key)
        try:
            n_ds = len(ALL_DATASETS)
            if data is None:
                np.random.seed(12)
                routing = np.random.dirichlet(np.ones(N_EXPERTS), size=n_ds).T
            else:
                routing = np.asarray(data["routing"])

            x     = np.arange(n_ds)
            bottoms = np.zeros(n_ds)
            fig, ax = plt.subplots(figsize=(18, 6))
            for i, (name, color) in enumerate(zip(EXPERT_NAMES, _COLORS)):
                ax.bar(x, routing[i] * 100, bottom=bottoms * 100,
                       color=color, label=EXPERT_LABELS[name], alpha=0.9, width=0.75)
                bottoms += routing[i]

            ax.set_xticks(x)
            ax.set_xticklabels(ALL_DATASETS, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("% Tokens Routed")
            ax.set_ylim(0, 100)
            title = "Expert Routing by Dataset"
            ax.set_title(title, fontweight="bold")
            ax.legend(ncol=4, fontsize=8, loc="upper right")

            # Domain group separators
            cum = 0
            for domain, ds_list in DATASET_DOMAINS.items():
                cum += len(ds_list)
                if cum < n_ds:
                    ax.axvline(cum - 0.5, color="black", lw=1.2, ls="--", alpha=0.5)

            out = _save_and_log(fig, out, self.logger, "plots/expert_analysis/routing_patterns/routing_by_dataset", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_routing_sankey_snapshot(self, data=None, step=None) -> Path:
        """Plot 2.1.4 – Sankey Snapshot (static alternative to animation)."""
        key = "routing_sankey_snapshot"
        out = PLOT_DIRS["routing_patterns"] / plot_filename("expert_analysis", "routing_patterns", key)
        try:
            # Use plotly if available, else fall back to a simple stacked-bar snapshot
            try:
                import plotly.graph_objects as go
                import plotly.io as pio

                if data is None:
                    np.random.seed(13)
                    token_flows = np.random.dirichlet(np.ones(N_EXPERTS), size=8)
                    # rows = input domains, cols = experts
                    mean_flow = token_flows.mean(axis=0)
                    source_labels = [d for d in DATASET_DOMAINS]
                    target_labels = [EXPERT_LABELS[e] for e in EXPERT_NAMES]

                sources, targets, values, link_colors = [], [], [], []
                n_src = len(source_labels)
                for si in range(n_src):
                    for ti in range(N_EXPERTS):
                        sources.append(si)
                        targets.append(n_src + ti)
                        values.append(float(token_flows[si % len(token_flows), ti]))
                        link_colors.append(_COLORS[ti])

                fig_sankey = go.Figure(go.Sankey(
                    node=dict(
                        label=source_labels + target_labels,
                        color=["#cccccc"] * n_src + _COLORS,
                        pad=15, thickness=20,
                    ),
                    link=dict(source=sources, target=targets, value=values,
                              color=[f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.38)" for c in link_colors]),
                ))
                title = "Token Routing Sankey (Snapshot)"
                fig_sankey.update_layout(title_text=title, font_size=11, height=500)
                pio.write_image(fig_sankey, str(out), width=1200, height=600, scale=2)
                self.logger.log_image(out, "plots/expert_analysis/routing_patterns/sankey_snapshot", step=step)
                self._generated.append(out)
                return out
            except (ImportError, Exception) as plotly_err:
                # Fallback: save a placeholder text file
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    f"Sankey plot requires plotly + kaleido.\nError: {plotly_err}\n"
                    "Install with: pip install plotly kaleido"
                )
                print(f"  [WARNING] Sankey plot skipped (plotly not available): {plotly_err}")
                return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    # ==================================================================
    # 2.2 Specialization Metrics
    # ==================================================================

    def plot_specialization_index(self, data=None, step=None) -> Path:
        """Plot 2.2.1 – Expert Specialization Index over Training."""
        key = "expert_specialization_index"
        out = PLOT_DIRS["specialization_metrics"] / plot_filename("expert_analysis", "specialization_metrics", key)
        try:
            if data is None:
                np.random.seed(20)
                n = 5000
                steps = np.arange(n)
                indices = {}
                for i, name in enumerate(EXPERT_NAMES):
                    # Specialization rises from ~0.1 to ~0.7-0.9
                    target = np.random.uniform(0.6, 0.95)
                    curve  = target * (1 - np.exp(-steps / (n * 0.35))) + \
                             np.random.normal(0, 0.015, n)
                    indices[name] = np.clip(curve, 0, 1)
                data = {"steps": steps, "indices": indices}

            steps   = np.asarray(data["steps"])
            indices = data["indices"]

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            for name in EXPERT_NAMES:
                if name in indices:
                    ax.plot(steps, indices[name],
                            color=EXPERT_COLORS[name], lw=VIZ_CONFIG["lw_main"],
                            label=EXPERT_LABELS[name], alpha=0.9)

            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Specialization Index (0=uniform, 1=perfect)")
            title = "Expert Specialization Index"
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=8, ncol=2)
            ax.set_ylim(0, 1.05)

            out = _save_and_log(fig, out, self.logger, "plots/expert_analysis/specialization_metrics/specialization_index", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_weight_sparsity_grid(self, data=None, step=None) -> Path:
        """Plot 2.2.2 – Expert Weight Sparsity (2×4 subplot grid)."""
        key = "expert_weight_sparsity_grid"
        out = PLOT_DIRS["specialization_metrics"] / plot_filename("expert_analysis", "specialization_metrics", key)
        try:
            if data is None:
                np.random.seed(21)
                # Synthetic ternary distributions per expert
                dists = {}
                for name in EXPERT_NAMES:
                    p_neg  = np.random.uniform(0.2, 0.35)
                    p_zero = np.random.uniform(0.25, 0.60)
                    p_pos  = 1 - p_neg - p_zero
                    dists[name] = {"neg": p_neg, "zero": p_zero, "pos": p_pos}
                data = {"dists_final": dists, "dists_init": {
                    n: {"neg": 0.33, "zero": 0.34, "pos": 0.33} for n in EXPERT_NAMES
                }}

            dists_final = data["dists_final"]
            dists_init  = data.get("dists_init", {
                n: {"neg": 0.33, "zero": 0.34, "pos": 0.33} for n in EXPERT_NAMES
            })

            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            for idx, (name, ax) in enumerate(zip(EXPERT_NAMES, axes)):
                d_i = dists_init.get(name, {"neg": 0.33, "zero": 0.34, "pos": 0.33})
                d_f = dists_final.get(name, {"neg": 0.33, "zero": 0.34, "pos": 0.33})
                x  = np.array([-1, 0, 1])
                init_vals  = [d_i["neg"], d_i["zero"], d_i["pos"]]
                final_vals = [d_f["neg"], d_f["zero"], d_f["pos"]]
                w = 0.35
                ax.bar(x - w/2, init_vals,  w, label="Initial", color="lightgray", edgecolor="black")
                ax.bar(x + w/2, final_vals, w, label="Final",   color=EXPERT_COLORS[name], alpha=0.8, edgecolor="black")
                ax.set_xticks([-1, 0, 1])
                ax.set_xticklabels(["-1", "0", "+1"])
                ax.set_title(f"{EXPERT_LABELS[name]}\nSparsity={d_f['zero']*100:.1f}%", fontsize=9, fontweight="bold")
                ax.set_ylabel("Proportion")
                if idx == 0:
                    ax.legend(fontsize=7)

            title = "Expert Ternary Weight Distributions (Initial vs Final)"
            fig.suptitle(title, fontsize=VIZ_CONFIG["font_title"], fontweight="bold")

            out = _save_and_log(fig, out, self.logger, "plots/expert_analysis/specialization_metrics/weight_sparsity_grid", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_output_variance_boxplot(self, data=None, step=None) -> Path:
        """Plot 2.2.3 – Expert Output Variance by Domain."""
        key = "expert_output_variance_boxplot"
        out = PLOT_DIRS["specialization_metrics"] / plot_filename("expert_analysis", "specialization_metrics", key)
        try:
            domains = list(DATASET_DOMAINS.keys())
            if data is None:
                np.random.seed(22)
                variance_data = {
                    name: {d: np.random.exponential(0.5, 50) + 0.1 for d in domains}
                    for name in EXPERT_NAMES
                }
                data = {"variance": variance_data}

            var_data = data["variance"]
            fig, axes = plt.subplots(1, len(domains), figsize=(20, 6), sharey=True)
            for di, domain in enumerate(domains):
                ax = axes[di]
                box_vals = [var_data.get(name, {}).get(domain, [0.0]) for name in EXPERT_NAMES]
                bp = ax.boxplot(
                    box_vals, patch_artist=True,
                    medianprops=dict(color="black", lw=2),
                )
                for patch, color in zip(bp["boxes"], _COLORS):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.75)
                ax.set_xticks(range(1, N_EXPERTS + 1))
                ax.set_xticklabels([f"E{i}" for i in range(N_EXPERTS)], fontsize=7)
                ax.set_title(domain.replace("_", "\n"), fontsize=8)
                if di == 0:
                    ax.set_ylabel("Output Activation Variance")

            title = "Expert Output Variance by Domain"
            fig.suptitle(title, fontsize=VIZ_CONFIG["font_title"], fontweight="bold")

            out = _save_and_log(fig, out, self.logger, "plots/expert_analysis/specialization_metrics/output_variance", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    # ==================================================================
    # 2.3 Expert Utilization
    # ==================================================================

    def plot_load_balancing(self, data=None, step=None) -> Path:
        """Plot 2.3.1 – Expert Load Balancing (imbalance coefficient)."""
        key = "expert_load_balancing"
        out = PLOT_DIRS["expert_utilization"] / plot_filename("expert_analysis", "expert_utilization", key)
        try:
            if data is None:
                np.random.seed(30)
                n = 5000
                steps = np.arange(n)
                imbalance = 0.8 * np.exp(-steps / (n * 0.5)) + 0.05 + \
                            np.random.normal(0, 0.02, n)
                imbalance = np.clip(imbalance, 0, None)
                data = {"steps": steps, "imbalance": imbalance, "target": 0.1}

            steps     = np.asarray(data["steps"])
            imbalance = np.asarray(data["imbalance"])
            target    = data.get("target", 0.1)

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            ax.plot(steps, imbalance, color="#1f77b4", lw=VIZ_CONFIG["lw_main"], label="Load imbalance")
            ax.axhline(target, color="red", lw=VIZ_CONFIG["lw_dashed"], ls="--", label=f"Target ({target})")
            ax.fill_between(steps, 0, imbalance, alpha=VIZ_CONFIG["alpha_fill"], color="#1f77b4")
            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Load Imbalance Coefficient\n(std / mean usage)")
            title = "Expert Load Balancing"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            out = _save_and_log(fig, out, self.logger, "plots/expert_analysis/expert_utilization/load_balancing", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_usage_violin(self, data=None, step=None) -> Path:
        """Plot 2.3.2 – Expert Usage Distribution (Violin)."""
        key = "expert_usage_violin"
        out = PLOT_DIRS["expert_utilization"] / plot_filename("expert_analysis", "expert_utilization", key)
        try:
            if data is None:
                np.random.seed(31)
                usage_data = {
                    name: np.random.beta(a, b, size=300) * 30
                    for name, (a, b) in zip(EXPERT_NAMES, [
                        (3.0, 1.5), (2.5, 2.0), (3.5, 1.2), (2.0, 2.5),
                        (1.5, 3.0), (3.0, 1.8), (2.8, 1.6), (2.2, 2.2),
                    ])
                }
                data = {"usage": usage_data}

            usage = data["usage"]
            import pandas as pd
            rows = []
            for name in EXPERT_NAMES:
                for val in usage.get(name, [0.0]):
                    rows.append({"expert": EXPERT_LABELS[name], "usage_pct": val})
            df = pd.DataFrame(rows)

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            palette = {EXPERT_LABELS[e]: EXPERT_COLORS[e] for e in EXPERT_NAMES}
            sns.violinplot(data=df, x="expert", y="usage_pct", ax=ax,
                           palette=palette, cut=0, inner="box")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
            ax.set_xlabel("Expert")
            ax.set_ylabel("Token Usage (% of batch)")
            # Ideal uniform line
            ax.axhline(100 / N_EXPERTS, color="black", ls="--", lw=1.5, label="Ideal uniform")
            ax.legend()
            title = "Expert Token Usage Distribution"
            ax.set_title(title, fontweight="bold")

            out = _save_and_log(fig, out, self.logger, "plots/expert_analysis/expert_utilization/usage_violin", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_dead_expert_heatmap(self, data=None, step=None) -> Path:
        """Plot 2.3.3 – Dead Expert Detection (binary heatmap)."""
        key = "dead_expert_detection"
        out = PLOT_DIRS["expert_utilization"] / plot_filename("expert_analysis", "expert_utilization", key)
        try:
            n_ckpts = 10
            if data is None:
                np.random.seed(32)
                # Most experts active; a few occasionally die early then recover
                active = np.ones((N_EXPERTS, n_ckpts), dtype=int)
                active[2, [0, 1]] = 0
                active[5, [0]] = 0
                ckpt_labels = [f"step\n{(i+1)*500}" for i in range(n_ckpts)]
            else:
                active = np.asarray(data["active"])
                ckpt_labels = data.get("ckpt_labels", [str(i) for i in range(active.shape[1])])

            cmap = plt.cm.RdYlGn  # red=dead, green=active
            fig, ax = plt.subplots(figsize=(12, 5))
            im = ax.imshow(active, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(n_ckpts))
            ax.set_xticklabels(ckpt_labels, fontsize=8)
            ax.set_yticks(range(N_EXPERTS))
            ax.set_yticklabels([EXPERT_LABELS[e] for e in EXPERT_NAMES])
            ax.set_xlabel("Training Checkpoint")
            ax.set_ylabel("Expert")
            cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
            cbar.set_ticklabels(["Dead (≤1%)", "Active (>1%)"])
            title = "Dead Expert Detection"
            ax.set_title(title, fontweight="bold")

            out = _save_and_log(fig, out, self.logger, "plots/expert_analysis/expert_utilization/dead_expert_heatmap", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    # ==================================================================
    # 2.4 Spider / Radar Charts
    # ==================================================================

    def _draw_radar(
        self, ax, values: np.ndarray, color: str,
        fill_alpha: float = 0.2, lw: float = 2.0, label: str = "",
        ls: str = "-",
    ):
        """Draw one polygon on a polar ax."""
        N = len(values)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        vals   = np.concatenate([values, [values[0]]])
        angs   = np.concatenate([angles, [angles[0]]])
        ax.plot(angs, vals, color=color, lw=lw, ls=ls, label=label)
        ax.fill(angs, vals, color=color, alpha=fill_alpha)

    def plot_per_expert_spider_charts(self, data=None, step=None) -> Path:
        """Plot 2.4.1 – Per-Expert Domain Proficiency (8 spider charts)."""
        key = "per_expert_spider_charts"
        out = PLOT_DIRS["spider_charts"] / plot_filename("expert_analysis", "spider_charts", key)
        try:
            domains = EXPERT_NAMES  # axes = domain names = expert names
            N = len(domains)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

            if data is None:
                np.random.seed(40)
                perf = {}
                for i, name in enumerate(EXPERT_NAMES):
                    scores = np.random.uniform(0.1, 0.5, N)
                    scores[i] = np.random.uniform(0.7, 1.0)   # high on own domain
                    perf[name] = scores

                ideal = {}
                for i, name in enumerate(EXPERT_NAMES):
                    ideal_scores = np.full(N, 0.1)
                    ideal_scores[i] = 1.0
                    ideal[name] = ideal_scores
                data = {"perf": perf, "ideal": ideal}

            perf  = data["perf"]
            ideal = data.get("ideal", {})

            fig, axes = plt.subplots(2, 4, figsize=(20, 10),
                                     subplot_kw=dict(polar=True))
            axes = axes.flatten()
            for idx, (name, ax) in enumerate(zip(EXPERT_NAMES, axes)):
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_xticks(angles)
                ax.set_xticklabels([d.replace("_", "\n") for d in domains], fontsize=7)
                ax.set_yticks([0.25, 0.5, 0.75, 1.0])
                ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=6)
                ax.set_ylim(0, 1)

                scores = np.asarray(perf.get(name, np.full(N, 0.5)))
                self._draw_radar(ax, scores, EXPERT_COLORS[name], fill_alpha=0.25, label="Actual")
                if name in ideal:
                    self._draw_radar(ax, ideal[name], "black", fill_alpha=0,
                                     lw=1.0, ls=":", label="Ideal")
                ax.set_title(EXPERT_LABELS[name], fontsize=9, fontweight="bold", pad=10)

            title = "Per-Expert Domain Proficiency"
            fig.suptitle(title, fontsize=VIZ_CONFIG["font_title"], fontweight="bold", y=1.01)

            out = _save_and_log(fig, out, self.logger, "plots/expert_analysis/spider_charts/per_expert", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_comparative_spider(self, data=None, step=None) -> Path:
        """Plot 2.4.2 – All Experts Overlaid on Single Spider Chart."""
        key = "comparative_spider_chart"
        out = PLOT_DIRS["spider_charts"] / plot_filename("expert_analysis", "spider_charts", key)
        try:
            domains = EXPERT_NAMES
            N = len(domains)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

            if data is None:
                np.random.seed(41)
                perf = {}
                for i, name in enumerate(EXPERT_NAMES):
                    scores = np.random.uniform(0.1, 0.5, N)
                    scores[i] = np.random.uniform(0.75, 1.0)
                    perf[name] = scores
                data = {"perf": perf}

            perf = data["perf"]
            fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles)
            ax.set_xticklabels([d.replace("_", "\n") for d in domains], fontsize=8)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
            ax.set_ylim(0, 1)

            for name in EXPERT_NAMES:
                scores = np.asarray(perf.get(name, np.full(N, 0.5)))
                self._draw_radar(ax, scores, EXPERT_COLORS[name],
                                 fill_alpha=VIZ_CONFIG["alpha_overlay"], label=EXPERT_LABELS[name])

            ax.legend(bbox_to_anchor=(1.35, 1.1), fontsize=8)
            title = "Comparative Expert Spider Chart – All Experts"
            ax.set_title(title, fontsize=VIZ_CONFIG["font_title"], fontweight="bold", pad=20)

            out = _save_and_log(fig, out, self.logger, "plots/expert_analysis/spider_charts/comparative", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_spider_temporal(self, data=None, step=None) -> Path:
        """Plot 2.4.3 – Spider Chart Evolution (2×2 temporal grid)."""
        key = "spider_temporal_evolution"
        out = PLOT_DIRS["spider_charts"] / plot_filename("expert_analysis", "spider_charts", key)
        try:
            domains = EXPERT_NAMES
            N = len(domains)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            ckpt_labels = ["Initial (random)", "25% Training", "50% Training", "Final"]
            ckpt_keys   = ["init", "25pct", "50pct", "final"]

            if data is None:
                np.random.seed(42)
                data = {}
                for ci, ck in enumerate(ckpt_keys):
                    perf = {}
                    specialization = ci / (len(ckpt_keys) - 1)  # 0→1
                    for i, name in enumerate(EXPERT_NAMES):
                        scores = np.random.uniform(0.1, 1 - specialization * 0.5, N)
                        scores[i] = np.random.uniform(0.4 + specialization * 0.5, 1.0)
                        perf[name] = np.clip(scores, 0, 1)
                    data[ck] = perf

            fig, axes = plt.subplots(2, 2, figsize=VIZ_CONFIG["figsize_grid22"],
                                     subplot_kw=dict(polar=True))
            axes = axes.flatten()
            for ci, (ck, ax) in enumerate(zip(ckpt_keys, axes)):
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_xticks(angles)
                ax.set_xticklabels([d.replace("_", "\n") for d in domains], fontsize=6)
                ax.set_yticks([0.5, 1.0])
                ax.set_yticklabels(["0.5", "1.0"], fontsize=6)
                ax.set_ylim(0, 1)
                perf = data.get(ck, {})
                for name in EXPERT_NAMES:
                    scores = np.asarray(perf.get(name, np.full(N, 0.5)))
                    self._draw_radar(ax, scores, EXPERT_COLORS[name], fill_alpha=0.15, lw=1.2)
                ax.set_title(ckpt_labels[ci], fontsize=10, fontweight="bold", pad=8)

            title = "Spider Chart Temporal Evolution (Expert Specialization)"
            fig.suptitle(title, fontsize=VIZ_CONFIG["font_title"], fontweight="bold")

            out = _save_and_log(fig, out, self.logger, "plots/expert_analysis/spider_charts/temporal_evolution", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    # ------------------------------------------------------------------
    # 2.5  Research-grade: stacked-area usage and routing entropy
    # ------------------------------------------------------------------

    def plot_expert_usage_stacked_area(self, data=None, step=None) -> Path:
        """Plot: Expert Usage over Time – Stacked Area (MoE paper style)."""
        key = "expert_usage_stacked_area"
        out = PLOT_DIRS["expert_utilization"] / plot_filename("expert_analysis", "expert_utilization", key)
        try:
            if data is None:
                np.random.seed(30)
                T = 200
                steps = np.arange(T) * 10
                # Simulate experts starting uniform and specializing
                probs = np.zeros((N_EXPERTS, T))
                for i in range(N_EXPERTS):
                    target = 1.0 / N_EXPERTS + (np.random.rand() - 0.5) * 0.06
                    probs[i] = np.clip(
                        target + (np.random.rand() - 0.5) * 0.02,
                        0.05, 0.4,
                    )
                probs /= probs.sum(axis=0, keepdims=True)
                data = {"steps": steps, "expert_probs": probs}

            steps       = np.asarray(data["steps"])
            expert_probs = np.asarray(data["expert_probs"])  # (N_experts, T)
            mode_tag    = data.get("mode_tag", "")

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            ax.stackplot(
                steps,
                [expert_probs[i] * 100 for i in range(N_EXPERTS)],
                labels=[EXPERT_LABELS[e] for e in EXPERT_NAMES],
                colors=_COLORS,
                alpha=0.85,
            )
            ax.axhline(100.0 / N_EXPERTS, color="white", lw=1.0, ls="--",
                       alpha=0.6, label="uniform (1/8)")
            ax.set_xlabel("Training Step")
            ax.set_ylabel("% of Tokens Routed to Expert")
            ax.set_ylim(0, 100)
            title = "Expert Token Utilization Over Training (Stacked Area)"
            if mode_tag: title += f"  {mode_tag}"
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
            out = _save_and_log(fig, out, self.logger, "plots/expert_analysis/expert_utilization/stacked_area_usage", step)
            self._generated.append(out); return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_routing_entropy_over_time(self, data=None, step=None) -> Path:
        """Plot: Routing Entropy over Training (line with confidence band)."""
        key = "routing_entropy_over_time"
        out = PLOT_DIRS["routing_patterns"] / plot_filename("expert_analysis", "routing_patterns", key)
        try:
            if data is None:
                np.random.seed(31)
                T = 300
                steps = np.arange(T) * 10
                max_entropy = np.log(N_EXPERTS)
                # Starts near max (uniform/unconfident), drops to ~50% of max
                base = max_entropy * (0.95 - 0.45 * (1 - np.exp(-steps / 1500)))
                noise = np.random.normal(0, 0.02, T)
                entropy = np.clip(base + noise, 0.1, max_entropy)
                data = {"steps": steps, "entropy": entropy}

            steps    = np.asarray(data["steps"])
            entropy  = np.asarray(data["entropy"])
            mode_tag = data.get("mode_tag", "")
            max_ent  = np.log(N_EXPERTS)
            # Confidence band via sliding std
            window   = max(10, len(steps) // 20)
            roll_std = np.array([
                entropy[max(0, i - window // 2): i + window // 2].std() + 0.01
                for i in range(len(entropy))
            ])

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            ax.plot(steps, entropy, color="#e377c2", lw=VIZ_CONFIG["lw_main"],
                    label="Routing entropy")
            ax.fill_between(steps, np.clip(entropy - roll_std, 0, None),
                            entropy + roll_std, color="#e377c2", alpha=0.20)
            ax.axhline(max_ent, color="gray", lw=1.0, ls="--",
                       label=f"Max entropy (uniform, {max_ent:.2f})")
            ax.axhline(max_ent * 0.5, color="gray", lw=0.8, ls=":",
                       label=f"50% of max entropy")
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Routing Entropy (nats)")
            ax.set_ylim(0, max_ent * 1.1)
            title = "Expert Routing Entropy over Training"
            if mode_tag: title += f"  {mode_tag}"
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=9)
            out = _save_and_log(fig, out, self.logger, "plots/expert_analysis/routing_patterns/routing_entropy_over_time", step)
            self._generated.append(out); return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    # ------------------------------------------------------------------
    # Convenience: generate all
    # ------------------------------------------------------------------
    def generate_all(self, data=None, step=None) -> List[Path]:
        d = data or {}
        methods = [
            (self.plot_expert_selection_heatmap,   d.get("selection_freq")),
            (self.plot_cooccurrence_matrix,        d.get("cooccurrence")),
            (self.plot_routing_by_dataset,         d.get("routing_by_dataset")),
            (self.plot_routing_sankey_snapshot,    d.get("sankey")),
            (self.plot_specialization_index,       d.get("specialization_index")),
            (self.plot_weight_sparsity_grid,       d.get("weight_sparsity")),
            (self.plot_output_variance_boxplot,    d.get("output_variance")),
            (self.plot_load_balancing,             d.get("load_balancing")),
            (self.plot_usage_violin,               d.get("usage_violin")),
            (self.plot_dead_expert_heatmap,        d.get("dead_expert")),
            (self.plot_per_expert_spider_charts,   d.get("per_expert_spider")),
            (self.plot_comparative_spider,         d.get("comparative_spider")),
            (self.plot_spider_temporal,            d.get("spider_temporal")),
            # --- new research-grade plots ---
            (self.plot_expert_usage_stacked_area,  d.get("stacked_area")),
            (self.plot_routing_entropy_over_time,  d.get("routing_entropy")),
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
