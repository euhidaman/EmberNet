"""
Dataset Analysis Visualizations

Covers:
  5.1  Token Statistics      – per-dataset tokens, sequence length, cumulative exposure
  5.2  Domain Distributions  – pie chart, dataset mixing schedule, expert-dataset alignment
  5.3  Sample Visualizations – representative sample grid, failure case analysis
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
from visualizations.training_dynamics import _save_and_log
from visualizations.wandb_utils import WandBLogger

apply_mpl_style()


class DatasetAnalysisPlotter:
    """Generates all §5 dataset-analysis plots."""

    def __init__(self, logger: Optional[WandBLogger] = None):
        self.logger = logger or WandBLogger(disabled=True)
        self._generated: List[Path] = []

    # ==================================================================
    # 5.1 Token Statistics
    # ==================================================================

    def plot_token_distribution(self, data=None, step=None) -> Path:
        """Plot 5.1.1 – Token Distribution per Dataset."""
        key = "token_distribution_per_dataset"
        out = PLOT_DIRS["token_statistics"] / plot_filename("dataset_analysis", "token_statistics", key)
        try:
            n_ds = len(ALL_DATASETS)
            if data is None:
                np.random.seed(100)
                img_tok  = np.random.uniform(5.0, 60.0, n_ds)   # Millions
                text_tok = np.random.uniform(10.0, 120.0, n_ds)
                data = {"datasets": ALL_DATASETS, "img_tokens_M": img_tok, "text_tokens_M": text_tok}

            datasets   = data.get("datasets", ALL_DATASETS)
            img_tok    = np.asarray(data["img_tokens_M"])
            text_tok   = np.asarray(data["text_tokens_M"])
            x = np.arange(len(datasets))

            fig, ax = plt.subplots(figsize=(18, 6))
            ax.bar(x, img_tok,  color="#AED6F1", edgecolor="black", alpha=0.9, label="Image tokens")
            ax.bar(x, text_tok, bottom=img_tok, color="#A9DFBF", edgecolor="black", alpha=0.9, label="Text tokens")
            ax.set_xticks(x)
            ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Total Tokens (Millions)")
            title = "Token Distribution per Dataset"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            # Domain separators
            cum = 0
            for domain, ds_list in DATASET_DOMAINS.items():
                cum += len(ds_list)
                if cum < len(datasets):
                    ax.axvline(cum - 0.5, color="black", lw=1.5, ls="--", alpha=0.4)

            out = _save_and_log(fig, out, self.logger, "plots/dataset_analysis/token_statistics/token_distribution", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_sequence_length_violin(self, data=None, step=None) -> Path:
        """Plot 5.1.2 – Sequence Length Distribution (Violin)."""
        key = "sequence_length_violin"
        out = PLOT_DIRS["token_statistics"] / plot_filename("dataset_analysis", "token_statistics", key)
        try:
            if data is None:
                import pandas as pd
                np.random.seed(101)
                rows = []
                for ds in ALL_DATASETS:
                    domain = _dataset_domain(ds)
                    mean_len = np.random.randint(128, 1024)
                    lengths  = np.random.lognormal(np.log(mean_len), 0.4, 200).clip(32, 2048)
                    for l in lengths:
                        rows.append({"dataset": ds, "length": l, "domain": domain})
                data = {"df": pd.DataFrame(rows)}

            import pandas as pd
            df = data["df"]

            fig, ax = plt.subplots(figsize=(20, 6))
            domain_palette = {ds: DOMAIN_COLORS[_dataset_domain(ds)] for ds in ALL_DATASETS}
            sns.violinplot(
                data=df, x="dataset", y="length", ax=ax, hue="dataset",
                palette=domain_palette, cut=0, inner="box", density_norm="width",
                order=ALL_DATASETS, legend=False,
            )
            ax.set_xticks(range(len(ALL_DATASETS)))
            ax.set_xticklabels(ALL_DATASETS, rotation=45, ha="right", fontsize=7)
            ax.set_xlabel("Dataset")
            ax.set_ylabel("Sequence Length (tokens)")
            title = "Sequence Length Distribution per Dataset"
            ax.set_title(title, fontweight="bold")

            out = _save_and_log(fig, out, self.logger, "plots/dataset_analysis/token_statistics/seq_length_violin", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_cumulative_token_exposure(self, data=None, step=None) -> Path:
        """Plot 5.1.3 – Cumulative Token Exposure by Stage."""
        key = "cumulative_token_exposure"
        out = PLOT_DIRS["token_statistics"] / plot_filename("dataset_analysis", "token_statistics", key)
        try:
            if data is None:
                n1, n2 = 2000, 3000
                steps  = np.arange(n1 + n2)
                cum_s1 = np.cumsum(np.full(n1, 4096 * 4))   # tokens per step
                cum_s2 = cum_s1[-1] + np.cumsum(np.full(n2, 4096 * 4))
                data = {
                    "steps_s1": np.arange(n1),
                    "steps_s2": np.arange(n1, n1 + n2),
                    "cum_s1": cum_s1,
                    "cum_s2": cum_s2,
                    "epoch_boundaries": [500, 1000, 1500, 2500, 3500, 4500],
                }

            s1_steps = np.asarray(data["steps_s1"])
            s2_steps = np.asarray(data["steps_s2"])
            cum_s1   = np.asarray(data["cum_s1"]) / 1e9
            cum_s2   = np.asarray(data["cum_s2"]) / 1e9
            epoch_b  = data.get("epoch_boundaries", [])

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            ax.plot(s1_steps, cum_s1, color=STAGE_COLORS[1], lw=VIZ_CONFIG["lw_main"], label="Stage 1")
            ax.plot(s2_steps, cum_s2, color=STAGE_COLORS[2], lw=VIZ_CONFIG["lw_main"], label="Stage 2")
            for eb in epoch_b:
                ax.axvline(eb, color="gray", lw=0.8, ls=":", alpha=0.6)

            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Cumulative Tokens (Billions)")
            title = "Cumulative Token Exposure by Training Stage"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            out = _save_and_log(fig, out, self.logger, "plots/dataset_analysis/token_statistics/cumulative_tokens", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    # ==================================================================
    # 5.2 Domain Distributions
    # ==================================================================

    def plot_domain_pie(self, data=None, step=None) -> Path:
        """Plot 5.2.1 – Domain Distribution Pie Chart."""
        key = "domain_distribution_pie"
        out = PLOT_DIRS["domain_distributions"] / plot_filename("dataset_analysis", "domain_distributions", key)
        try:
            if data is None:
                domain_tok = {d: np.random.uniform(0.5, 3.0) for d in DATASET_DOMAINS}
                data = {"domain_tokens_B": domain_tok}

            domain_tok = data["domain_tokens_B"]
            labels  = [d.replace("_", "\n") for d in domain_tok]
            sizes   = list(domain_tok.values())
            colors  = [DOMAIN_COLORS.get(d, "#cccccc") for d in domain_tok]

            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, auto = ax.pie(
                sizes, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90,
                pctdistance=0.8,
                wedgeprops=dict(edgecolor="white", lw=1.5),
            )
            for t in texts:
                t.set_fontsize(9)
            title = "Domain Token Distribution"
            ax.set_title(title, fontsize=VIZ_CONFIG["font_title"], fontweight="bold", pad=15)

            out = _save_and_log(fig, out, self.logger, "plots/dataset_analysis/domain_distributions/domain_pie", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_dataset_mixing_schedule(self, data=None, step=None) -> Path:
        """Plot 5.2.2 – Dataset Mixing Schedule (Stage 2, stacked area)."""
        key = "dataset_mixing_schedule"
        out = PLOT_DIRS["domain_distributions"] / plot_filename("dataset_analysis", "domain_distributions", key)
        try:
            n_steps = 3000
            n_ds    = len(ALL_DATASETS)
            if data is None:
                np.random.seed(110)
                steps   = np.arange(n_steps)
                # Uniform start → curriculum shift toward harder datasets
                mix = np.zeros((n_ds, n_steps))
                for i in range(n_ds):
                    target = np.random.dirichlet(np.ones(n_ds))[i]
                    start  = 1.0 / n_ds
                    mix[i] = np.linspace(start, target, n_steps) + \
                             np.random.normal(0, 0.002, n_steps)
                # Normalize to sum = 1
                mix = np.abs(mix)
                mix /= mix.sum(axis=0, keepdims=True)
                mixing = {ds: mix[i] * 100 for i, ds in enumerate(ALL_DATASETS)}
                data = {"steps": steps, "mixing": mixing}

            steps  = np.asarray(data["steps"])
            mixing = data["mixing"]

            fig, ax = plt.subplots(figsize=(14, 7))
            # Group colors from domain
            bottom = np.zeros(len(steps))
            for ds in ALL_DATASETS:
                if ds in mixing:
                    domain = _dataset_domain(ds)
                    c = DOMAIN_COLORS.get(domain, "#aaaaaa")
                    ax.fill_between(steps, bottom, bottom + mixing[ds],
                                    color=c, alpha=0.85, label=ds)
                    bottom += mixing[ds]

            ax.set_xlim(0, steps[-1])
            ax.set_ylim(0, 100)
            ax.set_xlabel("Training Steps (Stage 2)")
            ax.set_ylabel("Batch Composition (%)")
            title = "Dataset Mixing Schedule (Curriculum Learning)"
            ax.set_title(title, fontweight="bold")
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, ncol=2)

            out = _save_and_log(fig, out, self.logger, "plots/dataset_analysis/domain_distributions/mixing_schedule", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_expert_dataset_alignment(self, data=None, step=None) -> Path:
        """Plot 5.2.3 – Expert-Dataset Alignment Matrix (designed vs learned)."""
        key = "expert_dataset_alignment_matrix"
        out = PLOT_DIRS["domain_distributions"] / plot_filename("dataset_analysis", "domain_distributions", key)
        try:
            n_experts = len(EXPERT_NAMES)
            n_ds      = len(ALL_DATASETS)
            if data is None:
                np.random.seed(111)
                # Designed alignment (block-diagonal based on domain mapping)
                designed = _build_designed_alignment()
                # Learned alignment: noisy version
                learned  = designed + np.random.normal(0, 0.05, designed.shape)
                learned  = np.clip(learned, 0, 1)
                data = {"designed": designed, "learned": learned}

            designed = np.asarray(data["designed"])
            learned  = np.asarray(data["learned"])

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
            kw = dict(
                xticklabels=ALL_DATASETS,
                yticklabels=[EXPERT_LABELS[e] for e in EXPERT_NAMES],
                cmap="Blues", vmin=0, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 6},
                linewidths=0.2,
            )
            sns.heatmap(designed, ax=ax1, cbar_kws={"label": "Alignment"}, **kw)
            ax1.set_title("Designed Alignment", fontweight="bold")
            ax1.set_xticklabels(ALL_DATASETS, rotation=45, ha="right", fontsize=6)

            sns.heatmap(learned, ax=ax2, cbar=True, cbar_kws={"label": "Routing Freq"}, **kw)
            ax2.set_title("Learned Alignment (Actual Routing)", fontweight="bold")
            ax2.set_xticklabels(ALL_DATASETS, rotation=45, ha="right", fontsize=6)

            title = "Expert-Dataset Alignment: Designed vs Learned"
            fig.suptitle(title, fontsize=VIZ_CONFIG["font_title"], fontweight="bold")

            out = _save_and_log(fig, out, self.logger, "plots/dataset_analysis/domain_distributions/expert_dataset_alignment", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    # ==================================================================
    # 5.3 Sample Visualizations
    # ==================================================================

    def plot_sample_grid(self, data=None, step=None) -> Path:
        """Plot 5.3.1 – Representative Samples Grid (placeholder tiles)."""
        key = "representative_samples_grid"
        out = PLOT_DIRS["sample_visualizations"] / plot_filename("dataset_analysis", "sample_visualizations", key)
        try:
            n_rows, n_cols = 4, 5
            n_samples = n_rows * n_cols  # exactly 20

            fig, axes = plt.subplots(n_rows, n_cols, figsize=VIZ_CONFIG["figsize_grid45"])
            for idx, ax in enumerate(axes.flatten()):
                if data is None or idx >= len(data.get("images", [])):
                    # Placeholder colored tile per dataset domain
                    ds = ALL_DATASETS[idx % len(ALL_DATASETS)]
                    domain = _dataset_domain(ds)
                    color  = DOMAIN_COLORS.get(domain, "#cccccc")
                    placeholder = np.full((64, 64, 3), int(color[1:3], 16))
                    ax.imshow(placeholder, vmin=0, vmax=255)
                    ax.set_title(ds, fontsize=7, fontweight="bold")
                    ax.set_xlabel("GT: [placeholder]\nPred: [placeholder]", fontsize=6)
                else:
                    img = data["images"][idx]
                    ax.imshow(img)
                    ax.set_title(ALL_DATASETS[idx], fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])

            title = "Representative Samples (one per dataset)"
            fig.suptitle(title, fontsize=VIZ_CONFIG["font_title"], fontweight="bold")

            out = _save_and_log(fig, out, self.logger, "plots/dataset_analysis/sample_visualizations/sample_grid", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_failure_cases(self, data=None, step=None) -> Path:
        """Plot 5.3.2 – Failure Case Analysis (2×4 grid)."""
        key = "failure_case_analysis"
        out = PLOT_DIRS["sample_visualizations"] / plot_filename("dataset_analysis", "sample_visualizations", key)
        try:
            failure_types = [
                "OCR error", "Spatial error", "Math error", "Diagram error",
                "Counting error", "Chart error", "Reasoning error", "Knowledge error",
            ]
            n_rows, n_cols = 2, 4

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
            for idx, (ax, ft) in enumerate(zip(axes.flatten(), failure_types)):
                placeholder = np.random.randint(180, 240, (64, 64, 3), dtype=np.uint8)
                ax.imshow(placeholder)
                ax.set_title(f"Failure: {ft}", fontsize=8, fontweight="bold", color="red")
                ax.set_xlabel("GT: [placeholder]\nPred: [wrong placeholder]", fontsize=6)
                ax.set_xticks([])
                ax.set_yticks([])

            title = "Failure Case Analysis"
            fig.suptitle(title, fontsize=VIZ_CONFIG["font_title"], fontweight="bold")

            out = _save_and_log(fig, out, self.logger, "plots/dataset_analysis/sample_visualizations/failure_cases", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def generate_all(self, data=None, step=None) -> List[Path]:
        d = data or {}
        methods = [
            (self.plot_token_distribution,          d.get("token_dist")),
            (self.plot_sequence_length_violin,      d.get("seq_len")),
            (self.plot_cumulative_token_exposure,   d.get("cum_tokens")),
            (self.plot_domain_pie,                  d.get("domain_pie")),
            (self.plot_dataset_mixing_schedule,     d.get("mixing")),
            (self.plot_expert_dataset_alignment,    d.get("alignment")),
            (self.plot_sample_grid,                 d.get("samples")),
            (self.plot_failure_cases,               d.get("failures")),
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataset_domain(ds: str) -> str:
    for domain, ds_list in DATASET_DOMAINS.items():
        if ds in ds_list:
            return domain
    return "alignment"


def _build_designed_alignment():
    """Build a block-diagonal expert-dataset alignment matrix."""
    n_experts = len(EXPERT_NAMES)
    n_ds      = len(ALL_DATASETS)
    mat = np.zeros((n_experts, n_ds))

    # Map experts to datasets by domain
    expert_domain_map = {
        "vision_ocr":          "vision_ocr",
        "vision_diagram":      "vision_ocr",      # diagrams close to OCR
        "code_math_chart":     "code_math_chart",
        "code_math_formula":   "code_math_chart",
        "spatial_scene":       "spatial_scene",
        "spatial_reasoning":   "spatial_scene",
        "agentic_knowledge":   "agentic_reasoning",
        "agentic_reasoning":   "agentic_reasoning",
    }

    for ei, expert_name in enumerate(EXPERT_NAMES):
        target_domain = expert_domain_map.get(expert_name, "alignment")
        domain_datasets = DATASET_DOMAINS.get(target_domain, [])
        for di, ds in enumerate(ALL_DATASETS):
            if ds in domain_datasets:
                mat[ei, di] = 0.8 + np.random.uniform(0, 0.2)
            else:
                mat[ei, di] = 0.0 + np.random.uniform(0, 0.1)
    return mat
