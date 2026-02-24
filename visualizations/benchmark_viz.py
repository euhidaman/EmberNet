"""
visualizations/benchmark_viz.py
================================
Publication-quality benchmark visualizations for EmberNet.

Generated automatically after lmms-eval completes at the end of training.
All plots are saved to {output_dir}/plots/benchmark_results/.

Plots:
  4.1  Expert Domain Spider Chart  — radar of 8 domain scores + baseline overlay
  4.2  Per-Task Score Bar Chart    — horizontal bars, color-coded by expert domain
  4.3  Domain Heatmap              — expert domain × metric matrix
  4.4  Score Distribution Summary  — box + strip plot across all benchmarks
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

from visualizations.config import (
    VIZ_CONFIG, PLOT_DIRS, EXPERT_NAMES, EXPERT_COLORS, EXPERT_LABELS,
    apply_mpl_style, log_plot_error,
)
from visualizations.wandb_utils import WandBLogger

apply_mpl_style()

# ---------------------------------------------------------------------------
# Task → expert domain mapping
# ---------------------------------------------------------------------------
TASK_TO_EXPERT = {
    "textvqa":                "vision_ocr",
    "docvqa_val":             "vision_ocr",
    "ocrvqa":                 "vision_ocr",
    "ocrbench":               "vision_ocr",
    "ai2d":                   "vision_diagram",
    "chartqa":                "code_math_chart",
    "charxiv_val_reasoning":  "code_math_chart",
    "charxiv_val_descriptive":"vision_diagram",
    "mathvista":              "code_math_formula",
    "vqav2":                  "spatial_scene",
    "gqa":                    "spatial_reasoning",
    "ok_vqa":                 "agentic_knowledge",
    "scienceqa_img":          "agentic_reasoning",
    "clevr":                  "agentic_reasoning",
    "pope":                   "general",
    "hallusion_bench_image":  "general",
    "mme":                    "general",
    "mmmu_val":               "general",
    "mmstar":                 "general",
    "seed_bench":             "general",
}

TASK_DISPLAY = {
    "textvqa":                "TextVQA",
    "docvqa_val":             "DocVQA",
    "ocrvqa":                 "OCR-VQA",
    "ocrbench":               "OCRBench",
    "ai2d":                   "AI2D",
    "chartqa":                "ChartQA",
    "charxiv_val_reasoning":  "CharXiv-Reasoning",
    "charxiv_val_descriptive":"CharXiv-Descriptive",
    "mathvista":              "MathVista",
    "vqav2":                  "VQAv2",
    "gqa":                    "GQA",
    "ok_vqa":                 "OK-VQA",
    "scienceqa_img":          "ScienceQA",
    "pope":                   "POPE",
    "hallusion_bench_image":  "HallusionBench",
    "mme":                    "MME",
    "mmmu_val":               "MMMU",
    "mmstar":                 "MMStar",
    "seed_bench":             "SEED-Bench",
}

GENERAL_COLOR = "#bcbd22"

# Domain → display label for spider axes
DOMAIN_LABELS = {
    "vision_ocr":          "Vision\nOCR",
    "vision_diagram":      "Vision\nDiagram",
    "code_math_chart":     "Code/Math\nChart",
    "code_math_formula":   "Code/Math\nFormula",
    "spatial_scene":       "Spatial\nScene",
    "spatial_reasoning":   "Spatial\nReas.",
    "agentic_knowledge":   "Agentic\nKnow.",
    "agentic_reasoning":   "Agentic\nReas.",
}

# Approximate published baselines for context (accuracy %, 0-100)
# Sourced from public leaderboards for comparable ~700M-1B models
BASELINES = {
    "InstructBLIP-7B": {
        "vision_ocr":        50.1,
        "vision_diagram":    40.7,
        "code_math_chart":   33.9,
        "code_math_formula": 25.3,
        "spatial_scene":     60.9,
        "spatial_reasoning": 49.2,
        "agentic_knowledge": 60.5,
        "agentic_reasoning": 60.5,
    },
    "LLaVA-1.5-7B": {
        "vision_ocr":        58.2,
        "vision_diagram":    55.5,
        "code_math_chart":   18.2,
        "code_math_formula": 26.4,
        "spatial_scene":     78.5,
        "spatial_reasoning": 62.0,
        "agentic_knowledge": 54.4,
        "agentic_reasoning": 66.8,
    },
}

# Baseline line styles
BASELINE_STYLES = [
    {"color": "#aaaaaa", "ls": "--",  "lw": 1.2, "alpha": 0.6},
    {"color": "#888888", "ls": "-.",  "lw": 1.2, "alpha": 0.6},
]


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def _save(fig: plt.Figure, path: Path, logger: WandBLogger, wandb_key: str, step=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    if logger:
        try:
            import wandb
            logger.log({wandb_key: wandb.Image(str(path))}, step=step)
        except Exception:
            pass


class BenchmarkVisualizer:
    """
    Generates all §4 benchmark result plots from a scores dict.

    scores dict format:
        {
            "task_name": float,   # 0-100 accuracy/score
            ...
        }
    e.g. {"textvqa": 52.3, "chartqa": 44.1, "mme": 61.8, ...}
    """

    def __init__(
        self,
        logger: Optional[WandBLogger] = None,
        plots_root: Optional[Path] = None,
    ):
        self.logger = logger or WandBLogger(disabled=True)
        # Use configured PLOT_DIRS, but allow override
        self._root = plots_root or PLOT_DIRS.get("benchmark_spider", Path("plots/benchmark_results"))
        self._generated: List[Path] = []

    def _dir(self, sub: str) -> Path:
        d = PLOT_DIRS.get(f"benchmark_{sub}", self._root / sub)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # 4.1  Expert Domain Spider / Radar Chart
    # ------------------------------------------------------------------

    def plot_expert_spider(
        self,
        scores: Dict[str, float],
        mode: str = "main",
        step: Optional[int] = None,
    ) -> Path:
        """
        Radar chart with 8 expert domain axes.

        EmberNet's score for each domain is the mean of its corresponding
        benchmark task(s).  Baseline models drawn as dashed overlays.
        """
        out = self._dir("spider") / f"benchmark_expert_spider_{mode}_{_ts()}.png"
        try:
            domain_scores = _aggregate_domain_scores(scores)
            domains = list(DOMAIN_LABELS.keys())
            values  = [domain_scores.get(d, 0.0) for d in domains]
            labels  = [DOMAIN_LABELS[d] for d in domains]
            N       = len(domains)

            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]   # close the polygon
            vals_plot = values + values[:1]

            fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

            # ---- concentric grid lines ----
            for ring in [20, 40, 60, 80, 100]:
                ax.plot(angles, [ring] * (N + 1), color="gray", lw=0.5, alpha=0.4)
                ax.text(0, ring + 2, f"{ring}%", ha="center", va="bottom",
                        fontsize=7, color="gray")

            # ---- baselines ----
            for (bname, bscores), bstyle in zip(BASELINES.items(), BASELINE_STYLES):
                bvals = [bscores.get(d, 0.0) for d in domains] + [bscores.get(domains[0], 0.0)]
                ax.plot(angles, bvals,
                        color=bstyle["color"], ls=bstyle["ls"],
                        lw=bstyle["lw"], alpha=bstyle["alpha"],
                        label=bname)
                ax.fill(angles, bvals, color=bstyle["color"], alpha=0.04)

            # ---- EmberNet ----
            ember_color = "#ff4444" if mode == "trial" else "#e03030"
            ax.plot(angles, vals_plot, color=ember_color, lw=2.5,
                    label=f"EmberNet ({mode.capitalize()})")
            ax.fill(angles, vals_plot, color=ember_color, alpha=0.18)

            # ---- score labels at each vertex ----
            for angle, val, lbl in zip(angles[:-1], values, labels):
                ha = "left" if np.cos(angle) > 0.1 else ("right" if np.cos(angle) < -0.1 else "center")
                va = "bottom" if np.sin(angle) > 0.1 else ("top" if np.sin(angle) < -0.1 else "center")
                offset = 11
                ax.text(angle, val + offset, f"{val:.1f}%",
                        ha=ha, va=va, fontsize=9, fontweight="bold",
                        color=ember_color)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=10)
            ax.set_ylim(0, 105)
            ax.set_yticks([])
            ax.spines["polar"].set_visible(False)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            title = f"EmberNet — Expert Domain Benchmark Spider Chart ({mode.capitalize()} Run)"
            avg = np.mean(values)
            ax.set_title(title + f"\nMean accuracy: {avg:.1f}%", fontweight="bold",
                         pad=20, fontsize=13)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=9)

            _save(fig, out, self.logger,
                  f"benchmark/spider_chart_{mode}", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error("expert_spider", e); plt.close("all"); return out

    # ------------------------------------------------------------------
    # 4.2  Per-Task Horizontal Bar Chart
    # ------------------------------------------------------------------

    def plot_task_bars(
        self,
        scores: Dict[str, float],
        mode: str = "main",
        step: Optional[int] = None,
    ) -> Path:
        """Horizontal bar chart of every benchmark result, color-coded by domain."""
        out = self._dir("task_scores") / f"benchmark_task_bars_{mode}_{_ts()}.png"
        try:
            tasks  = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
            values = [scores[t] for t in tasks]
            labels = [TASK_DISPLAY.get(t, t) for t in tasks]
            colors = [
                EXPERT_COLORS.get(TASK_TO_EXPERT.get(t, "general"), GENERAL_COLOR)
                for t in tasks
            ]

            fig, ax = plt.subplots(figsize=(10, max(4, len(tasks) * 0.65)))
            y = np.arange(len(tasks))
            bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.5)

            # value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", ha="left", fontsize=9)

            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=10)
            ax.set_xlabel("Score (%)", fontsize=11)
            ax.set_xlim(0, max(max(values) * 1.18, 10) if values else 100)
            ax.axvline(sum(values) / len(values) if values else 0,
                       color="black", ls="--", lw=1.2, label="Mean")
            ax.legend(fontsize=9)

            # domain legend patches
            seen = set()
            patches = []
            for t in tasks:
                d = TASK_TO_EXPERT.get(t, "general")
                if d not in seen:
                    seen.add(d)
                    c = EXPERT_COLORS.get(d, GENERAL_COLOR)
                    patches.append(mpatches.Patch(color=c, label=EXPERT_LABELS.get(d, d)))
            ax.legend(handles=patches, loc="lower right", fontsize=8,
                      title="Expert Domain", title_fontsize=8)

            ax.set_title(
                f"EmberNet ({mode.capitalize()}) — Per-Task Benchmark Scores",
                fontweight="bold", fontsize=13,
            )
            fig.tight_layout()

            _save(fig, out, self.logger, f"benchmark/task_bars_{mode}", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error("task_bars", e); plt.close("all"); return out

    # ------------------------------------------------------------------
    # 4.3  Domain Heatmap + Baseline Delta
    # ------------------------------------------------------------------

    def plot_domain_heatmap(
        self,
        scores: Dict[str, float],
        mode: str = "main",
        step: Optional[int] = None,
    ) -> Path:
        """2-row heatmap: EmberNet scores and delta vs LLaVA-1.5-7B baseline."""
        out = self._dir("domain_analysis") / f"benchmark_domain_heatmap_{mode}_{_ts()}.png"
        try:
            import seaborn as sns

            domain_scores   = _aggregate_domain_scores(scores)
            domains         = list(DOMAIN_LABELS.keys())
            ember_row       = np.array([domain_scores.get(d, 0.0) for d in domains])
            baseline_row    = np.array([BASELINES["LLaVA-1.5-7B"].get(d, 0.0) for d in domains])
            delta_row       = ember_row - baseline_row

            matrix = np.vstack([ember_row, baseline_row, delta_row])
            row_labels = ["EmberNet", "LLaVA-1.5-7B", "Δ (EmberNet − LLaVA)"]
            col_labels = [DOMAIN_LABELS[d].replace("\n", " ") for d in domains]

            fig, axes = plt.subplots(1, 2, figsize=(16, 3.5),
                                     gridspec_kw={"width_ratios": [2, 1]})

            # left: absolute scores for EmberNet + baseline
            sns.heatmap(
                matrix[:2], ax=axes[0],
                xticklabels=col_labels, yticklabels=row_labels[:2],
                cmap="YlOrRd", vmin=0, vmax=100,
                annot=True, fmt=".1f",
                cbar_kws={"label": "Score (%)"},
            )
            axes[0].set_title("Domain Scores (%)", fontweight="bold")

            # right: delta (green=better, red=worse)
            sns.heatmap(
                delta_row.reshape(1, -1), ax=axes[1],
                xticklabels=col_labels, yticklabels=["Δ"],
                cmap="RdYlGn", center=0, vmin=-30, vmax=30,
                annot=True, fmt="+.1f",
                cbar_kws={"label": "Delta (pp)"},
            )
            axes[1].set_title("vs LLaVA-1.5-7B", fontweight="bold")

            fig.suptitle(
                f"EmberNet ({mode.capitalize()}) — Domain Score Heatmap",
                fontweight="bold", fontsize=13, y=1.02,
            )
            fig.tight_layout()

            _save(fig, out, self.logger, f"benchmark/domain_heatmap_{mode}", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error("domain_heatmap", e); plt.close("all"); return out

    # ------------------------------------------------------------------
    # 4.4  Summary Dashboard (2×2 grid combining spider + bars + heatmap)
    # ------------------------------------------------------------------

    def plot_summary_dashboard(
        self,
        scores: Dict[str, float],
        mode: str = "main",
        training_loss: float = 0.0,
        step: Optional[int] = None,
    ) -> Path:
        """Single-figure dashboard with spider, bars, domain chart, and meta stats."""
        out = self._dir("dashboard") / f"benchmark_dashboard_{mode}_{_ts()}.png"
        try:
            domain_scores = _aggregate_domain_scores(scores)
            domains       = list(DOMAIN_LABELS.keys())
            values        = [domain_scores.get(d, 0.0) for d in domains]
            N             = len(domains)
            angles        = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles       += angles[:1]
            vals_plot     = values + values[:1]

            fig = plt.figure(figsize=(20, 14))
            gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.45)

            # ---- [0,0] Spider ----
            ax_spider = fig.add_subplot(gs[0, 0], polar=True)
            for ring in [25, 50, 75, 100]:
                ax_spider.plot(angles, [ring] * (N + 1), color="gray", lw=0.4, alpha=0.3)
            for (bname, bscores), bstyle in zip(BASELINES.items(), BASELINE_STYLES):
                bvals = [bscores.get(d, 0.0) for d in domains] + [bscores.get(domains[0], 0.0)]
                ax_spider.plot(angles, bvals, color=bstyle["color"],
                               ls=bstyle["ls"], lw=1.0, alpha=bstyle["alpha"], label=bname)
            ember_c = "#e03030"
            ax_spider.plot(angles, vals_plot, color=ember_c, lw=2.5,
                           label=f"EmberNet ({mode})")
            ax_spider.fill(angles, vals_plot, color=ember_c, alpha=0.18)
            ax_spider.set_xticks(angles[:-1])
            ax_spider.set_xticklabels([DOMAIN_LABELS[d] for d in domains], size=8)
            ax_spider.set_ylim(0, 110)
            ax_spider.set_yticks([])
            ax_spider.spines["polar"].set_visible(False)
            ax_spider.set_theta_offset(np.pi / 2)
            ax_spider.set_theta_direction(-1)
            ax_spider.set_title("Expert Domain Spider", fontweight="bold", pad=20, fontsize=11)
            ax_spider.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.4, 1.1))

            # ---- [0,1] Task bars ----
            ax_bars = fig.add_subplot(gs[0, 1:])
            tasks_sorted = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
            bar_colors   = [
                EXPERT_COLORS.get(TASK_TO_EXPERT.get(t, "general"), GENERAL_COLOR)
                for t in tasks_sorted
            ]
            ypos  = np.arange(len(tasks_sorted))
            hbars = ax_bars.barh(ypos, [scores[t] for t in tasks_sorted],
                                 color=bar_colors, edgecolor="white", lw=0.4)
            for bar, t in zip(hbars, tasks_sorted):
                ax_bars.text(bar.get_width() + 0.5,
                             bar.get_y() + bar.get_height() / 2,
                             f"{scores[t]:.1f}%", va="center", ha="left", fontsize=8)
            ax_bars.set_yticks(ypos)
            ax_bars.set_yticklabels([TASK_DISPLAY.get(t, t) for t in tasks_sorted], fontsize=9)
            ax_bars.set_xlabel("Score (%)")
            mean_score = np.mean(list(scores.values())) if scores else 0
            ax_bars.axvline(mean_score, color="black", ls="--", lw=1.2)
            ax_bars.set_title("Per-Task Scores", fontweight="bold", fontsize=11)

            # ---- [1,0] Domain bar (vs baseline) ----
            ax_dom = fig.add_subplot(gs[1, 0])
            ember_vals  = [domain_scores.get(d, 0.0) for d in domains]
            llava_vals  = [BASELINES["LLaVA-1.5-7B"].get(d, 0.0) for d in domains]
            x = np.arange(N)
            w = 0.35
            ax_dom.bar(x - w/2, ember_vals, w, label="EmberNet", color="#e03030", alpha=0.85)
            ax_dom.bar(x + w/2, llava_vals, w, label="LLaVA-1.5-7B", color="#888888", alpha=0.6)
            ax_dom.set_xticks(x)
            ax_dom.set_xticklabels(
                [DOMAIN_LABELS[d].replace("\n", " ") for d in domains],
                rotation=35, ha="right", fontsize=7,
            )
            ax_dom.set_ylabel("Score (%)")
            ax_dom.legend(fontsize=8)
            ax_dom.set_title("Domain vs Baseline", fontweight="bold", fontsize=11)

            # ---- [1,1] Delta heatmap ----
            ax_heat = fig.add_subplot(gs[1, 1])
            try:
                import seaborn as sns
                delta = np.array(ember_vals) - np.array(llava_vals)
                sns.heatmap(
                    delta.reshape(1, -1), ax=ax_heat,
                    xticklabels=[DOMAIN_LABELS[d].replace("\n", " ") for d in domains],
                    yticklabels=["Δ"],
                    cmap="RdYlGn", center=0, vmin=-30, vmax=30,
                    annot=True, fmt="+.1f", annot_kws={"fontsize": 8},
                    cbar_kws={"label": "pp vs LLaVA"},
                )
                ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=35, ha="right", fontsize=7)
                ax_heat.set_title("Δ vs LLaVA-1.5-7B", fontweight="bold", fontsize=11)
            except ImportError:
                ax_heat.text(0.5, 0.5, "seaborn required", ha="center", va="center")

            # ---- [1,2] Summary stats panel ----
            ax_stats = fig.add_subplot(gs[1, 2])
            ax_stats.axis("off")
            stats_lines = [
                f"Mode:             {mode.capitalize()}",
                f"Tasks evaluated:  {len(scores)}",
                f"Mean score:       {mean_score:.1f}%",
                f"Best task:        {max(scores, key=scores.get) if scores else '-'} ({max(scores.values()) if scores else 0:.1f}%)",
                f"Weakest task:     {min(scores, key=scores.get) if scores else '-'} ({min(scores.values()) if scores else 0:.1f}%)",
                f"Training loss:    {training_loss:.4f}" if training_loss > 0 else "",
                "",
                "Best Expert Domains:",
            ]
            if domain_scores:
                sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
                for rank, (d, s) in enumerate(sorted_domains[:3], 1):
                    stats_lines.append(f"  #{rank}  {DOMAIN_LABELS[d].replace(chr(10), ' '):<20} {s:.1f}%")

            text = "\n".join(stats_lines)
            ax_stats.text(0.05, 0.95, text, transform=ax_stats.transAxes,
                          va="top", fontsize=9, family="monospace",
                          bbox=dict(boxstyle="round", facecolor="#f8f8f8", alpha=0.6))
            ax_stats.set_title("Summary", fontweight="bold", fontsize=11)

            fig.suptitle(
                f"EmberNet — Benchmark Dashboard  ({mode.capitalize()} Run)",
                fontweight="bold", fontsize=15, y=1.01,
            )

            _save(fig, out, self.logger, f"benchmark/dashboard_{mode}", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error("benchmark_dashboard", e); plt.close("all"); return out

    # ------------------------------------------------------------------
    # Convenience: generate all 4 plots at once
    # ------------------------------------------------------------------

    def plot_all(
        self,
        scores: Dict[str, float],
        mode: str = "main",
        training_loss: float = 0.0,
        step: Optional[int] = None,
    ) -> List[Path]:
        generated = []
        for fn, kwargs in [
            (self.plot_expert_spider,    dict(scores=scores, mode=mode, step=step)),
            (self.plot_task_bars,        dict(scores=scores, mode=mode, step=step)),
            (self.plot_domain_heatmap,   dict(scores=scores, mode=mode, step=step)),
            (self.plot_summary_dashboard,dict(scores=scores, mode=mode,
                                              training_loss=training_loss, step=step)),
        ]:
            try:
                p = fn(**kwargs)
                generated.append(p)
            except Exception as e:
                log_plot_error(fn.__name__, e)
        return generated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aggregate_domain_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Average per-task scores into per-domain scores (0-100 scale)."""
    domain_totals: Dict[str, List[float]] = {}
    for task, score in scores.items():
        domain = TASK_TO_EXPERT.get(task, "general")
        if domain == "general":
            continue
        domain_totals.setdefault(domain, []).append(float(score))
    return {d: float(np.mean(vals)) for d, vals in domain_totals.items()}


# Task-specific primary metric keys (checked before METRIC_PRIORITY).
# These are the canonical lmms-eval JSON result keys for tasks whose
# metric name doesn't match the generic priority list.
_TASK_METRIC_OVERRIDE = {
    "pope":                    "pope_f1_score,none",       # F1 best captures hallucination tradeoff
    "ocrbench":                "ocrbench_accuracy,none",
    "hallusion_bench_image":   "aAcc,none",                # answer-level accuracy
    "mmstar":                  "average,none",
    "charxiv_val_reasoning":   "reasoning_acc,none",
    "charxiv_val_descriptive": "descriptive_acc,none",
}


def extract_scores_from_lmms_results(results_dict: dict) -> Dict[str, float]:
    """
    Parse lmms-eval results dict into a flat {task: score_0_to_100} mapping.

    Handles the different metric names across benchmarks.
    """
    # Generic priority order (used when no task-specific override exists)
    METRIC_PRIORITY = [
        "accuracy,none",
        "exact_match,none",
        "relaxed_accuracy,none",
        "overall_accuracy,none",
        "acc,none",
        "score,none",
    ]
    # Special normalization for MME (raw perception+cognition out of ~2480)
    MME_MAX = 2480.0

    task_results = results_dict.get("results", {})
    scores: Dict[str, float] = {}

    def _coerce(v) -> Optional[float]:
        """float for numeric, 0.0 for None, None for non-numeric (list/str/etc)."""
        if isinstance(v, (int, float)):
            return float(v)
        if v is None:
            return 0.0
        return None

    for task_name, metrics in task_results.items():
        if not isinstance(metrics, dict):
            continue
        # normalize task name (strip dataset sub-splits like "textvqa_val")
        base = task_name.split(",")[0].lower().replace("-", "_")

        raw = None

        # task-specific metric override (checked first)
        if base in _TASK_METRIC_OVERRIDE:
            override_key = _TASK_METRIC_OVERRIDE[base]
            raw = _coerce(metrics.get(override_key))

        # special case: MME returns perception + cognition
        elif "mme" in base:
            perception = _coerce(metrics.get("mme_perception_score,none")) or 0.0
            cognition  = _coerce(metrics.get("mme_cognition_score,none"))  or 0.0
            total      = perception + cognition
            raw = min(total / MME_MAX * 100.0, 100.0)  # 0.0 if total==0
        else:
            for mkey in METRIC_PRIORITY:
                if mkey in metrics:
                    r = _coerce(metrics[mkey])
                    if r is not None:  # skip list/string values; try next key
                        raw = r
                        break
            # If still None, scan for first coercible key
            if raw is None:
                for k, v in metrics.items():
                    if k.endswith("_stderr") or k.endswith(",stderr"):
                        continue
                    r = _coerce(v)
                    if r is not None:
                        raw = r
                        break

        if raw is not None:
            # Convert [0,1] → [0,100] if expressed as fraction
            val = float(raw)
            if 0.0 <= val <= 1.0:
                val *= 100.0
            scores[base] = round(val, 2)

    return scores


def save_scores_json(scores: Dict[str, float], output_dir: Path, mode: str) -> Path:
    """Persist the normalized scores dict alongside the plots."""
    out = output_dir / f"benchmark_scores_{mode}_{_ts()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"mode": mode, "scores": scores,
                   "timestamp": datetime.now().isoformat()}, f, indent=2)
    return out
