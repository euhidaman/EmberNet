"""
Publication-quality visualizations for the Robot Risk Benchmarking Suite.

Generates:
  1. overview_metrics.{png,pdf}         — grouped bar chart across benchmarks
  2. veri_confusion.{png,pdf}           — 2x2 confusion matrix for VERI-Emergency
  3. veri_category_rates.{png,pdf}      — false alarm / missed-emergency per category
  4. robot_selection_heatmap.{png,pdf}  — scenario-type vs robot heatmap
  5. geobench_tasks.{png,pdf}           — per-task risk accuracy for GEOBench
  6. severity_distributions.{png,pdf}   — risk severity histograms

Invoked from eval/robot_risk_benchmarks.py automatically after evaluation,
or standalone via generate_all_plots.py --fig fig_robot_risk_benchmarks.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Try to import EmberNet viz config; fall back to sensible defaults
try:
    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from visualizations.config import VIZ_CONFIG, apply_mpl_style
    _HAS_CONFIG = True
except ImportError:
    _HAS_CONFIG = False

# ── Palette ──────────────────────────────────────────────────────────────────
C_SAFE    = "#2ca02c"   # green
C_DANGER  = "#d62728"   # red
C_NEUTRAL = "#1f77b4"   # blue
C_WARN    = "#ff7f0e"   # orange
C_GRAY    = "#7f7f7f"

BENCHMARK_COLORS = {
    "robot_selection": "#0d7377",
    "geobench_vlm":    "#c45b28",
    "veri_emergency":  "#6a3d9a",
}

DPI = 300


def _apply_style():
    if _HAS_CONFIG:
        apply_mpl_style()
    else:
        plt.rcParams.update({
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
        })


def _save(fig, path_stem: Path):
    for ext in (".png", ".pdf"):
        fig.savefig(str(path_stem) + ext, dpi=DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)


def _load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _load_metrics(results_dir: Path) -> Optional[dict]:
    for name in ("metrics_summary_main.json", "metrics_summary_trial.json"):
        p = results_dir / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


# ── 1. Overview metrics bar chart ────────────────────────────────────────────

def plot_overview(metrics: dict, save_dir: Path):
    _apply_style()
    benchmarks = metrics.get("benchmarks", {})
    if not benchmarks:
        return

    names = []
    acc_vals, far_vals, mer_vals = [], [], []
    colors = []
    for key in ("robot_selection", "geobench_vlm", "veri_emergency"):
        if key not in benchmarks:
            continue
        m = benchmarks[key]
        label = key.replace("_", " ").title()
        names.append(label)
        acc_vals.append(m.get("risk_accuracy", 0))
        far_vals.append(m.get("false_alarm_rate", 0))
        mer_vals.append(m.get("missed_emergency_rate", 0))
        colors.append(BENCHMARK_COLORS.get(key, C_NEUTRAL))

    if not names:
        return

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#f5f5f5")

    ax.bar(x - width, acc_vals, width, label="Risk Accuracy", color=C_NEUTRAL, alpha=0.85)
    ax.bar(x, far_vals, width, label="False Alarm Rate", color=C_WARN, alpha=0.85)
    ax.bar(x + width, mer_vals, width, label="Missed Emergency Rate", color=C_DANGER, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Rate")
    ax.set_title("Robot Risk Benchmark — Overview Metrics", fontsize=13, fontweight="semibold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linewidth=0.4, alpha=0.3, color="#cccccc")

    _save(fig, save_dir / "overview_metrics")


# ── 2. VERI confusion matrix ────────────────────────────────────────────────

def plot_veri_confusion(results_dir: Path, save_dir: Path):
    _apply_style()
    rows = _load_jsonl(results_dir / "veri_emergency_results_main.jsonl")
    if not rows:
        rows = _load_jsonl(results_dir / "veri_emergency_results_trial.jsonl")
    if not rows:
        return

    valid = [r for r in rows if r.get("risk_label_pred") in ("safe", "danger")]
    if not valid:
        return

    labels = ["safe", "danger"]
    mat = np.zeros((2, 2), dtype=int)
    for r in valid:
        gt_idx = labels.index(r["risk_label_gt"]) if r["risk_label_gt"] in labels else -1
        pr_idx = labels.index(r["risk_label_pred"]) if r["risk_label_pred"] in labels else -1
        if gt_idx >= 0 and pr_idx >= 0:
            mat[gt_idx, pr_idx] += 1

    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor("#fafafa")

    cmap = mcolors.LinearSegmentedColormap.from_list("rg", [C_SAFE, "#f5f5f5", C_DANGER])
    im = ax.imshow(mat, cmap=cmap, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Safe", "Pred: Danger"])
    ax.set_yticklabels(["GT: Safe", "GT: Danger"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if mat[i, j] > mat.max() * 0.6 else "black")

    ax.set_title("VERI-Emergency Confusion Matrix", fontsize=12, fontweight="semibold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    _save(fig, save_dir / "veri_confusion")


# ── 3. VERI per-category false alarm / missed emergency ──────────────────────

def plot_veri_categories(results_dir: Path, save_dir: Path):
    _apply_style()
    rows = _load_jsonl(results_dir / "veri_emergency_results_main.jsonl")
    if not rows:
        rows = _load_jsonl(results_dir / "veri_emergency_results_trial.jsonl")
    if not rows:
        return

    cats = defaultdict(lambda: {"fp": 0, "fn": 0, "safe_total": 0, "danger_total": 0})
    for r in rows:
        cat = r.get("category", "unknown")
        gt = r.get("risk_label_gt", "")
        pred = r.get("risk_label_pred", "")
        if gt == "safe":
            cats[cat]["safe_total"] += 1
            if pred == "danger":
                cats[cat]["fp"] += 1
        elif gt == "danger":
            cats[cat]["danger_total"] += 1
            if pred == "safe":
                cats[cat]["fn"] += 1

    cat_names = sorted(cats.keys())
    if not cat_names:
        return

    far = [cats[c]["fp"] / max(cats[c]["safe_total"], 1) for c in cat_names]
    mer = [cats[c]["fn"] / max(cats[c]["danger_total"], 1) for c in cat_names]

    x = np.arange(len(cat_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#f5f5f5")

    ax.bar(x - width / 2, far, width, label="False Alarm Rate", color=C_WARN, alpha=0.85)
    ax.bar(x + width / 2, mer, width, label="Missed Emergency Rate", color=C_DANGER, alpha=0.85)

    ax.set_xticks(x)
    cat_labels = {"AB": "Accidents &\nBehaviors", "PME": "Personal\nMedical", "ND": "Natural\nDisasters"}
    ax.set_xticklabels([cat_labels.get(c, c) for c in cat_names], fontsize=9)
    ax.set_ylabel("Rate")
    ax.set_title("VERI-Emergency — Error Rates by Category", fontsize=12, fontweight="semibold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linewidth=0.4, alpha=0.3, color="#cccccc")
    fig.tight_layout()
    _save(fig, save_dir / "veri_category_rates")


# ── 4. Robot selection heatmap ───────────────────────────────────────────────

def plot_robot_heatmap(results_dir: Path, save_dir: Path):
    _apply_style()
    rows = _load_jsonl(results_dir / "robot_selection_results_main.jsonl")
    if not rows:
        rows = _load_jsonl(results_dir / "robot_selection_results_trial.jsonl")
    if not rows:
        return

    ROBOTS = ["Drone", "Humanoid", "Robot with Legs", "Robot with Wheels", "Underwater Robot"]

    # Bin scenarios by primary keyword
    SCENARIO_BINS = {
        "aerial/inspection": ["aerial", "inspect", "survey", "roof", "bridge", "high-rise"],
        "search & rescue": ["search", "rescue", "disaster", "rubble", "collapse"],
        "underwater": ["underwater", "marine", "ocean", "sea", "dive", "submerged"],
        "indoor/warehouse": ["indoor", "warehouse", "factory", "office", "hospital"],
        "transport/delivery": ["transport", "deliver", "cargo", "payload", "carry"],
        "outdoor/terrain": ["outdoor", "terrain", "mountain", "forest", "rough"],
    }

    def _bin_scenario(task: str) -> str:
        lower = task.lower() if task else ""
        for bin_name, keywords in SCENARIO_BINS.items():
            if any(k in lower for k in keywords):
                return bin_name
        return "other"

    # Build GT and pred matrices
    scenario_names = list(SCENARIO_BINS.keys()) + ["other"]
    gt_counts = np.zeros((len(scenario_names), len(ROBOTS)))
    pred_counts = np.zeros_like(gt_counts)

    for r in rows:
        # Infer scenario from task_name field or context
        task = r.get("task_name", "")
        sc = _bin_scenario(task)
        si = scenario_names.index(sc)

        for rb in r.get("robot_gt", []):
            if rb in ROBOTS:
                gt_counts[si, ROBOTS.index(rb)] += 1
        for rb in r.get("robot_pred", []):
            if rb in ROBOTS:
                pred_counts[si, ROBOTS.index(rb)] += 1

    # Normalize per-row to show proportions
    gt_norm = gt_counts / np.maximum(gt_counts.sum(axis=1, keepdims=True), 1)
    pred_norm = pred_counts / np.maximum(pred_counts.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#fafafa")

    for ax, data, title in [(axes[0], gt_norm, "Ground Truth"),
                             (axes[1], pred_norm, "Model Prediction")]:
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(ROBOTS)))
        ax.set_xticklabels([r.replace(" ", "\n") for r in ROBOTS], fontsize=8)
        ax.set_yticks(range(len(scenario_names)))
        ax.set_yticklabels(scenario_names, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="semibold")

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if v > 0.01:
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.5 else "black")

    fig.suptitle("Robot Selection — Scenario vs Robot Distribution",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.colorbar(im, ax=axes, shrink=0.7, label="Selection Proportion")
    _save(fig, save_dir / "robot_selection_heatmap")


# ── 5. GEOBench per-task accuracy ───────────────────────────────────────────

def plot_geobench_tasks(results_dir: Path, save_dir: Path):
    _apply_style()
    rows = _load_jsonl(results_dir / "geobench_vlm_results_main.jsonl")
    if not rows:
        rows = _load_jsonl(results_dir / "geobench_vlm_results_trial.jsonl")
    if not rows:
        return

    tasks = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in rows:
        task = r.get("task_name", "unknown")
        tasks[task]["total"] += 1
        if r.get("correct"):
            tasks[task]["correct"] += 1

    task_names = sorted(tasks.keys(), key=lambda t: tasks[t]["total"], reverse=True)[:15]
    if not task_names:
        return

    acc = [tasks[t]["correct"] / max(tasks[t]["total"], 1) for t in task_names]
    counts = [tasks[t]["total"] for t in task_names]

    fig, ax = plt.subplots(figsize=(10, max(4, len(task_names) * 0.4)))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#f5f5f5")

    y = np.arange(len(task_names))
    colors = [C_DANGER if a < 0.5 else C_NEUTRAL for a in acc]
    bars = ax.barh(y, acc, color=colors, alpha=0.85)

    for i, (a, n) in enumerate(zip(acc, counts)):
        ax.text(a + 0.02, i, f"{a:.0%} (n={n})", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([t[:35] for t in task_names], fontsize=8)
    ax.set_xlabel("Risk Classification Accuracy")
    ax.set_title("GEOBench-VLM — Per-Task Risk Accuracy", fontsize=12, fontweight="semibold")
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()
    ax.grid(True, linewidth=0.4, alpha=0.3, color="#cccccc")
    fig.tight_layout()
    _save(fig, save_dir / "geobench_tasks")


# ── 6. Risk severity distributions ──────────────────────────────────────────

def plot_severity_distributions(results_dir: Path, save_dir: Path):
    _apply_style()

    severity_map = {"low": 1, "medium": 2, "high": 3}
    datasets = {
        "veri_emergency": ("VERI-Emergency", C_DANGER),
        "geobench_vlm": ("GEOBench-VLM", C_NEUTRAL),
        "robot_selection": ("Robot-Selection", BENCHMARK_COLORS["robot_selection"]),
    }

    panels = []
    for ds_key, (label, color) in datasets.items():
        for suffix in ("_main.jsonl", "_trial.jsonl"):
            rows = _load_jsonl(results_dir / f"{ds_key}_results{suffix}")
            if rows:
                break
        if not rows:
            continue

        safe_sev = [severity_map.get(r.get("risk_severity_pred", ""), 0)
                     for r in rows if r.get("risk_label_gt") == "safe" and r.get("risk_severity_pred") in severity_map]
        danger_sev = [severity_map.get(r.get("risk_severity_pred", ""), 0)
                       for r in rows if r.get("risk_label_gt") == "danger" and r.get("risk_severity_pred") in severity_map]
        if safe_sev or danger_sev:
            panels.append((label, safe_sev, danger_sev))

    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5))
    fig.patch.set_facecolor("#fafafa")
    if len(panels) == 1:
        axes = [axes]

    bins = [0.5, 1.5, 2.5, 3.5]
    for ax, (label, safe_sev, danger_sev) in zip(axes, panels):
        ax.set_facecolor("#f5f5f5")
        if safe_sev:
            ax.hist(safe_sev, bins=bins, alpha=0.6, color=C_SAFE, label="GT: Safe", density=True)
        if danger_sev:
            ax.hist(danger_sev, bins=bins, alpha=0.6, color=C_DANGER, label="GT: Danger", density=True)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Low", "Medium", "High"])
        ax.set_xlabel("Predicted Severity")
        ax.set_ylabel("Density")
        ax.set_title(label, fontsize=11, fontweight="semibold")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.grid(True, linewidth=0.4, alpha=0.3, color="#cccccc")

    fig.suptitle("Predicted Risk Severity — Safe vs Dangerous Ground Truth",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, save_dir / "severity_distributions")


# ── Entry points ─────────────────────────────────────────────────────────────

def generate(results_dir: Optional[Path] = None, save_dir: Optional[Path] = None,
             model=None):
    """Main entry point.

    Called from eval/robot_risk_benchmarks.py or from
    generate_all_plots.py --fig fig_robot_risk_benchmarks.
    """
    _apply_style()

    if results_dir is None:
        results_dir = Path("benchmarks") / "results"
    else:
        results_dir = Path(results_dir)

    if save_dir is None:
        save_dir = Path("benchmarks") / "plots" / "robot_risk_benchmarks"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = _load_metrics(results_dir)
    if metrics is None:
        print("  [fig_robot_risk_benchmarks] No metrics found in", results_dir)
        print("  Run evaluation first:")
        print("    python eval/robot_risk_benchmarks.py --trial --model-path <path>")
        return None

    print(f"  Generating robot risk benchmark plots → {save_dir}")

    plot_overview(metrics, save_dir)
    plot_veri_confusion(results_dir, save_dir)
    plot_veri_categories(results_dir, save_dir)
    plot_robot_heatmap(results_dir, save_dir)
    plot_geobench_tasks(results_dir, save_dir)
    plot_severity_distributions(results_dir, save_dir)

    print(f"  All plots saved to {save_dir}")
    return save_dir
