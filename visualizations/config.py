"""
Shared visualization configuration for EmberNet.

Defines the consistent color palette, font specs, figure sizes, folder paths,
expert metadata, and dataset metadata used across all plots.
"""

from pathlib import Path
from datetime import datetime
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Root plot directory
# ---------------------------------------------------------------------------
PLOTS_ROOT = Path("plots")

# ---------------------------------------------------------------------------
# Hierarchical folder structure
# ---------------------------------------------------------------------------
PLOT_DIRS = {
    # Training Dynamics
    "loss_curves":           PLOTS_ROOT / "training_dynamics" / "loss_curves",
    "learning_rates":        PLOTS_ROOT / "training_dynamics" / "learning_rates",
    "gradient_stats":        PLOTS_ROOT / "training_dynamics" / "gradient_stats",
    "convergence":           PLOTS_ROOT / "training_dynamics" / "convergence",
    # Expert Analysis
    "routing_patterns":      PLOTS_ROOT / "expert_analysis" / "routing_patterns",
    "specialization_metrics":PLOTS_ROOT / "expert_analysis" / "specialization_metrics",
    "expert_utilization":    PLOTS_ROOT / "expert_analysis" / "expert_utilization",
    "spider_charts":         PLOTS_ROOT / "expert_analysis" / "spider_charts",
    # Architecture
    "model_diagrams":        PLOTS_ROOT / "architecture_visualizations" / "model_diagrams",
    "attention_maps":        PLOTS_ROOT / "architecture_visualizations" / "attention_maps",
    "token_flow":            PLOTS_ROOT / "architecture_visualizations" / "token_flow",
    # Quantization
    "weight_distributions":  PLOTS_ROOT / "quantization_analysis" / "weight_distributions",
    "activation_histograms": PLOTS_ROOT / "quantization_analysis" / "activation_histograms",
    "bitwidth_efficiency":   PLOTS_ROOT / "quantization_analysis" / "bitwidth_efficiency",
    # Dataset Analysis
    "token_statistics":      PLOTS_ROOT / "dataset_analysis" / "token_statistics",
    "domain_distributions":  PLOTS_ROOT / "dataset_analysis" / "domain_distributions",
    "sample_visualizations": PLOTS_ROOT / "dataset_analysis" / "sample_visualizations",
    # Performance Metrics
    "accuracy_curves":       PLOTS_ROOT / "performance_metrics" / "accuracy_curves",
    "perplexity_progression":PLOTS_ROOT / "performance_metrics" / "perplexity_progression",
    "benchmark_comparisons": PLOTS_ROOT / "performance_metrics" / "benchmark_comparisons",
    # Stage Comparison
    "stage1_vs_stage2":      PLOTS_ROOT / "stage_comparison" / "stage1_vs_stage2",
    "ablation_studies":      PLOTS_ROOT / "stage_comparison" / "ablation_studies",
    # Benchmark Results
    "benchmark_spider":      PLOTS_ROOT / "benchmark_results" / "spider",
    "benchmark_task_scores": PLOTS_ROOT / "benchmark_results" / "task_scores",
    "benchmark_domain":      PLOTS_ROOT / "benchmark_results" / "domain_analysis",
    "benchmark_dashboard":   PLOTS_ROOT / "benchmark_results" / "dashboard",
    # Errors
    "errors":                PLOTS_ROOT / "errors",
}


def ensure_plot_dirs():
    """Create all plot directories if they do not exist."""
    for d in PLOT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


# Relative sub-paths (below whatever root is active).
# Used by set_plots_root() to rebuild PLOT_DIRS when the root changes.
_PLOT_DIR_SUFFIXES = {
    "loss_curves":            Path("training_dynamics/loss_curves"),
    "learning_rates":         Path("training_dynamics/learning_rates"),
    "gradient_stats":         Path("training_dynamics/gradient_stats"),
    "convergence":            Path("training_dynamics/convergence"),
    "routing_patterns":       Path("expert_analysis/routing_patterns"),
    "specialization_metrics": Path("expert_analysis/specialization_metrics"),
    "expert_utilization":     Path("expert_analysis/expert_utilization"),
    "spider_charts":          Path("expert_analysis/spider_charts"),
    "model_diagrams":         Path("architecture_visualizations/model_diagrams"),
    "attention_maps":         Path("architecture_visualizations/attention_maps"),
    "token_flow":             Path("architecture_visualizations/token_flow"),
    "weight_distributions":   Path("quantization_analysis/weight_distributions"),
    "activation_histograms":  Path("quantization_analysis/activation_histograms"),
    "bitwidth_efficiency":    Path("quantization_analysis/bitwidth_efficiency"),
    "token_statistics":       Path("dataset_analysis/token_statistics"),
    "domain_distributions":   Path("dataset_analysis/domain_distributions"),
    "sample_visualizations":  Path("dataset_analysis/sample_visualizations"),
    "accuracy_curves":        Path("performance_metrics/accuracy_curves"),
    "perplexity_progression": Path("performance_metrics/perplexity_progression"),
    "benchmark_comparisons":  Path("performance_metrics/benchmark_comparisons"),
    "stage1_vs_stage2":       Path("stage_comparison/stage1_vs_stage2"),
    "ablation_studies":       Path("stage_comparison/ablation_studies"),
    "benchmark_spider":       Path("benchmark_results/spider"),
    "benchmark_task_scores":  Path("benchmark_results/task_scores"),
    "benchmark_domain":       Path("benchmark_results/domain_analysis"),
    "benchmark_dashboard":    Path("benchmark_results/dashboard"),
    "errors":                 Path("errors"),
}


def set_plots_root(new_root):
    """Redirect all plot output directories under a new root folder.

    Call this before instantiating any plotter to control where files land.
    Typically called with ``{output_dir}/plots`` so plots sit next to checkpoints.
    """
    global PLOTS_ROOT, PLOT_DIRS
    PLOTS_ROOT = Path(new_root)
    for key, suffix in _PLOT_DIR_SUFFIXES.items():
        PLOT_DIRS[key] = PLOTS_ROOT / suffix
    ensure_plot_dirs()


# ---------------------------------------------------------------------------
# Expert metadata
# ---------------------------------------------------------------------------
EXPERT_NAMES = [
    "vision_ocr",
    "vision_diagram",
    "code_math_chart",
    "code_math_formula",
    "spatial_scene",
    "spatial_reasoning",
    "agentic_knowledge",
    "agentic_reasoning",
]

EXPERT_COLORS = {
    "vision_ocr":          "#1f77b4",   # blue
    "vision_diagram":      "#ff7f0e",   # orange
    "code_math_chart":     "#2ca02c",   # green
    "code_math_formula":   "#d62728",   # red
    "spatial_scene":       "#9467bd",   # purple
    "spatial_reasoning":   "#8c564b",   # brown
    "agentic_knowledge":   "#e377c2",   # pink
    "agentic_reasoning":   "#7f7f7f",   # gray
    "shared_expert":       "#bcbd22",   # yellow-green
}

EXPERT_LABELS = {
    "vision_ocr":          "E0: vision_ocr",
    "vision_diagram":      "E1: vision_diagram",
    "code_math_chart":     "E2: code_math_chart",
    "code_math_formula":   "E3: code_math_formula",
    "spatial_scene":       "E4: spatial_scene",
    "spatial_reasoning":   "E5: spatial_reasoning",
    "agentic_knowledge":   "E6: agentic_knowledge",
    "agentic_reasoning":   "E7: agentic_reasoning",
}

# ---------------------------------------------------------------------------
# Stage colors
# ---------------------------------------------------------------------------
STAGE_COLORS = {
    1: "#17becf",   # cyan  – Stage 1
    2: "#ff6b6b",   # coral – Stage 2
}

# ---------------------------------------------------------------------------
# Dataset metadata grouped by domain
# ---------------------------------------------------------------------------
DATASET_DOMAINS = {
    "vision_ocr": [
        "TextVQA", "DocVQA", "OCR-VQA", "InfoVQA", "AI2D",
    ],
    "code_math_chart": [
        "ChartQA", "PlotQA", "FigureQA", "DVQA", "MathVista",
    ],
    "spatial_scene": [
        "VQAv2", "GQA", "VisualGenome",
    ],
    "agentic_reasoning": [
        "ScienceQA", "OK-VQA", "A-OKVQA", "CLEVR",
    ],
    "alignment": [
        "LLaVA-Instruct", "ShareGPT4V", "ALLaVA",
    ],
}

ALL_DATASETS = [ds for datasets in DATASET_DOMAINS.values() for ds in datasets]

DOMAIN_COLORS = {
    "vision_ocr":       "#1f77b4",
    "code_math_chart":  "#2ca02c",
    "spatial_scene":    "#9467bd",
    "agentic_reasoning":"#ff7f0e",
    "alignment":        "#d62728",
}

# ---------------------------------------------------------------------------
# Matplotlib font and style settings  (publication quality)
# ---------------------------------------------------------------------------
VIZ_CONFIG = {
    # Figure sizes (width × height in inches)
    "figsize_single":  (10, 6),
    "figsize_dual":    (12, 6),
    "figsize_grid22":  (12, 12),
    "figsize_grid45":  (20, 12),
    "figsize_grid24":  (16, 8),
    # DPI
    "dpi": 300,
    # Font sizes
    "font_title":   14,
    "font_label":   12,
    "font_tick":    10,
    "font_legend":  10,
    # Line widths
    "lw_main":   2.0,
    "lw_thin":   1.2,
    "lw_dashed": 1.5,
    # Alpha
    "alpha_fill":    0.15,
    "alpha_overlay": 0.3,
    # LaTeX rendering (set False if LaTeX not installed)
    "use_latex": False,
}

# ---------------------------------------------------------------------------
# Matplotlib RC params
# ---------------------------------------------------------------------------
def apply_mpl_style():
    """Apply publication-quality matplotlib style."""
    mpl.rcParams.update({
        "text.usetex":          VIZ_CONFIG["use_latex"],
        "font.family":          "sans-serif",
        "font.size":            VIZ_CONFIG["font_label"],
        "axes.titlesize":       VIZ_CONFIG["font_title"],
        "axes.labelsize":       VIZ_CONFIG["font_label"],
        "xtick.labelsize":      VIZ_CONFIG["font_tick"],
        "ytick.labelsize":      VIZ_CONFIG["font_tick"],
        "legend.fontsize":      VIZ_CONFIG["font_legend"],
        "figure.dpi":           VIZ_CONFIG["dpi"],
        "savefig.dpi":          VIZ_CONFIG["dpi"],
        "savefig.bbox":         "tight",
        "axes.grid":            True,
        "grid.alpha":           0.3,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
    })


# ---------------------------------------------------------------------------
# File naming helper
# ---------------------------------------------------------------------------
def plot_filename(category: str, subcategory: str, name: str, ext: str = "png") -> str:
    """
    Returns a timestamped filename following the convention:
        {category}_{subcategory}_{name}_{YYYYMMDD_HHMM}.{ext}
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    stem = f"{category}_{subcategory}_{name}_{ts}"
    return f"{stem}.{ext}"


# ---------------------------------------------------------------------------
# Error logging helper
# ---------------------------------------------------------------------------
def log_plot_error(plot_name: str, error: Exception):
    """Write a plot error to the errors directory."""
    err_dir = PLOT_DIRS["errors"]
    err_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    err_file = err_dir / f"{plot_name}_error_{ts}.txt"
    with open(err_file, "w") as f:
        import traceback
        f.write(f"Plot: {plot_name}\n")
        f.write(f"Time: {ts}\n\n")
        f.write(traceback.format_exc())
    print(f"  [ERROR] {plot_name}: {error} — logged to {err_file}")
