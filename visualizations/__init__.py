"""
EmberNet Visualization Package

Comprehensive visualizations for ECCV/top-tier AI-ML conference papers.
All plots are logged to W&B and saved locally in a hierarchical folder structure.
"""

from visualizations.config import VIZ_CONFIG, EXPERT_COLORS, PLOT_DIRS
from visualizations.training_dynamics import TrainingDynamicsPlotter
from visualizations.expert_analysis import ExpertAnalysisPlotter
from visualizations.architecture import ArchitecturePlotter
from visualizations.quantization import QuantizationPlotter
from visualizations.dataset_analysis import DatasetAnalysisPlotter
from visualizations.performance_metrics import PerformanceMetricsPlotter
from visualizations.stage_comparison import StageComparisonPlotter
from visualizations.wandb_utils import WandBLogger
from visualizations.live_plotter import LivePlotter
from visualizations.benchmark_viz import BenchmarkVisualizer

__all__ = [
    "VIZ_CONFIG",
    "EXPERT_COLORS",
    "PLOT_DIRS",
    "TrainingDynamicsPlotter",
    "ExpertAnalysisPlotter",
    "ArchitecturePlotter",
    "QuantizationPlotter",
    "DatasetAnalysisPlotter",
    "PerformanceMetricsPlotter",
    "StageComparisonPlotter",
    "WandBLogger",
    "LivePlotter",
    "BenchmarkVisualizer",
]
