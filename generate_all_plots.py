#!/usr/bin/env python3
"""
generate_all_plots.py
=====================
Master script for EmberNet visualizations.

Actions performed:
  1. Loads W&B run history (optional)
  2. Generates ALL plots across 7 categories
  3. Saves to hierarchical `plots/` folder structure
  4. Uploads plot images back to W&B
  5. Generates a summary report `plots/REPORT_<timestamp>.md`

Usage:
------
    # Generate all plots (training + paper figures) with synthetic data
    python generate_all_plots.py --all

    # Generate only the 7 ECCV/AAAI paper figures
    python generate_all_plots.py --all --paper-only

    # Generate a single paper figure by name
    python generate_all_plots.py --fig fig_architecture_overview
    python generate_all_plots.py --fig fig_ternary_stats
    python generate_all_plots.py --fig fig_moe_routing
    python generate_all_plots.py --fig fig_latency_energy
    python generate_all_plots.py --fig fig_va_token_effects
    python generate_all_plots.py --fig fig_va_answer_level
    python generate_all_plots.py --fig fig_qualitative_grid

    # Load data from a specific W&B run
    python generate_all_plots.py --wandb-run EmberNet/<run_id>

    # Load data from local checkpoint directory
    python generate_all_plots.py --checkpoint-dir ./checkpoints/checkpoint_step_5000.pt

    # Generate only one training-viz section
    python generate_all_plots.py --section expert_analysis

    # Enable W&B upload of generated plots
    python generate_all_plots.py --all --upload-to-wandb

Available --section values:
    training_dynamics  expert_analysis  architecture  quantization
    dataset_analysis   performance      stage_comparison  all (default)

Available --fig values (paper figures):
    fig_architecture_overview  fig_ternary_stats   fig_moe_routing
    fig_latency_energy         fig_va_token_effects fig_va_answer_level
    fig_qualitative_grid
"""

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Add repo root to sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from visualizations.config import (
    PLOT_DIRS, VIZ_CONFIG, ensure_plot_dirs, apply_mpl_style,
)
from visualizations.training_dynamics   import TrainingDynamicsPlotter
from visualizations.expert_analysis     import ExpertAnalysisPlotter
from visualizations.architecture        import ArchitecturePlotter
from visualizations.quantization        import QuantizationPlotter
from visualizations.dataset_analysis    import DatasetAnalysisPlotter
from visualizations.performance_metrics import PerformanceMetricsPlotter
from visualizations.stage_comparison    import StageComparisonPlotter
from visualizations.wandb_utils         import WandBLogger

apply_mpl_style()


# ===========================================================================
# Paper-figure registry
# ===========================================================================

_PAPER_FIGS = {
    "fig_architecture_overview": "visualizations.fig_architecture_overview",
    "fig_ternary_stats":         "visualizations.fig_ternary_stats",
    "fig_moe_routing":           "visualizations.fig_moe_routing",
    "fig_latency_energy":        "visualizations.fig_latency_energy",
    "fig_va_token_effects":      "visualizations.fig_va_token_effects",
    "fig_va_answer_level":       "visualizations.fig_va_answer_level",
    "fig_qualitative_grid":      "visualizations.fig_qualitative_grid",
}


def generate_paper_fig(fig_name: str, save_dir=None, model=None):
    """
    Import and run the `generate()` function from the named paper-figure module.

    Parameters
    ----------
    fig_name : str
        One of the keys in _PAPER_FIGS (e.g. "fig_ternary_stats").
    save_dir : Path, optional
        Override default output directory.
    model : EmberNetVLM, optional
        Live model for real-data extraction.

    Returns
    -------
    Path or None
    """
    from pathlib import Path as _Path
    import importlib

    if fig_name not in _PAPER_FIGS:
        print(f"  [WARNING] Unknown paper figure: '{fig_name}'.  "
              f"Available: {list(_PAPER_FIGS)}")
        return None

    module = importlib.import_module(_PAPER_FIGS[fig_name])
    out_dir = _Path(save_dir) if save_dir else _Path("plots/paper_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    return module.generate(save_dir=out_dir, model=model)


# ===========================================================================
# W&B data loader
# ===========================================================================

def load_wandb_data(run_path: str) -> Dict:
    """
    Load logged scalars from a W&B run into a dict of NumPy arrays.

    Parameters
    ----------
    run_path : str
        Full run path in the format ``<entity>/<project>/<run_id>`` or
        ``<project>/<run_id>``.

    Returns
    -------
    Dict
        Flat dict mapping W&B metric name → NumPy array of all logged values.
        Empty dict on failure.
    """
    try:
        import wandb
        api = wandb.Api()
        run = api.run(run_path)
        history = run.scan_history()

        import pandas as pd
        import numpy as np

        rows = list(history)
        if not rows:
            print(f"  [WARNING] W&B run {run_path} has no history rows.")
            return {}

        df = pd.DataFrame(rows)
        data = {}
        for col in df.columns:
            if col.startswith("_"):
                continue
            vals = df[col].dropna().values
            if len(vals):
                try:
                    data[col] = vals.astype(float)
                except (ValueError, TypeError):
                    data[col] = vals   # keep as-is for string cols

        print(f"  Loaded {len(data)} metrics from W&B run: {run_path}")
        return data

    except Exception as e:
        print(f"  [WARNING] Could not load W&B data from '{run_path}': {e}")
        return {}


def load_checkpoint_data(checkpoint_path: str) -> Dict:
    """
    Extract scalar training info from a saved checkpoint file.
    """
    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        data = {}
        if "global_step" in checkpoint:
            import numpy as np
            data["train/step"] = np.array([checkpoint["global_step"]])
        print(f"  Loaded checkpoint info from: {checkpoint_path}")
        return data
    except Exception as e:
        print(f"  [WARNING] Could not load checkpoint '{checkpoint_path}': {e}")
        return {}


# ===========================================================================
# Plot registry
# ===========================================================================

def _build_context(raw_data: Dict) -> Dict:
    """
    Convert a flat W&B history dict into the nested format expected
    by each plotter.  Falls back to None (synthetic data) for any missing key.
    """
    import numpy as np

    def get_arr(key):
        return raw_data.get(key)

    # ---- Training dynamics ----
    td = {}
    if "train/loss" in raw_data:
        n = len(raw_data["train/loss"])
        td["loss"] = {
            "stage1_steps":      raw_data.get("train/step",     np.arange(n))[:n // 2],
            "stage2_steps":      raw_data.get("train/step",     np.arange(n))[n // 2:],
            "stage1_train_loss": raw_data["train/loss"][:n // 2],
            "stage2_train_loss": raw_data["train/loss"][n // 2:],
            "stage1_val_loss":   raw_data.get("val/loss", raw_data["train/loss"])[:n // 2],
            "stage2_val_loss":   raw_data.get("val/loss", raw_data["train/loss"])[n // 2:],
        }
    if "train/lr" in raw_data:
        td["lr_schedule"] = {
            "steps": raw_data.get("train/step"),
            "actual_lr": raw_data["train/lr"],
        }
    if "train/grad_norm" in raw_data:
        td["grad_norms"] = {
            "steps":       raw_data.get("train/step"),
            "global_norm": raw_data["train/grad_norm"],
            "clip_threshold": 1.0,
        }

    # ---- Energy / CO₂ (CodeCarbon-logged) ----
    if "energy/stage1_kwh" in raw_data or "energy/stage2_kwh" in raw_data:
        n = len(raw_data.get("train/loss", np.array([]))) or 1
        steps_all = raw_data.get("train/step", np.arange(n))
        half = len(steps_all) // 2

        td["energy"] = {
            "s1_steps":    steps_all[:half],
            "s1_energy":   raw_data.get("energy/stage1_kwh", np.zeros(half)),
            "s2_steps":    steps_all[half:],
            "s2_energy":   raw_data.get("energy/stage2_kwh", np.zeros(len(steps_all) - half)),
        }
        td["co2"] = {
            "s1_steps":  steps_all[:half],
            "s1_co2_kg": raw_data.get("energy/stage1_co2_kg", np.zeros(half)),
            "s2_steps":  steps_all[half:],
            "s2_co2_kg": raw_data.get("energy/stage2_co2_kg", np.zeros(len(steps_all) - half)),
        }
        # Per-token efficiency ‑ requires cumulative_tokens key
        if "train/cumulative_tokens" in raw_data:
            cum_tok = raw_data["train/cumulative_tokens"]
            s1_e = raw_data.get("energy/stage1_kwh", np.zeros_like(cum_tok[:half]))
            s2_e = raw_data.get("energy/stage2_kwh", np.zeros_like(cum_tok[half:]))
            cum_e = np.concatenate([s1_e, s2_e])
            tok_m = np.maximum(cum_tok / 1e6, 1e-9)
            td["energy_per_token"] = {
                "tokens_m":     tok_m,
                "kwh_per_m_tok": cum_e / tok_m,
                "steps":         steps_all,
            }
        td["stage_energy"] = {
            "stage_names":  ["Stage 1", "Stage 2"],
            "s1_steps":     steps_all[:half],
            "s2_steps":     steps_all[half:],
            "s1_energy":    raw_data.get("energy/stage1_kwh", np.zeros(half)),
            "s2_energy":    raw_data.get("energy/stage2_kwh", np.zeros(len(steps_all) - half)),
            "s1_co2_kg":    raw_data.get("energy/stage1_co2_kg", np.zeros(half)),
            "s2_co2_kg":    raw_data.get("energy/stage2_co2_kg", np.zeros(len(steps_all) - half)),
        }

    # ---- Expert analysis ----
    ea = {}
    if "train/routing_entropy" in raw_data:
        ea["routing_entropy"] = {
            "steps":   raw_data.get("train/step"),
            "entropy": raw_data["train/routing_entropy"],
        }
    # Stacked expert probabilities (expert_0..expert_7 columns)
    _expert_cols = [k for k in raw_data if k.startswith("train/expert_") and k[len("train/expert_"):].isdigit()]
    if len(_expert_cols) == 8:
        _expert_cols_sorted = sorted(_expert_cols, key=lambda k: int(k.rsplit("_", 1)[-1]))
        _n = min(len(raw_data[c]) for c in _expert_cols_sorted)
        ea["stacked_area"] = {
            "steps":         raw_data.get("train/step", np.arange(_n))[:_n],
            "expert_probs":  np.column_stack([raw_data[c][:_n] for c in _expert_cols_sorted]),
        }

    return {"training_dynamics": td, "expert_analysis": ea}


# ===========================================================================
# Main orchestrator
# ===========================================================================

SECTION_NAMES = [
    "training_dynamics",
    "expert_analysis",
    "architecture",
    "quantization",
    "dataset_analysis",
    "performance",
    "stage_comparison",
]


def generate_section(section: str, logger: WandBLogger, context: Dict) -> List[Path]:
    """Generate all plots for a single section. Returns list of saved paths."""
    step = None
    td_data = context.get("training_dynamics", {})

    if section == "training_dynamics":
        print("\n[§1] Training Dynamics")
        plotter = TrainingDynamicsPlotter(logger=logger)
        return plotter.generate_all(data=td_data, step=step)

    elif section == "expert_analysis":
        print("\n[§2] Expert Analysis")
        plotter = ExpertAnalysisPlotter(logger=logger)
        return plotter.generate_all(data=context.get("expert_analysis"), step=step)

    elif section == "architecture":
        print("\n[§3] Architecture Visualizations")
        plotter = ArchitecturePlotter(logger=logger)
        return plotter.generate_all(data=context.get("architecture"), step=step)

    elif section == "quantization":
        print("\n[§4] Quantization Analysis")
        plotter = QuantizationPlotter(logger=logger)
        return plotter.generate_all(data=context.get("quantization"), step=step)

    elif section == "dataset_analysis":
        print("\n[§5] Dataset Analysis")
        plotter = DatasetAnalysisPlotter(logger=logger)
        return plotter.generate_all(data=context.get("dataset_analysis"), step=step)

    elif section == "performance":
        print("\n[§6] Performance Metrics")
        plotter = PerformanceMetricsPlotter(logger=logger)
        return plotter.generate_all(data=context.get("performance"), step=step)

    elif section == "stage_comparison":
        print("\n[§7] Stage Comparison & Ablations")
        plotter = StageComparisonPlotter(logger=logger)
        return plotter.generate_all(data=context.get("stage_comparison"), step=step)

    else:
        print(f"  [WARNING] Unknown section: {section}")
        return []


def write_report(
    generated: List[Path],
    failed: List[str],
    warnings: List[str],
    wandb_project: str,
) -> Path:
    """Write a Markdown summary report listing all generated plots."""
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rpt_path = Path("plots") / f"REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    rpt_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# EmberNet Training Visualization Report",
        "",
        f"**Generated:** {ts}  ",
        f"**W&B Project:** {wandb_project}  ",
        f"**Stages:** Stage 1, Stage 2  ",
        "",
        "---",
        "",
    ]

    # Group by section
    sections = {
        "1. Training Dynamics":          "training_dynamics",
        "2. Expert Analysis":            "expert_analysis",
        "3. Architecture Visualizations":"architecture_visualizations",
        "4. Quantization Analysis":      "quantization_analysis",
        "5. Dataset Analysis":           "dataset_analysis",
        "6. Performance Metrics":        "performance_metrics",
        "7. Stage Comparison":           "stage_comparison",
        "8. Paper Figures (ECCV/AAAI)":  "paper_figures",
    }

    for section_title, folder_fragment in sections.items():
        section_plots = [
            p for p in generated
            if folder_fragment in p.as_posix()
        ]
        if not section_plots:
            continue
        lines.append(f"## {section_title}")
        lines.append("")
        for p in section_plots:
            rel = p.as_posix()
            lines.append(f"- [{p.name}]({rel})")
        lines.append("")

    # Summary footer
    lines += [
        "---",
        "",
        f"**Total Plots Generated:** {len(generated)}  ",
        f"**Failed Plots:** {len(failed)}  ",
        f"**Warnings:** {len(warnings)}  ",
        "",
    ]

    if failed:
        lines += ["### Failed Plots", ""]
        for f in failed:
            lines.append(f"- {f}")
        lines.append("")

    if warnings:
        lines += ["### Warnings", ""]
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    rpt_path.write_text("\n".join(lines), encoding="utf-8")
    return rpt_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate all EmberNet training visualizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--wandb-run", type=str, default=None,
        help="W&B run path (entity/project/run_id) to load metric history from.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Path to a checkpoint .pt file to load step info from.",
    )
    parser.add_argument(
        "--section", type=str, default="all",
        choices=SECTION_NAMES + ["all"],
        help="Generate only this section (default: all).",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="EmberNet",
        help="W&B project name (for report metadata).",
    )
    parser.add_argument(
        "--upload-to-wandb", action="store_true", default=False,
        help="Upload generated plots back to W&B as images.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots",
        help="Root output directory (default: plots/).",
    )
    parser.add_argument(
        "--all", action="store_true", default=False,
        help="Generate ALL plots: both training-viz sections AND all 7 paper figures.",
    )
    parser.add_argument(
        "--paper-only", action="store_true", default=False,
        help="Generate only the 7 ECCV/AAAI paper figures (skips training-viz sections).",
    )
    parser.add_argument(
        "--fig", type=str, default=None,
        choices=list(_PAPER_FIGS.keys()),
        metavar="FIG_NAME",
        help=("Generate a single paper figure by module name.  "
              "Choices: " + ", ".join(_PAPER_FIGS.keys())),
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to an EmberNet checkpoint to use for real-data extraction in paper figures.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  EmberNet – Generate All Visualizations")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Create all plot directories
    # ------------------------------------------------------------------
    ensure_plot_dirs()
    print("✓ Plot directories created")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    raw_data: Dict = {}
    warnings: List[str] = []

    if args.wandb_run:
        print(f"\nLoading W&B history: {args.wandb_run}")
        raw_data = load_wandb_data(args.wandb_run)
    elif args.checkpoint_dir:
        print(f"\nLoading checkpoint: {args.checkpoint_dir}")
        raw_data = load_checkpoint_data(args.checkpoint_dir)
    else:
        print("\nNo data source specified → using synthetic placeholder data for all plots.")
        warnings.append("No data source — all plots use synthetic placeholder data.")

    context = _build_context(raw_data)

    # ------------------------------------------------------------------
    # W&B logger
    # ------------------------------------------------------------------
    try:
        import wandb
        logger = WandBLogger(disabled=not args.upload_to_wandb)
        if args.upload_to_wandb and wandb.run is None:
            wandb.init(project=args.wandb_project, job_type="visualize",
                       name=f"viz_{datetime.now().strftime('%Y%m%d_%H%M')}")
    except ImportError:
        logger = WandBLogger(disabled=True)
        warnings.append("wandb not installed — W&B upload disabled.")

    # ------------------------------------------------------------------
    # Load optional EmberNet model for real-data paper figures
    # ------------------------------------------------------------------
    live_model = None
    if args.model:
        try:
            from inference.infer import EmberVLM
            print(f"\nLoading model for paper figures: {args.model}")
            live_model = EmberVLM(model_path=args.model)
        except Exception as e:
            warnings.append(f"Model load failed ({e}) — paper figures will use synthetic data.")
            print(f"  [WARNING] {warnings[-1]}")

    # ------------------------------------------------------------------
    # Handle --fig (single paper figure shortcut)
    # ------------------------------------------------------------------
    if args.fig:
        paper_dir = Path(args.output_dir) / "paper_figures"
        print(f"\n[paper fig] Generating: {args.fig}")
        result = generate_paper_fig(args.fig, save_dir=paper_dir, model=live_model)
        if result and result.exists():
            print(f"  ✓ Saved → {result}")
            return 0
        else:
            print(f"  ✗ Failed")
            return 1

    # ------------------------------------------------------------------
    # Generate training-viz sections (skipped when --paper-only)
    # ------------------------------------------------------------------
    all_paths: List[Path] = []
    failed:    List[str]  = []

    if not args.paper_only:
        sections_to_run = SECTION_NAMES if args.section == "all" else [args.section]
        for section in sections_to_run:
            try:
                paths = generate_section(section, logger, context)
                all_paths.extend([p for p in paths if p and p.exists()])
            except Exception as e:
                msg = f"  [ERROR] Section '{section}' failed: {e}\n{traceback.format_exc()}"
                print(msg)
                failed.append(f"{section}: {e}")

    # ------------------------------------------------------------------
    # Generate paper figures (when --all or --paper-only)
    # ------------------------------------------------------------------
    if args.all or args.paper_only:
        paper_dir = Path(args.output_dir) / "paper_figures"
        print("\n[§8] ECCV/AAAI Paper Figures")
        for fig_name in _PAPER_FIGS:
            try:
                result = generate_paper_fig(fig_name, save_dir=paper_dir, model=live_model)
                if result and result.exists():
                    all_paths.append(result)
                    print(f"  ✓ {fig_name}")
                else:
                    failed.append(f"{fig_name}: returned None or file missing")
            except Exception as e:
                print(f"  [ERROR] {fig_name}: {e}\n{traceback.format_exc()}")
                failed.append(f"{fig_name}: {e}")

    # ------------------------------------------------------------------
    # Upload generated plots to W&B
    # ------------------------------------------------------------------
    if args.upload_to_wandb and not logger.disabled:
        print(f"\nUploading {len(all_paths)} plots to W&B …")
        logger.upload_plots(all_paths)

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    report_path = write_report(all_paths, failed, warnings, args.wandb_project)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  DONE — {len(all_paths)} plots generated, {len(failed)} failed")
    print(f"  Report: {report_path}")
    print("=" * 70)

    if failed:
        print("\nFailed sections:")
        for f in failed:
            print(f"  ✗ {f}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
