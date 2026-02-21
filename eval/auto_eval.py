"""
eval/auto_eval.py
=================
Automatic post-training evaluation.

Called from train.py main() after ALL stages complete — no extra commands needed.
Silently skips if lmms-eval is not installed.

Trial mode : quick suite (5 tasks), 50 samples each  (~3-5 min on GPU)
Main  mode : core  suite (9 tasks), full dataset      (~1-2 hr on GPU)

Results and visualizations land in:
    {output_dir}/plots/benchmark_results/
        spider/           ← expert domain radar chart
        task_scores/      ← per-task horizontal bar chart
        domain_analysis/  ← domain heatmap + baseline delta
        dashboard/        ← full 4-panel summary
    {output_dir}/plots/benchmark_results/benchmark_scores_*.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Benchmark suites (same as run_eval.py but embedded so no import needed)
# ---------------------------------------------------------------------------

# Each task maps to one expert domain for the spider chart
SUITE_TRIAL = [
    "textvqa",       # E0: vision_ocr     (fastest OCR benchmark)
    "ai2d",          # E1: vision_diagram
    "chartqa",       # E2: code_math_chart
    "scienceqa_img", # E7: agentic_reasoning
    "mme",           # General
]

SUITE_CORE = [
    "textvqa",       # E0
    "ai2d",          # E1
    "chartqa",       # E2
    "mathvista",     # E3
    "vqav2",         # E4  (large; skipped in trial)
    "gqa",           # E5
    "ok_vqa",        # E6
    "scienceqa_img", # E7
    "mme",           # General
]


def run_auto_eval(
    checkpoint_path: str,
    output_dir: str,
    is_trial: bool,
    best_loss: float = 0.0,
    device: str = "auto",
) -> None:
    """
    Run lmms-eval on the final checkpoint, then generate benchmark visualizations.

    Parameters
    ----------
    checkpoint_path : str  Path to final_model.pt (or HF repo ID).
    output_dir      : str  Training output dir; plots will go under plots/benchmark_results/.
    is_trial        : bool Trial mode → 50-sample limit; core → full.
    best_loss       : float Final best training loss (shown in dashboard).
    device          : str  Device for inference ("auto", "cuda", "cpu").
    """
    _banner("POST-TRAINING BENCHMARK EVALUATION")

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    repo_root   = Path(__file__).resolve().parent.parent
    eval_dir    = repo_root / "eval"
    plots_root  = Path(output_dir) / "plots"
    bench_dir   = plots_root / "benchmark_results"
    bench_dir.mkdir(parents=True, exist_ok=True)

    mode   = "trial" if is_trial else "main"
    tasks  = SUITE_TRIAL if is_trial else SUITE_CORE
    limit  = 50 if is_trial else None

    print(f"  Mode    : {mode.upper()}")
    print(f"  Tasks   : {', '.join(tasks)}")
    print(f"  Limit   : {limit if limit else 'full dataset'}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Plots → : {bench_dir.resolve()}\n")

    # ------------------------------------------------------------------
    # Check lmms-eval is installed
    # ------------------------------------------------------------------
    try:
        import lmms_eval  # noqa: F401
    except ImportError:
        _warn(
            "lmms-eval not installed — benchmark evaluation skipped.\n"
            "  Install: git clone https://github.com/EvolvingLMMs-Lab/lmms-eval ~/lmms-eval\n"
            "           cd ~/lmms-eval && pip install -e '[all]'"
        )
        return

    # ------------------------------------------------------------------
    # Run lmms-eval via subprocess (most reliable across v0.5/v0.6)
    # ------------------------------------------------------------------
    results_dir = bench_dir / "lmms_raw"
    results_dir.mkdir(parents=True, exist_ok=True)

    pretrained  = str(Path(checkpoint_path).resolve())
    model_args  = f"pretrained={pretrained}"
    if device and device != "auto":
        model_args += f",device={device}"
    model_args += ",dtype=bfloat16"

    cmd = [
        sys.executable, str(repo_root / "eval" / "lmms_launcher.py"),
        "--model",        "embernet",
        "--model_args",   model_args,
        "--tasks",        ",".join(tasks),
        "--batch_size",   "1",
        "--output_path",  str(results_dir),
        "--trust_remote_code",
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    print("  Running: " + " ".join(cmd[:6]) + f" ... ({len(tasks)} tasks)")
    try:
        result = subprocess.run(cmd, capture_output=False, timeout=7200)  # 2hr max
        if result.returncode != 0:
            _warn(f"lmms-eval exited with code {result.returncode}")
    except subprocess.TimeoutExpired:
        _warn("lmms-eval timed out after 2 hours — partial results may be available")
    except Exception as e:
        _warn(f"lmms-eval subprocess error: {e}")

    # ------------------------------------------------------------------
    # Parse results JSON written by lmms-eval
    # ------------------------------------------------------------------
    scores = _load_scores(results_dir, tasks)

    if not scores:
        _warn("No benchmark scores found — skipping visualization.")
        return

    print(f"\n  Benchmark scores extracted ({len(scores)} tasks):")
    for t, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"    {t:<22} {s:.1f}%")

    # ------------------------------------------------------------------
    # Generate visualizations
    # ------------------------------------------------------------------
    _banner("Generating benchmark visualizations")
    try:
        import matplotlib
        matplotlib.use("Agg")

        from visualizations.config import set_plots_root, ensure_plot_dirs
        set_plots_root(plots_root)
        ensure_plot_dirs()

        from visualizations.benchmark_viz import BenchmarkVisualizer, save_scores_json

        viz = BenchmarkVisualizer(plots_root=bench_dir)
        generated = viz.plot_all(scores=scores, mode=mode, training_loss=best_loss)

        # Also persist the raw scores JSON
        json_path = save_scores_json(scores, bench_dir, mode)

        print(f"\n  {len(generated)} benchmark plots saved:")
        for p in generated:
            print(f"    {p.relative_to(repo_root)}")
        print(f"  Scores JSON: {json_path.relative_to(repo_root)}")

    except Exception as e:
        _warn(f"Visualization error (non-fatal): {e}")
        import traceback
        traceback.print_exc()

    _banner("EVALUATION COMPLETE")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_scores(results_dir: Path, expected_tasks: List[str]) -> Dict[str, float]:
    """
    Scan lmms-eval output directory for result JSON files and extract scores.
    lmms-eval v0.5+ writes a JSON per task run under results_dir.
    """
    from visualizations.benchmark_viz import extract_scores_from_lmms_results

    scores: Dict[str, float] = {}

    # Try the combined results.json first (v0.6 behavior)
    for json_file in sorted(results_dir.rglob("results*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            if "results" in data:
                extracted = extract_scores_from_lmms_results(data)
                scores.update(extracted)
        except Exception:
            pass

    # Also scan for per-task JSON files (v0.5 behavior / per-task logs)
    for json_file in sorted(results_dir.rglob("*.json")):
        if "results" in json_file.name and json_file not in scores:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                if "results" in data:
                    extracted = extract_scores_from_lmms_results(data)
                    scores.update(extracted)
            except Exception:
                pass

    return scores


def _banner(msg: str):
    print(f"\n{'='*64}")
    print(f"  {msg}")
    print(f"{'='*64}")


def _warn(msg: str):
    print(f"\n  [auto_eval] WARNING: {msg}")
