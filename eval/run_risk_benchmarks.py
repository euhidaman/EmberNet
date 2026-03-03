#!/usr/bin/env python3
"""
eval/run_risk_benchmarks.py
============================
Master script: run all three risk benchmark evaluations and aggregate results.

Usage:
    python eval/run_risk_benchmarks.py --model checkpoints/stage2/final_model.pt
    python eval/run_risk_benchmarks.py --model checkpoints/stage2/final_model.pt --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def _load_model_once(model_path: str, device: str | None):
    """Load EmberVLM once and return the instance."""
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from inference import EmberVLM
    return EmberVLM(model_path=model_path, device=device)


def run_topn_robots(model_path: str, benchmarks_dir: str, output_dir: str,
                     device: str | None, limit: int | None,
                     model=None) -> dict:
    from eval.eval_topn_robots import run
    dataset = str(Path(benchmarks_dir) / "robot-selection-dataset" /
                  "multi_robot_selection_dataset.json")
    args = SimpleNamespace(model=model_path, dataset=dataset, output_dir=output_dir,
                           device=device, limit=limit)
    return run(args, model=model)


def run_veri(model_path: str, benchmarks_dir: str, output_dir: str,
             device: str | None, limit: int | None,
             model=None) -> dict:
    from eval.eval_veri_emergency import run
    args = SimpleNamespace(model=model_path, benchmarks_dir=benchmarks_dir,
                           split="train", output_dir=output_dir,
                           device=device, limit=limit)
    return run(args, model=model)


def run_geobench(model_path: str, benchmarks_dir: str, output_dir: str,
                  device: str | None, limit: int | None,
                  model=None) -> dict:
    from eval.eval_geobench_vlm import run
    args = SimpleNamespace(model=model_path, benchmarks_dir=benchmarks_dir,
                           hf_dataset="aialliance/GEOBench-VLM", split="single",
                           output_dir=output_dir, device=device, limit=limit)
    return run(args, model=model)


BENCHMARKS = {
    "topn_robots": run_topn_robots,
    "veri_emergency": run_veri,
    "geobench_vlm": run_geobench,
}


def main():
    parser = argparse.ArgumentParser(description="Run all EmberNet risk benchmarks")
    parser.add_argument("--model", type=str, default="checkpoints/stage2/final_model.pt")
    parser.add_argument("--benchmarks-dir", type=str, default="benchmarks")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=str, nargs="*", default=[],
                        choices=list(BENCHMARKS.keys()),
                        help="Benchmarks to skip")
    args = parser.parse_args()

    # Ensure project root is on path
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model ONCE and share across all benchmarks
    print("  Loading model (once for all benchmarks)...")
    shared_model = _load_model_once(args.model, args.device)
    print("  Model ready.")

    summary = {}
    timings = {}
    skip_set = set(args.skip or [])

    for name, func in BENCHMARKS.items():
        if name in skip_set:
            print(f"\n>>> SKIP: {name}")
            continue
        print(f"\n{'#'*60}")
        print(f"  Running: {name}")
        print(f"{'#'*60}")
        t0 = time.time()
        metrics = func(args.model, args.benchmarks_dir, args.output_dir,
                        args.device, args.limit, model=shared_model)
        elapsed = time.time() - t0
        timings[name] = round(elapsed, 1)
        summary[name] = metrics

    # Build compact summary
    compact = {}
    if "topn_robots" in summary:
        m = summary["topn_robots"]
        compact["topn_robots"] = {
            "exact_set_accuracy": m.get("exact_set_accuracy"),
            "macro_f1": m.get("macro_f1"),
            "num_examples": m.get("num_examples"),
        }
    if "veri_emergency" in summary:
        m = summary["veri_emergency"]
        compact["veri_emergency"] = {
            "accuracy": m.get("accuracy"),
            "overreaction_rate": m.get("overreaction_rate"),
            "underreaction_rate": m.get("underreaction_rate"),
            "num_examples": m.get("num_examples"),
        }
    if "geobench_vlm" in summary:
        m = summary["geobench_vlm"]
        compact["geobench_vlm"] = {
            "overall_accuracy": m.get("overall_accuracy"),
            "num_examples": m.get("num_examples"),
        }
    compact["timings_sec"] = timings

    summary_file = out_dir / "risk_benchmark_summary.json"
    summary_file.write_text(json.dumps(compact, indent=2))

    # Markdown table
    print(f"\n{'='*70}")
    print("  Risk Benchmark Summary")
    print(f"{'='*70}")
    print(f"  | {'Benchmark':<25} | {'Key Metric':<18} | {'Value':>8} | {'N':>6} |")
    print(f"  |{'-'*27}|{'-'*20}|{'-'*10}|{'-'*8}|")

    if "topn_robots" in compact:
        c = compact["topn_robots"]
        print(f"  | {'Top-N Robot Selection':<25} | {'Exact Set Acc':<18} | {_fmt(c['exact_set_accuracy']):>8} | {c['num_examples']:>6} |")
        print(f"  | {'':<25} | {'Macro F1':<18} | {_fmt(c['macro_f1']):>8} | {'':<6} |")
    if "veri_emergency" in compact:
        c = compact["veri_emergency"]
        print(f"  | {'VERI-Emergency':<25} | {'Accuracy':<18} | {_fmt(c['accuracy']):>8} | {c['num_examples']:>6} |")
        print(f"  | {'':<25} | {'Overreaction':<18} | {_fmt(c['overreaction_rate']):>8} | {'':<6} |")
        print(f"  | {'':<25} | {'Underreaction':<18} | {_fmt(c['underreaction_rate']):>8} | {'':<6} |")
    if "geobench_vlm" in compact:
        c = compact["geobench_vlm"]
        print(f"  | {'GEO-Bench-VLM':<25} | {'Overall Acc':<18} | {_fmt(c['overall_accuracy']):>8} | {c['num_examples']:>6} |")

    print(f"{'='*70}")
    for name, sec in timings.items():
        print(f"  {name}: {sec}s")
    print(f"\n  Saved → {summary_file}")


def _fmt(v):
    if v is None:
        return "N/A"
    return f"{v:.4f}"


if __name__ == "__main__":
    main()
