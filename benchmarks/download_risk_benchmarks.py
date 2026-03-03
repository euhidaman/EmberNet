#!/usr/bin/env python3
"""
Download and prepare all benchmark data for the EmberNet risk evaluation suite.

Handles three datasets:
  1. Top-N Robot Selection  — local JSON, verify only
  2. VERI-Emergency         — HuggingFace  Dasool/VERI-Emergency
  3. GEO-Bench-VLM         — HuggingFace  aialliance/GEOBench-VLM

Usage:
    python benchmarks/download_risk_benchmarks.py
    python benchmarks/download_risk_benchmarks.py --skip-veri
    python benchmarks/download_risk_benchmarks.py --skip-geobench
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _require_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError:
        sys.exit("ERROR: `datasets` library required. Run: pip install datasets")


def verify_robot_selection(benchmarks_dir: Path) -> bool:
    path = benchmarks_dir / "robot-selection-dataset" / "multi_robot_selection_dataset.json"
    if not path.exists():
        print(f"  [MISSING] Robot-selection JSON not found at {path}")
        print(f"           It must be present in the repo (no download needed).")
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"  [OK] Robot-selection dataset: {len(data)} scenarios")
    return True


def download_veri(benchmarks_dir: Path) -> bool:
    from datasets import load_dataset

    dest = benchmarks_dir / "veri_emergency"
    dest.mkdir(parents=True, exist_ok=True)
    meta_file = dest / "_meta.json"

    print("  Downloading Dasool/VERI-Emergency ...")
    ds = load_dataset("Dasool/VERI-Emergency", split="train")

    images_dir = dest / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, row in enumerate(ds):
        rec = {}
        for k, v in row.items():
            if k == "image" and v is not None:
                img_id = row.get("image_id", f"veri_{i:04d}")
                fname = f"{img_id}.png"
                v.save(images_dir / fname)
                rec["image_path"] = str(images_dir / fname)
            else:
                rec[k] = v
        records.append(rec)

    jsonl = dest / "veri_emergency.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {"dataset": "Dasool/VERI-Emergency", "split": "train",
            "num_samples": len(records)}
    meta_file.write_text(json.dumps(meta, indent=2))
    print(f"  [OK] VERI-Emergency: {len(records)} samples → {jsonl}")
    return True


def download_geobench(benchmarks_dir: Path) -> bool:
    from datasets import load_dataset

    dest = benchmarks_dir / "geobench_vlm"
    dest.mkdir(parents=True, exist_ok=True)
    meta_file = dest / "_meta.json"

    print("  Downloading aialliance/GEOBench-VLM (split=single) ...")
    ds = load_dataset("aialliance/GEOBench-VLM", split="single")

    images_dir = dest / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    tasks_seen = set()
    for i, row in enumerate(ds):
        rec = {}
        for k, v in row.items():
            if k == "image" and v is not None:
                fname = f"geobench_{i:05d}.png"
                v.save(images_dir / fname)
                rec["image_path"] = str(images_dir / fname)
            else:
                rec[k] = v
        records.append(rec)
        if "task" in rec:
            tasks_seen.add(rec["task"])

    jsonl = dest / "geobench_single.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {"dataset": "aialliance/GEOBench-VLM", "split": "single",
            "num_samples": len(records), "tasks": sorted(tasks_seen)}
    meta_file.write_text(json.dumps(meta, indent=2))
    print(f"  [OK] GEO-Bench-VLM: {len(records)} samples, {len(tasks_seen)} tasks → {jsonl}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download/prepare benchmark data for EmberNet risk evaluation.")
    parser.add_argument("--benchmarks-dir", type=str, default="benchmarks")
    parser.add_argument("--skip-veri", action="store_true")
    parser.add_argument("--skip-geobench", action="store_true")
    parser.add_argument("--skip-robots", action="store_true")
    args = parser.parse_args()

    benchmarks_dir = Path(args.benchmarks_dir)
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  EmberNet Risk Benchmark — Data Preparation")
    print(f"{'='*60}\n")

    ok = True

    if not args.skip_robots:
        print("[1/3] Verifying robot-selection dataset ...")
        ok &= verify_robot_selection(benchmarks_dir)
    else:
        print("[1/3] Skipping robot-selection (--skip-robots)")

    if not args.skip_veri:
        _require_datasets()
        print("\n[2/3] VERI-Emergency ...")
        ok &= download_veri(benchmarks_dir)
    else:
        print("\n[2/3] Skipping VERI-Emergency (--skip-veri)")

    if not args.skip_geobench:
        _require_datasets()
        print("\n[3/3] GEO-Bench-VLM ...")
        ok &= download_geobench(benchmarks_dir)
    else:
        print("\n[3/3] Skipping GEO-Bench-VLM (--skip-geobench)")

    print(f"\n{'='*60}")
    if ok:
        print("  All datasets ready.")
    else:
        print("  Some datasets had issues — check messages above.")
    print(f"  Data location: {benchmarks_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
