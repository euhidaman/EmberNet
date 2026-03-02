"""
Download all external benchmark datasets into benchmarks/.

Trial download (fast sanity check — ~20 samples each):
    python benchmarks/download_benchmarks.py --mode trial --output-dir benchmarks

Full download (complete datasets):
    python benchmarks/download_benchmarks.py --mode full --output-dir benchmarks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _ensure_datasets_lib():
    try:
        import datasets  # noqa: F401
    except ImportError:
        sys.exit("ERROR: `datasets` library not found. Run: pip install datasets")


def download_geobench(output_dir: Path, trial: bool = False):
    """Download aialliance/GEOBench-VLM into output_dir/geobench_vlm/."""
    from datasets import load_dataset

    dest = output_dir / "geobench_vlm"
    dest.mkdir(parents=True, exist_ok=True)

    meta_file = dest / "_download_meta.json"
    print(f"\n{'='*60}")
    print(f"  GEOBench-VLM  ({'trial' if trial else 'full'} mode)")
    print(f"{'='*60}")

    # GEOBench-VLM has a single config 'default' with split 'single'
    split = "single"
    if trial:
        ds = load_dataset("aialliance/GEOBench-VLM", split=f"{split}[:30]")
    else:
        ds = load_dataset("aialliance/GEOBench-VLM", split=split)

    out_path = dest / "single.jsonl"
    images_dir = dest / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, row in enumerate(ds):
        rec = {}
        for k, v in row.items():
            if k == "image" and v is not None:
                fname = f"geobench_{i:05d}.png"
                v.save(images_dir / fname)
                rec["image_path"] = f"images/{fname}"
            else:
                rec[k] = v
        records.append(rec)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {"dataset": "aialliance/GEOBench-VLM", "split": split,
            "num_samples": len(records), "trial": trial}
    meta_file.write_text(json.dumps(meta, indent=2))
    print(f"  Saved {len(records)} samples → {out_path}")
    print(f"  Images → {images_dir}/")


def download_veri(output_dir: Path, trial: bool = False):
    """Download Dasool/VERI-Emergency into output_dir/veri_emergency/."""
    from datasets import load_dataset

    dest = output_dir / "veri_emergency"
    dest.mkdir(parents=True, exist_ok=True)

    meta_file = dest / "_download_meta.json"
    print(f"\n{'='*60}")
    print(f"  VERI-Emergency  ({'trial' if trial else 'full'} mode)")
    print(f"{'='*60}")

    if trial:
        ds = load_dataset("Dasool/VERI-Emergency", split="train[:20]")
    else:
        ds = load_dataset("Dasool/VERI-Emergency", split="train")

    out_path = dest / "veri_emergency.jsonl"
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
                rec["image_path"] = f"images/{fname}"
            else:
                rec[k] = v
        records.append(rec)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {"dataset": "Dasool/VERI-Emergency", "split": "train",
            "num_samples": len(records), "trial": trial}
    meta_file.write_text(json.dumps(meta, indent=2))
    print(f"  Saved {len(records)} samples → {out_path}")
    print(f"  Images → {images_dir}/")


def verify_robot_selection(output_dir: Path):
    """Verify the local robot-selection-dataset exists in benchmarks/."""
    rs_dir = output_dir / "robot-selection-dataset"
    multi = rs_dir / "multi_robot_selection_dataset.json"
    if not multi.exists():
        print(f"\n  WARNING: Robot-selection dataset not found at {multi}")
        print(f"  Expected location: benchmarks/robot-selection-dataset/")
        return
    data = json.loads(multi.read_text(encoding="utf-8"))
    print(f"\n  Robot-selection dataset: {len(data)} scenarios (OK)")


def main():
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets for EmberNet risk evaluation.")
    parser.add_argument("--mode", choices=["trial", "full"], default="trial",
                        help="trial = small subset for testing; full = complete datasets")
    parser.add_argument("--output-dir", type=str, default="benchmarks",
                        help="Root directory for all benchmarks (default: benchmarks)")
    args = parser.parse_args()

    _ensure_datasets_lib()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial = args.mode == "trial"

    print(f"Benchmark download — mode={args.mode}, output={output_dir.resolve()}")

    verify_robot_selection(output_dir)
    download_geobench(output_dir, trial=trial)
    download_veri(output_dir, trial=trial)

    print(f"\n{'='*60}")
    print("  All downloads complete.")
    print(f"  Data location: {output_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
