#!/usr/bin/env python3
"""
eval/eval_geobench_vlm.py
==========================
Evaluate EmberNet on GEO-Bench-VLM (geospatial MCQ benchmark).

Dataset: aialliance/GEOBench-VLM  (split=single, ~3211 MCQ samples)
All questions are multiple-choice with options A–E.

Metrics:
  - Overall accuracy
  - Per-task accuracy
  - Per-category accuracy (derived from task groupings)

Usage:
    python eval/eval_geobench_vlm.py --model checkpoints/stage2/final_model.pt
    python eval/eval_geobench_vlm.py --model checkpoints/stage2/final_model.pt --limit 100
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

_BLANK_IMAGE = Image.new("RGB", (224, 224), (255, 255, 255))

# Task → high-level category mapping based on the GEO-Bench-VLM paper
TASK_CATEGORY = {
    "Ship Type Classification": "Object Classification",
    "Aircraft Type Classification": "Object Classification",
    "Vehicle Type Classification": "Object Classification",
    "Building Type Classification": "Object Classification",
    "Bridge Type Classification": "Object Classification",
    "Land Use Classification": "Scene Classification",
    "Scene Classification": "Scene Classification",
    "Cloud Type Classification": "Scene Classification",
    "Object Counting": "Counting",
    "Ship Counting": "Counting",
    "Vehicle Counting": "Counting",
    "Building Counting": "Counting",
    "Localization": "Localization",
    "Object Localization": "Localization",
    "Fine-Grained Categorization": "Fine-Grained",
    "Change Detection": "Temporal",
    "Temporal Analysis": "Temporal",
    "Damage Assessment": "Risk/Disaster",
    "Disaster Assessment": "Risk/Disaster",
    "Flood Detection": "Risk/Disaster",
}

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def load_geobench(benchmarks_dir: str, hf_dataset: str, split: str,
                   limit: int | None = None):
    """Load GEO-Bench-VLM. Try local JSONL first, fall back to HF."""
    local_jsonl = Path(benchmarks_dir) / "geobench_vlm" / "geobench_single.jsonl"
    if local_jsonl.exists():
        print(f"  Loading from local JSONL: {local_jsonl}")
        records = []
        with open(local_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if limit:
            records = records[:limit]
        return records

    print(f"  Loading from HuggingFace: {hf_dataset} split={split} ...")
    from datasets import load_dataset
    if limit:
        ds = load_dataset(hf_dataset, split=f"{split}[:{limit}]")
    else:
        ds = load_dataset(hf_dataset, split=split)
    return list(ds)


def build_prompt(row: dict) -> str:
    """Build MCQ prompt from a GEO-Bench-VLM row."""
    prompts = row.get("prompts", [])
    question = prompts[0] if prompts else "What do you see in this image?"
    options_str = row.get("options", "")
    return (
        f"{question}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Answer with only the option letter (e.g. A, B, C, D, or E)."
    )


def parse_option_letter(text: str, options_list: list[str] | None = None) -> str:
    """Extract the predicted option letter from model output."""
    text = text.strip()

    # Direct single-letter match
    if len(text) == 1 and text.upper() in OPTION_LETTERS:
        return text.upper()

    # First character is a letter followed by punctuation/space
    m = re.match(r"^([A-Ha-h])[.\s:)\-]", text)
    if m:
        return m.group(1).upper()

    # Look for "answer is X" or "option X" patterns
    m = re.search(r"(?:answer|option)\s*(?:is\s*)?([A-Ha-h])\b", text, re.I)
    if m:
        return m.group(1).upper()

    # Match against option text if available
    if options_list:
        text_lower = text.lower().strip()
        for idx, opt in enumerate(options_list):
            if opt.lower().strip() == text_lower and idx < len(OPTION_LETTERS):
                return OPTION_LETTERS[idx]

    # Last resort: first capital letter in A-E range
    for ch in text:
        if ch in "ABCDE":
            return ch

    return ""


def get_category(task: str) -> str:
    """Map task name to category, with fallback."""
    if task in TASK_CATEGORY:
        return TASK_CATEGORY[task]
    task_lower = task.lower()
    for key, cat in TASK_CATEGORY.items():
        if key.lower() in task_lower or task_lower in key.lower():
            return cat
    return "Other"


def compute_metrics(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"num_examples": 0}

    correct = sum(1 for r in results if r["correct"])

    # Per-task
    task_correct = defaultdict(int)
    task_total = defaultdict(int)
    for r in results:
        t = r["task"]
        task_total[t] += 1
        if r["correct"]:
            task_correct[t] += 1

    per_task = {}
    for t in sorted(task_total.keys()):
        acc = task_correct[t] / task_total[t] if task_total[t] > 0 else 0.0
        per_task[t] = {"accuracy": round(acc, 4), "count": task_total[t],
                        "correct": task_correct[t]}

    # Per-category
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    for r in results:
        cat = r["category"]
        cat_total[cat] += 1
        if r["correct"]:
            cat_correct[cat] += 1

    per_category = {}
    for c in sorted(cat_total.keys()):
        acc = cat_correct[c] / cat_total[c] if cat_total[c] > 0 else 0.0
        per_category[c] = {"accuracy": round(acc, 4), "count": cat_total[c],
                            "correct": cat_correct[c]}

    return {
        "num_examples": n,
        "overall_accuracy": round(correct / n, 4),
        "per_task": per_task,
        "per_category": per_category,
    }


def _resolve_image(row, benchmarks_dir):
    """Resolve image_path relative to project root, or return PIL image."""
    img = row.get("image_path")
    if img and isinstance(img, str):
        p = Path(img)
        if not p.is_absolute():
            project_root = Path(benchmarks_dir).parent
            p = project_root / img
        if p.exists():
            return str(p)
    return row.get("image")


def run(args, model=None):
    data = load_geobench(args.benchmarks_dir, args.hf_dataset, args.split,
                          limit=args.limit)
    print(f"  Loaded {len(data)} GEO-Bench-VLM samples")

    sys.path.insert(0, str(Path(__file__).parent.parent))
    if model is None:
        from inference import EmberVLM
        model = EmberVLM(model_path=args.model, device=args.device)

    results = []
    for i, row in enumerate(tqdm(data, desc="  GEO-Bench-VLM", unit="sample")):
        task = row.get("task", "unknown")
        gt_option = row.get("ground_truth_option", "").strip().upper()
        options_list = row.get("options_list", [])
        image = _resolve_image(row, args.benchmarks_dir)

        prompt = build_prompt(row)

        if image is not None:
            response = model.chat(
                image=image, prompt=prompt, reset=True, max_tokens=16,
            )
        else:
            response = model.chat(
                image=_BLANK_IMAGE, prompt=prompt, reset=True, max_tokens=16,
            )

        if i < 3:
            print(f"  [debug] geo sample {i} raw: {response[:200]!r}")

        pred_letter = parse_option_letter(response, options_list)
        if not pred_letter:
            n_opts = len(options_list) if options_list else 5
            pred_letter = random.choice(OPTION_LETTERS[:n_opts])
        is_correct = pred_letter == gt_option

        results.append({
            "task": task,
            "category": get_category(task),
            "gold": gt_option,
            "pred": pred_letter,
            "correct": is_correct,
            "raw": response,
        })

    metrics = compute_metrics(results)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "geobench_vlm.json"
    out_file.write_text(json.dumps(metrics, indent=2))

    # Print table
    print(f"\n{'='*60}")
    print(f"  GEO-Bench-VLM Results")
    print(f"{'='*60}")
    print(f"  Examples: {metrics['num_examples']}")
    print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"\n  {'Task':<40} {'Acc':>6}  {'N':>5}")
    print(f"  {'-'*55}")
    for t, m in metrics["per_task"].items():
        print(f"  {t:<40} {m['accuracy']:>6.4f}  {m['count']:>5}")
    print(f"\n  {'Category':<30} {'Acc':>6}  {'N':>5}")
    print(f"  {'-'*45}")
    for c, m in metrics["per_category"].items():
        print(f"  {c:<30} {m['accuracy']:>6.4f}  {m['count']:>5}")
    print(f"{'='*60}")
    print(f"  Saved → {out_file}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate EmberNet on GEO-Bench-VLM")
    parser.add_argument("--model", type=str, default="checkpoints/stage2/final_model.pt")
    parser.add_argument("--benchmarks-dir", type=str, default="benchmarks")
    parser.add_argument("--hf-dataset", type=str, default="aialliance/GEOBench-VLM")
    parser.add_argument("--split", type=str, default="single")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
