#!/usr/bin/env python3
"""
eval/eval_veri_emergency.py
===========================
Evaluate EmberNet on the VERI-Emergency benchmark.

Dataset: Dasool/VERI-Emergency (200 images, 100 contrastive pairs)
Labels: "danger" vs "safe"

Metrics:
  - Accuracy
  - Confusion matrix (danger=positive)
  - Overreaction rate  (safe predicted as danger)
  - Underreaction rate (danger predicted as safe)

Usage:
    python eval/eval_veri_emergency.py --model checkpoints/stage2/final_model.pt
    python eval/eval_veri_emergency.py --model checkpoints/stage2/final_model.pt --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

PROMPT = (
    "Look at this image carefully. Is this scene showing a dangerous emergency "
    "situation, or is it a safe situation? Answer with exactly one word: "
    "danger or safe."
)


def load_veri(benchmarks_dir: str, split: str = "train", limit: int | None = None):
    """Load VERI-Emergency. Try local JSONL first, fall back to HF datasets."""
    local_jsonl = Path(benchmarks_dir) / "veri_emergency" / "veri_emergency.jsonl"
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

    print(f"  Local JSONL not found, loading from HuggingFace ...")
    from datasets import load_dataset
    if limit:
        ds = load_dataset("Dasool/VERI-Emergency", split=f"{split}[:{limit}]")
    else:
        ds = load_dataset("Dasool/VERI-Emergency", split=split)
    return list(ds)


def classify_response(text: str) -> str:
    """Map model output to 'danger' or 'safe'."""
    t = text.lower().strip()
    # Check for explicit keywords
    if "danger" in t or "emergency" in t or "hazard" in t or "unsafe" in t:
        return "danger"
    if "safe" in t or "normal" in t or "no danger" in t or "not dangerous" in t:
        return "safe"
    # First word heuristic
    first = t.split()[0] if t.split() else ""
    if first.startswith("danger"):
        return "danger"
    if first.startswith("safe"):
        return "safe"
    return "unknown"


def compute_metrics(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"num_examples": 0}

    # Confusion matrix: danger = positive
    tp = fp = tn = fn = 0
    correct = 0
    for r in results:
        gt = r["gold"]
        pred = r["pred"]
        if gt == pred:
            correct += 1
        if gt == "danger" and pred == "danger":
            tp += 1
        elif gt == "safe" and pred == "danger":
            fp += 1
        elif gt == "safe" and pred == "safe":
            tn += 1
        elif gt == "danger" and pred == "safe":
            fn += 1

    n_safe = fp + tn
    n_danger = tp + fn

    accuracy = correct / n if n > 0 else 0.0
    overreaction = fp / n_safe if n_safe > 0 else 0.0   # safe → danger
    underreaction = fn / n_danger if n_danger > 0 else 0.0  # danger → safe

    return {
        "num_examples": n,
        "accuracy": round(accuracy, 4),
        "overreaction_rate": round(overreaction, 4),
        "underreaction_rate": round(underreaction, 4),
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "num_danger": n_danger,
        "num_safe": n_safe,
        "num_unknown_pred": sum(1 for r in results if r["pred"] == "unknown"),
    }


def _resolve_image(row, benchmarks_dir):
    """Resolve image_path relative to project root, or return PIL image."""
    img = row.get("image_path")
    if img and isinstance(img, str):
        p = Path(img)
        if not p.is_absolute():
            # Try relative to project root (parent of benchmarks dir)
            project_root = Path(benchmarks_dir).parent
            p = project_root / img
        if p.exists():
            return str(p)
    return row.get("image")


def run(args, model=None):
    data = load_veri(args.benchmarks_dir, split=args.split, limit=args.limit)
    print(f"  Loaded {len(data)} VERI-Emergency samples")

    sys.path.insert(0, str(Path(__file__).parent.parent))
    if model is None:
        from inference import EmberVLM
        model = EmberVLM(model_path=args.model, device=args.device)

    results = []
    for row in tqdm(data, desc="  VERI-Emergency", unit="sample"):
        # Gold label
        gold = row.get("risk_identification", "").strip().lower()
        if gold not in ("danger", "safe"):
            continue

        # Image — resolve relative path
        image = _resolve_image(row, args.benchmarks_dir)
        if image is None:
            continue

        # Inference
        response = model.answer(image=image, question=PROMPT)
        pred = classify_response(response)

        results.append({"gold": gold, "pred": pred, "raw": response,
                        "image_id": row.get("image_id", "")})

    metrics = compute_metrics(results)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "veri_emergency.json"
    out_file.write_text(json.dumps(metrics, indent=2))

    print(f"\n{'='*50}")
    print(f"  VERI-Emergency Results")
    print(f"{'='*50}")
    print(f"  Examples:          {metrics['num_examples']}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Overreaction:      {metrics['overreaction_rate']:.4f}  (safe→danger)")
    print(f"  Underreaction:     {metrics['underreaction_rate']:.4f}  (danger→safe)")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion:  TP={cm['TP']}  FP={cm['FP']}  TN={cm['TN']}  FN={cm['FN']}")
    print(f"  Unknown predictions: {metrics['num_unknown_pred']}")
    print(f"{'='*50}")
    print(f"  Saved → {out_file}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate EmberNet on VERI-Emergency")
    parser.add_argument("--model", type=str, default="checkpoints/stage2/final_model.pt")
    parser.add_argument("--benchmarks-dir", type=str, default="benchmarks")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
