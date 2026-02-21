#!/usr/bin/env python3
"""
eval/run_eval.py
================
Convenience launcher for evaluating EmberNet on lmms-eval benchmarks.

SETUP (one-time, on your training server):
    git clone https://github.com/EvolvingLMMs-Lab/lmms-eval ~/lmms-eval
    cd ~/lmms-eval && pip install -e ".[all]"
    export HF_TOKEN="hf_xxx"          # needed if tasks auto-download datasets

QUICKSTART:
    # Evaluate the latest trial checkpoint on core benchmarks
    python eval/run_eval.py --checkpoint ./checkpoints/trial/stage2/final_model.pt

    # Full suite matching all 8 expert domains
    python eval/run_eval.py --checkpoint ./checkpoints/main/stage2/final_model.pt --suite full

    # Single benchmark
    python eval/run_eval.py --checkpoint ./checkpoints/trial/stage2/final_model.pt --tasks chartqa

    # From HF Hub (after push-per-epoch has run)
    python eval/run_eval.py --hub euhidaman/EmberNet-Trial --tasks mme,textvqa

    # Dry-run: print the command without running
    python eval/run_eval.py --checkpoint ... --dry-run
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------

# Core = a fast subset that exercises all 8 expert domains + general capability
SUITE_CORE = [
    "textvqa",          # E0: vision_ocr
    "ai2d",             # E1: vision_diagram
    "chartqa",          # E2: code_math_chart
    "mathvista",        # E3: code_math_formula
    "vqav2",            # E4: spatial_scene
    "gqa",              # E5: spatial_reasoning
    "ok_vqa",           # E6: agentic_knowledge
    "scienceqa_img",    # E7: agentic_reasoning
    "mme",              # general VLM capability
]

# Full = core + additional per-domain + ranking benchmarks
SUITE_FULL = SUITE_CORE + [
    "docvqa_val",       # E0: vision_ocr (harder)
    "mmstar",           # general reasoning (hard)
    "mmmu_val",         # general (college-level multi-discipline)
    "seed_bench",       # general scene understanding
]

# Quick = fastest sanity-check (3–5 min on a single GPU)
SUITE_QUICK = [
    "mme",
    "textvqa",
    "scienceqa_img",
]

SUITES = {
    "quick":  SUITE_QUICK,
    "core":   SUITE_CORE,
    "full":   SUITE_FULL,
}

# ---------------------------------------------------------------------------
# Expert domain annotation (for logging)
# ---------------------------------------------------------------------------
TASK_EXPERT = {
    "textvqa":       "E0: vision_ocr",
    "docvqa_val":    "E0: vision_ocr",
    "ai2d":          "E1: vision_diagram",
    "chartqa":       "E2: code_math_chart",
    "mathvista":     "E3: code_math_formula",
    "vqav2":         "E4: spatial_scene",
    "gqa":           "E5: spatial_reasoning",
    "ok_vqa":        "E6: agentic_knowledge",
    "scienceqa_img": "E7: agentic_reasoning",
    "mme":           "General",
    "mmstar":        "General",
    "mmmu_val":      "General",
    "seed_bench":    "General",
}


def build_command(args, tasks: list[str]) -> list[str]:
    """Build the lmms_eval CLI command list."""
    repo_root = Path(__file__).resolve().parent.parent

    # model_args
    if args.hub:
        pretrained = args.hub
    elif args.checkpoint:
        pretrained = str(Path(args.checkpoint).resolve())
    else:
        raise ValueError("Provide --checkpoint or --hub")

    model_args = f"pretrained={pretrained}"
    if args.device:
        model_args += f",device={args.device}"
    if args.dtype:
        model_args += f",dtype={args.dtype}"
    if args.max_new_tokens:
        model_args += f",max_new_tokens={args.max_new_tokens}"

    # output path
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    suffix = Path(pretrained).stem if not args.hub else args.hub.replace("/", "_")
    out_dir = Path(args.output_dir) / f"embernet_{suffix}_{ts}"

    cmd = [
        sys.executable, "-m", "lmms_eval",
        "--model",         "embernet",
        "--model_args",    model_args,
        "--tasks",         ",".join(tasks),
        "--batch_size",    "1",
        "--include-path",  str(repo_root / "eval"),
        "--output_path",   str(out_dir),
        "--log_samples",
    ]

    if args.limit:
        cmd += ["--limit", str(args.limit)]

    if args.num_fewshot is not None:
        cmd += ["--num_fewshot", str(args.num_fewshot)]

    return cmd, out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run EmberNet evaluation on lmms-eval benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Source
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--checkpoint", type=str,
                     help="Path to local checkpoint .pt file")
    src.add_argument("--hub",        type=str,
                     help="HuggingFace Hub repo ID, e.g. euhidaman/EmberNet-Trial")

    # Benchmark selection
    bench = parser.add_mutually_exclusive_group()
    bench.add_argument("--suite", choices=["quick", "core", "full"], default="core",
                       help="Pre-defined benchmark suite (default: core)")
    bench.add_argument("--tasks", type=str,
                       help="Comma-separated list of lmms-eval task names")

    # Hardware
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu / auto (default: auto-select)")
    parser.add_argument("--dtype",  type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])

    # Generation
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=256)

    # Eval options
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per task (useful for quick testing)")
    parser.add_argument("--num-fewshot", dest="num_fewshot", type=int, default=None,
                        help="Number of few-shot examples (default: task default)")

    # Output
    parser.add_argument("--output-dir", dest="output_dir", type=str,
                        default="./eval_results",
                        help="Directory to write JSON results (default: ./eval_results)")

    parser.add_argument("--dry-run", action="store_true",
                        help="Print the command without running it")

    args = parser.parse_args()

    if not args.checkpoint and not args.hub:
        parser.error("Provide --checkpoint <path> or --hub <repo>")

    # Resolve tasks
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = SUITES[args.suite]

    # Print plan
    print("\n" + "=" * 64)
    print("  EmberNet — lmms-eval evaluation plan")
    print("=" * 64)
    src_str = args.hub or args.checkpoint
    print(f"  Model   : {src_str}")
    print(f"  Tasks   :")
    for t in tasks:
        expert = TASK_EXPERT.get(t, "?")
        print(f"            {t:<22}  [{expert}]")
    print(f"  Device  : {args.device or 'auto'}")
    print(f"  dtype   : {args.dtype}")
    if args.limit:
        print(f"  Limit   : {args.limit} samples per task")
    print("=" * 64 + "\n")

    cmd, out_dir = build_command(args, tasks)

    print("Command:")
    print("  " + " \\\n    ".join(cmd))
    print()

    if args.dry_run:
        print("[dry-run] Not executing.")
        return

    # Check lmms_eval is importable
    try:
        import lmms_eval  # noqa: F401
    except ImportError:
        print("ERROR: lmms-eval is not installed.\n")
        print("Install it with:")
        print("  git clone https://github.com/EvolvingLMMs-Lab/lmms-eval ~/lmms-eval")
        print("  cd ~/lmms-eval && pip install -e '.[all]'")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be written to: {out_dir.resolve()}\n")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
