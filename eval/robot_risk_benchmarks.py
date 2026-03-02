"""
Robot Risk-Aware Benchmarking Suite for EmberNet.

Evaluates EmberNet on three benchmarks that test environmental-risk understanding
and robot-deployment decision-making:

  1. Robot-Selection Dataset   — multi-robot selection in hazardous scenarios
  2. GEOBench-VLM              — geospatial disaster / hazard assessment
  3. VERI-Emergency            — emergency vs safe scene discrimination

Trial run (sanity check, ~20 samples per benchmark):

    python eval/robot_risk_benchmarks.py \\
        --model-path ./checkpoints/stage2/final_model.pt \\
        --benchmarks-dir ./benchmarks \\
        --top-n 3 --limit 20 --trial

Full evaluation:

    python eval/robot_risk_benchmarks.py \\
        --model-path ./checkpoints/stage2/final_model.pt \\
        --benchmarks-dir ./benchmarks \\
        --top-n 3 --main

Results are saved under benchmarks/results/ and plots under
benchmarks/plots/robot_risk_benchmarks/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

KNOWN_ROBOTS = [
    "Drone",
    "Humanoid",
    "Robot with Legs",
    "Robot with Wheels",
    "Underwater Robot",
]

RISK_KEYWORDS_DANGEROUS = {
    "fire", "flame", "smoke", "explosion", "hazardous", "toxic", "radiation",
    "flood", "earthquake", "collapse", "debris", "unstable", "chemical",
    "biohazard", "nuclear", "volcano", "lava", "storm", "hurricane",
    "tornado", "lightning", "avalanche", "landslide", "sinkhole", "wreckage",
    "contamination", "gas leak", "high voltage", "electrocution", "extreme heat",
    "extreme cold", "blizzard", "war zone", "minefield", "active shooter",
    "bomb", "explosive", "structural damage", "cave-in", "underwater",
    "deep sea", "ocean floor", "submerged", "drowning",
}

RISK_KEYWORDS_MODERATE = {
    "inspect", "survey", "monitor", "patrol", "search", "rescue",
    "damaged", "disaster", "emergency", "hazard", "risk", "danger",
    "rubble", "wreck", "ruin", "aftermath",
}

# GEOBench risk-related tasks (substring matches against the 'task' field)
GEOBENCH_RISK_TASKS = {
    "damage": "high",
    "disaster": "high",
    "flood": "high",
    "fire": "high",
    "destruction": "high",
    "change detection": "medium",
    "building": "medium",
    "infrastructure": "medium",
    "urban": "low",
    "land use": "low",
    "ship": "low",
    "aircraft": "low",
    "vehicle": "low",
}


# ── Unified sample representation ────────────────────────────────────────────


@dataclass
class RiskSample:
    sample_id: str
    dataset_name: str
    task_name: str = ""
    image_path: Optional[str] = None
    context_text: str = ""
    risk_label: str = "unknown"           # safe | danger | unknown
    risk_severity: str = "unknown"        # low | medium | high | unknown
    robot_selection_gt: List[str] = field(default_factory=list)
    no_deployment_gt: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Predictions (filled after inference)
    pred_risk_label: str = ""
    pred_risk_severity: str = ""
    pred_selected_robots: List[str] = field(default_factory=list)
    pred_no_deployment: bool = False
    pred_explanation: str = ""
    pred_raw: str = ""
    parse_valid: bool = True


# ── Dataset loaders ──────────────────────────────────────────────────────────


def _infer_risk_from_text(text: str) -> Tuple[str, str]:
    """Heuristic: derive risk_label and risk_severity from scenario text."""
    lower = text.lower()
    has_dangerous = any(k in lower for k in RISK_KEYWORDS_DANGEROUS)
    has_moderate = any(k in lower for k in RISK_KEYWORDS_MODERATE)
    if has_dangerous:
        return "danger", "high"
    if has_moderate:
        return "danger", "medium"
    return "safe", "low"


def _score_robot_scenario(robot: str, task_text: str) -> float:
    """Deterministic scoring: how suitable is *robot* for *task_text*."""
    lower = task_text.lower()
    s = 0.0

    ROBOT_STRENGTHS = {
        "Drone": {
            "aerial": 3, "inspect": 2, "survey": 2, "monitor": 2,
            "high-rise": 3, "roof": 3, "bridge": 2, "overhead": 3,
            "surveillance": 3, "search": 2, "outdoor": 2, "fire": 1,
            "flood": 2, "disaster": 2, "large area": 3, "mapping": 3,
        },
        "Humanoid": {
            "indoor": 3, "manipulation": 3, "tool": 3, "human": 2,
            "stairs": 2, "door": 2, "complex": 2, "assemble": 3,
            "operate": 3, "interact": 3, "hospital": 2, "office": 2,
        },
        "Robot with Legs": {
            "rough terrain": 3, "rubble": 3, "stairs": 2, "search and rescue": 3,
            "mountain": 3, "uneven": 3, "outdoor": 2, "disaster": 2,
            "inspect": 2, "industrial": 2, "forest": 2, "load": 2,
        },
        "Robot with Wheels": {
            "warehouse": 3, "flat": 3, "indoor": 2, "road": 3,
            "payload": 3, "transport": 3, "delivery": 3, "patrol": 2,
            "factory": 3, "hospital": 2, "efficient": 2, "fast": 2,
        },
        "Underwater Robot": {
            "underwater": 3, "marine": 3, "ocean": 3, "sea": 3,
            "pipe": 2, "pool": 2, "submerged": 3, "dam": 2,
            "reef": 2, "flood": 1, "deep": 3, "dive": 3,
        },
    }

    ROBOT_WEAKNESSES = {
        "Drone": {"indoor": -1, "manipulation": -2, "heavy": -2, "payload": -2},
        "Humanoid": {"fast": -1, "aerial": -2, "underwater": -2, "rough terrain": -1},
        "Robot with Legs": {"aerial": -2, "underwater": -2, "manipulation": -1, "flat": -1},
        "Robot with Wheels": {"stairs": -2, "rubble": -2, "rough": -2, "underwater": -2,
                              "aerial": -2, "uneven": -2},
        "Underwater Robot": {"land": -2, "indoor": -2, "aerial": -2, "stairs": -2,
                             "road": -2, "warehouse": -2, "factory": -2},
    }

    for kw, bonus in ROBOT_STRENGTHS.get(robot, {}).items():
        if kw in lower:
            s += bonus
    for kw, penalty in ROBOT_WEAKNESSES.get(robot, {}).items():
        if kw in lower:
            s += penalty
    return s


def _rank_robots(task_text: str, top_n: int = 3) -> List[str]:
    """Return top-N robots ranked by suitability for the scenario."""
    scores = {r: _score_robot_scenario(r, task_text) for r in KNOWN_ROBOTS}
    ranked = sorted(scores, key=lambda r: scores[r], reverse=True)
    return ranked[:top_n]


def _should_not_deploy(task_text: str) -> bool:
    """Heuristic: is the scenario so dangerous that no robot should be sent?"""
    lower = task_text.lower()
    extreme = {"nuclear meltdown", "active minefield", "active volcano eruption",
               "chemical weapons", "nerve agent", "radiation leak critical"}
    return any(k in lower for k in extreme)


def load_robot_selection(benchmarks_dir: Path, limit: Optional[int] = None,
                         top_n: int = 3) -> List[RiskSample]:
    """Load the local robot-selection dataset."""
    rs_dir = benchmarks_dir / "robot-selection-dataset"
    multi_file = rs_dir / "multi_robot_selection_dataset.json"
    if not multi_file.exists():
        print(f"  [SKIP] Robot-selection dataset not found at {multi_file}")
        print(f"         Expected: benchmarks/robot-selection-dataset/multi_robot_selection_dataset.json")
        return []

    data = json.loads(multi_file.read_text(encoding="utf-8"))
    samples = []
    for i, entry in enumerate(data):
        if limit and i >= limit:
            break
        task_text = entry.get("input", "")
        risk_label, risk_severity = _infer_risk_from_text(task_text)

        # Ground-truth robots from subtasks
        gt_robots = []
        for st in entry.get("subtasks", []):
            r = st.get("assigned_robot", "")
            if r and r not in gt_robots:
                gt_robots.append(r)

        # Also use original_single_robot_output
        orig = entry.get("original_single_robot_output", "")
        if isinstance(orig, list):
            for r in orig:
                if r not in gt_robots:
                    gt_robots.append(r)
        elif isinstance(orig, str) and orig:
            for r in [x.strip() for x in orig.split(",")]:
                if r and r not in gt_robots:
                    gt_robots.append(r)

        scored_ranking = _rank_robots(task_text, top_n=top_n)

        samples.append(RiskSample(
            sample_id=f"rs_{i:04d}",
            dataset_name="robot_selection",
            task_name="multi_robot_selection",
            context_text=task_text,
            risk_label=risk_label,
            risk_severity=risk_severity,
            robot_selection_gt=gt_robots if gt_robots else scored_ranking,
            no_deployment_gt=_should_not_deploy(task_text),
            metadata={
                "subtasks": entry.get("subtasks", []),
                "scored_ranking": scored_ranking,
            },
        ))

    print(f"  Robot-selection: loaded {len(samples)} scenarios")
    return samples


def load_geobench(benchmarks_dir: Path, limit: Optional[int] = None) -> List[RiskSample]:
    """Load GEOBench-VLM from benchmarks/geobench_vlm/."""
    geo_dir = benchmarks_dir / "geobench_vlm"
    jsonl = geo_dir / "single.jsonl"
    if not jsonl.exists():
        print(f"  [SKIP] GEOBench-VLM not found at {jsonl}")
        print(f"         Run: python benchmarks/download_benchmarks.py --mode trial --output-dir benchmarks")
        return []

    samples = []
    with open(jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            row = json.loads(line)
            task = row.get("task", "unknown")

            # Derive risk from task name
            risk_label, risk_severity = "safe", "low"
            task_lower = task.lower()
            for kw, sev in GEOBENCH_RISK_TASKS.items():
                if kw in task_lower:
                    risk_label = "danger" if sev in ("high", "medium") else "safe"
                    risk_severity = sev
                    break

            prompts = row.get("prompts", [])
            prompt_text = prompts[0] if prompts else ""
            options = row.get("options", "")
            gt = row.get("ground_truth", "")

            context = f"{prompt_text}\n{options}".strip() if options else prompt_text

            img_path = row.get("image_path", "")
            if img_path:
                full_img = str(geo_dir / img_path)
            else:
                full_img = None

            samples.append(RiskSample(
                sample_id=f"geo_{i:05d}",
                dataset_name="geobench_vlm",
                task_name=task,
                image_path=full_img,
                context_text=context,
                risk_label=risk_label,
                risk_severity=risk_severity,
                metadata={
                    "ground_truth": gt,
                    "ground_truth_option": row.get("ground_truth_option", ""),
                    "options_list": row.get("options_list", []),
                },
            ))

    print(f"  GEOBench-VLM: loaded {len(samples)} samples")
    return samples


def load_veri(benchmarks_dir: Path, limit: Optional[int] = None) -> List[RiskSample]:
    """Load VERI-Emergency from benchmarks/veri_emergency/."""
    veri_dir = benchmarks_dir / "veri_emergency"
    jsonl = veri_dir / "veri_emergency.jsonl"
    if not jsonl.exists():
        print(f"  [SKIP] VERI-Emergency not found at {jsonl}")
        print(f"         Run: python benchmarks/download_benchmarks.py --mode trial --output-dir benchmarks")
        return []

    CATEGORY_SEVERITY = {"AB": "high", "PME": "high", "ND": "high"}

    samples = []
    with open(jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            row = json.loads(line)
            risk_id = row.get("risk_identification", "").strip().lower()
            risk_label = "danger" if risk_id == "danger" else "safe"
            category = row.get("category", "unknown")
            risk_severity = CATEGORY_SEVERITY.get(category, "medium") if risk_label == "danger" else "low"

            img_path = row.get("image_path", "")
            if img_path:
                full_img = str(veri_dir / img_path)
            else:
                full_img = None

            caption = row.get("caption", "")

            samples.append(RiskSample(
                sample_id=row.get("image_id", f"veri_{i:04d}"),
                dataset_name="veri_emergency",
                task_name="risk_identification",
                image_path=full_img,
                context_text=caption,
                risk_label=risk_label,
                risk_severity=risk_severity,
                metadata={
                    "category": category,
                    "emergency_response": row.get("emergency_response", ""),
                },
            ))

    print(f"  VERI-Emergency: loaded {len(samples)} samples")
    return samples


# ── Prompt templates ─────────────────────────────────────────────────────────

RISK_PROMPT_GENERAL = (
    "You are a robot operations coordinator deciding whether to deploy robots "
    "into real-world environments. Given the image and any accompanying text, "
    "you must:\n"
    "1) assess whether the situation is SAFE or DANGEROUS for robot deployment, and\n"
    "2) if dangerous, briefly describe the main risks.\n\n"
    'Respond in JSON with fields: "risk_label" (one of "safe", "danger"), '
    '"risk_severity" (one of "low", "medium", "high"), and '
    '"explanation" (short text).'
)

ROBOT_SELECTION_PROMPT = (
    "You are a robot operations coordinator. Below is a list of available robots:\n\n"
    "1. Drone — aerial surveillance, inspection, lightweight transport; limited payload, weather dependent\n"
    "2. Humanoid — manipulation, tool use, human interaction, stairs; slow, balance issues\n"
    "3. Robot with Legs — rough terrain, rubble, search and rescue, load carrying; limited manipulation\n"
    "4. Robot with Wheels — fast movement, payload, warehouses, roads; flat surfaces only\n"
    "5. Underwater Robot — underwater navigation, marine inspection; water environments only\n\n"
    "Given the scenario below, decide:\n"
    "(a) Is this SAFE or DANGEROUS for robot deployment?\n"
    "(b) Rate severity: low / medium / high.\n"
    "(c) Select up to {top_n} robots most suitable for this scenario, sorted best to worst.\n"
    "(d) If the scenario is too dangerous for any robot, set no_deployment to true.\n\n"
    "Scenario: {scenario}\n\n"
    'Respond ONLY in JSON: {{"risk_label": "safe"|"danger", "risk_severity": "low"|"medium"|"high", '
    '"explanation": "...", "selected_robots": ["Robot Name", ...], "no_deployment": true|false}}'
)

GEOBENCH_RISK_PROMPT = (
    "You are a robot operations coordinator assessing a satellite/aerial image for "
    "ground-level robot deployment safety.\n\n"
    "Task context: {task}\n"
    "Question: {question}\n\n"
    "Decide whether ground robots would face a DANGEROUS or SAFE situation in this "
    "location, based on disaster indicators, damage, or obstacles. Also estimate "
    "risk severity (low/medium/high).\n\n"
    'Respond ONLY in JSON: {{"risk_label": "safe"|"danger", "risk_severity": "low"|"medium"|"high", '
    '"explanation": "..."}}'
)

VERI_RISK_PROMPT = (
    "You are a robot operations coordinator evaluating whether this scene requires "
    "emergency robot intervention.\n\n"
    "IMPORTANT: Many images look dangerous but are actually safe, and vice versa. "
    "Be careful not to overreact. Look for genuine indicators of emergency.\n\n"
    "Scene description: {caption}\n\n"
    "Decide if this scene is a genuine emergency requiring robot intervention "
    '("danger") or not ("safe"). Provide a one-sentence explanation.\n\n'
    'Respond ONLY in JSON: {{"risk_label": "safe"|"danger", "risk_severity": "low"|"medium"|"high", '
    '"explanation": "..."}}'
)


def build_prompt(sample: RiskSample, top_n: int = 3) -> str:
    """Build the appropriate prompt for a sample based on its dataset."""
    if sample.dataset_name == "robot_selection":
        return ROBOT_SELECTION_PROMPT.format(
            top_n=top_n, scenario=sample.context_text)

    if sample.dataset_name == "geobench_vlm":
        return GEOBENCH_RISK_PROMPT.format(
            task=sample.task_name, question=sample.context_text)

    if sample.dataset_name == "veri_emergency":
        return VERI_RISK_PROMPT.format(caption=sample.context_text)

    return RISK_PROMPT_GENERAL + f"\n\nContext: {sample.context_text}"


# ── Output parsing ───────────────────────────────────────────────────────────


def _extract_json(text: str) -> Optional[dict]:
    """Try to extract JSON from model output."""
    # Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Try to find JSON block
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Try with nested braces
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _normalize_risk_label(val: str) -> str:
    val = val.strip().lower()
    if val in ("danger", "dangerous", "hazardous", "emergency", "unsafe"):
        return "danger"
    if val in ("safe", "ok", "normal", "benign", "no danger"):
        return "safe"
    return "unknown"


def _normalize_severity(val: str) -> str:
    val = val.strip().lower()
    if val in ("low", "l", "minor", "minimal"):
        return "low"
    if val in ("medium", "med", "m", "moderate"):
        return "medium"
    if val in ("high", "h", "severe", "critical", "extreme"):
        return "high"
    return "unknown"


def _normalize_robot_name(name: str) -> str:
    name = name.strip()
    for known in KNOWN_ROBOTS:
        if name.lower() == known.lower():
            return known
    for known in KNOWN_ROBOTS:
        if name.lower() in known.lower() or known.lower() in name.lower():
            return known
    return name


def parse_model_output(sample: RiskSample, raw_output: str):
    """Parse model output and fill prediction fields on the sample."""
    sample.pred_raw = raw_output
    parsed = _extract_json(raw_output)

    if parsed is None:
        sample.parse_valid = False
        # Fallback: look for keywords
        lower = raw_output.lower()
        if "danger" in lower or "dangerous" in lower or "emergency" in lower:
            sample.pred_risk_label = "danger"
        elif "safe" in lower:
            sample.pred_risk_label = "safe"
        else:
            sample.pred_risk_label = "unknown"
        sample.pred_risk_severity = "unknown"
        sample.pred_explanation = raw_output[:200]
        return

    sample.pred_risk_label = _normalize_risk_label(str(parsed.get("risk_label", "")))
    sample.pred_risk_severity = _normalize_severity(str(parsed.get("risk_severity", "")))
    sample.pred_explanation = str(parsed.get("explanation", ""))[:500]
    sample.pred_no_deployment = bool(parsed.get("no_deployment", False))

    robots = parsed.get("selected_robots", [])
    if isinstance(robots, list):
        sample.pred_selected_robots = [_normalize_robot_name(r) for r in robots]
    elif isinstance(robots, str):
        sample.pred_selected_robots = [_normalize_robot_name(r) for r in robots.split(",")]

    sample.parse_valid = True


# ── Model interface ──────────────────────────────────────────────────────────


def _load_embernet(model_path: str, use_va_refiner: bool = False, device: str = "auto"):
    """Load EmberNet model for inference."""
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from inference import load_model, EmberVLM
    except ImportError:
        print("WARNING: Could not import EmberNet inference module.")
        print("         Running in mock mode (random predictions).")
        return None

    try:
        vlm = EmberVLM(
            model_path=model_path,
            use_va_refiner=use_va_refiner,
            device=device if device != "auto" else None,
        )
        return vlm
    except Exception as e:
        print(f"WARNING: Could not load model from {model_path}: {e}")
        print("         Running in mock mode (random predictions).")
        return None


def _mock_inference(sample: RiskSample, top_n: int = 3) -> str:
    """Generate a mock JSON response for testing the pipeline without a model."""
    import random
    label = random.choice(["safe", "danger"])
    severity = random.choice(["low", "medium", "high"])
    robots = random.sample(KNOWN_ROBOTS, min(top_n, len(KNOWN_ROBOTS)))
    return json.dumps({
        "risk_label": label,
        "risk_severity": severity,
        "explanation": "Mock prediction for pipeline testing.",
        "selected_robots": robots,
        "no_deployment": False,
    })


def run_inference(model, samples: List[RiskSample], top_n: int = 3):
    """Run inference on all samples and fill prediction fields."""
    from PIL import Image

    for i, sample in enumerate(samples):
        prompt = build_prompt(sample, top_n=top_n)

        if model is None:
            raw = _mock_inference(sample, top_n=top_n)
        else:
            try:
                img = None
                if sample.image_path and os.path.isfile(sample.image_path):
                    img = Image.open(sample.image_path).convert("RGB")
                raw = model.answer(image=img, question=prompt)
            except Exception as e:
                raw = json.dumps({
                    "risk_label": "unknown", "risk_severity": "unknown",
                    "explanation": f"Inference error: {e}",
                })

        parse_model_output(sample, raw)

        if (i + 1) % 50 == 0 or i == len(samples) - 1:
            print(f"    Inference: {i+1}/{len(samples)} samples done")


# ── Metrics ──────────────────────────────────────────────────────────────────


@dataclass
class BenchmarkMetrics:
    dataset_name: str
    total_samples: int = 0
    valid_samples: int = 0
    risk_accuracy: float = 0.0
    false_alarm_rate: float = 0.0
    missed_emergency_rate: float = 0.0
    per_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Robot selection (only for robot_selection)
    top1_accuracy: float = 0.0
    topn_recall: float = 0.0
    deployment_safety: float = 0.0
    confusion: Dict[str, int] = field(default_factory=dict)


def compute_metrics(samples: List[RiskSample], dataset_name: str) -> BenchmarkMetrics:
    """Compute all metrics for a set of evaluated samples."""
    m = BenchmarkMetrics(dataset_name=dataset_name)
    valid = [s for s in samples if s.parse_valid and s.pred_risk_label in ("safe", "danger")]
    m.total_samples = len(samples)
    m.valid_samples = len(valid)

    if not valid:
        return m

    # Confusion counts
    tp = sum(1 for s in valid if s.risk_label == "danger" and s.pred_risk_label == "danger")
    tn = sum(1 for s in valid if s.risk_label == "safe" and s.pred_risk_label == "safe")
    fp = sum(1 for s in valid if s.risk_label == "safe" and s.pred_risk_label == "danger")
    fn = sum(1 for s in valid if s.risk_label == "danger" and s.pred_risk_label == "safe")

    m.confusion = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    m.risk_accuracy = (tp + tn) / len(valid) if valid else 0.0

    safe_total = sum(1 for s in valid if s.risk_label == "safe")
    danger_total = sum(1 for s in valid if s.risk_label == "danger")
    m.false_alarm_rate = fp / safe_total if safe_total > 0 else 0.0
    m.missed_emergency_rate = fn / danger_total if danger_total > 0 else 0.0

    # Per-category/task breakdown
    cats = defaultdict(list)
    for s in valid:
        cat_key = s.metadata.get("category", s.task_name)
        cats[cat_key].append(s)

    for cat, cat_samples in cats.items():
        correct = sum(1 for s in cat_samples if s.pred_risk_label == s.risk_label)
        m.per_category[cat] = {
            "count": len(cat_samples),
            "accuracy": correct / len(cat_samples) if cat_samples else 0.0,
        }

    # Robot selection metrics
    if dataset_name == "robot_selection":
        top1_correct = 0
        topn_recall_sum = 0
        topn_count = 0
        no_deploy_correct = 0
        no_deploy_total = 0

        for s in valid:
            gt_robots = s.robot_selection_gt
            pred_robots = s.pred_selected_robots

            if gt_robots and pred_robots:
                if pred_robots[0] in gt_robots:
                    top1_correct += 1

                gt_set = set(gt_robots)
                pred_set = set(pred_robots)
                overlap = gt_set & pred_set
                topn_recall_sum += len(overlap) / len(gt_set) if gt_set else 0
                topn_count += 1

            if s.no_deployment_gt:
                no_deploy_total += 1
                if s.pred_no_deployment:
                    no_deploy_correct += 1

        m.top1_accuracy = top1_correct / len(valid) if valid else 0.0
        m.topn_recall = topn_recall_sum / topn_count if topn_count > 0 else 0.0
        m.deployment_safety = no_deploy_correct / no_deploy_total if no_deploy_total > 0 else 1.0

    return m


# ── Result logging ───────────────────────────────────────────────────────────


def save_results(samples: List[RiskSample], output_path: Path):
    """Write per-sample results as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            row = {
                "sample_id": s.sample_id,
                "dataset_name": s.dataset_name,
                "task_name": s.task_name,
                "risk_label_gt": s.risk_label,
                "risk_severity_gt": s.risk_severity,
                "risk_label_pred": s.pred_risk_label,
                "risk_severity_pred": s.pred_risk_severity,
                "correct": s.pred_risk_label == s.risk_label,
                "robot_gt": s.robot_selection_gt,
                "robot_pred": s.pred_selected_robots,
                "no_deployment_gt": s.no_deployment_gt,
                "no_deployment_pred": s.pred_no_deployment,
                "explanation": s.pred_explanation,
                "parse_valid": s.parse_valid,
            }
            if s.metadata.get("category"):
                row["category"] = s.metadata["category"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Results saved → {output_path}")


def save_metrics_summary(all_metrics: List[BenchmarkMetrics], output_path: Path,
                         trial: bool = False):
    """Save aggregated metrics JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "trial" if trial else "main",
        "benchmarks": {},
    }
    for m in all_metrics:
        summary["benchmarks"][m.dataset_name] = asdict(m)
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"  Metrics summary → {output_path}")


def print_summary(all_metrics: List[BenchmarkMetrics]):
    """Print a concise summary table to stdout."""
    print(f"\n{'='*72}")
    print(f"  ROBOT RISK BENCHMARK RESULTS")
    print(f"{'='*72}")
    print(f"  {'Benchmark':<22} {'Acc':>7} {'FAlarm':>7} {'MissEm':>7} {'Top1':>7} {'TopN-R':>7}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for m in all_metrics:
        top1 = f"{m.top1_accuracy:.1%}" if m.dataset_name == "robot_selection" else "  —"
        topn = f"{m.topn_recall:.1%}" if m.dataset_name == "robot_selection" else "  —"
        print(f"  {m.dataset_name:<22} "
              f"{m.risk_accuracy:>6.1%} "
              f"{m.false_alarm_rate:>6.1%} "
              f"{m.missed_emergency_rate:>6.1%} "
              f"{top1:>7} "
              f"{topn:>7}")
    print(f"{'='*72}")

    for m in all_metrics:
        if m.per_category:
            print(f"\n  {m.dataset_name} — per category/task:")
            for cat, vals in sorted(m.per_category.items()):
                print(f"    {cat:<35} n={vals['count']:>4}  acc={vals['accuracy']:.1%}")
    print()


# ── Visualization trigger ────────────────────────────────────────────────────


def generate_plots(all_metrics: List[BenchmarkMetrics], results_dir: Path,
                   plots_dir: Path):
    """Invoke the visualization module to generate all benchmark plots."""
    try:
        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from visualizations.fig_robot_risk_benchmarks import generate
        generate(results_dir=results_dir, save_dir=plots_dir)
    except ImportError as e:
        print(f"  [plots] Could not import visualization module: {e}")
    except Exception as e:
        print(f"  [plots] Plot generation failed: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Robot risk-aware benchmarking suite for EmberNet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument("--model-path", type=str,
                        default="./checkpoints/stage2/final_model.pt",
                        help="Path to EmberNet checkpoint")
    parser.add_argument("--benchmarks-dir", type=str, default="./benchmarks",
                        help="Root directory containing benchmark datasets")
    parser.add_argument("--top-n", type=int, default=3,
                        help="Number of robots to select (robot-selection benchmark)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per benchmark (for quick testing)")
    parser.add_argument("--use-va-refiner", action="store_true",
                        help="Enable VA Refiner during evaluation")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for inference (auto/cuda/cpu)")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trial", action="store_true",
                      help="Trial mode: small subset (~20 samples) for sanity check")
    mode.add_argument("--main", action="store_true",
                      help="Main mode: full evaluation on all data")

    args = parser.parse_args()

    benchmarks_dir = Path(args.benchmarks_dir)
    if not benchmarks_dir.exists():
        sys.exit(f"ERROR: Benchmarks directory not found: {benchmarks_dir}\n"
                 f"       Run: python benchmarks/download_benchmarks.py --mode trial --output-dir benchmarks")

    limit = args.limit
    if args.trial and limit is None:
        limit = 20

    print(f"Mode: {'trial' if args.trial else 'main'}")
    print(f"Benchmarks dir: {benchmarks_dir.resolve()}")
    print(f"Model: {args.model_path}")
    print(f"Top-N: {args.top_n}, Limit: {limit or 'none'}")
    print(f"VA Refiner: {'ON' if args.use_va_refiner else 'OFF'}")

    # Load model
    print("\nLoading model...")
    model = _load_embernet(args.model_path, use_va_refiner=args.use_va_refiner,
                           device=args.device)

    # Load datasets
    print("\nLoading datasets...")
    rs_samples = load_robot_selection(benchmarks_dir, limit=limit, top_n=args.top_n)
    geo_samples = load_geobench(benchmarks_dir, limit=limit)
    veri_samples = load_veri(benchmarks_dir, limit=limit)

    all_samples = {
        "robot_selection": rs_samples,
        "geobench_vlm": geo_samples,
        "veri_emergency": veri_samples,
    }

    # Run inference per dataset
    all_metrics = []
    mode_tag = "trial" if args.trial else "main"
    results_dir = benchmarks_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for name, samples in all_samples.items():
        if not samples:
            continue
        print(f"\nEvaluating {name} ({len(samples)} samples)...")
        run_inference(model, samples, top_n=args.top_n)

        metrics = compute_metrics(samples, name)
        all_metrics.append(metrics)

        out_file = results_dir / f"{name}_results_{mode_tag}.jsonl"
        save_results(samples, out_file)

    # Save metrics and print summary
    if all_metrics:
        metrics_file = results_dir / f"metrics_summary_{mode_tag}.json"
        save_metrics_summary(all_metrics, metrics_file, trial=args.trial)
        print_summary(all_metrics)

        # Generate plots
        plots_dir = benchmarks_dir / "plots" / "robot_risk_benchmarks"
        plots_dir.mkdir(parents=True, exist_ok=True)
        print("Generating visualizations...")
        generate_plots(all_metrics, results_dir, plots_dir)

    print("Done.")


if __name__ == "__main__":
    main()
