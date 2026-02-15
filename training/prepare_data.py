#!/usr/bin/env python3
"""
Data Preparation Script for EmberNet VLM

This script downloads and prepares ALL required datasets for training a
high-quality EmberNet VLM. Run this BEFORE training.

================================================================================
EXPERT ARCHITECTURE (8 Domain Experts + 1 Shared Expert)
================================================================================

EmberNet uses a Mixture-of-Experts (MoE) architecture with 8 specialized experts.
Each token is routed to the top-2 most relevant experts + the shared expert.

┌─────────────────────────────────────────────────────────────────────────────┐
│  EXPERT 0: vision_ocr        │  Reads text in images, OCR                  │
│  EXPERT 1: vision_diagram    │  Understands diagrams, infographics         │
│  EXPERT 2: code_math_chart   │  Analyzes charts, graphs, plots             │
│  EXPERT 3: code_math_formula │  Handles math equations, formulas           │
│  EXPERT 4: spatial_scene     │  Scene understanding, object detection      │
│  EXPERT 5: spatial_reasoning │  Spatial relationships, counting            │
│  EXPERT 6: agentic_knowledge │  Knowledge-based QA, facts                  │
│  EXPERT 7: agentic_reasoning │  Multi-step reasoning, logic                │
│  SHARED:   (always active)   │  Common patterns across all domains         │
└─────────────────────────────────────────────────────────────────────────────┘

DATASET → EXPERT MAPPING:
─────────────────────────
Stage 1 (Alignment): Trains PROJECTOR, not experts
  - LLaVA-Instruct, ShareGPT4V, ALLaVA → Projector learns vision-language mapping

Stage 2 (Expert SFT): Trains EXPERTS based on domain
  - TextVQA, DocVQA, OCR-VQA      → Expert 0 (vision_ocr)
  - AI2D, InfoVQA                  → Expert 1 (vision_diagram)
  - ChartQA, PlotQA, FigureQA      → Expert 2 (code_math_chart)
  - MathVista                      → Expert 3 (code_math_formula)
  - VQAv2, Visual Genome           → Expert 4 (spatial_scene)
  - GQA                            → Expert 5 (spatial_reasoning)
  - OK-VQA, A-OKVQA                → Expert 6 (agentic_knowledge)
  - ScienceQA, CLEVR               → Expert 7 (agentic_reasoning)

================================================================================
COMPLETE DATASET LIST
================================================================================

STAGE 1 - VISION-LANGUAGE ALIGNMENT (Projector Training):
----------------------------------------------------------
These datasets teach the model to connect visual features with language.
The projector learns to map SigLIP image embeddings into the LLM's embedding space.

- LLaVA-Instruct-150K: High-quality image-text conversations
- ShareGPT4V: Detailed image descriptions and conversations
- ALLaVA: Diverse visual instruction data

STAGE 2 - EXPERT SPECIALIZATION (Domain-Specific Fine-tuning):
--------------------------------------------------------------
Each expert cluster gets specialized data:

VISION/OCR EXPERTS (0, 1):
- TextVQA: Reading text in natural scene images
- DocVQA: Understanding documents, forms, receipts
- AI2D: Scientific diagrams with annotations
- InfoVQA: Infographics understanding
- OCR-VQA: Book covers, signs, product labels

CODE/MATH EXPERTS (2, 3):
- ChartQA: Bar charts, line graphs, pie charts
- PlotQA: Scientific plots and figures
- FigureQA: Figure understanding and reasoning
- DVQA: Data visualization QA
- MathVista: Mathematical visual reasoning

SPATIAL/SCENE EXPERTS (4, 5):
- VQAv2: General visual question answering
- GQA: Scene graph based visual reasoning
- Visual Genome: Dense scene annotations

REASONING EXPERTS (6, 7):
- OK-VQA: Outside knowledge VQA
- A-OKVQA: Augmented outside knowledge VQA
- ScienceQA: Science diagrams and questions
- CLEVR: Compositional reasoning

USAGE:
======
    python training/prepare_data.py --all          # Download everything (~100GB)
    python training/prepare_data.py --recommended  # Good balance (~70GB)
    python training/prepare_data.py --minimal      # Quick testing (~10GB)
    python training/prepare_data.py --list         # Show all datasets
    python training/prepare_data.py --explain      # Explain alignment & experts

REQUIREMENTS:
=============
    pip install datasets pillow huggingface_hub
"""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict

# =============================================================================
# COMPLETE DATASET CONFIGURATIONS
# =============================================================================

DATASETS = {
    # =========================================================================
    # STAGE 1: VISION-LANGUAGE ALIGNMENT
    # These are critical for teaching the projector to align image and text
    # NOTE: Stage 1 trains the PROJECTOR, not the experts
    # =========================================================================

    "llava_instruct_150k": {
        "hf_name": "lmms-lab/LLaVA-Instruct-150K",
        "config": "default",
        "description": "High-quality GPT-4 generated visual conversations",
        "stage": 1,
        "domain": "general",
        "expert": "Projector (not experts)",
        "size_gb": 5.0,
        "priority": "critical",
        "samples": "150K",
    },
    "sharegpt4v": {
        "hf_name": "lmms-lab/ShareGPT4V",
        "config": "default",
        "description": "Detailed image descriptions from GPT-4V",
        "stage": 1,
        "domain": "general",
        "expert": "Projector (not experts)",
        "size_gb": 8.0,
        "priority": "critical",
        "samples": "100K",
    },
    "allava_instruct": {
        "hf_name": "FreedomIntelligence/ALLaVA-4V",
        "config": "allava_vflan",
        "description": "Diverse visual instruction tuning data",
        "stage": 1,
        "domain": "general",
        "expert": "Projector (not experts)",
        "size_gb": 6.0,
        "priority": "recommended",
        "samples": "700K",
    },

    # =========================================================================
    # STAGE 2: VISION/OCR EXPERTS (Expert 0 & 1)
    # For reading text, understanding documents, diagrams
    # =========================================================================

    "textvqa": {
        "hf_name": "lmms-lab/TextVQA",
        "config": "default",
        "description": "Text reading in natural scene images",
        "stage": 2,
        "domain": "vision_ocr",
        "expert": "Expert 0: vision_ocr",
        "size_gb": 2.0,
        "priority": "critical",
        "samples": "45K",
    },
    "docvqa": {
        "hf_name": "lmms-lab/DocVQA",
        "config": "DocVQA",
        "description": "Document understanding - forms, receipts, letters",
        "stage": 2,
        "domain": "vision_ocr",
        "expert": "Expert 0: vision_ocr",
        "size_gb": 3.0,
        "priority": "critical",
        "samples": "50K",
    },
    "infovqa": {
        "hf_name": "lmms-lab/InfographicVQA",
        "config": "default",
        "description": "Infographics understanding",
        "stage": 2,
        "domain": "vision_diagram",
        "expert": "Expert 1: vision_diagram",
        "size_gb": 2.5,
        "priority": "recommended",
        "samples": "30K",
    },
    "ocrvqa": {
        "hf_name": "howard-hou/OCR-VQA",
        "config": "default",
        "description": "OCR on book covers, signs, products",
        "stage": 2,
        "domain": "vision_ocr",
        "expert": "Expert 0: vision_ocr",
        "size_gb": 4.0,
        "priority": "recommended",
        "samples": "200K",
    },
    "ai2d": {
        "hf_name": "lmms-lab/ai2d",
        "config": "default",
        "description": "Scientific diagram understanding",
        "stage": 2,
        "domain": "vision_diagram",
        "expert": "Expert 1: vision_diagram",
        "size_gb": 1.5,
        "priority": "critical",
        "samples": "15K",
    },

    # =========================================================================
    # STAGE 2: CODE/MATH/CHART EXPERTS (Expert 2 & 3)
    # For understanding charts, graphs, mathematical figures
    # =========================================================================

    "chartqa": {
        "hf_name": "ahmed-masry/ChartQA",
        "config": "default",
        "description": "Chart understanding - bar, line, pie charts",
        "stage": 2,
        "domain": "code_math_chart",
        "expert": "Expert 2: code_math_chart",
        "size_gb": 1.0,
        "priority": "critical",
        "samples": "32K",
    },
    "plotqa": {
        "hf_name": "lmms-lab/PlotQA",
        "config": "default",
        "description": "Scientific plot understanding",
        "stage": 2,
        "domain": "code_math_chart",
        "expert": "Expert 2: code_math_chart",
        "size_gb": 8.0,
        "priority": "recommended",
        "samples": "224K",
    },
    "figureqa": {
        "hf_name": "lmms-lab/FigureQA",
        "config": "default",
        "description": "Figure understanding and visual reasoning",
        "stage": 2,
        "domain": "code_math_chart",
        "expert": "Expert 2: code_math_chart",
        "size_gb": 5.0,
        "priority": "recommended",
        "samples": "180K",
    },
    "dvqa": {
        "hf_name": "lmms-lab/DVQA",
        "config": "default",
        "description": "Data visualization QA",
        "stage": 2,
        "domain": "code_math_chart",
        "expert": "Expert 2: code_math_chart",
        "size_gb": 3.0,
        "priority": "optional",
        "samples": "300K",
    },
    "mathvista": {
        "hf_name": "AI4Math/MathVista",
        "config": "default",
        "description": "Mathematical visual reasoning",
        "stage": 2,
        "domain": "code_math_formula",
        "expert": "Expert 3: code_math_formula",
        "size_gb": 1.0,
        "priority": "critical",
        "samples": "6K",
    },

    # =========================================================================
    # STAGE 2: SPATIAL/SCENE UNDERSTANDING EXPERTS
    # For understanding scenes, spatial relationships, counting
    # =========================================================================

    "vqav2": {
        "hf_name": "lmms-lab/VQAv2",
        "config": "default",
        "description": "General visual question answering",
        "stage": 2,
        "domain": "spatial_scene",
        "size_gb": 25.0,
        "priority": "critical",
        "samples": "1.1M",
        "expert": "Expert 4: spatial_scene",
    },
    "gqa": {
        "hf_name": "lmms-lab/GQA",
        "config": "train_balanced_instructions",
        "description": "Scene graph based visual reasoning",
        "stage": 2,
        "domain": "spatial_reasoning",
        "size_gb": 15.0,
        "priority": "recommended",
        "samples": "22M",
        "expert": "Expert 5: spatial_reasoning",
    },
    "visual_genome": {
        "hf_name": "lmms-lab/VisualGenome",
        "config": "default",
        "description": "Dense scene annotations and relationships",
        "stage": 2,
        "domain": "spatial_scene",
        "expert": "Expert 4: spatial_scene",
        "size_gb": 15.0,
        "priority": "optional",
        "samples": "108K images",
    },
    "okvqa": {
        "hf_name": "lmms-lab/OK-VQA",
        "config": "default",
        "description": "Outside knowledge visual QA",
        "stage": 2,
        "domain": "agentic_knowledge",
        "expert": "Expert 6: agentic_knowledge",
        "size_gb": 1.0,
        "priority": "recommended",
        "samples": "14K",
    },

    # =========================================================================
    # STAGE 2: REASONING EXPERTS (Expert 6 & 7)
    # For complex multi-step reasoning, science, compositional understanding
    # =========================================================================

    "aokvqa": {
        "hf_name": "HuggingFaceM4/A-OKVQA",
        "config": "default",
        "description": "Augmented outside knowledge VQA with rationales",
        "stage": 2,
        "domain": "agentic_knowledge",
        "expert": "Expert 6: agentic_knowledge",
        "size_gb": 1.5,
        "priority": "recommended",
        "samples": "25K",
    },
    "scienceqa": {
        "hf_name": "derek-thomas/ScienceQA",
        "config": "default",
        "description": "Science questions with diagrams",
        "stage": 2,
        "domain": "agentic_reasoning",
        "expert": "Expert 7: agentic_reasoning",
        "size_gb": 2.0,
        "priority": "critical",
        "samples": "21K",
    },
    "clevr": {
        "hf_name": "lmms-lab/CLEVR",
        "config": "default",
        "description": "Compositional visual reasoning",
        "stage": 2,
        "domain": "agentic_reasoning",
        "expert": "Expert 7: agentic_reasoning",
        "size_gb": 18.0,
        "priority": "optional",
        "samples": "850K",
    },
}

# =============================================================================
# DATASET GROUPINGS
# =============================================================================

# Absolute minimum for testing
MINIMAL_DATASETS = [
    "llava_instruct_150k",  # Stage 1 alignment
    "textvqa",              # OCR
    "chartqa",              # Charts
    "vqav2",                # General VQA
]

# Critical datasets - model won't work well without these
CRITICAL_DATASETS = [k for k, v in DATASETS.items() if v["priority"] == "critical"]

# Recommended - good quality model
RECOMMENDED_DATASETS = [k for k, v in DATASETS.items() if v["priority"] in ["critical", "recommended"]]

# All datasets for best performance
ALL_DATASETS = list(DATASETS.keys())


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import datasets
    except ImportError:
        missing.append("datasets")

    try:
        from PIL import Image
    except ImportError:
        missing.append("pillow")

    try:
        import huggingface_hub
    except ImportError:
        missing.append("huggingface_hub")

    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    print("✓ All dependencies installed")


def get_total_size(dataset_keys: List[str]) -> float:
    """Calculate total size of datasets."""
    return sum(DATASETS[k]["size_gb"] for k in dataset_keys if k in DATASETS)


def download_dataset(
    dataset_key: str,
    output_dir: Path,
    force: bool = False,
) -> bool:
    """
    Download a single dataset from HuggingFace.
    """
    from datasets import load_dataset

    if dataset_key not in DATASETS:
        print(f"ERROR: Unknown dataset: {dataset_key}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        return False

    info = DATASETS[dataset_key]
    hf_name = info["hf_name"]
    config_name = info.get("config", None)
    save_path = output_dir / dataset_key

    # Check if already exists
    if save_path.exists() and not force:
        print(f"✓ {dataset_key} already downloaded at {save_path}")
        return True

    print(f"\n{'='*70}")
    print(f"DOWNLOADING: {dataset_key}")
    print(f"{'='*70}")
    print(f"  Source:      {hf_name}")
    if config_name:
        print(f"  Config:      {config_name}")
    print(f"  Description: {info['description']}")
    print(f"  Samples:     {info['samples']}")
    print(f"  Size:        ~{info['size_gb']} GB")
    print(f"  Domain:      {info['domain']}")
    print(f"  Stage:       {info['stage']}")
    print(f"{'='*70}\n")

    start_time = time.time()

    try:
        # Load dataset from HuggingFace
        # We try to use trust_remote_code=True as many VLM datasets use custom loaders
        # but fallback if the library version doesn't support it or if it's not needed.
        try:
            if config_name:
                ds = load_dataset(hf_name, config_name, trust_remote_code=True)
            else:
                ds = load_dataset(hf_name, trust_remote_code=True)
        except Exception:
            # Fallback for datasets that don't need/support trust_remote_code
            if config_name:
                ds = load_dataset(hf_name, config_name)
            else:
                ds = load_dataset(hf_name)

        # Save to disk
        save_path.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(save_path))

        download_time = time.time() - start_time

        # Save detailed metadata
        metadata = {
            "dataset_key": dataset_key,
            "hf_name": hf_name,
            "config": config_name,
            "domain": info["domain"],
            "expert": info.get("expert", "N/A"),
            "stage": info["stage"],
            "priority": info["priority"],
            "description": info["description"],
            "splits": list(ds.keys()),
            "num_samples": {split: len(ds[split]) for split in ds.keys()},
            "total_samples": sum(len(ds[split]) for split in ds.keys()),
            "estimated_size_gb": info["size_gb"],
            "download_time_seconds": round(download_time, 2),
            "save_path": str(save_path.absolute()),
            "download_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Successfully saved {dataset_key} to {save_path}")
        print(f"  Download time: {download_time:.1f} seconds")
        return True

    except Exception as e:
        print(f"✗ FAILED to download {dataset_key}: {e}")
        return False

