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
        "hf_name": "liuhaotian/LLaVA-Instruct-150K",
        "description": "High-quality GPT-4 generated visual conversations",
        "stage": 1,
        "domain": "general",
        "expert": "Projector (not experts)",
        "size_gb": 5.0,
        "priority": "critical",
        "samples": "150K",
    },
    "sharegpt4v": {
        "hf_name": "Lin-Chen/ShareGPT4V",
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
        "hf_name": "textvqa",
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
        "hf_name": "HuggingFaceM4/VQAv2",
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
        "description": "Scene graph based visual reasoning",
        "stage": 2,
        "domain": "spatial_reasoning",
        "size_gb": 15.0,
        "priority": "recommended",
        "samples": "22M",
        "expert": "Expert 5: spatial_reasoning",
    },
    "visual_genome": {
        "hf_name": "visual_genome",
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
        "config": None,
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
        "config": None,
        "description": "Science questions with diagrams",
        "stage": 2,
        "domain": "agentic_reasoning",
        "expert": "Expert 7: agentic_reasoning",
        "size_gb": 2.0,
        "priority": "critical",
        "samples": "21K",
    },
    "clevr": {
        "hf_name": "clevr/clevr",
        "config": None,
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
        # Load dataset from HuggingFace (without trust_remote_code)
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


def download_all_datasets(
    dataset_keys: List[str],
    output_dir: Path,
    force: bool = False,
) -> Dict[str, bool]:
    """Download multiple datasets with progress tracking and manifest logging."""
    results = {}
    download_session = {
        "session_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets_requested": dataset_keys,
        "output_dir": str(output_dir.absolute()),
        "downloads": []
    }

    total_size = get_total_size(dataset_keys)
    print(f"\n{'#'*70}")
    print(f"# EMBERNET DATA PREPARATION")
    print(f"{'#'*70}")
    print(f"\nDatasets to download: {len(dataset_keys)}")
    print(f"Estimated total size: ~{total_size:.1f} GB")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"\nThis may take a while depending on your internet connection...")
    print(f"{'#'*70}\n")

    session_start_time = time.time()

    for i, key in enumerate(dataset_keys, 1):
        print(f"\n[{i}/{len(dataset_keys)}] Processing {key}...")
        success = download_dataset(key, output_dir, force)
        results[key] = success

        # Load metadata from individual dataset folder if successful
        if success:
            metadata_path = output_dir / key / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    dataset_metadata = json.load(f)
                    download_session["downloads"].append({
                        "dataset_key": key,
                        "status": "success",
                        "metadata": dataset_metadata
                    })
        else:
            download_session["downloads"].append({
                "dataset_key": key,
                "status": "failed",
                "error": "Download failed"
            })

    session_total_time = time.time() - session_start_time
    download_session["session_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    download_session["total_time_seconds"] = round(session_total_time, 2)
    download_session["successful_downloads"] = sum(1 for v in results.values() if v)
    download_session["failed_downloads"] = sum(1 for v in results.values() if not v)

    # Save download session manifest
    manifest_path = output_dir / "download_manifest.json"

    # Load existing manifest if it exists
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest_data = json.load(f)
            if "sessions" not in manifest_data:
                manifest_data = {"sessions": []}
    else:
        manifest_data = {"sessions": []}

    # Append this session
    manifest_data["sessions"].append(download_session)
    manifest_data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n✓ Download manifest saved to {manifest_path}")

    return results


def create_dataset_index(output_dir: Path):
    """Create an index file mapping datasets to training stages and domains."""
    index = {
        "version": "1.0",
        "datasets": {},
        "by_stage": {
            "1": [],  # Alignment datasets
            "2": [],  # Expert SFT datasets
        },
        "by_domain": {},
    }

    for dataset_key in DATASETS:
        dataset_path = output_dir / dataset_key
        if dataset_path.exists():
            metadata_path = dataset_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)

                index["datasets"][dataset_key] = {
                    "path": str(dataset_path),
                    **metadata
                }

                # Index by stage
                stage = str(metadata["stage"])
                index["by_stage"][stage].append(dataset_key)

                # Index by domain
                domain = metadata["domain"]
                if domain not in index["by_domain"]:
                    index["by_domain"][domain] = []
                index["by_domain"][domain].append(dataset_key)

    index_path = output_dir / "dataset_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n✓ Dataset index saved to {index_path}")
    return index


def print_dataset_info():
    """Print detailed information about all available datasets."""
    print("\n" + "="*80)
    print(" EMBERNET TRAINING DATASETS")
    print("="*80)

    # Stage 1
    print("\n" + "-"*80)
    print(" STAGE 1: VISION-LANGUAGE ALIGNMENT")
    print(" These datasets train the projector to connect images with text")
    print("-"*80)

    stage1_size = 0
    for key, info in DATASETS.items():
        if info["stage"] == 1:
            marker = "★" if info["priority"] == "critical" else "●"
            print(f"\n  {marker} {key}")
            print(f"      HuggingFace: {info['hf_name']}")
            print(f"      Description: {info['description']}")
            print(f"      Samples: {info['samples']} | Size: ~{info['size_gb']} GB")
            stage1_size += info['size_gb']
    print(f"\n  Stage 1 Total: ~{stage1_size:.1f} GB")

    # Stage 2 by domain
    print("\n" + "-"*80)
    print(" STAGE 2: EXPERT SPECIALIZATION")
    print(" These datasets train domain-specific experts")
    print("-"*80)

    domains = {}
    for key, info in DATASETS.items():
        if info["stage"] == 2:
            domain = info["domain"]
            if domain not in domains:
                domains[domain] = []
            domains[domain].append((key, info))

    stage2_size = 0
    for domain, items in domains.items():
        domain_size = sum(info['size_gb'] for _, info in items)
        stage2_size += domain_size

        print(f"\n  [{domain.upper()}] (~{domain_size:.1f} GB)")
        for key, info in items:
            marker = {"critical": "★", "recommended": "●", "optional": "○"}[info["priority"]]
            print(f"    {marker} {key}: {info['description']}")
            print(f"        Samples: {info['samples']} | Size: ~{info['size_gb']} GB")

    print(f"\n  Stage 2 Total: ~{stage2_size:.1f} GB")

    # Summary
    print("\n" + "="*80)
    print(" SUMMARY")
    print("="*80)
    print(f"\n  --minimal:     {len(MINIMAL_DATASETS)} datasets, ~{get_total_size(MINIMAL_DATASETS):.1f} GB")
    print(f"  --critical:    {len(CRITICAL_DATASETS)} datasets, ~{get_total_size(CRITICAL_DATASETS):.1f} GB")
    print(f"  --recommended: {len(RECOMMENDED_DATASETS)} datasets, ~{get_total_size(RECOMMENDED_DATASETS):.1f} GB")
    print(f"  --all:         {len(ALL_DATASETS)} datasets, ~{get_total_size(ALL_DATASETS):.1f} GB")
    print("\n  Legend: ★ Critical  ● Recommended  ○ Optional")
    print("="*80)


def print_alignment_explanation():
    """Print explanation of how vision-language alignment works."""
    print("""
================================================================================
HOW VISION-LANGUAGE ALIGNMENT WORKS IN EMBERNET
================================================================================

EmberNet aligns images with text through a multi-stage process:

┌─────────────────────────────────────────────────────────────────────────────┐
│ IMAGE INPUT                                                                  │
│    ↓                                                                        │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ VISION ENCODER (SigLIP - Frozen)                                        │ │
│ │   • Pretrained on 400M image-text pairs                                 │ │
│ │   • Extracts 196 visual tokens (14x14 grid)                            │ │
│ │   • Each token = 768-dim feature vector                                │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│    ↓                                                                        │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ TOKEN COMPRESSION (Trainable)                                           │ │
│ │   • Pixel Shuffle: 196 tokens → 49 tokens (2x2 merge)                  │ │
│ │   • Adaptive Pooling: 49 → 64 tokens (learnable queries)               │ │
│ │   • Reduces computation while preserving information                    │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│    ↓                                                                        │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ PROJECTOR (BitLinear MLP - Stage 1 Training Target)                    │ │
│ │   • 2-layer MLP with ternary weights                                   │ │
│ │   • Maps vision space → language space                                 │ │
│ │   • This is WHERE alignment happens!                                   │ │
│ │   • Input: 768-dim visual features                                     │ │
│ │   • Output: 768-dim language-compatible embeddings                     │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│    ↓                                                                        │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ MERGED SEQUENCE                                                          │ │
│ │   [BOS] [IMG_1] [IMG_2] ... [IMG_64] [User: What's in this image?] ... │ │
│ │         ↑ visual tokens              ↑ text tokens                      │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│    ↓                                                                        │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ BITNET MOE DECODER (Stage 2 Training Target)                            │ │
│ │   • 16 transformer layers                                               │ │
│ │   • Processes visual + text tokens together                             │ │
│ │   • MoE routes to specialized experts based on content                 │ │
│ │   • Generates text response autoregressively                           │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│    ↓                                                                        │
│ TEXT OUTPUT: "This image shows a sunset over the ocean..."                 │
└─────────────────────────────────────────────────────────────────────────────┘

TRAINING STAGES:
================

STAGE 1 - PROJECTOR ALIGNMENT (~3 epochs):
  • Freeze: Vision encoder + Language decoder
  • Train: Projector + Token compression
  • Data: LLaVA-Instruct, ShareGPT4V (high-quality image descriptions)
  • Goal: Learn to map visual features into language embedding space
  • Loss: Next-token prediction on image captions/conversations

STAGE 2 - EXPERT SPECIALIZATION (~10 epochs):
  • Freeze: Vision encoder + Embeddings
  • Train: MoE experts + Router
  • Data: Domain-specific datasets (OCR, charts, diagrams, etc.)
  • Goal: Specialize experts for different visual tasks
  • Loss: Next-token prediction + Router load balancing

WHY THIS WORKS:
===============

1. SigLIP already understands images (pretrained on 400M pairs)
2. The projector learns a linear-ish mapping between:
   - SigLIP's "visual concept" space
   - LLM's "language concept" space
3. Since both spaces encode similar semantic concepts,
   the mapping is relatively simple to learn
4. After alignment, visual tokens "look like" language tokens
   to the decoder, enabling natural multimodal understanding

================================================================================
""")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for EmberNet VLM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training/prepare_data.py --list              # Show all datasets
  python training/prepare_data.py --explain           # Explain alignment process
  python training/prepare_data.py --minimal           # Quick testing (~10GB)
  python training/prepare_data.py --critical          # Essential only (~45GB)
  python training/prepare_data.py --recommended       # Good quality (~70GB)
  python training/prepare_data.py --all               # Best quality (~100GB)
  python training/prepare_data.py --dataset textvqa   # Specific dataset
        """
    )

    # Dataset selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--minimal", action="store_true",
                       help="Download minimal set for quick testing (~10GB)")
    group.add_argument("--critical", action="store_true",
                       help="Download critical datasets only (~45GB)")
    group.add_argument("--recommended", action="store_true",
                       help="Download recommended datasets (~70GB)")
    group.add_argument("--all", action="store_true",
                       help="Download ALL datasets for best quality (~100GB)")
    group.add_argument("--dataset", nargs="+", metavar="NAME",
                       help="Download specific dataset(s) by name")
    group.add_argument("--list", action="store_true",
                       help="List all available datasets")
    group.add_argument("--explain", action="store_true",
                       help="Explain how vision-language alignment works")

    # Options
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Output directory (default: ./data)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if exists")

    args = parser.parse_args()

    # Info commands
    if args.list:
        print_dataset_info()
        return

    if args.explain:
        print_alignment_explanation()
        return

    # Check dependencies
    check_dependencies()

    # Determine datasets to download
    if args.minimal:
        dataset_keys = MINIMAL_DATASETS
        print("\n⚠️  MINIMAL mode: Good for testing, not for production use")
    elif args.critical:
        dataset_keys = CRITICAL_DATASETS
        print("\n📦 CRITICAL mode: Essential datasets for a working model")
    elif args.recommended:
        dataset_keys = RECOMMENDED_DATASETS
        print("\n✅ RECOMMENDED mode: Good balance of quality and size")
    elif args.all:
        dataset_keys = ALL_DATASETS
        print("\n🚀 ALL mode: Maximum quality, downloading everything")
    elif args.dataset:
        dataset_keys = args.dataset
        print(f"\n📦 CUSTOM mode: Downloading {len(dataset_keys)} specific dataset(s)")
    else:
        print("\nNo selection specified. Use --list to see options.")
        print("Defaulting to --recommended for good quality...")
        dataset_keys = RECOMMENDED_DATASETS

    # Download
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = download_all_datasets(dataset_keys, output_dir, args.force)

    # Create index
    index = create_dataset_index(output_dir)

    # Summary
    print("\n" + "="*70)
    print(" DOWNLOAD COMPLETE")
    print("="*70)

    success = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]

    print(f"\n✓ Succeeded: {len(success)}")
    for k in success:
        print(f"    - {k}")

    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for k in failed:
            print(f"    - {k}")

    print(f"\nData directory: {output_dir.absolute()}")
    print(f"Stage 1 datasets: {len(index['by_stage']['1'])}")
    print(f"Stage 2 datasets: {len(index['by_stage']['2'])}")

    # Manifest file info
    manifest_path = output_dir / "download_manifest.json"
    if manifest_path.exists():
        print(f"\n📋 Download tracking manifest: {manifest_path}")
        print(f"   This file contains detailed info about all downloaded datasets")

    # Next steps
    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)
    print("""
1. Stage 1 - Train the projector (vision-language alignment):
   python training/train.py --stage 1 --data-dir ./data --epochs 3 --batch-size 8

2. Stage 2 - Train the experts (domain specialization):
   python training/train.py --stage 2 --data-dir ./data --epochs 10 --batch-size 4

3. Convert to optimized format:
   python inference/convert.py checkpoints/final_model.pt embernet.pt

4. Run inference:
   python inference/infer.py --model embernet.pt --interactive
""")


if __name__ == "__main__":
    main()

