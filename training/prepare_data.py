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
  - TextVQA, DocVQA                → Expert 0 (vision_ocr)
  - AI2D                            → Expert 1 (vision_diagram)
  - ChartQA, PlotQA                 → Expert 2 (code_math_chart)
  - MathVista                       → Expert 3 (code_math_formula)
  - VQAv2                           → Expert 4 (spatial_scene)
  - GQA                             → Expert 5 (spatial_reasoning)
  - OK-VQA, A-OKVQA                 → Expert 6 (agentic_knowledge)
  - ScienceQA                       → Expert 7 (agentic_reasoning)

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
- COCO Captions: General image captioning and description
- Conceptual Captions: Large-scale diverse image captioning (optional)

STAGE 2 - EXPERT SPECIALIZATION (Domain-Specific Fine-tuning):
--------------------------------------------------------------
Each expert cluster gets specialized data:

VISION/OCR EXPERTS (0, 1):
- TextVQA: Reading text in natural scene images
- DocVQA: Understanding documents, forms, receipts
- AI2D: Scientific diagrams with annotations
(Note: InfoVQA, OCR-VQA not available as standalone datasets)

CODE/MATH EXPERTS (2, 3):
- ChartQA: Bar charts, line graphs, pie charts
- PlotQA: Scientific plots and figures (alternative source: achang/plot_qa)
- MathVista: Mathematical visual reasoning
(Note: FigureQA, DVQA not available as standalone datasets)

SPATIAL/SCENE EXPERTS (4, 5):
- VQAv2: General visual question answering
- GQA: Scene graph based visual reasoning
- RefCOCO: Visual grounding and referring expressions (optional)
- Visual Genome: Dense region descriptions (optional)
- NLVR2: Natural language visual reasoning with image pairs
- VSR: Visual spatial reasoning
(Note: Original Visual Genome dataset not available as standalone)

REASONING EXPERTS (6, 7):
- OK-VQA: Outside knowledge VQA
- A-OKVQA: Augmented outside knowledge VQA
- ScienceQA: Science diagrams and questions
- VCR: Visual Commonsense Reasoning with rationales
- Winoground: Visio-linguistic compositional reasoning (optional)
(Note: CLEVR not available as standalone dataset)

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
import os
import json
import argparse
import time
import tarfile
import zipfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Optional
import concurrent.futures
import hashlib

# =============================================================================
# COMPLETE DATASET CONFIGURATIONS
# =============================================================================

DATASETS = {
    # =========================================================================
    # STAGE 1: VISION-LANGUAGE ALIGNMENT
    # These are critical for teaching the projector to align image and text
    # NOTE: Stage 1 trains the PROJECTOR, not the experts
    # =========================================================================

    # NOTE: Using alternative high-quality instruction datasets that don't require loading scripts
    "llava_instruct_150k": {
        "hf_name": "adamo1139/llava-instruct-150k-with-images",
        "config": "default",
        "description": "LLaVA-Instruct-150K with images included",
        "stage": 1,
        "domain": "general",
        "expert": "Projector (not experts)",
        "preferred_download": "snapshot",
        "size_gb": 5.0,
        "priority": "critical",
        "samples": "150K",
    },
    "sharegpt4v": {
        "hf_name": "Lin-Chen/ShareGPT4V",
        "config": "ShareGPT4V",
        "description": "Detailed image descriptions from GPT-4V",
        "stage": 1,
        "domain": "general",
        "expert": "Projector (not experts)",
        "download_images_from_urls": True,
        "image_url_field": "image",
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
        "download_images_from_urls": True,
        "image_url_field": "image",
        "size_gb": 6.0,
        "priority": "recommended",
        "samples": "700K",
    },
    # NOTE: COCO captions from HuggingFaceM4/COCO
    "coco_captions": {
        "hf_name": "HuggingFaceM4/COCO",
        "config": "2014_captions",
        "description": "COCO image captions for general visual understanding",
        "stage": 1,
        "domain": "general",
        "expert": "Projector (not experts)",
        "preferred_download": "snapshot",
        "size_gb": 3.0,
        "priority": "recommended",
        "samples": "118K",
    },
    "conceptual_captions": {
        "hf_name": "google-research-datasets/conceptual_captions",
        "config": "unlabeled",
        "description": "Large-scale image captioning (diverse domains)",
        "stage": 1,
        "domain": "general",
        "expert": "Projector (not experts)",
        "download_images_from_urls": True,
        "image_url_field": "image_url",
        "size_gb": 12.0,
        "priority": "optional",
        "samples": "3.3M",
    },

    # =========================================================================
    # STAGE 2: VISION/OCR EXPERTS (Expert 0 & 1)
    # For reading text, understanding documents, diagrams
    # =========================================================================

    "textvqa": {
        "hf_name": "lmms-lab/textvqa",
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
    # NOTE: InfographicVQA removed - exists as subset in DocVQA/Cauldron, not standalone
    # "infovqa": {
    #     "hf_name": "lmms-lab/InfographicVQA",
    #     "config": "default",
    #     "description": "Infographics understanding (NOT AVAILABLE - use DocVQA subset)",
    #     "stage": 2,
    #     "domain": "vision_diagram",
    #     "expert": "Expert 1: vision_diagram",
    #     "size_gb": 2.5,
    #     "priority": "optional",
    #     "samples": "30K",
    # },
    # REPLACED: OCR-VQA path could not be verified - commented out
    # "ocrvqa": {
    #     "hf_name": "howard-hou/OCR-VQA",
    #     "config": "default",
    #     "description": "OCR on book covers, signs, products (PATH NOT VERIFIED)",
    #     "stage": 2,
    #     "domain": "vision_ocr",
    #     "expert": "Expert 0: vision_ocr",
    #     "size_gb": 4.0,
    #     "priority": "optional",
    #     "samples": "200K",
    # },
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
        "download_images_from_urls": True,
        "image_url_field": "imgname",
        "size_gb": 1.0,
        "priority": "critical",
        "samples": "32K",
    },
    # NOTE: PlotQA alternative path (lmms-lab version doesn't exist)
    "plotqa": {
        "hf_name": "achang/plot_qa",
        "config": "default",
        "description": "Scientific plot understanding (alternative source)",
        "stage": 2,
        "domain": "code_math_chart",
        "expert": "Expert 2: code_math_chart",
        "size_gb": 8.0,
        "priority": "optional",
        "samples": "224K",
    },
    # NOTE: FigureQA doesn't exist as standalone - exists in HuggingFaceM4/the_cauldron
    # "figureqa": {
    #     "hf_name": "lmms-lab/FigureQA",
    #     "config": "default",
    #     "description": "Figure understanding and visual reasoning (NOT AVAILABLE)",
    #     "stage": 2,
    #     "domain": "code_math_chart",
    #     "expert": "Expert 2: code_math_chart",
    #     "size_gb": 5.0,
    #     "priority": "optional",
    #     "samples": "180K",
    # },
    # NOTE: DVQA doesn't exist as standalone - exists in larger collections
    # "dvqa": {
    #     "hf_name": "lmms-lab/DVQA",
    #     "config": "default",
    #     "description": "Data visualization QA (NOT AVAILABLE)",
    #     "stage": 2,
    #     "domain": "code_math_chart",
    #     "expert": "Expert 2: code_math_chart",
    #     "size_gb": 3.0,
    #     "priority": "optional",
    #     "samples": "300K",
    # },
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
        "requires_coco_images": True,
    },
    # NOTE: Visual Genome doesn't exist as standalone in this format
    # "visual_genome": {
    #     "hf_name": "lmms-lab/VisualGenome",
    #     "config": "default",
    #     "description": "Dense scene annotations and relationships (NOT AVAILABLE)",
    #     "stage": 2,
    #     "domain": "spatial_scene",
    #     "expert": "Expert 4: spatial_scene",
    #     "size_gb": 15.0,
    #     "priority": "optional",
    #     "samples": "108K images",
    # },
    # NOTE: Visual Genome - Using ranjaykrishna/visual_genome (region_descriptions_v1.2.0)
    "visual_genome_region": {
        "hf_name": "ranjaykrishna/visual_genome",
        "config": "region_descriptions",
        "description": "Dense region descriptions and relationships",
        "stage": 2,
        "domain": "spatial_scene",
        "expert": "Expert 4: spatial_scene",
        "preferred_download": "snapshot",
        "size_gb": 10.0,
        "priority": "optional",
        "samples": "108K",
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
    # NOTE: RefCOCO - lmms-lab formatted version
    "refcoco": {
        "hf_name": "lmms-lab/RefCOCO",
        "config": "default",
        "description": "Referring expressions and visual grounding",
        "stage": 2,
        "domain": "spatial_scene",
        "expert": "Expert 4: spatial_scene",
        "size_gb": 2.3,
        "priority": "optional",
        "samples": "142K",
    },

    # =========================================================================
    # STAGE 2: ACTION & EMBODIED AI DATASETS
    # For action recognition, instruction following, navigation
    # =========================================================================

    "nlvr2": {
        "hf_name": "lmms-lab/NLVR2",
        "config": "default",
        "description": "Natural language visual reasoning (pair images)",
        "stage": 2,
        "domain": "spatial_reasoning",
        "expert": "Expert 5: spatial_reasoning",
        "size_gb": 3.0,
        "priority": "recommended",
        "samples": "107K",
    },
    "vsr": {
        "hf_name": "cambridgeltl/vsr_random",
        "config": "default",
        "description": "Visual Spatial Reasoning dataset",
        "stage": 2,
        "domain": "spatial_reasoning",
        "expert": "Expert 5: spatial_reasoning",
        "download_images_from_urls": True,
        "image_url_field": "image",
        "size_gb": 0.5,
        "priority": "optional",
        "samples": "10K",
    },
    "winoground": {
        "hf_name": "facebook/winoground",
        "config": "default",
        "description": "Visio-linguistic compositional reasoning",
        "stage": 2,
        "domain": "agentic_reasoning",
        "expert": "Expert 7: agentic_reasoning",
        "size_gb": 0.2,
        "priority": "optional",
        "samples": "800",
        "requires_hf_token": True,
    },
    # NOTE: VCR is not available on HuggingFace Hub; requires manual download.
    # "visual_commonsense": {
    #     "hf_name": "visual_commonsense",
    #     "config": None,
    #     "description": "Visual Commonsense Reasoning with rationales (manual download)",
    #     "stage": 2,
    #     "domain": "agentic_reasoning",
    #     "expert": "Expert 7: agentic_reasoning",
    #     "size_gb": 6.0,
    #     "priority": "recommended",
    #     "samples": "290K",
    # },

    # NOTE: Visual7W path could not be verified
    # "visual7w": {
    #     "hf_name": "nlphuji/visual7w",
    #     "config": None,
    #     "description": "Visual question answering with pointing (PATH NOT VERIFIED)",
    #     "stage": 2,
    #     "domain": "spatial_reasoning",
    #     "expert": "Expert 5: spatial_reasoning",
    #     "size_gb": 3.0,
    #     "priority": "optional",
    #     "samples": "47K",
    # },
    # NOTE: CLEVR doesn't exist as standalone in lmms-lab
    # "clevr": {
    #     "hf_name": "lmms-lab/CLEVR",
    #     "config": "default",
    #     "description": "Compositional visual reasoning (NOT AVAILABLE)",
    #     "stage": 2,
    #     "domain": "agentic_reasoning",
    #     "expert": "Expert 7: agentic_reasoning",
    #     "size_gb": 18.0,
    #     "priority": "optional",
    #     "samples": "850K",
    # },
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


def _is_text_feature(feature) -> bool:
    from datasets import Sequence, Value

    if isinstance(feature, Value) and feature.dtype == "string":
        return True
    if isinstance(feature, Sequence):
        return _is_text_feature(feature.feature)
    if isinstance(feature, dict):
        return any(_is_text_feature(sub_feature) for sub_feature in feature.values())
    return False


def _is_image_feature(feature) -> bool:
    from datasets import Image, Sequence

    if isinstance(feature, Image):
        return True
    if isinstance(feature, Sequence):
        return _is_image_feature(feature.feature)
    if isinstance(feature, dict):
        return any(_is_image_feature(sub_feature) for sub_feature in feature.values())
    return False


def _zip_contains(base_dir: Path, rel_path: Path) -> bool:
    import zipfile

    rel_posix = rel_path.as_posix()
    for zip_path in base_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                if rel_posix in zf.namelist():
                    return True
        except Exception:
            continue
    return False


def _image_value_present(value, base_dir: Path | None = None) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, Path)):
        value_str = str(value)
        if value_str.startswith("http://") or value_str.startswith("https://"):
            return True
        path = Path(value_str)
        if path.is_absolute() and path.exists():
            return True
        if path.exists():
            return True
        if base_dir is not None:
            if (base_dir / path).exists():
                return True
            if _zip_contains(base_dir, path):
                return True
        return False
    if isinstance(value, dict):
        url_value = value.get("url") or value.get("image_url")
        if url_value:
            return True
        path = value.get("path") or value.get("filename")
        if path:
            path = Path(path)
            if path.is_absolute() and path.exists():
                return True
            if path.exists():
                return True
            if base_dir is not None:
                if (base_dir / path).exists():
                    return True
                if _zip_contains(base_dir, path):
                    return True
        return value.get("bytes") is not None
    if hasattr(value, "convert"):
        return True
    return True


def _image_columns(features) -> List[str]:
    from datasets import Value

    columns = []
    for name, feature in features.items():
        if _is_image_feature(feature):
            columns.append(name)
        elif isinstance(feature, Value) and feature.dtype == "string":
            if name in {"image", "image_path", "img", "picture", "image_url", "image_uri", "url"}:
                columns.append(name)
    return columns


def _load_dataset_with_fallback(
    hf_name: str,
    config_name: str | None = None,
    token: str | None = None,
):
    """Load a dataset from Hugging Face with robust config fallback."""
    from datasets import get_dataset_config_names, load_dataset

    errors = []
    candidates: List[str | None] = []

    if config_name:
        candidates.append(config_name)

    candidates.append(None)

    try:
        for cfg in get_dataset_config_names(hf_name):
            if cfg not in candidates:
                candidates.append(cfg)
    except Exception:
        pass

    for cfg in candidates:
        try:
            if cfg is None:
                ds = load_dataset(hf_name, token=token)
            else:
                ds = load_dataset(
                    hf_name,
                    cfg,
                    token=token,
                )
            return ds, cfg
        except Exception as exc:
            errors.append(f"config={cfg!r}: {exc}")

    joined = " | ".join(errors) if errors else "unknown error"
    raise RuntimeError(f"Could not load dataset '{hf_name}' with any config. {joined}")


def _extract_archives_in_dir(dataset_dir: Path) -> int:
    """Extract archives found under dataset_dir. Returns count extracted."""
    import zipfile

    extracted = 0
    archive_patterns = ["*.zip", "*.tar", "*.tar.gz", "*.tgz", "*.tar.bz2", "*.tbz2"]
    archive_files = []
    for pattern in archive_patterns:
        archive_files.extend(dataset_dir.rglob(pattern))

    for archive_path in archive_files:
        target_dir = archive_path.parent
        try:
            suffixes = [s.lower() for s in archive_path.suffixes]
            if archive_path.suffix.lower() == ".zip":
                with zipfile.ZipFile(archive_path) as zf:
                    zf.extractall(target_dir)
                extracted += 1
            elif ".tar" in suffixes or archive_path.suffix.lower() in {".tgz", ".tbz2"}:
                with tarfile.open(archive_path) as tf:
                    tf.extractall(target_dir)
                extracted += 1
        except Exception as exc:
            print(f"Warning: Failed to extract {archive_path}: {exc}")

    return extracted


def _snapshot_download_dataset(
    hf_name: str,
    save_path: Path,
    token: str | None = None,
    extract_archives: bool = False,
) -> Dict[str, object]:
    """Download dataset repository files directly (zip/tar/parquet/etc)."""
    from huggingface_hub import snapshot_download

    save_path.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=hf_name,
        repo_type="dataset",
        local_dir=str(save_path),
        token=token,
    )

    extracted_count = 0
    if extract_archives:
        extracted_count = _extract_archives_in_dir(save_path)

    return {
        "snapshot_path": str(save_path.absolute()),
        "archives_extracted": extracted_count,
    }


def _download_image_from_url(url: str, save_path: Path, timeout: int = 30) -> bool:
    """Download a single image from URL with retry logic."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(response.read())
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
        return False


def _download_images_from_dataset(
    dataset,
    save_path: Path,
    url_field: str,
    max_workers: int = 8,
    max_images: Optional[int] = None,
) -> tuple[int, int]:
    """Download images from URLs in dataset to local directory."""
    images_dir = save_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    failed = 0
    
    def download_sample(idx_row):
        idx, row = idx_row
        if max_images and idx >= max_images:
            return None
            
        url = row.get(url_field)
        if not url or not isinstance(url, str):
            return False
            
        # Generate filename from URL hash to avoid collisions
        url_hash = hashlib.md5(url.encode()).hexdigest()
        ext = url.split('.')[-1].split('?')[0].lower()
        if ext not in {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}:
            ext = 'jpg'
        
        img_path = images_dir / f"{idx:08d}_{url_hash}.{ext}"
        
        if img_path.exists():
            return True
            
        return _download_image_from_url(url, img_path)
    
    print(f"  Downloading images from URLs (using {max_workers} workers)...")
    
    # Use ThreadPoolExecutor for concurrent downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, row in enumerate(dataset):
            if max_images and idx >= max_images:
                break
            futures.append(executor.submit(download_sample, (idx, row)))
        
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result is True:
                downloaded += 1
            elif result is False:
                failed += 1
            
            if (idx + 1) % 100 == 0:
                print(f"    Progress: {idx + 1}/{len(futures)} ({downloaded} ok, {failed} failed)")
    
    print(f"  ✓ Downloaded {downloaded} images, {failed} failed")
    return downloaded, failed


def _download_coco_images(save_path: Path, splits: List[str] = None) -> bool:
    """Download COCO images needed for datasets like GQA."""
    if splits is None:
        splits = ["train2017", "val2017"]
    
    images_dir = save_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Downloading COCO images for splits: {splits}")
    
    for split in splits:
        zip_url = f"http://images.cocodataset.org/zips/{split}.zip"
        zip_path = save_path / f"{split}.zip"
        
        if (images_dir / split).exists():
            print(f"    {split} already exists, skipping")
            continue
        
        try:
            print(f"    Downloading {split}.zip...")
            urllib.request.urlretrieve(zip_url, zip_path)
            
            print(f"    Extracting {split}.zip...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(images_dir)
            
            zip_path.unlink()
            print(f"    ✓ {split} downloaded and extracted")
        except Exception as e:
            print(f"    ✗ Failed to download {split}: {e}")
            return False
    
    return True


def download_dataset(
    dataset_key: str,
    output_dir: Path,
    force: bool = False,
    method: str = "auto",
    extract_archives: bool = False,
) -> bool:
    """
    Download a single dataset from HuggingFace.
    """
    if dataset_key not in DATASETS:
        print(f"ERROR: Unknown dataset: {dataset_key}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        return False

    info = DATASETS[dataset_key]
    hf_name = info["hf_name"]
    config_name = info.get("config", None)
    save_path = output_dir / dataset_key
    preferred_download = info.get("preferred_download", "datasets")
    allow_missing_images = bool(info.get("allow_missing_images", False))
    resolved_method = method
    if resolved_method == "auto":
        resolved_method = preferred_download
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )

    if info.get("requires_hf_token", False) and not hf_token:
        print(
            f"✗ {dataset_key} requires Hugging Face access token. "
            "Set HF_TOKEN and accept dataset access terms on the dataset page."
        )
        return False

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
    print(f"  Method:      {resolved_method}")
    print(f"{'='*70}\n")

    start_time = time.time()

    try:
        if resolved_method == "snapshot":
            snapshot_meta = _snapshot_download_dataset(
                hf_name,
                save_path,
                token=hf_token,
                extract_archives=extract_archives,
            )
            resolved_config = config_name
            image_columns = []
            has_text = True
            image_validation_passed = False
            sample_count = 0
            missing_images = 0
            splits = []
            num_samples = {}
            total_samples = 0
        else:
            # Load dataset from HuggingFace via datasets library
            ds, resolved_config = _load_dataset_with_fallback(
                hf_name,
                config_name,
                token=hf_token,
            )

            first_split = next(iter(ds.keys()))
            features = ds[first_split].features
            image_columns = _image_columns(features)
            has_text = _is_text_feature(features)
            image_validation_passed = True

            if not image_columns:
                if allow_missing_images:
                    image_validation_passed = False
                    print(f"! Warning: No image feature found in dataset: {hf_name}")
                else:
                    raise ValueError(f"No image feature found in dataset: {hf_name}")
            if not has_text:
                raise ValueError(f"No text feature found in dataset: {hf_name}")

            sample_count = min(32, len(ds[first_split]))
            missing_images = 0
            for idx in range(sample_count):
                row = ds[first_split][idx]
                has_image = any(_image_value_present(row.get(col), save_path) for col in image_columns)
                if not has_image:
                    missing_images += 1
            if sample_count > 0 and missing_images == sample_count:
                if allow_missing_images:
                    image_validation_passed = False
                    print(f"! Warning: No images present in sample rows for dataset: {hf_name}")
                else:
                    raise ValueError(f"No images present in sample rows for dataset: {hf_name}")

            # Save to disk
            save_path.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(save_path))
            splits = list(ds.keys())
            num_samples = {split: len(ds[split]) for split in ds.keys()}
            total_samples = sum(len(ds[split]) for split in ds.keys())
            snapshot_meta = {}
            
            # Download images from URLs if needed
            if info.get("download_images_from_urls", False):
                url_field = info.get("image_url_field", "image")
                print(f"\n  Dataset contains image URLs - downloading actual images...")
                downloaded_count = 0
                failed_count = 0
                
                for split_name in ds.keys():
                    print(f"  Processing split: {split_name}")
                    down, fail = _download_images_from_dataset(
                        ds[split_name],
                        save_path,
                        url_field=url_field,
                        max_workers=16,
                        max_images=10000 if info.get("priority") == "optional" else None,
                    )
                    downloaded_count += down
                    failed_count += fail
                
                if downloaded_count > 0:
                    image_validation_passed = True
                    print(f"\n  ✓ Downloaded {downloaded_count} images total")
                    snapshot_meta["images_downloaded"] = downloaded_count
                    snapshot_meta["images_failed"] = failed_count
            
            # Download COCO images if needed (for datasets like GQA)
            if info.get("requires_coco_images", False):
                print(f"\n  Dataset requires COCO images - downloading...")
                coco_success = _download_coco_images(save_path)
                if coco_success:
                    image_validation_passed = True
                    print(f"  ✓ COCO images downloaded")
                    snapshot_meta["coco_images_downloaded"] = True
                else:
                    print(f"  ✗ Failed to download COCO images")
                    snapshot_meta["coco_images_downloaded"] = False

        download_time = time.time() - start_time

        # Save detailed metadata
        metadata = {
            "dataset_key": dataset_key,
            "hf_name": hf_name,
            "config": resolved_config,
            "requested_config": config_name,
            "domain": info["domain"],
            "expert": info.get("expert", "N/A"),
            "stage": info["stage"],
            "priority": info["priority"],
            "description": info["description"],
            "download_method": resolved_method,
            "image_columns": image_columns,
            "text_feature_present": has_text,
            "image_samples_checked": sample_count,
            "image_samples_missing": missing_images,
            "image_validation_passed": image_validation_passed,
            "splits": splits,
            "num_samples": num_samples,
            "total_samples": total_samples,
            "estimated_size_gb": info["size_gb"],
            "download_time_seconds": round(download_time, 2),
            "save_path": str(save_path.absolute()),
            "download_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        metadata.update(snapshot_meta)
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Successfully saved {dataset_key} to {save_path}")
        print(f"  Download time: {download_time:.1f} seconds")

        # Special handling for llava_instruct_150k - auto extract if needed
        if dataset_key == "llava_instruct_150k":
            zip_file = save_path / "train2017.zip"
            if zip_file.exists():
                print(f"  Extracting images from train2017.zip...")
                try:
                    import zipfile
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(save_path)
                    print(f"  ✓ Images extracted successfully")
                except Exception as e:
                    print(f"  ! Warning: Failed to extract images: {e}")
                    print(f"    You can manually extract {zip_file}")

        return True

    except Exception as e:
        err_msg = str(e)
        if resolved_method == "datasets" and "Dataset scripts are no longer supported" in err_msg:
            print(f"! {dataset_key}: datasets API blocked script-based loading; retrying with snapshot method")
            try:
                start_time = time.time()
                save_path.mkdir(parents=True, exist_ok=True)
                snapshot_meta = _snapshot_download_dataset(
                    hf_name,
                    save_path,
                    token=hf_token,
                    extract_archives=extract_archives,
                )
                download_time = time.time() - start_time

                metadata = {
                    "dataset_key": dataset_key,
                    "hf_name": hf_name,
                    "config": config_name,
                    "requested_config": config_name,
                    "domain": info["domain"],
                    "expert": info.get("expert", "N/A"),
                    "stage": info["stage"],
                    "priority": info["priority"],
                    "description": info["description"],
                    "download_method": "snapshot",
                    "image_columns": [],
                    "text_feature_present": True,
                    "image_samples_checked": 0,
                    "image_samples_missing": 0,
                    "image_validation_passed": False,
                    "splits": [],
                    "num_samples": {},
                    "total_samples": 0,
                    "estimated_size_gb": info["size_gb"],
                    "download_time_seconds": round(download_time, 2),
                    "save_path": str(save_path.absolute()),
                    "download_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                metadata.update(snapshot_meta)
                with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                print(f"✓ Successfully saved {dataset_key} to {save_path} (snapshot fallback)")
                print(f"  Download time: {download_time:.1f} seconds")
                return True
            except Exception as fallback_e:
                print(f"✗ FAILED fallback snapshot for {dataset_key}: {fallback_e}")
                return False

        print(f"✗ FAILED to download {dataset_key}: {e}")
        return False


def download_multiple_datasets(
    dataset_keys: List[str],
    output_dir: Path,
    force: bool = False,
    method: str = "auto",
    extract_archives: bool = False,
) -> Dict[str, bool]:
    """
    Download multiple datasets.
    Returns a dict mapping dataset_key -> success status.
    """
    results = {}
    total = len(dataset_keys)

    print(f"\n{'='*70}")
    print(f"DOWNLOADING {total} DATASETS")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Total estimated size: ~{get_total_size(dataset_keys):.1f} GB")
    print(f"{'='*70}\n")

    for i, dataset_key in enumerate(dataset_keys, 1):
        print(f"\n[{i}/{total}] Starting download: {dataset_key}")
        success = download_dataset(
            dataset_key,
            output_dir,
            force,
            method=method,
            extract_archives=extract_archives,
        )
        results[dataset_key] = success

        if success:
            print(f"✓ [{i}/{total}] Completed: {dataset_key}")
        else:
            print(f"✗ [{i}/{total}] Failed: {dataset_key}")

    return results


def print_dataset_list():
    """Print all available datasets with details."""
    print(f"\n{'='*70}")
    print("AVAILABLE DATASETS")
    print(f"{'='*70}\n")

    # Group by stage
    stage1 = {k: v for k, v in DATASETS.items() if v["stage"] == 1}
    stage2 = {k: v for k, v in DATASETS.items() if v["stage"] == 2}

    print("STAGE 1 - Vision-Language Alignment:")
    print("-" * 70)
    for key, info in stage1.items():
        print(f"  {key:25s} {info['samples']:>8s} samples  ~{info['size_gb']:>4.1f} GB  [{info['priority']}]")
        print(f"    {info['hf_name']}")

    print(f"\nSTAGE 2 - Expert Specialization:")
    print("-" * 70)

    # Group by domain
    domains = {}
    for key, info in stage2.items():
        domain = info["domain"]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append((key, info))

    for domain in sorted(domains.keys()):
        print(f"\n  {domain.upper().replace('_', ' ')}:")
        for key, info in domains[domain]:
            print(f"    {key:23s} {info['samples']:>8s} samples  ~{info['size_gb']:>4.1f} GB  [{info['priority']}]")
            print(f"      {info['hf_name']}")

    print(f"\n{'='*70}")
    print(f"Total: {len(DATASETS)} datasets")
    print(f"  Critical: {len(CRITICAL_DATASETS)} datasets (~{get_total_size(CRITICAL_DATASETS):.1f} GB)")
    print(f"  Recommended: {len(RECOMMENDED_DATASETS)} datasets (~{get_total_size(RECOMMENDED_DATASETS):.1f} GB)")
    print(f"  All: {len(ALL_DATASETS)} datasets (~{get_total_size(ALL_DATASETS):.1f} GB)")
    print(f"{'='*70}\n")


def explain_alignment():
    """Explain vision-language alignment and expert training."""
    print(f"\n{'='*70}")
    print("VISION-LANGUAGE ALIGNMENT EXPLAINED")
    print(f"{'='*70}\n")

    print("""
EmberNet uses a two-stage training approach:

STAGE 1 - PROJECTOR ALIGNMENT (Vision-Language Connection):
-----------------------------------------------------------
Goal: Teach the model to "see" by connecting visual features to language.

What happens:
1. SigLIP vision encoder extracts visual features (196 tokens) [FROZEN]
2. Token compression reduces to 64 tokens [TRAINABLE]
3. Projector MLP maps vision → language space [TRAINABLE]
4. Language model generates text [MOSTLY FROZEN]

Why: The projector learns that "this visual pattern" = "the word dog"

Datasets: LLaVA-Instruct, ShareGPT4V, ALLaVA, COCO Captions
Training: 3-5 epochs, learning rate ~1e-3


STAGE 2 - EXPERT SPECIALIZATION (Domain-Specific Skills):
---------------------------------------------------------
Goal: Train domain experts to be good at specific visual tasks.

What happens:
1. Vision encoder + Projector [FROZEN - already aligned]
2. MoE Router learns to route tokens to experts [TRAINABLE]
3. Domain experts specialize on their data [TRAINABLE]

Why: Different tasks need different skills. OCR needs different 
     processing than chart analysis or spatial reasoning.

Expert Assignment:
  - Expert 0: OCR, text reading (TextVQA, DocVQA)
  - Expert 1: Diagrams, infographics (AI2D)
  - Expert 2: Charts, graphs (ChartQA, PlotQA)
  - Expert 3: Math, formulas (MathVista)
  - Expert 4: Scene understanding (VQAv2, RefCOCO)
  - Expert 5: Spatial reasoning (GQA, NLVR2)
  - Expert 6: World knowledge (OK-VQA, A-OKVQA)
  - Expert 7: Complex reasoning (ScienceQA, VCR)

Datasets: Domain-specific datasets for each expert
Training: 10-15 epochs, learning rate ~1e-4

Result: A tiny VLM that can handle diverse visual tasks efficiently!
""")

    print(f"{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download datasets for EmberNet VLM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all --output-dir ./data
  %(prog)s --recommended --output-dir ./data
  %(prog)s --minimal --output-dir ./data
  %(prog)s --dataset textvqa chartqa --output-dir ./data
  %(prog)s --list
  %(prog)s --explain
        """
    )

    # Action arguments
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets (~110GB)"
    )
    action_group.add_argument(
        "--recommended",
        action="store_true",
        help="Download critical + recommended datasets (~80GB)"
    )
    action_group.add_argument(
        "--critical",
        action="store_true",
        help="Download only critical datasets (~50GB)"
    )
    action_group.add_argument(
        "--minimal",
        action="store_true",
        help="Download minimal datasets for testing (~40GB)"
    )
    action_group.add_argument(
        "--dataset",
        nargs="+",
        metavar="NAME",
        help="Download specific datasets by name"
    )
    action_group.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets and exit"
    )
    action_group.add_argument(
        "--explain",
        action="store_true",
        help="Explain vision-language alignment and exit"
    )

    # Configuration arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for datasets (default: ./data)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download datasets even if they exist"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "datasets", "snapshot"],
        help="Download method: auto (per-dataset), datasets (load+save), snapshot (raw repo files)"
    )
    parser.add_argument(
        "--extract-archives",
        action="store_true",
        help="When using snapshot method, extract zip/tar archives after download"
    )

    args = parser.parse_args()

    # Handle info commands
    if args.list:
        print_dataset_list()
        return

    if args.explain:
        explain_alignment()
        return

    # Check dependencies
    check_dependencies()

    # Determine which datasets to download
    if args.all:
        datasets_to_download = ALL_DATASETS
        mode = "ALL"
    elif args.recommended:
        datasets_to_download = RECOMMENDED_DATASETS
        mode = "RECOMMENDED"
    elif args.critical:
        datasets_to_download = CRITICAL_DATASETS
        mode = "CRITICAL"
    elif args.minimal:
        datasets_to_download = MINIMAL_DATASETS
        mode = "MINIMAL"
    elif args.dataset:
        datasets_to_download = args.dataset
        mode = "CUSTOM"
        # Validate dataset names
        invalid = [d for d in datasets_to_download if d not in DATASETS]
        if invalid:
            print(f"ERROR: Unknown datasets: {', '.join(invalid)}")
            print(f"Available datasets: {', '.join(DATASETS.keys())}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"EMBERNET DATASET DOWNLOADER")
    print(f"{'='*70}")
    print(f"Mode: {mode}")
    print(f"Datasets: {len(datasets_to_download)}")
    print(f"Output: {output_dir.absolute()}")
    print(f"Total size: ~{get_total_size(datasets_to_download):.1f} GB")
    print(f"{'='*70}\n")

    # Download datasets
    results = download_multiple_datasets(
        datasets_to_download,
        output_dir,
        args.force,
        method=args.method,
        extract_archives=args.extract_archives,
    )

    # Print summary
    successful = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]

    print(f"\n{'='*70}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {len(results)}")
    print(f"✓ Successful: {len(successful)}")
    print(f"✗ Failed: {len(failed)}")

    if failed:
        print(f"\nFailed datasets:")
        for dataset in failed:
            print(f"  ✗ {dataset}")

    print(f"\n{'='*70}")

    # Create download manifest
    manifest_path = output_dir / "download_manifest.json"
    manifest = {
        "download_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "download_method": args.method,
        "extract_archives": bool(args.extract_archives),
        "total_datasets": len(results),
        "successful": successful,
        "failed": failed,
        "datasets": {k: DATASETS[k] for k in datasets_to_download if k in DATASETS}
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Create dataset index consumed by training/data.py
    index_path = output_dir / "dataset_index.json"
    index = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "datasets": {},
    }
    for dataset_key in successful:
        info = dict(DATASETS[dataset_key])
        metadata_path = output_dir / dataset_key / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as meta_file:
                    metadata = json.load(meta_file)
                info["resolved_config"] = metadata.get("config")
                info["download_method"] = metadata.get("download_method")
                info["splits"] = metadata.get("splits", [])
                info["num_samples"] = metadata.get("num_samples", {})
            except Exception:
                pass
        index["datasets"][dataset_key] = info

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"\nDownload manifest saved to: {manifest_path}")
    print(f"Dataset index saved to: {index_path}")

    if successful:
        print(f"\n✓ Successfully downloaded {len(successful)} datasets to {output_dir.absolute()}")
        print("\nNext steps:")
        print("  1. Start Stage 1 training:")
        print("     python training/train.py --stage 1 --data-dir ./data --epochs 3")
        print("  2. After Stage 1 completes, run Stage 2:")
        print("     python training/train.py --stage 2 --data-dir ./data --epochs 10 --resume <checkpoint>")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

