# EmberNet - Tiny BitNet MoE Vision-Language Model

A high-quality, tiny BitNet b1.58-style Mixture-of-Experts Vision-Language Model (~300M ternary parameters) designed for edge deployment.

**What it does:** You give it an image + a question, it gives you an intelligent answer. OCR, chart analysis, document understanding, visual reasoning - all in <500MB.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [How It Works](#how-it-works)
3. [Complete Dataset List](#complete-dataset-list)
4. [Training Guide](#training-guide)
5. [Usage Examples](#usage-examples)
6. [Architecture Details](#architecture-details)

---

## Quick Start

### Complete Setup & Training Guide

Follow these steps in order to set up and train EmberNet:

#### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd EmberNet

# Install all required packages
pip install -r requirements.txt
```

**Required packages:**
- `torch>=2.0.0` - Deep learning framework
- `transformers>=4.36.0` - HuggingFace transformers (for SigLIP)
- `datasets>=2.14.0` - HuggingFace datasets
- `wandb>=0.16.0` - Experiment tracking
- `huggingface_hub>=0.19.0` - HuggingFace authentication
- `Pillow>=9.0.0` - Image processing
- `einops>=0.7.0` - Tensor operations
- `numpy>=1.24.0` - Numerical computing

---

#### Step 2: Authentication Setup

**2a. HuggingFace Login** (REQUIRED - for downloading datasets):
```bash
# Login to HuggingFace
huggingface-cli login

# When prompted, enter your HuggingFace token
# Get your token from: https://huggingface.co/settings/tokens
# Create a token with "read" permissions
```

**2b. Weights & Biases Login** (OPTIONAL - for experiment tracking):
```bash
# Login to W&B
wandb login

# When prompted, enter your W&B API key
# Get your API key from: https://wandb.ai/authorize
# This enables training visualization and metric tracking
```

---

#### Step 3: Download Training Datasets

**Choose one option based on your needs:**

**Option A: All Datasets (~100GB)** - Best quality, all datasets
```bash
python training/prepare_data.py --all --output-dir ./data
```
Downloads all 20 datasets for maximum model quality.

**Option B: Recommended (~70GB)** - Good balance of quality and size
```bash
python training/prepare_data.py --recommended --output-dir ./data
```
Downloads critical + recommended datasets (excludes optional ones).

**Option C: Critical Only (~50GB)** - Core datasets only
```bash
python training/prepare_data.py --critical --output-dir ./data
```
Downloads only the most important datasets.

**Option D: Minimal (~10GB)** - Quick testing
```bash
python training/prepare_data.py --minimal --output-dir ./data
```
Downloads a tiny subset for testing the pipeline.

**Other useful commands:**
```bash
# List all available datasets before downloading
python training/prepare_data.py --list

# Download specific datasets only
python training/prepare_data.py --dataset textvqa chartqa vqav2 --output-dir ./data

# Explain how alignment works
python training/prepare_data.py --explain
```

**What gets downloaded:**
- Stage 1 datasets: LLaVA-Instruct, ShareGPT4V, ALLaVA, COCO Captions, etc.
- Stage 2 datasets: TextVQA, DocVQA, AI2D, ChartQA, PlotQA, VQAv2, GQA, OK-VQA, A-OKVQA, ScienceQA, RefCOCO, NLVR2, VSR, and more

**Note:** Some datasets (ShareGPT4V, ALLaVA, ChartQA, GQA, VSR) download images from URLs, which may take longer than loading pre-packaged datasets.

**What gets saved:**
After downloading, you'll have:
- `./data/{dataset_name}/` - Each dataset in its own folder
- `./data/download_manifest.json` - **Central tracking file** with all download info
- `./data/dataset_index.json` - Index mapping datasets to stages/domains
- `./data/{dataset_name}/metadata.json` - Individual dataset metadata

The **download_manifest.json** tracks:
- All download sessions with timestamps
- Dataset metadata (samples, size, domain, expert)
- Download times and success/failure status
- File paths and HuggingFace IDs

#### Step 4: Train the Model

EmberNet uses a **two-stage training pipeline**. Use `--trial` for quick validation or `--main` for full training.

---

**🚀 Quick Start: Single-Command Training**

```bash
# Trial Run - Quick pipeline validation (minutes, not hours)
python training/train.py --trial --data-dir ./data

# Main Run - Full production training (hours/days)
python training/train.py --main --data-dir ./data
```

Both commands automatically run **Stage 1 → Stage 2** sequentially.

---

**Trial Mode (`--trial`)**

Validates the entire pipeline with minimal data:

```bash
python training/train.py --trial --data-dir ./data
```

| Setting | Value |
|---------|-------|
| Samples per dataset | 50 |
| Epochs per stage | 1 |
| Batch size | 2 |
| Gradient accumulation | 1 |
| W&B logging | Disabled |
| Output | `./checkpoints/trial/stage{1,2}/` |

Use this to verify everything works before committing to full training.

---

**Main Mode (`--main`)**

Full production training with all data:

```bash
python training/train.py --main --data-dir ./data
```

| Setting | Stage 1 | Stage 2 |
|---------|---------|---------|
| Samples | ALL | ALL |
| Epochs | 3 | 10 |
| Batch size | 8 | 4 |
| Gradient accumulation | 4 | 4 |
| W&B logging | Enabled | Enabled |
| Output | `./checkpoints/stage1/` | `./checkpoints/stage2/` |

---

**Run a Specific Stage Only**

```bash
# Stage 1 only (projector alignment)
python training/train.py --trial --stage 1 --data-dir ./data

# Stage 2 only (expert specialization) - requires Stage 1 checkpoint
python training/train.py --trial --stage 2 --data-dir ./data \
    --resume ./checkpoints/trial/stage1/final_model.pt
```

---

**Training Options**

```bash
# Custom epochs and batch size
python training/train.py --main --data-dir ./data --epochs 5 --batch-size 4

# Limit samples per dataset (useful for debugging)
python training/train.py --main --data-dir ./data --max-samples-per-dataset 1000

# Disable specific features
python training/train.py --main --data-dir ./data \
    --no-wandb \          # Disable W&B logging
    --no-ema \            # Disable EMA
    --no-curriculum \     # Disable curriculum learning
    --no-adaptive-clip    # Disable adaptive gradient clipping

# Hardware adjustments
python training/train.py --main --data-dir ./data \
    --device cpu \        # Force CPU (slower)
    --no-amp \            # Disable mixed precision
    --num-workers 2       # Reduce data loading workers
```

---

**Output Structure**

After training completes:

```
checkpoints/
├── trial/                    # Trial mode outputs
│   ├── stage1/
│   │   ├── final_model.pt    # Stage 1 checkpoint
│   │   └── checkpoint_epoch_1.pt
│   └── stage2/
│       ├── final_model.pt    # Stage 2 checkpoint (final model)
│       └── checkpoint_epoch_1.pt
│
├── stage1/                   # Main mode outputs
│   ├── final_model.pt
│   ├── best_model.pt
│   └── checkpoint_epoch_*.pt
│
└── stage2/
    ├── final_model.pt        # ← Use this for inference
    ├── best_model.pt
    └── checkpoint_epoch_*.pt
```

---

**What Each Stage Does**

| Stage | Purpose | What Trains | What's Frozen |
|-------|---------|-------------|---------------|
| **Stage 1** | Vision-Language Alignment | CrossModal Projector, Pooler, Compressor | Vision Encoder, LM Decoder |
| **Stage 2** | Expert Specialization | MoE Router, Domain Experts | Vision Encoder, Projector, Embeddings |

---

#### Step 5: Convert Model (Optional)

Convert to optimized ternary format for deployment:

```bash
python inference/convert.py \
    ./checkpoints/stage2/final_model.pt \
    ./embernet_optimized.pt
```

This packs ternary weights to 2-bit representation, reducing model size to <500MB.

---

#### Step 6: Run Inference

**Interactive Mode:**
```bash
python inference/infer.py \
    --model ./checkpoints/stage2/final_model.pt \
    --interactive
```

**Interactive commands:**
- `/load image.jpg` - Load an image
- `/describe` - Describe the image
- `/ocr` - Extract text from image
- `/chart` - Analyze chart/graph
- `/clear` - Reset conversation
- `/quit` - Exit

**Single Query:**
```bash
python inference/infer.py \
    --model ./checkpoints/stage2/final_model.pt \
    --image photo.jpg \
    --prompt "What's in this image?"
```

---

### Quick Reference: All Commands in Order

```bash
# 1. Install dependencies
cd EmberNet
pip install -r requirements.txt

# 2. Authenticate (HF required, W&B optional)
huggingface-cli login   # Enter token from https://huggingface.co/settings/tokens
wandb login             # Optional: Enter API key from https://wandb.ai/authorize

# 3. Download training data
python training/prepare_data.py --recommended --output-dir ./data   # ~70GB
# OR: python training/prepare_data.py --all --output-dir ./data     # ~100GB (best quality)
# OR: python training/prepare_data.py --minimal --output-dir ./data # ~10GB (testing only)

# 4. Train the model (SINGLE COMMAND - runs Stage 1 → Stage 2 automatically)
python training/train.py --trial --data-dir ./data   # Quick test (~minutes)
# OR:
python training/train.py --main --data-dir ./data    # Full training (~hours/days)

# 5. Run interactive inference
python inference/infer.py --model ./checkpoints/stage2/final_model.pt --interactive
```

---

## How It Works

### Vision-Language Alignment Explained

EmberNet connects images to language through a carefully designed pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           IMAGE INPUT                                    │
│                              ↓                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ 1. VISION ENCODER (SigLIP - Frozen)                            │    │
│  │    • Pretrained on 400M image-text pairs from Google           │    │
│  │    • Extracts 196 visual tokens (14×14 grid)                   │    │
│  │    • Each token is a 768-dimensional feature vector            │    │
│  │    • Already "understands" visual concepts                     │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ 2. TOKEN COMPRESSION (Trainable)                               │    │
│  │    • Pixel Shuffle: 196 → 49 tokens (merges 2×2 neighbors)     │    │
│  │    • Adaptive Pooling: 49 → 64 tokens (learned queries)        │    │
│  │    • Preserves important visual information                    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ 3. PROJECTOR (BitLinear MLP) ← THIS IS WHERE ALIGNMENT HAPPENS │    │
│  │    • 2-layer MLP with ternary weights {-1, 0, +1}              │    │
│  │    • Maps: Vision embedding space → Language embedding space   │    │
│  │    • Trained in Stage 1 on image-caption pairs                 │    │
│  │    • After training, visual tokens "look like" word tokens     │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ 4. MERGED INPUT SEQUENCE                                       │    │
│  │    [BOS] [IMG₁] [IMG₂] ... [IMG₆₄] [User: Describe this] ...  │    │
│  │          └── visual tokens ──┘     └── text tokens ──────┘     │    │
│  │    The LLM processes both as if they're all "text"             │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ 5. BITNET MOE DECODER                                          │    │
│  │    • 16 transformer layers with ternary weights                │    │
│  │    • MoE FFN: 8 specialized experts + 1 shared expert          │    │
│  │    • Router sends tokens to relevant experts:                  │    │
│  │      - OCR expert for text reading                             │    │
│  │      - Chart expert for graphs                                 │    │
│  │      - Diagram expert for technical drawings                   │    │
│  │      - etc.                                                    │    │
│  │    • Generates response token by token                         │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│                        TEXT OUTPUT                                      │
│            "This image shows a bar chart comparing..."                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Two Training Stages?

**Stage 1 - Projector Alignment:**
- Goal: Teach the model to "see" by connecting visual features to language
- What's trained: Projector MLP + Token compression layers
- What's frozen: Vision encoder + Language decoder
- Data: High-quality image descriptions (LLaVA-Instruct, ShareGPT4V)
- Why: The projector needs to learn that "this visual pattern" = "the concept of a dog"

**Stage 2 - Expert Specialization:**
- Goal: Make experts good at specific tasks (OCR, charts, diagrams, etc.)
- What's trained: MoE experts + Router
- What's frozen: Vision encoder + Embeddings
- Data: Domain-specific datasets (TextVQA, ChartQA, DocVQA, etc.)
- Why: Different visual tasks need different skills - one expert can't do everything well

### Expert Architecture & Dataset Mapping

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    EMBERNET MoE EXPERT ARCHITECTURE                          │
│                                                                              │
│  ROUTING: Each token → TOP-2 Experts + SHARED Expert (always active)        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  EXPERT 0: vision_ocr                                                        │
│  ├─ Specialty: Read text in images, OCR, document parsing                   │
│  └─ Datasets: TextVQA, DocVQA, OCR-VQA                                      │
│                                                                              │
│  EXPERT 1: vision_diagram                                                    │
│  ├─ Specialty: Understand diagrams, infographics, technical drawings        │
│  └─ Datasets: AI2D, InfoVQA                                                 │
│                                                                              │
│  EXPERT 2: code_math_chart                                                   │
│  ├─ Specialty: Analyze charts, graphs, plots, data visualizations           │
│  └─ Datasets: ChartQA, PlotQA, FigureQA, DVQA                              │
│                                                                              │
│  EXPERT 3: code_math_formula                                                 │
│  ├─ Specialty: Handle math equations, formulas, numerical reasoning         │
│  └─ Datasets: MathVista                                                     │
│                                                                              │
│  EXPERT 4: spatial_scene                                                     │
│  ├─ Specialty: Scene understanding, object detection, descriptions          │
│  └─ Datasets: VQAv2, Visual Genome                                          │
│                                                                              │
│  EXPERT 5: spatial_reasoning                                                 │
│  ├─ Specialty: Spatial relationships, counting, positional reasoning        │
│  └─ Datasets: GQA                                                           │
│                                                                              │
│  EXPERT 6: agentic_knowledge                                                 │
│  ├─ Specialty: Knowledge-based QA, facts requiring world knowledge          │
│  └─ Datasets: OK-VQA, A-OKVQA                                               │
│                                                                              │
│  EXPERT 7: agentic_reasoning                                                 │
│  ├─ Specialty: Multi-step reasoning, logic, science questions               │
│  └─ Datasets: ScienceQA, CLEVR                                              │
│                                                                              │
│  SHARED EXPERT (Always Active)                                               │
│  ├─ Specialty: Common patterns, language generation, general knowledge      │
│  └─ Datasets: ALL datasets (learns shared representations)                  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

Example Routing:
  User: "What does this chart show? The title says Q3 Sales."
         │
         ├─> EXPERT 0 (vision_ocr)    - reads "Q3 Sales" text
         ├─> EXPERT 2 (chart)          - analyzes chart structure  
         └─> SHARED EXPERT             - general language/context
         
  Combined output: "This bar chart shows Q3 sales data..."
```

---

## Complete Dataset List

### Stage 1: Vision-Language Alignment

| Dataset | HuggingFace ID | Description | Samples | Size |
|---------|----------------|-------------|---------|------|
| **LLaVA-Instruct-150K** ★ | `lmms-lab/LLaVA-Instruct-150K` | GPT-4 generated visual conversations | 150K | ~5GB |
| **ShareGPT4V** ★ | `lmms-lab/ShareGPT4V` | Detailed image descriptions from GPT-4V | 100K | ~8GB |
| **ALLaVA** | `FreedomIntelligence/ALLaVA-4V` | Diverse visual instructions | 711K | ~6GB |

### Stage 2: Expert Specialization

#### Vision/OCR Expert Datasets

| Dataset | HuggingFace ID | Description | Samples | Size |
|---------|----------------|-------------|---------|------|
| **TextVQA** ★ | `lmms-lab/TextVQA` | Text in natural scenes | 45K | ~2GB |
| **DocVQA** ★ | `lmms-lab/DocVQA` | Documents, forms, receipts | 50K | ~3GB |
| **AI2D** ★ | `lmms-lab/ai2d` | Scientific diagrams | 15K | ~1.5GB |
| **InfoVQA** | `lmms-lab/InfographicVQA` | Infographics | 30K | ~2.5GB |
| **OCR-VQA** | `howard-hou/OCR-VQA` | Book covers, signs | 200K | ~4GB |

#### Code/Math/Chart Expert Datasets

| Dataset | HuggingFace ID | Description | Samples | Size |
|---------|----------------|-------------|---------|------|
| **ChartQA** ★ | `ahmed-masry/ChartQA` | Bar, line, pie charts | 32K | ~1GB |
| **MathVista** ★ | `AI4Math/MathVista` | Mathematical visual reasoning | 6K | ~1GB |
| **PlotQA** | `lmms-lab/PlotQA` | Scientific plots | 224K | ~8GB |
| **FigureQA** | `lmms-lab/FigureQA` | Figure understanding | 180K | ~5GB |
| **DVQA** | `lmms-lab/DVQA` | Data visualization | 300K | ~3GB |

#### Spatial/Scene Expert Datasets

| Dataset | HuggingFace ID | Description | Samples | Size |
|---------|----------------|-------------|---------|------|
| **VQAv2** ★ | `lmms-lab/VQAv2` | General visual QA | 1.1M | ~25GB |
| **GQA** | `lmms-lab/GQA` | Scene graph reasoning | 22M | ~15GB |
| **Visual Genome** | `lmms-lab/VisualGenome` | Dense scene annotations | 108K | ~15GB |

#### Reasoning Expert Datasets

| Dataset | HuggingFace ID | Description | Samples | Size |
|---------|----------------|-------------|---------|------|
| **ScienceQA** ★ | `derek-thomas/ScienceQA` | Science with diagrams | 21K | ~2GB |
| **OK-VQA** | `lmms-lab/OK-VQA` | Outside knowledge VQA | 14K | ~1GB |
| **A-OKVQA** | `lmms-lab/A-OKVQA` | Augmented knowledge VQA | 25K | ~1.5GB |
| **CLEVR** | `lmms-lab/CLEVR` | Compositional reasoning | 850K | ~18GB |

**★ = Critical (included in --critical mode)**

### Download Options

```bash
# Minimal (~10GB) - For quick testing only
python training/prepare_data.py --minimal
# Includes: llava_instruct_150k, textvqa, chartqa, vqav2

# Critical (~45GB) - Essential for a working model
python training/prepare_data.py --critical
# Includes: All ★ marked datasets above

# Recommended (~70GB) - Good quality model
python training/prepare_data.py --recommended
# Includes: Critical + recommended datasets

# All (~100GB) - Maximum quality
python training/prepare_data.py --all
# Includes: Everything
```

---

## Training Guide

### Hardware Requirements

| Stage | Minimum | Recommended | Notes |
|-------|---------|-------------|-------|
| Stage 1 | 8GB VRAM | 16GB VRAM | Can use CPU with 16GB RAM (slower) |
| Stage 2 | 12GB VRAM | 24GB VRAM | Can use CPU with 32GB RAM (slower) |

### Expected Training Time

| Stage | GPU (A100) | GPU (RTX 3090) | CPU (32 cores) |
|-------|------------|----------------|----------------|
| Stage 1 (3 epochs) | 6-12 hours | 12-24 hours | 3-5 days |
| Stage 2 (10 epochs) | 24-48 hours | 48-96 hours | 10-15 days |

### Training Commands Reference

**Basic Training (Recommended):**

```bash
# Stage 1: Projector Alignment
python training/train.py \
    --stage 1 \
    --data-dir ./data \
    --epochs 3 \
    --batch-size 8 \
    --output-dir ./checkpoints

# Stage 2: Expert Specialization  
python training/train.py \
    --stage 2 \
    --data-dir ./data \
    --epochs 10 \
    --batch-size 4 \
    --resume ./checkpoints/checkpoint_epoch_3.pt \
    --output-dir ./checkpoints
```

**With W&B Logging (Recommended):**

```bash
# Stage 1
python training/train.py \
    --stage 1 \
    --data-dir ./data \
    --epochs 3 \
    --batch-size 8 \
    --wandb-project EmberNet \
    --wandb-run-name stage1_projector

# Stage 2
python training/train.py \
    --stage 2 \
    --data-dir ./data \
    --epochs 10 \
    --batch-size 4 \
    --resume ./checkpoints/checkpoint_epoch_3.pt \
    --wandb-project EmberNet \
    --wandb-run-name stage2_experts
```

**Custom Learning Rate:**

```bash
python training/train.py \
    --stage 1 \
    --data-dir ./data \
    --epochs 3 \
    --lr 5e-4
```

**Resume from Checkpoint:**

```bash
python training/train.py \
    --stage 2 \
    --data-dir ./data \
    --resume ./checkpoints/checkpoint_epoch_5.pt
```

### Troubleshooting

**Out of Memory (OOM) Errors:**
```bash
# Reduce batch size and increase gradient accumulation
python training/train.py \
    --stage 1 \
    --data-dir ./data \
    --batch-size 2 \
    --grad-accum 16
# Effective batch size = 2 * 16 = 32
```

**Slow Training:**
```bash
# Check if GPU is being used
python training/train.py --stage 1 --data-dir ./data --device cuda

# Increase number of data loading workers
python training/train.py --stage 1 --data-dir ./data --num-workers 8
```

**Mixed Precision Issues:**
```bash
# Disable automatic mixed precision
python training/train.py --stage 1 --data-dir ./data --no-amp
```

**CPU Training:**
```bash
# Force CPU (much slower but works without GPU)
python training/train.py --stage 1 --data-dir ./data --device cpu --batch-size 1
```

### All Training Arguments

```
Training Settings:
  --stage {1,2}              Training stage (1=projector, 2=expert SFT)
  --epochs N                 Number of training epochs (default: 3)
  --batch-size N             Training batch size (default: 4)
  --lr LR                    Learning rate (auto-set: 1e-3 for stage1, 1e-4 for stage2)
  --grad-accum N             Gradient accumulation steps (default: 4)

Paths:
  --data-dir DIR             Data directory (default: ./data)
  --output-dir DIR           Output directory for checkpoints (default: ./checkpoints)
  --resume PATH              Resume from checkpoint

Hardware:
  --device DEVICE            Device: cuda/cpu/auto (default: auto)
  --no-amp                   Disable mixed precision training
  --num-workers N            Data loading workers (default: 4)

Experiment Tracking:
  --wandb                    Use W&B logging (default: True)
  --no-wandb                 Disable W&B logging
  --wandb-project NAME       W&B project name (default: EmberNet)
  --wandb-run-name NAME      W&B run name (default: auto-generated)
```

### What Gets Saved

During training, the following checkpoints are saved to `--output-dir`:

- `checkpoint_epoch_N.pt` - Saved after each epoch
- `checkpoint_step_N.pt` - Saved every N steps (configurable)
- `best_model.pt` - Best model based on validation loss
- `final_model.pt` - Final model after all epochs

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Training configuration
- Current epoch and step
- Best loss so far

### Monitoring Training

**With Weights & Biases:**
Visit https://wandb.ai/your-username/EmberNet to see:
- Real-time loss curves
- Learning rate schedules
- Token statistics
- Gradient norms
- Model parameters
- System metrics (GPU/CPU usage, memory)

**Console Output:**
```
======================================================================
Starting Stage 1 Training
======================================================================
Epochs: 3
Steps per epoch: 1000
Total steps: 3000
Batch size: 8
Gradient accumulation: 4
Effective batch size: 32
Learning rate: 0.001
Device: cuda

--- Token Statistics (per sample) ---
Total tokens:  2,048
  ├─ Image tokens: 64
  └─ Text tokens:  1,984

--- Token Statistics (per batch) ---
Total tokens:  16,384
  ├─ Image tokens: 512
  └─ Text tokens:  15,872

--- Total Training Tokens (all epochs) ---
Total tokens:  49,152,000
  ├─ Image tokens: 1,536,000
  └─ Text tokens:  47,616,000
======================================================================

Epoch 1/3 | Step 100 | Loss: 2.3456 | Avg Loss: 2.4567 | LR: 1.00e-03
```

---

## Usage Examples

### Python API

```python
from inference.infer import EmberVLM

# Load model
model = EmberVLM("checkpoints/embernet.pt")

# Basic image Q&A
response = model.chat(
    image="photo.jpg",
    prompt="What's in this image?"
)
print(response)

# OCR - Read text from image
response = model.chat(
    image="document.png",
    prompt="Extract all text from this document"
)

# Chart analysis
response = model.chat(
    image="chart.png",
    prompt="What does this chart show? What's the maximum value?"
)

# Multi-turn conversation (remembers context)
model.chat(image="scene.jpg", prompt="Describe this image")
model.chat(prompt="How many people are there?")  # Uses same image
model.chat(prompt="What are they doing?")        # Continues conversation

# Reset and start fresh
model.clear_history()
```

### Interactive CLI

```bash
python inference/infer.py --interactive

# Commands:
#   /load image.jpg  - Load an image
#   /describe        - Describe the image
#   /ocr             - Extract text
#   /chart           - Analyze as chart
#   /clear           - Reset conversation
#   /quit            - Exit
```

---

## Architecture Details

```
EmberNet VLM (~300M total parameters)
│
├── Vision Encoder: SigLIP-base (FROZEN)
│   ├── Parameters: ~85M (not counted in trainable)
│   ├── Input: 224×224 RGB image
│   └── Output: 196 tokens × 768 dims
│
├── Token Compression (TRAINABLE in Stage 1)
│   ├── Pixel Shuffle: 196 → 49 tokens
│   ├── Adaptive Pooling: 49 → 64 tokens
│   └── Parameters: ~2M
│
├── Projector (TRAINABLE in Stage 1)
│   ├── BitLinear MLP (768 → 768 → 768)
│   ├── Ternary weights {-1, 0, +1}
│   └── Parameters: ~3M
│
└── Language Decoder: BitNet MoE (TRAINABLE in Stage 2)
    ├── Layers: 16 transformer blocks
    ├── Hidden size: 768
    ├── Attention: GQA (12 heads, 6 KV heads) — all projections are BitLinear
    ├── MoE FFN:
    │   ├── 8 Domain Experts (top-2 routing)
    │   │   ├── vision_ocr, vision_diagram
    │   │   ├── code_math_chart, code_math_formula
    │   │   ├── spatial_reasoning (×2)
    │   │   └── agentic_reasoning (×2)
    │   └── 1 Shared Expert (always active)
    ├── All weights: Ternary {-1, 0, +1}
    └── Parameters: ~250M (50M active per forward pass)

Total Trainable: ~255M ternary parameters
Active per Forward: ~55M parameters
Model Size on Disk: <500MB
```

---

## Quantization Implementation Details

### Which modules are ternary vs. full-precision?

| Module | Precision | Notes |
|---|---|---|
| `BitNetAttention` — Q/K/V/O projections | **Ternary (1.58-bit)** | All four projections are `BitLinear` |
| `BitNetExpert` — gate / up / down projections | **Ternary (1.58-bit)** | All 8 domain experts + shared expert |
| `VisionProjector` — fc1, fc2 | **Ternary (1.58-bit)** | `BitLinear` MLP in `models/vision.py` |
| `PixelShuffleCompressor` — proj | **FP16** | Standard `nn.Linear`; small (~600K params) |
| `AdaptivePooler` — cross-attention | **FP16** | Standard `nn.MultiheadAttention` |
| `RMSNorm` layers (all) | **FP16** | Learnable scale only; ~negligible params |
| Token embeddings + LM head (tied) | **FP16** | `nn.Embedding` + tied `nn.Linear` |
| MoE router (`nn.Linear`) | **FP16** | Small (768 × 8); routing must stay precise |
| SigLIP vision encoder | **FP16, frozen** | Not counted in 255M trainable params |
| VA Refiner MLP classifier | **FP32 (small)** | 2-layer MLP, ~12K params; explicitly not ternary |

> **Summary**: ~250M decoder params (+ ~1M projector) are ternary. Non-quantized
> components (router, layernorms, embeddings, compressor) account for ≈25M FP16 params.
> The "BitNet-b1.58-style" description is accurate for all learnable weight matrices
> in the decoder and projector.

### How ternary quantization works at runtime

`BitLinear.forward()` in `models/bitnet_moe.py` applies two Straight-Through Estimator (STE) operations:

```python
# Activation quantization (per-token, 8-bit symmetric)
x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()

# Weight quantization (ternary: sign(W - mean(W)) * mean|W|)
w_quant = w + (weight_quant(w) - w).detach()
```

`weight_quant()` maps float weights to `{-scale, 0, +scale}` using the signed mean:
`u = sign(w - mean(w)) * mean(|w|)`.  The STE means gradients flow through
as if the quantization were identity — the underlying float weights are
updated continuously during training and re-quantized at each forward pass.

### On-disk packing (`inference/convert.py`)

After training, `convert_to_ternary()` calls `convert_bitlinear_to_ternary()` on
every `BitLinear` module.  For each:
1. `weight_quant(module.weight)` snaps weights to `{-scale, 0, +scale}`.
2. `pack_ternary_weights()` encodes `{-1, 0, +1}` as `{0b00, 0b01, 0b10}` and
   packs 4 values per byte (2 bits per weight).
3. A FP32 scalar `scale = mean(|w|)` is stored alongside.
4. The resulting `TernaryLinear` module pre-unpacks weights for fast inference.

Non-BitLinear layers (embeddings, layernorms, router, compressor) are quantized
to INT8 via `torch.quantization.quantize_dynamic` in the same pass.

---

## Project Structure

```
EmberNet/
├── models/
│   ├── bitnet_moe.py      # BitLinear + MoE decoder
│   ├── vision.py          # SigLIP encoder + compression
│   ├── va_refiner.py      # VA Refiner hallucination mitigation
│   └── vlm.py             # Complete VLM
├── training/
│   ├── prepare_data.py    # Dataset download script
│   ├── data.py            # Data loading
│   └── train.py           # Training loop
├── inference/
│   ├── convert.py         # Model optimization
│   └── infer.py           # User interface
├── visualizations/
│   ├── fig_architecture_overview.py  # Fig 1: architecture + params
│   ├── fig_ternary_stats.py          # Fig 2: ternary weight stats
│   ├── fig_moe_routing.py            # Fig 3: MoE routing patterns
│   ├── fig_latency_energy.py         # Fig 4: latency/energy benchmark
│   ├── fig_va_token_effects.py       # Fig 5: VA token-level dynamics
│   ├── fig_va_answer_level.py        # Fig 6: VA hallucination metrics
│   ├── fig_qualitative_grid.py       # Fig 7: qualitative examples
│   └── ...                           # existing training-viz scripts
├── generate_all_plots.py  # Master visualization orchestrator
├── requirements.txt
└── README.md
```

---

## Hallucination Mitigation (VA Refiner)

EmberNet ships an optional **Visual Absence Refiner (VA Refiner)** that detects and suppresses hallucinated tokens at inference time — particularly tokens that describe visual attributes (colors, objects, counts, spatial relations) that are not grounded in the image.

### How it works

The VA Refiner operates in three layers:

1. **Neuron-level monitoring** — Forward hooks are placed on `shared_expert.down_proj` in the BitNet MoE layers specified by `va_layer_indices` (default: layers 6–11). The top-K neuron activations are fed to a lightweight 2-layer MLP that predicts a per-step "visual hallucination score" `p_neuron ∈ [0, 1]`.

2. **Logit discrepancy** — A null baseline is computed once at the start of generation by running the decoder with image tokens zeroed out. At each step the L1 distance between the live logit distribution and the null baseline is collapsed to a score `p_logit = 1 − tanh(dist / scale)`. High similarity to the null baseline means the model is generating as if there were no image — a sign of hallucination.

3. **Temporal burst detection** — A sliding window tracks the blended score `p = α·p_neuron + (1−α)·p_logit`. If the window mean exceeds `va_burst_threshold`, a "burst" is declared and a soft log-prob penalty (`va_soft_penalty`) is added to every token in `VISUAL_KEYWORDS` until the window cools.

After generation, `calibrate_answer_prefix()` checks the mean VA score for the response. If it exceeds 0.70, the response is prefixed with *"I cannot see that clearly, but …"*. If it exceeds 0.40, the prefix is *"I might be wrong, but …"*.

### CLI usage

```bash
# Single inference with VA Refiner
python inference/infer.py \
  --model checkpoints/embernet.pt \
  --image photo.jpg \
  --prompt "What color is the car?" \
  --use-va-refiner \
  --va-threshold 0.70 \
  --va-burst-threshold 0.70 \
  --va-soft-penalty 5.0 \
  --va-alpha 0.5
```

### Python API

```python
from inference.infer import EmberVLM

model = EmberVLM(
    model_path="checkpoints/embernet.pt",
    use_va_refiner=True,
    va_threshold=0.70,
    va_burst_threshold=0.70,
    va_soft_penalty=5.0,
    va_alpha=0.5,
)
response = model.chat(image="photo.jpg", prompt="What color is the car?")
print(response)
```

### Configuration knobs

| Argument | Default | Description |
|---|---|---|
| `--use-va-refiner` | off | Enable VA Refiner |
| `--va-threshold` | 0.70 | VA score threshold per token (non-visual) |
| `--va-burst-threshold` | 0.70 | Window mean that triggers burst mode |
| `--va-soft-penalty` | 5.0 | Log-prob penalty during bursts |
| `--va-alpha` | 0.5 | Blend: 1.0 = pure neuron, 0.0 = pure logit discrepancy |

Fine-grained knobs (`va_layer_indices`, `va_neuron_k`, `va_window_size`, `va_decay_factor`, `va_logit_scale`) can be set programmatically via `VARefinerConfig` in `models/va_refiner.py`.

### Trade-offs

- **Overhead**: one extra decoder forward pass (null baseline) at generation start, plus per-step hook collection. On a single A100 this adds ~5–10% latency per response.
- **Conservative by design**: the refiner penalises rather than blocks tokens, so factual accuracy is preserved. Use `va_soft_penalty < 3.0` for a lighter touch.

---

## Visualization Suite

EmberNet ships a publication-grade visualization suite in `visualizations/`.
Generate all figures or individual ones via `generate_all_plots.py`.

### Quick commands

```bash
# Generate ALL paper figures (Figs 1–7, synthetic data, no model required)
python generate_all_plots.py --paper-only

# Generate ALL plots (training-viz + paper figures)
python generate_all_plots.py --all

# Single paper figure
python generate_all_plots.py --fig fig_ternary_stats

# With a real checkpoint for data-driven figures
python generate_all_plots.py --paper-only --model checkpoints/stage2/final_model.pt
```

### Figure catalogue

| Figure | Script | Description |
|---|---|---|
| Fig 1 | `fig_architecture_overview.py` | Pipeline block diagram + parameter/bitwidth breakdown per component |
| Fig 2 | `fig_ternary_stats.py` | Per-layer ternary weight sparsity heatmaps and {-1,0,+1} composition bars |
| Fig 3 | `fig_moe_routing.py` | MoE expert routing frequency matrix across vision-language datasets |
| Fig 4 | `fig_latency_energy.py` | Latency, energy, and throughput: ternary vs FP16 baseline (±std bars) |
| Fig 5 | `fig_va_token_effects.py` | Per-token p_VA trajectories with burst-mode and penalisation markers |
| Fig 6 | `fig_va_answer_level.py` | Answer-level hallucination rate with/without VA Refiner, by visual category |
| Fig 7 | `fig_qualitative_grid.py` | Multi-domain qualitative panel: baseline vs VA-refined answers + p_VA sparklines |

All figures are saved as both `.pdf` (vector, for LaTeX) and `.png` (300 DPI) in `plots/paper_figures/`.

---

## License

MIT License

