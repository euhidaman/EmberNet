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

**Option A: Recommended (~70GB)** - Best balance of quality and size
```bash
python training/prepare_data.py --recommended --output-dir ./data
```

**Option B: All Datasets (~100GB)** - Maximum quality
```bash
python training/prepare_data.py --all --output-dir ./data
```

**Option C: Minimal (~10GB)** - Quick testing only
```bash
python training/prepare_data.py --minimal --output-dir ./data
```

**Other useful commands:**
```bash
# List all available datasets before downloading
python training/prepare_data.py --list

# Download specific datasets only
python training/prepare_data.py --dataset textvqa chartqa --output-dir ./data

# Explain how alignment works
python training/prepare_data.py --explain
```

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

---

#### Step 4: Train the Model

**Stage 1: Projector Alignment (3 epochs, ~6-12 hours on GPU)**

Trains the vision-to-language projector to connect SigLIP visual features with the language model.

```bash
python training/train.py \
    --stage 1 \
    --data-dir ./data \
    --epochs 3 \
    --batch-size 8 \
    --lr 1e-3 \
    --output-dir ./checkpoints \
    --wandb-project EmberNet \
    --wandb-run-name stage1_projector
```

**Stage 2: Expert Specialization (10 epochs, ~24-48 hours on GPU)**

Trains domain-specific experts on specialized datasets.

```bash
python training/train.py \
    --stage 2 \
    --data-dir ./data \
    --epochs 10 \
    --batch-size 4 \
    --lr 1e-4 \
    --resume ./checkpoints/checkpoint_epoch_3.pt \
    --output-dir ./checkpoints \
    --wandb-project EmberNet \
    --wandb-run-name stage2_experts
```

**Training without W&B:**
```bash
# Disable Weights & Biases logging
python training/train.py --stage 1 --data-dir ./data --epochs 3 --batch-size 8 --no-wandb
```

**Adjust for your hardware:**
```bash
# If you get OOM (Out of Memory) errors:
python training/train.py --stage 1 --data-dir ./data --batch-size 2 --grad-accum 16

# Force CPU training (slower):
python training/train.py --stage 1 --data-dir ./data --device cpu

# Disable mixed precision (if issues):
python training/train.py --stage 1 --data-dir ./data --no-amp
```

---

#### Step 5: Convert Model (Optional)

Convert to optimized ternary format for deployment:

```bash
python inference/convert.py \
    ./checkpoints/final_model.pt \
    ./embernet_optimized.pt
```

This packs ternary weights to 2-bit representation, reducing model size to <500MB.

---

#### Step 6: Run Inference

**Interactive Mode:**
```bash
python inference/infer.py \
    --model ./checkpoints/final_model.pt \
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
    --model ./checkpoints/final_model.pt \
    --image photo.jpg \
    --prompt "What's in this image?"
```

---

### Quick Reference: All Commands in Order

```bash
# 1. Install
cd EmberNet
pip install -r requirements.txt

# 2. Authenticate
huggingface-cli login
wandb login

# 3. Download data
python training/prepare_data.py --recommended --output-dir ./data

# 4. Train Stage 1
python training/train.py --stage 1 --data-dir ./data --epochs 3 --batch-size 8

# 5. Train Stage 2
python training/train.py --stage 2 --data-dir ./data --epochs 10 --batch-size 4 --resume ./checkpoints/checkpoint_epoch_3.pt

# 6. Run inference
python inference/infer.py --model ./checkpoints/final_model.pt --interactive
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
| **LLaVA-Instruct-150K** ★ | `liuhaotian/LLaVA-Instruct-150K` | GPT-4 generated visual conversations | 150K | ~5GB |
| **ShareGPT4V** ★ | `Lin-Chen/ShareGPT4V` | Detailed image descriptions from GPT-4V | 100K | ~8GB |
| **ALLaVA** | `FreedomIntelligence/ALLaVA-4V` | Diverse visual instructions | 700K | ~6GB |

### Stage 2: Expert Specialization

#### Vision/OCR Expert Datasets

| Dataset | HuggingFace ID | Description | Samples | Size |
|---------|----------------|-------------|---------|------|
| **TextVQA** ★ | `textvqa` | Text in natural scenes | 45K | ~2GB |
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
| **VQAv2** ★ | `HuggingFaceM4/VQAv2` | General visual QA | 1.1M | ~25GB |
| **GQA** | `lmms-lab/GQA` | Scene graph reasoning | 22M | ~15GB |
| **Visual Genome** | `visual_genome` | Dense scene annotations | 108K | ~15GB |

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
    ├── Attention: GQA (12 heads, 6 KV heads)
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

## Project Structure

```
EmberNet/
├── models/
│   ├── bitnet_moe.py      # BitLinear + MoE decoder
│   ├── vision.py          # SigLIP encoder + compression
│   └── vlm.py             # Complete VLM
├── training/
│   ├── prepare_data.py    # Dataset download script
│   ├── data.py            # Data loading
│   └── train.py           # Training loop
├── inference/
│   ├── convert.py         # Model optimization
│   └── infer.py           # User interface
├── requirements.txt
└── README.md
```

---

## License

MIT License

