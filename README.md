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

### Step 1: Install Dependencies

```bash
cd EmberNet
pip install -r requirements.txt
```

### Step 2: Download Training Data

```bash
# See all available datasets
python training/prepare_data.py --list

# Download recommended datasets (~70GB) - BEST BALANCE
python training/prepare_data.py --recommended

# OR download ALL datasets for maximum quality (~100GB)
python training/prepare_data.py --all

# OR minimal for quick testing (~10GB)
python training/prepare_data.py --minimal
```

### Step 3: Train the Model

```bash
# Stage 1: Vision-Language Alignment (3 epochs)
python training/train.py --stage 1 --data-dir ./data --epochs 3 --batch-size 8

# Stage 2: Expert Specialization (10 epochs)
python training/train.py --stage 2 --data-dir ./data --epochs 10 --batch-size 4
```

### Step 4: Run Inference

```bash
python inference/infer.py --model checkpoints/final_model.pt --interactive
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

| Stage | Minimum | Recommended |
|-------|---------|-------------|
| Stage 1 | 8GB VRAM | 16GB VRAM |
| Stage 2 | 12GB VRAM | 24GB VRAM |
| CPU-only | 16GB RAM | 32GB RAM |

### Training Commands

```bash
# Full training pipeline

# 1. Prepare data (do this first!)
python training/prepare_data.py --recommended --output-dir ./data

# 2. Stage 1: Projector alignment
python training/train.py \
    --stage 1 \
    --data-dir ./data \
    --epochs 3 \
    --batch-size 8 \
    --lr 1e-3 \
    --output-dir ./checkpoints

# 3. Stage 2: Expert fine-tuning (continues from Stage 1)
python training/train.py \
    --stage 2 \
    --data-dir ./data \
    --epochs 10 \
    --batch-size 4 \
    --lr 1e-4 \
    --resume ./checkpoints/checkpoint_epoch_3.pt \
    --output-dir ./checkpoints

# 4. Convert to optimized format
python inference/convert.py ./checkpoints/final_model.pt ./embernet_final.pt
```

### Training Tips

- **Out of memory?** Reduce `--batch-size` and increase `--grad-accum`
- **Slow training?** Make sure you're using GPU (`--device cuda`)
- **Resume training:** Use `--resume path/to/checkpoint.pt`
- **Monitor progress:** Check `checkpoints/` for saved models

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

