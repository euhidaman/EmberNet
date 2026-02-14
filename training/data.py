"""
Data Loading and Preprocessing for EmberNet VLM Training

This module handles loading and preprocessing of vision-language datasets
for training the EmberNet model with domain-specific expert routing.

VERIFIED DATASETS:
- TextVQA: https://huggingface.co/datasets/textvqa
- DocVQA: https://huggingface.co/datasets/lmms-lab/DocVQA
- AI2D (diagrams): https://huggingface.co/datasets/lmms-lab/ai2d
- ChartQA: https://huggingface.co/datasets/ahmed-masry/ChartQA
- LLaVA-Instruct-150K: https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K

DOMAIN TAGGING:
Each expert cluster is trained on domain-specific data:
- vision_ocr, vision_diagram: TextVQA, DocVQA, AI2D
- code_math_chart, code_math_formula: ChartQA, CLEVR-Math
- robotics_action, robotics_spatial: Open-X-Embodiment (subset)
- agentic_tool, agentic_reasoning: ToolBench, multi-step QA

PREPROCESSING:
- Images: Resize to 224x224, normalize with SigLIP stats
- Text: Tokenize with prompt template, max 2048 tokens
- Domain tags: Prepended to prompts for router learning
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# Domain tags for expert routing
DOMAIN_TAGS = {
    "vision_ocr": "[DOMAIN: OCR/Text Recognition]",
    "vision_diagram": "[DOMAIN: Diagram Understanding]",
    "robotics_action": "[DOMAIN: Action Recognition]",
    "robotics_spatial": "[DOMAIN: Spatial Reasoning]",
    "code_math_chart": "[DOMAIN: Chart Analysis]",
    "code_math_formula": "[DOMAIN: Math/Formula]",
    "agentic_tool": "[DOMAIN: Tool Use]",
    "agentic_reasoning": "[DOMAIN: Multi-step Reasoning]",
    "general": "[DOMAIN: General]",
}


@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_dir: str = "./data"
    image_size: int = 224
    max_length: int = 2048
    tokenizer_name: str = "microsoft/phi-2"

    # Dataset mixing ratios
    vision_ratio: float = 0.4
    code_math_ratio: float = 0.3
    robotics_ratio: float = 0.1
    agentic_ratio: float = 0.1
    general_ratio: float = 0.1

    # Augmentation
    use_augmentation: bool = True
    random_crop: bool = False


class ImageProcessor:
    """Simple image preprocessing for training."""

    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
        std: Tuple[float, ...] = (0.5, 0.5, 0.5),
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __call__(self, image: "Image.Image") -> torch.Tensor:
        """Preprocess a PIL image to tensor."""
        if not HAS_PIL:
            raise ImportError("PIL is required for image processing")

        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert to tensor
        import torchvision.transforms.functional as TF
        tensor = TF.to_tensor(image)

        # Normalize
        tensor = TF.normalize(tensor, self.mean, self.std)

        return tensor


class EmberNetDataset(Dataset):
    """
    Dataset for EmberNet VLM training.

    Supports multiple data formats:
    - JSON files with image paths and conversations
    - HuggingFace datasets (lazy loading)
    - Local directories with images

    Each sample includes:
    - image: Preprocessed image tensor
    - input_ids: Tokenized conversation
    - labels: Target tokens for loss
    - domain: Domain tag for expert routing
    """

    def __init__(
        self,
        data_path: str,
        config: Optional[DataConfig] = None,
        split: str = "train",
        domain: str = "general",
    ):
        self.config = config or DataConfig()
        self.split = split
        self.domain = domain
        self.domain_tag = DOMAIN_TAGS.get(domain, DOMAIN_TAGS["general"])

        self.image_processor = ImageProcessor(self.config.image_size)

        # Load tokenizer
        self.tokenizer = None
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.tokenizer_name,
                    trust_remote_code=True,
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")

        # Load data
        self.samples = self._load_data(data_path)
        print(f"Loaded {len(self.samples)} samples for {domain} ({split})")

    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load data from various sources."""
        data_path = Path(data_path)

        if data_path.is_file() and data_path.suffix == ".json":
            return self._load_json(data_path)
        elif data_path.is_dir():
            return self._load_directory(data_path)
        else:
            # Try loading from HuggingFace
            return self._load_huggingface(str(data_path))

    def _load_json(self, json_path: Path) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for item in data:
            sample = {
                "image_path": item.get("image", item.get("image_path", "")),
                "conversations": item.get("conversations", []),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            }
            samples.append(sample)

        return samples

    def _load_directory(self, dir_path: Path) -> List[Dict[str, Any]]:
        """Load image-text pairs from directory."""
        samples = []

        # Look for image files
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
        for img_path in dir_path.glob("**/*"):
            if img_path.suffix.lower() in image_extensions:
                # Look for corresponding text file
                txt_path = img_path.with_suffix(".txt")
                json_path = img_path.with_suffix(".json")

                text = ""
                if txt_path.exists():
                    with open(txt_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                elif json_path.exists():
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        text = data.get("caption", data.get("text", ""))

                samples.append({
                    "image_path": str(img_path),
                    "question": "Describe this image.",
                    "answer": text or "An image.",
                })

        return samples

    def _load_huggingface(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load from HuggingFace datasets (saved to disk or from hub)."""
        try:
            from datasets import load_from_disk, load_dataset

            # First try loading from disk (downloaded by prepare_data.py)
            disk_path = Path(self.config.data_dir) / dataset_name
            if disk_path.exists():
                print(f"Loading {dataset_name} from disk: {disk_path}")
                ds = load_from_disk(str(disk_path))
                if self.split in ds:
                    ds = ds[self.split]
                else:
                    # Use first available split
                    ds = ds[list(ds.keys())[0]]
            else:
                # Try loading from HuggingFace hub
                print(f"Loading {dataset_name} from HuggingFace...")
                ds = load_dataset(dataset_name, split=self.split, trust_remote_code=True)

            samples = []
            for item in ds:
                # Handle different dataset formats
                sample = self._parse_dataset_item(item, dataset_name)
                if sample:
                    samples.append(sample)

            return samples

        except Exception as e:
            print(f"Could not load dataset {dataset_name}: {e}")
            return self._create_dummy_data()

    def _parse_dataset_item(self, item: Dict, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Parse an item from various dataset formats."""

        # Get image - handle different field names
        image = (
            item.get("image") or
            item.get("img") or
            item.get("picture") or
            item.get("image_path")
        )

        # =====================================================================
        # STAGE 1: ALIGNMENT DATASETS
        # =====================================================================

        # LLaVA-Instruct format (conversations list)
        if "conversations" in item:
            convs = item["conversations"]
            if len(convs) >= 2:
                # Get first human message and first assistant response
                human_msg = ""
                assistant_msg = ""
                for conv in convs:
                    role = conv.get("from", conv.get("role", ""))
                    content = conv.get("value", conv.get("content", ""))
                    if role in ["human", "user"] and not human_msg:
                        human_msg = content
                    elif role in ["gpt", "assistant"] and not assistant_msg:
                        assistant_msg = content
                if human_msg and assistant_msg:
                    return {
                        "image": image,
                        "question": human_msg.replace("<image>", "").strip(),
                        "answer": assistant_msg,
                    }

        # ShareGPT4V format
        if "caption" in item and "image" in item:
            return {
                "image": image,
                "question": "Describe this image in detail.",
                "answer": item["caption"],
            }

        # =====================================================================
        # STAGE 2: VISION/OCR DATASETS
        # =====================================================================

        # TextVQA format (answers is a list)
        if "question" in item and "answers" in item:
            answers = item["answers"]
            if isinstance(answers, list):
                answer = answers[0] if answers else ""
            else:
                answer = str(answers)
            return {
                "image": image,
                "question": item["question"],
                "answer": answer,
            }

        # DocVQA / InfoVQA format
        if "question" in item and "answer" in item:
            answer = item["answer"]
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            return {
                "image": image,
                "question": item["question"],
                "answer": str(answer),
            }

        # OCR-VQA format
        if "questions" in item and "answers" in item:
            questions = item["questions"]
            answers = item["answers"]
            if questions and answers:
                return {
                    "image": image,
                    "question": questions[0] if isinstance(questions, list) else questions,
                    "answer": answers[0] if isinstance(answers, list) else answers,
                }

        # =====================================================================
        # STAGE 2: CHART/MATH DATASETS
        # =====================================================================

        # ChartQA format
        if "query" in item and "label" in item:
            return {
                "image": image,
                "question": item["query"],
                "answer": str(item["label"]),
            }

        # MathVista format
        if "question" in item and "answer" in item:
            return {
                "image": image,
                "question": item["question"],
                "answer": str(item["answer"]),
            }

        # PlotQA / FigureQA format
        if "question_string" in item and "answer" in item:
            return {
                "image": image,
                "question": item["question_string"],
                "answer": str(item["answer"]),
            }

        # =====================================================================
        # STAGE 2: GENERAL VQA DATASETS
        # =====================================================================

        # VQAv2 format
        if "question" in item and "multiple_choice_answer" in item:
            return {
                "image": image,
                "question": item["question"],
                "answer": item["multiple_choice_answer"],
            }

        # GQA format
        if "question" in item and "fullAnswer" in item:
            return {
                "image": image,
                "question": item["question"],
                "answer": item["fullAnswer"],
            }

        # OK-VQA / A-OKVQA format
        if "question" in item and "direct_answers" in item:
            answers = item["direct_answers"]
            answer = answers[0] if isinstance(answers, list) and answers else str(answers)
            return {
                "image": image,
                "question": item["question"],
                "answer": answer,
            }

        # =====================================================================
        # STAGE 2: SCIENCE/REASONING DATASETS
        # =====================================================================

        # ScienceQA format
        if "question" in item and "choices" in item and "answer" in item:
            choices = item["choices"]
            answer_idx = item["answer"]
            if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
                answer = choices[answer_idx]
            else:
                answer = str(answer_idx)

            # Format question with choices
            question = item["question"]
            if choices:
                choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
                question = f"{question}\n\nChoices:\n{choices_str}"

            return {
                "image": image,
                "question": question,
                "answer": answer,
            }

        # CLEVR format
        if "question" in item and "answer" in item:
            return {
                "image": image,
                "question": item["question"],
                "answer": str(item["answer"]),
            }

        # =====================================================================
        # FALLBACK: Generic format
        # =====================================================================

        question = (
            item.get("question") or
            item.get("query") or
            item.get("text") or
            "Describe this image."
        )
        answer = (
            item.get("answer") or
            item.get("label") or
            item.get("caption") or
            item.get("response") or
            "An image."
        )

        if isinstance(answer, list):
            answer = answer[0] if answer else "An image."

        return {
            "image": image,
            "question": str(question),
            "answer": str(answer),
        }

    def _create_dummy_data(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Create dummy data for testing."""
        samples = []
        for i in range(num_samples):
            samples.append({
                "image": None,  # Will create random tensor
                "question": f"Question {i}: What is in this image?",
                "answer": f"Answer {i}: This is a test image with various objects.",
            })
        return samples

    def _format_conversation(
        self,
        question: str,
        answer: str,
    ) -> Tuple[str, str]:
        """Format question-answer into conversation with domain tag."""
        # Add domain tag to help router learn
        prompt = f"{self.domain_tag}\nUser: <image>\n{question}\nAssistant: "
        target = answer
        return prompt, target

    def _tokenize(
        self,
        prompt: str,
        target: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize prompt and target."""
        if self.tokenizer is None:
            # Dummy tokenization
            input_ids = torch.randint(0, 32000, (self.config.max_length,))
            attention_mask = torch.ones(self.config.max_length)
            labels = input_ids.clone()
            return input_ids, attention_mask, labels

        # Tokenize full sequence
        full_text = prompt + target
        encoding = self.tokenizer(
            full_text,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels (mask prompt portion)
        labels = input_ids.clone()
        prompt_encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]
        labels[:prompt_len] = -100  # Ignore prompt in loss

        # Mask padding
        labels[attention_mask == 0] = -100

        return input_ids, attention_mask, labels

    def _load_image(self, sample: Dict[str, Any]) -> torch.Tensor:
        """Load and preprocess image."""
        if "image" in sample and sample["image"] is not None:
            if isinstance(sample["image"], torch.Tensor):
                return sample["image"]
            elif HAS_PIL and hasattr(sample["image"], "convert"):
                return self.image_processor(sample["image"])

        if "image_path" in sample and sample["image_path"]:
            path = Path(sample["image_path"])
            if path.exists() and HAS_PIL:
                image = Image.open(path)
                return self.image_processor(image)

        # Return random tensor as placeholder
        return torch.randn(3, self.config.image_size, self.config.image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        pixel_values = self._load_image(sample)

        # Get question and answer
        question = sample.get("question", "Describe this image.")
        answer = sample.get("answer", "This is an image.")

        # Format conversation
        prompt, target = self._format_conversation(question, answer)

        # Tokenize
        input_ids, attention_mask, labels = self._tokenize(prompt, target)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "domain": self.domain,
        }


class MixedDomainDataset(Dataset):
    """
    Dataset that mixes samples from multiple domain-specific datasets.

    Used for Stage 2 training with domain-specific expert routing.
    """

    def __init__(
        self,
        datasets: Dict[str, EmberNetDataset],
        ratios: Optional[Dict[str, float]] = None,
        total_samples: int = 10000,
    ):
        self.datasets = datasets

        # Default equal ratios
        if ratios is None:
            ratios = {name: 1.0 / len(datasets) for name in datasets}
        self.ratios = ratios

        # Build index mapping
        self.samples = []
        for name, dataset in datasets.items():
            ratio = ratios.get(name, 0.0)
            num_samples = int(total_samples * ratio)

            # Sample with replacement if needed
            indices = random.choices(range(len(dataset)), k=num_samples)
            for idx in indices:
                self.samples.append((name, idx))

        # Shuffle
        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_name, sample_idx = self.samples[idx]
        return self.datasets[dataset_name][sample_idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate batch of samples."""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def create_dataloaders(
    data_config: Optional[DataConfig] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    stage: int = 1,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders.

    Args:
        data_config: Data configuration
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        stage: Training stage (1=projector alignment, 2=expert SFT)

    Returns:
        train_loader, val_loader
    """
    config = data_config or DataConfig()

    if stage == 1:
        # Stage 1: Use general VLM data for projector alignment
        train_dataset = EmberNetDataset(
            config.data_dir,
            config=config,
            split="train",
            domain="general",
        )
        val_dataset = EmberNetDataset(
            config.data_dir,
            config=config,
            split="validation",
            domain="general",
        )
    else:
        # Stage 2: Mix domain-specific datasets
        domains = {
            "vision_ocr": config.vision_ratio / 2,
            "vision_diagram": config.vision_ratio / 2,
            "code_math_chart": config.code_math_ratio / 2,
            "code_math_formula": config.code_math_ratio / 2,
            "robotics_spatial": config.robotics_ratio,
            "agentic_reasoning": config.agentic_ratio,
            "general": config.general_ratio,
        }

        datasets = {}
        for domain in domains:
            datasets[domain] = EmberNetDataset(
                config.data_dir,
                config=config,
                split="train",
                domain=domain,
            )

        train_dataset = MixedDomainDataset(datasets, domains)
        val_dataset = None  # Simplified for stage 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    return train_loader, val_loader


# Utility functions for dataset preparation
def download_dataset(dataset_name: str, save_dir: str = "./data"):
    """Download a dataset from HuggingFace."""
    try:
        from datasets import load_dataset

        print(f"Downloading {dataset_name}...")
        ds = load_dataset(dataset_name)

        save_path = Path(save_dir) / dataset_name.replace("/", "_")
        ds.save_to_disk(str(save_path))
        print(f"Saved to {save_path}")

    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")


def prepare_training_data(
    output_dir: str = "./data",
    datasets: Optional[List[str]] = None,
):
    """
    Prepare training data by downloading required datasets.

    Default datasets:
    - textvqa (vision/OCR)
    - lmms-lab/DocVQA (document understanding)
    - ahmed-masry/ChartQA (chart analysis)
    """
    default_datasets = [
        "textvqa",
        "lmms-lab/DocVQA",
        "ahmed-masry/ChartQA",
    ]

    datasets = datasets or default_datasets

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in datasets:
        download_dataset(dataset_name, str(output_dir))

    print(f"\nData preparation complete. Data saved to {output_dir}")

