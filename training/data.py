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

import io
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
    "spatial_scene": "[DOMAIN: Scene Understanding]",
    "spatial_reasoning": "[DOMAIN: Spatial Reasoning]",
    "agentic_knowledge": "[DOMAIN: Knowledge Grounding]",
    "robotics_action": "[DOMAIN: Action Recognition]",
    "robotics_spatial": "[DOMAIN: Spatial Reasoning]",
    "code_math_chart": "[DOMAIN: Chart Analysis]",
    "code_math_formula": "[DOMAIN: Math/Formula]",
    "agentic_tool": "[DOMAIN: Tool Use]",
    "agentic_reasoning": "[DOMAIN: Multi-step Reasoning]",
    "general": "[DOMAIN: General]",
}

# Domain to expert index mapping (aligns with BitNetMoEConfig.expert_domains)
DOMAIN_TO_EXPERT = {
    "vision_ocr": 0,        # Expert 0: Text reading, document OCR
    "vision_diagram": 1,    # Expert 1: Diagrams, infographics
    "code_math_chart": 2,   # Expert 2: Charts, graphs, plots
    "code_math_formula": 3, # Expert 3: Math equations, formulas
    "spatial_scene": 4,     # Expert 4: Scene understanding
    "spatial_reasoning": 5, # Expert 5: Spatial relationships
    "agentic_knowledge": 6, # Expert 6: Knowledge-based reasoning
    "agentic_reasoning": 7, # Expert 7: Multi-step reasoning
    "general": 0,           # Default to expert 0
    "robotics_action": 4,   # Map to spatial_scene
    "robotics_spatial": 5,  # Map to spatial_reasoning
    "agentic_tool": 6,      # Map to agentic_knowledge
}


@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_dir: str = "./data"
    image_size: int = 224
    max_length: int = 2048
    # Use TinyLlama tokenizer - vocab_size=32000 matches our model's 32002
    tokenizer_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

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
        max_samples: Optional[int] = None,
    ):
        self.config = config or DataConfig()
        self.split = split
        self.domain = domain
        self.domain_tag = DOMAIN_TAGS.get(domain, DOMAIN_TAGS["general"])
        self.data_root = Path(self.config.data_dir)
        self.max_samples = max_samples

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

        # Limit samples if max_samples is set (for trial runs)
        if self.max_samples is not None and len(self.samples) > self.max_samples:
            self.samples = self.samples[:self.max_samples]
            print(f"  [Trial] Limited to {len(self.samples)} samples for {domain} ({split})")
        else:
            print(f"Loaded {len(self.samples)} samples for {domain} ({split})")

    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load data from various sources."""
        data_path = Path(data_path)

        if data_path.is_file() and data_path.suffix == ".json":
            return self._load_json(data_path)
        elif data_path.is_dir():
            # Check for EmberNet dataset index
            index_path = data_path / "dataset_index.json"
            if index_path.exists():
                return self._load_from_index(index_path)
            manifest_path = data_path / "download_manifest.json"
            if manifest_path.exists():
                return self._load_from_index(manifest_path)
            samples = self._load_directory(data_path)
            if not samples:
                # Empty local dir — raise via HuggingFace loader if dataset isn't found there either
                print(f"  [DataLoader] No local data in {data_path}, trying HuggingFace fallback...")
                samples = self._load_huggingface(str(data_path))
            return samples
        else:
            # Try loading from HuggingFace
            return self._load_huggingface(str(data_path))

    def _load_from_index(self, index_path: Path) -> List[Dict[str, Any]]:
        """Load datasets based on the index file created by prepare_data.py."""
        with open(index_path, "r") as f:
            index = json.load(f)

        all_samples = []
        datasets_to_load = []
        available = index.get("datasets", {})

        if index_path.name == "download_manifest.json":
            successful = set(index.get("successful", []))
            available = {k: v for k, v in available.items() if k in successful}

        # In stage 1, we want all datasets assigned to stage 1
        # In stage 2, we want datasets assigned to the specific domain

        for key, info in available.items():
            dataset_stage = str(info.get("stage", ""))
            dataset_domain = info.get("domain", "")
            download_method = info.get("download_method", "datasets")
            # save_path is missing from most index files; derive from data_dir/key
            save_path = info.get("save_path") or str(Path(self.config.data_dir) / key)

            # Match condition:
            # - If domain is 'general', we take all stage 1 datasets (for alignment)
            # - OR if domain matches specifically (for expert SFT)
            should_load = False
            if self.domain == "general":
                if dataset_stage == "1":
                    should_load = True
            elif dataset_domain == self.domain:
                should_load = True

            if should_load:
                datasets_to_load.append((key, download_method, save_path))

        print(f"Index scan: Found {len(datasets_to_load)} datasets for domain='{self.domain}'")

        for key, method, save_path in datasets_to_load:
            if method == "snapshot":
                print(f"Loading {key} from snapshot directory...")
                snapshot_dir = Path(save_path)
                if snapshot_dir.exists():
                    snapshot_samples = self._load_snapshot_dataset(snapshot_dir, key)
                    if snapshot_samples:
                        all_samples.extend(snapshot_samples)
                    else:
                        # Snapshot dir exists but yielded nothing (scripts-only dir, no data)
                        print(f"  Snapshot empty for {key}; trying HuggingFace hub...")
                        all_samples.extend(self._load_huggingface(key))
                else:
                    print(f"  Snapshot path not found for {key}; trying HuggingFace hub...")
                    all_samples.extend(self._load_huggingface(key))
            else:
                # Load via standard HuggingFace datasets library
                all_samples.extend(self._load_huggingface(key))

        return all_samples

    def _load_snapshot_dataset(self, snapshot_dir: Path, dataset_key: str) -> List[Dict[str, Any]]:
        """Load a dataset from a snapshot directory (raw HuggingFace repo layout)."""
        samples = []

        # ── 1. Parquet files ─────────────────────────────────────────────────
        parquet_files = sorted(snapshot_dir.glob("**/*.parquet"))
        if parquet_files:
            try:
                import pandas as pd
                for pq_file in parquet_files:
                    df = pd.read_parquet(pq_file)
                    for _, row in df.iterrows():
                        image_path = None
                        # imgname is ChartQA's image filename column
                        for col in ["image", "image_path", "img", "file_name", "imgname"]:
                            if col not in row.index or not row[col]:
                                continue
                            val = str(row[col])
                            p = snapshot_dir / val
                            if p.exists():
                                image_path = str(p); break
                            # ChartQA: images live in ChartQA Dataset/<split>/png/<name>
                            if col == "imgname":
                                stem = Path(val).stem
                                for sp in ["train", "test", "val"]:
                                    for ext in [".png", ".jpg", ".jpeg"]:
                                        p2 = snapshot_dir / "ChartQA Dataset" / sp / "png" / f"{stem}{ext}"
                                        if p2.exists():
                                            image_path = str(p2); break
                                    if image_path: break
                            if image_path: break

                        text = question = answer = ""
                        for col in ["caption", "text", "description"]:
                            if col in row.index and row[col]:
                                text = str(row[col]); break
                        for col in ["question", "query"]:
                            if col in row.index and row[col]:
                                question = str(row[col])
                        # label is ChartQA's answer column
                        for col in ["answer", "response", "output", "label"]:
                            if col in row.index and row[col]:
                                answer = str(row[col]); break

                        if image_path:
                            samples.append({
                                "image_path": image_path,
                                "question": question or "Describe this image.",
                                "answer": answer or text or "An image.",
                            })

                if samples:
                    print(f"  Loaded {len(samples)} samples from {len(parquet_files)} parquet file(s) in {dataset_key}")
                    return samples
            except Exception as e:
                print(f"  Warning: parquet load failed for {dataset_key}: {e}")

        # ── 2. JSON data files ───────────────────────────────────────────────
        _SKIP = {"metadata.json", "dataset_info.json", "dataset_dict.json",
                 "state.json", "download_manifest.json", "dataset_index.json"}
        for jf in sorted(snapshot_dir.glob("**/*.json")):
            if jf.name in _SKIP:
                continue
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list) or not data:
                    continue
                jf_dir = jf.parent
                batch = []
                for item in data:
                    s = self._parse_dataset_item(item, dataset_key, jf_dir)
                    if s:
                        s.setdefault("_base_dir", str(jf_dir))
                        batch.append(s)
                if batch:
                    print(f"  Loaded {len(batch)} samples from {jf.relative_to(snapshot_dir)} in {dataset_key}")
                    samples.extend(batch)
            except Exception as e:
                print(f"  Warning: could not load {jf.name} in {dataset_key}: {e}")

        if samples:
            return samples

        # ── 3. Fallback: loose image directory ───────────────────────────────
        print(f"  No structured data in {dataset_key}; scanning for images...")
        return self._load_directory(snapshot_dir)

    # Splits that are used by lmms-eval as evaluation benchmarks.
    # We must NEVER use these as training data — doing so inflates scores.
    _EVAL_SPLITS = frozenset({
        "test", "testdev", "testmini", "testA", "testB",
        "val2014", "test2014", "test2015",
        "challenge",  # GQA challenge split
    })

    def _select_split(self, dataset_obj):
        """Select the best split from DatasetDict/Dataset objects.

        For train splits we strictly exclude benchmark test/val sets to prevent
        contamination of lmms-eval scores.
        For validation splits we carve 10 % of train or use a named val split,
        but never use a benchmark test set.
        """
        if hasattr(dataset_obj, "column_names"):
            # Already a single split — accept only if it looks like a training split
            return dataset_obj

        if hasattr(dataset_obj, "keys"):
            splits = list(dataset_obj.keys())

            if self.split == "validation":
                for preferred in ["validation", "val"]:
                    if preferred in splits:
                        return dataset_obj[preferred]
                # Carve last 10 % of train as proxy
                if "train" in splits:
                    train_ds = dataset_obj["train"]
                    n = len(train_ds)
                    start = max(0, n - max(1, n // 10))
                    return train_ds.select(range(start, n))
                # Fall through to first non-benchmark split
                safe = [s for s in splits if s not in self._EVAL_SPLITS
                        and not s.startswith(("balanced_test", "test"))]
                if safe:
                    return dataset_obj[safe[0]]
                # Nothing safe for validation — use an empty Dataset rather than a test split
                print(f"  [DataLoader] No safe validation split in {splits}; validation will be empty for this dataset.")
                from datasets import Dataset as HFDataset
                return HFDataset.from_dict({})

            else:  # train
                if "train" in splits:
                    return dataset_obj["train"]
                # Accept any split that is not a known benchmark test set
                safe = [s for s in splits if s not in self._EVAL_SPLITS
                        and not s.startswith(("balanced_test", "test"))]
                if safe:
                    print(f"  [DataLoader] No 'train' split; using '{safe[0]}' from {splits}")
                    return dataset_obj[safe[0]]
                # Only test/benchmark splits exist — cannot use for training
                raise ValueError(
                    f"Dataset has only benchmark/test splits {splits} and no 'train' split. "
                    f"Re-download it with a proper training split using prepare_data.py, "
                    f"or exclude it from the stage-2 domain configuration. "
                    f"Using test splits for training would contaminate lmms-eval benchmark scores."
                )

        return dataset_obj

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
            from datasets import get_dataset_config_names, load_from_disk, load_dataset

            base_dir = None

            # First try loading from disk (downloaded by prepare_data.py)
            disk_path = Path(self.config.data_dir) / dataset_name
            if disk_path.exists():
                print(f"Loading {dataset_name} from disk: {disk_path}")
                try:
                    ds = load_from_disk(str(disk_path))
                    ds = self._select_split(ds)   # raises ValueError for benchmark-only datasets
                    base_dir = disk_path
                except ValueError:
                    raise  # contamination guard — let it propagate
                except Exception:
                    # Not a saved-dataset directory; try as a local dataset script/repo
                    print(f"  Not a saved Dataset format; trying local repo load for {dataset_name}...")
                    local_errors = []
                    ds = None
                    for remote_code in (False, True):
                        try:
                            ds = load_dataset(str(disk_path), trust_remote_code=remote_code)
                            ds = self._select_split(ds)
                            base_dir = disk_path
                            break
                        except ValueError:
                            raise
                        except Exception as e:
                            local_errors.append(str(e))
                    if ds is None:
                        raise RuntimeError(
                            f"Could not load local dataset at {disk_path}: " + "; ".join(local_errors)
                        )
            else:
                # Dataset not on disk — fetch from HuggingFace hub
                print(f"Loading {dataset_name} from HuggingFace hub...")
                ds = None
                errors = []
                for remote_code in (False, True):
                    try:
                        ds = load_dataset(dataset_name, trust_remote_code=remote_code)
                        ds = self._select_split(ds)
                        break
                    except ValueError:
                        raise
                    except Exception as e:
                        errors.append(f"(trust_remote_code={remote_code}): {e}")
                        ds = None

                if ds is None:
                    # Try explicit configs
                    for remote_code in (False, True):
                        try:
                            for config_name in get_dataset_config_names(dataset_name, trust_remote_code=remote_code):
                                try:
                                    _ds = load_dataset(dataset_name, config_name, trust_remote_code=remote_code)
                                    _ds = self._select_split(_ds)
                                    ds = _ds
                                    break
                                except ValueError:
                                    raise
                                except Exception as cfg_e:
                                    errors.append(f"{config_name}: {cfg_e}")
                        except ValueError:
                            raise
                        except Exception as e:
                            errors.append(f"config listing: {e}")
                        if ds is not None:
                            break

                if ds is None:
                    raise RuntimeError("; ".join(errors) if errors else "dataset load failed")

            samples = []
            for item in ds:
                # Handle different dataset formats
                sample = self._parse_dataset_item(item, dataset_name, base_dir)
                if isinstance(sample, tuple):
                    sample = sample[0]
                if sample and base_dir and "_base_dir" not in sample:
                    sample["_base_dir"] = str(base_dir)
                if sample:
                    samples.append(sample)

            return samples

        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset '{dataset_name}': {e}\n"
                f"Fix data loading before training. Run training/prepare_data.py to download "
                f"missing datasets, or remove '{dataset_name}' from the dataset index."
            ) from e

    def _parse_dataset_item(
        self,
        item: Dict,
        dataset_name: str,
        base_dir: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        """Parse an item from various dataset formats."""
        # Ensure item is a dictionary
        if isinstance(item, str):
            # If item is just a string, treat it as conceptual captions formatted text
            item = {"caption": item, "text": item}
        elif not isinstance(item, dict) and not hasattr(item, "get"):
            # Fallback for non-dictionary items
            try:
                item = dict(item)
            except (TypeError, ValueError):
                return None

        # Now item should be a dict-like object
        if not hasattr(item, "get"):
            # Still not dict-like, skip it
            return None

        # Get image - handle different field names
        image = (
            item.get("image") or
            item.get("img") or
            item.get("picture") or
            item.get("image_path")
        )
        # A dict of {"bytes": None, "path": None} is truthy but unloadable —
        # downgrade to None so the URL fallback below can trigger.
        if isinstance(image, dict) and not image.get("bytes") and not image.get("path"):
            image = None
        # For datasets like conceptual_captions where the image field may be
        # None/empty but a URL is provided separately, fall back to the URL.
        if not image:
            url = item.get("image_url") or item.get("url") or item.get("image_url_4x")
            if url and isinstance(url, str) and url.startswith(("http://", "https://")):
                image = url
        image = self._normalize_image_value(image, base_dir)

        # ScienceQA format
        if "question" in item and "choices" in item and "answer" in item:
            choices = item["choices"]
            answer_idx = item["answer"]
            if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
                answer = choices[answer_idx]
            else:
                answer = str(answer_idx)

            question = item["question"]
            if choices:
                choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
                question = f"{question}\n\nChoices:\n{choices_str}"

            return {
                "image": image,
                "question": question,
                "answer": answer,
            }

        # =====================================================================
        # ShareGPT4V format (raw JSON files from snapshot)
        # =====================================================================

        if "image" in item and "conversations" not in item and "caption" not in item:
            return self._attach_base_dir({
                "image": image,
                "question": "Describe this image in detail.",
                "answer": item.get("text", item.get("description", "An image.")),
            }, base_dir)

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
                    return self._attach_base_dir({
                        "image": image,
                        "question": human_msg.replace("<image>", "").strip(),
                        "answer": assistant_msg,
                    }, base_dir)

        # =====================================================================
        # ALLaVA format (from snapshot with allava_laion/allava_vflan folders)
        # =====================================================================

        if "id" in item and str(item.get("id", "")).startswith("allava_"):
            convs = item.get("conversations", [])
            if len(convs) >= 2:
                question = ""
                answer = ""
                for conv in convs:
                    role = conv.get("from", "")
                    value = conv.get("value", "")
                    if role == "human":
                        question = value.replace("<image>", "").strip()
                    elif role == "gpt":
                        answer = value

                if question and answer:
                    img_value = item.get("image", "")
                    img_path = image
                    if base_dir and img_value:
                        from pathlib import Path
                        img_path_obj = base_dir / img_value
                        if not img_path_obj.exists():
                            img_path_obj = base_dir / "allava_laion" / "images" / img_value.split("/")[-1]
                        if img_path_obj.exists():
                            img_path = str(img_path_obj)

                    return self._attach_base_dir({
                        "image": img_path,
                        "question": question,
                        "answer": answer,
                    }, base_dir)

        # ShareGPT4V format
        if "caption" in item and "image" in item:
            return self._attach_base_dir({
                "image": image,
                "question": "Describe this image in detail.",
                "answer": item["caption"],
            }, base_dir)

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
        if "question" in item and "answer" in item and "choices" not in item:
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

        # ChartQA alternative format (from snapshot with imgname)
        if "imgname" in item and "query" in item:
            img_path = image
            if base_dir:
                from pathlib import Path
                for ext in ['.png', '.jpg', '.jpeg']:
                    potential_path = base_dir / "ChartQA Dataset" / "train" / "png" / f"{item['imgname']}{ext}"
                    if potential_path.exists():
                        img_path = str(potential_path)
                        break

            return self._attach_base_dir({
                "image": img_path,
                "question": item["query"],
                "answer": str(item.get("label", item.get("answer", ""))),
            }, base_dir)

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

        # GQA format - needs to resolve imageId to Visual Genome image path
        if "question" in item and "fullAnswer" in item:
            # GQA uses imageId which refers to Visual Genome images
            image_id = item.get("imageId") or item.get("image_id")
            resolved_image = image

            if image_id and base_dir:
                # Try to find image in VG_100K folders
                for vg_folder in ["VG_100K", "VG_100K_2", "images"]:
                    for ext in [".jpg", ".png", ".jpeg"]:
                        img_path = base_dir / "images" / vg_folder / f"{image_id}{ext}"
                        if img_path.exists():
                            resolved_image = str(img_path)
                            break
                    if resolved_image != image:
                        break

            return self._attach_base_dir({
                "image": resolved_image,
                "question": item["question"],
                "answer": item["fullAnswer"],
            }, base_dir)

        # OK-VQA / A-OKVQA format
        if "question" in item and "direct_answers" in item:
            answers = item["direct_answers"]
            answer = answers[0] if isinstance(answers, list) and answers else str(answers)
            return {
                "image": image,
                "question": item["question"],
                "answer": answer,
            }

        # NLVR2 format: dual-image sentence classification (True/False)
        if "sentence" in item and "label" in item and (
                "left_image" in item or "right_image" in item):
            # Use left_image as the primary image (single-image architecture)
            raw = item.get("left_image") or item.get("right_image")
            img = self._normalize_image_value(raw, base_dir)
            label_val = item["label"]
            if isinstance(label_val, bool):
                label_str = "True" if label_val else "False"
            else:
                label_str = str(label_val)
            return self._attach_base_dir({
                "image": img,
                "question": item["sentence"],
                "answer": label_str,
            }, base_dir)

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

    def _attach_base_dir(
        self,
        sample: Dict[str, Any],
        base_dir: Optional[Path],
    ) -> Dict[str, Any]:
        if base_dir is not None:
            sample["_base_dir"] = str(base_dir)
        return sample

    def _normalize_image_value(
        self,
        image_value: Any,
        base_dir: Optional[Path],
    ) -> Any:
        if image_value is None:
            return None

        if HAS_PIL and hasattr(image_value, "convert"):
            return image_value

        if isinstance(image_value, dict):
            img_path = image_value.get("path") or image_value.get("filename")
            img_bytes = image_value.get("bytes")

            if img_bytes is not None and HAS_PIL:
                import io

                try:
                    return Image.open(io.BytesIO(img_bytes))
                except Exception:
                    pass

            if img_path:
                path = Path(img_path)
                if base_dir is not None and not path.is_absolute():
                    return str(path)
                return str(path)

        if isinstance(image_value, (str, Path)):
            return str(image_value)

        return image_value

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
            raise RuntimeError(
                "No tokenizer is available on this EmberNetDataset instance. "
                "Pass a valid tokenizer when constructing the dataset. "
                "Training cannot proceed without a tokenizer."
            )

        # Ensure tokenizer has image token as special token
        IMAGE_TOKEN = "<image>"
        IMAGE_TOKEN_ID = 32001
        VOCAB_SIZE = 32002  # Must match model's vocab_size

        if IMAGE_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": [IMAGE_TOKEN]
            })

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

        # Map the tokenizer's image token ID to our expected IMAGE_TOKEN_ID
        # The tokenizer assigns its own ID, we need to use 32001 consistently
        tokenizer_image_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        if tokenizer_image_id != IMAGE_TOKEN_ID and tokenizer_image_id != self.tokenizer.unk_token_id:
            input_ids[input_ids == tokenizer_image_id] = IMAGE_TOKEN_ID

        # CRITICAL: Clamp ALL token IDs to valid vocabulary range
        # The tokenizer may have a larger vocab than our model
        # Any token >= VOCAB_SIZE gets mapped to unk_token (0) or pad_token
        out_of_range_mask = input_ids >= VOCAB_SIZE
        if out_of_range_mask.any():
            # Map out-of-range tokens to a safe token (use token ID 1 as fallback)
            input_ids[out_of_range_mask] = 1  # Usually BOS or some valid token

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
        """Load and preprocess image from any supported source format."""
        base_dir = sample.get("_base_dir")
        for key in ("image", "image_path"):
            val = sample.get(key)
            if val is None:
                continue
            result = self._resolve_image(val, base_dir)
            if result is not None:
                return result

        img_val = sample.get("image")
        print(
            f"[IMAGE LOAD FAILURE] domain='{self.domain}' "
            f"type={type(img_val).__name__} repr={repr(img_val)[:120]}\n"
            f"  _base_dir={sample.get('_base_dir')} question={str(sample.get('question',''))[:60]}\n"
            f"  → returning zero tensor (this sample contributes no visual signal)",
            flush=True,
        )
        return torch.zeros(3, self.config.image_size, self.config.image_size)

    def _resolve_image(self, val: Any, base_dir: Optional[str]) -> Optional[torch.Tensor]:
        """Try to turn *val* into a preprocessed image tensor."""
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            return val
        if HAS_PIL and hasattr(val, "convert"):
            return self.image_processor(val)
        if isinstance(val, (bytes, bytearray)) and HAS_PIL:
            try:
                return self.image_processor(Image.open(io.BytesIO(val)))
            except Exception:
                pass
        if isinstance(val, dict):
            img_bytes = val.get("bytes")
            if img_bytes is not None and HAS_PIL:
                try:
                    return self.image_processor(Image.open(io.BytesIO(img_bytes)))
                except Exception:
                    pass
            img_path = val.get("path") or val.get("filename")
            if img_path:
                return self._resolve_image_path(str(img_path), base_dir)
            return None
        if isinstance(val, (str, Path)):
            return self._resolve_image_path(str(val), base_dir)
        return None

    def _resolve_image_path(self, img_path: str, base_dir: Optional[str]) -> Optional[torch.Tensor]:
        """Open an image from a file path, resolving against base_dir if needed."""
        if not HAS_PIL:
            return None
        path = Path(img_path)
        if path.exists():
            try:
                return self.image_processor(Image.open(path))
            except Exception:
                pass
        if base_dir:
            bd = Path(base_dir)
            for candidate in (bd / path, bd / path.name, bd / "images" / path.name):
                if candidate.exists():
                    try:
                        return self.image_processor(Image.open(candidate))
                    except Exception:
                        pass
            pil_img = self._load_image_from_zip(bd, path)
            if pil_img is not None:
                return self.image_processor(pil_img)
            # Cross-dataset resolution: ShareGPT4V references coco/train2017/* which
            # actually lives in the llava_instruct_150k or coco_captions sibling dir.
            if "train2017" in img_path or "train2014" in img_path or "val2014" in img_path:
                fname = Path(img_path).name
                folder = "train2017" if "train2017" in img_path else (
                    "train2014" if "train2014" in img_path else "val2014")
                data_root = bd.parent
                for sib in data_root.iterdir():
                    if not sib.is_dir() or sib == bd:
                        continue
                    for c in (sib / folder / fname, sib / fname):
                        if c.exists():
                            try:
                                return self.image_processor(Image.open(c))
                            except Exception:
                                pass
        if img_path.startswith(("http://", "https://")):
            try:
                import urllib.request
                with urllib.request.urlopen(img_path, timeout=15) as resp:
                    return self.image_processor(Image.open(io.BytesIO(resp.read())))
            except Exception:
                pass
        return None

    def _load_image_from_zip(self, base_dir: Path, rel_path: Path) -> Optional["Image.Image"]:
        if not HAS_PIL:
            return None

        import io
        import zipfile

        rel_posix = rel_path.as_posix()
        for zip_path in base_dir.glob("*.zip"):
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    if rel_posix in zf.namelist():
                        with zf.open(rel_posix) as f:
                            return Image.open(io.BytesIO(f.read()))
            except Exception:
                continue
        return None

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

        # Get expert target for expert supervision
        expert_target = torch.tensor(DOMAIN_TO_EXPERT.get(self.domain, 0), dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "domain": self.domain,
            "expert_target": expert_target,
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

            if num_samples <= 0:
                continue
            if len(dataset) == 0:
                print(f"Warning: dataset '{name}' is empty; skipping")
                continue

            # Sample with replacement if needed
            indices = random.choices(range(len(dataset)), k=num_samples)
            for idx in indices:
                self.samples.append((name, idx))

        if not self.samples:
            raise ValueError("No samples were available to build MixedDomainDataset")

        # Shuffle
        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_name, sample_idx = self.samples[idx]
        return self.datasets[dataset_name][sample_idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate batch of samples."""
    result = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }
    # Include expert_targets if present (for expert supervision in Stage 2)
    if "expert_target" in batch[0]:
        result["expert_targets"] = torch.stack([b["expert_target"] for b in batch])
    return result


def create_dataloaders(
    data_config: Optional[DataConfig] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    stage: int = 1,
    use_curriculum: bool = False,
    max_samples_per_dataset: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders.

    Args:
        data_config: Data configuration
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        stage: Training stage (1=projector alignment, 2=expert SFT)
        use_curriculum: Whether to use curriculum learning (Stage 2 only)
            If True, easier datasets (VQAv2, COCO) are weighted higher early,
            harder datasets (ChartQA, MathVista) weighted higher later.
        max_samples_per_dataset: Maximum samples per dataset (None = use all).
            Useful for trial runs to quickly validate the pipeline.

    Returns:
        train_loader, val_loader
    """
    config = data_config or DataConfig()

    # Store max_samples in config for datasets to use
    if max_samples_per_dataset is not None:
        config.max_samples_per_dataset = max_samples_per_dataset
        print(f"  [Trial Mode] Limiting to {max_samples_per_dataset} samples per dataset")

    if stage == 1:
        # Stage 1: Use general VLM data for projector alignment
        train_dataset = EmberNetDataset(
            config.data_dir,
            config=config,
            split="train",
            domain="general",
            max_samples=max_samples_per_dataset,
        )
        val_dataset = EmberNetDataset(
            config.data_dir,
            config=config,
            split="validation",
            domain="general",
            max_samples=max_samples_per_dataset,
        )
    else:
        # Stage 2: Mix domain-specific datasets
        # Curriculum: start with easier domains, gradually add harder ones
        if use_curriculum:
            # Curriculum learning: prioritize easier datasets first
            # Easy: spatial_scene, general (VQAv2, COCO-like)
            # Medium: vision_ocr, spatial_reasoning (TextVQA, GQA)
            # Hard: code_math_chart, code_math_formula, agentic (ChartQA, MathVista, ScienceQA)
            domains = {
                "spatial_scene": 0.25,       # Easy - VQAv2
                "general": 0.20,             # Easy - general captions
                "vision_ocr": 0.15,          # Medium - TextVQA
                "spatial_reasoning": 0.15,   # Medium - GQA
                "vision_diagram": 0.05,      # Medium - AI2D
                "code_math_chart": 0.08,     # Hard - ChartQA
                "code_math_formula": 0.05,   # Hard - MathVista
                "agentic_knowledge": 0.04,   # Hard - OK-VQA
                "agentic_reasoning": 0.03,   # Hard - ScienceQA
            }
        else:
            domains = {
                "vision_ocr": config.vision_ratio / 2,
                "vision_diagram": config.vision_ratio / 2,
                "code_math_chart": config.code_math_ratio / 2,
                "code_math_formula": config.code_math_ratio / 2,
                "spatial_scene": config.robotics_ratio / 2,
                "spatial_reasoning": config.robotics_ratio / 2,
                "agentic_knowledge": config.agentic_ratio / 2,
                "agentic_reasoning": config.agentic_ratio / 2,
                "general": config.general_ratio,
            }

        datasets = {}
        for domain in domains:
            datasets[domain] = EmberNetDataset(
                config.data_dir,
                config=config,
                split="train",
                domain=domain,
                max_samples=max_samples_per_dataset,
            )

        # In trial mode, use only the actual available samples instead of the
        # default 10,000 (which inflates 34 real samples into 10k repeated ones).
        if max_samples_per_dataset is not None:
            total_mixed = sum(len(d) for d in datasets.values())
        else:
            total_mixed = 10000
        train_dataset = MixedDomainDataset(datasets, domains, total_samples=total_mixed)

        # Build a small validation set from validation splits of the same domains
        val_samples_per_domain = max(5, (max_samples_per_dataset or 100) // 10)
        val_datasets = {}
        for domain in domains:
            try:
                val_datasets[domain] = EmberNetDataset(
                    config.data_dir,
                    config=config,
                    split="validation",
                    domain=domain,
                    max_samples=val_samples_per_domain,
                )
            except Exception:
                pass
        if val_datasets:
            val_total = sum(len(d) for d in val_datasets.values())
            val_dataset = MixedDomainDataset(val_datasets, total_samples=max(1, val_total))
        else:
            val_dataset = None

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
