"""
eval/embernet_lmms_adapter.py
==============================
lmms-eval adapter for EmberNet.

Place this file in a directory and pass that directory via --include-path
when calling lmms-eval, e.g.:

    python -m lmms_eval \
        --model embernet \
        --model_args pretrained=./checkpoints/trial/stage2/final_model.pt \
        --tasks textvqa,mme,chartqa,ai2d,scienceqa_img \
        --batch_size 1 \
        --include-path ./eval \
        --output_path ./eval_results

The adapter follows lmms-eval's Simple Model (legacy) interface because
EmberNet does not use a HF transformers chat template.

Supported benchmarks (matched to EmberNet's 8 expert domains):
    vision_ocr        : textvqa, docvqa, ocrvqa
    vision_diagram    : ai2d
    code_math_chart   : chartqa
    code_math_formula : mathvista
    spatial_scene     : vqav2
    spatial_reasoning : gqa
    agentic_knowledge : ok_vqa
    agentic_reasoning : scienceqa_img, clevr
    general VLM       : mme, mmmu, mmstar
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure EmberNet repo is importable regardless of CWD
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from lmms_eval.api.registry import register_model
    from lmms_eval.api.model import lmms
except ImportError as e:
    raise ImportError(
        "lmms-eval not found. Install it first:\n"
        "  git clone https://github.com/EvolvingLMMs-Lab/lmms-eval\n"
        "  cd lmms-eval && pip install -e '.[all]'\n"
        f"Original error: {e}"
    )

from loguru import logger  # lmms-eval ships loguru


# ---------------------------------------------------------------------------
# Task → expert domain mapping (informational, logged at eval start)
# ---------------------------------------------------------------------------
TASK_EXPERT_MAP = {
    "textvqa":       "vision_ocr",
    "docvqa":        "vision_ocr",
    "ocrvqa":        "vision_ocr",
    "ai2d":          "vision_diagram",
    "chartqa":       "code_math_chart",
    "mathvista":     "code_math_formula",
    "vqav2":         "spatial_scene",
    "gqa":           "spatial_reasoning",
    "ok_vqa":        "agentic_knowledge",
    "scienceqa_img": "agentic_reasoning",
    "clevr":         "agentic_reasoning",
    "mme":           "general",
    "mmmu":          "general",
    "mmstar":        "general",
    "seed_bench":    "general",
}


@register_model("embernet")
class EmberNetLMMS(lmms):
    """
    lmms-eval wrapper for EmberNet (BitNet b1.58 MoE VLM).

    model_args (passed via --model_args key=value,key=value):
        pretrained       : Path to checkpoint .pt file, or HF repo ID
                           e.g. pretrained=./checkpoints/trial/stage2/final_model.pt
                           e.g. pretrained=euhidaman/EmberNet-Trial
        device           : cuda / cpu / auto  (default: auto)
        dtype            : float32 / bfloat16 / float16  (default: bfloat16)
        max_new_tokens   : max tokens to generate  (default: 256)
        temperature      : sampling temperature   (default: 0.0 for greedy)
        top_p            : nucleus sampling       (default: 1.0)
        batch_size       : 1 (EmberNet doesn't support batched generate yet)
        tokenizer_name   : override tokenizer  (default: microsoft/phi-2)
        trust_remote_code: True / False  (default: True)
    """

    # ---- lmms-eval flags ----
    is_simple = True   # use doc_to_visual + doc_to_text interface

    def __init__(
        self,
        pretrained: str = "./checkpoints/trial/stage2/final_model.pt",
        device: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tokenizer_name: str = "microsoft/phi-2",
        trust_remote_code: bool = True,
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__()

        # ---- dtype ----
        _dtype_map = {
            "float32":   torch.float32,
            "bfloat16":  torch.bfloat16,
            "float16":   torch.float16,
        }
        self._torch_dtype = _dtype_map.get(dtype, torch.bfloat16)

        # ---- device ----
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._max_new_tokens = int(max_new_tokens)
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._batch_size = 1   # always 1

        logger.info(f"[EmberNet] Loading model from: {pretrained}")
        logger.info(f"[EmberNet] Device: {self._device} | dtype: {dtype}")

        # ---- load tokenizer ----
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- load model ----
        from models.vlm import EmberNetVLM, EmberNetConfig
        config = EmberNetConfig()
        self._model = EmberNetVLM(config)

        self._pretrained = pretrained
        if Path(pretrained).exists():
            # local checkpoint
            ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = self._model.load_state_dict(state, strict=False)
            if missing:
                logger.warning(f"[EmberNet] Missing keys ({len(missing)}): {missing[:5]}…")
            if unexpected:
                logger.warning(f"[EmberNet] Unexpected keys ({len(unexpected)}): {unexpected[:5]}…")
            logger.info(f"[EmberNet] Loaded local checkpoint (step {ckpt.get('global_step', '?')})")
        else:
            # HF Hub repo — download pytorch_model.bin
            logger.info(f"[EmberNet] Downloading from HF Hub: {pretrained}")
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(repo_id=pretrained, filename="pytorch_model.bin")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            self._model.load_state_dict(state, strict=False)
            logger.info(f"[EmberNet] Loaded HF Hub checkpoint")

        self._model = self._model.to(self._torch_dtype).to(self._device)
        self._model.eval()

        total = sum(p.numel() for p in self._model.parameters())
        logger.info(f"[EmberNet] Model ready — {total:,} parameters on {self._device}")

    # ------------------------------------------------------------------
    # Required lmms-eval properties
    # ------------------------------------------------------------------

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 4096

    @property
    def max_new_tokens(self):
        return self._max_new_tokens

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    # ------------------------------------------------------------------
    # generate_until — core evaluation method
    # ------------------------------------------------------------------

    def generate_until(self, requests) -> List[str]:
        """
        Generate responses for a list of lmms-eval requests.

        lmms-eval v0.5+ passes request.arguments as:
            (context, gen_kwargs, doc_to_visual_fn, doc_id, task, split)
        where doc_to_visual_fn is a *callable* that returns a list of PIL images
        when called with the document dict.  Older versions may pass a list
        directly.  We handle both cases robustly.
        """
        results = []

        for request in tqdm(requests, desc="EmberNet eval", dynamic_ncols=True):
            # lmms-eval ≥ 0.5 stores the full 6-tuple in request.arguments:
            #   (context, gen_kwargs, doc_to_visual_fn, doc_id, task, split)
            # Older versions or the base lm-eval-harness use request.args
            # (a 2-tuple: context, until).  Try the lmms-eval attribute first.
            try:
                args = request.arguments
            except AttributeError:
                args = request.args

            # --- Unpack positional arguments ---
            context       = args[0] if len(args) > 0 else ""
            gen_kwargs    = args[1] if len(args) > 1 else {}
            doc_to_visual = args[2] if len(args) > 2 else None
            doc_id        = args[3] if len(args) > 3 else None
            task          = args[4] if len(args) > 4 else None
            split         = args[5] if len(args) > 5 else None

            # --- Resolve visuals ---
            # doc_to_visual is task.doc_to_visual — a bound method that must be
            # called with the raw dataset doc from self.task_dict[task][split][doc_id].
            # This is the pattern used by every lmms-eval model implementation
            # (cambrians, vila, qwen_vl, vllm, etc.).
            visuals: list = []
            if callable(doc_to_visual):
                try:
                    doc = self.task_dict[task][split][doc_id]
                    visuals = doc_to_visual(doc) or []
                except Exception as _ve:
                    logger.warning(f"[EmberNet] doc_to_visual error (doc {doc_id}): {_ve}")
                    visuals = []
            elif isinstance(doc_to_visual, (list, tuple)):
                visuals = list(doc_to_visual)

            if not visuals:
                # Fallback: pull image field directly from the doc dict
                try:
                    doc = self.task_dict[task][split][doc_id]
                    _img = doc.get('image') or doc.get('img') or doc.get('images')
                    if _img is not None:
                        visuals = _img if isinstance(_img, list) else [_img]
                except Exception:
                    pass

            # --- Guard: gen_kwargs must be a dict ---
            if not isinstance(gen_kwargs, dict):
                gen_kwargs = {}

            # generation params from request, with instance defaults as fallback
            max_new = int(gen_kwargs.get("max_new_tokens", self._max_new_tokens))
            temp    = float(gen_kwargs.get("temperature", self._temperature))
            top_p   = float(gen_kwargs.get("top_p", self._top_p))

            # pick first image (EmberNet processes one image per forward pass)
            image = visuals[0] if visuals else None

            # Strip the <image> placeholder that tasks inject — EmberNet handles
            # the image via the vision encoder, not via a text token.
            prompt = context.replace("<image>", "").strip()
            if not prompt:
                prompt = "Describe the image."

            try:
                with torch.inference_mode():
                    response = self._model.generate(
                        image=image,
                        prompt=prompt,
                        max_new_tokens=max_new,
                        temperature=temp if temp > 0 else None,
                        top_p=top_p if temp > 0 else None,
                        do_sample=(temp > 0),
                    )
                response = response.strip() if isinstance(response, str) else ""
            except Exception as e:
                logger.warning(f"[EmberNet] generate error (doc {doc_id}): {e}")
                if image is None:
                    logger.warning(
                        f"[EmberNet] image was None — task={task}, split={split}, "
                        f"doc_id={doc_id}, doc_to_visual type={type(doc_to_visual).__name__}, "
                        f"task_dict has task={task in (self.task_dict or {})}"
                    )
                response = ""

            results.append(response)

        return results

    # ------------------------------------------------------------------
    # loglikelihood — needed by some tasks (VQAv2, ScienceQA MCQ etc.)
    # We return stub values; these tasks will fall back to generation.
    # ------------------------------------------------------------------

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        logger.warning(
            "[EmberNet] loglikelihood not implemented — returning stub (0.0, False). "
            "Use generation-based tasks (output_type: generate_until)."
        )
        return [(0.0, False)] * len(requests)

    def loglikelihood_rolling(self, requests):
        return [(0.0,)] * len(requests)

    # ------------------------------------------------------------------
    # generate_until_multi_round — required abstract method in newer
    # versions of lmms-eval for multi-turn / interleaved evaluation.
    # EmberNet doesn't support multi-round conversations, so we flatten
    # each round's context and delegate to generate_until.
    # ------------------------------------------------------------------

    def generate_until_multi_round(self, requests) -> List[str]:
        """
        Multi-round generation stub.
        lmms-eval calls this for interleaved / multi-turn benchmarks.
        EmberNet treats the full concatenated context as a single prompt.
        """
        return self.generate_until(requests)
