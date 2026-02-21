"""
eval/lmms_launcher.py
=====================
Thin launcher that:
1. Imports embernet_lmms_adapter FIRST — triggers @register_model("embernet")
   into both lmms_eval's legacy MODEL_REGISTRY and registry_v2 before the CLI
   ever looks up the model name.
2. Forwards all remaining argv to lmms_eval's cli_evaluate().

Why this exists
---------------
Newer lmms-eval versions (post-0.2.x) use a registry_v2 that does NOT
auto-import Python files placed in --include_path.  They only scan that
directory for task YAML configs.  So passing --include_path ./eval is not
enough to register our custom EmberNet model — the adapter module must be
explicitly imported before cli_evaluate() is called.

Usage (called by eval/auto_eval.py — not intended for direct use):
    python eval/lmms_launcher.py \
        --model embernet \
        --model_args pretrained=/path/to/final_model.pt \
        --tasks textvqa,ai2d \
        --batch_size 1 \
        --output_path ./results
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# ── 0. Make sure repo root is on sys.path ──────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── 1. Import EmberNet adapter so the class is available ──────────────────
try:
    from eval.embernet_lmms_adapter import EmberNetLMMS
    print("[lmms_launcher] EmberNetLMMS imported successfully.")
except Exception as e:
    print(f"[lmms_launcher] FATAL: Could not import EmberNetLMMS: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── 2. Monkey-patch lmms_eval.models.get_model ────────────────────────────
# This is the ONLY 100% reliable way to inject a custom model regardless of
# which registry version the installed lmms-eval uses.
# get_model() is called by evaluator.simple_evaluate(); we intercept the
# "embernet" lookup and return the class directly, bypassing all registries.
try:
    import lmms_eval.models as _lmms_models

    _original_get_model = _lmms_models.get_model

    def _patched_get_model(model_name, force_simple=False):
        if model_name == "embernet":
            print("[lmms_launcher] get_model intercepted → returning EmberNetLMMS")
            return EmberNetLMMS
        return _original_get_model(model_name, force_simple)

    _lmms_models.get_model = _patched_get_model
    print("[lmms_launcher] Patched lmms_eval.models.get_model successfully.")
except Exception as e:
    print(f"[lmms_launcher] WARNING: get_model patch failed: {e}")
    traceback.print_exc()

# ── 3. Hand off to lmms_eval CLI ──────────────────────────────────────────
try:
    from lmms_eval.__main__ import cli_evaluate
    cli_evaluate()
except SystemExit as e:
    sys.exit(e.code)
