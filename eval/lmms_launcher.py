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
import os
from pathlib import Path

# ── 0. Make sure repo root is on sys.path ──────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── 1. Import adapter → triggers @register_model("embernet") ──────────────
#       Must happen before lmms_eval.evaluator or models.__init__ are loaded.
try:
    import eval.embernet_lmms_adapter as _adapter  # noqa: F401
    print("[lmms_launcher] EmberNet model registered successfully.")
except Exception as e:
    print(f"[lmms_launcher] WARNING: Failed to register EmberNet model: {e}")
    import traceback
    traceback.print_exc()

# ── 2. Register into registry_v2 using register_manifest() ───────────────
# registry_v2 uses ModelManifest with a dotted class path string — it does
# NOT pick up @register_model decorators from the legacy registry.
try:
    from lmms_eval.models.registry_v2 import MODEL_REGISTRY_V2, ModelManifest  # type: ignore

    manifest = ModelManifest(
        model_id="embernet",
        simple_class_path="eval.embernet_lmms_adapter.EmberNetLMMS",
    )
    MODEL_REGISTRY_V2.register_manifest(manifest, overwrite=True)
    print("[lmms_launcher] Registered EmberNet into registry_v2 via register_manifest().")
except ImportError:
    pass  # older lmms-eval without registry_v2 — @register_model already handled it
except Exception as e:
    print(f"[lmms_launcher] registry_v2 patch failed (non-fatal): {e}")

# ── 3. Hand off to lmms_eval CLI ──────────────────────────────────────────
try:
    from lmms_eval.__main__ import cli_evaluate
    cli_evaluate()
except SystemExit as e:
    sys.exit(e.code)
