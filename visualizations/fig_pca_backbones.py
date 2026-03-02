"""
PCA Backbone Comparison — Vision vs Text feature spaces.

Generates a publication-grade two-panel scatter plot showing how the
vision backbone (SigLIP) and text backbone (BitNet MoE decoder) features
distribute in a shared 2-D PCA space.

Called periodically during training (every N optimizer steps) and can also
be invoked offline from a checkpoint via generate_all_plots.py:

    python generate_all_plots.py --fig fig_pca_backbones --model <ckpt>

Saved to:  plots/pca_backbones/pca_backbones_step_XXXXX.{png,pdf}
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualizations.config import VIZ_CONFIG, apply_mpl_style, PLOTS_ROOT

# ── Palette ──────────────────────────────────────────────────────────────────
COLOR_VISION = "#0d7377"   # deep teal
COLOR_TEXT   = "#c45b28"   # rich amber

PCA_DIR_NAME = "pca_backbones"


def _pca_dir() -> Path:
    d = PLOTS_ROOT / PCA_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Feature extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_backbone_features(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    max_samples: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vision_feats, text_feats) each of shape [N, D].

    * vision_feats: mean-pooled SigLIP hidden states *before* the compressor/
      projector — raw encoder output.
    * text_feats: mean-pooled decoder hidden state at the last layer over
      non-padding text token positions.
    """
    # Unwrap DataParallel
    raw = model.module if isinstance(model, nn.DataParallel) else model

    device = next(raw.parameters()).device
    dtype  = next(raw.parameters()).dtype

    pixel_values = batch["pixel_values"].to(device, dtype=dtype)
    input_ids    = batch["input_ids"].to(device)
    attn_mask    = batch.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    B = min(pixel_values.size(0), max_samples)
    pixel_values = pixel_values[:B]
    input_ids    = input_ids[:B]
    if attn_mask is not None:
        attn_mask = attn_mask[:B]

    # ── Vision: raw SigLIP features (before compressor/pooler/projector) ──
    vis_enc = raw.vision_encoder
    if pixel_values.dim() == 5:
        pixel_values = pixel_values[:, 0]  # take first image
    vis_hidden = vis_enc.get_image_features(pixel_values)  # [B, 196, 768]
    vision_feats = vis_hidden.float().mean(dim=1).cpu().numpy()  # [B, D]

    # ── Text: decoder hidden state at last layer ──
    safe_ids = input_ids.clamp(0, raw.config.vocab_size - 1)
    text_embeds = raw.decoder.embed_tokens(safe_ids)  # [B, seq, D]

    h = text_embeds
    for layer in raw.decoder.layers:
        h, _, _ = layer(h)

    h = h.float()  # [B, seq, D]
    if attn_mask is not None:
        mask = attn_mask.unsqueeze(-1).float()  # [B, seq, 1]
        text_feats = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        text_feats = h.mean(dim=1)

    text_feats = text_feats.cpu().numpy()  # [B, D]

    return vision_feats, text_feats


# ── PCA plotting ─────────────────────────────────────────────────────────────

def _run_pca(vision: np.ndarray, text: np.ndarray):
    """Standardize, PCA-2, and return (vis_2d, txt_2d, explained_var)."""
    combined = np.concatenate([vision, text], axis=0)  # [2N, D]
    mean = combined.mean(axis=0)
    std  = combined.std(axis=0) + 1e-8
    combined = (combined - mean) / std

    # SVD-based PCA (no sklearn dependency)
    U, S, Vt = np.linalg.svd(combined, full_matrices=False)
    pc = combined @ Vt[:2].T  # [2N, 2]

    total_var = (S ** 2).sum()
    explained = (S[:2] ** 2) / total_var * 100  # percent

    n_vis = vision.shape[0]
    return pc[:n_vis], pc[n_vis:], explained


def plot_pca(
    vision_feats: np.ndarray,
    text_feats: np.ndarray,
    global_step: int,
    save_dir: Optional[Path] = None,
    stage: int = 1,
) -> Path:
    """Create the two-panel PCA scatter and save it."""
    apply_mpl_style()

    vis_2d, txt_2d, explained = _run_pca(vision_feats, text_feats)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.patch.set_facecolor("#fafafa")

    _scatter_kw_vis  = dict(s=14, alpha=0.75, edgecolors="none", color=COLOR_VISION, label="Vision backbone")
    _scatter_kw_txt  = dict(s=14, alpha=0.75, edgecolors="none", color=COLOR_TEXT,   label="Text backbone")

    # ── Left panel: standard PCA scatter ──
    ax = axes[0]
    ax.set_facecolor("#f5f5f5")
    ax.scatter(vis_2d[:, 0], vis_2d[:, 1], **_scatter_kw_vis)
    ax.scatter(txt_2d[:, 0], txt_2d[:, 1], **_scatter_kw_txt)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% var)")
    ax.set_title("Vision vs Text Backbone (PCA space)", fontsize=12, fontweight="semibold")
    ax.legend(loc="best", framealpha=0.9, edgecolor="none")
    ax.grid(True, linewidth=0.4, alpha=0.35, color="#cccccc")

    # ── Right panel: density view with centroids ──
    ax2 = axes[1]
    ax2.set_facecolor("#f5f5f5")
    ax2.scatter(vis_2d[:, 0], vis_2d[:, 1], s=22, alpha=0.35, edgecolors="none", color=COLOR_VISION)
    ax2.scatter(txt_2d[:, 0], txt_2d[:, 1], s=22, alpha=0.35, edgecolors="none", color=COLOR_TEXT)

    # Centroids
    vc = vis_2d.mean(axis=0)
    tc = txt_2d.mean(axis=0)
    ax2.scatter(*vc, s=120, marker="X", color=COLOR_VISION, edgecolors="white", linewidths=1.2, zorder=5)
    ax2.scatter(*tc, s=120, marker="X", color=COLOR_TEXT,   edgecolors="white", linewidths=1.2, zorder=5)
    ax2.annotate("Vision centroid", vc, textcoords="offset points", xytext=(8, 8),
                 fontsize=8, color=COLOR_VISION, fontweight="bold")
    ax2.annotate("Text centroid",   tc, textcoords="offset points", xytext=(8, -12),
                 fontsize=8, color=COLOR_TEXT, fontweight="bold")

    # Centroid distance annotation
    dist = np.linalg.norm(vc - tc)
    ax2.set_xlabel(f"PC1 ({explained[0]:.1f}% var)")
    ax2.set_ylabel(f"PC2 ({explained[1]:.1f}% var)")
    ax2.set_title(f"Centroid Distance = {dist:.2f}", fontsize=12, fontweight="semibold")
    ax2.grid(True, linewidth=0.4, alpha=0.35, color="#cccccc")

    fig.suptitle(f"Backbone Feature Alignment — Stage {stage}  ·  Step {global_step:,}",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_dir = save_dir or _pca_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pca_backbones_step_{global_step:05d}"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return png_path


# ── Training hook ────────────────────────────────────────────────────────────

def pca_backbones_hook(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    global_step: int,
    stage: int = 1,
    save_dir: Optional[Path] = None,
    max_samples: int = 256,
):
    """Lightweight hook called from the training loop every N steps."""
    was_training = model.training
    model.eval()
    try:
        vision_feats, text_feats = extract_backbone_features(model, batch, max_samples=max_samples)
        path = plot_pca(vision_feats, text_feats, global_step, save_dir=save_dir, stage=stage)
        print(f"  [PCA] Saved backbone PCA → {path}")
    except Exception as exc:
        print(f"  [PCA] Error at step {global_step}: {exc}")
    finally:
        if was_training:
            model.train()


# ── Offline generation (for generate_all_plots.py) ──────────────────────────

def generate(
    save_dir: Optional[Path] = None,
    model=None,
    step: int = 0,
) -> Optional[Path]:
    """Entry point for generate_all_plots.py --fig fig_pca_backbones.

    If *model* is provided, runs a dummy forward pass to extract real features.
    Otherwise generates a synthetic demo plot.
    """
    apply_mpl_style()
    out_dir = Path(save_dir) if save_dir else _pca_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if model is not None:
        # Build a tiny dummy batch on the model's device
        device = next(model.parameters()).device
        dtype  = next(model.parameters()).dtype
        raw = model.module if isinstance(model, nn.DataParallel) else model
        B = 64
        dummy_batch = {
            "pixel_values":   torch.randn(B, 3, 224, 224, device=device, dtype=dtype),
            "input_ids":      torch.randint(0, raw.config.vocab_size, (B, 64), device=device),
            "attention_mask":  torch.ones(B, 64, device=device, dtype=torch.long),
        }
        vision_feats, text_feats = extract_backbone_features(model, dummy_batch, max_samples=B)
    else:
        # Synthetic demo
        rng = np.random.RandomState(42)
        vision_feats = rng.randn(128, 64).astype(np.float32) + np.array([2.0, 0] * 32)
        text_feats   = rng.randn(128, 64).astype(np.float32) + np.array([0, 2.0] * 32)

    return plot_pca(vision_feats, text_feats, global_step=step, save_dir=out_dir)
