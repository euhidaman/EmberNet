"""
fig_architecture_overview.py
============================
Figure 1 — Architecture and quantization overview.

Scientific story
----------------
EmberNet is a tiny BitNet-b1.58 MoE VLM that compresses ~300M parameters
to ≈2× / 16× below FP16 / FP32 through ternary {-1, 0, +1} weights.
This figure situates the reader: left panel shows the full pipeline as a
simplified block diagram; right panel breaks down parameter counts and
effective bit-widths per component, making the "1.58-bit" claim legible.

Intended paper usage
--------------------
Figure 1 in the main text ("Model Overview", Section 3 / Architecture),
paired with the architecture table in the appendix.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from visualizations.config import VIZ_CONFIG, apply_mpl_style, skip_no_data

apply_mpl_style()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TERNARY_COLOR = "#1f77b4"   # blue  – ternary BitLinear
_FP16_COLOR    = "#ff7f0e"   # orange – FP16 ancillary
_ACCENT        = "#2ca02c"   # green  – shared expert / highlight

# Synthetic parameter estimates (realistic for EmberNet config)
# Updated from actual model counts if a model is provided
_DEFAULT_PARAMS = {
    "SigLIP Encoder\n(FP16, frozen)": dict(params=86_000_000, bits=16, color=_FP16_COLOR),
    "Pixel Shuffle\nCompressor (FP16)": dict(params=600_000,  bits=16, color=_FP16_COLOR),
    "Ternary\nProjector (2-bit)":       dict(params=1_180_000, bits=1.58, color=_TERNARY_COLOR),
    "BitNet MoE\nDecoder (2-bit)":      dict(params=300_000_000,bits=1.58, color=_TERNARY_COLOR),
    "Embeddings +\nLayer Norms (FP16)": dict(params=25_000_000, bits=16,  color=_FP16_COLOR),
}


def _extract_params_from_model(model) -> dict:
    """Extract accurate param counts from a live EmberNet model."""
    import torch.nn as nn
    try:
        from models.bitnet_moe import BitLinear
    except ImportError:
        return None

    counts = {k: dict(**v) for k, v in _DEFAULT_PARAMS.items()}

    def _count(module) -> int:
        return sum(p.numel() for p in module.parameters())

    if hasattr(model, "vision_encoder"):
        counts["SigLIP Encoder\n(FP16, frozen)"]["params"] = _count(model.vision_encoder)
    if hasattr(model, "compressor") or hasattr(model.vision_encoder, "compressor"):
        comp = getattr(model.vision_encoder, "compressor", None)
        if comp is not None:
            counts["Pixel Shuffle\nCompressor (FP16)"]["params"] = _count(comp)
    if hasattr(model, "projector"):
        counts["Ternary\nProjector (2-bit)"]["params"] = _count(model.projector)
    if hasattr(model, "decoder"):
        # Separate embeddings from the rest
        dec = model.decoder
        embed_p = sum(p.numel() for n, p in dec.named_parameters()
                      if "embed_tokens" in n or "lm_head" in n or "norm" in n)
        rest_p = sum(p.numel() for n, p in dec.named_parameters()
                     if "embed_tokens" not in n and "lm_head" not in n
                     and "norm" not in n)
        counts["BitNet MoE\nDecoder (2-bit)"]["params"] = rest_p
        counts["Embeddings +\nLayer Norms (FP16)"]["params"] = embed_p

    return counts


def _draw_block(ax, x, y, w, h, label, color, fontsize=9, text_color="white", edgecolor="none", alpha=0.92):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor=edgecolor, alpha=alpha, zorder=3
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color=text_color, zorder=4, multialignment="center")


def _draw_arrow(ax, x0, y0, x1, y1):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5),
        zorder=2
    )


def draw_pipeline_diagram(ax):
    """Left panel: pipeline block diagram."""
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")
    ax.set_title("EmberNet Pipeline", fontsize=12, fontweight="bold", pad=10)

    # Vertical layout — evenly spaced blocks along the centre column
    cx = 0.42          # centre x for the main pipeline column
    bw = 0.52          # standard block width
    bh = 0.095         # standard block height
    gap = 0.025        # gap between blocks

    # y-centres, top to bottom
    y_img   = 0.92
    y_sig   = y_img  - bh - gap
    y_comp  = y_sig  - bh - gap
    y_proj  = y_comp - bh - gap
    y_moe   = y_proj - bh - gap - 0.01   # slightly larger gap before decoder
    y_lm    = y_moe  - bh - gap - 0.04   # extra room for MoE annotation

    # Main pipeline blocks
    _draw_block(ax, cx, y_img,  0.30, bh * 0.85,  "Image\nInput",  "#888888", fontsize=9)
    _draw_block(ax, cx, y_sig,  bw,   bh,  "SigLIP Encoder\n(FP16, frozen)", _FP16_COLOR, fontsize=9)
    _draw_block(ax, cx, y_comp, bw,   bh,  "Pixel Shuffle Compressor\n(FP16)", "#ef8c2c", fontsize=9)
    _draw_block(ax, cx, y_proj, bw,   bh,  "Ternary Projector\n(BitLinear, 1.58-bit)", _TERNARY_COLOR, fontsize=9)
    _draw_block(ax, cx, y_moe,  bw + 0.04, bh + 0.01,
                "BitNet-b1.58 MoE Decoder\n(16 layers, 8+1 experts)", _TERNARY_COLOR, fontsize=9)
    _draw_block(ax, cx, y_lm,   bw,   bh,  "LM Head \u2192 Output Text", "#555555", fontsize=9)

    # Text prompt block — positioned to the right, feeding into MoE Decoder
    tx = 0.88
    _draw_block(ax, tx, y_moe + 0.065, 0.28, bh * 0.80,
                "Text Prompt\n(FP16 embed)", "#999999", fontsize=7.5, text_color="white")

    # Vertical arrows (main column)
    for y_top, y_bot in [
        (y_img  - bh * 0.85 / 2, y_sig  + bh / 2),
        (y_sig  - bh / 2,        y_comp + bh / 2),
        (y_comp - bh / 2,        y_proj + bh / 2),
        (y_proj - bh / 2,        y_moe  + (bh + 0.01) / 2),
        (y_moe  - (bh + 0.01) / 2, y_lm + bh / 2),
    ]:
        _draw_arrow(ax, cx, y_top, cx, y_bot)

    # Side arrow: text prompt → MoE decoder (horizontal-ish)
    _draw_arrow(ax, tx - 0.14, y_moe + 0.065 - bh * 0.80 / 2,
                cx + (bw + 0.04) / 2 - 0.02, y_moe + (bh + 0.01) / 2 - 0.01)

    # MoE annotation
    ax.text(cx, y_moe - (bh + 0.01) / 2 - 0.015,
            "Top-2 routing  |  Shared Expert",
            ha="center", va="top", fontsize=7.5, color="#444444", style="italic")

    # Legend
    leg = [
        mpatches.Patch(color=_TERNARY_COLOR, label="Ternary (1.58-bit)"),
        mpatches.Patch(color=_FP16_COLOR, label="FP16 (frozen / ancillary)"),
    ]
    ax.legend(handles=leg, loc="lower left", fontsize=7.5, framealpha=0.85)


def draw_param_breakdown(ax_top, ax_bot, param_dict: dict):
    """Right panels: parameter count bar + effective bit-size bar."""
    labels = list(param_dict.keys())
    params = np.array([v["params"] for v in param_dict.values()]) / 1e6  # millions
    bits   = np.array([v["bits"]   for v in param_dict.values()])
    colors = [v["color"] for v in param_dict.values()]
    eff_bits = params * bits  # effective Mbit

    short_labels = [
        lbl.replace("\n", "\n") for lbl in labels
    ]

    x = np.arange(len(labels))
    bar_w = 0.6

    # Top: raw params
    bars = ax_top.bar(x, params, width=bar_w, color=colors, edgecolor="white", linewidth=0.8)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(short_labels, fontsize=7, ha="center", multialignment="center")
    ax_top.set_ylabel("Parameters (millions)", fontsize=9)
    ax_top.set_title("Parameter Count per Component", fontsize=10, fontweight="bold")
    for bar, p in zip(bars, params):
        if p > 2:
            ax_top.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{p:.0f}M", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Bot: effective bits (params × bitwidth)
    ax_bot.bar(x, eff_bits, width=bar_w, color=colors, edgecolor="white", linewidth=0.8)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(short_labels, fontsize=7, ha="center", multialignment="center")
    ax_bot.set_ylabel("Effective size (Mbit = M·params × bits)", fontsize=9)
    ax_bot.set_title("Effective Storage Footprint (Mbit)", fontsize=10, fontweight="bold")

    # Annotate total compression
    total_fp32_mbit = params.sum() * 32
    total_eff_mbit  = eff_bits.sum()
    ratio = total_fp32_mbit / max(total_eff_mbit, 1)
    ax_bot.text(0.98, 0.97,
                f"vs FP32: {ratio:.1f}× smaller",
                transform=ax_bot.transAxes, ha="right", va="top",
                fontsize=8.5, fontweight="bold", color="#d62728",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.8))

    # Color legend
    leg = [
        mpatches.Patch(color=_TERNARY_COLOR, label="Ternary BitLinear (1.58-bit)"),
        mpatches.Patch(color=_FP16_COLOR,    label="FP16 (frozen / misc)"),
    ]
    ax_bot.legend(handles=leg, fontsize=7.5, loc="upper left", framealpha=0.9)


def generate(save_dir: Optional[Path] = None, model=None) -> Path:
    """
    Generate Figure 1 and save it.

    Parameters
    ----------
    save_dir : Path, optional
        Directory to save the figure.  Defaults to plots/paper_figures/.
    model : EmberNetVLM, optional
        Live model object for accurate parameter extraction.

    Returns
    -------
    Path : path to the saved figure.
    """
    if save_dir is None:
        save_dir = Path("plots/paper_figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        skip_no_data("fig1_architecture_overview")
        return save_dir / "fig1_architecture_overview.png"

    param_dict = _extract_params_from_model(model)
    if param_dict is None:
        skip_no_data("fig1_architecture_overview (extraction failed)")
        return save_dir / "fig1_architecture_overview.png"

    fig = plt.figure(figsize=(16, 9), dpi=VIZ_CONFIG["dpi"])
    # Layout: left 45% = pipeline, right 55% = 2 stacked bars
    gs = fig.add_gridspec(2, 2, width_ratios=[0.85, 1.15], hspace=0.50, wspace=0.35)

    ax_left  = fig.add_subplot(gs[:, 0])
    ax_right_top = fig.add_subplot(gs[0, 1])
    ax_right_bot = fig.add_subplot(gs[1, 1])

    draw_pipeline_diagram(ax_left)
    draw_param_breakdown(ax_right_top, ax_right_bot, param_dict)

    fig.suptitle(
        "EmberNet: Architecture Overview and Quantization Breakdown",
        fontsize=13, fontweight="bold", y=1.01
    )

    out_path = save_dir / "fig1_architecture_overview.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    out_png = save_dir / "fig1_architecture_overview.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] Saved → {out_path}")
    return out_png


if __name__ == "__main__":
    generate()
