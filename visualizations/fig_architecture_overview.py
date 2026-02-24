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
from visualizations.config import VIZ_CONFIG, apply_mpl_style

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
    "Ternary\nProjector (2-bit)":       dict(params=1_180_000, bits=1.58, color=_ternary_color := _TERNARY_COLOR),
    "BitNet MoE\nDecoder (2-bit)":      dict(params=300_000_000,bits=1.58, color=_TERNARY_COLOR),
    "Embeddings +\nLayer Norms (FP16)": dict(params=25_000_000, bits=16,  color=_FP16_COLOR),
}


def _extract_params_from_model(model) -> dict:
    """Extract accurate param counts from a live EmberNet model."""
    import torch.nn as nn
    try:
        from models.bitnet_moe import BitLinear
    except ImportError:
        return _DEFAULT_PARAMS

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


def _draw_block(ax, x, y, w, h, label, color, fontsize=8.5, text_color="white", edgecolor="none", alpha=0.92):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.03",
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
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("EmberNet Pipeline", fontsize=11, fontweight="bold", pad=6)

    # Blocks: (x, y, width, height, label, color)
    blocks = [
        (0.50, 0.90, 0.30, 0.10, "Image\nInput", "#888888"),
        (0.50, 0.75, 0.35, 0.10, "SigLIP Encoder\n(FP16, frozen)", _FP16_COLOR),
        (0.50, 0.60, 0.35, 0.09, "Pixel Shuffle Compressor\n(FP16)", "#ef8c2c"),
        (0.50, 0.46, 0.35, 0.10, "Ternary Projector\n(BitLinear, 1.58-bit)", _TERNARY_COLOR),
        (0.50, 0.31, 0.38, 0.10, "BitNet-b1.58 MoE Decoder\n(16 layers, 8+1 experts)", _TERNARY_COLOR),
        (0.82, 0.31, 0.26, 0.09, "Text Prompt\n(FP16 embeddings)", "#aaaaaa"),
        (0.50, 0.14, 0.35, 0.10, "LM Head → Output Text", "#555555"),
    ]
    for bx, by, bw, bh, blabel, bcolor in blocks:
        _draw_block(ax, bx, by, bw, bh, blabel, bcolor, fontsize=7.5)

    # Arrows
    arrows = [
        (0.50, 0.85, 0.50, 0.80),  # Image → SigLIP
        (0.50, 0.70, 0.50, 0.645),  # SigLIP → Compressor
        (0.50, 0.555, 0.50, 0.51),  # Compressor → Projector
        (0.50, 0.41, 0.50, 0.36),  # Projector → MoE Decoder
        (0.82, 0.355, 0.67, 0.315),  # Text → MoE Decoder
        (0.50, 0.26, 0.50, 0.19),  # MoE Decoder → LM Head
    ]
    for x0, y0, x1, y1 in arrows:
        _draw_arrow(ax, x0, y0, x1, y1)

    # MoE sub-diagram annotation
    ax.text(0.50, 0.24, "Top-2 routing + Shared Expert", ha="center", va="top",
            fontsize=6.5, color="#333333", style="italic")

    # Legend patches
    leg = [
        mpatches.Patch(color=_TERNARY_COLOR, label="Ternary (1.58-bit)"),
        mpatches.Patch(color=_FP16_COLOR, label="FP16 (frozen / ancillary)"),
    ]
    ax.legend(handles=leg, loc="lower left", fontsize=7, framealpha=0.85)


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

    param_dict = _extract_params_from_model(model) if model is not None else _DEFAULT_PARAMS

    fig = plt.figure(figsize=(16, 8), dpi=VIZ_CONFIG["dpi"])
    # Layout: left 40% = pipeline, right 60% = 2 stacked bars
    gs = fig.add_gridspec(2, 2, width_ratios=[0.90, 1.10], hspace=0.55, wspace=0.30)

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
