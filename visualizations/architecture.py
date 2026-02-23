"""
Architecture Visualizations

Covers:
  3.1  Model Diagrams      – full flowchart, MoE layer detail, BitLinear quant flow
  3.2  Attention Maps      – cross-modal heatmap, layer-wise grid, avg distance
  3.3  Token Flow          – Sankey diagram, expert activation timeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import seaborn as sns

from visualizations.config import (
    VIZ_CONFIG, PLOT_DIRS, EXPERT_NAMES, EXPERT_COLORS,
    apply_mpl_style, plot_filename, log_plot_error,
)
from visualizations.training_dynamics import _save_and_log
from visualizations.wandb_utils import WandBLogger

apply_mpl_style()


class ArchitecturePlotter:
    """Generates all §3 architecture visualizations."""

    def __init__(self, logger: Optional[WandBLogger] = None):
        self.logger = logger or WandBLogger(disabled=True)
        self._generated: List[Path] = []

    # ==================================================================
    # 3.1 Model Diagrams
    # ==================================================================

    def plot_architecture_flowchart(self, data=None, step=None) -> Path:
        """Plot 3.1.1 – Full EmberNet Architecture Flowchart."""
        key = "full_architecture_flowchart"
        out = PLOT_DIRS["model_diagrams"] / plot_filename("architecture", "model_diagrams", key)
        try:
            fig, ax = plt.subplots(figsize=(10, 14))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 14)
            ax.axis("off")

            def box(ax, x, y, w, h, text, color="#AED6F1", fontsize=9, bold=False):
                rect = mpatches.FancyBboxPatch(
                    (x - w / 2, y - h / 2), w, h,
                    boxstyle="round,pad=0.15", fc=color, ec="black", lw=1.2,
                )
                ax.add_patch(rect)
                fw = "bold" if bold else "normal"
                ax.text(x, y, text, ha="center", va="center",
                        fontsize=fontsize, fontweight=fw, wrap=True)

            def arrow(ax, x1, y1, x2, y2, label=""):
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                             arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
                if label:
                    mx, my = (x1 + x2) / 2 + 0.15, (y1 + y2) / 2
                    ax.text(mx, my, label, fontsize=7, color="gray")

            # Blocks (y decreasing = top to bottom)
            blk = [
                (5, 13.0, 7, 0.6, "Input: Image (224×224×3) + Text tokens", "#FAD7A0"),
                (3, 11.8, 4, 0.6, "SigLIP Vision Encoder\n(FROZEN, gray)", "#D5D8DC"),
                (7, 11.8, 4, 0.6, "Text Tokenizer →\nEmbeddings", "#D5D8DC"),
                (3, 10.4, 4, 0.6, "Visual Compressor\n196 → 49 → 64 tokens\n(Stage1, blue)", "#AED6F1"),
                (3, 9.0,  4, 0.6, "Projector (BitLinear MLP)\n64 visual tokens\n(Stage1, blue)", "#AED6F1"),
                (5, 7.6,  7, 0.6, "[BOS] [IMG₁…IMG₆₄] [Text tokens…]", "#A9DFBF"),
                (5, 6.2,  7, 0.6, "16× BitNet MoE Decoder Layers\n(Attn + MoE FFN)  (Stage2, red)", "#F1948A"),
                (5, 4.8,  7, 0.6, "RMSNorm → LM Head (BitLinear)\n→ Logits (vocab_size=32002)", "#A9DFBF"),
                (5, 3.5,  7, 0.6, "Output Tokens / Loss", "#FAD7A0"),
            ]
            for bx, by, bw, bh, bt, bc in blk:
                box(ax, bx, by, bw, bh, bt, color=bc, bold=True)

            # Arrows
            arrow(ax, 3, 12.7, 3, 12.1)   # image → SigLIP
            arrow(ax, 7, 12.7, 7, 12.1)   # text → tokenizer
            arrow(ax, 3, 11.5, 3, 10.7)   # SigLIP → compressor
            arrow(ax, 3, 10.1, 3, 9.3)    # compressor → projector
            arrow(ax, 3, 8.7, 5, 7.9, label="64 img toks")   # projector → merge
            arrow(ax, 7, 11.5, 5, 7.9, label="text toks")    # embeddings → merge
            arrow(ax, 5, 7.3, 5, 6.5)     # merge → decoder
            arrow(ax, 5, 5.9, 5, 5.1)     # decoder → lm head
            arrow(ax, 5, 4.5, 5, 3.8)     # lm head → output

            # Legend
            legend_items = [
                mpatches.Patch(fc="#D5D8DC", ec="black", label="Frozen (SigLIP)"),
                mpatches.Patch(fc="#AED6F1", ec="black", label="Stage 1 trainable (Projector)"),
                mpatches.Patch(fc="#F1948A", ec="black", label="Stage 2 trainable (Decoder)"),
            ]
            ax.legend(handles=legend_items, loc="lower left", fontsize=8)
            ax.set_title("EmberNet Architecture Overview", fontsize=14, fontweight="bold", pad=10)

            out = _save_and_log(fig, out, self.logger, "plots/architecture/model_diagrams/full_flowchart", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_moe_layer_detail(self, data=None, step=None) -> Path:
        """Plot 3.1.2 – MoE Layer Detailed View."""
        key = "moe_layer_detailed_view"
        out = PLOT_DIRS["model_diagrams"] / plot_filename("architecture", "model_diagrams", key)
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 9)
            ax.axis("off")

            def box(x, y, w, h, text, color="#AED6F1", fs=8):
                rect = mpatches.FancyBboxPatch(
                    (x - w/2, y - h/2), w, h,
                    boxstyle="round,pad=0.1", fc=color, ec="black", lw=1.0,
                )
                ax.add_patch(rect)
                ax.text(x, y, text, ha="center", va="center", fontsize=fs, wrap=True)

            def arr(x1, y1, x2, y2, lbl=""):
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                             arrowprops=dict(arrowstyle="->", lw=1.0))
                if lbl:
                    ax.text((x1+x2)/2+0.1, (y1+y2)/2+0.1, lbl, fontsize=6.5, color="gray")

            # Input
            box(1.5, 4.5, 2.0, 0.7, "Hidden States\n[B, seq, 768]", "#FAD7A0")
            # Router
            box(4.5, 4.5, 2.0, 0.7, "Router\nLinear(768→8)\n+ Softmax", "#AED6F1")
            # Top-2 select
            box(7.0, 4.5, 1.8, 0.7, "Top-2\nSelection", "#A9DFBF")
            # Shared expert
            box(9.5, 7.5, 2.0, 0.7, "Shared Expert\n(always active)", "#BCBD22", fs=7)
            # Experts
            expert_ys = [7.0, 6.0, 5.0, 4.0, 3.0, 2.5, 1.8, 1.2]
            n_exp = len(EXPERT_NAMES)
            for i, (e, ey) in enumerate(zip(EXPERT_NAMES, expert_ys)):
                c = EXPERT_COLORS[e] + "99"
                box(9.5, ey, 2.0, 0.55, f"E{i}: {e.split('_')[0]}\nBitFFN", c, fs=6.5)
            # Weighted sum
            box(12.5, 4.5, 2.0, 0.7, "Weighted Sum\n(expert outputs)", "#FAD7A0")

            # Arrows
            arr(2.5, 4.5, 3.5, 4.5, lbl="hidden")
            arr(5.5, 4.5, 6.1, 4.5)
            arr(7.9, 4.5, 8.5, 4.5, lbl="route")
            for ey in expert_ys:
                arr(8.5, 4.5, 8.5, ey)
                arr(8.5, ey, 9.5 - 1.0, ey)
                arr(9.5 + 1.0, ey, 11.5, ey)
                arr(11.5, ey, 11.5, 4.5)
            arr(9.5 - 1.0, 7.5, 8.5, 7.5)   # shared expert line
            arr(9.5 + 1.0, 7.5, 11.5, 7.5)
            arr(11.5, 4.5, 11.5, 4.5)
            arr(11.5, 4.5, 11.5, 4.5)
            arr(11.5, 4.5, 12.5 - 1.0, 4.5)

            ax.set_title("MoE FFN Layer – Detailed Data Flow", fontsize=14, fontweight="bold")
            out = _save_and_log(fig, out, self.logger, "plots/architecture/model_diagrams/moe_layer_detail", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_bitlinear_quantization_flow(self, data=None, step=None) -> Path:
        """Plot 3.1.3 – BitLinear Quantization Step-by-Step."""
        key = "bitlinear_quantization_flow"
        out = PLOT_DIRS["model_diagrams"] / plot_filename("architecture", "model_diagrams", key)
        try:
            np.random.seed(99)
            fp_weights = np.random.normal(0, 0.5, (5, 8))
            gamma = np.mean(np.abs(fp_weights))
            q_weights = np.round(fp_weights / (gamma + 1e-8)).clip(-1, 1)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Step 1: FP16 distribution
            axes[0].hist(fp_weights.flatten(), bins=20, color="#AED6F1", edgecolor="black", alpha=0.85)
            axes[0].axvline(0, color="black", lw=1.0, ls="--")
            axes[0].set_xlabel("Weight value")
            axes[0].set_ylabel("Count")
            axes[0].set_title(f"Step 1: FP16 Weights\nγ = {gamma:.4f}", fontweight="bold")

            # Step 2: Quantized (ternary) distribution
            unique, counts = np.unique(q_weights.flatten(), return_counts=True)
            bar_colors = {-1: "#d62728", 0: "#aaaaaa", 1: "#1f77b4"}
            bcolors = [bar_colors.get(int(v), "gray") for v in unique]
            axes[1].bar(unique, counts, color=bcolors, edgecolor="black", width=0.4)
            axes[1].set_xticks([-1, 0, 1])
            axes[1].set_xlabel("Quantized value {-1, 0, +1}")
            axes[1].set_ylabel("Count")
            sparsity = (q_weights == 0).mean() * 100
            axes[1].set_title(f"Step 2: Ternary Weights W_q\nSparsity={sparsity:.1f}%", fontweight="bold")

            # Step 3: Example mini weight matrix
            im = axes[2].imshow(q_weights, cmap="bwr", vmin=-1, vmax=1, aspect="auto")
            axes[2].set_title("Step 3: Weight Matrix W_q ∈ {−1, 0, +1}\n(red=−1, white=0, blue=+1)", fontweight="bold")
            axes[2].set_xlabel("Input dimension")
            axes[2].set_ylabel("Output dimension")
            fig.colorbar(im, ax=axes[2], ticks=[-1, 0, 1])

            fig.suptitle("BitNet b1.58 Quantization Flow", fontsize=VIZ_CONFIG["font_title"], fontweight="bold")
            out = _save_and_log(fig, out, self.logger, "plots/architecture/model_diagrams/bitlinear_quant_flow", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    # ==================================================================
    # 3.2 Attention Maps
    # ==================================================================

    def plot_cross_modal_attention(self, data=None, step=None) -> Path:
        """Plot 3.2.1 – Cross-Modal Attention Heatmap."""
        key = "cross_modal_attention_heatmap"
        out = PLOT_DIRS["attention_maps"] / plot_filename("architecture", "attention_maps", key)
        try:
            incomplete = data is None
            n_img_tokens  = 64
            n_text_tokens = 20
            n_query_tokens = 20
            if data is None:
                np.random.seed(50)
                attn = np.random.dirichlet(
                    np.ones(n_img_tokens + n_text_tokens), size=n_query_tokens
                )
            else:
                attn = np.asarray(data["attn"])
                n_img_tokens  = data.get("n_img_tokens",  64)
                n_text_tokens = data.get("n_text_tokens", 20)

            fig, ax = plt.subplots(figsize=(14, 6))
            im = ax.imshow(attn, cmap="hot", aspect="auto", vmin=0)
            ax.axvline(n_img_tokens - 0.5, color="cyan", lw=2.0, label="Img/Text boundary")
            ax.set_xlabel("Key Tokens (img tokens | text tokens)")
            ax.set_ylabel("Query Text Tokens")
            ax.set_xticks([n_img_tokens // 2 - 0.5, n_img_tokens + n_text_tokens // 2])
            ax.set_xticklabels(["Image tokens", "Text tokens"])
            title = "Cross-Modal Attention Weights"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            fig.colorbar(im, ax=ax, label="Attention weight")
            ax.legend()

            out = _save_and_log(fig, out, self.logger, "plots/architecture/attention_maps/cross_modal", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_layerwise_attention(self, data=None, step=None) -> Path:
        """Plot 3.2.2 – Self-Attention across 16 Layers (4×4 grid)."""
        key = "layerwise_attention_grid"
        out = PLOT_DIRS["attention_maps"] / plot_filename("architecture", "attention_maps", key)
        try:
            incomplete = data is None
            n_layers = 16
            seq_len  = 32
            if data is None:
                np.random.seed(51)
                attn_all = []
                for l in range(n_layers):
                    a = np.random.dirichlet(np.ones(seq_len), size=seq_len)
                    attn_all.append(a)
            else:
                attn_all = [np.asarray(a) for a in data["attn_all"]]

            fig, axes = plt.subplots(4, 4, figsize=VIZ_CONFIG["figsize_grid22"])
            for l, ax in enumerate(axes.flatten()):
                if l < len(attn_all):
                    im = ax.imshow(attn_all[l], cmap="Blues", aspect="auto", vmin=0, vmax=attn_all[l].max())
                    ax.set_title(f"Layer {l+1}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])

            title = "Self-Attention Maps Across 16 Decoder Layers"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            fig.suptitle(title, fontsize=VIZ_CONFIG["font_title"], fontweight="bold")
            out = _save_and_log(fig, out, self.logger, "plots/architecture/attention_maps/layerwise_grid", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_avg_attention_distance(self, data=None, step=None) -> Path:
        """Plot 3.2.3 – Average Attention Distance per Layer."""
        key = "avg_attention_distance_per_layer"
        out = PLOT_DIRS["attention_maps"] / plot_filename("architecture", "attention_maps", key)
        try:
            incomplete = data is None
            n_layers = 16
            if data is None:
                np.random.seed(52)
                layers = np.arange(1, n_layers + 1)
                distances = 5 + layers * 1.8 + np.random.normal(0, 1, n_layers)
                data = {"layers": layers, "distances": distances}

            layers    = np.asarray(data["layers"])
            distances = np.asarray(data["distances"])

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            ax.plot(layers, distances, color="#1f77b4", lw=VIZ_CONFIG["lw_main"], marker="o", ms=5)
            ax.fill_between(layers, distances, alpha=VIZ_CONFIG["alpha_fill"], color="#1f77b4")
            ax.set_xlabel("Layer Number")
            ax.set_ylabel("Average Attention Distance (tokens)")
            title = "Average Attention Distance by Layer"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.set_xticks(layers)

            out = _save_and_log(fig, out, self.logger, "plots/architecture/attention_maps/avg_attn_distance", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    # ==================================================================
    # 3.3 Token Flow
    # ==================================================================

    def plot_token_routing_sankey(self, data=None, step=None) -> Path:
        """Plot 3.3.1 – Token Routing Sankey."""
        key = "token_routing_sankey"
        out = PLOT_DIRS["token_flow"] / plot_filename("architecture", "token_flow", key)
        try:
            try:
                import plotly.graph_objects as go
                import plotly.io as pio
                np.random.seed(60)
                n_input = 2  # img / text groups
                in_labels = ["Image tokens (64)", "Text tokens (64)"]
                expert_labels = [f"E{i}:{EXPERT_NAMES[i].split('_')[0]}" for i in range(len(EXPERT_NAMES))]
                out_labels = ["Output tokens"]
                all_labels = in_labels + expert_labels + out_labels

                flows = np.random.dirichlet(np.ones(len(EXPERT_NAMES)), size=n_input)

                sources, targets, values, colors = [], [], [], []
                for si in range(n_input):
                    for ei in range(len(EXPERT_NAMES)):
                        sources.append(si)
                        targets.append(n_input + ei)
                        values.append(float(flows[si, ei]) * 64)
                        colors.append(list(EXPERT_COLORS.values())[ei] + "80")

                for ei in range(len(EXPERT_NAMES)):
                    sources.append(n_input + ei)
                    targets.append(n_input + len(EXPERT_NAMES))
                    values.append(sum(flows[:, ei]) * 32)
                    colors.append(list(EXPERT_COLORS.values())[ei] + "80")

                fig_s = go.Figure(go.Sankey(
                    node=dict(label=all_labels, pad=15, thickness=18,
                              color=["#cccccc"] * n_input + list(EXPERT_COLORS.values()) + ["#aaaaaa"]),
                    link=dict(source=sources, target=targets, value=values, color=colors),
                ))
                fig_s.update_layout(title_text="Token Routing Sankey", font_size=11, height=500)
                pio.write_image(fig_s, str(out), width=1400, height=600, scale=2)
                self.logger.log_image(out, "plots/architecture/token_flow/routing_sankey", step=step)
            except (ImportError, Exception) as e_plotly:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(f"Requires plotly+kaleido: {e_plotly}")
                print(f"  [WARNING] Sankey skipped: {e_plotly}")
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_expert_activation_timeline(self, data=None, step=None) -> Path:
        """Plot 3.3.2 – Expert Activation Heatmap across Token Positions."""
        key = "expert_activation_timeline"
        out = PLOT_DIRS["token_flow"] / plot_filename("architecture", "token_flow", key)
        try:
            incomplete = data is None
            n_experts = len(EXPERT_NAMES)
            n_tokens  = 128
            if data is None:
                np.random.seed(61)
                act_map = np.random.dirichlet(np.ones(n_experts), size=n_tokens).T
            else:
                act_map = np.asarray(data["act_map"])

            fig, ax = plt.subplots(figsize=(16, 5))
            im = ax.imshow(act_map, cmap="hot", aspect="auto", vmin=0, vmax=act_map.max())
            ax.set_yticks(range(n_experts))
            ax.set_yticklabels([f"E{i}" for i in range(n_experts)])
            ax.set_xlabel("Token Position")
            ax.set_ylabel("Expert")
            ax.axvline(63.5, color="cyan", lw=1.5, ls="--", label="Img/Text boundary")
            ax.legend(fontsize=8)
            title = "Expert Activation by Token Position"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            fig.colorbar(im, ax=ax, label="Routing Probability")

            out = _save_and_log(fig, out, self.logger, "plots/architecture/token_flow/expert_activation_timeline", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def generate_all(self, data=None, step=None) -> List[Path]:
        d = data or {}
        methods = [
            (self.plot_architecture_flowchart,       d.get("arch")),
            (self.plot_moe_layer_detail,             d.get("moe_detail")),
            (self.plot_bitlinear_quantization_flow,  d.get("bitlinear")),
            (self.plot_cross_modal_attention,        d.get("cross_modal_attn")),
            (self.plot_layerwise_attention,          d.get("layerwise_attn")),
            (self.plot_avg_attention_distance,       d.get("attn_distance")),
            (self.plot_token_routing_sankey,         d.get("token_sankey")),
            (self.plot_expert_activation_timeline,   d.get("expert_timeline")),
        ]
        paths = []
        for fn, dat in methods:
            try:
                p = fn(dat, step=step)
                paths.append(p)
                print(f"  ✓ {p.name}")
            except Exception as e:
                print(f"  ✗ {fn.__name__}: {e}")
        return paths
