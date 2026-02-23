"""
Quantization Analysis Visualizations

Covers:
  4.1  Weight Distributions  – before/after quant, layer-wise sparsity, weight magnitude decay
  4.2  Activation Histograms – 4-bit distribution, clipping frequency, layer-wise scale
  4.3  Bitwidth Efficiency   – model size breakdown, effective bitwidth, quant-error pareto
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from visualizations.config import (
    VIZ_CONFIG, PLOT_DIRS, STAGE_COLORS,
    apply_mpl_style, plot_filename, log_plot_error,
)
from visualizations.training_dynamics import _save_and_log
from visualizations.wandb_utils import WandBLogger

apply_mpl_style()


class QuantizationPlotter:
    """Generates all §4 quantization-analysis plots."""

    def __init__(self, logger: Optional[WandBLogger] = None):
        self.logger = logger or WandBLogger(disabled=True)
        self._generated: List[Path] = []

    # ==================================================================
    # 4.1 Weight Distributions
    # ==================================================================

    def plot_weight_before_after(self, data=None, step=None) -> Path:
        """Plot 4.1.1 – Ternary Weight Histogram Before/After Quantization."""
        key = "ternary_weight_histogram"
        out = PLOT_DIRS["weight_distributions"] / plot_filename("quantization", "weight_distributions", key)
        try:
            incomplete = data is None
            if data is None:
                np.random.seed(70)
                fp16_weights   = np.random.normal(0, 0.35, 50000)
                gamma          = np.mean(np.abs(fp16_weights))
                q_weights      = np.round(fp16_weights / (gamma + 1e-8)).clip(-1, 1)
                sparsity       = (q_weights == 0).mean() * 100
                data = {"fp16": fp16_weights, "quant": q_weights, "sparsity": sparsity}

            fp16     = np.asarray(data["fp16"])
            q        = np.asarray(data["quant"])
            sparsity = data.get("sparsity", (q == 0).mean() * 100)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VIZ_CONFIG["figsize_dual"])

            ax1.hist(fp16, bins=80, color="#AED6F1", edgecolor="black", log=True, alpha=0.85)
            ax1.set_xlabel("Weight Value")
            ax1.set_ylabel("Count (log scale)")
            ax1.set_title("FP16 Weights\n(before quantization)", fontweight="bold")

            uniq, cnts = np.unique(q.astype(int), return_counts=True)
            colors = {-1: "#d62728", 0: "#aaaaaa", 1: "#1f77b4"}
            ax2.bar([str(v) for v in uniq], cnts,
                    color=[colors.get(int(v), "gray") for v in uniq],
                    edgecolor="black", alpha=0.9)
            ax2.set_xlabel("Quantized Value {-1, 0, +1}")
            ax2.set_ylabel("Count (log scale)")
            ax2.set_yscale("log")
            ax2.set_title(f"Ternary Weights W_q\nSparsity = {sparsity:.1f}%", fontweight="bold")

            title = "Weight Distributions: FP16 vs Ternary"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            fig.suptitle(title, fontsize=VIZ_CONFIG["font_title"], fontweight="bold")

            out = _save_and_log(fig, out, self.logger, "plots/quantization/weight_distributions/before_after_hist", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_layerwise_sparsity(self, data=None, step=None) -> Path:
        """Plot 4.1.2 – Layer-wise Weight Sparsity."""
        key = "layerwise_weight_sparsity"
        out = PLOT_DIRS["weight_distributions"] / plot_filename("quantization", "weight_distributions", key)
        try:
            incomplete = data is None
            layers = (
                ["embed"] + [f"L{i}" for i in range(16)] + ["lm_head"]
            )
            if data is None:
                np.random.seed(71)
                sparsity_s1 = np.random.uniform(25, 45, len(layers))
                sparsity_s2 = np.random.uniform(28, 50, len(layers))
                sparsity_s2[0] = 0.0  # embed not quantized
                sparsity_s1[0] = 0.0
                data = {"layers": layers, "stage1": sparsity_s1, "stage2": sparsity_s2}

            x = np.arange(len(layers))
            w = 0.35
            s1 = np.asarray(data.get("stage1", np.zeros(len(layers))))
            s2 = np.asarray(data.get("stage2", np.zeros(len(layers))))

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.bar(x - w/2, s1, w, label="Stage 1", color=STAGE_COLORS[1], alpha=0.85)
            ax.bar(x + w/2, s2, w, label="Stage 2", color=STAGE_COLORS[2], alpha=0.85)
            ax.axhline(33.33, color="green", ls="--", lw=1.5, label="Target (33.3%)")
            ax.set_xticks(x)
            ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Sparsity (% zeros)")
            ax.set_ylim(0, 60)
            title = "Layer-wise Weight Sparsity"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            out = _save_and_log(fig, out, self.logger, "plots/quantization/weight_distributions/layerwise_sparsity", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_weight_magnitude_decay(self, data=None, step=None) -> Path:
        """Plot 4.1.3 – Weight Magnitude Decay Over Training."""
        key = "weight_magnitude_decay"
        out = PLOT_DIRS["weight_distributions"] / plot_filename("quantization", "weight_distributions", key)
        try:
            incomplete = data is None
            layer_types = ["attention", "FFN", "projector"]
            layer_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
            if data is None:
                np.random.seed(72)
                n = 5000
                steps = np.arange(n)
                mag = {}
                for lt, start in zip(layer_types, [0.45, 0.55, 0.6]):
                    curve = start * np.exp(-steps / (n * 0.8)) + 0.1 + \
                            np.random.normal(0, 0.005, n)
                    mag[lt] = np.clip(curve, 0.05, None)
                data = {"steps": steps, "magnitudes": mag}

            steps = np.asarray(data["steps"])
            mag   = data["magnitudes"]

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            for lt, color in zip(layer_types, layer_colors):
                if lt in mag:
                    ax.plot(steps, mag[lt], color=color, lw=VIZ_CONFIG["lw_main"], label=lt)

            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Average |weight| (before quantization)")
            title = "Weight Magnitude Decay Over Training"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            out = _save_and_log(fig, out, self.logger, "plots/quantization/weight_distributions/magnitude_decay", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    # ==================================================================
    # 4.2 Activation Histograms
    # ==================================================================

    def plot_4bit_activation_dist(self, data=None, step=None) -> Path:
        """Plot 4.2.1 – 4-bit Activation Distribution."""
        key = "4bit_activation_distribution"
        out = PLOT_DIRS["activation_histograms"] / plot_filename("quantization", "activations", key)
        try:
            incomplete = data is None
            if data is None:
                np.random.seed(80)
                fp16_act = np.random.normal(0, 2.5, 100000)
                # Quantize to integer range [-8, 7]
                scale = 8.0 / (np.max(np.abs(fp16_act)) + 1e-8)
                int_act = np.round(fp16_act * scale).clip(-8, 7)
                rmse = float(np.sqrt(np.mean((fp16_act - int_act / scale) ** 2)))
                data = {"fp16": fp16_act, "quant": int_act, "rmse": rmse}

            fp16_act = np.asarray(data["fp16"])
            int_act  = np.asarray(data["quant"])
            rmse     = data.get("rmse", 0.0)

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            ax.hist(fp16_act, bins=100, color="#AED6F1", alpha=0.6, density=True, label="FP16 activations")
            ax.hist(int_act,  bins=range(-9, 9), color="#F1948A", alpha=0.7, density=True,
                    label="4-bit quantized", align="mid")
            ax.set_xlabel("Activation Value")
            ax.set_ylabel("Density")
            title = f"Activation Distribution: FP16 vs 4-bit  (RMSE={rmse:.4f})"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.axvline(-8, color="red", ls=":", lw=1.2, label="4-bit clip boundary")
            ax.axvline(7,  color="red", ls=":", lw=1.2)
            ax.legend()

            out = _save_and_log(fig, out, self.logger, "plots/quantization/activations/4bit_dist", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_activation_clipping_frequency(self, data=None, step=None) -> Path:
        """Plot 4.2.2 – Activation Clipping Frequency over Training."""
        key = "activation_clipping_frequency"
        out = PLOT_DIRS["activation_histograms"] / plot_filename("quantization", "activations", key)
        try:
            incomplete = data is None
            if data is None:
                np.random.seed(81)
                n = 5000
                steps = np.arange(n)
                clip_ratio = 0.3 * np.exp(-steps / (n * 0.4)) + 0.02 + \
                             np.random.normal(0, 0.01, n)
                clip_ratio = np.clip(clip_ratio * 100, 0, 100)
                data = {"steps": steps, "clip_ratio": clip_ratio}

            steps      = np.asarray(data["steps"])
            clip_ratio = np.asarray(data["clip_ratio"])

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            ax.plot(steps, clip_ratio, color=STAGE_COLORS[2], lw=VIZ_CONFIG["lw_main"])
            ax.fill_between(steps, 0, clip_ratio, alpha=VIZ_CONFIG["alpha_fill"], color=STAGE_COLORS[2])
            ax.axhline(5.0, color="red", ls="--", lw=1.2, label="5% threshold")
            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Activation Clipping Frequency (%)")
            title = "Activation Clipping Frequency (outside [-8, 7])"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            out = _save_and_log(fig, out, self.logger, "plots/quantization/activations/clipping_frequency", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_activation_scale_boxplot(self, data=None, step=None) -> Path:
        """Plot 4.2.3 – Layer-wise Activation Scale."""
        key = "layerwise_activation_scale"
        out = PLOT_DIRS["activation_histograms"] / plot_filename("quantization", "activations", key)
        try:
            incomplete = data is None
            layers = ["embed"] + [f"L{i}" for i in range(16)] + ["lm_head"]
            if data is None:
                np.random.seed(82)
                dist = {l: np.random.exponential(1.0 + i * 0.1, 100) for i, l in enumerate(layers)}
                data = {"dist": dist}

            dist = data["dist"]
            box_vals = [dist.get(l, [0.0]) for l in layers]

            fig, ax = plt.subplots(figsize=(14, 5))
            bp = ax.boxplot(box_vals, patch_artist=True,
                            medianprops=dict(color="black", lw=2),
                            flierprops=dict(marker=".", ms=3, alpha=0.3))
            for patch in bp["boxes"]:
                patch.set_facecolor("#AED6F1")
                patch.set_alpha(0.75)
            ax.set_xticks(range(1, len(layers) + 1))
            ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Activation Magnitude (pre-quantization)")
            title = "Layer-wise Activation Scale"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")

            out = _save_and_log(fig, out, self.logger, "plots/quantization/activations/scale_boxplot", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    # ==================================================================
    # 4.3 Bitwidth Efficiency
    # ==================================================================

    def plot_model_size_breakdown(self, data=None, step=None) -> Path:
        """Plot 4.3.1 – Model Size Breakdown (stacked bar)."""
        key = "model_size_breakdown"
        out = PLOT_DIRS["bitwidth_efficiency"] / plot_filename("quantization", "bitwidth", key)
        try:
            incomplete = data is None
            components = ["Vision\nEncoder", "Projector", "Decoder", "Total"]
            if data is None:
                fp16_mb    = np.array([400.0,  20.0, 350.0, 770.0])
                ternary_mb = np.array([0.0,    12.0, 110.0, 122.0])
                act_mb     = np.array([50.0,    5.0,  80.0, 135.0])
                data = {"fp16": fp16_mb, "ternary": ternary_mb, "activations": act_mb}

            fp16_mb    = np.asarray(data["fp16"])
            ternary_mb = np.asarray(data["ternary"])
            act_mb     = np.asarray(data["activations"])
            x = np.arange(len(components))

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.bar(x, fp16_mb,    label="FP16 weights",    color="#aaaaaa", alpha=0.85, edgecolor="black")
            ax.bar(x, ternary_mb, bottom=fp16_mb,          label="Ternary weights", color="#AED6F1", alpha=0.9, edgecolor="black")
            ax.bar(x, act_mb,     bottom=fp16_mb + ternary_mb, label="Activations", color="#FAD7A0", alpha=0.9, edgecolor="black")

            # Compression ratio annotations
            for i, (fp, t) in enumerate(zip(fp16_mb, ternary_mb)):
                if t > 0 and fp > 0:
                    ratio = fp / t
                    ax.text(i, fp + t + act_mb[i] + 15, f"×{ratio:.1f}", ha="center", fontsize=8, color="green", fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(components)
            ax.set_ylabel("Memory (MB)")
            title = "Model Size Breakdown by Component"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            out = _save_and_log(fig, out, self.logger, "plots/quantization/bitwidth/model_size_breakdown", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_effective_bitwidth(self, data=None, step=None) -> Path:
        """Plot 4.3.2 – Effective Bitwidth per Layer."""
        key = "effective_bitwidth_per_layer"
        out = PLOT_DIRS["bitwidth_efficiency"] / plot_filename("quantization", "bitwidth", key)
        try:
            incomplete = data is None
            layers = ["embed"] + [f"L{i}" for i in range(16)] + ["lm_head"]
            if data is None:
                np.random.seed(90)
                # Bit entropy: ideal ternary = 1.585 bits; embed = 16 bits
                bits = np.random.uniform(1.2, 1.58, len(layers))
                bits[0]  = 16.0  # embedding
                bits[-1] = 16.0  # lm head
                data = {"layers": layers, "bits": bits}

            bits = np.asarray(data["bits"])

            fig, ax = plt.subplots(figsize=(14, 5))
            colors = ["#AED6F1" if (0 < i < len(layers) - 1) else "#F1948A"
                      for i in range(len(layers))]
            ax.bar(range(len(layers)), bits, color=colors, edgecolor="black", alpha=0.85)
            ax.axhline(1.585, color="green", ls="--", lw=1.5, label="Ideal ternary (1.585 bits)")
            ax.axhline(16.0,  color="gray",  ls=":",  lw=1.2, label="FP16 (16 bits)")
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Effective bits per parameter")
            title = "Effective Bitwidth per Layer  (H(p_{-1}, p_{0}, p_{+1}))"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            out = _save_and_log(fig, out, self.logger, "plots/quantization/bitwidth/effective_bitwidth", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_quantization_pareto(self, data=None, step=None) -> Path:
        """Plot 4.3.3 – Quantization Error vs Model Size (Pareto frontier)."""
        key = "quantization_pareto"
        out = PLOT_DIRS["bitwidth_efficiency"] / plot_filename("quantization", "bitwidth", key)
        try:
            incomplete = data is None
            if data is None:
                methods  = ["FP32", "FP16", "Ternary", "Binary"]
                sizes_mb = [1540.0, 770.0, 135.0, 90.0]
                ppl      = [8.5,    9.0,   11.5,  15.0]
                colors   = ["#d62728", "#AED6F1", "#2ca02c", "#9467bd"]
                data = {"methods": methods, "sizes_mb": sizes_mb, "ppl": ppl, "colors": colors}

            methods  = data["methods"]
            sizes_mb = np.asarray(data["sizes_mb"])
            ppl      = np.asarray(data["ppl"])
            colors   = data.get("colors", ["#1f77b4"] * len(methods))

            # Pareto frontier (sort by size)
            sort_idx = np.argsort(sizes_mb)
            pareto_s = sizes_mb[sort_idx]
            pareto_p = np.minimum.accumulate(ppl[sort_idx])

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            for m, s, p, c in zip(methods, sizes_mb, ppl, colors):
                marker = "*" if "Ternary" in m else "o"
                ms     = 16 if marker == "*" else 10
                ax.scatter(s, p, color=c, s=ms**2, marker=marker, zorder=5, edgecolors="black", lw=0.8)
                ax.annotate(m, (s, p), textcoords="offset points", xytext=(6, 4), fontsize=9)

            ax.plot(pareto_s, pareto_p, color="black", lw=1.2, ls="--", label="Pareto frontier", zorder=3)
            ax.set_xlabel("Model Size (MB)")
            ax.set_ylabel("Validation Perplexity (lower is better)")
            title = "Quantization Pareto: Size vs Perplexity"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            out = _save_and_log(fig, out, self.logger, "plots/quantization/bitwidth/quant_pareto", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def generate_all(self, data=None, step=None) -> List[Path]:
        d = data or {}
        methods = [
            (self.plot_weight_before_after,         d.get("weight_hist")),
            (self.plot_layerwise_sparsity,          d.get("sparsity")),
            (self.plot_weight_magnitude_decay,      d.get("weight_mag")),
            (self.plot_4bit_activation_dist,        d.get("activation_dist")),
            (self.plot_activation_clipping_frequency, d.get("act_clip_freq")),
            (self.plot_activation_scale_boxplot,    d.get("act_scale")),
            (self.plot_model_size_breakdown,        d.get("model_size")),
            (self.plot_effective_bitwidth,          d.get("eff_bitwidth")),
            (self.plot_quantization_pareto,         d.get("quant_pareto")),
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
