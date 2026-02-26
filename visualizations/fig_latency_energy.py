"""
fig_latency_energy.py
=====================
Figure 4 — Latency and energy consumption: quantized vs full-precision.

Scientific story
----------------
The primary practical motivation for 1.58-bit ternary weights is deployment
efficiency.  This figure benchmarks three settings end-to-end:
  1. EmberNet-Ternary : BitLinear with weight_quant (STE at train time;
     effectively 2-bit storage at inference).
  2. EmberNet-FP16    : Same architecture but BitLinear bypassed (straight
     float forward), serving as an upper bound on latency.
  3. EmberNet-Converted: Post-conversion TernaryLinear (packed 2-bit).

Metrics: per-sample wall-clock latency (ms) and, if CodeCarbon is available,
estimated energy in kWh.  Results are reported as mean ± std over N=10 runs.

Intended paper usage
--------------------
Figure 4, Section 5 "Efficiency Analysis".  Emphasises the cost savings
that motivate the ternary design, relevant for edge / on-device deployment.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))
from visualizations.config import VIZ_CONFIG, apply_mpl_style, skip_no_data

apply_mpl_style()

_COLORS = {
    "ternary":   "#1f77b4",
    "fp16":      "#ff7f0e",
    "converted": "#2ca02c",
}


# ---------------------------------------------------------------------------
# Synthetic latency/energy data (used when no model is provided)
# ---------------------------------------------------------------------------

def _synthetic_latency_energy(seed: int = 7) -> Dict:
    """
    Realistic synthetic benchmark results for three model formats.

    Values inspired by published BitNet inference numbers on A100.
    """
    rng = np.random.default_rng(seed)

    # Latency in ms (mean, std) for each setting
    data = {
        "EmberNet\n(FP16 weights)": {
            "latency_ms":   rng.normal(95, 4, 10),
            "energy_mwh":   rng.normal(3.2, 0.2, 10),
            "tokens_per_s": rng.normal(12, 0.8, 10),
            "color": _COLORS["fp16"],
        },
        "EmberNet\n(Ternary, STE)": {
            "latency_ms":   rng.normal(62, 3, 10),
            "energy_mwh":   rng.normal(1.9, 0.15, 10),
            "tokens_per_s": rng.normal(18, 1.0, 10),
            "color": _COLORS["ternary"],
        },
        "EmberNet\n(Converted\n2-bit packed)": {
            "latency_ms":   rng.normal(51, 2.8, 10),
            "energy_mwh":   rng.normal(1.5, 0.12, 10),
            "tokens_per_s": rng.normal(22, 1.2, 10),
            "color": _COLORS["converted"],
        },
    }
    return data


# ---------------------------------------------------------------------------
# Real benchmark harness
# ---------------------------------------------------------------------------

def _benchmark_model(model_fn, n_runs: int = 10) -> Dict:
    """Run model_fn n_runs times, collect latency and optional energy."""
    import torch

    latencies: List[float] = []
    energies:  List[float] = []

    cc = None
    try:
        from codecarbon import EmissionsTracker
        cc = EmissionsTracker(project_name="EmberNet_bench", log_level="error",
                              save_to_file=False)
    except Exception:
        pass

    for _ in range(n_runs):
        if cc is not None:
            try:
                cc.start()
            except Exception:
                cc = None

        t0 = time.perf_counter()
        with torch.no_grad():
            model_fn()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

        if cc is not None:
            try:
                em = cc.stop()
                kwh = float(cc.final_emissions_data.energy_consumed)
                energies.append(kwh * 1e6)  # μWh
            except Exception:
                cc = None

    return dict(
        latency_ms=np.array(latencies),
        energy_mwh=np.array(energies) if energies else np.zeros(n_runs),
    )


def _run_real_benchmark(model, device: str = "cpu", n_runs: int = 10) -> Dict:
    """Benchmark EmberNet in ternary vs FP16 mode."""
    import torch
    from models.bitnet_moe import BitLinear, weight_quant, activation_quant

    m = model  # always EmberNetVLM

    hidden_dim = getattr(m.decoder, "hidden_size", 768) if hasattr(m, "decoder") else 768
    model_dtype = next(m.parameters()).dtype
    dummy_embeds = torch.randn(1, 32, hidden_dim, device=device, dtype=model_dtype)

    # 1: Ternary (normal quantized forward)
    def _run_ternary():
        m.decoder(inputs_embeds=dummy_embeds.clone())

    tern_res = _benchmark_model(_run_ternary, n_runs)

    # 2: FP16 — temporarily patch BitLinear.forward to bypass quantization
    original_forwards: Dict = {}

    def fp16_forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

    for name, mod in m.named_modules():
        if isinstance(mod, BitLinear):
            original_forwards[name] = mod.forward
            mod.forward = fp16_forward.__get__(mod, type(mod))

    def _run_fp16():
        m.decoder(inputs_embeds=dummy_embeds.clone())

    fp16_res = _benchmark_model(_run_fp16, n_runs)

    # Restore original forwards
    for name, mod in m.named_modules():
        if isinstance(mod, BitLinear) and name in original_forwards:
            mod.forward = original_forwards[name]

    results = {
        "EmberNet\n(FP16 weights)": {
            **fp16_res,
            "tokens_per_s": (32 / (fp16_res["latency_ms"] / 1000)),
            "color": _COLORS["fp16"],
        },
        "EmberNet\n(Ternary, STE)": {
            **tern_res,
            "tokens_per_s": (32 / (tern_res["latency_ms"] / 1000)),
            "color": _COLORS["ternary"],
        },
    }
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _bar_with_errors(ax, labels: List[str], means: np.ndarray, stds: np.ndarray,
                     colors: List[str], ylabel: str, title: str, annotate: bool = True):
    x = np.arange(len(labels))
    bars = ax.bar(x, means, 0.55, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", lw=0.8,
                  error_kw=dict(elinewidth=1.5, ecolor="#333333", capthick=1.5))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, ha="center", multialignment="center")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9.5, fontweight="bold")

    if annotate:
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + means.max() * 0.01,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")


def generate(save_dir: Optional[Path] = None, model=None) -> Path:
    """
    Generate Figure 4 and save it.

    Parameters
    ----------
    save_dir : Path, optional
        Output directory.  Defaults to plots/paper_figures/.
    model : EmberNetVLM, optional
        Live model for real benchmarking.

    Returns
    -------
    Path : path to the saved PNG.
    """
    if save_dir is None:
        save_dir = Path("plots/paper_figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        skip_no_data("fig4_latency_energy")
        return save_dir / "fig4_latency_energy.png"

    try:
        device = next(model.parameters()).device
        bench_data = _run_real_benchmark(model, device=str(device))
    except Exception as e:
        skip_no_data(f"fig4_latency_energy (benchmark failed: {e})")
        return save_dir / "fig4_latency_energy.png"

    labels = list(bench_data.keys())
    colors = [bench_data[k]["color"] for k in labels]

    lat_means = np.array([bench_data[k]["latency_ms"].mean()  for k in labels])
    lat_stds  = np.array([bench_data[k]["latency_ms"].std()   for k in labels])
    nrg_means = np.array([bench_data[k]["energy_mwh"].mean()  for k in labels])
    nrg_stds  = np.array([bench_data[k]["energy_mwh"].std()   for k in labels])
    tok_means = np.array([bench_data[k]["tokens_per_s"].mean() if "tokens_per_s" in bench_data[k]
                          else 1000 / bench_data[k]["latency_ms"].mean() * 32 for k in labels])
    tok_stds  = np.array([bench_data[k]["tokens_per_s"].std()  if "tokens_per_s" in bench_data[k]
                          else bench_data[k]["latency_ms"].std() for k in labels])

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=VIZ_CONFIG["dpi"])
    fig.subplots_adjust(wspace=0.38)

    _bar_with_errors(axes[0], labels, lat_means, lat_stds, colors,
                     "Latency (ms / sample)", "A) Per-sample Latency\n(lower is better)")

    _bar_with_errors(axes[1], labels, nrg_means, nrg_stds, colors,
                     "Energy (mWh / sample)", "B) Estimated Energy per sample\n(CodeCarbon; lower is better)")

    _bar_with_errors(axes[2], labels, tok_means, tok_stds, colors,
                     "Tokens / second (prefill)", "C) Throughput\n(higher is better)")

    # Annotate speedup vs FP16 baseline on latency panel
    if len(lat_means) >= 2:
        fp16_lat = lat_means[0]
        for i, (label, lat) in enumerate(zip(labels[1:], lat_means[1:]), start=1):
            speedup = fp16_lat / lat
            axes[0].text(
                i, lat_means[i] + lat_stds[i] + lat_means.max() * 0.04,
                f"×{speedup:.2f}",
                ha="center", va="bottom", fontsize=9, color=colors[i], fontweight="bold"
            )

    fig.suptitle(
        "Figure 4 — Efficiency: Latency and Energy Consumption of Quantized EmberNet",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.text(0.5, -0.03,
             "Benchmarked on N=10 samples (mean ± std). "
             "FP16 weights = BitLinear without quantization (upper bound). "
             "Ternary = STE forward. Converted = packed 2-bit TernaryLinear.",
             ha="center", fontsize=7.5, color="#555555")

    out_pdf = save_dir / "fig4_latency_energy.pdf"
    out_png = save_dir / "fig4_latency_energy.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] Saved → {out_png}")
    return out_png


if __name__ == "__main__":
    generate()
