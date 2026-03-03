#!/usr/bin/env python3
"""
generate_all_plots.py
=====================
Master script for EmberNet visualizations.

Actions performed:
  1. Loads W&B run history (optional)
  2. Generates ALL plots across 7 categories
  3. Saves to hierarchical `plots/` folder structure
  4. Uploads plot images back to W&B
  5. Generates a summary report `plots/REPORT_<timestamp>.md`

Usage:
------
    # Generate all plots (training + paper figures) with synthetic data
    python generate_all_plots.py --all

    # Generate only the 7 ECCV/AAAI paper figures
    python generate_all_plots.py --all --paper-only

    # Generate a single paper figure by name
    python generate_all_plots.py --fig fig_architecture_overview
    python generate_all_plots.py --fig fig_ternary_stats
    python generate_all_plots.py --fig fig_moe_routing
    python generate_all_plots.py --fig fig_latency_energy
    python generate_all_plots.py --fig fig_va_token_effects
    python generate_all_plots.py --fig fig_va_answer_level
    python generate_all_plots.py --fig fig_qualitative_grid

    # Load data from a specific W&B run
    python generate_all_plots.py --wandb-run EmberNet/<run_id>

    # Load data from local checkpoint directory
    python generate_all_plots.py --checkpoint-dir ./checkpoints/checkpoint_step_5000.pt

    # Generate only one training-viz section
    python generate_all_plots.py --section expert_analysis

    # Enable W&B upload of generated plots
    python generate_all_plots.py --all --upload-to-wandb

Available --section values:
    training_dynamics  expert_analysis  architecture  quantization
    dataset_analysis   performance      stage_comparison  all (default)

Available --fig values (paper figures):
    fig_architecture_overview  fig_ternary_stats   fig_moe_routing
    fig_latency_energy         fig_va_token_effects fig_va_answer_level
    fig_qualitative_grid
"""

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Add repo root to sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from visualizations.config import (
    PLOT_DIRS, VIZ_CONFIG, ensure_plot_dirs, apply_mpl_style,
)
from visualizations.training_dynamics   import TrainingDynamicsPlotter
from visualizations.expert_analysis     import ExpertAnalysisPlotter
from visualizations.architecture        import ArchitecturePlotter
from visualizations.quantization        import QuantizationPlotter
from visualizations.dataset_analysis    import DatasetAnalysisPlotter
from visualizations.performance_metrics import PerformanceMetricsPlotter
from visualizations.stage_comparison    import StageComparisonPlotter
from visualizations.wandb_utils         import WandBLogger

apply_mpl_style()

import numpy as np


# ===========================================================================
# Paper-figure registry
# ===========================================================================

_PAPER_FIGS = {
    "fig_architecture_overview": "visualizations.fig_architecture_overview",
    "fig_ternary_stats":         "visualizations.fig_ternary_stats",
    "fig_moe_routing":           "visualizations.fig_moe_routing",
    "fig_latency_energy":        "visualizations.fig_latency_energy",
    "fig_va_token_effects":      "visualizations.fig_va_token_effects",
    "fig_va_answer_level":       "visualizations.fig_va_answer_level",
    "fig_qualitative_grid":      "visualizations.fig_qualitative_grid",
    "fig_hallucination_activation": "visualizations.fig_hallucination_activation",
    "fig_pca_backbones":        "visualizations.fig_pca_backbones",
    "fig_robot_risk_benchmarks": "visualizations.fig_robot_risk_benchmarks",
}


def generate_paper_fig(fig_name: str, save_dir=None, model=None):
    """
    Import and run the `generate()` function from the named paper-figure module.

    Parameters
    ----------
    fig_name : str
        One of the keys in _PAPER_FIGS (e.g. "fig_ternary_stats").
    save_dir : Path, optional
        Override default output directory.
    model : EmberNetVLM, optional
        Live model for real-data extraction.

    Returns
    -------
    Path or None
    """
    from pathlib import Path as _Path
    import importlib

    if fig_name not in _PAPER_FIGS:
        print(f"  [WARNING] Unknown paper figure: '{fig_name}'.  "
              f"Available: {list(_PAPER_FIGS)}")
        return None

    module = importlib.import_module(_PAPER_FIGS[fig_name])
    out_dir = _Path(save_dir) if save_dir else _Path("plots/paper_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    return module.generate(save_dir=out_dir, model=model)


# ===========================================================================
# W&B data loader
# ===========================================================================

def load_wandb_data(run_path: str) -> Dict:
    """
    Load logged scalars from a W&B run into a dict of NumPy arrays.

    Parameters
    ----------
    run_path : str
        Full run path in the format ``<entity>/<project>/<run_id>`` or
        ``<project>/<run_id>``.

    Returns
    -------
    Dict
        Flat dict mapping W&B metric name → NumPy array of all logged values.
        Empty dict on failure.
    """
    try:
        import wandb
        api = wandb.Api()
        run = api.run(run_path)
        history = run.scan_history()

        import pandas as pd
        import numpy as np

        rows = list(history)
        if not rows:
            print(f"  [WARNING] W&B run {run_path} has no history rows.")
            return {}

        df = pd.DataFrame(rows)
        data = {}
        for col in df.columns:
            if col.startswith("_"):
                continue
            vals = df[col].dropna().values
            if len(vals):
                try:
                    data[col] = vals.astype(float)
                except (ValueError, TypeError):
                    data[col] = vals   # keep as-is for string cols

        print(f"  Loaded {len(data)} metrics from W&B run: {run_path}")
        return data

    except Exception as e:
        print(f"  [WARNING] Could not load W&B data from '{run_path}': {e}")
        return {}


def load_checkpoint_data(checkpoint_path: str) -> Dict:
    """
    Extract scalar training info from a saved checkpoint file.
    """
    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        data = {}
        if "global_step" in checkpoint:
            data["train/step"] = np.array([checkpoint["global_step"]])
        print(f"  Loaded checkpoint info from: {checkpoint_path}")
        return data
    except Exception as e:
        print(f"  [WARNING] Could not load checkpoint '{checkpoint_path}': {e}")
        return {}


# ===========================================================================
# Local training metrics loader (replaces W&B when unavailable)
# ===========================================================================

def _load_local_training_metrics(output_dir: str) -> Dict:
    """
    Load training_metrics.json from both stage directories and merge them
    into a single dict with the same key format as W&B raw_data.

    Searches for:
      {output_dir}/stage1/training_metrics.json
      {output_dir}/stage2/training_metrics.json
    and concatenates arrays from both stages.
    """
    import json

    ckpt_base = Path(output_dir)
    merged: Dict = {}

    for stage_dir in ("stage1", "stage2"):
        metrics_file = ckpt_base / stage_dir / "training_metrics.json"
        if not metrics_file.exists():
            continue

        try:
            data = json.loads(metrics_file.read_text())
            if not data:
                continue
            n_steps = len(next(iter(data.values())))
            print(f"  [local] Loaded {n_steps} metric steps from {metrics_file}")

            for key, vals in data.items():
                arr = np.array(vals, dtype=float)
                if key in merged:
                    merged[key] = np.concatenate([merged[key], arr])
                else:
                    merged[key] = arr
        except Exception as e:
            print(f"  [local] Could not load {metrics_file}: {e}")

    if merged:
        print(f"  [local] Merged {len(merged)} metric keys from local training_metrics.json files")
    else:
        print(f"  [local] No training_metrics.json found in {ckpt_base}/stage*/")

    return merged


# ===========================================================================
# Plot registry
# ===========================================================================

def _build_context(
    raw_data: Dict,
    model=None,
    output_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict:
    """
    Build the nested context dict expected by each plotter section.

    Data sources:
      - raw_data      : W&B history (training_dynamics, expert_analysis)
      - model         : live EmberNetVLM (architecture, quantization)
      - output_dir    : path to training output (benchmark JSON for performance)
      - checkpoint_path: path to final checkpoint (stage comparison)
    """
    # ---- Load local training_metrics.json if W&B raw_data is empty ----
    if not raw_data and output_dir is not None:
        raw_data = _load_local_training_metrics(output_dir)

    # ---- Training dynamics (from W&B or local metrics) ----
    td = {}
    if "train/loss" in raw_data:
        n = len(raw_data["train/loss"])
        steps = raw_data.get("train/step", np.arange(n))
        loss = raw_data["train/loss"]
        val_loss = raw_data.get("val/loss", raw_data.get("train/window_avg_loss", loss))
        half = n // 2

        td["loss"] = {
            "stage1_steps":      steps[:half],
            "stage2_steps":      steps[half:],
            "stage1_train_loss": loss[:half],
            "stage2_train_loss": loss[half:],
            "stage1_val_loss":   val_loss[:half],
            "stage2_val_loss":   val_loss[half:],
        }

        # Loss spike detection
        td["loss_spike"] = {
            "steps": steps,
            "losses": loss,
        }

        # Convergence rate (uses val/window_avg as proxy for val loss)
        if half > 0:
            td["convergence"] = {
                "s1_steps": steps[:half],
                "s1_val":   val_loss[:half],
                "s2_steps": steps[half:],
                "s2_val":   val_loss[half:],
            }

    if "train/lr" in raw_data:
        td["lr_schedule"] = {
            "steps": raw_data.get("train/step"),
            "actual_lr": raw_data["train/lr"],
        }

    if "train/grad_norm" in raw_data:
        td["grad_norms"] = {
            "steps":       raw_data.get("train/step"),
            "global_norm": raw_data["train/grad_norm"],
            "clip_threshold": 1.0,
        }

    # Gradient clipping frequency (aggregate per-step clipped flag by epoch)
    if "train/grad_clipped" in raw_data and "train/epoch" in raw_data:
        epochs_arr = raw_data["train/epoch"]
        clipped_arr = raw_data["train/grad_clipped"]
        unique_epochs = sorted(set(int(e) for e in epochs_arr))
        clip_rates = []
        for ep in unique_epochs:
            mask = epochs_arr == ep
            if mask.sum() > 0:
                clip_rates.append(float(clipped_arr[mask].mean()))
            else:
                clip_rates.append(0.0)
        td["grad_clip_freq"] = {
            "epochs": np.array(unique_epochs),
            "fixed": np.array(clip_rates),
        }
        td["grad_clip_line"] = {
            "epochs": np.array(unique_epochs),
            "s1_clip_rate": np.array(clip_rates[:len(unique_epochs)//2 or len(unique_epochs)]),
            "s2_clip_rate": np.array(clip_rates[len(unique_epochs)//2:]) if len(unique_epochs) > 1 else np.array([]),
        }

    # Loss vs tokens
    if "train/cumulative_tokens" in raw_data and "train/loss" in raw_data:
        cum_tok = raw_data["train/cumulative_tokens"]
        loss_arr = raw_data["train/loss"]
        n_min = min(len(cum_tok), len(loss_arr))
        val_arr = raw_data.get("val/loss", raw_data.get("train/window_avg_loss", loss_arr))
        td["loss_vs_tokens"] = {
            "tokens_m": cum_tok[:n_min] / 1e6,
            "train_loss": loss_arr[:n_min],
            "val_loss": val_arr[:n_min],
        }

        # Tokens to convergence
        thresholds = [4.0, 3.0, 2.5, 2.0, 1.8, 1.5]
        tokens_needed = []
        for th in thresholds:
            idx = np.where(loss_arr[:n_min] <= th)[0]
            if len(idx) > 0:
                tokens_needed.append(float(cum_tok[idx[0]] / 1e6))
            else:
                tokens_needed.append(float(cum_tok[n_min - 1] / 1e6))
        td["tokens_convergence"] = {
            "thresholds": thresholds,
            "tokens_needed_m": np.array(tokens_needed),
        }

    # Training efficiency
    if "train/loss" in raw_data and "train/cumulative_tokens" in raw_data:
        n_steps = len(raw_data["train/loss"])
        # Estimate time uniformly across steps (no real time available)
        time_h = np.linspace(0, n_steps / 3600, n_steps)
        td["efficiency"] = {
            "time_h": time_h,
            "train_loss": raw_data["train/loss"],
            "val_loss": raw_data.get("val/loss", raw_data.get("train/window_avg_loss", raw_data["train/loss"])),
            "cum_tokens": raw_data["train/cumulative_tokens"][:n_steps],
        }

    if "energy/stage1_kwh" in raw_data or "energy/stage2_kwh" in raw_data:
        n = len(raw_data.get("train/loss", np.array([]))) or 1
        steps_all = raw_data.get("train/step", np.arange(n))
        half = len(steps_all) // 2

        td["energy"] = {
            "s1_steps":    steps_all[:half],
            "s1_energy":   raw_data.get("energy/stage1_kwh", np.zeros(half)),
            "s2_steps":    steps_all[half:],
            "s2_energy":   raw_data.get("energy/stage2_kwh", np.zeros(len(steps_all) - half)),
        }
        td["co2"] = {
            "s1_steps":  steps_all[:half],
            "s1_co2_kg": raw_data.get("energy/stage1_co2_kg", np.zeros(half)),
            "s2_steps":  steps_all[half:],
            "s2_co2_kg": raw_data.get("energy/stage2_co2_kg", np.zeros(len(steps_all) - half)),
        }
        if "train/cumulative_tokens" in raw_data:
            cum_tok = raw_data["train/cumulative_tokens"]
            s1_e = raw_data.get("energy/stage1_kwh", np.zeros_like(cum_tok[:half]))
            s2_e = raw_data.get("energy/stage2_kwh", np.zeros_like(cum_tok[half:]))
            cum_e = np.concatenate([s1_e, s2_e])
            tok_m = np.maximum(cum_tok / 1e6, 1e-9)
            td["energy_per_token"] = {
                "tokens_m":     tok_m,
                "kwh_per_m_tok": cum_e / tok_m,
                "steps":         steps_all,
            }
        td["stage_energy"] = {
            "stage_names":  ["Stage 1", "Stage 2"],
            "s1_steps":     steps_all[:half],
            "s2_steps":     steps_all[half:],
            "s1_energy":    raw_data.get("energy/stage1_kwh", np.zeros(half)),
            "s2_energy":    raw_data.get("energy/stage2_kwh", np.zeros(len(steps_all) - half)),
            "s1_co2_kg":    raw_data.get("energy/stage1_co2_kg", np.zeros(half)),
            "s2_co2_kg":    raw_data.get("energy/stage2_co2_kg", np.zeros(len(steps_all) - half)),
        }

    # ---- Expert analysis (from W&B or local metrics) ----
    ea = {}
    if "train/routing_entropy" in raw_data:
        ea["routing_entropy"] = {
            "steps":   raw_data.get("train/step"),
            "entropy": raw_data["train/routing_entropy"],
        }

    _expert_cols = [k for k in raw_data if k.startswith("train/expert_") and k[len("train/expert_"):].isdigit()]
    if _expert_cols:
        _expert_cols_sorted = sorted(_expert_cols, key=lambda k: int(k.rsplit("_", 1)[-1]))
        n_experts = len(_expert_cols_sorted)
        _n = min(len(raw_data[c]) for c in _expert_cols_sorted)
        steps_ea = raw_data.get("train/step", np.arange(_n))[:_n]
        expert_probs = np.column_stack([raw_data[c][:_n] for c in _expert_cols_sorted])

        ea["stacked_area"] = {
            "steps":        steps_ea,
            "expert_probs": expert_probs,
        }

        # Load balancing: imbalance = std/mean of expert probs per step
        ep_std = expert_probs.std(axis=1)
        ep_mean = np.maximum(expert_probs.mean(axis=1), 1e-9)
        ea["load_balancing"] = {
            "steps": steps_ea,
            "imbalance": ep_std / ep_mean,
        }

        # Expert usage violin: per-expert distribution of selection probabilities
        try:
            from visualizations.config import EXPERT_NAMES
        except ImportError:
            EXPERT_NAMES = [f"expert_{i}" for i in range(n_experts)]
        usage_dict = {}
        for ei, col in enumerate(_expert_cols_sorted):
            ename = EXPERT_NAMES[ei] if ei < len(EXPERT_NAMES) else f"expert_{ei}"
            usage_dict[ename] = raw_data[col][:_n].tolist()
        ea["usage_violin"] = {"usage": usage_dict}

        # Selection frequency heatmap: expert probs across checkpoint intervals
        n_ckpts = min(10, _n)
        chunk = max(1, _n // n_ckpts)
        freq_data = np.zeros((n_experts, n_ckpts))
        ckpt_labels = []
        for ci in range(n_ckpts):
            start = ci * chunk
            end = min(start + chunk, _n)
            freq_data[:, ci] = expert_probs[start:end].mean(axis=0)
            ckpt_labels.append(f"step {int(steps_ea[min(start, _n-1)])}")
        ea["selection_freq"] = {
            "freq": freq_data,
            "ckpt_labels": ckpt_labels,
        }

        # Dead expert detection: mark experts with < 1% avg usage as inactive
        active = (freq_data > 0.01).astype(float)
        ea["dead_expert"] = {
            "active": active,
            "ckpt_labels": ckpt_labels,
        }

        # Expert specialization index over time
        spec_steps = []
        spec_indices = {EXPERT_NAMES[ei] if ei < len(EXPERT_NAMES) else f"expert_{ei}": []
                        for ei in range(n_experts)}
        for ci in range(n_ckpts):
            start = ci * chunk
            end = min(start + chunk, _n)
            spec_steps.append(float(steps_ea[min(start, _n-1)]))
            probs_chunk = expert_probs[start:end]
            for ei in range(n_experts):
                ename = EXPERT_NAMES[ei] if ei < len(EXPERT_NAMES) else f"expert_{ei}"
                ep_vals = probs_chunk[:, ei]
                # Specialization = how concentrated routing is to this expert
                spec_indices[ename].append(float(ep_vals.std()))
        ea["specialization_index"] = {
            "steps": np.array(spec_steps),
            "indices": {k: np.array(v) for k, v in spec_indices.items()},
        }

        # Co-occurrence matrix (approximate from per-step top-2 experts)
        cooc = np.zeros((n_experts, n_experts))
        for t in range(min(_n, 500)):  # sample up to 500 steps
            top2 = np.argsort(expert_probs[t])[-2:]
            cooc[top2[0], top2[1]] += 1
            cooc[top2[1], top2[0]] += 1
        if cooc.sum() > 0:
            cooc /= cooc.sum()
        ea["cooccurrence"] = {"matrix": cooc}

    # ---- Architecture (from live model) ----
    arch = _extract_architecture(model) if model is not None else None

    # ---- Quantization (from live model) ----
    quant = _extract_quantization(model) if model is not None else None

    # ---- Dataset analysis (from training data on disk) ----
    ds_analysis = _extract_dataset_analysis(output_dir)

    # ---- Performance (from benchmark JSON) ----
    perf = _extract_performance(output_dir)

    # ---- Stage comparison (from W&B + checkpoints) ----
    sc = _extract_stage_comparison(raw_data, output_dir, checkpoint_path)

    return {
        "training_dynamics": td,
        "expert_analysis":   ea,
        "architecture":      arch,
        "quantization":      quant,
        "dataset_analysis":  ds_analysis,
        "performance":       perf,
        "stage_comparison":  sc,
    }


# ---------------------------------------------------------------------------
# Architecture extraction (from live model forward pass)
# ---------------------------------------------------------------------------

def _extract_architecture(model) -> Optional[Dict]:
    """Extract attention maps, routing patterns, expert activations from model."""
    try:
        import torch
        from models.bitnet_moe import BitLinear

        if not (hasattr(model, "decoder") and hasattr(model.decoder, "layers")):
            print("  [arch] Model has no decoder.layers — skipping architecture extraction")
            return None

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        num_layers = len(model.decoder.layers)
        hidden_size = model.decoder.config.hidden_size
        seq_len = 32
        n_img = 16

        dummy = torch.randn(1, seq_len, hidden_size, device=device, dtype=dtype)

        # Collect attention weights from all layers via hooks
        attn_maps = []
        router_logits_all = []

        def _attn_hook(layer_idx):
            def hook(mod, inp, out):
                # BitNetAttention returns (output, kv_cache)
                # To get attn_weights we need to recompute; store input instead
                pass
            return hook

        # Run forward to collect router logits
        model.eval()
        with torch.no_grad():
            h = dummy
            for li, layer in enumerate(model.decoder.layers):
                h, r_logits, _ = layer(h)
                if r_logits is not None:
                    probs = torch.softmax(r_logits.float(), dim=-1)
                    router_logits_all.append(probs.detach().cpu().numpy().squeeze())

        # Expert activation timeline: [n_experts, seq_len]
        if router_logits_all:
            n_experts = router_logits_all[0].shape[-1] if router_logits_all[0].ndim > 1 else 8
            act_map = np.zeros((n_experts, num_layers))
            for li, rp in enumerate(router_logits_all):
                if rp.ndim == 2:
                    act_map[:, li] = rp.mean(axis=0)[:n_experts]
                elif rp.ndim == 1:
                    act_map[:, li] = rp[:n_experts]
        else:
            act_map = None

        # Compute attention distances per layer
        layer_dists = []
        with torch.no_grad():
            h = dummy
            for li, layer in enumerate(model.decoder.layers):
                residual = h
                h_normed = layer.input_layernorm(h)
                q = layer.attention.q_proj(h_normed)
                k = layer.attention.k_proj(h_normed)
                bsz, sl, _ = q.shape
                hd = layer.attention.head_dim
                nh = layer.attention.num_heads
                nkv = layer.attention.num_kv_heads
                q = q.view(bsz, sl, nh, hd).transpose(1, 2)
                k = k.view(bsz, sl, nkv, hd).transpose(1, 2)
                cos, sin = layer.attention.rotary_emb(h_normed, sl)
                from models.bitnet_moe import apply_rotary_pos_emb
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                if layer.attention.num_kv_groups > 1:
                    k = k.unsqueeze(2).expand(-1, -1, layer.attention.num_kv_groups, -1, -1)
                    k = k.reshape(bsz, nh, sl, hd)
                scale = 1.0 / (hd ** 0.5)
                attn_w = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn_w = torch.softmax(attn_w, dim=-1)
                # mean over heads and batch -> [seq, seq]
                attn_mean = attn_w.mean(dim=(0, 1)).cpu().numpy()
                attn_maps.append(attn_mean)
                # Average attention distance
                pos = np.arange(sl)
                dist_matrix = np.abs(pos[:, None] - pos[None, :])
                avg_dist = (attn_mean * dist_matrix).sum() / attn_mean.sum()
                layer_dists.append(float(avg_dist))
                # Continue forward for next layer
                h, _, _ = layer(residual)

        result = {
            "arch": True,
            "moe_detail": True,
            "bitlinear": True,
            "token_sankey": True,
        }
        if attn_maps:
            result["cross_modal_attn"] = {
                "attn": attn_maps[0],
                "n_img_tokens": n_img,
                "n_text_tokens": seq_len - n_img,
            }
            result["layerwise_attn"] = {"attn_all": attn_maps}
        if layer_dists:
            result["attn_distance"] = {
                "layers": np.arange(num_layers),
                "distances": np.array(layer_dists),
            }
        if act_map is not None:
            result["expert_timeline"] = {"act_map": act_map}

        print(f"  [arch] Extracted {len(result)} architecture data keys from model")
        return result

    except Exception as e:
        print(f"  [arch] Architecture extraction failed: {e}")
        import traceback; traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Quantization extraction (from model weights)
# ---------------------------------------------------------------------------

def _extract_quantization(model) -> Optional[Dict]:
    """Extract weight histograms, sparsity, and size data from model."""
    try:
        import torch
        from models.bitnet_moe import BitLinear, weight_quant

        if not (hasattr(model, "decoder") and hasattr(model.decoder, "layers")):
            return None

        num_layers = len(model.decoder.layers)

        # Collect FP16 and quantized weights from all BitLinear layers
        all_fp16 = []
        all_quant = []
        layer_sparsity = []
        layer_bits = []
        act_scales = {}

        # Layer names: embed, L0..L15, lm_head
        layer_names = ["embed"] + [f"L{i}" for i in range(num_layers)] + ["lm_head"]
        sparsity_per_layer = np.zeros(len(layer_names))

        # Embedding sparsity
        embed_w = model.decoder.embed_tokens.weight.detach().float().cpu()
        sparsity_per_layer[0] = float((embed_w.abs() < 1e-6).sum()) / embed_w.numel()
        layer_bits.append(16.0)

        for li, layer in enumerate(model.decoder.layers):
            layer_fp16 = []
            layer_qnt = []
            for mod in [layer.attention.q_proj, layer.attention.k_proj,
                        layer.attention.v_proj, layer.attention.o_proj]:
                if isinstance(mod, BitLinear):
                    w = mod.weight.detach().float()
                    wq = weight_quant(w).cpu()
                    w = w.cpu()
                    layer_fp16.append(w.flatten())
                    layer_qnt.append(wq.flatten())
            for exp in layer.moe.experts:
                for proj in [exp.gate_proj, exp.up_proj, exp.down_proj]:
                    if isinstance(proj, BitLinear):
                        w = proj.weight.detach().float()
                        wq = weight_quant(w).cpu()
                        w = w.cpu()
                        layer_fp16.append(w.flatten())
                        layer_qnt.append(wq.flatten())
            if hasattr(layer.moe, 'shared_expert'):
                for proj in [layer.moe.shared_expert.gate_proj,
                             layer.moe.shared_expert.up_proj,
                             layer.moe.shared_expert.down_proj]:
                    if isinstance(proj, BitLinear):
                        w = proj.weight.detach().float()
                        wq = weight_quant(w).cpu()
                        w = w.cpu()
                        layer_fp16.append(w.flatten())
                        layer_qnt.append(wq.flatten())

            if layer_fp16:
                cat_fp = torch.cat(layer_fp16)
                cat_q = torch.cat(layer_qnt)
                all_fp16.append(cat_fp)
                all_quant.append(cat_q)
                sparsity_per_layer[1 + li] = float((cat_q.abs() < 0.5).sum()) / cat_q.numel()
                # Effective bitwidth: ternary = ~1.58
                layer_bits.append(1.58)
                act_scales[layer_names[1 + li]] = [float(cat_q.abs().mean())]

        # LM head sparsity (shared with embed)
        sparsity_per_layer[-1] = sparsity_per_layer[0]
        layer_bits.append(16.0)

        result = {}

        if all_fp16 and all_quant:
            fp16_flat = torch.cat(all_fp16).numpy()
            quant_flat = torch.cat(all_quant).numpy()
            # Subsample for histogram
            if len(fp16_flat) > 100000:
                idx = np.random.default_rng(42).choice(len(fp16_flat), 100000, replace=False)
                fp16_flat = fp16_flat[idx]
                quant_flat = quant_flat[idx]
            zero_frac = float((np.abs(quant_flat) < 0.5).sum()) / len(quant_flat)
            result["weight_hist"] = {
                "fp16": fp16_flat,
                "quant": quant_flat,
                "sparsity": zero_frac * 100,
            }

        result["sparsity"] = {
            "stage2": sparsity_per_layer * 100,
        }

        result["eff_bitwidth"] = {"bits": np.array(layer_bits)}

        result["act_scale"] = {"dist": act_scales}

        # Model size breakdown: Vision Encoder, Projector, Decoder, Total
        def _count_params(module):
            return sum(p.numel() for p in module.parameters())

        enc_p = _count_params(model.vision_encoder) if hasattr(model, "vision_encoder") else 0
        proj_p = _count_params(model.projector) if hasattr(model, "projector") else 0
        dec_p = _count_params(model.decoder) if hasattr(model, "decoder") else 0
        total_p = enc_p + proj_p + dec_p

        fp16_mb = np.array([enc_p * 2, proj_p * 2, dec_p * 2, total_p * 2]) / 1e6
        ternary_mb = np.array([enc_p * 2, proj_p * 1.58 / 8, dec_p * 1.58 / 8,
                               enc_p * 2 + proj_p * 1.58 / 8 + dec_p * 1.58 / 8]) / 1e6
        act_mb = np.array([0, 0, dec_p * 1 / 8, dec_p * 1 / 8]) / 1e6  # 8-bit activations

        result["model_size"] = {
            "fp16": fp16_mb,
            "ternary": ternary_mb,
            "activations": act_mb,
        }

        print(f"  [quant] Extracted {len(result)} quantization data keys from model")
        return result

    except Exception as e:
        print(f"  [quant] Quantization extraction failed: {e}")
        import traceback; traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Dataset analysis extraction (from training data on disk)
# ---------------------------------------------------------------------------

def _extract_dataset_analysis(output_dir: Optional[str]) -> Optional[Dict]:
    """Extract dataset statistics from disk data."""
    try:
        from visualizations.config import ALL_DATASETS, DATASET_DOMAINS

        if output_dir is None:
            return None

        ckpt_base = Path(output_dir)
        # Look for dataset_stats.json written during data loading
        stats_file = None
        for candidate in [
            ckpt_base / "dataset_stats.json",
            ckpt_base.parent / "dataset_stats.json",
            ckpt_base / "stage1" / "dataset_stats.json",
            ckpt_base / "stage2" / "dataset_stats.json",
        ]:
            if candidate.exists():
                stats_file = candidate
                break

        result = {}

        # Domain distribution from config (this is architectural, not data-dependent)
        domain_tokens = {}
        for domain, datasets in DATASET_DOMAINS.items():
            domain_tokens[domain] = float(len(datasets))
        result["domain_pie"] = {"domain_tokens_B": domain_tokens}

        result["samples"] = {}
        result["failures"] = {}

        if stats_file is not None:
            import json
            stats = json.loads(stats_file.read_text())
            if "token_counts" in stats:
                result["token_dist"] = stats["token_counts"]
            if "domain_samples" in stats:
                # Use real domain sample counts instead of architectural defaults
                real_domain_tokens = {}
                for domain_name, count in stats["domain_samples"].items():
                    real_domain_tokens[domain_name] = float(count)
                result["domain_pie"] = {"domain_tokens_B": real_domain_tokens}
            print(f"  [dataset] Loaded stats from {stats_file}")
        else:
            print(f"  [dataset] No dataset_stats.json found — domain_pie available, others will skip")

        return result if result else None

    except Exception as e:
        print(f"  [dataset] Dataset analysis extraction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Performance extraction (from benchmark JSON)
# ---------------------------------------------------------------------------

def _extract_performance(output_dir: Optional[str]) -> Optional[Dict]:
    """Load benchmark scores from auto_eval JSON output."""
    try:
        import json

        if output_dir is None:
            return None

        ckpt_base = Path(output_dir)
        bench_dir = ckpt_base / "plots" / "benchmark_results"

        # Find benchmark_scores_*.json files
        scores = {}
        if bench_dir.exists():
            for jf in sorted(bench_dir.glob("benchmark_scores_*.json")):
                try:
                    data = json.loads(jf.read_text())
                    if isinstance(data, dict):
                        # save_scores_json wraps as {"mode":..., "scores":{...}, "timestamp":...}
                        inner = data.get("scores", data)
                        if isinstance(inner, dict):
                            scores.update({k: float(v) for k, v in inner.items()
                                           if isinstance(v, (int, float))})
                except Exception:
                    pass

        # Also try lmms_raw dir
        raw_dir = bench_dir / "lmms_raw"
        if raw_dir.exists():
            try:
                from visualizations.benchmark_viz import extract_scores_from_lmms_results
                for jf in sorted(raw_dir.rglob("*.json")):
                    try:
                        data = json.loads(jf.read_text())
                        if isinstance(data, dict) and "results" in data:
                            extracted = extract_scores_from_lmms_results(data)
                            scores.update(extracted)
                    except Exception:
                        pass
            except ImportError:
                pass

        if not scores:
            print("  [perf] No benchmark scores found")
            return None

        print(f"  [perf] Loaded {len(scores)} benchmark scores")

        result = {}

        # bench_acc: radar/bar of benchmark scores
        benchmarks = sorted(scores.keys())
        bench_scores = np.array([scores[b] for b in benchmarks])
        result["bench_acc"] = {
            "benchmarks": benchmarks,
            "scores": {"EmberNet": bench_scores},
        }

        return result

    except Exception as e:
        print(f"  [perf] Performance extraction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Stage comparison extraction (from W&B data + both stage checkpoints)
# ---------------------------------------------------------------------------

def _extract_stage_comparison(
    raw_data: Dict,
    output_dir: Optional[str],
    checkpoint_path: Optional[str],
) -> Optional[Dict]:
    """Build cross-stage comparison data from training logs."""
    try:
        result = {}

        # Loss side-by-side (from W&B or local metrics)
        if "train/loss" in raw_data:
            n = len(raw_data["train/loss"])
            steps = raw_data.get("train/step", np.arange(n))
            loss = raw_data["train/loss"]
            val_loss = raw_data.get("val/loss", raw_data.get("train/window_avg_loss", loss))
            half = n // 2

            if half > 0:
                result["loss_sbs"] = {
                    "s1_steps": steps[:half],
                    "s1_train": loss[:half],
                    "s1_val":   val_loss[:half] if len(val_loss) >= half else loss[:half],
                    "s2_steps": steps[half:],
                    "s2_train": loss[half:],
                    "s2_val":   val_loss[half:] if len(val_loss) > half else loss[half:],
                }

        # Routing before/after from expert probability columns
        _expert_cols = [k for k in raw_data if k.startswith("train/expert_") and k[len("train/expert_"):].isdigit()]
        if _expert_cols:
            try:
                from visualizations.config import EXPERT_NAMES
            except ImportError:
                EXPERT_NAMES = [f"expert_{i}" for i in range(len(_expert_cols))]
            _expert_cols_sorted = sorted(_expert_cols, key=lambda k: int(k.rsplit("_", 1)[-1]))
            _n = min(len(raw_data[c]) for c in _expert_cols_sorted)
            half = _n // 2
            if half > 0:
                before = {}
                after = {}
                for ei, col in enumerate(_expert_cols_sorted):
                    ename = EXPERT_NAMES[ei] if ei < len(EXPERT_NAMES) else f"expert_{ei}"
                    vals = raw_data[col][:_n]
                    before[ename] = np.array([float(np.mean(vals[:half]))])
                    after[ename] = np.array([float(np.mean(vals[half:]))])
                result["routing_ba"] = {"before": before, "after": after}

        if not result:
            print("  [stage] No stage comparison data available (need W&B logs)")
            return None

        print(f"  [stage] Extracted {len(result)} stage-comparison data keys")
        return result

    except Exception as e:
        print(f"  [stage] Stage comparison extraction failed: {e}")
        return None


# ===========================================================================
# Main orchestrator
# ===========================================================================

SECTION_NAMES = [
    "training_dynamics",
    "expert_analysis",
    "architecture",
    "quantization",
    "dataset_analysis",
    "performance",
    "stage_comparison",
]


def generate_section(section: str, logger: WandBLogger, context: Dict) -> List[Path]:
    """Generate all plots for a single section. Returns list of saved paths."""
    step = None
    td_data = context.get("training_dynamics", {})

    if section == "training_dynamics":
        print("\n[§1] Training Dynamics")
        plotter = TrainingDynamicsPlotter(logger=logger)
        return plotter.generate_all(data=td_data, step=step)

    elif section == "expert_analysis":
        print("\n[§2] Expert Analysis")
        plotter = ExpertAnalysisPlotter(logger=logger)
        return plotter.generate_all(data=context.get("expert_analysis"), step=step)

    elif section == "architecture":
        print("\n[§3] Architecture Visualizations")
        plotter = ArchitecturePlotter(logger=logger)
        return plotter.generate_all(data=context.get("architecture"), step=step)

    elif section == "quantization":
        print("\n[§4] Quantization Analysis")
        plotter = QuantizationPlotter(logger=logger)
        return plotter.generate_all(data=context.get("quantization"), step=step)

    elif section == "dataset_analysis":
        print("\n[§5] Dataset Analysis")
        plotter = DatasetAnalysisPlotter(logger=logger)
        return plotter.generate_all(data=context.get("dataset_analysis"), step=step)

    elif section == "performance":
        print("\n[§6] Performance Metrics")
        plotter = PerformanceMetricsPlotter(logger=logger)
        return plotter.generate_all(data=context.get("performance"), step=step)

    elif section == "stage_comparison":
        print("\n[§7] Stage Comparison & Ablations")
        plotter = StageComparisonPlotter(logger=logger)
        return plotter.generate_all(data=context.get("stage_comparison"), step=step)

    else:
        print(f"  [WARNING] Unknown section: {section}")
        return []


def write_report(
    generated: List[Path],
    failed: List[str],
    warnings: List[str],
    wandb_project: str,
) -> Path:
    """Write a Markdown summary report listing all generated plots."""
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rpt_path = Path("plots") / f"REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    rpt_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# EmberNet Training Visualization Report",
        "",
        f"**Generated:** {ts}  ",
        f"**W&B Project:** {wandb_project}  ",
        f"**Stages:** Stage 1, Stage 2  ",
        "",
        "---",
        "",
    ]

    # Group by section
    sections = {
        "1. Training Dynamics":          "training_dynamics",
        "2. Expert Analysis":            "expert_analysis",
        "3. Architecture Visualizations":"architecture_visualizations",
        "4. Quantization Analysis":      "quantization_analysis",
        "5. Dataset Analysis":           "dataset_analysis",
        "6. Performance Metrics":        "performance_metrics",
        "7. Stage Comparison":           "stage_comparison",
        "8. Paper Figures (ECCV/AAAI)":  "paper_figures",
    }

    for section_title, folder_fragment in sections.items():
        section_plots = [
            p for p in generated
            if folder_fragment in p.as_posix()
        ]
        if not section_plots:
            continue
        lines.append(f"## {section_title}")
        lines.append("")
        for p in section_plots:
            rel = p.as_posix()
            lines.append(f"- [{p.name}]({rel})")
        lines.append("")

    # Summary footer
    lines += [
        "---",
        "",
        f"**Total Plots Generated:** {len(generated)}  ",
        f"**Failed Plots:** {len(failed)}  ",
        f"**Warnings:** {len(warnings)}  ",
        "",
    ]

    if failed:
        lines += ["### Failed Plots", ""]
        for f in failed:
            lines.append(f"- {f}")
        lines.append("")

    if warnings:
        lines += ["### Warnings", ""]
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    rpt_path.write_text("\n".join(lines), encoding="utf-8")
    return rpt_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate all EmberNet training visualizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--wandb-run", type=str, default=None,
        help="W&B run path (entity/project/run_id) to load metric history from.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Path to a checkpoint .pt file to load step info from.",
    )
    parser.add_argument(
        "--section", type=str, default="all",
        choices=SECTION_NAMES + ["all"],
        help="Generate only this section (default: all).",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="EmberNet",
        help="W&B project name (for report metadata).",
    )
    parser.add_argument(
        "--upload-to-wandb", action="store_true", default=False,
        help="Upload generated plots back to W&B as images.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots",
        help="Root output directory (default: plots/).",
    )
    parser.add_argument(
        "--all", action="store_true", default=False,
        help="Generate ALL plots: both training-viz sections AND all 7 paper figures.",
    )
    parser.add_argument(
        "--paper-only", action="store_true", default=False,
        help="Generate only the 7 ECCV/AAAI paper figures (skips training-viz sections).",
    )
    parser.add_argument(
        "--fig", type=str, default=None,
        choices=list(_PAPER_FIGS.keys()),
        metavar="FIG_NAME",
        help=("Generate a single paper figure by module name.  "
              "Choices: " + ", ".join(_PAPER_FIGS.keys())),
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to an EmberNet checkpoint to use for real-data extraction in paper figures.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  EmberNet – Generate All Visualizations")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Create all plot directories
    # ------------------------------------------------------------------
    ensure_plot_dirs()
    print("✓ Plot directories created")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    raw_data: Dict = {}
    warnings: List[str] = []

    if args.wandb_run:
        print(f"\nLoading W&B history: {args.wandb_run}")
        raw_data = load_wandb_data(args.wandb_run)
    elif args.checkpoint_dir:
        print(f"\nLoading checkpoint: {args.checkpoint_dir}")
        raw_data = load_checkpoint_data(args.checkpoint_dir)
    else:
        print("\nNo data source specified → using synthetic placeholder data for all plots.")
        warnings.append("No data source — all plots use synthetic placeholder data.")

    # ------------------------------------------------------------------
    # Load optional EmberNet model for real-data extraction
    # ------------------------------------------------------------------
    live_model = None
    if args.model:
        try:
            from inference.infer import EmberVLM
            print(f"\nLoading model: {args.model}")
            live_model = EmberVLM(model_path=args.model)
        except Exception as e:
            warnings.append(f"Model load failed ({e}) — model-dependent plots will skip.")
            print(f"  [WARNING] {warnings[-1]}")

    _output_base = args.output_dir
    _ckpt_for_ctx = args.checkpoint_dir or args.model
    context = _build_context(
        raw_data,
        model=live_model,
        output_dir=_output_base,
        checkpoint_path=_ckpt_for_ctx,
    )

    # ------------------------------------------------------------------
    # W&B logger
    # ------------------------------------------------------------------
    try:
        import wandb
        logger = WandBLogger(disabled=not args.upload_to_wandb)
        if args.upload_to_wandb and wandb.run is None:
            wandb.init(project=args.wandb_project, job_type="visualize",
                       name=f"viz_{datetime.now().strftime('%Y%m%d_%H%M')}")
    except ImportError:
        logger = WandBLogger(disabled=True)
        warnings.append("wandb not installed — W&B upload disabled.")

    # ------------------------------------------------------------------
    # Handle --fig (single paper figure shortcut)
    # ------------------------------------------------------------------
    if args.fig:
        paper_dir = Path(args.output_dir) / "paper_figures"
        print(f"\n[paper fig] Generating: {args.fig}")
        result = generate_paper_fig(args.fig, save_dir=paper_dir, model=live_model)
        if result and result.exists():
            print(f"  ✓ Saved → {result}")
            return 0
        else:
            print(f"  ✗ Failed")
            return 1

    # ------------------------------------------------------------------
    # Generate training-viz sections (skipped when --paper-only)
    # ------------------------------------------------------------------
    all_paths: List[Path] = []
    failed:    List[str]  = []

    if not args.paper_only:
        sections_to_run = SECTION_NAMES if args.section == "all" else [args.section]
        for section in sections_to_run:
            try:
                paths = generate_section(section, logger, context)
                all_paths.extend([p for p in paths if p and p.exists()])
            except Exception as e:
                msg = f"  [ERROR] Section '{section}' failed: {e}\n{traceback.format_exc()}"
                print(msg)
                failed.append(f"{section}: {e}")

    # ------------------------------------------------------------------
    # Generate paper figures (when --all or --paper-only)
    # ------------------------------------------------------------------
    if args.all or args.paper_only:
        paper_dir = Path(args.output_dir) / "paper_figures"
        print("\n[§8] ECCV/AAAI Paper Figures")
        for fig_name in _PAPER_FIGS:
            try:
                result = generate_paper_fig(fig_name, save_dir=paper_dir, model=live_model)
                if result and result.exists():
                    all_paths.append(result)
                    print(f"  ✓ {fig_name}")
                else:
                    failed.append(f"{fig_name}: returned None or file missing")
            except Exception as e:
                print(f"  [ERROR] {fig_name}: {e}\n{traceback.format_exc()}")
                failed.append(f"{fig_name}: {e}")

    # ------------------------------------------------------------------
    # Upload generated plots to W&B
    # ------------------------------------------------------------------
    if args.upload_to_wandb and not logger.disabled:
        print(f"\nUploading {len(all_paths)} plots to W&B …")
        logger.upload_plots(all_paths)

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    report_path = write_report(all_paths, failed, warnings, args.wandb_project)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  DONE — {len(all_paths)} plots generated, {len(failed)} failed")
    print(f"  Report: {report_path}")
    print("=" * 70)

    if failed:
        print("\nFailed sections:")
        for f in failed:
            print(f"  ✗ {f}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
