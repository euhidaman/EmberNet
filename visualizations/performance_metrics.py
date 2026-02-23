"""
Performance Metrics Visualizations

Covers:
  6.1  Accuracy Curves          – per-dataset, domain-aggregated, per-expert on target
  6.2  Perplexity Progression   – over training, by token position
  6.3  Benchmark Comparisons    – accuracy bars, size vs perf pareto, inference speed
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from visualizations.config import (
    VIZ_CONFIG, PLOT_DIRS, STAGE_COLORS, EXPERT_NAMES, EXPERT_COLORS, EXPERT_LABELS,
    ALL_DATASETS, DATASET_DOMAINS, DOMAIN_COLORS,
    apply_mpl_style, plot_filename, log_plot_error,
)
from visualizations.training_dynamics import _save_and_log
from visualizations.wandb_utils import WandBLogger

apply_mpl_style()


class PerformanceMetricsPlotter:
    """Generates all §6 performance-metrics plots."""

    def __init__(self, logger: Optional[WandBLogger] = None):
        self.logger = logger or WandBLogger(disabled=True)
        self._generated: List[Path] = []

    # ==================================================================
    # 6.1 Accuracy Curves
    # ==================================================================

    def plot_per_dataset_accuracy(self, data=None, step=None) -> Path:
        """Plot 6.1.1 – Multi-Dataset Accuracy Progression."""
        key = "per_dataset_accuracy_curves"
        out = PLOT_DIRS["accuracy_curves"] / plot_filename("performance", "accuracy_curves", key)
        try:
            incomplete = data is None
            n_steps = 5000
            if data is None:
                np.random.seed(120)
                steps = np.arange(0, n_steps, 500)
                acc_curves = {}
                for i, ds in enumerate(ALL_DATASETS):
                    target = np.random.uniform(55, 90)
                    curve  = target * (1 - np.exp(-steps / 2000)) + \
                             np.random.normal(0, 1.5, len(steps))
                    acc_curves[ds] = np.clip(curve, 0, 100)
                data = {"steps": steps, "acc": acc_curves}

            steps  = np.asarray(data["steps"])
            acc    = data["acc"]
            cmap   = plt.cm.tab20

            fig, ax = plt.subplots(figsize=(12, 7))
            for i, ds in enumerate(ALL_DATASETS):
                if ds in acc:
                    domain = _dataset_domain(ds)
                    color  = DOMAIN_COLORS.get(domain, cmap(i % 20))
                    ax.plot(steps, acc[ds], color=color, lw=VIZ_CONFIG["lw_thin"],
                            alpha=0.75, label=ds)

            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 100)
            title = "Per-Dataset Accuracy Progression"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=6, ncol=3, bbox_to_anchor=(1.01, 1), loc="upper left")

            out = _save_and_log(fig, out, self.logger, "plots/performance_metrics/accuracy_curves/per_dataset", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_domain_accuracy(self, data=None, step=None) -> Path:
        """Plot 6.1.2 – Domain-Aggregated Accuracy."""
        key = "domain_aggregated_accuracy"
        out = PLOT_DIRS["accuracy_curves"] / plot_filename("performance", "accuracy_curves", key)
        try:
            incomplete = data is None
            n_steps = 5000
            domains = list(DATASET_DOMAINS.keys())
            if data is None:
                np.random.seed(121)
                steps = np.arange(0, n_steps, 500)
                domain_acc = {}
                for domain in domains:
                    target = np.random.uniform(60, 88)
                    curve  = target * (1 - np.exp(-steps / 2000)) + \
                             np.random.normal(0, 1.0, len(steps))
                    domain_acc[domain] = np.clip(curve, 0, 100)
                data = {"steps": steps, "domain_acc": domain_acc}

            steps      = np.asarray(data["steps"])
            domain_acc = data["domain_acc"]

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            for domain in domains:
                if domain in domain_acc:
                    ax.plot(steps, domain_acc[domain],
                            color=DOMAIN_COLORS.get(domain, "#1f77b4"),
                            lw=VIZ_CONFIG["lw_main"], label=domain.replace("_", " "))

            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Average Accuracy (%)")
            ax.set_ylim(0, 100)
            title = "Domain-Aggregated Accuracy"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            out = _save_and_log(fig, out, self.logger, "plots/performance_metrics/accuracy_curves/domain_accuracy", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_per_expert_target_accuracy(self, data=None, step=None) -> Path:
        """Plot 6.1.3 – Per-Expert Accuracy on Target vs Other Domains."""
        key = "per_expert_target_accuracy"
        out = PLOT_DIRS["accuracy_curves"] / plot_filename("performance", "accuracy_curves", key)
        try:
            incomplete = data is None
            if data is None:
                np.random.seed(122)
                target_acc = np.random.uniform(70, 92, len(EXPERT_NAMES))
                others_acc = np.random.uniform(40, 65, len(EXPERT_NAMES))
                data = {"target_acc": target_acc, "others_acc": others_acc}

            target_acc = np.asarray(data["target_acc"])
            others_acc = np.asarray(data["others_acc"])
            x = np.arange(len(EXPERT_NAMES))
            w = 0.35

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            ax.bar(x - w/2, target_acc, w, label="Target domain",
                   color=[EXPERT_COLORS[e] for e in EXPERT_NAMES], edgecolor="black", alpha=0.9)
            ax.bar(x + w/2, others_acc, w, label="Other domains",
                   color="lightgray", edgecolor="black", alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels([f"E{i}" for i in range(len(EXPERT_NAMES))], fontsize=9)
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 100)

            # Expert name annotations
            for i, name in enumerate(EXPERT_NAMES):
                ax.text(i, 2, name.replace("_", "\n"), ha="center", fontsize=5.5, rotation=0)

            title = "Per-Expert Accuracy: Target Domain vs Others"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            out = _save_and_log(fig, out, self.logger, "plots/performance_metrics/accuracy_curves/per_expert_target_acc", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    # ==================================================================
    # 6.2 Perplexity Progression
    # ==================================================================

    def plot_perplexity_over_training(self, data=None, step=None) -> Path:
        """Plot 6.2.1 – Validation Perplexity Over Training."""
        key = "validation_perplexity_over_training"
        out = PLOT_DIRS["perplexity_progression"] / plot_filename("performance", "perplexity", key)
        try:
            incomplete = data is None
            domains = list(DATASET_DOMAINS.keys())
            if data is None:
                np.random.seed(130)
                steps = np.arange(0, 5000, 500)
                overall_ppl = 80 * np.exp(-steps / 3000) + 10 + np.random.normal(0, 1, len(steps))
                domain_ppl  = {}
                for domain in domains:
                    t = np.random.uniform(55, 120)
                    domain_ppl[domain] = t * np.exp(-steps / 3000) + np.random.uniform(8, 15) + \
                                         np.random.normal(0, 0.5, len(steps))
                data = {"steps": steps, "overall": overall_ppl, "domain_ppl": domain_ppl}

            steps       = np.asarray(data["steps"])
            overall_ppl = np.asarray(data["overall"])
            domain_ppl  = data.get("domain_ppl", {})

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            ax.semilogy(steps, overall_ppl, color="black", lw=2.0, label="Overall")
            for domain in domains:
                if domain in domain_ppl:
                    ax.semilogy(steps, domain_ppl[domain],
                                color=DOMAIN_COLORS.get(domain, "#aaaaaa"),
                                lw=VIZ_CONFIG["lw_thin"], alpha=0.75, label=domain.replace("_"," "))

            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Perplexity (log scale)")
            title = "Validation Perplexity over Training"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=8)

            out = _save_and_log(fig, out, self.logger, "plots/performance_metrics/perplexity/val_perplexity", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_perplexity_by_position(self, data=None, step=None) -> Path:
        """Plot 6.2.2 – Perplexity by Token Position."""
        key = "perplexity_by_token_position"
        out = PLOT_DIRS["perplexity_progression"] / plot_filename("performance", "perplexity", key)
        try:
            incomplete = data is None
            seq_len = 2048
            if data is None:
                np.random.seed(131)
                positions   = np.arange(seq_len)
                img_ppl     = np.random.uniform(5, 15, seq_len)
                early_txt   = 20 * np.exp(-positions[:512] / 200) + 8 + np.random.normal(0, 0.5, 512)
                late_txt    = 10 * np.exp(-positions[512:] / 500) + 5 + np.random.normal(0, 0.3, seq_len - 512)
                all_ppl     = np.concatenate([img_ppl[:64], early_txt[:min(seq_len - 64, 512)],
                                              late_txt[:max(0, seq_len - 576)]])
                # Pad to seq_len
                all_ppl = np.pad(all_ppl, (0, max(0, seq_len - len(all_ppl))))[:seq_len]
                data = {"positions": positions, "ppl": all_ppl, "n_img": 64}

            positions = np.asarray(data["positions"])
            ppl       = np.asarray(data["ppl"])
            n_img     = data.get("n_img", 64)

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            ax.plot(positions[:n_img], ppl[:n_img],
                    color="#AED6F1", lw=VIZ_CONFIG["lw_main"], label="Image tokens")
            ax.plot(positions[n_img:n_img+512], ppl[n_img:n_img+512],
                    color=STAGE_COLORS[1], lw=VIZ_CONFIG["lw_main"], label="Early text")
            ax.plot(positions[n_img+512:], ppl[n_img+512:],
                    color=STAGE_COLORS[2], lw=VIZ_CONFIG["lw_main"], label="Late text")
            ax.axvline(n_img - 0.5,       color="black", ls=":", lw=1.2)
            ax.axvline(n_img + 511.5,     color="black", ls=":", lw=1.2)
            ax.set_xlabel("Token Position in Sequence")
            ax.set_ylabel("Average Perplexity")
            title = "Perplexity by Token Position"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            out = _save_and_log(fig, out, self.logger, "plots/performance_metrics/perplexity/ppl_by_position", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    # ==================================================================
    # 6.3 Benchmark Comparisons
    # ==================================================================

    def plot_benchmark_accuracy(self, data=None, step=None) -> Path:
        """Plot 6.3.1 – EmberNet vs Baselines Accuracy."""
        key = "benchmark_accuracy_comparison"
        out = PLOT_DIRS["benchmark_comparisons"] / plot_filename("performance", "benchmark", key)
        try:
            incomplete = data is None
            benchmarks = ["TextVQA", "DocVQA", "ChartQA", "VQAv2", "ScienceQA", "A-OKVQA", "MathVista"]
            models     = ["EmberNet", "SmolVLM", "MobileVLM", "Baseline"]
            if data is None:
                np.random.seed(140)
                scores = {
                    "EmberNet":   np.random.uniform(55, 80, len(benchmarks)),
                    "SmolVLM":    np.random.uniform(45, 72, len(benchmarks)),
                    "MobileVLM":  np.random.uniform(40, 68, len(benchmarks)),
                    "Baseline":   np.random.uniform(25, 50, len(benchmarks)),
                }
                data = {"benchmarks": benchmarks, "scores": scores}

            benchmarks = data.get("benchmarks", benchmarks)
            scores = data["scores"]
            x = np.arange(len(benchmarks))
            w = 0.2
            model_colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]

            fig, ax = plt.subplots(figsize=(12, 6))
            for j, (model, color) in enumerate(zip(models, model_colors)):
                if model in scores:
                    offset = (j - len(models) / 2 + 0.5) * w
                    ax.bar(x + offset, scores[model], w,
                           label=model, color=color, edgecolor="black", alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(benchmarks, fontsize=9)
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 100)
            title = "Benchmark Accuracy: EmberNet vs Baselines"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            out = _save_and_log(fig, out, self.logger, "plots/performance_metrics/benchmark/accuracy_comparison", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_size_vs_performance(self, data=None, step=None) -> Path:
        """Plot 6.3.2 – Model Size vs Accuracy Pareto."""
        key = "model_size_vs_performance"
        out = PLOT_DIRS["benchmark_comparisons"] / plot_filename("performance", "benchmark", key)
        try:
            incomplete = data is None
            if data is None:
                models_info = {
                    "EmberNet":   (135, 72.5, "#d62728", "*"),
                    "SmolVLM":    (450, 69.0, "#1f77b4", "o"),
                    "MobileVLM":  (800, 71.0, "#2ca02c", "o"),
                    "Baseline":   (1540, 58.0, "#9467bd", "s"),
                }
                data = {"models": models_info}

            models_info = data["models"]
            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])

            sizes, accs = [], []
            for name, (size, acc, color, marker) in models_info.items():
                ms = 18 if marker == "*" else 10
                ax.scatter(size, acc, s=ms**2, color=color, marker=marker,
                           edgecolors="black", lw=0.8, zorder=5, label=name)
                ax.annotate(name, (size, acc), xytext=(6, 4),
                            textcoords="offset points", fontsize=8)
                sizes.append(size)
                accs.append(acc)

            # Pareto frontier
            sorted_idx = np.argsort(sizes)
            pareto_s = np.array(sizes)[sorted_idx]
            pareto_a = np.maximum.accumulate(np.array(accs)[sorted_idx][::-1])[::-1]
            ax.plot(pareto_s, pareto_a, "k--", lw=1.2, alpha=0.6, label="Pareto frontier")

            ax.set_xlabel("Model Size (MB)")
            ax.set_ylabel("Average Accuracy across Benchmarks (%)")
            title = "Model Size vs Performance Trade-off"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            out = _save_and_log(fig, out, self.logger, "plots/performance_metrics/benchmark/size_vs_performance", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_inference_speed(self, data=None, step=None) -> Path:
        """Plot 6.3.3 – Inference Speed Comparison."""
        key = "inference_speed_comparison"
        out = PLOT_DIRS["benchmark_comparisons"] / plot_filename("performance", "benchmark", key)
        try:
            incomplete = data is None
            devices = ["CPU", "GPU (RTX 3090)", "Edge (Raspberry Pi)"]
            models  = ["EmberNet", "SmolVLM", "MobileVLM"]
            if data is None:
                np.random.seed(141)
                speeds = {
                    "EmberNet":  np.array([45.0, 280.0, 8.5]),
                    "SmolVLM":   np.array([25.0, 180.0, 4.5]),
                    "MobileVLM": np.array([18.0, 130.0, 2.8]),
                }
                data = {"devices": devices, "speeds": speeds}

            devices = data.get("devices", devices)
            speeds  = data["speeds"]
            x = np.arange(len(devices))
            w = 0.25
            model_colors = ["#d62728", "#1f77b4", "#2ca02c"]

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            for j, (model, color) in enumerate(zip(models, model_colors)):
                if model in speeds:
                    offset = (j - len(models) / 2 + 0.5) * w
                    bars = ax.bar(x + offset, speeds[model], w,
                                  label=model, color=color, edgecolor="black", alpha=0.85)
                    # Latency annotation
                    for bar, spd in zip(bars, speeds[model]):
                        if spd > 0:
                            lat = 1000 / spd  # ms/token
                            ax.text(bar.get_x() + bar.get_width() / 2,
                                    bar.get_height() + 1,
                                    f"{lat:.1f}ms", ha="center", fontsize=6, rotation=90)

            ax.set_xticks(x)
            ax.set_xticklabels(devices, fontsize=9)
            ax.set_ylabel("Tokens / Second")
            title = "Inference Speed Comparison"
            if incomplete:
                title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend()

            out = _save_and_log(fig, out, self.logger, "plots/performance_metrics/benchmark/inference_speed", step)
            self._generated.append(out)
            return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_inference_energy_distribution(self, data=None, step=None) -> Path:
        """Plot: Energy per Inference Query by Task Type (violin / boxplot)."""
        key = "inference_energy_distribution"
        out = PLOT_DIRS["efficiency_tradeoffs"] / plot_filename("performance", "efficiency", key)
        try:
            incomplete = data is None
            task_types = ["OCR", "Chart", "Document", "Scene", "Generic VQA"]
            if data is None:
                np.random.seed(150)
                energy_data = {t: np.abs(np.random.normal(
                    np.random.uniform(0.0001, 0.0008), 0.0001, 30
                )) for t in task_types}
                data = {"task_types": task_types, "energy_joules": energy_data}

            task_types  = data.get("task_types", task_types)
            energy      = data.get("energy_joules", {})
            mode_tag    = data.get("mode_tag", "")

            vals = [energy.get(t, np.array([0.0])) * 1000 for t in task_types]  # → mJ
            task_colors = plt.cm.Set2(np.linspace(0, 1, len(task_types)))

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            vp = ax.violinplot(vals, positions=np.arange(len(task_types)),
                               showmedians=True, showextrema=True)
            for patch, c in zip(vp["bodies"], task_colors):
                patch.set_facecolor(c); patch.set_alpha(0.7)
            ax.set_xticks(np.arange(len(task_types)))
            ax.set_xticklabels(task_types, rotation=20, ha="right")
            ax.set_ylabel("Energy per Query (mJ)")
            title = "Inference Energy Distribution by Task Type"
            if mode_tag: title += f"  {mode_tag}"
            if incomplete: title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            out = _save_and_log(fig, out, self.logger, "plots/performance_metrics/efficiency/inference_energy_distribution", step)
            self._generated.append(out); return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_accuracy_vs_energy_pareto(self, data=None, step=None) -> Path:
        """Plot: Accuracy vs Energy (Pareto frontier, emulating ML-systems efficiency papers)."""
        key = "accuracy_vs_energy_pareto"
        out = PLOT_DIRS["efficiency_tradeoffs"] / plot_filename("performance", "efficiency", key)
        try:
            incomplete = data is None
            models = ["EmberNet", "SmolVLM-500M", "MobileVLM-3B", "LLaVA-7B", "Qwen-VL-7B"]
            if data is None:
                np.random.seed(151)
                bench_data = {
                    "EmberNet":       {"acc": 72.5, "energy_kwh": 0.0008},
                    "SmolVLM-500M":   {"acc": 61.0, "energy_kwh": 0.0012},
                    "MobileVLM-3B":   {"acc": 68.0, "energy_kwh": 0.0025},
                    "LLaVA-7B":       {"acc": 78.0, "energy_kwh": 0.0060},
                    "Qwen-VL-7B":     {"acc": 80.0, "energy_kwh": 0.0065},
                }
                data = {"models": models, "data": bench_data}

            models    = data.get("models", models)
            bdata     = data["data"]
            mode_tag  = data.get("mode_tag", "")
            model_colors = plt.cm.tab10(np.linspace(0, 0.9, len(models)))

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            xs, ys = [], []
            for m, c in zip(models, model_colors):
                d = bdata.get(m, {})
                e = d.get("energy_kwh", 0.001) * 1000  # → mWh
                a = d.get("acc", 50.0)
                xs.append(e); ys.append(a)
                is_ember = m == "EmberNet"
                ax.scatter(e, a, color=c, s=120 if is_ember else 70,
                           zorder=5 if is_ember else 3,
                           marker="*" if is_ember else "o",
                           edgecolors="black" if is_ember else "none", lw=0.8,
                           label=m)
                ax.annotate(m, (e, a), textcoords="offset points",
                            xytext=(6, 4), fontsize=8)

            # Pareto frontier
            pts  = sorted(zip(xs, ys))
            best = []; front = -np.inf
            for x_, y_ in pts:
                if y_ > front: front = y_; best.append((x_, y_))
            if len(best) >= 2:
                px, py = zip(*best)
                ax.step(px, py, where="post", color="black", lw=1.2, ls="--",
                        alpha=0.5, label="Pareto frontier")

            ax.set_xlabel("Energy per Benchmark (mWh)")
            ax.set_ylabel("Average Accuracy (%)")
            title = "Accuracy vs Energy – Efficiency Frontier"
            if mode_tag: title += f"  {mode_tag}"
            if incomplete: title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=8, ncol=2)
            out = _save_and_log(fig, out, self.logger, "plots/performance_metrics/efficiency/accuracy_vs_energy_pareto", step)
            self._generated.append(out); return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def plot_model_efficiency_tradeoff(self, data=None, step=None) -> Path:
        """Plot: Model Size vs Accuracy vs Energy (bubble scatter, 3-dimensional)."""
        key = "model_efficiency_tradeoff"
        out = PLOT_DIRS["efficiency_tradeoffs"] / plot_filename("performance", "efficiency", key)
        try:
            incomplete = data is None
            if data is None:
                models_info = {
                    "EmberNet":       {"size_mb": 420,  "acc": 72.5, "energy_mwh": 0.8},
                    "SmolVLM-500M":   {"size_mb": 980,  "acc": 61.0, "energy_mwh": 1.2},
                    "MobileVLM-3B":   {"size_mb": 6000, "acc": 68.0, "energy_mwh": 2.5},
                    "LLaVA-7B":       {"size_mb": 14000,"acc": 78.0, "energy_mwh": 6.0},
                    "Qwen-VL-7B":     {"size_mb": 15000,"acc": 80.0, "energy_mwh": 6.5},
                }
                data = {"models": models_info}

            models_info = data.get("models", {})
            mode_tag    = data.get("mode_tag", "")
            cmap        = plt.cm.plasma

            sizes_mb = np.array([v["size_mb"] for v in models_info.values()])
            accs     = np.array([v["acc"]     for v in models_info.values()])
            energies = np.array([v["energy_mwh"] for v in models_info.values()])
            names    = list(models_info.keys())

            # Color = energy (low=good=green, high=bad=red): use plasma colormap
            norm_e   = (energies - energies.min()) / ((energies.max() - energies.min()) + 1e-9)
            colors_  = cmap(norm_e)
            # Bubble area ∝ energy (use fixed scaling)
            bubble_s = (energies / energies.max() * 500 + 50)

            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figsize_single"])
            scatter = ax.scatter(sizes_mb, accs, s=bubble_s, c=colors_, alpha=0.8,
                                 edgecolors="black", lw=0.7)
            for name_, x_, y_, e_ in zip(names, sizes_mb, accs, energies):
                is_e = name_ == "EmberNet"
                ax.annotate(
                    f"{name_}\n{e_:.1f}mWh",
                    (x_, y_),
                    textcoords="offset points",
                    xytext=(8, 4 if not is_e else -14),
                    fontsize=7,
                    fontweight="bold" if is_e else "normal",
                )
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                        norm=plt.Normalize(energies.min(), energies.max()))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="Energy per Benchmark (mWh)")
            ax.set_xscale("log")
            ax.set_xlabel("Model Size (MB, log scale)")
            ax.set_ylabel("Average Accuracy (%)")
            title = "Model Size × Accuracy × Energy Efficiency"
            if mode_tag: title += f"  {mode_tag}"
            if incomplete: title += "  [Incomplete – placeholder data]"
            ax.set_title(title, fontweight="bold")
            out = _save_and_log(fig, out, self.logger, "plots/performance_metrics/efficiency/model_efficiency_tradeoff", step)
            self._generated.append(out); return out
        except Exception as e:
            log_plot_error(key, e); plt.close("all"); return out

    def generate_all(self, data=None, step=None) -> List[Path]:
        d = data or {}
        methods = [
            (self.plot_per_dataset_accuracy,       d.get("per_ds_acc")),
            (self.plot_domain_accuracy,            d.get("domain_acc")),
            (self.plot_per_expert_target_accuracy, d.get("expert_acc")),
            (self.plot_perplexity_over_training,   d.get("ppl")),
            (self.plot_perplexity_by_position,     d.get("ppl_pos")),
            (self.plot_benchmark_accuracy,         d.get("bench_acc")),
            (self.plot_size_vs_performance,        d.get("size_perf")),
            (self.plot_inference_speed,            d.get("speed")),
            # --- new research-grade plots ---
            (self.plot_inference_energy_distribution, d.get("infer_energy")),
            (self.plot_accuracy_vs_energy_pareto,     d.get("acc_energy_pareto")),
            (self.plot_model_efficiency_tradeoff,     d.get("efficiency_tradeoff")),
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


def _dataset_domain(ds: str) -> str:
    from visualizations.config import DATASET_DOMAINS
    for domain, ds_list in DATASET_DOMAINS.items():
        if ds in ds_list:
            return domain
    return "alignment"
