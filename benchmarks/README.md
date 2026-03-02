# Robot Risk-Aware Benchmarking Suite

Three benchmarks evaluate EmberNet's ability to understand environmental risk and make robot deployment decisions.

## Benchmarks

| Benchmark | What It Measures | Source |
|---|---|---|
| **Robot-Selection Dataset** | Multi-robot selection in hazardous scenarios — top-N robot ranking and unsafe-deployment avoidance | Local (`robot-selection-dataset/`) |
| **GEOBench-VLM** | Geospatial disaster/hazard assessment from satellite imagery — damage detection, flooding, fire, infrastructure risk | [aialliance/GEOBench-VLM](https://huggingface.co/datasets/aialliance/GEOBench-VLM) |
| **VERI-Emergency** | Emergency vs safe scene discrimination — avoiding overreaction on visually similar scenes | [Dasool/VERI-Emergency](https://huggingface.co/datasets/Dasool/VERI-Emergency) |

## Quick Start

### 1. Download benchmark datasets

**Trial download** (fast, ~20 samples each):
```bash
python benchmarks/download_benchmarks.py --mode trial --output-dir benchmarks
```

**Full download** (complete datasets):
```bash
python benchmarks/download_benchmarks.py --mode full --output-dir benchmarks
```

### 2. Run evaluation

**Trial run** (pipeline sanity check):
```bash
python eval/robot_risk_benchmarks.py \
    --model-path ./checkpoints/stage2/final_model.pt \
    --benchmarks-dir ./benchmarks \
    --top-n 3 --limit 20 --trial
```

**Full evaluation**:
```bash
python eval/robot_risk_benchmarks.py \
    --model-path ./checkpoints/stage2/final_model.pt \
    --benchmarks-dir ./benchmarks \
    --top-n 3 --main
```

Add `--use-va-refiner` to evaluate with the VA Refiner hallucination mitigation module enabled.

### 3. View results

- **Per-sample logs**: `benchmarks/results/*.jsonl`
- **Metrics summary**: `benchmarks/results/metrics_summary_{trial|main}.json`
- **Plots**: `benchmarks/plots/robot_risk_benchmarks/`

## Metrics

| Metric | Description |
|---|---|
| Risk Accuracy | % correct safe/danger classification |
| False Alarm Rate | Safe scenes predicted as dangerous (overreaction) |
| Missed Emergency Rate | Dangerous scenes predicted as safe |
| Top-1 Accuracy | Correct top robot pick (robot-selection only) |
| Top-N Recall | Fraction of GT robots in predicted set (robot-selection only) |
| Deployment Safety | Correct refusal for extreme-risk scenarios (robot-selection only) |

## Directory Structure

```
benchmarks/
  robot-selection-dataset/     # Local dataset (250 scenarios, 5 robots)
  geobench_vlm/                # Downloaded GEOBench-VLM data
  veri_emergency/              # Downloaded VERI-Emergency data
  results/                     # Evaluation output (JSONL + metrics JSON)
  plots/robot_risk_benchmarks/ # Publication-quality figures (PNG + PDF)
  download_benchmarks.py       # Dataset download script
  README.md                    # This file
```

## Plots Generated

| File | Description |
|---|---|
| `overview_metrics` | Grouped bar chart: accuracy, false alarm, missed emergency across benchmarks |
| `veri_confusion` | 2×2 confusion matrix for VERI-Emergency |
| `veri_category_rates` | False alarm / missed emergency per VERI category (AB, PME, ND) |
| `robot_selection_heatmap` | Scenario type × robot selection proportions (GT vs predicted) |
| `geobench_tasks` | Per-task risk accuracy for GEOBench-VLM |
| `severity_distributions` | Predicted risk severity histograms for safe vs dangerous GT |
