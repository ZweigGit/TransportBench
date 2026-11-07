# 🌪️ Task II: Cylinder Parameter Generalization

This directory contains the code, models, and evaluation scripts for **Task II** of TransportBench. This task evaluates the robustness of neural architectures against wide unstructured parameter variations (Knudsen number $Kn$ and Mach number $Ma$) and strong non-equilibrium wake expansion effects around a circular cylinder.

## 📂 Directory Structure
- `train.py`: Unified training script with auto-saving logic and dynamic input parsing.
- `eval.py`: Standardized evaluation script to compute MAE / Relative $L_2$ errors and generate pure contour visualizations with solid obstacle masking.
- `data_loader.py`: Handles geometric masking and data parsing for diverse topological learners.
- `make_log_rho_dataset.py`: Builds the log10-rho variant of the dataset (`cylinder_full_2400_logrho.pt`).
- `model_*.py`: Model definitions for all 6 baseline architectures configured to $\sim$1M parameters.
- `checkpoints/`: Contains the pre-trained best weights for all evaluated models.

## 📊 Datasets

| File | rho channel | Notes |
|------|-------------|-------|
| `cylinder_full_2400.pt` | linear min-max to `[-1, 1]` | original, used for all existing checkpoints |
| `cylinder_full_2400_logrho.pt` | `log10` then min-max to `[-1, 1]` | built by `make_log_rho_dataset.py` |

`rho` spans ~5 decades ($8.7\times10^{-9}$ – $1.1\times10^{-3}$), so a shared linear container
gives it almost no relative resolution (median round-trip error $4.7\times10^{-5}$, up to
$6.9\times10^{-3}$). Storing `log10(rho)` cuts that to $3.1\times10^{-7}$ median, $2.0\times10^{-6}$
max. Values below `--rho_floor` (default `1e-8`, 2 points) are clamped.

The two files are drop-in interchangeable: `fit_train_normalizer()`, `train.py` and `eval.py`
all work unchanged, and the `[0, 1]` values seen by the loss are log-space for rho in the
`logrho` file. `CylinderDataset.denormalize()` inverts the transform, so it returns physical
values for both files.

## 🚀 Quick Start

### 1. Evaluation (Inference)
To evaluate a pre-trained model (e.g., Point Transformer) and reproduce the pure flow-field visualizations, run:
```bash
python eval.py --model pt --data_path ./data/cylinder_full_2400.pt

### 2. Training from Scratch
To train a model from scratch using the unified pipeline:
```bash
python train.py --model pt --epochs 2500 --batch_size 16 --lr 5e-4 --data_path ./data/cylinder_full_2400.pt