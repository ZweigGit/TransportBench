# 🚀 Task IV: Double Cone (Representational Stress Test)

This directory contains the codebase for **Task IV**, the ultimate representational stress test of TransportBench. While physically governed by macroscopic continuum mechanics, this Mach-10 hypersonic shock-wave problem pushes neural architectures to their absolute limits through three extreme constraints:
1. **Extreme Sparsity**: A $17 \times 384$ high-aspect-ratio grid.
2. **Extreme Few-Shot**: Only 45 training samples available.
3. **Extreme Gradients**: Severe mathematical discontinuities across the shockwave.

Crucially, this directory implements an **Ablation Study** to test the "Cure vs. Poison" hypothesis of high-frequency Fourier Feature (FF) injection.

## 📂 Directory Structure
- `train.py`: Unified script supporting the Curriculum Learning protocol and the `--use_fourier` ablation flag.
- `eval.py`: Evaluation script that automatically loads the corresponding ablated weights and plots the 1D near-wall pressure distributions.
- `data_loader.py`: Dataset loading, the `MinMaxNormalizer`, the vacuum mask and the few-shot split logic.
- `model_*.py`: Model definitions scaled to the Extreme Regime budget ($\sim$33M parameters), featuring dynamic positional encoding.
- `checkpoints/`: Directory for pre-trained weights (Note: weights are hosted externally due to GitHub's 100MB file size limit).

## 📊 Targets, Masking and Metric Space

- Pressure is stored as `log10(p + 1e-6)`: raw `p` spans 79.8 – 3.365e5 (3.63 decades), so a linear
  container would squeeze the median pressure into 0.18% of `[0, 1]`.
- 313 domain-filler cells (body-interior wedge + outflow corners) are exactly zero in every sample.
  The mask must be taken on the **raw** targets: after the log transform those cells read
  `log10(1e-6) = -6`, never `0`, so a mask taken afterwards silently matches nothing.
- Min/max statistics for the targets are computed on **fluid cells only**, on the train split.
  Including the vacuum floor would stretch the pressure span from 3.63 to 11.53 and leave real
  targets occupying just 31% of `[0, 1]`.
- MAE / MSE / Relative $L_2$ are reported in the normalized `[0, 1]` space (p in `log10`), vacuum
  cells excluded. They are **not** physical units and are not comparable across Tasks I–III.

## 🚀 Quick Start

### 1. Evaluation (Ablation Inference)
To evaluate the Vanilla configuration (without high-frequency injection):
```bash
python eval.py --model unet
```
To evaluate the Fourier Feature injected model:
```bash
python eval.py --model unet --use_fourier
```

### 2. Training (with Curriculum Protocol)
Train the Vanilla model:
```bash
python train.py --model unet --lr 1e-3
```
Train with Fourier Feature injection:
```bash
python train.py --model unet --lr 1e-3 --use_fourier
```

## 🏆 Performance Overview (Pressure Field Rel $L_2$ Error)
The results shatter the myth that explicit high-frequency injection is a universal cure. **Local receptive fields (U-Net) naturally dominate sharp shock capturing**, while Fourier injection acts as a double-edged sword, triggering severe Gibbs phenomena or non-physical noise in global models.

| Model | Vanilla (No FF) | With Fourier Feature (+FF) |
| :--- | :--- | :--- |
| **U-Net** | **12.3% (SOTA)** | 15.0% *(Micro-noise introduced)* |
| DeepONet | 16.6% *(Smoothed shocks)*| 21.1% *(Uncontrollable noise)* |
| FNO | 17.6% *(Lost features)*| 17.1% *(Severe Gibbs oscillations)* |
| AutoEncoder | 26.2% | 20.1% |
| ViT | 31.9% | 23.8% |
| Point Trans. | 42.9% | 54.6% *(Catastrophic Aliasing)* |
```
