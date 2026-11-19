"""Estimate the DSMC statistical-noise floor of the Task1 fields.

DSMC particle noise is approximately spatially white; for white noise the
median absolute DWT coefficient at the FINEST scale relates to the noise
std as sigma = median(|c|) / 0.6745 (Donoho-Johnstone MAD estimator).
The smooth mean field contributes almost nothing there, and the median is
robust to the localized real high-frequency content (wake, shock).

The floor is translated into the exact eval protocol: per-sample rel-L2 /
MSE / MAE over fluid points in the [0,1] train-normalized space, averaged
over the 20 seed-42 test samples.
"""

import torch
from pytorch_wavelets import DWTForward
from torch.utils.data import random_split

from data_loader import AirfoilDataset
from model_wien_deeponet import harmonic_fill

dev = "cuda" if torch.cuda.is_available() else "cpu"

ds = AirfoilDataset("../data/airfoil_unified_128x128.pt", mode="fno")
tr, _ = random_split(ds, [80, 20], generator=torch.Generator().manual_seed(42))
ds.fit_train_normalizer(tr.indices)
test_idx = sorted(set(range(100)) - set(tr.indices))

fields = ds.flow_label[test_idx].permute(0, 3, 1, 2).float().to(dev)  # (20,4,H,W)
fluid = ds.geo_mask[test_idx].to(dev).bool()
filled = harmonic_fill(fields, fluid)  # protect the estimator from body zeros

dwt = DWTForward(J=3, wave="db4", mode="symmetric").to(dev)
with torch.no_grad():
    _, Yh = dwt(filled)
finest = Yh[0]                                                   # (20,4,3,67,67)
sigma = finest.abs().flatten(2).median(dim=2).values / 0.6745     # (20,4)

nf = fluid.flatten(1).sum(1).float()                             # fluid cells
sq = (fields * fluid).sum(dim=(2, 3))                            # (20,4) signal energy
noise_e = (sigma.pow(2) * nf[:, None]).sum(1)                    # per-sample noise energy

rl2 = (noise_e / sq.sum(1)).sqrt()                               # per-sample floor
mse = noise_e / (nf * 4)                                         # eval-MSE convention
mae = (sigma * 0.7979).mean(1)                                   # E|N(0,s)| = 0.798 s

print(f"noise sigma per channel ([0,1] units), mean over test samples:")
for c, name in enumerate(["rho", "u", "v", "T"]):
    rel_ch = (sigma[:, c].pow(2) * nf).sum(0).sqrt() / sq[:, c].sum().sqrt()
    print(f"  {name}: sigma = {sigma[:, c].mean():.5f} "
          f"(+- {sigma[:, c].std():.5f}),  ch-rel-L2 floor = {rel_ch:.4f}")
print()
print(f"pooled floor (eval protocol, mean over 20 test samples):")
print(f"  RL2E floor : {rl2.mean():.5f}  (+- {rl2.std():.5f})")
print(f"  MSE  floor : {mse.mean():.3e}")
print(f"  MAE  floor : {mae.mean():.5f}")
print()
print("observed best models:  RL2E 0.01662-0.01681 | MSE 8.46e-05-8.68e-05 | MAE 0.00646-0.00653")
