"""Per-level wavelet-coefficient errors of a trained WIEN-DeepONet (test set).

Reports, per DWT level: normalized-space MSE and rel-L2 (the training
objective space), rel-L2 in denormalized coefficient units, and the level's
share of total coefficient energy (for context).
"""

import torch
from model_wien_deeponet import WIENDeepONet
from data_loader import AirfoilDataset

dev = "cuda" if torch.cuda.is_available() else "cpu"

cache = torch.load("output/wien_deeponet/coef_cache.pt", map_location="cpu",
                   weights_only=True)
model = WIENDeepONet().to(dev)
model.load_state_dict(torch.load("output/wien_deeponet/best_model.pth",
                                 map_location=dev, weights_only=True))
model.eval()

test_idx = sorted(set(range(cache["targets"][0].shape[0]))
                  - set(cache["train_indices"].tolist()))
ds = AirfoilDataset("../data/airfoil_unified_128x128.pt", mode="deeponet")
xb = ds.geo_points[test_idx].reshape(len(test_idx), -1).float().to(dev)

names = ["L0 Yl    22x22  (low-freq)",
         "L1 Yh(J) 22x22  (detail, coarsest)",
         "L2 Yh    37x37  (detail)",
         "L3 Yh    67x67  (detail, finest)"]

with torch.no_grad():
    preds = model(xb)

stats, energies = [], []
for l in range(4):
    o = model.out_dims[l]
    tgt = cache["targets"][l][test_idx].to(dev)          # (N, P, o) normalized
    p = preds[l]
    mse = ((p - tgt) ** 2).mean(dim=(1, 2))               # per sample
    rl2 = torch.linalg.norm(p - tgt, dim=(1, 2)) / torch.linalg.norm(tgt, dim=(1, 2))
    mean, maxx = model.coef_mean[l, :o], model.coef_maxx[l, :o]
    pd, td = p * maxx + mean, tgt * maxx + mean           # denormalized
    rl2d = (torch.linalg.norm(pd - td, dim=(1, 2))
            / torch.linalg.norm(td, dim=(1, 2)))
    stats.append((mse.mean().item(), rl2.mean().item(), rl2d.mean().item()))
    energies.append((torch.linalg.norm(td, dim=(1, 2)) ** 2).mean().item())

tot_e = sum(energies)
print(f"{'level':<34}{'MSE(norm)':>11}{'RL2E(norm)':>12}{'RL2E(coef)':>12}"
      f"{'energy%':>9}")
for n, (m, r, rd), e in zip(names, stats, energies):
    print(f"{n:<34}{m:>11.3e}{r:>12.4f}{rd:>12.4f}{100 * e / tot_e:>8.1f}%")
