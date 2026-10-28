"""Rebuild the Task4 dataset with correctly paired parameter rows.

The released double_cone_dataset_with_physics.pt mispairs (Ma, Tv, Re) rows
with flow fields (46/51 rows carry another case's parameters). The freestream
state is exactly recoverable from the noise-free inlet columns, and a
Hungarian assignment on (Ma, Re) against the stored table restores the
pairing with Ma error < 0.02 / Re error < 1% on all 51 cases.

Usage: python repair_dataset.py [in.pt] [out.pt]
Writes double_cone_dataset_repaired.pt next to the input by default; the
original is never modified. The permutation is stored as 'repair_perm'.
"""
import sys
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

R_N2 = 296.8  # specific gas constant [J/(kg K)]


def freestream(y):
    """Per-case (Ma_inf, per-meter Re proxy) from the exact inlet columns."""
    rho, u, p = [], [], []
    for s in range(y.shape[0]):
        b = y[s, :, :, 0:3]
        rho.append(b[0, 8, 0].item()); u.append(b[1, 8, 0].item()); p.append(b[3, 8, 0].item())
    rho, u, p = np.array(rho), np.array(u), np.array(p)
    T = p / (rho * R_N2)
    ma = u / np.sqrt(1.4 * R_N2 * T)
    mu = 1.663e-5 * (T / 273.15) ** 1.5 * (273.15 + 107) / (T + 107)  # Sutherland N2
    return ma, rho * u / mu


def repair(inp, outp):
    d = torch.load(inp, weights_only=False)
    x, y = d['x'].float(), d['y'].float()
    pi = d['param_info']
    mn = torch.tensor(pi['min'], dtype=torch.float32)
    mx = torch.tensor(pi['max'], dtype=torch.float32)
    stored = (x[:, 2:5, 0, 0] * (mx - mn) + mn).numpy()

    ma_t, re_raw = freestream(y)
    # one global scale (unit Reynolds per meter -> table Reynolds range)
    re_s = re_raw * (stored[:, 2].min() / re_raw.min())
    C = (((stored[None, :, 0] - ma_t[:, None]) / 0.1) ** 2
         + ((stored[None, :, 2] - re_s[:, None]) / re_s[:, None] / 0.01) ** 2)
    _, perm = linear_sum_assignment(C)

    ok_ma = np.abs(stored[perm, 0] - ma_t) < 0.02
    ok_re = np.abs(stored[perm, 2] - re_s) / re_s < 0.01
    print(f'matched {int((ok_ma & ok_re).sum())}/51 within Ma<0.02, Re<1%; '
          f'{int((perm == np.arange(len(perm))).sum())} rows were already correct')

    x_rep = x.clone()
    x_rep[:, 2:5] = x[perm][:, 2:5]
    torch.save({**d, 'x': x_rep, 'repair_perm': torch.tensor(perm),
                'repair_note': 'param rows re-paired to fields: row perm[i] of the '
                               'original x[:,2:5] belongs to field i'}, outp)
    print(f'wrote {outp}')


if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else '../data/double_cone_dataset_with_physics.pt'
    outp = sys.argv[2] if len(sys.argv) > 2 else '../data/double_cone_dataset_repaired.pt'
    repair(inp, outp)
