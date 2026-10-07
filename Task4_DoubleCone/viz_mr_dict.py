"""Visualize the frozen multi-resolution dictionary of MR_DeepONet.

Panel per dictionary group: support rectangle [t, t + (dec_len-1)*s]^2 of
every tensor atom (alpha stacking shows overlap), unit domain outlined in
black. Final panel: aggregate support-coverage count per point of [0,1]^2,
summed over all groups. Run: python viz_mr_dict.py
"""

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from model_mr_deeponet import MR_DeepONet, _wavefun

FILL = '#4C78A8'
VIEW = (-0.35, 1.35)


def coverage_count(bank, xs, width):
    """[n] how many atoms of `bank` cover each point of xs."""
    lo = bank.t[None, :]
    return ((lo <= xs[:, None]) & (xs[:, None] <= lo + width / bank.inv_s)).sum(1)


def main():
    m = MR_DeepONet(branch_dim=3, trunk_dim=2, hidden_dim=385, num_outputs=4,
                    depth=4, levels=4, activation='GELU', wavelet='db4', stride=2)
    plan = m.trunk_net.plan
    width = _wavefun('db4')[0] - 1  # support extent in units of s

    fig, axes = plt.subplots(4, 4, figsize=(16, 15))
    titles = [('s=1', '(φ,φ)')] + \
             [(f'j={j}', f'({a},{b})') for j in range(4)
              for a, b in (('φ', 'ψ'), ('ψ', 'φ'), ('ψ', 'ψ'))]

    n = 501
    xs = torch.linspace(0.0, 1.0, n)
    cov = torch.zeros(n, n)
    total = 0

    for ax, (fx, fy), (scale_tag, fam) in zip(axes.flat, plan, titles):
        wx, wy = width / fx.inv_s, width / fy.inv_s
        rects = [Rectangle((tx.item(), ty.item()), wx, wy)
                 for tx in fx.t for ty in fy.t]
        ax.add_collection(PatchCollection(rects, facecolor=FILL, alpha=0.08,
                                          edgecolor='none'))
        ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', lw=1.6))
        ax.set_xlim(*VIEW)
        ax.set_ylim(*VIEW)
        ax.set_aspect('equal')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.tick_params(labelsize=7)
        n_atoms = fx.n_atoms * fy.n_atoms
        total += n_atoms
        ax.set_title(f'{scale_tag} {fam}  ({fx.n_atoms}×{fy.n_atoms} = {n_atoms} atoms)',
                     fontsize=10)
        cov += coverage_count(fx, xs, width)[:, None] * coverage_count(fy, xs, width)[None, :]

    ax = axes.flat[13]
    im = ax.imshow(cov, origin='lower', extent=(0, 1, 0, 1), cmap='Blues')
    ax.set_title(f'coverage count, all {total} atoms', fontsize=10)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes.flat[14:]:
        ax.set_visible(False)

    fig.suptitle('MR_DeepONet db4 dictionary: atom supports on [0,1]$^2$ '
                 '(black = domain)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig('mr_dict_viz.png', dpi=150)
    print(f'total atoms = {total}')


if __name__ == '__main__':
    main()
