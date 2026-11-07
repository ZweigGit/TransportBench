"""Build the log10-rho variant of the Task II cylinder dataset.

Why: rho spans ~5 decades (8.7e-9 .. 1.1e-3) while u/v/T span ~1-2 decades, so
the shared linear [-1, 1] min-max container gives rho almost no relative
resolution. Storing log10(rho) puts all four channels on a comparable scale.

Layout of the output file (only the rho channel changes):

  flow_label[..., 0]    = 2*(log10(clamp(rho, floor)) - lo) / (hi - lo) - 1
  flow_label[..., 1:4]  = u, v, T copied verbatim (already [-1, 1])
  flow_stats['rho']     = {'min': lo, 'max': hi,
                           'transform': 'log10', 'floor': floor}
                          lo/hi are the log10 bounds over FLUID points only,
                          so the decode used everywhere else is
                              rho = 10 ** (z * (hi - lo) + lo)
                          with z the [0, 1] train-split value produced by
                          CylinderDataset.fit_train_normalizer().
  everything else       = copied verbatim

The [-1, 1] container is kept, so CylinderDataset.fit_train_normalizer() and the
loss/metric code in train.py / eval.py need no change: they operate in the
stored space, which is now log space for rho. Use CylinderDataset.denormalize()
to go back to physical values.

Cylinder-interior cells are non-physical fillers (constant rho = 5.4e-4). They
are transformed exactly like fluid cells, so the file contains no -inf/NaN and
their stored value stays inside [lo, hi].
"""
import os
import argparse
import torch

CHANNELS = ['rho', 'u', 'v', 'T']


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_src = os.path.join(script_dir, '..', 'data', 'cylinder_full_2400.pt')
    default_dst = os.path.join(script_dir, '..', 'data', 'cylinder_full_2400_logrho.pt')

    p = argparse.ArgumentParser(description="Build the log10-rho variant of the Task II cylinder dataset")
    p.add_argument('--src', default=default_src, help='Source dataset (.pt)')
    p.add_argument('--dst', default=default_dst, help='Output dataset (.pt)')
    p.add_argument('--rho_floor', type=float, default=1e-8,
                   help='Lower clamp applied before log10 (rho is strictly positive; default 1e-8)')
    p.add_argument('--overwrite', action='store_true', help='Overwrite --dst if it exists')
    args = p.parse_args()

    if os.path.exists(args.dst) and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {args.dst} (pass --overwrite)")

    print(f"Loading {args.src} ...")
    d = torch.load(args.src, map_location='cpu', weights_only=False)
    for k in ['input_params', 'flow_label', 'mask', 'grid_coords', 'flow_stats']:
        if k not in d:
            raise SystemExit(f"Source dataset is missing key: {k}")

    label = d['flow_label']                      # [N, H, W, 4], rho channel in [-1, 1]
    mask = d['mask']                             # [N, 1, H, W]
    stats = d['flow_stats']
    if list(stats.keys()) != CHANNELS:
        raise SystemExit(f"Unexpected flow_stats key order {list(stats.keys())}, expected {CHANNELS}")

    st_rho = stats['rho']
    gmin, gmax = float(st_rho['min']), float(st_rho['max'])
    if gmin <= 0.0:
        raise SystemExit(f"rho stats are not strictly positive (min={gmin}); cannot take log10")

    z = label[..., 0]
    rho = (z + 1.0) / 2.0 * (gmax - gmin) + gmin   # physical rho
    if float(rho.min()) <= 0.0:
        raise SystemExit(f"Decoded physical rho has non-positive values (min={float(rho.min())}); abort")

    fluid = (mask[:, 0] >= 0.5)                    # [N, H, W], same rule as CylinderDataset
    n_clip = int((rho < args.rho_floor).sum())
    n_fluid = int(fluid.sum())
    print(f"Fluid points: {n_fluid}/{fluid.numel()}")
    print(f"rho physical range: [{float(rho.min()):.4g}, {float(rho.max()):.4g}]")
    print(f"Points below floor {args.rho_floor:g}: {n_clip} ({100.0 * n_clip / rho.numel():.3g}% of all points)")

    log_rho = torch.log10(rho.clamp_min(args.rho_floor))
    lo, hi = float(log_rho[fluid].min()), float(log_rho[fluid].max())
    if hi - lo <= 0.0:
        raise SystemExit("Degenerate log10 range over fluid points")
    print(f"log10(rho) over fluid points: [{lo:.6f}, {hi:.6f}]  span={hi - lo:.4f} decades")

    # Re-encode into the same [-1, 1] container the loader expects.
    label[..., 0] = 2.0 * (log_rho - lo) / (hi - lo) - 1.0

    stats['rho'] = {'min': lo, 'max': hi, 'transform': 'log10', 'floor': args.rho_floor}
    d['flow_label'] = label
    d['flow_stats'] = stats
    d['description'] = (str(d.get('description', 'Unified Grid 128x192')) +
                        ' | rho stored as log10 (floor %g)' % args.rho_floor)

    print("Channel ranges in the new file (stored space):")
    for i, name in enumerate(CHANNELS):
        c = label[..., i]
        print(f"  {name:4s} [{float(c.min()):+.4f}, {float(c.max()):+.4f}]  "
              f"{stats[name].get('transform', 'identity')}")

    torch.save(d, args.dst)
    print(f"Saved {args.dst} ({os.path.getsize(args.dst) / 1e9:.2f} GB)")


if __name__ == '__main__':
    main()
