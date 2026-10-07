"""Print parameter counts for all Task4 models (configs as in train.py)."""

from train import build_model

names = ['ae', 'deeponet', 'fno', 'pt', 'unet', 'vit',
         'hyperdeeponet', 'mscale_deeponet', 'hyper_mscale_deeponet', 'c_hyperdeeponet', 'c_hyper_mscale_deeponet',
         'fusion_deeponet', 'residual_fusion_deeponet', 'mr_deeponet', 'wavelet_deeponet', 'mwavelet_deeponet']

print(f"{'model':<22} {'params':>10}")
print("-" * 34)
total = 0
for n in names:
    m = build_model(n, use_fourier=True)
    # complex params count as 2 real dof (wavelet_deeponet trunk)
    p = sum(x.numel() * (2 if x.is_complex() else 1) for x in m.parameters())
    total += p
    print(f"{n:<22} {p/1e6:8.2f} M")
print("-" * 34)
print(f"{'TOTAL':<22} {total/1e6:8.2f} M")
