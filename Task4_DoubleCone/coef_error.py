"""Per-band wavelet-coefficient errors of the trained wien_deeponet (best ckpt).

Reports the training metric per band: energy-normalized relative MSE,
support-weighted — same formula as the model's aux_loss, but decomposed.
"""
import torch
from data_loader import get_dataloader_and_stats
from train import build_model
from wavelet_haar import dwt2_stack, support_weights

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader, x_norm, y_norm, vac = get_dataloader_and_stats(
    '../data/double_cone_dataset.pt', 9, dev)

ck = torch.load('output/wien_deeponet/best_model.pth',
                map_location=dev, weights_only=True)
cfg = ck['config']
ctx = {'n_param': cfg['n_param'], 'vac': vac.to(dev),
       **{k: cfg[k] for k in ('levels', 'aux_weight', 'wavelet') if k in cfg}}
m = build_model('wien_deeponet', False, ctx).to(dev)
m.load_state_dict(ck['model_state'])
m.eval()
print(f"epoch {ck['epoch']}, best field test MSE {ck['best_test_loss']:.6f}, "
      f"levels={m.levels}, wavelet={m.wavelet}")


def band_errs(loader):
    vals, n = None, 0
    with torch.no_grad():
        for x, y in loader:
            xe, ye = x_norm.encode(x), y_norm.encode(y)
            _, bands = m(xe, ye, return_bands=True)
            A_t, Ds_t, hw = dwt2_stack(ye, m.levels, m.wavelet)
            w = support_weights(~ctx['vac'], hw)
            errs = [m._rel_mse(bands['A'], A_t, w[0]).item()]
            errs += [m._rel_mse(torch.cat(bd, 1), torch.cat(bt, 1), wj).item()
                     for bd, bt, wj in zip(bands['D'], Ds_t, w)]
            vals = errs if vals is None else [a + b for a, b in zip(vals, errs)]
            n += 1
    return [v / n for v in vals]


J = m.levels
labels = ['A_J'] + [f'D_{J - j}' for j in range(J)]
for tag, loader in [('train', train_loader), ('test', test_loader)]:
    e = band_errs(loader)
    row = '  '.join(f'{l} {v:.4f} (L2 {v ** 0.5:.3f})' for l, v in zip(labels, e))
    print(f'{tag}: {row}')
    print(f'      mean rel-MSE {sum(e) / len(e):.4f}')
