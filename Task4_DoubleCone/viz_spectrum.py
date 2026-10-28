# Compare power spectra of model predictions vs CFD ground truth for one test
# sample: deeponet_fourier (Fourier-feature trunk) vs wavelet_deeponet (Gabor trunk).
# Spectrum = row-averaged 1D power spectrum along the streamwise (x) axis.
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from data_loader import get_split_indices, vacuum_mask
from train import build_model, GridAdapter

CHANNELS = ['Density rho', 'Velocity u', 'Velocity v', 'Pressure p']
MODELS = {  # run_name -> model_name (checkpoint at output/<run_name>/best_model.pth)
    'deeponet_fourier': 'deeponet',
    'wavelet_deeponet': 'wavelet_deeponet',
}
COORD = {'wavelet_deeponet'}
COLORS = {'Ground Truth': 'black', 'deeponet_fourier': '#0072B2', 'wavelet_deeponet': '#D55E00'}


def load_model(run_name, device):
    ckpt = torch.load(os.path.join('output', run_name, 'best_model.pth'),
                      map_location=device, weights_only=True)
    name = MODELS[run_name]
    model = build_model(name, ckpt.get('config', {}).get('use_fourier', True))
    if name in COORD:
        model = GridAdapter(model)
    state = ckpt.get('model_state', ckpt)
    if name in COORD and not any(k.startswith('m.') for k in state):
        state = {f'm.{k}': v for k, v in state.items()}  # legacy pre-adapter ckpt
    model.load_state_dict(state)
    return model.to(device).eval()


def predict_phys(model, x_in, x_norm, y_norm):
    """Encoded grid in -> physical [4, H, W] (pressure channel back from log10)."""
    with torch.no_grad():
        pred = y_norm.decode(model(x_norm.encode(x_in)))
    pred[:, 3] = 10 ** pred[:, 3]
    return pred.squeeze(0).cpu().numpy()


def streamwise_spectrum(field, valid):
    """Row-averaged 1D power spectrum along x. Vacuum cells are filled with the
    row's fluid mean (avoids a zero step edge leaking into high k), each row is
    detrended, and only rows containing fluid cells contribute."""
    total = None
    n = 0
    for row, vrow in zip(field, valid):
        if not vrow.any():
            continue
        row = row.copy()
        row[~vrow] = row[vrow].mean()
        row -= row.mean()
        s = np.abs(np.fft.rfft(row)) ** 2
        total = s if total is None else total + s
        n += 1
    return total / n


def main():
    parser = argparse.ArgumentParser(description='Spectral comparison on one test sample')
    parser.add_argument('--data_path', type=str, default='../data/double_cone_dataset.pt')
    parser.add_argument('--test_sample_idx', type=int, default=0, help='Index into the TEST set')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset stores numpy arrays (needs pickle); local trusted file, same as eval.py
    data = torch.load(args.data_path, weights_only=False)
    x_data, y_data = data['x'].float(), data['y'].float()
    valid = ~vacuum_mask(y_data).cpu().numpy()

    # Same seeded split / train-only stats as train.py & eval.py
    train_idx, test_idx = get_split_indices(x_data.shape[0])
    y_log = y_data.clone()
    y_log[:, 3] = torch.log10(y_data[:, 3] + 1e-6)
    from data_loader import MinMaxNormalizer
    x_norm = MinMaxNormalizer(torch.amin(x_data[train_idx], dim=(0, 2, 3), keepdim=True),
                              torch.amax(x_data[train_idx], dim=(0, 2, 3), keepdim=True)).to(device)
    y_norm = MinMaxNormalizer(torch.amin(y_log[train_idx], dim=(0, 2, 3), keepdim=True),
                              torch.amax(y_log[train_idx], dim=(0, 2, 3), keepdim=True)).to(device)

    gi = test_idx[args.test_sample_idx].item()
    x_in = x_data[gi].unsqueeze(0).to(device)
    y_true = y_data[gi].numpy()

    freqs = np.fft.rfftfreq(y_true.shape[2])  # cycles per grid cell, along W (x)
    spectra = {'Ground Truth': y_true}
    for run_name in MODELS:
        model = load_model(run_name, device)
        spectra[run_name] = predict_phys(model, x_in, x_norm, y_norm)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Streamwise Power Spectrum | Test sample {args.test_sample_idx} '
                 f'(global idx {gi}) | fluid rows only, vacuum filled with row mean', fontsize=12)
    for ax, ch in zip(axes.flat, range(4)):
        for label, field in spectra.items():
            spec = streamwise_spectrum(field[ch], valid)
            ax.loglog(freqs[1:], spec[1:], color=COLORS[label],
                      lw=2.5 if label == 'Ground Truth' else 1.5,
                      ls='-' if label == 'Ground Truth' else '--', label=label)
        ax.set_title(CHANNELS[ch])
        ax.set_xlabel('Wavenumber (cycles / cell)')
        ax.set_ylabel('Power')
        ax.grid(True, which='both', alpha=0.3)
    axes[0, 0].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = 'spectrum_deeponet_fourier_vs_wavelet.png'
    plt.savefig(out, dpi=150)
    plt.close(fig)

    # Numeric summary: relative L2 of each model's spectrum vs GT, per channel
    for run_name in MODELS:
        errs = []
        for ch in range(4):
            s_gt = streamwise_spectrum(spectra['Ground Truth'][ch], valid)
            s_m = streamwise_spectrum(spectra[run_name][ch], valid)
            errs.append(np.linalg.norm(s_m - s_gt) / (np.linalg.norm(s_gt) + 1e-12))
        print(f'{run_name}: spectral rel L2 per channel = '
              + ', '.join(f'{CHANNELS[ch]} {errs[ch]:.3f}' for ch in range(4)))
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
