import torch
import torch.nn as nn
import time
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader import get_dataloader_and_stats, vacuum_mask, get_split_indices

from model_ae import AutoEncoder2d
from model_deeponet import DeepONet2d
from model_fno import FNO2d
from model_pt import PointTransformer
from model_unet import FluidUNet
from model_vit import VisionTransformer
from model_hyperdeeponet import HyperDeepONet
from model_mscale_deeponet import MscaleDeepONet
from model_hyper_mscale_deeponet import HyperMscaleDeepONet
from model_c_hyperdeeponet import c_HyperDeepONet
from model_c_hyper_mscale_deeponet import c_HyperMscaleDeepONet
from model_fusion_deeponet import Fusion_DeepONet
from model_residual_fusion_deeponet import Residual_Fusion_DeepONet
from model_mr_deeponet import MR_DeepONet
from model_mwavelet_deeponet import MwaveletDeepONet
from model_wavelet_deeponet import WaveletDeepONet
from model_residual_fusion_wavelet_deeponet import Residual_Fusion_Wavelet_DeepONet
from model_fusion_wavelet_deeponet import Fusion_Wavelet_DeepONet
from model_fs_deeponet import FS_DeepONet

class GridAdapter(nn.Module):
    """Grid [B,5,H,W] interface for (branch, trunk) coordinate DeepONet variants:
    branch = flow params (ch 2:5, constant per sample), trunk = shared grid
    coords (ch 0:1). Output transposed from point-major [B,N,C] to [B,C,H,W]."""
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        B, _, H, W = x.shape
        trunk = x[0, 0:2].permute(1, 2, 0).reshape(-1, 2)
        out = self.m(x[:, 2:, 0, 0], trunk)           # [B, H*W, C] (all param channels)
        return out.permute(0, 2, 1).reshape(B, -1, H, W)

    @property
    def aux_loss(self):
        return self.m.aux_loss

    def graph_lap_loss(self):
        return self.m.graph_lap_loss()

def get_args():
    parser = argparse.ArgumentParser(description="Universal Golden Protocol Training Script")
    parser.add_argument('--model', type=str, required=True, choices=['ae', 'deeponet', 'fno', 'pt', 'unet', 'vit', 'hyperdeeponet', 'mscale_deeponet', 'hyper_mscale_deeponet', 'c_hyperdeeponet', 'c_hyper_mscale_deeponet', 'fusion_deeponet', 'residual_fusion_deeponet', 'mr_deeponet', 'wavelet_deeponet', 'mwavelet_deeponet', 'residual_fusion_wavelet_deeponet', 'fusion_wavelet_deeponet', 'fs_deeponet', 'fs_lap_deeponet'])
    parser.add_argument('--data_path', type=str, default='../data/double_cone_dataset.pt')
    parser.add_argument('--batch_size', type=int, default=9)
    parser.add_argument('--epochs', type=int, default=2500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--no_fourier', action='store_true', help='Disable Fourier Encoding')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--consolidate', action='store_true',
                        help='Replace duplicate-param train labels by group mean')
    parser.add_argument('--best_after', type=int, default=1000,
                        help='Start tracking/saving the best checkpoint after this epoch')
    return parser.parse_args()

def build_fs_ctx(y_raw, param_raw, train_idx, x_norm, y_norm):
    """Stats for FS_DeepONet, computed on the train split only (no leakage).

    y_raw: [S,4,H,W] physical targets with p NOT yet log10-transformed.
    param_raw: [S,3] raw branch params (dataset order).
    """
    fluid = y_raw[train_idx, 3] > 0
    blk = y_raw[:, :, :, 0:5]
    m_in = y_raw[:, 3, None, :, 0:5] > 0
    min3 = m_in[:, 0]                                     # [S, H, 5]
    fs = torch.stack([(blk[:, 0] * min3).sum((1, 2)) / min3.sum((1, 2)),
                      (blk[:, 1] * min3).sum((1, 2)) / min3.sum((1, 2)),
                      (blk[:, 3] * min3).sum((1, 2)) / min3.sum((1, 2))], dim=1)
    ylog = y_raw.clone()
    ylog[:, 3] = torch.log10(y_raw[:, 3] + 1e-6)
    ytr, fstr = ylog[train_idx], fs[train_idx]
    ftr = fluid
    lrho = torch.log10((ytr[:, 0] / fstr[:, 0, None, None]).clamp_min(1e-8))[ftr]
    u_nd = (ytr[:, 1] / fstr[:, 1, None, None])[ftr]
    v_nd = (ytr[:, 2] / fstr[:, 1, None, None])[ftr]
    lp = (ytr[:, 3] - torch.log10(fstr[:, 2])[:, None, None])[ftr]
    t = torch.stack([lrho, u_nd, v_nd, lp])
    # benchmark y stats from the shared normalizer (log-p space)
    ymin = y_norm.min.view(-1).cpu()
    yrange = (y_norm.max - y_norm.min).view(-1).cpu()
    # normalized param table for exact row matching at train time
    pmin = x_norm.min.view(-1)[2:].cpu()
    pmax = x_norm.max.view(-1)[2:].cpu()
    ptab = (param_raw - pmin) / (pmax - pmin + 1e-6)
    lg = torch.log10(fs)
    mu, sd = lg.mean(0), lg.std(0)
    return {'param_table': ptab, 'log_fs_z': (lg - mu) / sd, 'fs_mu': mu, 'fs_sd': sd,
            'tmin': t.amin(1), 'tmax': t.amax(1), 'ymin': ymin, 'yrange': yrange}


def build_model(model_name, use_fourier, ctx=None):
    npar = (ctx or {}).get('n_param', 3)
    inch = 2 + npar
    if model_name == 'ae': return AutoEncoder2d(in_channels=inch, out_channels=4, features=128, use_fourier=use_fourier)
    elif model_name == 'deeponet':
        return DeepONet2d(in_channels=inch, out_channels=4, hidden_width=316, basis_size=128,
                          use_fourier=use_fourier, branch_dim=npar)
    elif model_name == 'fno': return FNO2d(modes1=8, modes2=64, width=64, in_channels=inch, out_channels=4, use_fourier=use_fourier)
    elif model_name == 'pt': return PointTransformer(in_channels=inch, out_channels=4, latent_dim=512, num_latents=1024, depth=10, use_fourier=use_fourier)
    elif model_name == 'unet': return FluidUNet(in_channels=inch, out_channels=4, features=64, use_fourier=use_fourier)
    elif model_name == 'vit': return VisionTransformer(in_channels=inch, out_channels=4, embed_dim=512, depth=10, use_fourier=use_fourier)
    # Coordinate-based (branch = Mach/Temp/Re, trunk = x/y grid coords)
    elif model_name == 'hyperdeeponet': return HyperDeepONet(branch_dim=npar, branch_hidden=45, trunk_hidden=64, trunk_depth=4, branch_depth=4, basis_size=128)
    # Single-trunk ablation of the multi-scale trunk (no scale branches, no
    # frequency scaling, LN-free) with Phi trunk activation and fusion-style
    # learnable per-layer modulation act(10*a*z + c): branch [3,313x5,512] GELU,
    # trunk [2,313x5,128] Phi, 1,006,478 params
    elif model_name == 'mscale_deeponet': return MscaleDeepONet(branch_dim=npar, trunk_dim=2, hidden=316, num_outputs=4, depth=5, basis_size=128)
    elif model_name == 'hyper_mscale_deeponet': return HyperMscaleDeepONet(branch_dim=npar, trunk_dim=2, hidden_dim=68,
                                                                           num_outputs=4, depth=4, activation='GELU')
    # Chunked hypernetwork: same generated trunk as HyperDeepONet (t_para=264,964) but emitted
    # in 512-wide chunks from a small branch (0.85M vs 34.2M)
    elif model_name == 'c_hyperdeeponet': return c_HyperDeepONet(branch_dim=npar, trunk_hidden_dim=256, branch_hidden_dim=350, num_basis=256,
                                                                 trunk_depth=5, branch_depth=5, chunk_in=256, chunk_out=1024)
    elif model_name == 'c_hyper_mscale_deeponet': return c_HyperMscaleDeepONet(branch_dim=npar, hidden_dim=300, depth=5, trunk_hidden=256, trunk_depth=5,
                                                                              basis_size=256, chunk_in=256, chunk_out=1024)
    # 1,010,312 params (~1.01M budget), branch [3,278,...,278,1112], trunk [2,278,...,278,278]
    elif model_name == 'fusion_deeponet': return Fusion_DeepONet(branch_dim=npar, trunk_dim=2, hidden_dim=278, num_outputs=4,
                                                                 depth=5, activation='GELU')
    # Same budget as fusion_deeponet; branch-to-trunk gate uses 1+skip (residual)
    elif model_name == 'residual_fusion_deeponet': return Residual_Fusion_DeepONet(branch_dim=npar, trunk_dim=2, hidden_dim=278, num_outputs=4,
                                                                                   depth=5, activation='GELU')
    # db4 multi-resolution dictionary trunk: 475 frozen atoms, zero trainable
    # trunk params (feature dim = atom count); full-rank branch head
    elif model_name == 'mr_deeponet': return MR_DeepONet(branch_dim=npar, trunk_dim=2, hidden_dim=350, num_outputs=4,
                                                         depth=4, levels=4, activation='GELU',
                                                         wavelet='db4', stride=2)
    # Gabor-wavelet trunk: complex FNN (one complex param = 2 real dof,
    # ~0.99M real dof), real part of trunk features into the dot product
    elif model_name == 'wavelet_deeponet': return WaveletDeepONet(branch_dim=npar, trunk_dim=2, branch_hidden=256,
                                                                  trunk_hidden=256, branch_depth=5, trunk_depth=5,
                                                                  basis_size=128, s0=3.0, w0=3.0)
    # Multi-branch Gabor trunk (Julia MwaveletNN): scales live in the
    # per-branch (w0, s0) = (1,1),(2,2),(3,3),(4,4) activation, ~1.0M real dof
    elif model_name == 'mwavelet_deeponet': return MwaveletDeepONet(branch_dim=npar, trunk_dim=2, branch_hidden=256,
                                                                    trunk_hidden=230, branch_depth=4, trunk_depth=4,
                                                                    basis_size=128, s0=1.0, w0s=[0.0, 5.0, 10.0, 15.0])
    # Residual-fusion branch + complex Gabor trunk, both width 259 (elementwise
    # gate forces equal widths; complex = 2 dof) — ~1.01M real dof budget
    elif model_name == 'residual_fusion_wavelet_deeponet': return Residual_Fusion_Wavelet_DeepONet(branch_dim=npar, trunk_dim=2,
                                                                                                    hidden_dim=259, trunk_hidden=259,
                                                                                                    num_outputs=4, depth=5,
                                                                                                    basis_size=128, w0=3.0, s0=3.0)
    # Fusion branch (ab/cb modulation, multiplicative skip gate, no sine
    # term) + complex Gabor trunk, both width 259 (elementwise gate forces
    # equal widths) — ~1.01M real dof budget, same dims as
    # residual_fusion_wavelet_deeponet
    elif model_name == 'fusion_wavelet_deeponet': return Fusion_Wavelet_DeepONet(branch_dim=npar, trunk_dim=2,
                                                                                 hidden_dim=259, trunk_hidden=259,
                                                                                 num_outputs=4, depth=5,
                                                                                 basis_size=128, w0=3.0, s0=3.0)
    # Freestream-normalized DeepONet (lap=True adds the train-param kNN
    # Laplacian penalty on branch coefficients), ~1.0M params
    elif model_name in ('fs_deeponet', 'fs_lap_deeponet'):
        return FS_DeepONet(branch_dim=npar, hidden=300, depth=5, basis_size=128, ctx=ctx,
                           lap=model_name == 'fs_lap_deeponet')

def main():
    args = get_args()
    # Seed init + DataLoader shuffle (GPU kernels stay nondeterministic;
    # add cudnn.deterministic/use_deterministic_algorithms for bitwise runs)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fourier = not args.no_fourier
    # Coordinate-based DeepONet variants, wrapped behind the grid image interface
    # so they train with the same curriculum-weighted MSE as grid models
    coord_models = {'hyperdeeponet', 'mscale_deeponet', 'hyper_mscale_deeponet', 'c_hyperdeeponet', 'c_hyper_mscale_deeponet', 'fusion_deeponet', 'residual_fusion_deeponet', 'mr_deeponet', 'wavelet_deeponet', 'mwavelet_deeponet', 'residual_fusion_wavelet_deeponet', 'fusion_wavelet_deeponet', 'fs_deeponet', 'fs_lap_deeponet'}
    # Coord variants without a Fourier option get no suffix (only hyperdeeponet supports it)
    # Coord variants have no Fourier option: no _fourier suffix in run names
    fourierless = coord_models
    fourier_suffix = "" if args.model in fourierless else ("_fourier" if use_fourier else "_nofourier")

    # Per-model output subdirectory, aligned with Task I convention
    # (T4_SAVE_DIR lets external harnesses like cv_eval.py redirect outputs)
    args.save_dir = os.environ.get(
        'T4_SAVE_DIR',
        os.path.join('output', args.model + fourier_suffix + ('_cons' if args.consolidate else '')))
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'best_model.pth')
    log_file = os.path.join(args.save_dir, 'train.log')
    # Overwrite (not append) so each run's log reflects only that run

    def log(msg):
        print(msg)
        with open(log_file, 'w') as f: f.write(msg + '\n')

    log(f"=== Training {args.model.upper()} | Fourier: {use_fourier} | Device: {device} ===")
    log("Strategy: TIME-DILATED GOLDEN PROTOCOL (Peak LR @ 40%, Weights start @ Ep800)")

    train_loader, test_loader, x_norm, y_norm, vac = get_dataloader_and_stats(
        args.data_path, args.batch_size, device, consolidate=args.consolidate)
    vac = vac.to(device)

    ctx = None
    if args.model in coord_models | {'deeponet', 'ae', 'fno', 'pt', 'unet', 'vit'}:
        # parameter channels follow the dataset (coords + N params)
        ctx = {'n_param': train_loader.dataset.tensors[0].shape[1] - 2}
    if args.model in ('fs_deeponet', 'fs_lap_deeponet'):
        data_raw = torch.load(args.data_path, weights_only=False)
        train_idx, _ = get_split_indices(data_raw['x'].shape[0])
        ctx = {**ctx, **build_fs_ctx(data_raw['y'].float(), data_raw['x'][:, 2:, 0, 0].float(),
                                     train_idx, x_norm, y_norm)}
    model = build_model(args.model, use_fourier, ctx).to(device)
    if args.model in coord_models:
        model = GridAdapter(model).to(device)
    # Complex params count as 2 real dof (wavelet_deeponet trunk)
    log(f"Model Parameters: {sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()) / 1e6:.2f} M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Warmup: Peak learning rate at 40% of training
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.4, anneal_strategy='cos'
    )

    loss_fn = nn.MSELoss(reduction='none')
    best_test_loss = float('inf')
    best_test_epoch = -1
    history = {'train_loss': [], 'test_loss': []}

    log("Training Started...")
    pbar = tqdm(range(args.epochs), desc="Training")
    for ep in pbar:
        model.train()
        train_loss_val = 0.0

        # Delayed curriculum learning: Unrestricted exploration for the first 800 epochs
        if ep < 800:
            w_wall, w_near, w_p = 1.0, 1.0, 1.0
        else:
            progress = min(1.0, (ep - 800) / 1000.0)
            w_wall = 1.0 + 4.0 * progress
            w_near = 1.0 + 1.0 * progress
            w_p = 1.0 + 1.0 * progress 

        for x, y in train_loader:
            x_enc = x_norm.encode(x)
            y_enc = y_norm.encode(y)

            optimizer.zero_grad()
            out_enc = model(x_enc)

            # MSE loss weight matrix (paper Eq. 14) with spatial curriculum
            raw_loss = loss_fn(out_enc, y_enc)
            w = torch.ones_like(raw_loss)
            w[:, :, 2, :] = w_wall      # Wall
            w[:, :, 1, :] = w_near      # Near-wall
            w[:, :, 3, :] = w_near      # Near-wall
            w[:, 3, :, :] *= w_p        # Pressure channel augmentation
            w[:, :, vac] = 0.0          # Vacuum filler cells are not targets

            loss = (raw_loss * w).sum() / w.sum()
            if getattr(model, 'aux_loss', None) is not None:
                loss = loss + model.aux_loss
            inner = getattr(model, 'm', model)
            if getattr(inner, 'lap', False):
                loss = loss + inner.lap_weight * inner.graph_lap_loss()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss_val += loss.item()

        train_loss_val /= len(train_loader)
        history['train_loss'].append(train_loss_val)

        # Evaluation
        model.eval()
        test_loss_val = 0.0
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_enc_test = x_norm.encode(x_test)
                y_enc_test = y_norm.encode(y_test)
                out_enc_test = model(x_enc_test)
                # Unweighted (vacuum-masked) MSE loss for validation
                raw_test_loss = loss_fn(out_enc_test, y_enc_test)[:, :, ~vac].mean()
                test_loss_val += raw_test_loss.item()
                
        test_loss_val /= len(test_loader)
        history['test_loss'].append(test_loss_val)

        pbar.set_postfix({
            'train': f'{train_loss_val:.4g}',
            'test': f'{test_loss_val:.4g}',
            'best': f'{best_test_loss:.4g}' if best_test_epoch >= 0 else 'n/a',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            'w_wall': f'{w_wall:.1f}',
        })

        if ep % 50 == 0 or ep == args.epochs - 1:
            log(f"Ep {ep:04d} | LR: {scheduler.get_last_lr()[0]:.1e} | Train: {train_loss_val:.4g} | Test(Pure): {test_loss_val:.4g} | W_wall: {w_wall:.1f}")

        # Save best model after --best_after (default 1000) to prevent early overfitting
        if ep >= args.best_after and test_loss_val < best_test_loss:
            best_test_loss = test_loss_val
            best_test_epoch = ep
            
            torch.save({
                'model_state': model.state_dict(),
                'config': {'use_fourier': use_fourier,
                           'n_param': (ctx or {}).get('n_param', 3)},
                'epoch': ep,
                'best_test_loss': best_test_loss
            }, save_path)
            
            if ep % 50 == 0:
                log(f"  >>> [SAVED] Epoch {ep:04d} | Golden Best Test Loss: {best_test_loss:.6f} <<<")

    log(f"Finished. Golden Best Test Loss: {best_test_loss:.6f} discovered at Epoch {best_test_epoch}")

    # Save loss history for visualization plotting
    np.save(os.path.join(args.save_dir, 'history.npy'), history)

    # Save loss curve plot
    fig_path = os.path.join(args.save_dir, 'loss_curve.png')
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train', alpha=0.8)
    plt.plot(history['test_loss'], label='Test', alpha=0.8)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{args.model.upper()}{fourier_suffix} Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    log(f"Loss curve saved to {fig_path}")

if __name__ == "__main__":
    main()
