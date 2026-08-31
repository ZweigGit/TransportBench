import torch
import torch.nn as nn
import time
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader import get_dataloader_and_stats

from model_ae import AutoEncoder2d
from model_deeponet import DeepONet2d
from model_fno import FNO2d
from model_pt import PointTransformer
from model_unet import FluidUNet
from model_vit import VisionTransformer
from model_hyperdeeponet import HyperDeepONet
from model_mscale_deeponet import MscaleDeepONet
from model_hyper_mscale_deeponet import HyperMscaleDeepONet

def get_args():
    parser = argparse.ArgumentParser(description="Universal Golden Protocol Training Script")
    parser.add_argument('--model', type=str, required=True, choices=['ae', 'deeponet', 'fno', 'pt', 'unet', 'vit', 'hyperdeeponet', 'mscale_deeponet', 'hyper_mscale_deeponet'])
    parser.add_argument('--data_path', type=str, default='../data/double_cone_dataset_with_physics.pt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--no_fourier', action='store_true', help='Disable Fourier Encoding')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    return parser.parse_args()

def build_model(model_name, use_fourier):
    if model_name == 'ae': return AutoEncoder2d(in_channels=5, out_channels=4, features=128, use_fourier=use_fourier)
    elif model_name == 'deeponet': return DeepONet2d(in_channels=5, out_channels=4, basis_size=256, use_fourier=use_fourier)
    elif model_name == 'fno': return FNO2d(modes1=8, modes2=64, width=64, in_channels=5, out_channels=4, use_fourier=use_fourier)
    elif model_name == 'pt': return PointTransformer(in_channels=5, out_channels=4, latent_dim=512, num_latents=1024, depth=10, use_fourier=use_fourier)
    elif model_name == 'unet': return FluidUNet(in_channels=5, out_channels=4, features=64, use_fourier=use_fourier)
    elif model_name == 'vit': return VisionTransformer(in_channels=5, out_channels=4, embed_dim=512, depth=10, use_fourier=use_fourier)
    # Coordinate-based (branch = Mach/Temp/Re, trunk = x/y grid coords)
    elif model_name == 'hyperdeeponet': return HyperDeepONet(branch_hidden=128, trunk_hidden=256, trunk_depth=4, branch_depth=4, basis_size=256)
    elif model_name == 'mscale_deeponet': return MscaleDeepONet(branch_hidden=2048, branch_depth=4, trunk_hidden=512, trunk_depth=4, basis_size=2048)
    elif model_name == 'hyper_mscale_deeponet': return HyperMscaleDeepONet(hidden_dim=128, depth=4, trunk_hidden=128, trunk_depth=3, basis_size=256)

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fourier = not args.no_fourier
    # Coordinate-based DeepONet variants take (branch, trunk) instead of a grid image
    coord_models = {'hyperdeeponet', 'mscale_deeponet', 'hyper_mscale_deeponet'}
    data_mode = 'coord' if args.model in coord_models else 'grid'
    fourier_suffix = "" if data_mode == 'coord' else ("_fourier" if use_fourier else "_nofourier")

    # Per-model output subdirectory, aligned with Task I convention
    args.save_dir = os.path.join('output', args.model + fourier_suffix)
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'best_model.pth')
    log_file = os.path.join(args.save_dir, 'train.log')

    def log(msg):
        print(msg)
        with open(log_file, 'a') as f: f.write(msg + '\n')

    log(f"=== Training {args.model.upper()} | Fourier: {use_fourier} | Device: {device} ===")
    log("Strategy: TIME-DILATED GOLDEN PROTOCOL (Peak LR @ 40%, Weights start @ Ep800)")

    train_loader, test_loader, x_norm, y_norm = get_dataloader_and_stats(args.data_path, args.batch_size, device)
    
    model = build_model(args.model, use_fourier).to(device)
    log(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Warmup: Peak learning rate at 40% of training
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.4, anneal_strategy='cos'
    )

    loss_fn = nn.L1Loss(reduction='none')
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
            if data_mode == 'coord':
                # Branch = flow params (ch 2-4: Mach/Temp/Re, constant per sample),
                # trunk = shared x/y grid coords (ch 0-1, identical across samples)
                branch = x_enc[:, 2:5, 0, 0]                                  # [B, 3]
                trunk = x_enc[0, 0:2].permute(1, 2, 0).reshape(-1, 2)          # [6528, 2]
                target = y_enc.permute(0, 2, 3, 1).reshape(x_enc.shape[0], -1, 4)  # [B, 6528, 4]
                out_enc = model(branch, trunk)
                # Plain L1: grid-position curriculum weights don't apply in flat space
                loss = loss_fn(out_enc, target).mean()
            else:
                out_enc = model(x_enc)

                # L1 Loss weight matrix
                raw_loss = loss_fn(out_enc, y_enc)
                w = torch.ones_like(raw_loss)
                w[:, :, 2, :] = w_wall      # Wall
                w[:, :, 1, :] = w_near      # Near-wall
                w[:, :, 3, :] = w_near      # Near-wall
                w[:, 3, :, :] *= w_p        # Pressure channel augmentation

                loss = (raw_loss * w).mean()
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
                if data_mode == 'coord':
                    branch = x_enc_test[:, 2:5, 0, 0]
                    trunk = x_enc_test[0, 0:2].permute(1, 2, 0).reshape(-1, 2)
                    target = y_enc_test.permute(0, 2, 3, 1).reshape(x_enc_test.shape[0], -1, 4)
                    out_enc_test = model(branch, trunk)
                    raw_test_loss = loss_fn(out_enc_test, target)
                else:
                    out_enc_test = model(x_enc_test)
                    # Unweighted L1 loss for validation
                    raw_test_loss = loss_fn(out_enc_test, y_enc_test)
                test_loss_val += raw_test_loss.mean().item()
                
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

        # Save best model after epoch 1000 to prevent early overfitting
        if ep >= 1000 and test_loss_val < best_test_loss:
            best_test_loss = test_loss_val
            best_test_epoch = ep
            
            torch.save({
                'model_state': model.state_dict(),
                'config': {'use_fourier': use_fourier},
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
