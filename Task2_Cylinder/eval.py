import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from model_deeponet import BoltzmannDeepONet
from model_fno import FNO2d
from model_unet import FluidUNet
from model_vit import VisionTransformer
from model_ae import Autoencoder
from model_pt import PointTransformer
from model_mscale_deeponet import MscaleDeepONet
from model_hyperdeeponet import HyperDeepONet
from data_loader import CylinderDataset

def get_args():
    parser = argparse.ArgumentParser(description="Evaluation Script for Task II: Cylinder Flow")
    parser.add_argument('--model', type=str, required=True,
                        choices=['deeponet', 'fno', 'unet', 'vit', 'ae', 'pt', 'mscale_deeponet', 'hyperdeeponet'],
                        help='Choose the baseline model to evaluate')
    parser.add_argument('--data_path', type=str, default='./data/cylinder_full_2400.pt', help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to weights (default: output/<model>/best_model.pth)')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save visualizations (default: output/<model>)')
    return parser.parse_args()

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Evaluation | Model: {args.model.upper()} | Device: {device}")

    # Create output directory
    if args.output_dir is None:
        args.output_dir = os.path.join('output', args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load test dataset
    data_mode = 'grid' if args.model in ['fno', 'unet', 'vit', 'ae'] else 'deeponet'
    dataset = CylinderDataset(args.data_path, mode=data_mode)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # 2. Initialize model
    if args.model == 'fno':
        model = FNO2d(modes1=12, modes2=12, width=32, in_channels=4, out_channels=4)
    elif args.model == 'unet':
        model = FluidUNet(in_channels=4, out_channels=4, base_dim=19)
    elif args.model == 'vit':
        model = VisionTransformer(img_size=(128, 192), patch_size=8, in_chans=4, out_chans=4, embed_dim=144, depth=4)
    elif args.model == 'ae':
        model = Autoencoder(in_channels=4, out_channels=4, base_width=36)
    elif args.model == 'deeponet':
        model = BoltzmannDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=280, num_outputs=4)
    elif args.model == 'pt':
        model = PointTransformer(in_dim=4, out_dim=4, embed_dim=144, depth=4)
    elif args.model == 'mscale_deeponet':
        model = MscaleDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=192, num_outputs=4,
                               scales=[1, 2, 4, 8, 16], depth=4, activation='GELU')
    elif args.model == 'hyperdeeponet':
        model = HyperDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=78, num_outputs=4,
                              trunk_depth=3, branch_depth=3, activation='GELU')

    model = model.to(device)

    # 3. Load model weights
    if args.checkpoint is None:
        args.checkpoint = os.path.join('output', args.model, 'best_model.pth')
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded weights from {ckpt_path}")

    # 4. Calculate evaluation metrics and collect samples for visualization
    total_mae = 0.0
    total_mse = 0.0
    total_l2_error = 0.0
    criterion_mae = nn.L1Loss(reduction='sum')
    criterion_mse = nn.MSELoss(reduction='sum')

    GRID_H, GRID_W = 128, 192
    samples_to_plot = []
    var_names = ['rho', 'u', 'v', 'T']

    # Load full dataset for cylinder mask (visualization only)
    full_dataset = CylinderDataset(args.data_path, mode='grid')

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if data_mode == 'grid':
                x, y = batch[0].to(device), batch[1].to(device)
                pred = model(x)
            else:
                x_branch, x_trunk, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                x_branch = x_branch[:, :2]
                x_trunk = x_trunk[0]
                pred = model(x_branch, x_trunk)

            total_mae += criterion_mae(pred, y).item()
            total_mse += criterion_mse(pred, y).item()
            l2_err = torch.norm(pred - y, p=2) / (torch.norm(y, p=2) + 1e-8)
            total_l2_error += l2_err.item()

            # Collect samples for visualization
            if i < args.num_samples:
                if data_mode == 'grid':
                    gt_sample = y[0].cpu().numpy()          # [4, H, W]
                    pred_sample = pred[0].cpu().numpy()     # [4, H, W]
                else:
                    gt_sample = y[0].view(GRID_H, GRID_W, 4).permute(2, 0, 1).cpu().numpy()
                    pred_sample = pred[0].view(GRID_H, GRID_W, 4).permute(2, 0, 1).cpu().numpy()

                # Cylinder mask for visualization
                mask_2d = full_dataset.mask[i][0].cpu().numpy()  # [H, W]

                samples_to_plot.append({
                    'gt': gt_sample,
                    'pred': pred_sample,
                    'mask': mask_2d,
                    'index': i
                })

    num_samples = len(test_loader.dataset)
    num_elements = num_samples * np.prod(y.shape[1:])

    final_mae = total_mae / num_elements
    final_mse = total_mse / num_elements
    final_rel_l2 = total_l2_error / num_samples

    print("-" * 50)
    print(f"Final Results for {args.model.upper()}:")
    print(f"Mean Absolute Error (MAE) : {final_mae:.5f}")
    print(f"Mean Squared Error (MSE)  : {final_mse:.5f}")
    print(f"Relative L2 Error         : {final_rel_l2:.5f}")
    print("-" * 50)

    # 5. Generate visualizations for multiple samples and variables
    print(f"\nGenerating visualizations for {len(samples_to_plot)} samples...")

    for sample_data in samples_to_plot:
        sample_idx = sample_data['index']
        gt = sample_data['gt']       # [4, H, W]
        pred = sample_data['pred']   # [4, H, W]
        mask = sample_data['mask']   # [H, W]

        fig, axes = plt.subplots(4, 3, figsize=(15, 16))

        for var_idx, var_name in enumerate(var_names):
            gt_var = gt[var_idx]
            pred_var = pred[var_idx]
            err_var = np.abs(gt_var - pred_var)

            # Apply mask for visualization
            gt_var_masked = np.ma.masked_where(mask < 0.5, gt_var)
            pred_var_masked = np.ma.masked_where(mask < 0.5, pred_var)
            err_var_masked = np.ma.masked_where(mask < 0.5, err_var)

            vmin = min(gt_var_masked.min(), pred_var_masked.min())
            vmax = max(gt_var_masked.max(), pred_var_masked.max())

            # Ground Truth
            im0 = axes[var_idx, 0].imshow(gt_var_masked, cmap='jet', vmin=vmin, vmax=vmax)
            axes[var_idx, 0].set_title(f"GT - {var_name}", fontsize=12)
            axes[var_idx, 0].axis('off')
            plt.colorbar(im0, ax=axes[var_idx, 0], fraction=0.046, pad=0.04)

            # Prediction
            im1 = axes[var_idx, 1].imshow(pred_var_masked, cmap='jet', vmin=vmin, vmax=vmax)
            axes[var_idx, 1].set_title(f"Pred - {var_name}", fontsize=12)
            axes[var_idx, 1].axis('off')
            plt.colorbar(im1, ax=axes[var_idx, 1], fraction=0.046, pad=0.04)

            # Error
            im2 = axes[var_idx, 2].imshow(err_var_masked, cmap='hot')
            axes[var_idx, 2].set_title(f"Error - {var_name}", fontsize=12)
            axes[var_idx, 2].axis('off')
            plt.colorbar(im2, ax=axes[var_idx, 2], fraction=0.046, pad=0.04)

        plt.suptitle(f"{args.model.upper()} - Sample {sample_idx}", fontsize=16, y=0.995)
        plt.tight_layout()

        save_fig_path = os.path.join(args.output_dir, f"sample_{sample_idx}.png")
        plt.savefig(save_fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_fig_path}")

    print(f"\nAll visualizations saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
