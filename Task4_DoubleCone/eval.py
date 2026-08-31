import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from data_loader import MinMaxNormalizer, get_split_indices
from model_deeponet import DeepONet2d
from model_fno import FNO2d
from model_unet import FluidUNet
from model_vit import VisionTransformer
from model_ae import AutoEncoder2d
from model_pt import PointTransformer

def get_args():
    parser = argparse.ArgumentParser(description="Evaluation for Task 4: Double Cone Flow")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['deeponet', 'fno', 'unet', 'vit', 'ae', 'pt'], 
                        help='Choose the model to evaluate')
    parser.add_argument('--no_fourier', action='store_true',
                        help='Model does not use Fourier encoding')
    parser.add_argument('--data_path', type=str, default='../data/double_cone_dataset_with_physics.pt',
                        help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (default: output/<model><_fourier|_nofourier>/best_model.pth)')
    parser.add_argument('--sample_idx', type=int, default=50,
                        help='Global sample index to visualize (Default: 50 for Benchmark Case)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save (default: output/<model><_fourier|_nofourier>)')
    return parser.parse_args()

def main():
    args = get_args()
    use_fourier = not args.no_fourier
    fourier_suffix = "_fourier" if use_fourier else "_nofourier"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    run_name = args.model + fourier_suffix

    # Per-model output subdirectory, aligned with Task I convention
    if args.output_dir is None:
        args.output_dir = os.path.join('output', run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Locate checkpoint
    if args.checkpoint is None:
        args.checkpoint = os.path.join('output', run_name, 'best_model.pth')
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found for {args.model}! Checked: {ckpt_path}")
            
    print(f"Loading checkpoint from: {ckpt_path}")

    # Load data and statistics
    print(f"Loading data from {args.data_path}...")
    data_full = torch.load(args.data_path, weights_only=False)
    x_data = data_full['x'].float()
    y_data = data_full['y'].float()
    
    y_data_log = y_data.clone()
    y_data_log[:, 3, :, :] = torch.log10(y_data[:, 3, :, :] + 1e-6)
    
    # Same seeded split as training; stats computed on TRAIN split only (no test leakage)
    train_idx, test_idx = get_split_indices(x_data.shape[0])

    x_train = x_data[train_idx]
    y_train = y_data_log[train_idx]
    x_min = torch.amin(x_train, dim=(0, 2, 3), keepdim=True).to(device)
    x_max = torch.amax(x_train, dim=(0, 2, 3), keepdim=True).to(device)
    y_min = torch.amin(y_train, dim=(0, 2, 3), keepdim=True).to(device)
    y_max = torch.amax(y_train, dim=(0, 2, 3), keepdim=True).to(device)

    x_norm = MinMaxNormalizer(min_val=x_min, max_val=x_max)
    y_norm = MinMaxNormalizer(min_val=y_min, max_val=y_max)

    # Load model and weights
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get('config', {})
    checkpoint_fourier = cfg.get('use_fourier', use_fourier)
    
    if args.model == 'fno':
        model = FNO2d(modes1=8, modes2=64, width=64, in_channels=5, out_channels=4, use_fourier=checkpoint_fourier)
    elif args.model == 'unet':
        model = FluidUNet(in_channels=5, out_channels=4, features=64, use_fourier=checkpoint_fourier)
    elif args.model == 'vit':
        model = VisionTransformer(in_channels=5, out_channels=4, embed_dim=512, depth=10, use_fourier=checkpoint_fourier)
    elif args.model == 'ae':
        model = AutoEncoder2d(in_channels=5, out_channels=4, features=128, use_fourier=checkpoint_fourier)
    elif args.model == 'deeponet':
        model = DeepONet2d(in_channels=5, out_channels=4, basis_size=256, use_fourier=checkpoint_fourier)
    elif args.model == 'pt':
        model = PointTransformer(in_channels=5, out_channels=4, latent_dim=512, num_latents=1024, depth=10, use_fourier=checkpoint_fourier)
    
    model = model.to(device)
    model.load_state_dict(checkpoint.get('model_state', checkpoint))
    model.eval()

    # Evaluate metrics on the test set (same split as training), in normalized [0,1] space
    x_test = x_data[test_idx].to(device)
    y_test_phys = y_data[test_idx].to(device)
    y_test_log = y_test_phys.clone()
    y_test_log[:, 3] = torch.log10(y_test_phys[:, 3] + 1e-6)  # pressure in log10 before normalization

    criterion_mae = nn.L1Loss(reduction='sum')
    criterion_mse = nn.MSELoss(reduction='sum')
    total_mae, total_mse = 0.0, 0.0
    total_l2_error, total_p_l2_error = 0.0, 0.0

    print(f"\nEvaluating on {len(test_idx)} test samples...")
    with torch.no_grad():
        bs = 8
        for s in range(0, x_test.shape[0], bs):
            pred_enc = model(x_norm.encode(x_test[s:s+bs]))
            y_enc = y_norm.encode(y_test_log[s:s+bs])  # target in normalized [0,1] space

            total_mae += criterion_mae(pred_enc, y_enc).item()
            total_mse += criterion_mse(pred_enc, y_enc).item()

            l2_err = torch.norm((pred_enc - y_enc).flatten(1), dim=1) / \
                     (torch.norm(y_enc.flatten(1), dim=1) + 1e-8)
            p_l2_err = torch.norm((pred_enc[:, 3:4] - y_enc[:, 3:4]).flatten(1), dim=1) / \
                       (torch.norm(y_enc[:, 3:4].flatten(1), dim=1) + 1e-8)
            total_l2_error += l2_err.sum().item()
            total_p_l2_error += p_l2_err.sum().item()

    final_mae = total_mae / y_test_phys.numel()
    final_mse = total_mse / y_test_phys.numel()
    final_rel_l2 = total_l2_error / len(test_idx)
    final_p_rel_l2 = total_p_l2_error / len(test_idx)

    print("-" * 50)
    print(f"Final Results for {args.model.upper()} (normalized [0,1] space):")
    print(f"Mean Absolute Error (MAE) : {final_mae:.4g}")
    print(f"Mean Squared Error (MSE)  : {final_mse:.4g}")
    print(f"Relative L2 Error (RL2E)  : {final_rel_l2:.4g}")
    print(f"RL2E (pressure, log10)    : {final_p_rel_l2:.4g}")
    print("-" * 50)

    # Save eval results
    eval_file = os.path.join(args.output_dir, 'eval_results.txt')
    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write(f"Model       : {args.model.upper()} ({'fourier' if use_fourier else 'nofourier'})\n")
        f.write(f"Checkpoint  : {ckpt_path}\n")
        f.write(f"Metric space: normalized [0,1] (p=log10)\n")
        f.write(f"MAE         : {final_mae:.4g}\n")
        f.write(f"MSE         : {final_mse:.4g}\n")
        f.write(f"RL2E        : {final_rel_l2:.4g}\n")
        f.write(f"RL2E (p,log10): {final_p_rel_l2:.4g}\n")
    print(f"Saved eval results to: {eval_file}")

    # Extract benchmark sample
    actual_idx = args.sample_idx
    print(f"Successfully extracted Benchmark Case (Global Idx {actual_idx}).")

    x_input = x_data[actual_idx].unsqueeze(0).to(device)
    y_true_phys = y_data[actual_idx].unsqueeze(0).to(device)

    with torch.no_grad():
        x_encoded = x_norm.encode(x_input)
        pred_encoded = model(x_encoded)
        y_pred_log = y_norm.decode(pred_encoded)

    y_pred_phys = y_pred_log.clone()
    y_pred_phys[:, 3, :, :] = torch.pow(10, y_pred_log[:, 3, :, :])
    error = torch.abs(y_true_phys - y_pred_phys)

    # Visualization
    def to_np(t): return t.squeeze(0).cpu().numpy()
    
    x_np, y_true, y_pred, err = to_np(x_input), to_np(y_true_phys), to_np(y_pred_phys), to_np(error)
    Grid_X, Grid_Y = x_np[0], x_np[1]
    
    fourier_text = "With Fourier" if checkpoint_fourier else "No Fourier"
    title_str = f"{args.model.upper()} | Benchmark Case (Global Idx: {actual_idx}) | {fourier_text}"
    plot_configs = [{'name': 'Velocity u (m/s)', 'idx': 1, 'cmap': 'jet'},
                    {'name': 'Pressure p (Pa)', 'idx': 3, 'cmap': 'magma'}]

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3)
    plt.suptitle(title_str, fontsize=16)

    for row_idx, cfg in enumerate(plot_configs):
        var_idx, cmap = cfg['idx'], cfg['cmap']
        gt, pred, e = y_true[var_idx], y_pred[var_idx], err[var_idx]
        
        l2_err = np.linalg.norm(e) / (np.linalg.norm(gt) + 1e-8)
        vmin, vmax = min(gt.min(), pred.min()), max(np.percentile(gt, 99), np.percentile(pred, 99))

        ax1 = fig.add_subplot(gs[row_idx, 0])
        im1 = ax1.pcolormesh(Grid_X, Grid_Y, gt, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        ax1.set_title(f"GT {cfg['name']}"); ax1.axis('equal'); ax1.axis('off'); plt.colorbar(im1, ax=ax1)

        ax2 = fig.add_subplot(gs[row_idx, 1])
        im2 = ax2.pcolormesh(Grid_X, Grid_Y, pred, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        ax2.set_title(f"Pred {cfg['name']}"); ax2.axis('equal'); ax2.axis('off'); plt.colorbar(im2, ax=ax2)

        ax3 = fig.add_subplot(gs[row_idx, 2])
        im3 = ax3.pcolormesh(Grid_X, Grid_Y, e, cmap='inferno', shading='gouraud')
        ax3.set_title(f"Error (Rel L2={l2_err:.1%})"); ax3.axis('equal'); ax3.axis('off'); plt.colorbar(im3, ax=ax3)

    # Extract wall curves
    wall_idx = 2
    wall_x = Grid_X[wall_idx, :]
    ax_wall = fig.add_subplot(gs[2, :])
    ax_wall.plot(wall_x, y_true[3][wall_idx, :], 'k-', lw=2.5, label='CFD Ground Truth')
    ax_wall.plot(wall_x, y_pred[3][wall_idx, :], 'r--', lw=2.5, label=f'{args.model.upper()} Pred ({fourier_text})')
    ax_wall.set_title(f"Near-Wall Pressure Distribution (Benchmark Case)")
    ax_wall.set_xlabel("X (m)"); ax_wall.set_ylabel("Pressure (Pa)"); ax_wall.legend(); ax_wall.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_img = os.path.join(args.output_dir, "inference_benchmark.png")
    plt.savefig(output_img, dpi=150)
    print(f"Saved visualization to: {output_img}")

if __name__ == "__main__":
    main()