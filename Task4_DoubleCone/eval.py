import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from data_loader import MinMaxNormalizer, get_split_indices, vacuum_mask
from train import build_model, GridAdapter  # single source of model configs, keeps eval in sync with train

def get_args():
    parser = argparse.ArgumentParser(description="Evaluation for Task 4: Double Cone Flow")
    parser.add_argument('--model', type=str, required=True,
                        choices=['deeponet', 'fno', 'unet', 'vit', 'ae', 'pt', 'hyperdeeponet', 'mscale_deeponet', 'hyper_mscale_deeponet', 'c_hyperdeeponet', 'c_hyper_mscale_deeponet', 'fusion_deeponet', 'wavelet_deeponet'],
                        help='Choose the model to evaluate')
    parser.add_argument('--no_fourier', action='store_true',
                        help='Model does not use Fourier encoding')
    parser.add_argument('--data_path', type=str, default='../data/double_cone_dataset_with_physics.pt',
                        help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (default: output/<model><_fourier|_nofourier>/best_model.pth)')
    parser.add_argument('--sample_idx', type=int, default=50,
                        help='Global sample index to visualize (Default: 50 for Benchmark Case)')
    parser.add_argument('--test_sample_idx', type=int, default=0,
                        help='Index into the TEST set to visualize (default: 0 = first test sample)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save (default: output/<model><_fourier|_nofourier>)')
    return parser.parse_args()

def main():
    args = get_args()
    use_fourier = not args.no_fourier
    fourier_suffix = "_fourier" if use_fourier else "_nofourier"
    # Coordinate-based DeepONet variants, wrapped behind the grid image interface
    coord_models = {'hyperdeeponet', 'mscale_deeponet', 'hyper_mscale_deeponet', 'c_hyperdeeponet', 'c_hyper_mscale_deeponet', 'fusion_deeponet', 'wavelet_deeponet'}
    # Coord variants without a Fourier option get no suffix (only hyperdeeponet supports it)
    fourierless = coord_models - {'hyperdeeponet'}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_name = args.model + ('' if args.model in fourierless else fourier_suffix)

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
    vac = vacuum_mask(y_data).to(device)      # domain-filler cells, not targets
    
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
    
    model = build_model(args.model, checkpoint_fourier).to(device)
    if args.model in coord_models:
        model = GridAdapter(model).to(device)
    state = checkpoint.get('model_state', checkpoint)
    if args.model in coord_models and not any(k.startswith('m.') for k in state):
        # legacy checkpoint saved before the grid adapter: remap raw keys
        state = {f'm.{k}': v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # Evaluate metrics on the test set (same split as training), in normalized [0,1] space
    x_test = x_data[test_idx].to(device)
    y_test_phys = y_data[test_idx].to(device)
    y_test_log = y_test_phys.clone()
    y_test_log[:, 3] = torch.log10(y_test_phys[:, 3] + 1e-6)  # pressure in log10 before normalization

    total_mae, total_mse = 0.0, 0.0
    total_l2_error, total_p_l2_error = 0.0, 0.0

    print(f"\nEvaluating on {len(test_idx)} test samples...")
    with torch.no_grad():
        bs = 8
        for s in range(0, x_test.shape[0], bs):
            x_enc_batch = x_norm.encode(x_test[s:s+bs])
            pred_enc = model(x_enc_batch)
            y_enc = y_norm.encode(y_test_log[s:s+bs])  # target in normalized [0,1] space

            # All metrics on the fluid domain only: zero out vacuum filler cells
            err = pred_enc - y_enc
            err[:, :, vac] = 0.0
            tgt = y_enc.clone()
            tgt[:, :, vac] = 0.0

            total_mae += err.abs().sum().item()
            total_mse += (err ** 2).sum().item()

            l2_err = torch.norm(err.flatten(1), dim=1) / \
                     (torch.norm(tgt.flatten(1), dim=1) + 1e-8)
            p_l2_err = torch.norm(err[:, 3:4].flatten(1), dim=1) / \
                       (torch.norm(tgt[:, 3:4].flatten(1), dim=1) + 1e-8)
            total_l2_error += l2_err.sum().item()
            total_p_l2_error += p_l2_err.sum().item()

    n_valid_el = len(test_idx) * 4 * (~vac).sum().item()
    final_mae = total_mae / n_valid_el
    final_mse = total_mse / n_valid_el
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
        f.write(f"Model       : {args.model.upper()} ({'coordinate-based' if args.model in fourierless else 'fourier' if checkpoint_fourier else 'nofourier'})\n")
        f.write(f"Checkpoint  : {ckpt_path}\n")
        f.write(f"Metric space: normalized [0,1] (p=log10), vacuum cells excluded\n")
        f.write(f"MAE         : {final_mae:.4g}\n")
        f.write(f"MSE         : {final_mse:.4g}\n")
        f.write(f"RL2E        : {final_rel_l2:.4g}\n")
        f.write(f"RL2E (p,log10): {final_p_rel_l2:.4g}\n")
    print(f"Saved eval results to: {eval_file}")

    # ---- Visualization: benchmark sample (global index, unchanged) + test sample ----
    def to_np(t): return t.squeeze(0).cpu().numpy()

    def visualize_sample(actual_idx, tag, out_name):
        """Predict and plot one sample (global index) with GT / pred / error fields."""
        x_input = x_data[actual_idx].unsqueeze(0).to(device)
        y_true_phys = y_data[actual_idx].unsqueeze(0).to(device)

        with torch.no_grad():
            x_encoded = x_norm.encode(x_input)
            pred_encoded = model(x_encoded)
            y_pred_log = y_norm.decode(pred_encoded)

        y_pred_phys = y_pred_log.clone()
        y_pred_phys[:, 3, :, :] = torch.pow(10, y_pred_log[:, 3, :, :])
        error = torch.abs(y_true_phys - y_pred_phys)

        x_np, y_true, y_pred, err = to_np(x_input), to_np(y_true_phys), to_np(y_pred_phys), to_np(error)
        Grid_X, Grid_Y = x_np[0], x_np[1]
        # blank vacuum filler cells (outside the fluid domain) in all panels
        vac_np = vac.cpu().numpy()
        valid2d = ~vac_np
        y_true[:, vac_np] = np.nan
        y_pred[:, vac_np] = np.nan
        err[:, vac_np] = np.nan

        fourier_text = ("Coordinate-based" if args.model in fourierless
                        else "With Fourier" if checkpoint_fourier else "No Fourier")
        title_str = f"{args.model.upper()} | {tag} (Global Idx: {actual_idx}) | {fourier_text}"
        plot_configs = [{'name': 'Velocity u (m/s)', 'idx': 1, 'cmap': 'jet'},
                        {'name': 'Pressure p (Pa)', 'idx': 3, 'cmap': 'magma'}]

        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 3)
        plt.suptitle(title_str, fontsize=16)

        for row_idx, cfg in enumerate(plot_configs):
            var_idx, cmap = cfg['idx'], cfg['cmap']
            gt, pred, e = y_true[var_idx], y_pred[var_idx], err[var_idx]

            l2_err = np.linalg.norm(e[valid2d]) / (np.linalg.norm(gt[valid2d]) + 1e-8)
            # fields carry NaN in vacuum cells; nan-unaware min/percentile would
            # poison the clim and matplotlib would fall back to (-0.1, 0.1)
            vmin = min(np.nanmin(gt), np.nanmin(pred))
            vmax = max(np.nanpercentile(gt, 99), np.nanpercentile(pred, 99))

            ax1 = fig.add_subplot(gs[row_idx, 0])
            im1 = ax1.pcolormesh(Grid_X, Grid_Y, gt, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
            ax1.set_title(f"GT {cfg['name']}"); ax1.axis('equal'); ax1.axis('off'); plt.colorbar(im1, ax=ax1)

            ax2 = fig.add_subplot(gs[row_idx, 1])
            im2 = ax2.pcolormesh(Grid_X, Grid_Y, pred, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
            ax2.set_title(f"Pred {cfg['name']}"); ax2.axis('equal'); ax2.axis('off'); plt.colorbar(im2, ax=ax2)

            ax3 = fig.add_subplot(gs[row_idx, 2])
            im3 = ax3.pcolormesh(Grid_X, Grid_Y, e, cmap='inferno', shading='gouraud')
            ax3.set_title(f"Error (Rel L2={l2_err:.1%})"); ax3.axis('equal'); ax3.axis('off'); plt.colorbar(im3, ax=ax3)

        # Wall pressure curves
        wall_idx = 2
        wall_x = Grid_X[wall_idx, :]
        ax_wall = fig.add_subplot(gs[2, :])
        ax_wall.plot(wall_x, y_true[3][wall_idx, :], 'k-', lw=2.5, label='CFD Ground Truth')
        ax_wall.plot(wall_x, y_pred[3][wall_idx, :], 'r--', lw=2.5, label=f'{args.model.upper()} Pred ({fourier_text})')
        ax_wall.set_title(f"Near-Wall Pressure Distribution ({tag})")
        ax_wall.set_xlabel("X (m)"); ax_wall.set_ylabel("Pressure (Pa)"); ax_wall.legend(); ax_wall.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_img = os.path.join(args.output_dir, out_name)
        plt.savefig(output_img, dpi=150)
        plt.close(fig)
        print(f"Saved visualization to: {output_img}")

    # Benchmark Case: global index 50, unchanged (train sample, literature benchmark)
    print(f"Extracting Benchmark Case (Global Idx {args.sample_idx}).")
    visualize_sample(args.sample_idx, "Benchmark Case", "inference_benchmark.png")

    # Test sample: drawn from the test set (same seeded split as training, no leakage)
    if not (0 <= args.test_sample_idx < len(test_idx)):
        raise ValueError(f"--test_sample_idx {args.test_sample_idx} out of range: "
                         f"test set has {len(test_idx)} samples (0..{len(test_idx)-1})")
    test_global_idx = test_idx[args.test_sample_idx].item()
    print(f"Extracting test sample (test_idx[{args.test_sample_idx}] -> Global Idx {test_global_idx}).")
    visualize_sample(test_global_idx, "Test Sample", "inference_test.png")

if __name__ == "__main__":
    main()