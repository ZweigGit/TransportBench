import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="TransportBench - Task II: Cylinder Flow")
    parser.add_argument('--model', type=str, required=True,
                        choices=['deeponet', 'fno', 'unet', 'vit', 'ae', 'pt', 'mscale_deeponet', 'hyperdeeponet'],
                        help='Choose the baseline model')
    parser.add_argument('--epochs', type=int, default=2500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data_path', type=str, default='./data/cylinder_full_2400.pt', help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    return parser.parse_args()

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Fix random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Starting Task II Training | Model: {args.model.upper()} | Device: {device}")

    # Save to output/<model>/, aligned with Task1 structure
    args.save_dir = os.path.join('output', args.model)
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"best_model.pth")

    # Determine data loading mode: grid-based vs coordinate-based
    data_mode = 'grid' if args.model in ['fno', 'unet', 'vit', 'ae'] else 'deeponet'

    dataset = CylinderDataset(args.data_path, mode=data_mode)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    if args.model == 'fno':
        model = FNO2d(modes1=12, modes2=12, width=32, in_channels=4, out_channels=4)
    elif args.model == 'unet':
        model = FluidUNet(in_channels=4, out_channels=4, base_dim=19)
    elif args.model == 'vit':
        model = VisionTransformer(img_size=(128, 192), patch_size=8, in_chans=4, out_chans=4, embed_dim=144, depth=4)
    elif args.model == 'ae':
        model = Autoencoder(in_channels=4, out_channels=4, base_width=36)
    elif args.model == 'deeponet':
        model = BoltzmannDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=280, num_outputs=4, depth=5)
    elif args.model == 'pt':
        model = PointTransformer(in_dim=4, out_dim=4, embed_dim=144, depth=4)
    elif args.model == 'mscale_deeponet':
        model = MscaleDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=280, num_outputs=4,
                               scales=[1, 2, 4, 8, 16], depth=4, activation='GELU')
    elif args.model == 'hyperdeeponet':
        model = HyperDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=68, num_outputs=4,
                              trunk_depth=3, branch_depth=3, activation='GELU')

    model = model.to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Paper: no scheduler for Tasks I-III
    criterion = nn.MSELoss()

    best_test_loss = float('inf')
    history = {'train_loss': [], 'test_loss': []}

    print("Training Started...")
    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        # --- Train Phase ---
        model.train()
        train_loss_acc = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            if data_mode == 'grid':
                x, y = batch[0].to(device), batch[1].to(device)
                pred = model(x)
            else:
                x_branch, x_trunk, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                x_branch = x_branch[:, :2]  # (Kn, Ma)
                x_trunk = x_trunk[0]  # All samples share the same grid
                pred = model(x_branch, x_trunk)

            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss_acc += loss.item()

        avg_train_loss = train_loss_acc / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- Test Phase ---
        model.eval()
        test_loss_acc = 0.0
        with torch.no_grad():
            for batch in test_loader:
                if data_mode == 'grid':
                    x, y = batch[0].to(device), batch[1].to(device)
                    pred = model(x)
                else:
                    x_branch, x_trunk, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    x_branch = x_branch[:, :2]
                    x_trunk = x_trunk[0]
                    pred = model(x_branch, x_trunk)

                loss = criterion(pred, y)
                test_loss_acc += loss.item()

        avg_test_loss = test_loss_acc / len(test_loader)
        history['test_loss'].append(avg_test_loss)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), save_path)
            saved_flag = True
        else:
            saved_flag = False

        pbar.set_postfix({
            'train': f'{avg_train_loss:.5f}',
            'test': f'{avg_test_loss:.5f}',
            'best': f'{best_test_loss:.5f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
        })
        if saved_flag and (epoch + 1) % 50 == 0:
            tqdm.write(f"  [BEST SAVED @ epoch {epoch+1}]")

    print(f"Training Complete! Best Test Loss: {best_test_loss:.5f}. Model saved to {save_path}")

    # Save loss history
    np.save(os.path.join(args.save_dir, 'history.npy'), history)

    # Save loss curve plot
    fig_path = os.path.join(args.save_dir, 'loss_curve.png')
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train', alpha=0.8)
    plt.plot(history['test_loss'], label='Test', alpha=0.8)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{args.model.upper()} Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to {fig_path}")

if __name__ == "__main__":
    main()
