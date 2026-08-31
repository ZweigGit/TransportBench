"""
Data loader for Task 4: Double Cone Flow
Compatible with original data_utils.py logic
"""
import torch
from torch.utils.data import TensorDataset, DataLoader

class MinMaxNormalizer:
    """Min-max normalization to [0,1], stats from train split only"""
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val
        self.eps = 1e-6

    def encode(self, x):
        return (x - self.min) / (self.max - self.min + self.eps)

    def decode(self, x):
        return x * (self.max - self.min + self.eps) + self.min

    def to(self, device):
        self.min = self.min.to(device)
        self.max = self.max.to(device)
        return self

    def state_dict(self):
        return {'min': self.min, 'max': self.max}

    def load_state_dict(self, state_dict):
        self.min = state_dict['min']
        self.max = state_dict['max']

def get_split_indices(n_total):
    """
    Train/test split with fixed seed 42.
    Shared by train and eval to guarantee the same test set.
    Returns: (train_idx, test_idx)
    """
    if n_total == 51:
        n_train = 45
        n_test = 6
        print(">>> Detected Full 51 Cases! Using 45 Train / 6 Test split. <<<")
    elif n_total == 32:
        n_train = 28
        n_test = 4
        print(">>> Warning: Only detected 32 Cases! Using 28 Train / 4 Test split. <<<")
    else:
        n_test = max(1, int(n_total * 0.1))
        n_train = n_total - n_test
        print(f">>> Detected {n_total} Cases. Using {n_train} Train / {n_test} Test. <<<")

    g_cpu = torch.Generator()
    g_cpu.manual_seed(42)
    indices = torch.randperm(n_total, generator=g_cpu)

    return indices[:n_train], indices[n_train:]

def get_dataloader_and_stats(data_path, batch_size, device):
    """
    Load dataset and compute normalization statistics
    Returns: train_loader, test_loader, x_norm, y_norm
    
    This function replicates the exact logic from original data_utils.py
    """
    print(f"Loading dataset from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    x_data = data['x'].float()
    y_data = data['y'].float()
    
    print("Applying Standard Log10-transform to Pressure channel (Index 3)...")
    y_data[:, 3, :, :] = torch.log10(y_data[:, 3, :, :] + 1e-6)

    n_total = x_data.shape[0]
    train_idx, test_idx = get_split_indices(n_total)

    print(f"Dataset Split -> Total: {n_total} | Train: {len(train_idx)} | Test: {len(test_idx)}")

    # Min-max statistics computed on TRAIN split only (no test leakage)
    print("Computing min-max statistics on train split...")
    x_train_raw, y_train_raw = x_data[train_idx], y_data[train_idx]
    x_min = torch.amin(x_train_raw, dim=(0, 2, 3), keepdim=True)
    x_max = torch.amax(x_train_raw, dim=(0, 2, 3), keepdim=True)
    y_min = torch.amin(y_train_raw, dim=(0, 2, 3), keepdim=True)
    y_max = torch.amax(y_train_raw, dim=(0, 2, 3), keepdim=True)

    x_train, y_train = x_data[train_idx].to(device), y_data[train_idx].to(device)
    x_test, y_test = x_data[test_idx].to(device), y_data[test_idx].to(device)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    x_norm = MinMaxNormalizer(x_min, x_max).to(device)
    y_norm = MinMaxNormalizer(y_min, y_max).to(device)

    return train_loader, test_loader, x_norm, y_norm
