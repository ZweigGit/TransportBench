import torch
from torch.utils.data import Dataset, DataLoader

class AirfoilDataset(Dataset):
    def __init__(self, pt_path, mode='deeponet'):
        """
        Unified data loader
        Args:
            pt_path: Path to the preprocessed .pt file
            mode: 'deeponet' or 'fno' (corresponds to FNO/UNet/etc.)
        """
        super().__init__()
        self.mode = mode.lower()
        
        # 1. Load entire dataset into memory (assuming small dataset size, direct loading is efficient)
        print(f"Loading data from {pt_path}...")
        data_dict = torch.load(pt_path)
        
        self.geo_points = data_dict["geometry_points"] # (N, 337, 2)
        self.geo_mask   = data_dict["geometry_mask"]   # (N, 1, 128, 128)
        self.flow_label = data_dict["flow_label"]      # (N, 128, 128, 4)
        self.grid_coords= data_dict["grid_coords"]     # (N, 128, 128, 2)
        self.stats      = data_dict["stats"]           # Statistical information
        
        self.n_samples = self.flow_label.shape[0]
        self.train_min = None
        self.train_max = None
        print(f"Data loaded. Mode: {self.mode}, Samples: {self.n_samples}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 1. Prepare targets (Consistent across models, though shapes may be adjusted)
        # Original: (128, 128, 4) -> Adjusted to (4, 128, 128) for PyTorch compatibility
        target = self.flow_label[idx].permute(2, 0, 1) # (C, H, W)
        
        if self.mode == 'deeponet':
            # --- DeepONet Specific ---
            # Branch Input: Geometry points (337, 2) -> flattened to (674,)
            branch_in = self.geo_points[idx].reshape(-1)
            
            # Trunk Input: Grid coordinates (128, 128, 2) -> reshaped to (16384, 2)
            trunk_in = self.grid_coords[idx].reshape(-1, 2)
            
            # Target: (4, 128, 128) -> flattened to (16384, 4)
            # Note: DeepONet output is typically (Batch, P, 4), so targets must be flattened accordingly
            target_flat = self.flow_label[idx].reshape(-1, 4)
            
            return branch_in, trunk_in, target_flat

        elif self.mode in ['fno', 'unet']:
            # --- FNO / U-Net Specific ---
            # Input: Mask (1, H, W) + Coords (2, H, W) -> Concatenated -> (3, H, W)
            
            mask = self.geo_mask[idx] # (1, 128, 128)
            coords = self.grid_coords[idx].permute(2, 0, 1) # (128, 128, 2) -> (2, 128, 128)
            
            # Concatenate inputs: Channel 0=Mask, Channel 1=X, Channel 2=Y
            x_input = torch.cat([mask, coords], dim=0) # (3, 128, 128)
            
            return x_input, target

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def fit_train_normalizer(self, train_indices):
        """Re-normalize flow_label in place: un-normalize stored [-1,1] (full-data stats),
        then min-max normalize to [0,1] with per-channel stats from TRAIN split only."""
        self.train_min, self.train_max = {}, {}
        for i, var in enumerate(self.stats.keys()):
            gmin, gmax = self.stats[var]['min'], self.stats[var]['max']
            v = self.flow_label[train_indices][..., i]  # (T, H, W) stored in [-1,1]
            vphys = (v + 1.0) / 2.0 * (gmax - gmin) + gmin
            tmin, tmax = vphys.min(), vphys.max()
            self.train_min[var], self.train_max[var] = float(tmin), float(tmax)
            rng = tmax - tmin
            rng = rng if rng > 1e-12 else 1.0
            # affine from stored z in [-1,1] to train-only [0,1]: v2 = a*z + b
            a = (gmax - gmin) / (2.0 * rng)
            b = ((gmax + gmin) / 2.0 - tmin) / rng
            self.flow_label[..., i] = self.flow_label[..., i] * a + b
        print(f"Train-only min-max normalization applied: {len(train_indices)} train samples.")

    # Auxiliary function: Denormalization
    def denormalize(self, tensor, var_name):
        """Restore [0, 1] train-only normalized tensors back to their physical quantities"""
        if self.train_min is None:
            raise RuntimeError("fit_train_normalizer() must be called before denormalize()")
        _min, _max = self.train_min[var_name], self.train_max[var_name]
        return tensor * (_max - _min) + _min