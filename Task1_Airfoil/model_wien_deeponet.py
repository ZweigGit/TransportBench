"""
WIEN-DeepONet for Task1 (airfoil flow), a fresh port of the WIEN-INR ideas
(Ni et al. 2026, "Multi-resolution enhancement for full-spectrum neural
representations", open-source code INC4Sci) to DeepONet operator learning.

Three WIEN ingredients, adapted for operator learning:
1. Train in wavelet-coefficient space: the model predicts DWT coefficients of
   the flow field instead of the field itself. Coefficients are normalized
   before learning and denormalized after (as in the source code), but with
   TRAIN-SET statistics per (level, channel, subband) -- the per-image
   statistics of the INR code are impossible at inference (no ground truth).
2. Scale separation: one shared GELU branch net + one SIREN trunk per DWT
   level (db4, J=3, symmetric on 128x128 -> 1 low-freq level of 22x22 with
   4 coefficients per point, and 3 high-freq levels of 22/37/67 with
   3 subbands x 4 channels = 12 coefficients per point). Each trunk is
   queried on the [-1,1] mgrid of its own coefficient grid shape
   (cf. INC4Sci code/Train/INR_Trainers.py get_pixels, line 1485).
3. Coarse-to-fine: every level's trunk receives, besides the query coords,
   the previous (coarser) level's coefficient prediction evaluated at the
   SAME query points. Applied at every level from low-freq to finest (the
   paper only enhances the finest level). All trunks + branch are trained
   jointly end-to-end.

Before the DWT, airfoil-interior cells are filled by a harmonic (Laplacian)
interpolation so the solid-body zeros do not inject artificial high
frequencies (protocol validated in wavelet_decompose_reconstruct.ipynb).

Parameter budget ~1M (v2, shared branch latent): trunk widths (35, 59, 151,
352) coarse->fine give a ~4x parameter ladder (8493 / 34073 / 129489 /
513600) at equal depth (4 SineLayers + linear output); one shared branch
latent [674,192,192,192,192,396]; total 1,002,851.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

# Coefficient grid shapes for db4/J=3/symmetric on 128x128, coarse -> fine.
LEVEL_SHAPES_128 = ((22, 22), (22, 22), (37, 37), (67, 67))
# Coefficients per query point per level: low = C, high = 3 subbands x C.
OUT_DIMS = (4, 12, 12, 12)


def get_mgrid(shape):
    """Flattened (x, y) grid in [-1, 1] matching a coefficient shape
    (source: INC4Sci base_models.get_mgrid)."""
    ty = torch.linspace(-1, 1, shape[0])
    tx = torch.linspace(-1, 1, shape[1])
    my, mx = torch.meshgrid(ty, tx, indexing="ij")
    return torch.stack([mx, my], dim=-1).reshape(-1, 2)  # (P, 2), row-major


class SineLayer(nn.Module):
    """SIREN layer with the original initialization scheme (fixed omega).

    is_first: U(-1/in, 1/in); otherwise U(-sqrt(6/in)/w0, sqrt(6/in)/w0) so
    activations stay bounded while gradients to the weights are boosted.

    Note: a learnable-omega variant (exp(log_omega) per layer, init 30.0)
    was tried and TRAINING DIVERGED at epoch ~575 (omega growth amplifies
    the sin gradient, a positive feedback loop); reverted to fixed omega.
    """

    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=True)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = math.sqrt(6.0 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class LevelTrunk(nn.Module):
    """Per-level trunk: `hidden_layers` SineLayers + linear output with the
    SIREN final init; or (ablation) the same stack with GELU activations and
    PyTorch-default init. Both variants have identical parameter counts."""

    def __init__(self, in_features, out_features, hidden_features, hidden_layers,
                 first_omega=30.0, hidden_omega=30.0, activation="siren"):
        super().__init__()
        if activation == "gelu":
            dims = [in_features] + [hidden_features] * hidden_layers + [out_features]
            layers = []
            for i in range(len(dims) - 2):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.GELU())
            layers.append(nn.Linear(dims[-2], dims[-1]))
        else:
            layers = [SineLayer(in_features, hidden_features, is_first=True,
                                omega_0=first_omega)]
            for _ in range(hidden_layers - 1):
                layers.append(SineLayer(hidden_features, hidden_features,
                                        is_first=False, omega_0=hidden_omega))
            final = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_features) / hidden_omega
                final.weight.uniform_(-bound, bound)
            layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WIENDeepONet(nn.Module):
    """Shared-branch multi-scale DeepONet in wavelet-coefficient space.

    forward(x_branch) -> per-level normalized coefficient predictions
        preds[l] of shape (B, P_l, o_l); low level channel layout is (C,),
        high levels are (K, C) flattened (K = LH/HL/HH subbands).
    predict_field(x_branch) -> denormalized IDWT reconstruction (B, 4, 128, 128).
    query_field(x_branch, phys_xy) -> field values at arbitrary physical
        coordinates (bilinear on the IDWT grid; the IDWT is a global operator,
        so a single-point query is answered by reconstructing the field and
        interpolating).

    The coarse-to-fine cascade evaluates trunk l on the grids of levels
    l..L-1: its own grid for the loss, and the finer grids as conditioning
    inputs for the deeper trunks (each finer trunk is fed the coarse level's
    prediction at that finer trunk's own query points).

    v2 changes (all trunks share ONE branch latent code; explicit affine
    mapping between physical Task1 coordinates and the [-1,1] trunk space;
    per-SineLayer learnable positive omega, init 30.0). Trunk widths default
    to (35, 59, 151, 352): with the shared branch (317,196 params) this keeps
    the ~4x parameter ladder (4.012/3.800/3.966) at 1,002,851 total.
    """

    def __init__(self, branch_dim=674, basis=32, trunk_widths=(35, 59, 151, 352),
                 trunk_layers=4, wavelet="db4", J=3, mode="symmetric",
                 grid_shapes=LEVEL_SHAPES_128, first_omega=30.0, hidden_omega=30.0,
                 branch_hidden=192, branch_layers=4, activation="siren", use_c2f=True,
                 xmin=-2.0, xmax=4.0, ymin=-3.0, ymax=3.0):
        super().__init__()
        assert len(trunk_widths) == J + 1
        self.J = J
        self.n_levels = J + 1
        self.basis = basis
        self.out_dims = OUT_DIMS  # fixed by C=4 and 3 DWT subbands
        self.wavelet_cfg = dict(wave=wavelet, J=J, mode=mode)
        self.grid_shapes = tuple(tuple(s) for s in grid_shapes)

        # --- physical <-> normalized coordinate mapping (Task1 domain) ---
        # x in [xmin, xmax], y in [ymin, ymax]; affine to [-1, 1]^2 per axis.
        self.register_buffer("phys_min", torch.tensor([xmin, ymin]))
        self.register_buffer("phys_span", torch.tensor(
            [xmax - xmin, ymax - ymin]))

        # --- trunks, one per level, 4x param ladder coarse->fine ---
        # use_c2f=False (ablation): coords only, no coarse-estimate input
        in_dims = [2]
        for l in range(1, self.n_levels):
            in_dims.append(2 + self.out_dims[l - 1] if use_c2f else 2)
        self.use_c2f = use_c2f
        self.trunks = nn.ModuleList([
            LevelTrunk(in_dims[l], self.out_dims[l] * basis, trunk_widths[l],
                       trunk_layers, first_omega, hidden_omega, activation)
            for l in range(self.n_levels)
        ])

        # --- shared branch (GELU MLP): ONE latent code for all trunks ---
        # o_max weight rows + o_max biases; the low-freq level uses the
        # first o_l rows (out_dims = (4, 12, 12, 12) all divide 12).
        o_max = max(self.out_dims)
        assert all(o_max % o == 0 for o in self.out_dims)
        branch_out = o_max * (self.basis + 1)  # 12 weight rows + 12 biases
        dims = [branch_dim] + [branch_hidden] * branch_layers + [branch_out]
        acts = [nn.GELU()] * (len(dims) - 2)
        self.branch_net = nn.Sequential(*[
            m for pair in zip(
                [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)],
                acts + [nn.Identity()]) for m in pair])

        # --- per-level query grids (constants, saved in state_dict) ---
        for l, sh in enumerate(self.grid_shapes):
            self.register_buffer(f"grid_{l}", get_mgrid(sh))

        # --- coefficient normalization buffers (train-set stats, filled by
        #     load_coeff_stats; restored from checkpoint at eval time) ---
        self.register_buffer("coef_mean", torch.zeros(self.n_levels, o_max))
        self.register_buffer("coef_maxx", torch.ones(self.n_levels, o_max))

        # lazy per-device IDWT (never a submodule: keep filters out of the
        # optimizer / state_dict)
        self._ifm_cache = {}

    # ------------------------------------------------------------------ #
    def load_coeff_stats(self, cache):
        """Copy train-set coefficient normalization stats into the buffers."""
        for l in range(self.n_levels):
            o = self.out_dims[l]
            self.coef_mean[l, :o] = cache["mean"][l]
            self.coef_maxx[l, :o] = cache["maxx"][l]

    # --- coordinate mapping ------------------------------------------ #
    def phys_to_norm(self, xy):
        """Physical coordinates (..., 2) -> normalized [-1, 1]^2."""
        return (xy - self.phys_min) / self.phys_span * 2.0 - 1.0

    def norm_to_phys(self, uv):
        """Normalized coordinates (..., 2) -> physical Task1 domain."""
        return (uv + 1.0) / 2.0 * self.phys_span + self.phys_min

    def _branch_slices(self, x_branch):
        """One shared latent: (B, o_max, basis) weights + (B, o_max) bias."""
        B = x_branch.shape[0]
        out = self.branch_net(x_branch)
        o_max = max(self.out_dims)
        w = out[:, :o_max * self.basis].view(B, 1, o_max, self.basis)
        b = out[:, o_max * self.basis:].view(B, 1, o_max)
        return [(w[:, :, :o], b[:, :, :o]) for o in self.out_dims]

    def forward(self, x_branch):
        """Per-level normalized coefficient predictions at each level's grid.

        x_branch: (B, 674). Returns list preds[l] of shape (B, P_l, o_l).
        """
        B = x_branch.shape[0]
        slices = self._branch_slices(x_branch)
        preds_own = [None] * self.n_levels
        if not self.use_c2f:  # ablation: each trunk sees only its own coords
            for l in range(self.n_levels):
                w, b = slices[l]
                o = self.out_dims[l]
                grid = getattr(self, f"grid_{l}")
                t = self.trunks[l](grid.unsqueeze(0).expand(B, -1, -1))
                preds_own[l] = (t.view(B, t.shape[1], o, self.basis) * w).sum(-1) + b
            return preds_own
        coarse = None  # level l-1 predictions at grids g = l..L-1
        for l in range(self.n_levels):
            w, b = slices[l]
            o = self.out_dims[l]
            new_coarse = []
            for g in range(l, self.n_levels):
                grid = getattr(self, f"grid_{g}")  # (P_g, 2)
                if l == 0:
                    inp = grid.unsqueeze(0).expand(B, -1, -1)
                else:
                    inp = torch.cat(
                        [grid.unsqueeze(0).expand(B, -1, -1), coarse[g - l + 1]],
                        dim=-1)
                t = self.trunks[l](inp)  # (B, P_g, o*basis)
                pred = (t.view(B, t.shape[1], o, self.basis) * w).sum(-1) + b
                new_coarse.append(pred)  # (B, P_g, o)
            preds_own[l] = new_coarse[0]
            coarse = new_coarse
        return preds_own

    def _denorm(self, pred, l):
        o = self.out_dims[l]
        return (pred * self.coef_maxx[l, :o].view(1, 1, o)
                + self.coef_mean[l, :o].view(1, 1, o))

    def _get_ifm(self, device):
        key = str(device)
        if key not in self._ifm_cache:
            self._ifm_cache[key] = DWTInverse(
                wave=self.wavelet_cfg["wave"], mode=self.wavelet_cfg["mode"]
            ).to(device)
        return self._ifm_cache[key]

    def predict_field(self, x_branch):
        """Reconstruct the flow field from predicted coefficients via IDWT.

        Returns (B, 4, 128, 128) in the same [0,1] train-normalized space
        the other Task1 models predict in.
        """
        B = x_branch.shape[0]
        preds = self.forward(x_branch)

        def to_band(l):
            d = self._denorm(preds[l], l)  # (B, P, o)
            h, wd = self.grid_shapes[l]
            if l == 0:  # (B, H, W, C) -> (B, C, H, W)
                return d.view(B, h, wd, 4).permute(0, 3, 1, 2)
            # (B, H, W, K, C) -> (B, C, K, H, W)
            return d.view(B, h, wd, 3, 4).permute(0, 4, 3, 1, 2)

        Yl = to_band(0)
        # model levels 1..J are coarse->fine; pytorch_wavelets Yh is fine->coarse
        Yh = [to_band(l) for l in range(self.J, 0, -1)]
        return self._get_ifm(x_branch.device)((Yl, Yh))

    def query_field(self, x_branch, phys_xy):
        """Flow-field values at arbitrary physical coordinates.

        phys_xy: (..., 2) tensor in the Task1 physical domain. The IDWT is a
        global operator (every coefficient contributes to every point), so a
        point query = full-field reconstruction + bilinear interpolation at
        the normalized location of phys_xy.

        Returns (..., 4) values in the [0,1] train-normalized space.
        """
        field = self.predict_field(x_branch)  # (B, 4, H, W), grid [-1,1]
        B = field.shape[0]
        uv = self.phys_to_norm(phys_xy)       # (..., 2)
        flat = uv.reshape(1, -1, 2).expand(B, -1, -1)  # (B, Q, 2)
        # grid_sample expects (x, y) in [-1, 1]; x = width index (last grid dim)
        grid = flat.unsqueeze(2)              # (B, Q, 1, 2)
        vals = F.grid_sample(field, grid, mode="bilinear", align_corners=True)
        vals = vals.squeeze(2)                # (B, 4, Q)
        if B == 1:                            # drop the batch (query semantics)
            return vals[0].T.reshape(*uv.shape[:-1], 4)
        return vals.permute(0, 2, 1).reshape(B, *uv.shape[:-1], 4)


# ---------------------------------------------------------------------- #
# Data pipeline: harmonic fill + DWT + train-set coefficient statistics  #
# ---------------------------------------------------------------------- #

def harmonic_fill(fields, fluid, n_iter=500, tol=1e-9):
    """Fill non-fluid cells by solving Laplace's equation with the fluid
    values as Dirichlet boundary (Jacobi iterations, batched).

    fields: (N, C, H, W); fluid: (N, 1, H, W) bool, True = keep.
    """
    f = fields.clone()
    body = ~fluid
    if not body.any():
        return f
    m = fluid.to(f.dtype)
    mean_v = (f * m).sum(dim=(2, 3), keepdim=True) / m.sum(dim=(2, 3), keepdim=True)
    f = torch.where(fluid, f, mean_v)  # flat init at the fluid mean
    k = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]],
                     device=f.device).view(1, 1, 3, 3)
    kc = k.expand(f.shape[1], 1, 3, 3)  # per-channel (grouped) kernel
    ones = torch.ones_like(m)           # (N, 1, H, W), broadcasts over C
    for _ in range(n_iter):
        nb_sum = F.conv2d(f, kc, padding=1, groups=f.shape[1])
        nb_cnt = F.conv2d(ones, k, padding=1)
        new = torch.where(fluid, f, nb_sum / nb_cnt)
        res = (new - f).abs().max()
        f = new
        if res < tol:
            break
    return f


def build_or_load_coef_cache(data_path, cache_path, wavelet="db4", J=3,
                             mode="symmetric", device=None):
    """Harmonic-fill + DWT the train-normalized fields, compute per-(level,
    channel[, subband]) mean / max-abs statistics over the TRAIN split, and
    cache the normalized flattened targets.

    Must reproduce the train.py pipeline exactly (same seed-42 split, same
    train-only [0,1] normalization), so the cache is valid for training.
    """
    if cache_path is not None:
        import os
        if os.path.exists(cache_path):
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            if (cache["wavelet_cfg"] == dict(wave=wavelet, J=J, mode=mode)
                    and cache["data_path"] == os.path.abspath(data_path)):
                print(f"Loaded coefficient cache: {cache_path}")
                return cache
            print("Coefficient cache metadata mismatch -- rebuilding.")

    from torch.utils.data import random_split
    from data_loader import AirfoilDataset

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = AirfoilDataset(data_path, mode="fno")
    train_size = int(0.8 * len(ds))
    sub_train, _ = random_split(ds, [train_size, len(ds) - train_size],
                                generator=torch.Generator().manual_seed(42))
    ds.fit_train_normalizer(sub_train.indices)  # same [0,1] space as training
    train_idx = torch.tensor(sub_train.indices)

    fields = ds.flow_label.permute(0, 3, 1, 2).to(device)  # (N, 4, 128, 128)
    fluid = ds.geo_mask.to(device).bool()                  # (N, 1, 128, 128)
    print("Harmonic-filling airfoil interior (all samples)...")
    filled = harmonic_fill(fields, fluid)
    dwt = DWTForward(J=J, wave=wavelet, mode=mode).to(device)
    with torch.no_grad():
        Yl, Yh = dwt(filled)

    # Levels coarse -> fine: Yl, Yh[J-1](coarsest detail), ..., Yh[0](finest)
    bands = [Yl] + [Yh[J - 1 - i] for i in range(J)]
    N = fields.shape[0]
    targets, means, maxxs, shapes = [], [], [], []
    for l, band in enumerate(bands):
        if band.dim() == 4:  # Yl (N, C, H, W)
            flat = band.permute(0, 2, 3, 1).reshape(N, -1, 4)
        else:                # Yh (N, C, K, H, W)
            flat = band.permute(0, 3, 4, 2, 1).reshape(N, -1, 12)
        tr = flat[train_idx.to(flat.device)]
        mean = tr.mean(dim=(0, 1))
        maxx = (tr - mean).abs().amax(dim=(0, 1)).clamp_min(1e-8)
        targets.append(((flat - mean) / maxx).cpu())  # (N, P, o), ~[-1, 1]
        means.append(mean.cpu())
        maxxs.append(maxx.cpu())
        shapes.append(tuple(band.shape[-2:]))
        del band, flat
    del fields, filled, Yl, Yh

    cache = {
        "wavelet_cfg": dict(wave=wavelet, J=J, mode=mode),
        "data_path": os.path.abspath(data_path) if cache_path else data_path,
        "shapes": shapes,
        "targets": targets,
        "mean": means,
        "maxx": maxxs,
        "train_indices": train_idx,
    }
    if cache_path is not None:
        import os
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
        print(f"Saved coefficient cache: {cache_path}")
    return cache


# ---------------------------------------------------------------------- #
# Self-check: parameter budget, fill/DWT round-trip, forward shapes      #
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. Parameter budget: ~4x ladder between adjacent trunks, ~1M total
    model = WIENDeepONet()
    trunk_p = [sum(p.numel() for p in t.parameters()) for t in model.trunks]
    branch_p = sum(p.numel() for p in model.branch_net.parameters())
    print("trunk params (coarse->fine):", trunk_p)
    print("trunk ratios:",
          [round(trunk_p[i + 1] / trunk_p[i], 3) for i in range(3)])
    print(f"branch params: {branch_p}, total: {sum(trunk_p) + branch_p:,}")
    assert trunk_p == [8493, 34073, 129489, 513600]
    assert branch_p == 317196 and sum(trunk_p) + branch_p == 1002851

    # 2. Harmonic fill + DWT/IDWT round-trip on a synthetic case
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f = torch.randn(2, 4, 128, 128, device=dev)
    fluid = torch.ones(2, 1, 128, 128, dtype=torch.bool, device=dev)
    fluid[:, 0, 60:64, 30:40] = False  # fake small body
    filled = harmonic_fill(f, fluid)
    dwt = DWTForward(J=3, wave="db4", mode="symmetric").to(dev)
    ifm = DWTInverse(wave="db4", mode="symmetric").to(dev)
    with torch.no_grad():
        Yl, Yh = dwt(filled)
        rec = ifm((Yl, Yh))
    err = (rec - filled).abs().max().item()
    print(f"fill+DWT/IDWT round-trip max err: {err:.2e}")
    assert err < 1e-5

    # 3. Forward / predict_field shapes (CPU and GPU path)
    xb = torch.randn(2, 674, device=dev)
    model = model.to(dev)
    preds = model(xb)
    print("pred shapes:", [tuple(p.shape) for p in preds])
    assert [tuple(p.shape) for p in preds] == [
        (2, 484, 4), (2, 484, 12), (2, 1369, 12), (2, 4489, 12)]
    field = model.predict_field(xb)
    print("field shape:", tuple(field.shape), "| device:", field.device)
    assert field.shape == (2, 4, 128, 128) and field.device.type == dev[:4]

    # 4. One optimization step works (loss = mean of level MSEs); omega grads
    tgt = [torch.randn_like(p) for p in preds]
    loss = sum(F.mse_loss(preds[l], tgt[l]) for l in range(4)) / 4.0
    loss.backward()
    assert all(p.grad is not None for t in model.trunks for p in t.parameters())
    print(f"backward ok, loss={loss.item():.3f}")

    # 5. Coordinate mapping round-trip + grid correspondence
    xy = torch.tensor([[2.0, 2.0], [-2.0, -3.0], [4.0, 3.0], [0.0, 0.0]],
                      device=dev)
    uv = model.phys_to_norm(xy)
    assert uv.min() >= -1.0 and uv.max() <= 1.0
    assert torch.allclose(model.norm_to_phys(uv), xy, atol=1e-6)
    # normalized physical grid == the trunk query grid (square uniform domain)
    base = torch.linspace(-2, 4, 128, device=dev), torch.linspace(-3, 3, 128, device=dev)
    gy, gx = torch.meshgrid(base[1], base[0], indexing="ij")
    phys_grid = torch.stack([gx, gy], dim=-1).reshape(-1, 2)
    l3 = model.grid_3.reshape(67, 67, 2)
    # L3 grid steps must match the normalized physical grid steps (67 ~ 64+3 pad)
    print("phys step (norm units):", (model.phys_span / 126).tolist())
    print("grid_3 x-step:", (l3[0, 1, 0] - l3[0, 0, 0]).abs().item(),
          "| grid_3 y-step:", (l3[1, 0, 1] - l3[0, 0, 1]).abs().item())
    print("coord mapping round-trip OK; grid correspondence approximate (pad)")

    # 6. Point-query interface: values match an independent grid_sample
    model.eval()
    with torch.no_grad():
        q = model.query_field(xb[:1], xy)          # (4, 4)
        field = model.predict_field(xb[:1])        # (1, 4, 128, 128)
    uvq = model.phys_to_norm(xy)                   # (4, 2)
    ref = F.grid_sample(field, uvq.reshape(1, 4, 1, 2), mode="bilinear",
                        align_corners=True)[0, :, :, 0].T          # (Q, 4)
    assert torch.allclose(q, ref, atol=1e-6), (q - ref).abs().max()
    # corners of the physical domain map to the corners of [-1, 1]^2
    assert torch.allclose(model.phys_to_norm(torch.tensor(
        [[-2., -3.], [4., 3.]], device=dev)),
        torch.tensor([[-1., -1.], [1., 1.]], device=dev), atol=1e-6)
    print("query_field matches grid_sample reference:", tuple(q.shape))

    # 7. Ablation variants: GELU trunks (identical param count), no-c2f
    m_gelu = WIENDeepONet(activation="gelu")
    tp = [sum(p.numel() for p in t.parameters()) for t in m_gelu.trunks]
    total_gelu = sum(tp) + sum(p.numel() for p in m_gelu.branch_net.parameters())
    assert total_gelu == 1002851
    m_nc = WIENDeepONet(use_c2f=False)
    tp_nc = [sum(p.numel() for p in t.parameters()) for t in m_nc.trunks]
    total_nc = sum(tp_nc) + sum(p.numel() for p in m_nc.branch_net.parameters())
    print(f"gelu total: {total_gelu:,} | noc2f trunks: {tp_nc}, total: {total_nc:,}")
    assert tp_nc == [8493, 33837, 127677, 509376] and total_nc == 996579
    xb2 = torch.randn(2, 674)
    expect = [(2, 484, 4), (2, 484, 12), (2, 1369, 12), (2, 4489, 12)]
    for name, m in [("gelu", m_gelu), ("noc2f", m_nc)]:
        ps = m(xb2)
        assert [tuple(p.shape) for p in ps] == expect, name
        f = m.predict_field(xb2)
        assert f.shape == (2, 4, 128, 128), name
    print("ALL SELF-CHECKS PASSED")
