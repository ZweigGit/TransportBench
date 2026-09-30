"""
WaveletDeepONet: DeepONet with a frozen 2D tensor-product wavelet MRA
dictionary as the trunk (WaveletMscaleNN2D-style fusion of wavelet bases and
MscaleNN, ported to the DeepONet setting).

Dictionary plan (one (factor_x, factor_y) pair per branch):
  - scaling branch   (phi_0, phi_0)
  - per dyadic scale s_j = 2^-j (j = 0..levels-1), the three wavelet
    families (phi_j, psi_j), (psi_j, phi_j), (psi_j, psi_j)
where phi and psi are the db10 MRA pair (PyWavelets cascade grids at
dyadic resolution 2^-10, linearly interpolated at evaluation time; for
dbN both functions are supported on [0, 2N - 1], i.e. [0, 19] for db10),
dilated/translated on the dyadic grid with the s^-1/2 factor
normalization (the 2D tensor normalization is the product of the two
factor normalizations).

The dictionary is parameter-free and frozen, so - like the precomputed_basis
philosophy of the reference project - the only trainable part is the branch
net (sensors -> per-atom coefficients); the DeepONet dot product plays the
role of the per-branch heads + linear fusion: each dictionary branch owns a
contiguous coefficient block, and prediction = sum over all atoms of
coefficient * feature.
"""

import math

import torch
import torch.nn as nn


_WAVEFUN = {}


def _wavefun(name, level=10):
    """(dec_len, grid, phi_vals, psi_vals) of the db-N MRA pair from the
    PyWavelets cascade algorithm, computed once per name and cached.

    pywt is imported lazily so this module's import stays free of the
    dependency for users of the other models.
    """
    if name not in _WAVEFUN:
        import pywt
        w = pywt.Wavelet(name)
        phi, psi, x = w.wavefun(level)
        _WAVEFUN[name] = (w.dec_len,
                          torch.tensor(x, dtype=torch.float32),
                          torch.tensor(phi, dtype=torch.float32),
                          torch.tensor(psi, dtype=torch.float32))
    return _WAVEFUN[name]


def _morlet_phi(u):
    """Gaussian 'scaling' factor of the Morlet plan (Morlet has no true
    scaling function; support +-4 by envelope truncation)."""
    return torch.exp(-u * u / 2)


def _morlet_psi(u):
    """Real Morlet wavelet cos(5u) exp(-u^2/2), truncated at |u| = 4."""
    return torch.cos(5 * u) * torch.exp(-u * u / 2)


class _AtomBank(nn.Module):
    """Frozen 1D dictionary of one db-N wavelet family:
    s^-1/2 * f((x - t_k) / s) with f interpolated on the cascade grid.

    kind: 'phi' or 'psi' (both supported on [0, dec_len - 1]).
    Translations sit on the grid t_k = k * stride * s (k stepping by
    `stride` over the full lattice), keeping every atom whose support
    [t_k, t_k + (dec_len - 1) s] reaches the domain. stride > 1 thins the
    translate frame uniformly at every scale (frame redundancy ~ W/stride).
    """
    def __init__(self, kind, scale, lo, hi, wavelet, stride=1):
        super().__init__()
        self.kind = kind
        self.closed = wavelet == 'morlet'
        if self.closed:
            w_lo, w_hi = -4.0, 4.0   # symmetric Gaussian-envelope support
        else:
            dec_len, grid, phi, psi = _wavefun(wavelet)
            w_lo, w_hi = 0.0, float(dec_len - 1)
            self.register_buffer('grid', grid)
            self.register_buffer('vals', phi if kind == 'phi' else psi)
        k_lo = math.ceil((lo - w_hi * scale) / scale)
        k_hi = math.floor((hi - w_lo * scale) / scale)
        t = torch.arange(k_lo, k_hi + 1, stride, dtype=torch.float32) * scale
        self.register_buffer('t', t)
        self.register_buffer('inv_s', torch.tensor(1.0 / scale))
        self.register_buffer('inv_sqrt_s', torch.tensor(scale ** -0.5))

    @property
    def n_atoms(self):
        return self.t.numel()

    def forward(self, x):
        """x: [N, 1] coordinates -> [N, n_atoms] features."""
        u = (x - self.t) * self.inv_s
        if self.closed:
            f = (_morlet_phi if self.kind == 'phi' else _morlet_psi)(u)
            f = f * (u.abs() <= 4.0).to(f.dtype)
            return f * self.inv_sqrt_s
        dx = self.grid[1] - self.grid[0]
        pos = (u - self.grid[0]) / dx
        inside = (pos >= 0) & (pos <= self.grid.numel() - 1)
        # ponytail: clamp keeps the gather in-bounds; outside-support
        # positions are zeroed by the mask below
        pos = pos.clamp(0, self.grid.numel() - 1.001)
        i0 = pos.floor().long()
        frac = pos - i0
        f = self.vals[i0] * (1 - frac) + self.vals[i0 + 1] * frac
        return f * inside.to(f.dtype) * self.inv_sqrt_s


class _WaveletTrunk(nn.Module):
    """Frozen 2D tensor-product dictionary: (phi_0, phi_0) first, then per
    scale the three families (phi_j, psi_j), (psi_j, phi_j), (psi_j, psi_j).
    Output rows follow the branch-then-kron order of the reference plan:
    branch blocks are concatenated, and within a block row (a-1)*ny + b
    holds fx_a * fy_b."""
    def __init__(self, domain, levels, wavelet, stride=1):
        super().__init__()
        lo, hi = domain
        s = [2.0 ** -j for j in range(levels)]
        phi = [_AtomBank('phi', sj, lo, hi, wavelet, stride) for sj in s]
        psi = [_AtomBank('psi', sj, lo, hi, wavelet, stride) for sj in s]
        self.plan = [(phi[0], phi[0])]
        for j in range(levels):
            self.plan += [(phi[j], psi[j]), (psi[j], phi[j]), (psi[j], psi[j])]
        self.banks = nn.ModuleList(b for pair in self.plan for b in pair)
        self.out_dim = sum(fx.n_atoms * fy.n_atoms for fx, fy in self.plan)

    def forward(self, x):
        """x: [N, 2] -> [N, out_dim] frozen features."""
        blocks = []
        for fx, fy in self.plan:
            Ax = fx(x[:, 0:1])  # [N, nx]
            Ay = fy(x[:, 1:2])  # [N, ny]
            nx, ny = fx.n_atoms, fy.n_atoms
            blocks.append((Ax[:, :, None] * Ay[:, None, :]).reshape(-1, nx * ny))
        return torch.cat(blocks, dim=-1)


class _FNN(nn.Module):
    """Fully-connected net: Linear -> LN -> Act -> ... -> Linear (LN like DeepONet2d)."""
    def __init__(self, dims, act):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(act)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WaveletDeepONet(nn.Module):
    """DeepONet whose trunk is the frozen tensor-product wavelet dictionary.

    Only the branch net is trainable: it maps sensor values to one
    coefficient per atom per output channel (num_outputs * out_dim blocks);
    the prediction is the dictionary expansion with those coefficients.

    Args:
        branch_dim:  Input dimension of the branch net (sensor values).
        trunk_dim:   Input dimension of the trunk net (coordinates, must be 2).
        hidden_dim:  Width of the branch net hidden layers.
        num_outputs: Output channels.
        depth:       Number of hidden layers in the branch net.
        levels:      Number of dyadic wavelet scales (s_j = 2^-j).
        activation:  Activation type ('GELU' or 'Tanh') for the branch net.
        wavelet:     PyWavelets db-N name for the MRA pair.
        domain:      Trunk coordinate domain (lo, hi) both axes, used only to
                     place the frozen translation grid.
        stride:      Translation-lattice stride (>1 thins the frame uniformly
                     at every scale; pure geometry, no data involved).
        lora_rank:   LoRA-style factorized branch head: hidden -> r -> out,
                     so the coefficient output lives in an r-dim subspace
                     learned from scratch. None = plain head.
    """
    def __init__(self, branch_dim=3, trunk_dim=2, hidden_dim=256,
                 num_outputs=4, depth=4, levels=5, activation='GELU',
                 wavelet='morlet', domain=(0.0, 1.0), stride=1, lora_rank=48):
        super().__init__()
        if trunk_dim != 2:
            raise ValueError("the tensor dictionary is 2D only")

        if activation == 'GELU':
            act = nn.GELU()
        elif activation == 'Tanh':
            act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.trunk_net = _WaveletTrunk(domain, levels, wavelet, stride)
        self.trunk_feat_dim = self.trunk_net.out_dim
        out_width = num_outputs * self.trunk_feat_dim

        if lora_rank is None:
            branch_dims = [branch_dim] + [hidden_dim] * depth + [out_width]
            self.branch_net = _FNN(branch_dims, act)
        else:
            # LoRA-style head: the FNN emits an r-dim code, a plain Linear
            # expands it to full coefficients (subspace = col of that Linear)
            self.branch_net = nn.Sequential(
                _FNN([branch_dim] + [hidden_dim] * depth + [lora_rank], act),
                nn.Linear(lora_rank, out_width))
        self.num_outputs = num_outputs

    def forward(self, x_branch, x_trunk):
        """Forward pass.

        Args:
            x_branch: [Batch, branch_dim]
            x_trunk:  [N_points, trunk_dim] or [Batch, N_points, trunk_dim]
                      (shared grid: the [Batch, 0] slice is evaluated)

        Returns:
            [Batch, N_points, num_outputs]
        """
        B = x_branch.shape[0]
        if x_trunk.dim() == 3:
            x_trunk = x_trunk[0]
        feat = self.trunk_net(x_trunk)                    # [N, n_atoms]
        assert feat.shape[-1] == self.trunk_feat_dim
        b = self.branch_net(x_branch)                     # [B, num_outputs * feat]
        b_out = b.view(B, self.num_outputs, self.trunk_feat_dim)
        return torch.einsum("bkh, nh -> bnk", b_out, feat)  # [B, N, num_outputs]


if __name__ == '__main__':
    model = WaveletDeepONet(hidden_dim=256, depth=4, levels=5,
                            wavelet='morlet', stride=1, lora_rank=48)
    n_params = sum(p.numel() for p in model.parameters())
    assert sum(p.numel() for p in model.trunk_net.parameters()) == 0, \
        "trunk dictionary must carry no parameters"
    xb = torch.randn(8, 3)
    xt = torch.rand(6528, 2)
    out = model(xb, xt)
    assert out.shape == (8, 6528, 4), out.shape
    out.sum().backward()  # grads reach the branch only, through frozen features
    assert model.branch_net[-1].weight.grad is not None
    # 3D trunk (shared grid) takes the [B, 0] slice
    xt3 = xt.unsqueeze(0).expand(8, -1, -1)
    assert torch.equal(model(xb, xt3), out)
    atoms = [fx.n_atoms * fy.n_atoms for fx, fy in model.trunk_net.plan]
    print(f"branches={len(atoms)} atoms={model.trunk_feat_dim} "
          f"(per branch {atoms}), lora_r=16, params={n_params/1e6:.2f}M, "
          f"out={tuple(out.shape)}")
