"""
MR_DeepONet: DeepONet whose trunk is a purely frozen multi-resolution
tensor-product wavelet atom dictionary (no trainable trunk parameters).

Dictionary (one (factor_x, factor_y) pair per group, all atoms frozen):
  - scaling group     (phi_0, phi_0)
  - per dyadic scale s_j = 2^-j (j = 0..levels-1), the three wavelet
    families (phi_j, psi_j), (psi_j, phi_j), (psi_j, psi_j)
where phi and psi are the db-N MRA pair (PyWavelets cascade grids at
dyadic resolution 2^-10, linearly interpolated at evaluation time),
dilated/translated on the dyadic grid with the s^-1/2 normalization.

The trunk feature of a point is the concatenated values of all atoms at
that point: feature dim = atom count, zero parameters. The branch net
maps sensors to one coefficient per atom per output channel; the
DeepONet dot product is the fusion.
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
        """x: [N, 1] coordinates -> [N, n_atoms] atom values."""
        u = (x - self.t) * self.inv_s
        dx = self.grid[1] - self.grid[0]
        pos = (u - self.grid[0]) / dx
        inside = (pos >= 0) & (pos <= self.grid.numel() - 1)
        # ponytail: clamp keeps the gather in-bounds (integer bound is exact
        # in fp32; outside-support positions are zeroed by the mask below)
        pos = pos.clamp(0, self.grid.numel() - 2)
        i0 = pos.floor().long()
        frac = pos - i0
        f = self.vals[i0] * (1 - frac) + self.vals[i0 + 1] * frac
        return f * inside.to(f.dtype) * self.inv_sqrt_s


class _MRTrunk(nn.Module):
    """Frozen 2D tensor-product wavelet dictionary: (phi_0, phi_0) first,
    then per scale the three families (phi_j, psi_j), (psi_j, phi_j),
    (psi_j, psi_j). Output feature dim = total atom count (all buffers,
    zero trainable parameters)."""
    def __init__(self, domain, levels, wavelet, stride):
        super().__init__()
        lo, hi = domain
        s = [2.0 ** -j for j in range(levels)]
        phi = [_AtomBank('phi', sj, lo, hi, wavelet, stride) for sj in s]
        psi = [_AtomBank('psi', sj, lo, hi, wavelet, stride) for sj in s]
        self.plan = [(phi[0], phi[0])]
        for j in range(levels):
            self.plan += [(phi[j], psi[j]), (psi[j], phi[j]), (psi[j], psi[j])]
        # registers the banks' buffers for .to(device); plan borrows refs
        self.banks = nn.ModuleList(b for pair in self.plan for b in pair)
        self.out_dim = sum(fx.n_atoms * fy.n_atoms for fx, fy in self.plan)

    def forward(self, x):
        """x: [N, 2] -> [N, out_dim] concatenated atom values."""
        feats = []
        for fx, fy in self.plan:
            ax = fx(x[:, 0:1])                        # [N, nx]
            ay = fy(x[:, 1:2])                        # [N, ny]
            feats.append((ax[:, :, None] * ay[:, None, :]).reshape(x.shape[0], -1))
        return torch.cat(feats, dim=-1)


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


class MR_DeepONet(nn.Module):
    """DeepONet whose trunk is the purely frozen multi-resolution wavelet
    dictionary: the only trainable part is the branch net, mapping sensor
    values to one coefficient per atom per output channel.

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
    """
    def __init__(self, branch_dim=3, trunk_dim=2, hidden_dim=385,
                 num_outputs=4, depth=4, activation='GELU',
                 wavelet='db4', domain=(0.0, 1.0), stride=2, levels=4):
        super().__init__()
        if trunk_dim != 2:
            raise ValueError("the tensor dictionary is 2D only")

        if activation == 'GELU':
            act = nn.GELU()
        elif activation == 'Tanh':
            act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.trunk_net = _MRTrunk(domain, levels, wavelet, stride)
        self.trunk_feat_dim = self.trunk_net.out_dim
        out_width = num_outputs * self.trunk_feat_dim
        self.branch_net = _FNN([branch_dim] + [hidden_dim] * depth + [out_width], act)
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
        b = self.branch_net(x_branch)                     # [B, num_outputs * n_atoms]
        b_out = b.view(B, self.num_outputs, self.trunk_feat_dim)
        return torch.einsum("bkh, nh -> bnk", b_out, feat)  # [B, N, num_outputs]


if __name__ == '__main__':
    model = MR_DeepONet(hidden_dim=385, depth=4, levels=4, wavelet='db4', stride=2)
    n_params = sum(p.numel() for p in model.parameters())
    trunk_params = sum(p.numel() for p in model.trunk_net.parameters())
    assert trunk_params == 0, "trunk must be parameter-free"
    assert model.trunk_feat_dim == sum(fx.n_atoms * fy.n_atoms
                                       for fx, fy in model.trunk_net.plan)
    xb = torch.randn(8, 3)
    xt = torch.rand(6528, 2)
    out = model(xb, xt)
    assert out.shape == (8, 6528, 4), out.shape
    out.sum().backward()
    assert model.branch_net.net[-1].weight.grad is not None
    # 3D trunk (shared grid) takes the [B, 0] slice
    xt3 = xt.unsqueeze(0).expand(8, -1, -1)
    assert torch.equal(model(xb, xt3), out)
    print(f"groups={len(model.trunk_net.plan)} frozen_atoms={model.trunk_feat_dim} "
          f"trunk_params={trunk_params}, params={n_params/1e6:.2f}M, "
          f"out={tuple(out.shape)}")
