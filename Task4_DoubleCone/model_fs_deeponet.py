"""Freestream-normalized DeepONet for Task 4.

Diagnosis that motivates this model (see analysis/task4_shockgate/): the
dominant test error of the vanilla DeepONet is low-wavenumber (93-98% of
error energy below k=16), not shock sharpness, and the rho channel fails by
amplitude miscalibration (predictions overshoot low-density samples by up to
10x). The inlet columns of every sample are exact prescribed freestream
constants (identical across all 85 inlet cells, no DSMC noise).

Idea: reparameterize the field by its exact inflow state. Internally predict
the nondimensional shape (log10(rho/rho_inf), u/u_inf, v/u_inf,
log10(p/p_inf)) whose cross-sample variability is roughly half of the raw
fields, plus a tiny head regressing (rho_inf, u_inf, p_inf) from the branch
params. The benchmark-space output is reconstructed exactly, so the training
loss, curriculum and eval protocol are unchanged.

Optional graph-Laplacian smoothness penalty on the branch coefficients over
the kNN graph of the 45 train params (targets the low-k interpolation error):
fs_lap_deeponet = same model with lap=True.
"""
import torch
import torch.nn as nn


class _FNN(nn.Module):
    """Linear -> LN -> GELU stack (same style as mscale/fusion variants)."""
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FS_DeepONet(nn.Module):
    """DeepONet over freestream-normalized fields with a param->freestream head.

    All data-derived constants (freestream table, internal shape stats,
    benchmark y stats) are registered buffers so a checkpoint carries
    everything eval needs; at build time in eval.py they are placeholders
    overwritten by load_state_dict.

    Args:
        branch_dim / trunk_dim: input dims (params / coords).
        hidden: width of the shape branch & trunk FNNs.
        depth: number of hidden layers in both FNNs.
        basis_size: dot-product width.
        num_outputs: output channels (4: rho, u, v, p).
        fs_hidden: width of the freestream head.
        lap: add graph-Laplacian regularizer on branch coefficients.
        lap_weight: weight of the Laplacian term.
        ctx: dict of numpy/tensor buffers computed from the train split:
            param_table [S,3] normalized params (row per dataset sample),
            log_fs_z    [S,3] standardized log10 freestreams,
            fs_mu/fs_sd [3] standardization stats,
            tmin/tmax   [4] internal shape-channel min/max (fluid cells),
            ymin/yrange [4] benchmark min/max in the p-log10 space.
            Defaults (None) build neutral placeholders for eval-time reloads.
    """
    def __init__(self, branch_dim=3, trunk_dim=2, hidden=300, depth=5,
                 basis_size=128, num_outputs=4, fs_hidden=64,
                 lap=False, lap_weight=0.05, aux_weight=1.0, ctx=None):
        super().__init__()
        self.num_outputs = num_outputs
        self.lap = lap
        self.lap_weight = lap_weight
        self.aux_weight = aux_weight
        # reconstruction is exact only if these buffers hold the train stats;
        # placeholder zeros keep eval-time construction shape-compatible
        z = lambda *s: torch.zeros(*s)
        ctx = ctx or {}
        self.register_buffer('param_table', ctx.get('param_table', z(51, branch_dim)))
        self.register_buffer('log_fs_z', ctx.get('log_fs_z', z(51, 3)))
        self.register_buffer('fs_mu', ctx.get('fs_mu', z(3)))
        self.register_buffer('fs_sd', ctx.get('fs_sd', torch.ones(3)))
        self.register_buffer('tmin', ctx.get('tmin', z(num_outputs)))
        self.register_buffer('tmax', ctx.get('tmax', torch.ones(num_outputs)))
        self.register_buffer('ymin', ctx.get('ymin', z(num_outputs)))
        self.register_buffer('yrange', ctx.get('yrange', torch.ones(num_outputs)))

        # shape nets: standard branch/trunk dot product in internal space
        self.trunk_net = _FNN([trunk_dim] + [hidden] * depth + [basis_size])
        self.branch_net = _FNN([branch_dim] + [hidden] * depth
                               + [num_outputs * basis_size])
        self.shape_bias = nn.Parameter(torch.zeros(num_outputs))

        # freestream head on standardized log10(rho_inf, u_inf, p_inf);
        # zero-init so it starts at the dataset mean freestream
        self.fs_head = _FNN([branch_dim, fs_hidden, fs_hidden, 3])
        nn.init.zeros_(self.fs_head.net[-1].weight)
        nn.init.zeros_(self.fs_head.net[-1].bias)

        # kNN graph over train params for the Laplacian penalty (built lazily
        # on first call so it follows the buffer device)
        self._lap_edges = None
        self.aux_loss = None

    def _build_lap_edges(self, k=6):
        p = self.param_table
        d = torch.cdist(p, p)
        d.fill_diagonal_(float('inf'))
        idx = d.topk(k, largest=False).indices              # [S, k]
        h = d[d.isfinite()].median()
        w = torch.exp(-(d.gather(1, idx) / h.clamp_min(1e-6)) ** 2)
        return idx, w

    def graph_lap_loss(self):
        """Smoothness of branch coefficients over the train-param kNN graph."""
        if self._lap_edges is None or self._lap_edges[0].device != self.param_table.device:
            self._lap_edges = self._build_lap_edges()
        idx, w = self._lap_edges
        c = self.branch_net(self.param_table)               # [S, C*K]
        cn = c[idx]                                          # [S, k, C*K]
        num = (w.unsqueeze(-1) * (cn - c.unsqueeze(1)).pow(2)).sum()
        den = w.sum() * c.pow(2).mean().clamp_min(1e-12)
        return num / den

    def coefficients(self, x_branch):
        B = x_branch.shape[0]
        return self.branch_net(x_branch).view(B, self.num_outputs, -1)

    def decode(self, shape_enc, log_fs_z):
        """Internal shape enc + predicted log-freestreams -> benchmark enc."""
        t = self.tmin + (self.tmax - self.tmin) * shape_enc    # internal units
        log_fs = (self.fs_mu + self.fs_sd * log_fs_z).unsqueeze(1)  # [B,1,3] log10 fs
        rho = torch.pow(10.0, t[..., 0:1] + log_fs[..., 0:1])
        vel = torch.pow(10.0, log_fs[..., 1:2])                 # u_inf
        u = vel * t[..., 1:2]
        v = vel * t[..., 2:3]
        log_p = t[..., 3:4] + log_fs[..., 2:3]
        return torch.cat([(rho - self.ymin[0]) / self.yrange[0],
                          (u - self.ymin[1]) / self.yrange[1],
                          (v - self.ymin[2]) / self.yrange[2],
                          (log_p - self.ymin[3]) / self.yrange[3]], dim=-1)

    def forward(self, x_branch, x_trunk):
        B, N = x_branch.shape[0], x_trunk.shape[0]
        t = self.trunk_net(x_trunk)                           # [N, K]
        b = self.coefficients(x_branch)                       # [B, C, K]
        shape_enc = torch.einsum('bck,nk->bnc', b, t) + self.shape_bias

        fs_z = self.fs_head(x_branch)                         # [B, 3]
        if self.training:
            # pin the head to the exact table row (train params match exactly)
            near = torch.cdist(x_branch, self.param_table).argmin(1)
            self.aux_loss = self.aux_weight * (fs_z - self.log_fs_z[near]).pow(2).mean()

        return self.decode(shape_enc, fs_z)


if __name__ == '__main__':
    # self-check: decode(encode(GT)) is the identity on a synthetic sample
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = FS_DeepONet().to(dev)
    # fabricate train-split-like stats
    m.tmin.copy_(torch.tensor([-0.5, -0.3, -0.3, -0.2], device=dev))
    m.tmax.copy_(torch.tensor([3.0, 1.0, 0.7, 3.1], device=dev))
    m.ymin.copy_(torch.tensor([0.0, -1.0, -1.0, 4.0], device=dev))
    m.yrange.copy_(torch.tensor([3.3, 7.0, 5.0, 3.0], device=dev))
    log_fs = torch.tensor([-2.8, 3.66, 2.24], device=dev)
    m.fs_mu.copy_(log_fs)
    m.fs_sd.copy_(torch.ones(3, device=dev))
    m.log_fs_z.copy_(torch.zeros(51, 3, device=dev))
    m.param_table.copy_(torch.rand(51, 3, device=dev))
    # encode: pick a GT in internal units, map to benchmark space via enc
    t = torch.tensor([1.2, 0.8, 0.1, 0.9], device=dev)
    enc = ((t - m.tmin) / (m.tmax - m.tmin)).view(1, 1, 4)
    bench = m.decode(enc, torch.zeros(1, 1, 3, device=dev))
    rho = 10 ** (t[0] + log_fs[0]); u = (10 ** log_fs[1]) * t[1]
    v = (10 ** log_fs[1]) * t[2]; lp = t[3] + log_fs[2]
    ref = torch.tensor([(rho - m.ymin[0]) / m.yrange[0], (u - m.ymin[1]) / m.yrange[1],
                        (v - m.ymin[2]) / m.yrange[2], (lp - m.ymin[3]) / m.yrange[3]], device=dev)
    assert torch.allclose(bench[0, 0], ref, atol=1e-5), (bench[0, 0], ref)
    # forward smoke test + Laplacian path
    m2 = FS_DeepONet(lap=True).to(dev)
    out = m2(torch.rand(2, 3, device=dev), torch.rand(100, 2, device=dev))
    assert out.shape == (2, 100, 4)
    m2.train()
    _ = m2(torch.rand(2, 3, device=dev), torch.rand(100, 2, device=dev))
    r = m2.graph_lap_loss()
    assert torch.isfinite(r)
    n = sum(p.numel() for p in m.parameters())
    print(f'fs_deeponet self-check OK, params {n / 1e6:.3f}M (excl. buffers), lap {r.item():.4f}')
