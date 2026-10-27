import torch
import torch.nn as nn

# Multi-branch trunk in the layout of the original MscaleDeepONet (parallel
# sub-trunks whose outputs are concatenated), but each sub-trunk is a real FNN
# whose hidden activation is the real Gabor wavelet
# psi(x) = exp(-(s0*x)^2) * cos(w0*x). All branches share one s0; branch i
# uses w0_i = i * w0 (distinct per branch), so the multi-scale structure lives
# in the per-branch activation constants instead of an input scaling layer.


class RealGaborActivation(nn.Module):
    """Real Gabor wavelet psi(x) = exp(-(s0*x)^2) * cos(w0*x).
    w0/s0 are fixed hyperparameters passed at construction, not learned."""
    def __init__(self, w0=3.0, s0=3.0):
        super().__init__()
        self.w0 = float(w0)
        self.s0 = float(s0)

    def forward(self, x):
        # Bounded in [-1, 1]: no overflow/clamp needed; large |s0| just
        # underflows env (and its gradient) to 0.
        env = torch.exp((self.s0 * x) ** 2 * -1.0)
        return env * torch.cos(self.w0 * x)


class _GaborTrunk(nn.Module):
    """Real FNN trunk: Linear -> RealGabor -> ... -> Linear (shared activation
    module so all layers update the same w0/s0 pair)."""
    def __init__(self, trunk_dim, hidden_dim, depth, out_dim, w0, s0):
        super().__init__()
        act = RealGaborActivation(w0, s0)
        dims = [trunk_dim] + [hidden_dim] * depth + [out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _FNN(nn.Module):
    """Real FNN: Linear -> GELU -> ... -> Linear (branch)."""
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _MwaveletTrunk(nn.Module):
    """Parallel real Gabor FNN sub-trunks, one per (w0_i, s0) pair. Each
    returns a real feature vector; outputs are concatenated, so trunk feature
    dim = n_branches * out_dim."""
    def __init__(self, trunk_dim, hidden_dim, depth, out_dim, w0s, s0):
        super().__init__()
        self.branches = nn.ModuleList([
            _GaborTrunk(trunk_dim, hidden_dim, depth, out_dim, w0_i, s0)
            for w0_i in w0s
        ])
        self.out_dim = len(w0s) * out_dim

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=-1)


class MwaveletDeepONet(nn.Module):
    """DeepONet with a multi-branch real-Gabor trunk: branch i's hidden
    activation is psi(z; w0_i, s0) = exp(-(s*z)^2) * cos(w0_i*z) applied at
    every layer, with s0 shared across branches and w0_i = i * w0 distinct
    per branch. Trunk feature vector = concatenation of the (equal-width)
    sub-network outputs.

    Args:
        branch_dim:   Input dimension of the branch net (flow params).
        trunk_dim:    Input dimension of the trunk net (coordinates).
        branch_hidden / branch_depth: Branch net width / hidden layers.
        trunk_hidden / trunk_depth:   Width / hidden layers of each sub-trunk.
        basis_size:   Total trunk feature dim (= n_branches * per-branch out
                      width); must be divisible by n_branches.
        num_outputs:  Output channels.
        w0s:          Per-branch Gabor frequencies w0_i (fixed, not learned);
                      default [1.0, 2.0, ..., n_branches].
        s0:           Gabor dilation hyperparameter shared by all branches
                      (fixed, not learned).
    """
    def __init__(self, branch_dim=3, trunk_dim=2, branch_hidden=256,
                 trunk_hidden=160, branch_depth=4, trunk_depth=4,
                 basis_size=128, num_outputs=4, w0s=None, s0=1.0):
        super().__init__()
        if w0s is None:
            w0s = [(i + 1) * 1.0 for i in range(4)]
        n_branches = len(w0s)
        if basis_size % n_branches != 0:
            raise ValueError(
                f"basis_size {basis_size} must be divisible by "
                f"n_branches {n_branches}"
            )
        out_dim = basis_size // n_branches

        self.trunk_net = _MwaveletTrunk(trunk_dim, trunk_hidden, trunk_depth,
                                        out_dim, w0s, s0)
        self.trunk_feat_dim = self.trunk_net.out_dim

        branch_dims = ([branch_dim] + [branch_hidden] * branch_depth
                       + [num_outputs * basis_size])
        self.branch_net = _FNN(branch_dims)

        self.num_outputs = num_outputs
        self.bias = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, x_branch, x_trunk):
        """Forward pass.

        Args:
            x_branch: [Batch, branch_dim]
            x_trunk:  [N_points, trunk_dim] or [Batch, N_points, trunk_dim]

        Returns:
            [Batch, N_points, num_outputs]
        """
        B = x_branch.shape[0]
        b = self.branch_net(x_branch).view(B, self.num_outputs, self.trunk_feat_dim)
        t = self.trunk_net(x_trunk)                       # [N, basis_size]
        if t.dim() == 2:
            t = t.unsqueeze(0).expand(B, -1, -1)          # [B, N, basis_size]
        pred = torch.einsum("bkh, bnh -> bnk", b, t)      # [B, N, num_outputs]
        return pred + self.bias.view(1, 1, -1)
