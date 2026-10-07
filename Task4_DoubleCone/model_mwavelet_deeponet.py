import torch
import torch.nn as nn

from model_wavelet_deeponet import _FNN, _GaborTrunk

# Ported from Multi-Resolution-WNN experiments/18_gabor_fnn_f1.jl (MwaveletNN):
# multi-branch Gabor trunk WITHOUT an input scaling layer — the scales live in
# the per-branch Gabor activation (w0, s0). Same layout as MscaleDeepONet, but
# each trunk sub-network is a complex Gabor FNN whose real-part output is
# concatenated with its siblings into the trunk feature vector.


class _MwaveletTrunk(nn.Module):
    """Parallel complex Gabor FNN sub-trunks, one per (w0, s0) pair. Each
    returns the real part of its complex output; outputs are concatenated,
    so trunk feature dim = n_wavelets * out_dim."""
    def __init__(self, trunk_dim, hidden_dim, depth, out_dim, wavelets):
        super().__init__()
        self.branches = nn.ModuleList([
            _GaborTrunk(trunk_dim, hidden_dim, depth, out_dim, w0, s0)
            for w0, s0 in wavelets
        ])
        self.out_dim = len(wavelets) * out_dim

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=-1)


class MwaveletDeepONet(nn.Module):
    """DeepONet with a multi-branch Gabor-wavelet trunk: branch i's atoms are
    psi(z; w0_i, s0_i) = exp(i*w0_i*z) * exp(-|s0_i*z|^2), so the multi-scale
    structure lives in the activation constants instead of an input scaling
    layer. Trunk feature vector = concatenation of the (equal-width) real-part
    sub-network outputs.

    Args:
        branch_dim:   Input dimension of the branch net (flow params).
        trunk_dim:    Input dimension of the trunk net (coordinates).
        branch_hidden / branch_depth: Branch net width / hidden layers.
        trunk_hidden / trunk_depth:   Width / hidden layers of each sub-trunk.
        basis_size:   Total trunk feature dim (= n_wavelets * per-branch out
                      width); must be divisible by len(wavelets).
        num_outputs:  Output channels.
        wavelets:     Per-branch (w0, s0) Gabor constants (fixed, not trained).
    """
    def __init__(self, branch_dim=3, trunk_dim=2, branch_hidden=256,
                 trunk_hidden=160, branch_depth=4, trunk_depth=4,
                 basis_size=128, num_outputs=4,
                 wavelets=None):
        super().__init__()
        if wavelets is None:
            wavelets = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
        if basis_size % len(wavelets) != 0:
            raise ValueError(
                f"basis_size {basis_size} must be divisible by "
                f"n_wavelets {len(wavelets)}"
            )
        out_dim = basis_size // len(wavelets)

        self.trunk_net = _MwaveletTrunk(trunk_dim, trunk_hidden, trunk_depth,
                                        out_dim, wavelets)
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
