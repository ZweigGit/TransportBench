import torch
import torch.nn as nn


class PhiActivation(nn.Module):
    """B-spline of order 3, compact support on [0, 3] (Liu et al. 2020)."""
    def forward(self, x):
        return (torch.relu(x) ** 2
                - 3 * torch.relu(x - 1) ** 2
                + 3 * torch.relu(x - 2) ** 2
                - torch.relu(x - 3) ** 2)


class SinActivation(nn.Module):
    """Sine activation for multi-scale frequency processing."""
    def forward(self, x):
        return torch.sin(x)


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


class _MscaleTrunk(nn.Module):
    """MscaleDNN trunk: parallel frequency-scaled branches stacked as output.
    Uses Phi (B-spline) activation for multi-scale frequency processing.
    No fusion layer — sub-network outputs are concatenated directly,
    so trunk feature dim = n_scales * out_dim."""
    def __init__(self, trunk_dim, hidden_dim, scales, depth, out_dim=None):
        super().__init__()
        act = PhiActivation()
        n_scales = len(scales)
        self.scales = nn.Parameter(
            torch.tensor(scales, dtype=torch.float32), requires_grad=False
        )
        # out_dim: per-scale output width; None → hidden_dim (output = hidden width)
        if out_dim is None:
            out_dim = hidden_dim
        branch_dims = [trunk_dim] + [hidden_dim] * depth + [out_dim]
        self.branches = nn.ModuleList([_FNN(branch_dims, act) for _ in range(n_scales)])
        self.out_dim = n_scales * out_dim

    def forward(self, x):
        single = x.dim() == 2
        if single:
            x = x.unsqueeze(0)
        outs = []
        for s, branch in zip(self.scales, self.branches):
            outs.append(branch(s * x))
        out = torch.cat(outs, dim=-1)
        if single:
            out = out.squeeze(0)
        return out


class MscaleDeepONet(nn.Module):
    """DeepONet with MscaleDNN trunk net for multi-scale coordinate processing.

    Trunk sub-network outputs are stacked directly (no fusion layer), so the
    trunk feature dim is n_scales * trunk_hidden. Branch width is controlled
    independently by branch_hidden.

    Args:
        branch_dim:    Input dimension of the branch net (sensor values).
        trunk_dim:     Input dimension of the trunk net (coordinates).
        branch_hidden: Width of the branch net hidden layers (independent of
                       the output dim, which is inferred as
                       num_outputs * trunk_feat_dim).
        trunk_hidden:  Width of each trunk sub-network (must satisfy
                       n_scales * trunk_hidden == trunk feature dim, else error).
        num_outputs:   Output channels.
        scales:        Frequency scaling factors for MscaleDNN trunk.
        branch_depth:  Number of hidden layers in the branch net.
        trunk_depth:   Number of hidden layers in each per-scale trunk FNN.
        activation:    Activation type ('GELU', 'Tanh', or 'Phi' for B-spline).
        basis_size:    Number of trunk basis functions (= trunk feature dim,
                       n_scales * out_dim). Sets each sub-network's OUTPUT
                       width (basis_size // n_scales); hidden width stays
                       trunk_hidden. Must be divisible by n_scales.
    """
    def __init__(self, branch_dim=3, trunk_dim=2, branch_hidden=512,
                 trunk_hidden=256, num_outputs=4, scales=None,
                 branch_depth=6, trunk_depth=6, activation='GELU',
                 basis_size=None):
        super().__init__()

        if scales is None:
            scales = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
        n_scales = len(scales)

        # basis_size = number of trunk basis functions = trunk feature dim.
        # It sets each sub-network's OUTPUT width (basis_size // n_scales),
        # leaving the hidden width (trunk_hidden) untouched.
        out_dim = None
        if basis_size is not None:
            if basis_size % n_scales != 0:
                raise ValueError(
                    f"basis_size {basis_size} must be divisible by n_scales {n_scales}"
                )
            out_dim = basis_size // n_scales

        if activation == 'GELU':
            act = nn.GELU()
        elif activation == 'Tanh':
            act = nn.Tanh()
        elif activation == 'Phi':
            act = PhiActivation()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Trunk Net (MscaleDNN) — uses PhiActivation internally
        self.trunk_net = _MscaleTrunk(trunk_dim, trunk_hidden, scales, trunk_depth, out_dim=out_dim)
        trunk_feat_dim = self.trunk_net.out_dim

        if trunk_feat_dim != len(scales) * (out_dim if out_dim is not None else trunk_hidden):
            raise ValueError(
                f"Trunk output dim {trunk_feat_dim} != sub-network output "
                f"({out_dim if out_dim is not None else trunk_hidden}) * n_scales ({len(scales)})"
            )

        # Branch Net — output dim inferred from trunk: num_outputs * trunk_feat_dim
        # (each output channel gets its own coefficient vector for the einsum)
        branch_dims = [branch_dim] + [branch_hidden] * branch_depth + [num_outputs * trunk_feat_dim]
        self.branch_net = _FNN(branch_dims, act)
        self.num_outputs = num_outputs
        self.trunk_feat_dim = trunk_feat_dim
        self.branch_hidden = branch_hidden

    def forward(self, x_branch, x_trunk):
        """Forward pass.

        Args:
            x_branch: [Batch, branch_dim]
            x_trunk:  [N_points, trunk_dim] or [Batch, N_points, trunk_dim]

        Returns:
            [Batch, N_points, num_outputs]
        """
        B = x_branch.shape[0]
        b = self.branch_net(x_branch)                       # [B, num_outputs * trunk_feat_dim]
        t = self.trunk_net(x_trunk)                         # [B_or_1, N, trunk_feat_dim]

        if t.dim() == 2:
            N = t.shape[0]
            t = t.unsqueeze(0).expand(B, -1, -1)            # [B, N, trunk_feat_dim]
        else:
            N = t.shape[1]

        # Guard: einsum feature dims must match (trunk actual output vs branch coefficient width)
        assert t.shape[-1] == self.trunk_feat_dim, \
            f"Trunk actual output dim {t.shape[-1]} != trunk_feat_dim {self.trunk_feat_dim}"

        b_out = b.view(B, self.num_outputs, self.trunk_feat_dim)  # [B, num_outputs, trunk_feat_dim]

        pred = torch.einsum("bkh, bnh -> bnk", b_out, t)    # [B, N, num_outputs]
        return pred
