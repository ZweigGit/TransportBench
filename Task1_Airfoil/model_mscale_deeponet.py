import torch
import torch.nn as nn


class PhiActivation(nn.Module):
    """B-spline of order 3, compact support on [0, 3] (Liu et al. 2020)."""
    def forward(self, x):
        return (torch.relu(x) ** 2
                - 3 * torch.relu(x - 1) ** 2
                + 3 * torch.relu(x - 2) ** 2
                - torch.relu(x - 3) ** 2)


class _FNN(nn.Module):
    """Fully-connected net."""
    def __init__(self, dims, act):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _MscaleTrunk(nn.Module):
    """MscaleDNN trunk: parallel frequency-scaled branches fused by a shared FNN.
    Uses PhiActivation (B-spline of order 3) per Liu et al. 2020."""
    def __init__(self, trunk_dim, hidden_dim, scales, depth):
        super().__init__()
        act = PhiActivation()
        n_scales = len(scales)
        self.scales = nn.Parameter(
            torch.tensor(scales, dtype=torch.float32), requires_grad=False
        )
        branch_dims = [trunk_dim] + [hidden_dim] * (depth - 1) + [hidden_dim]
        self.branches = nn.ModuleList([_FNN(branch_dims, act) for _ in range(n_scales)])
        fusion_dims = [n_scales * hidden_dim, hidden_dim, hidden_dim]
        self.fusion = _FNN(fusion_dims, act)

    def forward(self, x):
        single = x.dim() == 2
        if single:
            x = x.unsqueeze(0)
        outs = []
        for s, branch in zip(self.scales, self.branches):
            outs.append(branch(s * x))
        out = self.fusion(torch.cat(outs, dim=-1))
        if single:
            out = out.squeeze(0)
        return out


class MscaleDeepONet(nn.Module):
    """DeepONet with MscaleDNN trunk net for multi-scale coordinate processing.

    Args:
        branch_dim: Input dimension of the branch net (sensor values).
        trunk_dim:  Input dimension of the trunk net (coordinates).
        hidden_dim: Width of hidden layers.
        num_outputs: Output channels.
        scales: Frequency scaling factors for MscaleDNN trunk.
        depth: Number of hidden layers in branch and per-scale trunk FNNs.
        activation: Activation type ('GELU', 'Tanh', or 'Phi' for B-spline).
    """
    def __init__(self, branch_dim=674, trunk_dim=2, hidden_dim=256, num_outputs=4,
                 scales=None, depth=4, activation='GELU'):
        super().__init__()

        if scales is None:
            scales = [1.0, 2.0, 4.0, 8.0, 16.0]

        if activation == 'GELU':
            act = nn.GELU()
        elif activation == 'Tanh':
            act = nn.Tanh()
        elif activation == 'Phi':
            act = PhiActivation()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Branch Net
        branch_dims = [branch_dim] + [hidden_dim] * depth
        self.branch_net = _FNN(branch_dims, act)

        # Trunk Net (MscaleDNN) — uses PhiActivation internally
        self.trunk_net = _MscaleTrunk(trunk_dim, hidden_dim, scales, depth)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim * num_outputs)
        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim

    def forward(self, x_branch, x_trunk):
        """Forward pass.

        Args:
            x_branch: [Batch, branch_dim]
            x_trunk:  [N_points, trunk_dim] or [Batch, N_points, trunk_dim]

        Returns:
            [Batch, N_points, num_outputs]
        """
        B = x_branch.shape[0]
        b = self.branch_net(x_branch)                       # [B, hidden_dim]
        t = self.trunk_net(x_trunk)                         # [B_or_1, N, hidden_dim]

        if t.dim() == 2:
            N = t.shape[0]
            t = t.unsqueeze(0).expand(B, -1, -1)            # [B, N, hidden]
        else:
            N = t.shape[1]

        b_out = self.output_proj(b)                         # [B, hidden * num_outputs]
        b_out = b_out.view(B, self.num_outputs, self.hidden_dim)  # [B, num_outputs, hidden]

        pred = torch.einsum("bkh, bnh -> bnk", b_out, t)    # [B, num_outputs, N]
        pred = pred.permute(0, 2, 1)                        # [B, N, num_outputs]
        return pred
