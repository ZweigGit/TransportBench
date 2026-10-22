import torch
import torch.nn as nn


class _FNN(nn.Module):
    """Fully-connected net: Linear -> Act -> ... -> Linear."""
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


class MscaleDeepONet(nn.Module):
    """DeepONet with a plain single FNN trunk — the multi-scale trunk (parallel
    frequency-scaled sub-networks) is replaced by one standard trunk; the branch
    net and the basis dot product are unchanged (same ~1M budget ablation).

    Args:
        branch_dim:    Input dimension of the branch net (flow params).
        trunk_dim:     Input dimension of the trunk net (coords).
        hidden:        Width of both the branch and the trunk net.
        num_outputs:   Output channels.
        depth:         Number of hidden layers in each net.
        basis_size:    Number of trunk basis functions (= trunk feature dim).
        activation:    Activation type ('GELU' or 'Tanh').
    """
    def __init__(self, branch_dim=3, trunk_dim=2, hidden=313, num_outputs=4,
                 depth=5, basis_size=128, activation='GELU'):
        super().__init__()
        if activation == 'GELU':
            act = nn.GELU()
        elif activation == 'Tanh':
            act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.trunk_net = _FNN([trunk_dim] + [hidden] * depth + [basis_size], act)
        self.branch_net = _FNN([branch_dim] + [hidden] * depth + [num_outputs * basis_size], act)
        self.num_outputs = num_outputs
        self.basis_size = basis_size

    def forward(self, x_branch, x_trunk):
        """Forward pass.

        Args:
            x_branch: [Batch, branch_dim]
            x_trunk:  [N_points, trunk_dim] or [Batch, N_points, trunk_dim]

        Returns:
            [Batch, N_points, num_outputs]
        """
        B = x_branch.shape[0]
        b = self.branch_net(x_branch)                       # [B, num_outputs * basis_size]
        t = self.trunk_net(x_trunk)                         # [B_or_1, N, basis_size]

        if t.dim() == 2:
            t = t.unsqueeze(0).expand(B, -1, -1)            # [B, N, basis_size]

        b_out = b.view(B, self.num_outputs, self.basis_size)
        return torch.einsum("bkh, bnh -> bnk", b_out, t)    # [B, N, num_outputs]
