"""
Residual Fusion DeepONet, ported from Task2 (Cylinder Flow).

Identical to Fusion_DeepONet except the branch-to-trunk fusion gate:
    Fusion_DeepONet:          x_trunk = x_trunk * skip[i]          (raw product)
    Residual_Fusion_DeepONet: x_trunk = x_trunk * (1.0 + skip[i])  (residual gate)

The identity term keeps the trunk signal/gradient path alive when the branch
signal is weak, preventing the branch-collapse failure mode of the raw
product. Same ~1.01M params as Fusion_DeepONet (Task4: branch_dim=3).
"""

import torch
import torch.nn as nn


class sin_act(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class Residual_Fusion_DeepONet(nn.Module):
    def __init__(self, branch_dim=3, trunk_dim=2, hidden_dim=278,
                 num_outputs=4, depth=5, activation='GELU'):
        """
        Residual Fusion DeepONet for Double Cone Flow Task.

        Args:
            branch_dim: 3 (Ma, T, Re) — flow params, constant per sample
            trunk_dim: 2 (x, y) — shared grid coordinates
            hidden_dim: 278 (target ~1.01M params)
            num_outputs: 4 (physics quantities)
            depth: number of hidden layers
        """
        super().__init__()

        if activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.act2 = sin_act()

        # Fusion-layer scalars
        self.L = depth + 1

        self.ab  = nn.Parameter(torch.full((self.L,), 0.1))
        self.cb  = nn.Parameter(torch.full((self.L,), 0.1))
        self.a1b = nn.Parameter(torch.zeros(self.L,))
        self.F1b = nn.Parameter(torch.full((self.L,), 0.1))
        self.c1b = nn.Parameter(torch.zeros(self.L,))

        self.at  = nn.Parameter(torch.full((self.L,), 0.1))
        self.ct  = nn.Parameter(torch.full((self.L,), 0.1))
        self.a1t = nn.Parameter(torch.zeros(self.L,))
        self.F1t = nn.Parameter(torch.full((self.L,), 0.1))
        self.c1t = nn.Parameter(torch.zeros(self.L,))

        # Branch Net
        self.branch_net = nn.ModuleList()
        self.branch_net.append(nn.Linear(branch_dim, hidden_dim))
        for _ in range(depth - 1):
            self.branch_net.append(nn.Linear(hidden_dim, hidden_dim))
        self.branch_net.append(nn.Linear(hidden_dim, hidden_dim * num_outputs))

        # Trunk Net: [Linear1, Linear2, ..., Last Linear]
        self.trunk_net = nn.ModuleList()
        self.trunk_net.append(nn.Linear(trunk_dim, hidden_dim))
        for _ in range(depth - 1):
            self.trunk_net.append(nn.Linear(hidden_dim, hidden_dim))
        self.trunk_net.append(nn.Linear(hidden_dim, hidden_dim))

        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim

    def forward(self, x_branch, x_trunk):
        """Forward pass.

        Args:
            x_branch: [B, branch_dim]
            x_trunk:  [N, trunk_dim] or [B, N, trunk_dim] (shared across batch)

        Returns:
            [B, N, num_outputs]
        """
        B = x_branch.shape[0]
        if x_trunk.dim() == 2:
            # Shared 2D trunk: [N, trunk_dim] -> [B, N, trunk_dim] (view, no copy)
            x_trunk = x_trunk.unsqueeze(0).expand(B, -1, -1)

        skip = []

        for i in range(self.L - 1):
            z_b = self.branch_net[i](x_branch)
            x_branch = self.act(10 * self.ab[i] * z_b + self.cb[i]) + \
                10 * self.a1b[i] * self.act2(10 * self.F1b[i] * z_b + self.c1b[i])
            skip.append(x_branch)

        # Cumulative skip accumulation
        for i in range(1, self.L - 1):
            skip[i] = skip[i - 1] + skip[i]

        for i in range(self.L - 1):
            z_t = self.trunk_net[i](x_trunk)
            x_trunk = self.act(10 * self.at[i] * z_t + self.ct[i]) + \
                10 * self.a1t[i] * self.act2(10 * self.F1t[i] * z_t + self.c1t[i])

            # Residual gate: identity path preserves trunk magnitude/gradient
            x_trunk = torch.einsum('bk,bik->bik', 1.0 + skip[i], x_trunk)

        x_branch = self.branch_net[-1](x_branch)
        x_trunk = self.trunk_net[-1](x_trunk)

        B_out_reshaped = x_branch.view(-1, self.num_outputs, self.hidden_dim)
        # [Batch, num_outputs, hidden_dim]

        prediction = torch.einsum("bnk, bik -> bin", B_out_reshaped, x_trunk)
        # [Batch, N_points, num_outputs]

        return prediction
