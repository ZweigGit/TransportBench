"""
HyperDeepONet: DeepONet with a hypernetwork trunk.

Key ideas:
- Branch net outputs ARE the trunk net's weights/biases (hypernetwork).
- Multiple branch subnets (one per output variable) fuse via element-wise product.
- No learned parameters in the trunk — all trunk params come from the branch output.

Reference: Lee & Shin, "HyperDeepONet: a hypernetwork-based deep operator learning framework"
"""

import torch
import torch.nn as nn


class _MLP(nn.Module):
    """Fully-connected stack: Linear -> Act -> ... -> Linear."""
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


class HyperDeepONet(nn.Module):
    """HyperDeepONet for operator learning.

    Args:
        branch_dim:   Input dimension of the branch net (sensor values).
        trunk_dim:    Input dimension of the trunk net (coordinates).
        hidden_dim:   Width of hidden layers in both branch and trunk.
        num_outputs:  Number of output channels.
        trunk_depth:  Number of hidden layers in the hypernetwork trunk.
        branch_depth: Number of hidden layers in each branch subnet.
        activation:   'Tanh' or 'GELU'.
    """
    def __init__(self, branch_dim, trunk_dim, hidden_dim=46, num_outputs=4,
                 trunk_depth=3, branch_depth=3, activation='Tanh'):
        super().__init__()

        if activation == 'Tanh':
            act = nn.Tanh
        elif activation == 'GELU':
            act = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Trunk architecture: [trunk_dim, hidden, ..., hidden, num_outputs]
        self.trunk_dims = [trunk_dim] + [hidden_dim] * trunk_depth + [num_outputs]

        # Total parameters needed to construct the trunk net
        t_para = 0
        for i in range(len(self.trunk_dims) - 1):
            t_para += self.trunk_dims[i] * self.trunk_dims[i + 1] + self.trunk_dims[i + 1]

        # Branch: num_outputs sub-networks -> element-wise product -> t_para
        branch_dims = [branch_dim] + [hidden_dim] * branch_depth + [t_para]

        self.branch_subnets = nn.ModuleList([
            _MLP(branch_dims, act) for _ in range(num_outputs)
        ])

        self.num_outputs = num_outputs

    def _branch_forward(self, x):
        """Run each branch subnet on x, fuse with element-wise product."""
        out = self.branch_subnets[0](x)
        for subnet in self.branch_subnets[1:]:
            out = out * subnet(x)
        return out  # [B, t_para]

    def _trunk_forward(self, params, x_trunk):
        """Hypernetwork trunk: params -> weights/biases -> forward pass."""
        # Normalize to 3D: [B, N, trunk_dim]
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0)  # [1, N, trunk_dim]

        B, N, _ = x_trunk.shape
        y = x_trunk  # [B, N, trunk_dim]
        start = 0

        for i in range(len(self.trunk_dims) - 2):
            d_in, d_out = self.trunk_dims[i], self.trunk_dims[i + 1]

            w_sz = d_in * d_out
            weight = params[:, start:start + w_sz].reshape(B, d_out, d_in)
            start += w_sz
            bias = params[:, start:start + d_out].reshape(B, 1, d_out)
            start += d_out

            y = torch.einsum("bij,bgj->bgi", weight, y) + bias  # [B, N, d_out]
            y = torch.tanh(y)

        # Last layer: no activation
        d_in, d_out = self.trunk_dims[-2], self.trunk_dims[-1]
        w_sz = d_in * d_out
        weight = params[:, start:start + w_sz].reshape(B, d_out, d_in)
        start += w_sz
        bias = params[:, start:start + d_out].reshape(B, 1, d_out)

        y = torch.einsum("bij,bgj->bgi", weight, y) + bias  # [B, N, num_outputs]
        return y

    def forward(self, x_branch, x_trunk):
        """
        Args:
            x_branch: [Batch, branch_dim]  sensor values
            x_trunk:  [N_points, trunk_dim] or [Batch, N_points, trunk_dim]  coordinates

        Returns:
            [Batch, N_points, num_outputs]
        """
        params = self._branch_forward(x_branch)  # [B, t_para]
        return self._trunk_forward(params, x_trunk)
