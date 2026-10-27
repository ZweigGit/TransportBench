"""
HyperMscaleDeepONet: HyperDeepONet with a single-scale trunk.

Branch net outputs all trunk parameters (a single FNN + output projection)
plus one learnable log-scale factor. No learned parameters in the trunk —
every weight, bias, and the scale comes from the branch output at runtime.
"""

import torch
import torch.nn as nn


def _phi(x):
    """B-spline of order 3, compact support on [0, 3]."""
    return (torch.relu(x) ** 2
            - 3 * torch.relu(x - 1) ** 2
            + 3 * torch.relu(x - 2) ** 2
            - torch.relu(x - 3) ** 2)


def _compute_weight_bias(dims):
    """Total parameter count for a linear stack of given dims (weights + biases)."""
    total = 0
    for i in range(len(dims) - 1):
        total += dims[i] * dims[i + 1] + dims[i + 1]
    return total


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


class HyperMscaleDeepONet(nn.Module):
    """HyperDeepONet with a single-scale trunk — branch net outputs all trunk parameters.

    Branch net outputs one trunk FNN (weights + biases), one output projection,
    and one learnable log-scale factor. No learned parameters in the trunk.

    Args:
        branch_dim:   Input dimension of the branch net (sensor values).
        trunk_dim:    Input dimension of the trunk net (coordinates).
        hidden_dim:   Width of hidden layers.
        num_outputs:  Number of output channels.
        depth:        Number of hidden layers in trunk FNN and branch net.
        activation:   'GELU' or 'Tanh' (branch activation; trunk always uses B-spline).
    """
    def __init__(self, branch_dim, trunk_dim, hidden_dim=256, num_outputs=4,
                 depth=4, activation='GELU'):
        super().__init__()

        if activation == 'GELU':
            act = nn.GELU()
        elif activation == 'Tanh':
            act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.num_outputs = num_outputs

        # --- Compute total parameters needed to construct the hyper-trunk ---
        # Trunk FNN: [trunk_dim] + [hidden_dim] * depth
        trunk_dims = [trunk_dim] + [hidden_dim] * depth
        trunk_params = _compute_weight_bias(trunk_dims)

        # Output layer: hidden_dim -> num_outputs
        output_dims = [hidden_dim, num_outputs]
        output_params = _compute_weight_bias(output_dims)

        t_para = trunk_params + output_params + 1  # +1 for log_scale

        # --- Branch net ---
        self.branch_net = _MLP([branch_dim] + [hidden_dim] * depth + [t_para], act)

        # --- Stash shapes for trunk forward ---
        self._trunk_dims = trunk_dims
        self._output_dims = output_dims

    @staticmethod
    def _apply_layer(params, x, d_in, d_out, start, act_fn=None):
        """Slice, reshape, apply Linear(d_in, d_out), advance start. Returns (out, new_start)."""
        B = params.shape[0]
        w_sz = d_in * d_out
        weight = params[:, start:start + w_sz].reshape(B, d_out, d_in)
        start += w_sz
        bias = params[:, start:start + d_out].reshape(B, 1, d_out)
        start += d_out
        y = torch.einsum("bij,bgj->bgi", weight, x) + bias
        if act_fn is not None:
            y = act_fn(y)
        return y, start

    def _trunk_forward(self, params, x_trunk):
        """Hypernetwork trunk forward using branch-provided weights/biases.

        params: [B, t_para] — flattened trunk weights, biases, and log_scale
        x_trunk: [N, trunk_dim] or [B, N, trunk_dim]
        """
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0)  # [1, N, trunk_dim]
        B = params.shape[0]

        # --- Extract log_scale from end of params ---
        log_scale = params[:, -1:]  # [B, 1]
        scale = torch.exp(log_scale)  # [B, 1], >0 guaranteed

        # Apply scale to input coordinates
        y = scale.view(B, 1, 1) * x_trunk  # [B, N, trunk_dim]

        # --- Trunk FNN ---
        start = 0
        for i in range(len(self._trunk_dims) - 1):
            d_in = self._trunk_dims[i]
            d_out = self._trunk_dims[i + 1]
            y, start = self._apply_layer(params, y, d_in, d_out, start,
                                         act_fn=_phi)

        # --- Output layer (no activation) ---
        d_oin, d_oout = self._output_dims[0], self._output_dims[1]
        y, start = self._apply_layer(params, y, d_oin, d_oout, start,
                                     act_fn=None)
        return y  # [B, N, num_outputs]

    def forward(self, x_branch, x_trunk):
        """
        Args:
            x_branch: [Batch, branch_dim]  sensor values
            x_trunk:  [N_points, trunk_dim] or [Batch, N_points, trunk_dim]

        Returns:
            [Batch, N_points, num_outputs]
        """
        params = self.branch_net(x_branch)  # [B, t_para]
        return self._trunk_forward(params, x_trunk)
