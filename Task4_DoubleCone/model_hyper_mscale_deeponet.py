"""
HyperMscaleDeepONet: HyperDeepONet with a multi-scale stacked trunk.

Branch net outputs all trunk parameters — one FNN per scale (weights + biases),
one output layer, and one learnable log-scale factor. No learned parameters in
the trunk — every weight, bias, and the scale comes from the branch output at
runtime. Trunk sub-network outputs are stacked directly (no fusion layer), so
trunk feature dim = n_scales * out_dim (same design as MscaleDeepONet).
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
    """HyperDeepONet with a multi-scale stacked trunk — branch net outputs all trunk parameters.

    Branch net outputs one trunk FNN per scale (weights + biases), one output
    layer, and one learnable log-scale factor. No learned parameters in the trunk.

    Args:
        branch_dim:   Input dimension of the branch net (sensor values).
        trunk_dim:    Input dimension of the trunk net (coordinates).
        hidden_dim:   Width of the branch net hidden layers.
        trunk_hidden: Width of each per-scale trunk FNN hidden layers (trunk
                      feature dim = n_scales * out_dim, where out_dim =
                      basis_size // n_scales, or trunk_hidden if basis_size
                      is None).
        num_outputs:  Number of output channels.
        depth:        Number of hidden layers in the branch net.
        trunk_depth:  Number of hidden layers in each per-scale trunk FNN.
        scales:       Frequency scaling factors for the trunk.
        activation:   'GELU' or 'Tanh' (branch activation; trunk always uses B-spline).
        basis_size:   Number of trunk basis functions (= trunk feature dim,
                      n_scales * out_dim). Sets each sub-network's OUTPUT
                      width (basis_size // n_scales); hidden width stays
                      trunk_hidden. Must be divisible by n_scales.
    """
    def __init__(self, branch_dim=3, trunk_dim=2, hidden_dim=68, trunk_hidden=68,
                 num_outputs=4, depth=4, trunk_depth=4, scales=None,
                 activation='GELU', basis_size=None):
        super().__init__()

        if activation == 'GELU':
            act = nn.GELU()
        elif activation == 'Tanh':
            act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

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

        self.num_outputs = num_outputs

        # --- Compute total parameters needed to construct the hyper-trunk ---
        # Per-scale trunk FNN: [trunk_dim] + [trunk_hidden]*trunk_depth + [out_dim]
        if out_dim is None:
            out_dim = trunk_hidden
        scale_dims = [trunk_dim] + [trunk_hidden] * trunk_depth + [out_dim]
        trunk_feat_dim = n_scales * out_dim

        trunk_params = n_scales * _compute_weight_bias(scale_dims)

        # Output layer: trunk_feat_dim -> num_outputs
        output_dims = [trunk_feat_dim, num_outputs]
        output_params = _compute_weight_bias(output_dims)

        t_para = trunk_params + output_params + 1  # +1 for log_scale

        # --- Branch net ---
        self.branch_net = _MLP([branch_dim] + [hidden_dim] * depth + [t_para], act)

        # --- Stash shapes and scales for trunk forward ---
        self._scale_dims = scale_dims
        self._output_dims = output_dims
        self.scales = scales

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

        # --- Per-scale trunk FNNs, outputs stacked (no fusion) ---
        start = 0
        outs = []
        for s in self.scales:
            y_s = y * s
            for i in range(len(self._scale_dims) - 1):
                d_in = self._scale_dims[i]
                d_out = self._scale_dims[i + 1]
                y_s, start = self._apply_layer(params, y_s, d_in, d_out, start,
                                               act_fn=_phi)
            outs.append(y_s)
        y = torch.cat(outs, dim=-1)  # [B, N, n_scales * trunk_hidden]

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
