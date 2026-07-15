"""
HyperMscaleDeepONet: HyperDeepONet idea applied to MscaleDeepONet.

Branch net learns ALL parameters of the MscaleTrunk (including per-scale branch
FNNs, fusion FNN weights/biases, and output projection). No learned parameters
in the trunk — every weight and bias comes from the branch output at runtime.
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
    """HyperDeepONet with MscaleDNN trunk — branch net outputs all trunk parameters.

    The branch net also outputs learnable log-scale factors (s_p = exp(log_s)) for
    the Mscale trunk, so scale factors adapt to sensor inputs.

    Args:
        branch_dim:   Input dimension of the branch net (sensor values).
        trunk_dim:    Input dimension of the trunk net (coordinates).
        hidden_dim:   Width of hidden layers.
        num_outputs:  Number of output channels.
        num_scales:   Number of learnable scale factors (default 5).
        depth:        Number of hidden layers in per-scale trunk FNNs and in branch net.
        activation:   'GELU' or 'Tanh' (branch activation; trunk always uses B-spline).
    """
    def __init__(self, branch_dim, trunk_dim, hidden_dim=256, num_outputs=4,
                 num_scales=5, depth=4, activation='GELU'):
        super().__init__()

        if activation == 'GELU':
            act = nn.GELU()
        elif activation == 'Tanh':
            act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        n_scales = num_scales
        self.num_outputs = num_outputs

        # --- Compute total parameters needed to construct the hyper-trunk ---
        # Each scale branch: [trunk_dim] + [hidden_dim] * depth  -> depth+1 dims, depth layers
        branch_dims = [trunk_dim] + [hidden_dim] * depth
        per_branch_params = _compute_weight_bias(branch_dims)

        # Fusion FNN: [n_scales * hidden_dim] -> [hidden_dim]  (single linear layer)
        fusion_dims = [n_scales * hidden_dim, hidden_dim]
        fusion_params = _compute_weight_bias(fusion_dims)

        # Output layer: hidden_dim -> num_outputs
        output_dims = [hidden_dim, num_outputs]
        output_params = _compute_weight_bias(output_dims)

        t_para = n_scales * per_branch_params + fusion_params + output_params + n_scales  # +n_scales for log_scales

        # --- Branch net ---
        self.branch_net = _MLP([branch_dim] + [hidden_dim] * depth + [t_para], act)

        # --- Stash shapes for trunk forward ---
        self._branch_dims = branch_dims
        self._fusion_dims = fusion_dims
        self._output_dims = output_dims
        self._n_scales = n_scales
        self._per_branch_params = per_branch_params
        self._fusion_params = fusion_params
        self._output_params = output_params

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
        """Hypernetwork MscaleTrunk forward using branch-provided weights/biases.

        params: [B, t_para] — flattened trunk weights & biases
        x_trunk: [1, N, trunk_dim] or [B, N, trunk_dim]
        """
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0)  # [1, N, trunk_dim]
        B = params.shape[0]
        _, N, _ = x_trunk.shape

        # --- Extract log-scales from end of params ---
        log_scales = params[:, -self._n_scales:]  # [B, n_scales]
        scales = torch.exp(log_scales)             # [B, n_scales], >0 guaranteed

        # --- Per-scale branches ---
        d_in0, d_out0 = self._branch_dims[0], self._branch_dims[1]
        branch_out = []
        start = 0

        for s_idx in range(self._n_scales):
            scale = scales[:, s_idx].view(-1, 1, 1)  # [B, 1, 1]
            y = scale * x_trunk  # [B_or_1, N, trunk_dim]

            # First layer: trunk_dim -> hidden_dim
            y, start = self._apply_layer(params, y, d_in0, d_out0, start,
                                         act_fn=_phi)
            # Remaining layers: hidden_dim -> hidden_dim
            for i in range(1, len(self._branch_dims) - 1):
                d_in = self._branch_dims[i]
                d_out = self._branch_dims[i + 1]
                y, start = self._apply_layer(params, y, d_in, d_out, start,
                                             act_fn=_phi)
            branch_out.append(y)  # [B, N, hidden_dim]

        # --- Fusion ---
        y = torch.cat(branch_out, dim=-1)  # [B, N, n_scales * hidden_dim]
        d_fin, d_fout = self._fusion_dims[0], self._fusion_dims[1]
        y, start = self._apply_layer(params, y, d_fin, d_fout, start,
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
