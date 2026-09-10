"""
Chunked HyperMscaleDeepONet: HyperMscaleDeepONet with the trunk parameters
emitted in chunks, keeping the branch MLP small.

- A learnable latent code per chunk turns one shared branch net into K chunk
  generators; the per-scale trunk FNNs and the output layer are assembled
  from the concatenated chunks (same flat layout as HyperMscaleDeepONet).
- The per-scale frequency factors and the global scale are FIXED constants
  (registered buffers, not learned): analysis of trained HyperMscaleDeepONet
  showed the branch learns sample-independent scales, so the adaptive
  pathway is dropped.
- No learned parameters in the trunk itself — trunk weights/biases are all
  branch-generated at runtime.
"""

import math

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


class c_HyperMscaleDeepONet(nn.Module):
    """HyperMscaleDeepONet whose branch output is produced in K chunks.

    Args:
        branch_dim:   Input dimension of the branch net (sensor values).
        trunk_dim:    Input dimension of the trunk net (coordinates).
        hidden_dim:   Width of the branch net hidden layers.
        trunk_hidden: Width of each per-scale trunk FNN hidden layers.
        num_outputs:  Number of output channels.
        depth:        Number of hidden layers in the branch net.
        trunk_depth:  Number of hidden layers in each per-scale trunk FNN.
        scales:       Fixed per-scale frequency factors (registered as buffers,
                      never trained).
        activation:   'GELU' or 'Tanh' (branch activation; trunk always uses B-spline).
        basis_size:   Number of trunk basis functions (= trunk feature dim,
                      n_scales * out_dim). Sets each sub-network's OUTPUT width
                      (basis_size // n_scales); hidden width stays trunk_hidden.
                      Must be divisible by n_scales.
        chunk_in:     Dim of each learnable latent chunk code.
        chunk_out:    Params emitted per chunk; K = ceil(param_size / chunk_out).
    """
    def __init__(self, branch_dim=3, trunk_dim=2, hidden_dim=128, trunk_hidden=128,
                 num_outputs=4, depth=4, trunk_depth=4, scales=None,
                 activation='GELU', basis_size=None, chunk_in=512, chunk_out=8192):
        super().__init__()

        if activation == 'GELU':
            act = nn.GELU()
        elif activation == 'Tanh':
            act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if scales is None:
            scales = [1.0, 2.0, 4.0, 8.0]
        n_scales = len(scales)

        out_dim = trunk_hidden
        if basis_size is not None:
            if basis_size % n_scales != 0:
                raise ValueError(
                    f"basis_size {basis_size} must be divisible by n_scales {n_scales}"
                )
            out_dim = basis_size // n_scales

        self.num_outputs = num_outputs
        self._n_scales = n_scales

        # --- Flat parameter layout, identical to HyperMscaleDeepONet (no tail) ---
        scale_dims = [trunk_dim] + [trunk_hidden] * trunk_depth + [out_dim]
        trunk_feat_dim = n_scales * out_dim
        output_dims = [trunk_feat_dim, num_outputs]
        self._scale_dims = scale_dims
        self._output_dims = output_dims

        self.param_size = (n_scales * _compute_weight_bias(scale_dims)
                           + _compute_weight_bias(output_dims))

        # --- Chunked generation: shared branch + per-chunk latent codes ---
        self.chunk_in = chunk_in
        self.chunk_out = chunk_out
        self.num_chunks = math.ceil(self.param_size / chunk_out)
        self.latent_chunk = nn.Parameter(torch.randn(self.num_chunks, chunk_in))

        branch_dims = [branch_dim + chunk_in] + [hidden_dim] * depth + [chunk_out]
        self.branch_net = _MLP(branch_dims, act)
        # Zero-init the branch output weights so generated trunk weight matrices
        # start at 0 (no amplitude blow-up). Keep the bias (default small): it
        # seeds nonzero trunk activations so gradients bootstrap through the
        # zeroed layers -- zeroing bias too would deadlock the trunk weights.
        nn.init.zeros_(self.branch_net.net[-1].weight)

        # --- Fixed scales (buffers, not trained) ---
        self.register_buffer('per_scale', torch.tensor(scales, dtype=torch.float32))
        self.register_buffer('global_scale', torch.ones(1))

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

        params: [B, param_size] — flattened chunked trunk weights/biases.
        x_trunk: [N, trunk_dim] or [B, N, trunk_dim]
        """
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0).expand(params.shape[0], -1, -1)
        B = params.shape[0]

        # Apply fixed global scale to input coordinates
        y = self.global_scale * x_trunk  # [B, N, trunk_dim]

        # --- Per-scale trunk FNNs, outputs stacked (no fusion) ---
        start = 0
        outs = []
        for s in range(self._n_scales):
            y_s = y * self.per_scale[s]  # fixed factor, broadcast
            for i in range(len(self._scale_dims) - 1):
                d_in = self._scale_dims[i]
                d_out = self._scale_dims[i + 1]
                y_s, start = self._apply_layer(params, y_s, d_in, d_out, start,
                                               act_fn=_phi)
            outs.append(y_s)
        y = torch.cat(outs, dim=-1)  # [B, N, n_scales * trunk_hidden]

        # --- Output layer (no activation) ---
        d_oin, d_oout = self._output_dims[0], self._output_dims[1]
        y, _ = self._apply_layer(params, y, d_oin, d_oout, start, act_fn=None)
        return y  # [B, N, num_outputs]

    def forward(self, x_branch, x_trunk):
        """
        Args:
            x_branch: [Batch, branch_dim]  sensor values
            x_trunk:  [N_points, trunk_dim] or [Batch, N_points, trunk_dim]

        Returns:
            [Batch, N_points, num_outputs]
        """
        B, K = x_branch.shape[0], self.num_chunks
        xb = x_branch.unsqueeze(1).repeat(1, K, 1)          # [B, K, branch_dim]
        z = self.latent_chunk.unsqueeze(0).expand(B, -1, -1)  # [B, K, chunk_in]

        params = self.branch_net(torch.cat([xb, z], dim=-1))  # [B, K, chunk_out]
        params = params.reshape(B, -1)[:, :self.param_size]   # [B, param_size]

        return self._trunk_forward(params, x_trunk)


if __name__ == '__main__':
    model = c_HyperMscaleDeepONet(hidden_dim=128, depth=4, trunk_hidden=128,
                                  trunk_depth=4, basis_size=512,
                                  chunk_in=512, chunk_out=8192)
    n_params = sum(p.numel() for p in model.parameters())
    xb = torch.randn(8, 3)
    xt = torch.rand(6528, 2)
    out = model(xb, xt)
    assert out.shape == (8, 6528, 4), out.shape
    out.sum().backward()  # grads flow through the chunked branch
    assert model.latent_chunk.grad is not None
    assert not model.per_scale.requires_grad and not model.global_scale.requires_grad
    print(f"K={model.num_chunks} chunks, params={n_params/1e6:.2f}M, "
          f"branch={sum(p.numel() for p in model.branch_net.parameters())/1e6:.2f}M, "
          f"out={tuple(out.shape)}")
