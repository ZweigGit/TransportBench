"""
HyperWaveletDeepONet: HyperMscaleDeepONet whose trunk is the complex Gabor FNN
of WaveletDeepONet.

Branch net outputs all trunk parameters — complex weights and biases (real and
imaginary parts packed in the flat output vector) plus the Gabor wavelet's w0
and s0 — so no learned parameters live in the trunk; the trunk FNN itself maps
coordinates directly to the num_outputs channels (no basis expansion).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_weight_bias(dims):
    """Total real dof for a complex linear stack of given dims (2 dof per complex param)."""
    total = 0
    for i in range(len(dims) - 1):
        total += 2 * (dims[i] * dims[i + 1] + dims[i + 1])
    return total


class _MLP(nn.Module):
    """Fully-connected stack: Linear -> GELU -> ... -> Linear (branch net)."""
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HyperWaveletDeepONet(nn.Module):
    """HyperDeepONet with a complex Gabor-wavelet trunk — branch net outputs all
    trunk parameters.

    Branch net outputs one complex trunk FNN (real/imag parts of every weight
    and bias) plus per-sample w0 and s0 for the Gabor activation
    psi(z) = exp(i*w0*z) * exp(-|s0*z|^2). The real part of the complex trunk
    output is the prediction; no learned parameters in the trunk.

    Args:
        branch_dim:   Input dimension of the branch net (sensor values).
        trunk_dim:    Input dimension of the trunk net (coordinates).
        branch_hidden / branch_depth: Branch net width / hidden layers.
        trunk_hidden / trunk_depth:   Trunk FNN width / hidden layers.
        num_outputs:  Number of output channels.
        w0, s0:       Gabor wavelet frequency / dilation init values (init 1.0).
                      Branch outputs are softplus-anchored so the decoded values
                      stay > 0 during training and a zero output decodes to 1.0.
    """
    def __init__(self, branch_dim=674, trunk_dim=2, branch_hidden=63,
                 trunk_hidden=48, branch_depth=4, trunk_depth=4,
                 num_outputs=4, w0=1.0, s0=1.0):
        super().__init__()

        self.num_outputs = num_outputs
        self._w0_default = math.log(math.expm1(w0))  # softplus inverse
        self._s0_default = math.log(math.expm1(s0))

        # --- Compute total parameters needed to construct the hyper-trunk ---
        # Trunk FNN: [trunk_dim] + [trunk_hidden] * trunk_depth
        trunk_dims = [trunk_dim] + [trunk_hidden] * trunk_depth

        # Output layer: trunk_hidden -> num_outputs
        output_dims = [trunk_hidden, num_outputs]

        t_para = _compute_weight_bias(trunk_dims) + _compute_weight_bias(output_dims) + 2  # +2: w0, s0

        # --- Branch net ---
        self.branch_net = _MLP([branch_dim] + [branch_hidden] * branch_depth + [t_para])

        # --- Stash shapes for trunk forward ---
        self._trunk_dims = trunk_dims
        self._output_dims = output_dims

    @staticmethod
    def _apply_layer(params, x, d_in, d_out, start):
        """Slice complex weight/bias, apply Complex(d_in, d_out), advance start."""
        B = params.shape[0]
        w_sz = d_in * d_out
        weight = torch.complex(params[:, start:start + w_sz].reshape(B, d_out, d_in),
                               params[:, start + w_sz:start + 2 * w_sz].reshape(B, d_out, d_in))
        start += 2 * w_sz
        bias = torch.complex(params[:, start:start + d_out].reshape(B, 1, d_out),
                             params[:, start + d_out:start + 2 * d_out].reshape(B, 1, d_out))
        start += 2 * d_out
        return torch.einsum("bij,bgj->bgi", weight, x) + bias, start

    def _trunk_forward(self, params, x_trunk):
        """Hypernetwork trunk forward using branch-provided weights/biases.

        params: [B, t_para] — flattened complex trunk weights, biases, w0, s0
        x_trunk: [N, trunk_dim] or [B, N, trunk_dim]
        """
        if x_trunk.dim() == 2:
            x_trunk = x_trunk.unsqueeze(0)  # [1, N, trunk_dim]
        B = params.shape[0]

        # --- Extract w0/s0 from end of params; softplus guarantees > 0 ---
        w0 = F.softplus(self._w0_default + params[:, -2])  # [B]
        s0 = F.softplus(self._s0_default + params[:, -1])  # [B]

        y = x_trunk.to(torch.complex64).expand(B, -1, -1)  # [B, N, trunk_dim]

        # --- Trunk FNN: ComplexLinear -> Gabor -> ... ---
        start = 0
        for i in range(len(self._trunk_dims) - 1):
            d_in = self._trunk_dims[i]
            d_out = self._trunk_dims[i + 1]
            y, start = self._apply_layer(params, y, d_in, d_out, start)
            # psi(z) = exp(i*w0*z) * exp(-|s0*z|^2); |s0*z|^2 = s0^2*|z|^2
            env = (s0 ** 2).view(B, 1, 1) * (y.real.square() + y.imag.square())
            y = torch.exp(1j * w0.view(B, 1, 1) * y - env)

        # --- Output layer (no activation), real part as prediction ---
        d_oin, d_oout = self._output_dims[0], self._output_dims[1]
        y, start = self._apply_layer(params, y, d_oin, d_oout, start)
        return y.real  # [B, N, num_outputs]

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


if __name__ == "__main__":
    m = HyperWaveletDeepONet()
    p = sum(x.numel() * (2 if x.is_complex() else 1) for x in m.parameters())
    print(f"real dof: {p / 1e6:.3f} M")
    xb = torch.randn(4, 674)
    xt = torch.rand(100, 2)
    out = m(xb, xt)
    assert out.shape == (4, 100, 4)
    assert torch.isfinite(out).all()
    anchor = math.log(math.e - 1.0)
    assert torch.allclose(F.softplus(anchor + torch.zeros(1)), torch.ones(1))  # zero offset -> 1.0
    with torch.no_grad():  # decoded w0/s0 stay > 0 for any branch output
        p = m.branch_net(xb)
        assert (F.softplus(anchor + p[:, -2]) > 0).all() and (F.softplus(anchor + p[:, -1]) > 0).all()
    out.sum().backward()  # grads reach every branch param
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters())
    print("forward/backward ok", out.shape)
