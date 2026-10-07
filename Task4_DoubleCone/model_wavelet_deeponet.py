import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ported from Multi-Resolution-WNN experiments/18_gabor_fnn_f1.jl (FNN-Gabor):
# the trunk is a complex FNN whose hidden activation is the continuous Gabor
# wavelet psi(z) = exp(i*w0*z) * exp(-|s0*z|^2) applied to the complex
# pre-activation. Complex weights/biases (one complex param = 2 real dof).
# The complex trunk feature vector is reduced to its real part before the
# inner product with the (real) branch coefficients.


class GaborActivation(nn.Module):
    """Complex Gabor wavelet psi(z) = exp(i*w0*z) * exp(-|s0*z|^2)."""
    def __init__(self, w0=1.0, s0=1.0):
        super().__init__()
        self.w0 = w0
        self.s0 = s0

    def forward(self, z):
        # |s0*z|^2 = s0^2 * |z|^2, computed without abs() (undefined grad at 0)
        env = (self.s0 ** 2) * (z.real.square() + z.imag.square())
        return torch.exp(1j * self.w0 * z - env)


class ComplexLinear(nn.Module):
    """Linear layer with complex weights (Julia gabor_init: real/imag each
    randn*sqrt(1/(2*fan_in)) so E|W|^2 = 1/fan_in; bias = 0.1 * complex noise)."""
    def __init__(self, in_features, out_features):
        super().__init__()
        g = (1.0 / (2 * in_features)) ** 0.5
        weight = torch.complex(torch.randn(out_features, in_features) * g,
                               torch.randn(out_features, in_features) * g)
        bias = torch.complex(torch.randn(out_features) * 0.1,
                             torch.randn(out_features) * 0.1)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class _GaborTrunk(nn.Module):
    """Complex FNN trunk: ComplexLinear -> Gabor -> ... -> ComplexLinear,
    real part of the complex output is the trunk feature vector."""
    def __init__(self, trunk_dim, hidden_dim, depth, out_dim, w0, s0):
        super().__init__()
        dims = [trunk_dim] + [hidden_dim] * depth + [out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(ComplexLinear(dims[i], dims[i + 1]))
            layers.append(GaborActivation(w0, s0))
        layers.append(ComplexLinear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.to(torch.complex64)).real


class _FNN(nn.Module):
    """Real FNN: Linear -> LN -> GELU -> ... -> Linear (branch, like MscaleDeepONet)."""
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WaveletDeepONet(nn.Module):
    """DeepONet whose trunk activation is the complex Gabor wavelet
    psi(z) = exp(i*w0*z) * exp(-|s0*z|^2) (w0 = omega_0, s0 = s_0 fixed
    hyperparameters, not trained). The trunk feature vector is the real part
    of the complex trunk output; the dot product with the branch coefficients
    is the usual DeepONet basis expansion.

    Args:
        branch_dim:   Input dimension of the branch net (flow params).
        trunk_dim:    Input dimension of the trunk net (coordinates).
        branch_hidden / branch_depth: Branch net width / hidden layers.
        trunk_hidden / trunk_depth:   Trunk net width / hidden layers.
        basis_size:   Trunk feature dim (= inner-product width).
        num_outputs:  Output channels.
        w0, s0:       Gabor wavelet frequency / dilation (default 1.0 / 1.0).
    """
    def __init__(self, branch_dim=3, trunk_dim=2, branch_hidden=300,
                 trunk_hidden=192, branch_depth=5, trunk_depth=5,
                 basis_size=192, num_outputs=4, w0=1.0, s0=1.0):
        super().__init__()
        self.trunk_net = _GaborTrunk(trunk_dim, trunk_hidden, trunk_depth,
                                     basis_size, w0, s0)
        self.trunk_feat_dim = basis_size

        branch_dims = ([branch_dim] + [branch_hidden] * branch_depth
                       + [num_outputs * basis_size])
        self.branch_net = _FNN(branch_dims)

        self.num_outputs = num_outputs
        self.bias = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, x_branch, x_trunk):
        """Forward pass.

        Args:
            x_branch: [Batch, branch_dim]
            x_trunk:  [N_points, trunk_dim] (shared grid coords)

        Returns:
            [Batch, N_points, num_outputs]
        """
        B = x_branch.shape[0]
        b = self.branch_net(x_branch).view(B, self.num_outputs, self.trunk_feat_dim)
        t = self.trunk_net(x_trunk)                       # [N, basis_size]
        t = t.unsqueeze(0).expand(B, -1, -1)              # [B, N, basis_size]
        pred = torch.einsum("bkh, bnh -> bnk", b, t)      # [B, N, num_outputs]
        return pred + self.bias.view(1, 1, -1)
