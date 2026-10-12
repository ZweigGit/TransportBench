import torch
import torch.nn as nn

# Hybrid of Residual_Fusion_DeepONet and WaveletDeepONet (ported from Task1):
# the residual-fusion branch (real Linears + scalar modulation + skip
# accumulation, gate 1+skip) is kept unchanged; the trunk is swapped for the
# wavelet trunk — a complex FNN whose hidden activation is the continuous
# Gabor wavelet psi(z) = exp(i*w0*z) * exp(-|s0*z|^2). The real (1+skip) gate
# broadcasts over the complex trunk states; the real part of the final complex
# output is the trunk feature vector for the usual DeepONet inner product.


class sin_act(nn.Module):
    def forward(self, x):
        return torch.sin(x)


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
        return torch.nn.functional.linear(x, self.weight, self.bias)


class Residual_Fusion_Wavelet_DeepONet(nn.Module):
    """Residual_Fusion_DeepONet whose trunk net is the WaveletDeepONet Gabor
    trunk. Branch side (Linears + scalar modulation ab/cb/a1b/F1b/c1b + skip
    accumulation) is identical to Residual_Fusion_DeepONet; each trunk layer
    is ComplexLinear -> Gabor(w0, s0) followed by the same residual gate
    x_trunk = x_trunk * (1.0 + skip[i]). The trunk-side modulation scalars of
    the original (at/ct/a1t/F1t/c1t, built for the GELU+sin trunk) are
    dropped: the wavelet trunk has none.

    Args:
        branch_dim: 3 (Ma, T, Re) — flow params, constant per sample
        trunk_dim: 2 (x, y) — shared grid coordinates
        hidden_dim: branch width, must equal trunk_hidden — the residual gate
            is elementwise ('bk,bik'), so the two widths cannot differ (259)
        trunk_hidden: complex trunk width (259; one complex param = 2 real dof)
        num_outputs: 4 (physics quantities)
        depth: hidden layers per side (5)
        basis_size: trunk feature dim = inner-product width (128)
        w0, s0: Gabor wavelet frequency / dilation (fixed, not trained)
        activation: branch activation ('GELU' or 'Tanh')
    """
    def __init__(self, branch_dim=3, trunk_dim=2, hidden_dim=259,
                 trunk_hidden=259, num_outputs=4, depth=5, basis_size=128,
                 w0=3.0, s0=3.0, activation='GELU'):
        if trunk_hidden != hidden_dim:
            raise ValueError(f"trunk_hidden ({trunk_hidden}) must equal hidden_dim "
                             f"({hidden_dim}): the residual gate is elementwise.")
        super().__init__()

        if activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.act2 = sin_act()
        self.act_t = GaborActivation(w0, s0)

        self.L = depth + 1

        # Branch-side scalar modulation (trunk side dropped with the GELU+sin trunk)
        self.ab  = nn.Parameter(torch.full((self.L,), 0.1))
        self.cb  = nn.Parameter(torch.full((self.L,), 0.1))
        self.a1b = nn.Parameter(torch.zeros(self.L,))
        self.F1b = nn.Parameter(torch.full((self.L,), 0.1))
        self.c1b = nn.Parameter(torch.zeros(self.L,))

        # Branch Net (same as Residual_Fusion_DeepONet; final Linear emits
        # num_outputs * basis_size coefficients)
        self.branch_net = nn.ModuleList()
        self.branch_net.append(nn.Linear(branch_dim, hidden_dim))
        for _ in range(depth - 1):
            self.branch_net.append(nn.Linear(hidden_dim, hidden_dim))
        self.branch_net.append(nn.Linear(hidden_dim, num_outputs * basis_size))

        # Trunk Net: [ComplexLinear1, ..., Last ComplexLinear]
        self.trunk_net = nn.ModuleList()
        self.trunk_net.append(ComplexLinear(trunk_dim, trunk_hidden))
        for _ in range(depth - 1):
            self.trunk_net.append(ComplexLinear(trunk_hidden, trunk_hidden))
        self.trunk_net.append(ComplexLinear(trunk_hidden, basis_size))

        self.num_outputs = num_outputs
        self.basis_size = basis_size

    def forward(self, x_branch, x_trunk):

        """Forward pass.

        Args:
            x_branch: [Batch, branch_dim]
            x_trunk:  [N_points, trunk_dim] (shared grid coords)

        Returns:
            [Batch, N_points, num_outputs]
        """
        if x_trunk.dim() == 2:
            # Shared 2D trunk: [N, trunk_dim] -> [B, N, trunk_dim] (view, no copy)
            B = x_branch.shape[0]
            x_trunk = x_trunk.unsqueeze(0).expand(B, -1, -1)
        x_trunk = x_trunk.to(torch.complex64)

        skip = []

        for i in range(self.L-1):

            z_b = self.branch_net[i](x_branch)
            x_branch = self.act(10*self.ab[i]*z_b+self.cb[i])+\
            10*self.a1b[i]*self.act2(10*self.F1b[i]*z_b+self.c1b[i])
            skip.append(x_branch)

        for i in range(1,self.L-1):
            skip[i] = skip[i-1]+skip[i]

        for i in range(self.L-1):

            z_t = self.trunk_net[i](x_trunk)
            x_trunk = self.act_t(z_t)

            # Residual gate: identity path preserves trunk magnitude/gradient
            x_trunk = torch.einsum('bk,bik->bik', 1.0 + skip[i], x_trunk)

        x_branch = self.branch_net[-1](x_branch)
        x_trunk = self.trunk_net[-1](x_trunk).real

        B_out_reshaped = x_branch.view(-1, self.num_outputs, self.basis_size)
        # [Batch, num_outputs, basis_size]

        prediction = torch.einsum("bnk, bik -> bin", B_out_reshaped, x_trunk)
        # [Batch, N_points, num_outputs]

        return prediction


if __name__ == "__main__":
    m = Residual_Fusion_Wavelet_DeepONet()
    p = sum(x.numel() * (2 if x.is_complex() else 1) for x in m.parameters())
    print(f"real dof: {p / 1e6:.3f} M")
    xb = torch.randn(4, 3)
    xt = torch.rand(100, 2)
    out = m(xb, xt)
    assert out.shape == (4, 100, 4)
    print("forward ok", out.shape)

    # one backward pass: gradients reach the complex trunk params
    out.sum().backward()
    g = m.trunk_net[0].weight.grad
    assert g is not None and torch.is_complex(g) and torch.isfinite(g).all()
    print("backward ok (trunk grad dtype", g.dtype, ")")
