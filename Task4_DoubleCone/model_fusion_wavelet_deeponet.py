import torch
import torch.nn as nn

from model_wavelet_deeponet import GaborActivation, ComplexLinear, _ComplexLN

# Hybrid of Fusion_DeepONet and WaveletDeepONet: the fusion branch (LayerNorm
# + scalar modulation ab/cb, cumulative skip; no sine term) is kept unchanged;
# the trunk's Linear -> LN -> Act stack becomes its complex
# form ComplexLinear -> ComplexLN -> Gabor(w0, s0), where the activation is
# the continuous Gabor wavelet psi(z) = exp(i*w0*z) * exp(-|s0*z|^2). The
# fusion gate (elementwise product with the branch skip features) broadcasts
# the real skip over the complex trunk states; the modulation scalars of the
# original GELU+sin fusion trunk (a1b/F1b/c1b on the branch, at/ct/a1t/F1t/
# c1t on the trunk) are dropped. The real
# part of the final complex trunk output feeds the usual DeepONet inner
# product.


class Fusion_Wavelet_DeepONet(nn.Module):
    """Fusion_DeepONet whose trunk activation is the complex Gabor wavelet.

    Args:
        branch_dim: 3 (Ma, T, Re) — flow params, constant per sample
        trunk_dim: 2 (x, y) — shared grid coordinates
        hidden_dim: branch width, must equal trunk_hidden — the fusion gate
            is elementwise ('bk,bik'), so the two widths cannot differ (259)
        trunk_hidden: complex trunk width (259; one complex param = 2 real dof)
        num_outputs: 4 (physics quantities)
        depth: hidden layers per side (5)
        basis_size: trunk feature dim = inner-product width (128)
        w0, s0: Gabor wavelet frequency / dilation (fixed, not trained)
        activation: branch activation ('GELU' or 'Tanh'); the branch has no
            sine second term (unlike Fusion_DeepONet)
    """
    def __init__(self, branch_dim=3, trunk_dim=2, hidden_dim=259,
                 trunk_hidden=259, num_outputs=4, depth=5, basis_size=128,
                 w0=3.0, s0=3.0, activation='GELU'):
        if trunk_hidden != hidden_dim:
            raise ValueError(f"trunk_hidden ({trunk_hidden}) must equal hidden_dim "
                             f"({hidden_dim}): the fusion gate is elementwise.")
        super().__init__()

        if activation == 'GELU':
            self.act = nn.GELU()
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.act_t = GaborActivation(w0, s0)

        self.L = depth + 1

        # Branch-side scalar modulation (sine-term scalars a1b/F1b/c1b and the
        # trunk-side set are dropped)
        self.ab  = nn.Parameter(torch.full((self.L,), 0.1))
        self.cb  = nn.Parameter(torch.full((self.L,), 0.1))

        # Branch Net (same as Fusion_DeepONet; final Linear emits
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

        # Per-hidden-layer LayerNorm on the linear pre-activation z (real LN
        # on the branch, joint-statistics ComplexLN on the complex trunk),
        # matching the Linear -> LN -> Act layout of Fusion_DeepONet
        self.ln_b = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(self.L - 1)])
        self.ln_t = nn.ModuleList(
            [_ComplexLN(trunk_hidden) for _ in range(self.L - 1)])

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

        for i in range(self.L - 1):
            z_b = self.ln_b[i](self.branch_net[i](x_branch))
            x_branch = self.act(10 * self.ab[i] * z_b + self.cb[i])
            skip.append(x_branch)

        # Cumulative skip accumulation
        for i in range(1, self.L - 1):
            skip[i] = skip[i - 1] + skip[i]

        for i in range(self.L - 1):
            z_t = self.ln_t[i](self.trunk_net[i](x_trunk))
            x_trunk = self.act_t(z_t)

            # Fusion gate: branch skip features gate the complex trunk states
            # channel-wise (real gate broadcasts over real/imag parts)
            x_trunk = torch.einsum('bk,bik->bik', skip[i], x_trunk)

        x_branch = self.branch_net[-1](x_branch)
        x_trunk = self.trunk_net[-1](x_trunk).real

        B_out_reshaped = x_branch.view(-1, self.num_outputs, self.basis_size)
        # [Batch, num_outputs, basis_size]

        prediction = torch.einsum("bnk, bik -> bin", B_out_reshaped, x_trunk)
        # [Batch, N_points, num_outputs]

        return prediction


if __name__ == "__main__":
    m = Fusion_Wavelet_DeepONet()
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
