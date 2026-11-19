"""
WIEN-Serial-DeepONet for Task1: per-scale branch-trunk PAIRS trained in
sequence, coarse -> fine.

Motivation (user's design): in the shared-branch WIEN-DeepONet, staged
(per-level) training is ill-defined -- freezing the coarser trunks without
freezing the shared branch does not actually freeze their outputs, so the
c2f conditioning the finer levels see keeps drifting. The fix here is to
give each level its OWN branch sub-network (hidden width following the
trunk's 4x ladder, shared 674-dim input), so a level is a self-contained
(branch_l, trunk_l) pair: train the pair, freeze it, move to the next
level. After stage l finishes, NOTHING upstream of level l changes, so the
coarse estimates fed to finer levels are exactly the functions that will be
used at inference -- staged training is fully consistent with deployment.

Architecture per level l (identical building blocks to WIENDeepONet):
  branch_l: GELU MLP [674, width_l x branch_layers, o_l*(basis+1)]
            (hidden width follows trunk_widths[l]; output = the level's
            latent code: o_l weight rows x basis + o_l biases)
  trunk_l:  SIREN (learnable positive omega, init 30) on the [-1,1] mgrid
            of its coefficient shape (+ coarse-level prediction when c2f)
  pred_l = sum(trunk_l_features * w_l) + b_l  (the DeepONet dot product)

Training protocol (staged, in train.py): stage l trains (branch_l, trunk_l)
only; all earlier pairs are frozen (requires_grad=False) and their outputs
are detached. The stage schedule and epoch budget are set in train.py.

The c2f switch, coordinate mapping, pointwise query_field, harmonic fill
and the train-set coefficient cache are reused verbatim from
model_wien_deeponet (imported, not copied).
"""

import torch
import torch.nn as nn

from model_wien_deeponet import (
    OUT_DIMS, LEVEL_SHAPES_128, SineLayer, LevelTrunk, get_mgrid,
    build_or_load_coef_cache, harmonic_fill,
)


def _mlp_gelu(dims):
    """Linear -> GELU -> ... -> Linear (no activation on the output)."""
    acts = [nn.GELU()] * (len(dims) - 2)
    return nn.Sequential(*[
        m for pair in zip(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)],
            acts + [nn.Identity()]) for m in pair])


class WIENSerialDeepONet(nn.Module):
    """Per-scale (branch, trunk) pairs, trained coarse -> fine in sequence.

    forward(x_branch) -> per-level normalized coefficient predictions
        preds[l] of shape (B, P_l, o_l), same contract as WIENDeepONet.
    predict_field / query_field / phys_to_norm / norm_to_phys /
    load_coeff_stats: same semantics as WIENDeepONet.
    """

    def __init__(self, branch_dim=674, basis=32, trunk_widths=(11, 28, 92, 252),
                 trunk_layers=4, wavelet="db4", J=3, mode="symmetric",
                 grid_shapes=LEVEL_SHAPES_128, first_omega=30.0, hidden_omega=30.0,
                 activation="siren", use_c2f=True, learnable_omega=False,
                 branch_widths=None,
                 xmin=-2.0, xmax=4.0, ymin=-3.0, ymax=3.0):
        super().__init__()
        assert len(trunk_widths) == J + 1
        self.J = J
        self.n_levels = J + 1
        self.basis = basis
        self.out_dims = OUT_DIMS
        self.wavelet_cfg = dict(wave=wavelet, J=J, mode=mode)
        self.grid_shapes = tuple(tuple(s) for s in grid_shapes)

        self.register_buffer("phys_min", torch.tensor([xmin, ymin]))
        self.register_buffer("phys_span", torch.tensor(
            [xmax - xmin, ymax - ymin]))

        # per-level trunk input dims (c2f: coords + previous level's output)
        in_dims = [2]
        for l in range(1, self.n_levels):
            in_dims.append(2 + self.out_dims[l - 1] if use_c2f else 2)
        self.use_c2f = use_c2f
        self.trunks = nn.ModuleList([
            LevelTrunk(in_dims[l], self.out_dims[l] * basis, trunk_widths[l],
                       trunk_layers, first_omega, hidden_omega, activation,
                       learnable_omega=learnable_omega)
            for l in range(self.n_levels)
        ])

        # per-level branch sub-networks; SAME hidden width as the level's
        # trunk (user constraint), so one width ladder drives both. With
        # widths (11, 28, 92, 252) each (branch_l, trunk_l) pair's parameter
        # count is a near-exact 4x ladder -- 11.4k / 46.6k / 187.4k / 753.8k
        # per level (ratios 4.10/4.02/4.02), 999,070 total (fixed omega).
        if branch_widths is None:
            branch_widths = trunk_widths
        self.branch_nets = nn.ModuleList()
        for l in range(self.n_levels):
            o = self.out_dims[l]
            dims = [branch_dim] + [branch_widths[l]] * trunk_layers + \
                [o * (basis + 1)]
            self.branch_nets.append(_mlp_gelu(dims))

        for l, sh in enumerate(self.grid_shapes):
            self.register_buffer(f"grid_{l}", get_mgrid(sh))

        o_max = max(self.out_dims)
        self.register_buffer("coef_mean", torch.zeros(self.n_levels, o_max))
        self.register_buffer("coef_maxx", torch.ones(self.n_levels, o_max))

        self._ifm_cache = {}

    # ------------------------------------------------------------------ #
    def load_coeff_stats(self, cache):
        for l in range(self.n_levels):
            o = self.out_dims[l]
            self.coef_mean[l, :o] = cache["mean"][l]
            self.coef_maxx[l, :o] = cache["maxx"][l]

    def phys_to_norm(self, xy):
        return (xy - self.phys_min) / self.phys_span * 2.0 - 1.0

    def norm_to_phys(self, uv):
        return (uv + 1.0) / 2.0 * self.phys_span + self.phys_min

    def forward(self, x_branch):
        """Per-level normalized coefficient predictions (same contract as
        WIENDeepONet.forward). Frozen-stage behavior is decided by the
        training loop via requires_grad / detach; forward itself just runs
        every level (unused levels cost nothing extra at inference)."""
        B = x_branch.shape[0]
        preds_own = [None] * self.n_levels
        if not self.use_c2f:
            for l in range(self.n_levels):
                o = self.out_dims[l]
                lat = self.branch_nets[l](x_branch)          # (B, o*(basis+1))
                w = lat[:, :o * self.basis].view(B, 1, o, self.basis)
                b = lat[:, o * self.basis:].view(B, 1, o)
                grid = getattr(self, f"grid_{l}")
                t = self.trunks[l](grid.unsqueeze(0).expand(B, -1, -1))
                preds_own[l] = (t.view(B, t.shape[1], o, self.basis) * w).sum(-1) + b
            return preds_own
        coarse = None
        for l in range(self.n_levels):
            o = self.out_dims[l]
            lat = self.branch_nets[l](x_branch)
            w = lat[:, :o * self.basis].view(B, 1, o, self.basis)
            b = lat[:, o * self.basis:].view(B, 1, o)
            new_coarse = []
            for g in range(l, self.n_levels):
                grid = getattr(self, f"grid_{g}")
                if l == 0:
                    inp = grid.unsqueeze(0).expand(B, -1, -1)
                else:
                    inp = torch.cat(
                        [grid.unsqueeze(0).expand(B, -1, -1), coarse[g - l + 1]],
                        dim=-1)
                t = self.trunks[l](inp)
                new_coarse.append(
                    (t.view(B, t.shape[1], o, self.basis) * w).sum(-1) + b)
            preds_own[l] = new_coarse[0]
            coarse = new_coarse
        return preds_own

    def _denorm(self, pred, l):
        o = self.out_dims[l]
        return (pred * self.coef_maxx[l, :o].view(1, 1, o)
                + self.coef_mean[l, :o].view(1, 1, o))

    def _get_ifm(self, device):
        key = str(device)
        if key not in self._ifm_cache:
            from pytorch_wavelets import DWTInverse
            self._ifm_cache[key] = DWTInverse(
                wave=self.wavelet_cfg["wave"], mode=self.wavelet_cfg["mode"]
            ).to(device)
        return self._ifm_cache[key]

    def predict_field(self, x_branch):
        """IDWT reconstruction of the predicted coefficients (B, 4, 128, 128)."""
        B = x_branch.shape[0]
        preds = self.forward(x_branch)

        def to_band(l):
            d = self._denorm(preds[l], l)
            h, wd = self.grid_shapes[l]
            if l == 0:
                return d.view(B, h, wd, 4).permute(0, 3, 1, 2)
            return d.view(B, h, wd, 3, 4).permute(0, 4, 3, 1, 2)

        Yl = to_band(0)
        Yh = [to_band(l) for l in range(self.J, 0, -1)]
        return self._get_ifm(x_branch.device)((Yl, Yh))

    def query_field(self, x_branch, phys_xy):
        """Flow values at arbitrary physical coordinates (bilinear on the
        IDWT grid); same contract as WIENDeepONet.query_field."""
        field = self.predict_field(x_branch)
        B = field.shape[0]
        uv = self.phys_to_norm(phys_xy)
        flat = uv.reshape(1, -1, 2).expand(B, -1, -1)
        grid = flat.unsqueeze(2)
        vals = nn.functional.grid_sample(field, grid, mode="bilinear",
                                         align_corners=True).squeeze(2)
        if B == 1:
            return vals[0].T.reshape(*uv.shape[:-1], 4)
        return vals.permute(0, 2, 1).reshape(B, *uv.shape[:-1], 4)

    # --- staged-training helpers -------------------------------------- #
    def stage_params(self, stage):
        """Trainable parameters of stage `stage` (0 = low-freq pair, ...);
        everything else is frozen in place."""
        assert 0 <= stage < self.n_levels
        return list(self.branch_nets[stage].parameters()) + \
            list(self.trunks[stage].parameters())

    def begin_stage(self, stage):
        """Freeze all pairs except `stage`; re-enable everything first so
        repeated calls are idempotent."""
        for l in range(self.n_levels):
            for p in self.branch_nets[l].parameters():
                p.requires_grad_(l == stage)
            for p in self.trunks[l].parameters():
                p.requires_grad_(l == stage)

    def eval_mode_frozen(self):
        """Convenience: unfreeze nothing, just a no-op placeholder for
        symmetry with begin_stage (kept for API clarity)."""
        pass


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. Parameter budget per level and total
    m = WIENSerialDeepONet()
    per_level = []
    for l in range(m.n_levels):
        bp = sum(p.numel() for p in m.branch_nets[l].parameters())
        tp = sum(p.numel() for p in m.trunks[l].parameters())
        per_level.append(bp + tp)
        print(f"level {l}: branch {bp:,} + trunk {tp:,} = {bp + tp:,}")
    total = sum(p.numel() for p in m.parameters())
    print(f"total: {total:,}")
    assert per_level == [11370, 46588, 187356, 753756], per_level
    assert total == 999070, total

    # 2. Forward / predict_field / query_field shapes
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(dev)
    xb = torch.randn(2, 674, device=dev)
    preds = m(xb)
    assert [tuple(p.shape) for p in preds] == [
        (2, 484, 4), (2, 484, 12), (2, 1369, 12), (2, 4489, 12)]
    f = m.predict_field(xb)
    assert f.shape == (2, 4, 128, 128)
    xy = torch.tensor([[2.0, 2.0], [-2.0, -3.0]], device=dev)
    q = m.query_field(xb[:1], xy)
    assert q.shape == (2, 4)
    print("forward / predict_field / query_field OK")

    # 3. Staged freezing: begin_stage(2) -> only pair 2 trainable
    m.begin_stage(2)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert all(("branch_nets.2." in n or "trunks.2." in n) for n in trainable)
    assert len(trainable) > 0
    print(f"begin_stage(2): {len(trainable)} trainable tensors, all in pair 2")

    # 4. Staged loss flow: loss at level 2 only reaches pair-2 params
    out = m(xb)
    loss = nn.functional.mse_loss(out[2], torch.randn_like(out[2]))
    loss.backward()
    staged = [n for n, p in m.named_parameters()
              if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    frozen_grad = [n for n, p in m.named_parameters()
                   if not p.requires_grad and p.grad is not None]
    assert not frozen_grad, frozen_grad[:3]
    print(f"level-2 loss reaches only pair-2 params ({len(staged)} tensors)")

    # 5. noc2f variant: default is FIXED omega; learnable_omega=True still works
    m_nc = WIENSerialDeepONet(use_c2f=False)
    omegas = [l for t in m_nc.trunks for l in t.net
              if isinstance(l, SineLayer) and l.learnable_omega]
    assert len(omegas) == 0  # default: fixed omega 30
    m_lw = WIENSerialDeepONet(use_c2f=False, learnable_omega=True)
    omegas_lw = [l for t in m_lw.trunks for l in t.net
                 if isinstance(l, SineLayer) and l.learnable_omega]
    assert len(omegas_lw) == 16
    ps = m_nc(torch.randn(2, 674))
    assert [tuple(p.shape) for p in ps] == [
        (2, 484, 4), (2, 484, 12), (2, 1369, 12), (2, 4489, 12)]
    print("noc2f: fixed omega default + learnable-omega option OK; "
          "ALL SELF-CHECKS PASSED")
