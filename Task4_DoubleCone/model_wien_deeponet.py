"""WIEN-DeepONet DeepONet: coefficient-domain coarse-to-fine operator model.

Paper-faithful data flow (redesign spec):
- ONE DWT: the encoded target is harmonically filled over its vacuum cells
  first (no artificial step at the body), then decomposed once per batch
  into per-scale coefficient targets. The training loss is entirely
  coefficient-domain per-band relative MSE (energy-normalized, masked to
  coefficients whose whole support is fluid); no field-domain loss enters
  training (coef_loss_only).
- Every module predicts WAVELET COEFFICIENTS at its own scale: CoarseNet
  -> A_J, Enh_j -> D_j = (LH, HL, HH). The coarse-to-fine signal is NOT a
  reconstruction recursion: each module's trunk is a coordinate function,
  re-evaluated at the next FINER grid (upsampled query coordinates), and
  a context head emits a rough prediction of that finer scale's detail
  coefficients, concatenated into the next Enh's detail head. No IDWT
  runs inside the chain.
- ONE IDWT: the predicted bands are assembled once at the end
  (idwt2_stack) into the output field. Training and inference share the
  same single prediction path; ground truth only ever enters losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from wavelet_haar import (dwt2_stack, idwt2_stack, dwt_level, support_masks,
                          harmonic_fill, harmonic_fill_operator)


def _mlp(in_dim, out_dim, hidden, depth):
    """Hidden-layer MLP, GELU + LayerNorm on hidden activations.

    LayerNorm everywhere: the band-split ports without it hit a slow
    bilinear runaway; keep the family convention even though this model
    has no bilinear forms.
    """
    layers = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class BranchNet(nn.Module):
    """Params (B,P) -> latent (B,L)."""

    def __init__(self, branch_dim, latent_dim, hidden_dim, depth=2):
        super().__init__()
        self.net = _mlp(branch_dim, latent_dim, hidden_dim, depth)

    def forward(self, params):
        return self.net(params)


class CoarseNet(nn.Module):
    """Coarsest-band predictor with a context head.

    trunk(z, xy) -> shared features; head_a: A_J approximation
    coefficients; head_ctx: rough D_J on the same grid (D_J lives on the
    A_J grid) to condition Enh_J.
    """

    def __init__(self, latent_dim, num_outputs, hidden_dim, depth=2):
        super().__init__()
        self.trunk = _mlp(latent_dim + 2, hidden_dim, hidden_dim, depth)
        self.head_a = nn.Linear(hidden_dim, num_outputs)
        self.head_ctx = nn.Linear(hidden_dim, 3 * num_outputs)

    def features(self, z, coords):
        B, N, _ = coords.shape
        h = torch.cat([z[:, None, :].expand(B, N, z.shape[1]), coords], dim=-1)
        return self.trunk(h)

    def forward(self, z, coords):
        return self.head_a(self.features(z, coords))


class EnhancementNet(nn.Module):
    """Detail-band net for one scale.

    trunk(z, xy) -> shared features (a coordinate function, queryable at
    ANY grid); detail head consumes [features + ctx] where ctx is the
    previous (coarser) module's rough prediction of THIS scale's
    coefficients; context head consumes the same trunk re-evaluated at
    the next finer grid, emitting a rough prediction of the NEXT scale's
    coefficients -- the coarse-to-fine information. The finest-scale net
    has no next scale and drops the context head.
    """

    def __init__(self, latent_dim, num_outputs, hidden_dim, depth=2,
                 with_ctx_head=True):
        super().__init__()
        self.trunk = _mlp(latent_dim + 2, hidden_dim, hidden_dim, depth)
        self.head_d = nn.Linear(hidden_dim + 3 * num_outputs, 3 * num_outputs)
        if with_ctx_head:
            self.head_ctx = nn.Linear(hidden_dim, 3 * num_outputs)

    def features(self, z, coords):
        B, N, _ = coords.shape
        h = torch.cat([z[:, None, :].expand(B, N, z.shape[1]), coords], dim=-1)
        return self.trunk(h)

    def detail(self, z, coords, ctx):
        """(z, xy_j, rough D_j) -> D_j = (LH,HL,HH), (B,N,3C)."""
        return self.head_d(torch.cat([self.features(z, coords), ctx], dim=-1))

    def context(self, z, coords):
        """Same trunk queried at the next finer grid -> rough D_{j-1}."""
        return self.head_ctx(self.features(z, coords))


class WienDeepONet(nn.Module):
    """Grid-interface model: x [B,2+P,H,W] -> field [B,C,H,W].

    Chain (identical in training and inference; ground truth only enters
    losses, never the chain):
        A_J   = head_a(CoarseNet.trunk(z, xy_J))
        ctx_J = head_ctx(CoarseNet.trunk(z, xy_J))
        D_j   = head_d(Enh_j.trunk(z, xy_j) + ctx_j)     j = J..1
        ctx_j = head_ctx(Enh_j.trunk(z, xy_{j-1}))       j = J..2
        out   = idwt2_stack(A_J, [D_J..D_1])             ONE inverse DWT
    with z = BranchNet(params). Level coordinates are the normalized mesh
    coords avg-pooled down the same ragged pyramid, so every scale sees
    physical positions, not pixel indices.

    forward(x, y=None): with y (encoded target), self.aux_loss carries the
    per-band coefficient relative MSE (energy-normalized, masked to
    coefficients with fully-fluid support; y is harmonically filled before
    its DWT) and coef_loss_only tells the trainer it is the WHOLE training
    loss; the vacuum mask for the fill and those masks comes from
    ctx['vac'] (train only, kept as non-persistent buffers so eval
    rebuilds strict-load cleanly).
    """

    uses_target = True
    coef_loss_only = True
    _lvl_cache = {}

    def __init__(self, branch_dim, num_outputs=4, latent_dim=64, hidden_dim=256,
                 branch_depth=2, depth=2, levels=4, aux_weight=1.0, ctx=None):
        super().__init__()
        ctx = ctx or {}
        # sweepable knobs (levels/aux_weight/wavelet) may arrive via ctx
        self.levels = ctx.get('levels', levels)
        self.num_outputs = num_outputs
        self.aux_weight = ctx.get('aux_weight', aux_weight)
        self.wavelet = ctx.get('wavelet', 'haar')
        self.aux_loss = None
        self.branch = BranchNet(branch_dim, latent_dim, hidden_dim, branch_depth)
        self.coarse = CoarseNet(latent_dim, num_outputs, hidden_dim, depth)
        self.enh = nn.ModuleList(
            EnhancementNet(latent_dim, num_outputs, hidden_dim, depth,
                           with_ctx_head=(j < self.levels - 1))
            for j in range(self.levels))
        vac = (ctx or {}).get('vac')
        if vac is not None:
            self.register_buffer('vac', vac.clone(), persistent=False)
            # pre-factorized Laplace fill for the encoded targets (train only,
            # both non-persistent so eval rebuilds strict-load cleanly)
            self.register_buffer('fill_M', harmonic_fill_operator(vac),
                                 persistent=False)

    def _level_shapes(self, H, W):
        """Per-level A_j shapes of this model's DWT on an (H, W) field,
        finest-first. Queried through the real dwt_level (cached) instead of
        replicating grid math: db-N symmetric bands are slightly larger than
        half size (db4 on 17x384 -> 12x195 -> 9x101), haar is ceil-half."""
        key = (H, W, self.levels, self.wavelet)
        if key not in WienDeepONet._lvl_cache:
            a = torch.zeros(1, 1, H, W)
            shapes = []
            for _ in range(self.levels):
                a = dwt_level(a, self.wavelet)[0]
                shapes.append(tuple(a.shape[-2:]))
            WienDeepONet._lvl_cache[key] = shapes
        return WienDeepONet._lvl_cache[key]

    def _rel_mse(self, pred, targ, w):
        w = w[None, None]
        num = (w * (pred - targ) ** 2).sum()
        den = (w * targ ** 2).sum().clamp_min(1e-8)
        return num / den

    def forward(self, x, y=None, return_bands=False):
        B, C = x.shape[0], self.num_outputs
        z = self.branch(x[:, 2:, 0, 0])

        xc = x[:, 0:2]
        lv = self._level_shapes(*xc.shape[-2:])            # finest-first
        # hw is level-indexed: hw[j] = A_j shape, finest (input) first
        hw = [tuple(xc.shape[-2:])] + lv
        # coordinate maps per level, coarsest-first, pooled onto each
        # level's band grid so every scale sees physical positions
        coords_lv = [F.adaptive_avg_pool2d(xc, s) for s in lv[::-1]]

        def flat(c):
            return c.permute(0, 2, 3, 1).reshape(B, -1, 2)

        # coarsest approximation + rough D_J context on the same grid
        fJ = self.coarse.features(z, flat(coords_lv[0]))
        a = self.coarse.head_a(fJ)                         # (B, N_J, C)
        ctx = self.coarse.head_ctx(fJ)

        bands = []
        for j in range(self.levels):                       # scales J..1
            bands.append(self.enh[j].detail(z, flat(coords_lv[j]), ctx))
            if j + 1 < self.levels:
                # upsampled query coords: this trunk at the next finer grid
                ctx = self.enh[j].context(z, flat(coords_lv[j + 1]))

        def unflat(d, hwj):                                # (B,N,k)->(B,k,H,W)
            return d.permute(0, 2, 1).reshape(B, -1, *hwj)

        # ONE IDWT: assemble bands coarsest-first
        a_map = unflat(a, hw[-1])
        Ds = []
        for j, d in enumerate(bands):
            t = unflat(d, hw[self.levels - j])             # (B,3C,h,w)
            Ds.append((t[:, :C], t[:, C:2 * C], t[:, 2 * C:]))
        out = idwt2_stack(a_map, Ds, hw, self.wavelet)

        self.aux_loss = None
        if y is not None and getattr(self, 'vac', None) is not None and self.aux_weight > 0:
            # harmonic fill FIRST: the DWT targets must see no artificial step
            # at the vacuum boundary; the fluid-support masks stay on the raw
            # mask so the loss never counts filled-in (invented) coefficients
            y = harmonic_fill(y, self.fill_M, self.vac)
            A_t, Ds_t, _ = dwt2_stack(y, self.levels, self.wavelet)
            w = support_masks(~self.vac, self.levels, self.wavelet)
            losses = [self._rel_mse(a_map, A_t, w[0])]
            losses += [self._rel_mse(torch.cat(bd, 1), torch.cat(bt, 1), wj)
                       for bd, bt, wj in zip(Ds, Ds_t, w)]
            self.aux_loss = self.aux_weight * sum(losses) / len(losses)

        if return_bands:
            return out, {'A': a_map, 'D': Ds}
        return out
