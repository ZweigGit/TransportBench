"""Differentiable 2D DWT/IDWT with ragged sizes (the WIEN-DeepONet stack).

wavelet='haar' uses a hand-written orthonormal fast path (analysis and
synthesis both scaled by 1/2, so ||coeffs|| == ||x|| and IDWT(DWT(x)) == x
exactly; odd sides are replicate-padded bottom/right before each analysis
step and cropped back after synthesis). Anything else is a pywt name routed
through pytorch_wavelets with SYMMETRIC extension: no circular wrap between
the two ends of the (non-periodic) domain, forward/inverse still a perfect
reconstruction; the mirror carries its own boundary coefficients, so db-N
bands are slightly larger than half size (e.g. 17x384 -> 12x195 -> 9x101 for
db4). The hw list returned by dwt2_stack records those shapes.

Scale indexing (matches the WIEN-DeepONet design doc): A_j has shape
hw[j] with hw[j] = the j-th DWT low-pass shape; D_j has the SAME shape as
A_j; Ds[0] is the COARSEST band D_J and Ds[-1] the finest D_1.
"""
import torch
import torch.nn.functional as F


def _pad_odd(x):
    """Replicate-pad last two dims by 1 (bottom/right) when odd."""
    ph, pw = x.shape[-2] % 2, x.shape[-1] % 2
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='replicate')
    return x


def haar_dwt2(x):
    """One Haar level: (B,C,H,W) -> (ll, lh, hl, hh), each (B,C,ceil(H/2),ceil(W/2)).

    lh keeps the vertical (row) detail, hl the horizontal (column) detail.
    """
    xp = _pad_odd(x)
    B, C, H, W = xp.shape
    x = xp.reshape(B, C, H // 2, 2, W // 2, 2)
    a, b = x[:, :, :, 0, :, 0], x[:, :, :, 0, :, 1]   # top row of each 2x2 block
    c, d = x[:, :, :, 1, :, 0], x[:, :, :, 1, :, 1]   # bottom row
    ll = (a + b + c + d) / 2
    lh = (a + b - c - d) / 2
    hl = (a - b + c - d) / 2
    hh = (a - b - c + d) / 2
    return ll, lh, hl, hh


def haar_idwt2(ll, lh, hl, hh, out_hw):
    """Exact inverse of haar_dwt2, cropped back to out_hw (the pre-pad size)."""
    s = (ll + lh + hl + hh) / 2
    t = (ll + lh - hl - hh) / 2
    u = (ll - lh + hl - hh) / 2
    v = (ll - lh - hl + hh) / 2
    B, C, Hh, Wh = ll.shape
    x = torch.empty(B, C, Hh, 2, Wh, 2, dtype=ll.dtype, device=ll.device)
    x[:, :, :, 0, :, 0], x[:, :, :, 0, :, 1] = s, t
    x[:, :, :, 1, :, 0], x[:, :, :, 1, :, 1] = u, v
    return x.reshape(B, C, Hh * 2, Wh * 2)[..., :out_hw[0], :out_hw[1]]


def dwt2_stack(x, J, wavelet='haar'):
    """J-level pyramid: (A, Ds, hw).

    A   : (B, C, H_J, W_J)                coarsest approximation
    Ds  : [(lh, hl, hh)] x J, COARSEST-first (Ds[0] = D_J, Ds[-1] = D_1)
    hw  : [(H_0, W_0), ..., (H_J, W_J)]   A_j shapes, hw[0] is the input

    wavelet='haar' uses the hand-written fast path above; anything else is a
    pywt name routed through pytorch_wavelets (symmetric extension). db-N
    bands are slightly larger than half size because the mirror extension
    carries its own boundary coefficients (e.g. db4 on 17x384: 12x195, 9x101).
    """
    hw = [tuple(x.shape[-2:])]
    Ds = []
    a = x
    for _ in range(J):
        ll, lh, hl, hh = dwt_level(a, wavelet)
        Ds.append((lh, hl, hh))
        a = ll
        hw.append(tuple(a.shape[-2:]))
    return a, Ds[::-1], hw


def idwt2_stack(A, Ds, hw, wavelet='haar'):
    """Inverse of dwt2_stack: recursively A_{j-1} = IDWT(A_j, D_j)."""
    a = A
    for j, (lh, hl, hh) in enumerate(Ds):
        level = len(Ds) - j            # Ds[0] is D_J
        a = idwt_level(a, lh, hl, hh, hw[level - 1], wavelet)
    return a


_pw_cache = {}


def _pw(wavelet, forward, device):
    """Cached per-(wavelet, direction, device) pytorch_wavelets module.

    Their filter banks are registered buffers created on CPU; a separate
    cache entry per device keeps them on the input's device (a plain CPU
    module crashes on CUDA inputs and vice versa). Symmetric extension: no
    circular wrap between the two ends of the domain, exact inverse.
    """
    key = (wavelet, forward, str(device))
    if key not in _pw_cache:
        from pytorch_wavelets import DWTForward, DWTInverse
        m = (DWTForward(J=1, wave=wavelet, mode='symmetric') if forward
             else DWTInverse(wave=wavelet, mode='symmetric'))
        _pw_cache[key] = m.to(device)
    return _pw_cache[key]


def dwt_level(x, wavelet='haar'):
    """One decomposition level of any supported wavelet."""
    if wavelet == 'haar':
        return haar_dwt2(x)
    YL, Yh = _pw(wavelet, True, x.device)(x)
    b = Yh[0]                                   # (B, C, 3, H', W')
    return YL, b[:, :, 0], b[:, :, 1], b[:, :, 2]


def idwt_level(ll, lh, hl, hh, out_hw, wavelet='haar'):
    """One synthesis level of any supported wavelet, cropped to out_hw."""
    if wavelet == 'haar':
        return haar_idwt2(ll, lh, hl, hh, out_hw)
    y = _pw(wavelet, False, ll.device)((ll, [torch.stack([lh, hl, hh], dim=2)]))
    return y[..., :out_hw[0], :out_hw[1]]


def support_weights(valid01, hw):
    """Per-coefficient fraction of fluid-valid support, mirroring the pads.

    Successive 2x2 average pooling of the [0,1] validity map (replicate-
    padding odd tails first), so each pooled value is the mean validity of
    the real cells that coefficient covers. Returns [w_J, ..., w_1],
    coarsest-first, matching the Ds indexing of dwt2_stack; w_J also
    weights the coarsest approximation band A_J.
    """
    w = valid01.float()[None, None]
    outs = []
    for _ in range(len(hw) - 1):
        w = F.avg_pool2d(_pad_odd(w), 2)
        outs.append(w[0, 0])
    return outs[::-1]        # [w_1, ..., w_J] -> coarsest-first like Ds


def harmonic_fill_operator(vac):
    """Prefactor the harmonic (Laplace) vacuum fill ONCE for a fixed mask.

    Vacuum cells satisfy 4u - sum(vacuum neighbours) = sum(fluid neighbours)
    with zero-flux at the grid edges (off-grid terms dropped). Returns the
    dense (n, n) matrix M = A^-1 of that linear system: vacuum values =
    rhs @ M.T, where rhs[k] is the sum of the fluid 4-neighbours of vacuum
    cell k (rhs is differentiable, so the fill trains through it). Builds
    with scipy once at setup; applying is a small matmul per batch.
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    H, W = vac.shape
    v = vac.cpu().numpy()                          # CUDA mask fine: build on CPU
    idx = np.full((H, W), -1, dtype=np.int64)
    idx[v] = np.arange(v.sum())                    # vacuum cell -> unknown id
    n = int(v.sum())
    rows, cols, vals = [], [], []
    for i, j in zip(*np.where(v)):
        k = int(idx[i, j])
        rows.append(k); cols.append(k); vals.append(4.0)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if not (0 <= ii < H and 0 <= jj < W):
                vals[-1] -= 1.0                    # zero-flux edge: drop the term
            elif v[ii, jj]:
                rows.append(k); cols.append(int(idx[ii, jj])); vals.append(-1.0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
    inv = spsolve(A, np.eye(n))
    return torch.from_numpy(np.asarray(inv)).float()


def harmonic_fill(y, fill_M, vac):
    """Harmonic inpaint of the vacuum cells of encoded targets y (B,C,H,W).

    Each vacuum cell takes the Laplace-solve value of its fluid 4-neighbours
    (Dirichlet boundary from the fluid, zero-flux at the grid edges), so the
    DWT below sees a field with no artificial step at the body. The solve is
    pre-factorized in fill_M (see harmonic_fill_operator); the applied part
    is differentiable w.r.t. y. Fluid cells pass through unchanged.
    """
    fl = (~vac).to(y.dtype)
    nb = torch.zeros_like(y)                       # sum of FLUID neighbours only
    nb[:, :, :-1, :] += y[:, :, 1:, :] * fl[1:, :]
    nb[:, :, 1:, :] += y[:, :, :-1, :] * fl[:-1, :]
    nb[:, :, :, :-1] += y[:, :, :, 1:] * fl[:, 1:]
    nb[:, :, :, 1:] += y[:, :, :, :-1] * fl[:, :-1]
    filled = y.clone()
    filled[:, :, vac] = nb[:, :, vac] @ fill_M.to(y.device).to(y.dtype).T
    return filled


def support_masks(valid01, J, wavelet='haar'):
    """Binary per-level masks of coefficients whose ENTIRE support is fluid.

    The [0,1] validity map is run through the SAME DWT used for the targets:
    a coefficient's low-pass response equals the all-fluid plateau value
    exactly when no vacuum cell (nor boundary extension of one) enters its
    filter reach, and drops below it otherwise. Thresholding recovers a
    strict fluid-only mask for the coefficient-domain loss. Returns
    [w_J, ..., w_1] coarsest-first, same layout as support_weights; w_J also
    masks the coarsest approximation band A_J.
    """
    v = valid01.float()[None, None]
    outs = []
    a = v
    for _ in range(J):
        plateau = dwt_level(torch.ones_like(a), wavelet)[0]
        a = dwt_level(a, wavelet)[0]
        outs.append((a[0, 0] >= 0.999 * plateau.max()).float())
    return outs[::-1]        # coarsest-first like Ds
