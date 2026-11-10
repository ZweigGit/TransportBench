"""Differentiable 2D Haar DWT/IDWT with ragged (pad-when-odd) sizes.

Orthonormal Haar: analysis and synthesis both use the 1/2 factor, so
||coeffs|| == ||x|| on even sizes and IDWT(DWT(x)) == x exactly for any
size. Odd sides are replicate-padded (bottom/right) before each analysis
step and cropped back after synthesis; the size list returned by
dwt2_stack carries that bookkeeping.

Scale indexing (matches the WIEN-DeepONet design doc): A_j has shape
hw[j] with hw[j] = ceil(hw[j-1] / 2); D_j has the SAME shape as A_j;
Ds[0] is the COARSEST band D_J and Ds[-1] the finest D_1.
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
    pywt name routed through pytorch_wavelets (periodization). Constraint:
    periodization needs signal >= filter length at every level, e.g. on the
    17x384 grid db2 works up to J=4, db4 only up to J=2.
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
    module crashes on CUDA inputs and vice versa).
    """
    key = (wavelet, forward, str(device))
    if key not in _pw_cache:
        from pytorch_wavelets import DWTForward, DWTInverse
        m = (DWTForward(J=1, wave=wavelet, mode='periodization') if forward
             else DWTInverse(wave=wavelet, mode='periodization'))
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
