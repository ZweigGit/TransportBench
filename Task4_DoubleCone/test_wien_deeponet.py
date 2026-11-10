"""Self-checks for the WIEN-DeepONet DeepONet (coefficient-domain redesign).
Tests accumulate per stage.

Run: python test_wien_deeponet.py   (plain asserts, no test framework)
"""
import torch

PASS = []


def check(name):
    PASS.append(name)
    print(f"  ok: {name}")


# ---------------------------------------------------------------- stage 1
def test_stage1():
    from wavelet_haar import (haar_dwt2, haar_idwt2, dwt2_stack,
                              idwt2_stack, support_weights)

    torch.manual_seed(0)

    # single level: shapes + perfect reconstruction on odd sizes
    x = torch.randn(2, 3, 17, 384)
    ll, lh, hl, hh = haar_dwt2(x)
    assert ll.shape == (2, 3, 9, 192), ll.shape
    assert all(b.shape == ll.shape for b in (lh, hl, hh))
    xr = haar_idwt2(ll, lh, hl, hh, (17, 384))
    assert xr.shape == x.shape
    assert (xr - x).abs().max().item() < 1e-5, (xr - x).abs().max().item()
    check("single-level perfect reconstruction on 17x384")

    # even sizes: perfect reconstruction + orthonormal energy conservation
    x = torch.randn(2, 4, 32, 512)
    ll, lh, hl, hh = haar_dwt2(x)
    xr = haar_idwt2(ll, lh, hl, hh, (32, 512))
    assert (xr - x).abs().max().item() < 1e-5
    e_fwd = x.pow(2).sum().item()
    e_c = sum(b.pow(2).sum().item() for b in (ll, lh, hl, hh))
    assert abs(e_c - e_fwd) / e_fwd < 1e-5, (e_c, e_fwd)
    check("even-size reconstruction + energy conservation")

    # tiny mixed-odd case
    x = torch.randn(2, 4, 5, 6)
    A, Ds, hw = dwt2_stack(x, 2)
    assert hw == [(5, 6), (3, 3), (2, 2)], hw
    assert A.shape == (2, 4, 2, 2)
    assert all(b.shape == (2, 4, 2, 2) for b in Ds[0])   # coarsest D_2
    assert all(b.shape == (2, 4, 3, 3) for b in Ds[1])   # finest D_1
    xr = idwt2_stack(A, Ds, hw)
    assert (xr - x).abs().max().item() < 1e-5
    check("stack J=2 shapes (coarsest-first) + reconstruction on 5x6")

    # full pyramid on the task4 grid
    x = torch.randn(2, 4, 17, 384)
    A, Ds, hw = dwt2_stack(x, 4)
    assert hw == [(17, 384), (9, 192), (5, 96), (3, 48), (2, 24)], hw
    assert A.shape == (2, 4, 2, 24)
    for j, exp in zip(Ds, [(2, 24), (3, 48), (5, 96), (9, 192)]):
        assert all(b.shape == (2, 4, *exp) for b in j), j
    xr = idwt2_stack(A, Ds, hw)
    assert (xr - x).abs().max().item() < 1e-5
    check("stack J=4 size table matches design (2,24)/(3,48)/(5,96)/(9,192)")

    # autograd: analytic grad vs central finite differences (float64)
    x = torch.randn(1, 2, 5, 6, dtype=torch.float64, requires_grad=True)
    A, Ds, hw = dwt2_stack(x, 2)
    y = idwt2_stack(A, Ds, hw)
    w = torch.randn_like(y)
    (y * w).sum().backward()
    g = x.grad.clone()
    eps = 1e-6
    for idx in [(0, 0, 0, 0), (0, 1, 2, 3), (0, 0, 4, 5)]:
        xp = x.detach().clone(); xp[idx] += eps
        xm = x.detach().clone(); xm[idx] -= eps
        def f(t):
            a, d, h = dwt2_stack(t, 2)
            return (idwt2_stack(a, d, h) * w).sum().item()
        num = (f(xp) - f(xm)) / (2 * eps)
        assert abs(num - g[idx].item()) < 1e-6, (idx, num, g[idx].item())
    check("autograd matches finite differences on DWT/IDWT round trip")

    # support weights: all-valid -> all ones at every scale
    valid = torch.ones(17, 384)
    ws = support_weights(valid, [(17, 384), (9, 192), (5, 96), (3, 48), (2, 24)])
    assert len(ws) == 4
    assert all(tuple(w.shape) == s and (w == 1).all().item()
               for w, s in zip(ws, [(2, 24), (3, 48), (5, 96), (9, 192)]))
    check("support weights all-ones on fully valid grid")

    # single masked pixel -> exact fractions through the halving chain
    valid = torch.ones(17, 384)
    valid[16, 100] = 0.0
    ws = support_weights(valid, [(17, 384), (9, 192), (5, 96), (3, 48), (2, 24)])
    # level-1 cell (8,50) covers real cells (16,100),(16,101): 1 of 2 valid
    # (the padded row replicates row 16, so its validity copies too)
    assert abs(ws[3][8, 100 // 2].item() - 0.5) < 1e-6
    # level-2 cell (4,25) covers real row 16, cols 100..103: 3 of 4 valid
    assert abs(ws[2][4, 25].item() - 0.75) < 1e-6
    assert ws[3].sum().item() == ws[3].numel() - 0.5              # exactly one 1/2 cell
    check("support weights exact fractions for a single masked pixel")


# ---------------------------------------------------------------- stage 2
def test_stage2():
    from model_wien_deeponet import (BranchNet, CoarseNet,
                                         EnhancementNet)

    torch.manual_seed(0)
    z = torch.randn(3, 64)
    coords = torch.randn(3, 48, 2)

    br = BranchNet(branch_dim=4, latent_dim=64, hidden_dim=256, depth=2)
    assert br(torch.randn(3, 4)).shape == (3, 64)

    cn = CoarseNet(latent_dim=64, num_outputs=4, hidden_dim=256, depth=2)
    f = cn.features(z, coords)
    assert f.shape == (3, 48, 256)
    assert cn(z, coords).shape == (3, 48, 4)             # A_J head
    assert cn.head_ctx(f).shape == (3, 48, 12)           # rough D_J head

    en = EnhancementNet(latent_dim=64, num_outputs=4, hidden_dim=256, depth=2)
    ctx = torch.randn(3, 48, 12)
    assert en.detail(z, coords, ctx).shape == (3, 48, 12)
    # same trunk queried at a finer grid: context for the next scale
    assert en.context(z, torch.randn(3, 192, 2)).shape == (3, 192, 12)
    check("branch/coarse/enhancement net shapes (detail + context heads)")

    # gradients flow branch -> context -> detail
    p = torch.randn(3, 4, requires_grad=True)
    ctx2 = en.context(br(p), coords)
    out = en.detail(br(p), coords, ctx2)
    out.sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    check("gradient flows branch -> context -> detail")


# ---------------------------------------------------------------- stage 3
def test_stage3():
    from model_wien_deeponet import WienDeepONet
    from wavelet_haar import dwt2_stack

    torch.manual_seed(0)
    B, C, H, W = 3, 4, 17, 384
    x = torch.rand(B, 6, H, W)
    y = torch.rand(B, C, H, W)
    vac = torch.zeros(H, W, dtype=torch.bool)
    vac[5:9, 40:60] = True
    ctx = {'n_param': 4, 'vac': vac}

    m = WienDeepONet(branch_dim=4, ctx=ctx)
    n_par = sum(p.numel() for p in m.parameters())
    assert 7e5 < n_par < 1.1e6, n_par
    out = m(x, y)
    assert out.shape == (B, C, H, W)
    assert m.aux_loss is not None and torch.isfinite(m.aux_loss)
    check(f"full forward shape + coefficient loss finite ({n_par/1e6:.2f}M params)")

    # per-level band shapes match the true pyramid
    field, bands = m(x, y, return_bands=True)
    A_t, Ds_t, hw = dwt2_stack(y, 4)
    for (lh, hl, hh), true in zip(bands['D'], Ds_t):
        assert lh.shape == true[0].shape and hl.shape == true[0].shape
    assert bands['A'].shape == A_t.shape
    check("predicted band shapes == true pyramid shapes at every level")

    # no teacher forcing: passing y never changes the output
    m.eval()
    with torch.no_grad():
        a, b = m(x), m(x, y)
    assert torch.equal(a, b)
    m.train()
    check("forward(x) bitwise-equal forward(x, y) (no teacher forcing)")

    # end-to-end gradient: every parameter finite; the context chain
    # (coarse head_ctx + Enh_4..2 head_ctx) actually carries gradient
    loss = m(x, y).pow(2).mean() + m.aux_loss
    loss.backward()
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert m.coarse.head_ctx.weight.grad.abs().sum() > 0
    for j in range(3):
        assert m.enh[j].head_ctx.weight.grad.abs().sum() > 0
    check("end-to-end backward; ctx heads of Coarse + Enh_4..2 receive gradient")

    # eval-style rebuild (no vac in ctx) strict-loads a saved state_dict
    state = m.state_dict()
    m2 = WienDeepONet(branch_dim=4, ctx={'n_param': 4})
    m2.load_state_dict(state)
    check("eval-style rebuild strict-loads train state_dict")


def test_stage3_real():
    """20 optimizer steps on the real 36-sample split: loss must decrease."""
    from data_loader import get_dataloader_and_stats
    from train import build_model
    from model_wien_deeponet import WienDeepONet

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, _, x_norm, y_norm, vac = get_dataloader_and_stats(
        '../data/double_cone_dataset.pt', 9, torch.device(dev))
    ctx = {'n_param': 4, 'vac': vac.to(dev)}

    m = build_model('wien_deeponet', False, ctx).to(dev)
    assert isinstance(m, WienDeepONet)
    assert m.coef_loss_only                 # coefficient-domain training loss
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    batches = list(train_loader)
    losses = []
    for step in range(20):
        x, y = batches[step % len(batches)]
        xe, ye = x_norm.encode(x), y_norm.encode(y)
        m(xe, ye)                           # field out exists for eval only
        loss = m.aux_loss                   # == trainer's coef_loss_only branch
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    print(f"    smoke losses: {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0], losses
    assert all(torch.isfinite(torch.tensor(losses)))
    check("20-step smoke train on real data (coef loss only): loss decreases")


# ---------------------------------------------------------------- basis sweep
def test_basis():
    from wavelet_haar import dwt2_stack, idwt2_stack
    torch.manual_seed(0)

    x = torch.randn(2, 4, 17, 384)
    # symmetric extension carries its own boundary coefficients, so db-N
    # bands are slightly larger than half size and there is no periodization
    # size constraint on J anymore
    A, Ds, hw = dwt2_stack(x, 4, wavelet='db2')
    assert hw == [(17, 384), (10, 193), (6, 98), (4, 50), (3, 26)], hw
    assert A.shape == (2, 4, 3, 26)
    xr = idwt2_stack(A, Ds, hw, wavelet='db2')
    assert (xr - x).abs().max().item() < 1e-5, (xr - x).abs().max().item()
    check("db2 J=4 pyramid (symmetric): size table + perfect reconstruction")

    A2, Ds2, hw2 = dwt2_stack(x, 4, wavelet='db4')
    assert hw2 == [(17, 384), (12, 195), (9, 101), (8, 54), (7, 30)], hw2
    xr2 = idwt2_stack(A2, Ds2, hw2, wavelet='db4')
    assert (xr2 - x).abs().max().item() < 1e-4
    check("db4 J=4 pyramid (symmetric, no J<=2 constraint): perfect reconstruction")


def test_harmonic_fill():
    """The pre-factorized fill must equal the notebook's sparse Laplace solve."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    from wavelet_haar import harmonic_fill, harmonic_fill_operator
    from data_loader import vacuum_mask

    data = torch.load('../data/double_cone_dataset.pt', weights_only=False)
    y = data['y'].float()
    vac = vacuum_mask(y)
    f = y[0, 0].clone()                       # raw channel, any values work
    f[vac] = 0.0

    # reference: per-unknown sparse solve (notebook formulation)
    H, W = vac.shape
    v = vac.numpy()
    idx = np.full((H, W), -1, dtype=np.int64); idx[v] = np.arange(v.sum())
    rows, cols, vals = [], [], []
    rhs = np.zeros(int(v.sum()))
    for i, j in zip(*np.where(v)):
        k = int(idx[i, j]); rows.append(k); cols.append(k); vals.append(4.0)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if not (0 <= ii < H and 0 <= jj < W): vals[-1] -= 1.0
            elif not v[ii, jj]: rhs[k] += float(f[ii, jj])
            else: rows.append(k); cols.append(int(idx[ii, jj])); vals.append(-1.0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(int(v.sum()),) * 2).tocsc()
    ref = f.clone()
    ref[vac] = torch.as_tensor(spsolve(A, rhs)).float()   # spsolve is float64

    M = harmonic_fill_operator(vac)
    got = harmonic_fill(f[None, None], M, vac)[0, 0]
    err = (got - ref).abs().max().item()
    assert err < 1e-4, err                    # float32 matmul vs float64 solve
    assert torch.equal(got[~vac], f[~vac])    # fluid passes through exactly
    check(f"harmonic_fill matches the sparse Laplace solve (max diff {err:.1e})")

    # differentiable in the fluid neighbours (rhs): grads reach the fluid cells
    g = y[0, 0].clone().requires_grad_()
    out = harmonic_fill(g[None, None], M, vac)
    out[0, 0][vac].sum().backward()
    assert g.grad is not None and torch.isfinite(g.grad).all()
    assert g.grad[~vac].abs().sum() > 0
    check("harmonic_fill gradients flow into the fluid boundary cells")


def test_support_masks():
    from wavelet_haar import support_masks
    from data_loader import vacuum_mask
    data = torch.load('../data/double_cone_dataset.pt', weights_only=False)
    vac = vacuum_mask(data['y'].float())

    # fully valid field -> every coefficient counts
    w = support_masks(torch.ones(17, 384), 2, 'db4')
    assert all((m == 1).all() for m in w)
    check("support_masks: all-valid grid -> all ones at every level")

    # real vacuum layout (cone interior + outflow corners, cells in every row):
    # coefficients whose filter reach crosses vacuum cells are excluded.
    # MEASURED on this mask (not derived): db4 level-1 band rows 0/2/3 sit
    # against vacuum-dominated real rows in the strip columns, rows >= 4 don't.
    w = support_masks((~vac).float(), 2, 'db4')
    w1 = w[-1]
    assert w1[0, 4:96].float().mean() < 0.1       # top-edge row mostly excluded
    assert bool((w1[4:, 4:96] == 1).all())        # deep rows over the strip stay
    frac = sum(m.sum() for m in w) / sum(m.numel() for m in w)
    assert 0.7 < frac < 0.95, frac            # most coefficients stay fluid-only
    check(f"support_masks: vacuum-strip coefficients excluded ({frac:.1%} kept)")


def test_sweep_wiring():
    from train import build_model
    from model_wien_deeponet import WienDeepONet

    x = torch.rand(2, 6, 17, 384)
    y = torch.rand(2, 4, 17, 384)
    m = build_model('wien_deeponet', False,
                    ctx={'n_param': 4, 'levels': 5,
                         'wavelet': 'db2', 'aux_weight': 0.0})
    assert isinstance(m, WienDeepONet)
    assert m.levels == 5 and m.wavelet == 'db2'
    out = m(x, y)
    assert out.shape == (2, 4, 17, 384)
    assert m.aux_loss is None                    # aux_weight=0 disables
    loss = (out - y).pow(2).mean()
    loss.backward()
    check("sweep overrides (J=5, db2, aux=0) build + backward")


def test_basis_cuda():
    """db2 path on CUDA: filters must follow the input device (regression:
    CPU-registered filter banks crashed CUDA training)."""
    import torch
    if not torch.cuda.is_available():
        return
    from wavelet_haar import dwt2_stack, idwt2_stack
    x = torch.randn(2, 4, 17, 384, device='cuda', requires_grad=True)
    A, Ds, hw = dwt2_stack(x, 4, wavelet='db2')
    y = idwt2_stack(A, Ds, hw, wavelet='db2')
    assert (y - x.detach()).abs().max().item() < 1e-4
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    check("db2 pyramid on CUDA: perfect reconstruction + grads")


if __name__ == "__main__":
    print("WIEN-DeepONet self-checks (coefficient-domain redesign)")
    test_stage1()
    test_stage2()
    test_stage3()
    test_stage3_real()
    test_basis()
    test_harmonic_fill()
    test_support_masks()
    test_sweep_wiring()
    test_basis_cuda()
    print(f"\nAll {len(PASS)} checks passed.")
