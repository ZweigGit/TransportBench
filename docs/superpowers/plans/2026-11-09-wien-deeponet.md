# WIEN-DeepONet DeepONet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A coarse-to-fine wavelet-recursive DeepONet ("WIEN-DeepONet", registry key `wien_deeponet`) for Task4: a branch net encodes the 4 flow params into a latent, a coarse-scale coordinate net predicts the coarsest Haar approximation band, and per-scale Enhancement nets predict the detail subbands from (latent, local patch of the model's own coarse reconstruction, coordinates); inverse DWT recurses level by level down to the full 17x384 field.

**Architecture:** Single new model file + a small Haar-transform utility file, plugged into the existing Task4 golden protocol (train.py curriculum-weighted MSE with vacuum mask, eval.py RL2E) so scores are directly comparable with the re-measured baselines (hyperdeeponet 0.1436, mscale 0.1457, wavelet 0.1567). Unlike the three failed 2026-11-09 band-split ports (independent per-band heads, RL2E 0.24-0.42), every scale here consumes the model's own reconstructed coarser scale — the cross-scale recursion the diagnosis said was missing — and the primary loss stays in field space.

**Tech Stack:** PyTorch 2.0.1+cu117 (`C:/Users/86177/Anaconda3/envs/pytorch/python.exe`), no new dependencies (hand-written Haar DWT/IDWT, no pywt).

**Spec:** User task message of 2026-11-09 (WIEN-DeepONet: sections 一/二/三, staged delivery with dimension check before coding, self-recursion during training). Reference: `_refs/wien_src` (INC4Sci), esp. `Train/INR_Trainers.py::KernelNetwork_MP` (per-position k x k kernel over patches of the previous scale's output — our E_j is its operator-learning version with concat-MLP instead of dot-product, the parametrization that failed 3x here).

## Global Constraints

- Run everything with `/c/Users/86177/Anaconda3/envs/pytorch/python.exe` (bash default python has no torch).
- Code comments in English only (conversation may be Chinese).
- No new dependencies. Pure-torch Haar.
- Golden-protocol comparability: the model must train through the *unmodified* loss path in train.py (curriculum-weighted MSE, vacuum mask, aux_loss hook) and evaluate through eval.py RL2E. Allowed edits to train.py/eval.py: model registration (import, choices, build_model branch), passing `vac` into ctx for this model, and the optional-target forward call. Nothing else.
- Data contract: `data/double_cone_dataset.pt` — 45 samples, x `[45, 2+4, 17, 384]` (ch 0:2 grid coords, ch 2: params, all min-max encoded to [0,1]), y `[45, 4, 17, 384]` (rho, u, v, p; p log10 then min-max; stats from train split, fluid cells only). 36/9 split seed 42. Vacuum mask constant `[17, 384]` bool.
- Param budget ~0.9M (family convention: all DeepONet variants ≈ 1M).

## Design — Stage-0 tensor contract (checked before coding)

### Scale table, Haar, ragged pad-when-odd (replicate), J=4, grid (17, 384)

Sizes obey `A_j = ceil(A_{j-1}/2)` per axis; a level pads by 1 (replicate, bottom/right) only when the input side is odd. Deterministic, no powers-of-2 padding of the whole grid (47% dead rows avoided).

| j | A_j / D_j shape (H, W) | H pad at this level? | E_j positions H_j*W_j |
|---|---|---|---|
| 0 | (17, 384) output | 17 odd → pad 18 | — |
| 1 | (9, 192) | 9 odd → pad 10 | 1728 |
| 2 | (5, 96) | 5 odd → pad 6 | 480 |
| 3 | (3, 48) | 3 odd → pad 4 | 144 |
| 4 | (2, 24) coarsest | (2 even) | 48 |

IDWT(A_j, D_j) produces 2*H_j rows, cropped back to H_{j-1} when H_{j-1} was odd: 4→3, 6→5, 10→9, 18→17. D_j always has the *same* shape as A_j.

### Forward dataflow (B=9, C=4, latent=64, window k=3, hidden=256)

```
x_enc [B,5,17,384]
  params = x_enc[:, 2:, 0, 0]                     [B,4]
  z = BranchMLP(params)                           [B,64]

  A4_hat = CoarseNet(z | xy_4)                    [B,4,2,24]     (48 positions, shared MLP)
  for j = 4..1:
      patches = replicate_unfold3x3(A_j_hat)      [B, 36, H_j*W_j]
      D_j_hat = E_j(z | patches | xy_j)           3 x [B,4,H_j,W_j]   (one net per scale)
      A_{j-1}_hat = haar_idwt2(A_j_hat, D_j_hat)  [B,4,H_{j-1},W_{j-1}]
  out = A_0_hat                                   [B,4,17,384]
```

`E_j` per position: `[z(64) ⊕ flatten(3x3xC patch)(36) ⊕ xy(2)] = 102 → 256 → 256 → 256 → 12` (GELU + LayerNorm hidden). CoarseNet: `[66 → 256 → 256 → 256 → 4]`. Branch: `[4 → 256 → 256 → 64]`. Params ≈ 4x161K + 150K + 84K ≈ 0.88M.

### Losses

- Primary: the stock train.py curriculum-weighted masked MSE on `out` (normalized field space) — unchanged code path.
- Aux (per spec: final reconstruction + per-scale coefficient supervision): inside `forward(x, y=None)`, when `y` is given, `dwt2_stack(y, J)` provides true A_4, D_4..D_1; `self.aux_loss = aux_weight * mean_j( relMSE_j )` with relMSE_j = `Σ w_j (D̂−D)² / Σ w_j D²` (per-band energy-normalized, wienb precedent = paper eq.2/3), A_4 supervised the same way. `w_j` = fraction of each coefficient's pixel support that is fluid-valid + non-padded, computed by the same ceil-halving chain from the vacuum mask (passed via ctx, non-persistent buffer). Default aux_weight=0.1.
- Training uses the model's own recursion exclusively (no teacher forcing): `y` enters only as a loss target, never as recursion input. Eval calls `model(x)` without `y`; output identical.

### Consistency invariants (the checks the spec demands)

1. DWT/IDWT exact inverses incl. ragged crops — tested (Task 1).
2. D_j shape == A_j shape at every level — asserted (Task 3).
3. Single forward code path for train and eval; recursion input is always the model's own reconstruction.
4. No per-band independent summation: the only path from D̂_j to the output is through IDWT of the recursively built A_j.

## Review Focus

1. **Ragged crop off-by-one** (pad bottom/right, crop back on inverse) — perfect-reconstruction test on odd sizes 17x384 and mixed 5x6 pins it (Task 1).
2. **Support-weight map must mirror the same ceil-halving**, else aux losses weight wrong coefficients — test: all-valid grid → all weights 1; single masked pixel → its ancestors at every scale carry the exact fractional weight (Task 1).
3. **Scale-shape mismatch between enhancement outputs and IDWT inputs** (D_j at A_{j-1} resolution is the classic bug) — assert shape equality per level in the full-model test (Task 3).
4. **Teacher-forcing creep** (true coefficients leaking into recursion) — test: `forward(x)` output bitwise-equal to `forward(x, y)[0]` (Task 3).
5. **eval.py rebuild mismatch** (buffer/config drift making strict load fail) — save/load roundtrip through eval.py's exact build path (Task 3).

---

### Task 1 (Stage 1): Haar DWT/IDWT utility + self-checks

**Files:**
- Create: `Task4_DoubleCone/wavelet_haar.py`
- Test: `Task4_DoubleCone/test_wien_deeponet.py` (grows per task; plain asserts, `python test_wien_deeponet.py`)

**Interfaces:**
- Produces (consumed by Task 3 and the tests):
  - `haar_dwt2(x: (B,C,H,W)) -> (ll, lh, hl, hh)` each `(B,C,ceil(H/2),ceil(W/2))`, orthonormal Haar (factor 1/2), ragged replicate pad when odd.
  - `haar_idwt2(ll, lh, hl, hh, out_hw) -> (B,C,H,W)` exact inverse (crop to `out_hw`).
  - `dwt2_stack(x, J) -> (A, Ds, hw)`; `A` `(B,C,H_J,W_J)`, `Ds[0]`=D_J (coarsest) … `Ds[J-1]`=D_1 (finest), each `(lh,hl,hh)`; `hw` list of A_j shapes j=0..J.
  - `idwt2_stack(A, Ds, hw) -> x`.
  - `support_weights(valid01: (H,W) float, hw) -> [w_J, ..., w_1]` same indexing as Ds (+ none for A: A uses `w_J` too; return list of length J, coarsest-first).

- [ ] **Step 1: write failing tests** — perfect reconstruction (`max|idwt(dwt(x)) - x| < 1e-5`) for shapes (2,3,17,384), (1,1,32,512), (2,4,5,6); orthonormal energy conservation on even sizes (`||pyramid|| ≈ ||x||`, rel tol 1e-5); autograd finite-difference check on `idwt2_stack(dwt2_stack(x))` (rel err < 1e-3, fp32); support_weights all-ones and single-pixel-mask cases; D_j shape == A_j shape for all levels.
- [ ] **Step 2: run, verify fail** — `cd Task4_DoubleCone && /c/.../python.exe test_wien_deeponet.py` → ImportError.
- [ ] **Step 3: implement `wavelet_haar.py`** — reshape-based Haar: `x.reshape(B,C,H//2,2,W//2,2)`, `ll=(a+b+c+d)/2` etc.; inverse `x00=(ll+lh+hl+hh)/2` etc. Track odd sides, slice crops on inverse.
- [ ] **Step 4: run, verify pass.**
- [ ] **Step 5: commit** — `git add Task4_DoubleCone/wavelet_haar.py Task4_DoubleCone/test_wien_deeponet.py && git commit -m "Task4: differentiable Haar DWT/IDWT with ragged padding (WIEN-DeepONet stage 1)"`

### Task 2 (Stage 2): building-block nets

**Files:**
- Create: `Task4_DoubleCone/model_wien_deeponet.py` (this task: Branch/Coarse/Enhancement classes only)
- Test: `Task4_DoubleCone/test_wien_deeponet.py` (append)

**Interfaces:**
- Consumes: nothing yet (plain nn.Modules).
- Produces:
  - `BranchNet(branch_dim, latent_dim, hidden_dim, depth)`, `forward(params: (B,P)) -> (B,latent)`.
  - `CoarseNet(latent_dim, num_outputs, hidden_dim, depth)`, `forward(z: (B,L), coords: (B,N,2)) -> (B,N,C)`.
  - `EnhancementNet(latent_dim, num_outputs, hidden_dim, depth, window)`, `forward(z: (B,L), patches: (B,N,C*k*k), coords: (B,N,2)) -> (B,N,3C)`.
  - `replicate_unfold(a: (B,C,H,W), k) -> (B, C*k*k, H*W)` — clamp-index replicate pad then `F.unfold` (works for H=1/2, any k).
  - All hidden layers GELU + LayerNorm (runaway lesson from the band-split ports), standard init (no zero-init heads — no bilinear forms anywhere).

- [ ] **Step 1: failing tests** — shapes for each net (random inputs, exact expected output dims); replicate_unfold equals manual `F.pad(replicate)+unfold` on a 4x5 input AND works on H=1 (where `F.pad(replicate)` itself raises).
- [ ] **Step 2: verify fail.** **Step 3: implement.** **Step 4: verify pass.**
- [ ] **Step 5: commit** — `"Task4: branch/coarse/enhancement nets for WIEN-DeepONet (stage 2)"`

### Task 3 (Stage 3): full model + harness integration

**Files:**
- Modify: `Task4_DoubleCone/model_wien_deeponet.py` (add `WienDeepONet`), `Task4_DoubleCone/train.py` (registration: import, choices, build_model, ctx vac, optional-target forward), `Task4_DoubleCone/eval.py` (choices + build_model branch; forward unchanged — `model(x)`).
- Test: `Task4_DoubleCone/test_wien_deeponet.py` (append)

**Interfaces:**
- Produces: `WienDeepONet(branch_dim, num_outputs=4, latent_dim=64, hidden_dim=256, branch_depth=2, depth=3, levels=4, window=3, aux_weight=0.1, ctx=None)`; `forward(x: (B,2+P,17,384), y=None) -> (B,4,17,384)`; sets `self.aux_loss` (None when y is None or no vac in ctx); class attr `uses_target = True`; `vac` stored as non-persistent buffer (state_dict stays clean, eval needs no vac).
- train.py forward call becomes: `out_enc = model(x_enc, y_enc) if getattr(model, 'uses_target', False) else model(x_enc)`.
- Per-scale coordinate grids `xy_j` in [0,1] built once in `__init__` (buffer per scale).

- [ ] **Step 1: failing tests** — full forward on real dataset shapes (random x, y): output shape; per-level `D̂_j` shapes equal `dwt2_stack` level shapes (assert inside a debug return or via hooks — simplest: `forward(..., return_bands=True)` used only by tests); `forward(x)` == `forward(x, y)` output (torch.equal); `loss = mse(out, y) + model.aux_loss; loss.backward()` — every parameter's grad is finite and non-None; gradclip+optimizer step smoke (10 steps on the real 36-sample split, loss decreases); save via train.py's exact checkpoint format → rebuild through `eval.py`'s `build_model` path → strict `load_state_dict` OK.
- [ ] **Step 2: verify fail.** **Step 3: implement** (model + the 3 small harness edits; nothing else in the loss path). **Step 4: verify pass.**
- [ ] **Step 5: commit** — `"Task4: WIEN-DeepONet DeepONet full model + golden-protocol integration (stage 3)"`

### Task 4 (Stage 4): train, evaluate, compare

**Files:** none created; runs only. Results reported in chat (+ optionally output/ artifacts the harness already writes).

- [ ] **Step 1:** smoke run 100 epochs (`--epochs 100 --best_after 50`), confirm loss decreases and no runaway (aux band norms bounded).
- [ ] **Step 2:** full golden run: `python train.py --model wien_deeponet` (defaults: 2500 epochs, bs 9, lr 1e-3), then `python eval.py --model wien_deeponet`.
- [ ] **Step 3:** seeds 43, 44 repeats → mean ± std; compare vs hyperdeeponet 0.1436 single / 0.1396±0.0029 (3-seed) and the 0.015 noise line; per-channel and per-sample breakdown if it beats baseline.
- [ ] **Step 4:** report; if ≥ baseline − noise, discuss; else record the negative result against the recursion hypothesis (three band-split failures + this would isolate "cross-scale recursion" as the remaining untested axis... which this run *is*).

No commit for Task 4 (no repo changes) unless numbers warrant a README note.
