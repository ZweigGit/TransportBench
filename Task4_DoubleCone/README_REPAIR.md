# Task4 dataset repair: provenance and usage

## What was wrong
The released `double_cone_dataset_with_physics.pt` mispairs parameter rows with flow
fields (46/51 rows). Root cause (verified against the HF source): `preprocess.py`
collects fields with `sorted(glob('snapshot_*'))` (lexicographic), while
`extract_artist_logs.py` sorts the parameter table numerically
(`df.sort_values('snapshot')`, which also crashes on 'snapshot_10-1'); the two
lists were then joined by position.

## Ground truth
`harvest_nml.py` range-fetches `artist-sts.nml` from each source zip at
huggingface.co/datasets/CFDML/TransportBench/tree/main/Doublecone (~1 KB per case,
no full download) and recovers all 51 run conditions. case10/case50 have the nml
nested under a `caseN/` subdirectory. The real DoE is
u_inf (7) x rho_inf (5) x T_inf (3) x Tv_inf (4) x T_wall (3); T_wall is NOT
exposed in the benchmark's x channels.

## Files
- `repair_dataset.py`  - Hungarian re-pairing on (Ma, Re) recovered from the fields
- `data/double_cone_dataset_repaired_v3.pt` - canonical repair: x rebuilt from nml
  conditions, with field2case / confidence / nml_conditions metadata
- (intermediates: `_repaired.pt` Ma/Re pairing only, `_repaired_v2.pt` + field-evidence Tv)

Field-to-case confidence: 29 exact, 20 field-evidence (high), 2 low
(samples 37/43, Tv 1000 vs 2000 within the Ma=17.61 block).

## Result (same model & protocol)
RL2E 0.147 (4-run mean, broken data) -> ~0.100 on any repaired version.

## 4-param variant (experimental, NOT recommended as default)
`data/double_cone_dataset_repaired_4param.pt` adds normalized T_wall as a 4th input
channel (x = coords + Ma/Tv/Re/Twall). DeepONet auto-adapts (`branch_dim` in the
ckpt config; pass a `4param` data path). Caveat: T_wall is only CERTAIN for 22/51
fields (see `twall_confidence`); within 8 groups the labels are an unverified
permutation (the wall-temperature signature is too weak in the 17x384 resampled
fields to resolve). Empirically this hurts: RL2E 0.1145 vs 0.1015 (3-param v3,
same seed/protocol) - the noisy 4th input costs more than the ~1-6% twall signal
it carries. Use v3 unless T_wall labels are verified upstream.
