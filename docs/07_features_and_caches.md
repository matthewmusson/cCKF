# 07 — From parquet rows to feature caches

Doc 06 ended with 18 rows for one decision: the candidates the CKF saw at
pixel layer 4, module 361, on branch 128909. A network cannot read a parquet
row. This document covers the step that turns rows into fixed-length
vectors and labels, and stores them in a form a training loop can stream:
which columns become which features, where the label comes from, how the
events are split, and what a cache directory contains. The code is
`cckf/features.py`, `cckf/labels.py`, `cckf/splits.py`, `cckf/cache.py` and
the two builders `scripts/build_gate_cache.py`, `scripts/build_value_cache.py`.

## The gate's 26 features

`cckf.features.GATE_FEATURES` is the ordered list; the network's input
index is the position in this list, and `acts_patches/cckf/CckfFeatures.hpp`
builds the same 26 numbers in the same order at inference (doc 10). They
are grouped in `GATE_GROUPS`, which is what the ablation flags refer to:

| Group | # | Features | How they are computed |
|---|---|---|---|
| `kalman` | 0 to 5 | `residual_l0`, `residual_l1`, `chol_S_00`, `chol_S_10`, `chol_S_11`, `chi2_inc` | residuals and χ² straight from the row; the Cholesky factor of S (L00 = √S00, L10 = S01/L00, L11 = √(S11 − L10²)) replaces the three covariance entries so the network sees the window's shape in millimetres |
| `cluster_raw` | 6 to 11 | `clus_s_u`, `clus_s_v`, `clus_q_tot`, `clus_sigma_uu`, `clus_sigma_uv`, `clus_sigma_vv` | the candidate's cluster size (channels), total charge and charge-weighted second moments |
| `cluster_norm` | 12 to 14 | `kappa_u`, `kappa_v`, `q_tilde` | cluster size and charge divided by what the track's incidence angle predicts: κ_u = s_u / (1 + t·|tan α_u| / pitch_u), q̃ = Q / (t·√(1 + tan²α_u + tan²α_v)); a hit from a different particle crosses at a different angle and does not match |
| `occupancy` | 15 | `n_window` | candidates in the box on this surface |
| `context` | 16 to 19 | `eta`, `state_qop`, `step_k`, `pathInX0_interval` | η from the predicted θ, the predicted q/p, how far along the branch, material since the last hit |
| `sensor` | 20 to 22 | `pitch_u`, `pitch_v`, `thickness` | the module's channel pitch and thickness |
| `history` | 23 to 25 | `n_hits`, `n_holes`, `n_seq_holes` | the branch's record before this surface |

`build_gate_features(df) -> (n_rows, 26) float32` does this for a DataFrame
of parquet rows; non-finite values become 0. The two candidates from doc 06,
as the network sees them (raw values, before standardisation):

| feature | 53230 (true) | 53229 |
|---|---|---|
| `residual_l0`, `residual_l1` (mm) | −0.0001, 0.011 | −1.950, −0.114 |
| `chol_S_00`, `chol_S_10`, `chol_S_11` (mm) | 0.612, 0.092, 2.356 | same (one S per state in the parquet) |
| `chi2_inc` | 2e−5 | 10.2 |
| `clus_s_u`, `clus_s_v` | 1, 5 | 1, 2 |
| `clus_q_tot` | 0.244 | 0.150 |
| `kappa_u`, `kappa_v`, `q_tilde` | 0.790, 0.945, 0.982 | 0.790, 0.378, 0.603 |
| `n_window` | 18 | 18 |
| `eta`, `state_qop`, `step_k`, `pathInX0_interval` | −1.304, −0.298, 19 (3 after the order fix), 0.025 | same |
| `pitch_u`, `pitch_v`, `thickness` | 0.05, 0.05, 0.125 | same |
| `n_hits`, `n_holes`, `n_seq_holes` | 8, 11, 1 as stored; 2, 1, 1 in the correct order | same |

The cluster columns are what separates them once the residual is not
decisive: the true hit is a 1×5 cluster, exactly the length a track at
this incidence angle should leave (κ_v = 0.95, i.e. 95% of the expected
size); the other is 1×2 with κ_v = 0.38 and 60% of the expected charge. Doc
08 shows what the trained gate makes of the two. Standardisation (subtract the training mean, divide by the training
standard deviation) is applied to every feature except the three integer
history counters and `window_nsigma` (`NO_STANDARDIZE`), whose raw scale is
meaningful.

## The gate's label

`cckf.labels.derive_labels(table)` returns, for every row,

- `label_same_particle` = 1 if `branch_majority_pid` is in `contrib_pids`,
  else 0. A merged cluster that contains the branch's particle is a
  positive: the Kalman update from it still carries the right position.
- `label_ambiguous` = the cluster has more than one contributor (kept at
  full weight; `--drop-ambiguous` in training is the ablation).
- `gate_row_mask` = the row is a real candidate (`cand_hit_id ≠ −1`) on a
  branch whose particle is defined (`majority_undefined` false). About 90%
  of all rows fail this: most branches come from fake seeds, whose
  "particle" is not a well-posed notion. Hole rows have nothing to score.

The positive fraction after masking is about 0.5%: for every true hit in a
10σ window there are roughly 190 wrong ones. That imbalance is handled in
training (doc 08), not in the label.

**Pure seeds.** The majority rule accepts a seed with two of three hits from
one particle. `cckf.seed_purity.classify_seed_purity` marks branches whose
first three accepted measurements all belong to the majority particle as
`pure`; `--pure-seeds-only` on either cache builder keeps only those (about
3% of rows) for the ablation that asks whether the 2-of-3 branches hurt.

## The value function's 11 features

`VALUE_FEATURES`, one row per **state** (not per candidate), built by
`scripts/build_value_cache.py::_state_features`:

| Feature | Meaning |
|---|---|
| `eta`, `state_qop` | as above, from the predicted state |
| `sigma2_l0`, `sigma2_l1` | the predicted position variances (`cov_00`, `cov_06`): how lost the branch is |
| `n_hits`, `n_holes`, `n_seq_holes` | the history counters |
| `sum_gate_logodds` | Σ over accepted hits of log(Λ/(1−Λ)) with Λ = exp(−χ²/2), clipped to [10⁻⁶, 1 − 10⁻⁶] (`features.chi2_log_odds`): the accumulated evidence, from χ² and not from the gate's own logit, so training and C++ agree |
| `min_gate_logodds` | the worst accepted hit so far |
| `step_k` | position along the branch |
| `x0_accumulated` | radiation lengths crossed so far (**always 0 in the C++**; see doc 10) |

`VALUE_FEATURES_WINDOWED` adds `window_nsigma`, a constant per cache build,
for the re-propagated targets that depend on the rollout window (doc 09).
The C++ has no windowed path, so a 12-feature model cannot be deployed yet.

The value target is not a parquet column: `cckf/value_target.py` computes
it at cache-build time (doc 09).

## The event split

`cckf/splits.py` fixes which events go where; `configs/event_split.json`
records the rule. Events 0 to 31 are divided into train, val and cal
(roughly 24 / 4 / 4, by event, never by row, because rows within an event
share pileup and detector state). The cal split exists for the Platt fit
and the calibration audit only. Events 32 to 63 are the sealed set;
`splits.assert_not_test` refuses them everywhere, and
`cckf/event_selection.py` parses `--only-events` against the split the
caller named so a staged smoke cache cannot leak across splits.

## Building a cache

```
python scripts/build_gate_cache.py  --split train --parquet-dir $SCRATCH/cckf/reexpanded --out-dir $SCRATCH/cckf/caches_v4/train
python scripts/build_gate_cache.py  --split val   ...  --out-dir .../val
python scripts/build_gate_cache.py  --split cal   ...  --out-dir .../cal
python scripts/build_value_cache.py --split train --parquet-dir $SCRATCH/cckf/reexpanded_sel --csv-dir <stage-1 csvs> --out-dir $SCRATCH/cckf/vcache_v4/train
```

(`scripts/build_caches_nersc.py` does the three gate splits in one go;
`scripts/nersc/build_caches.sbatch` and `build_vcache.sbatch` are the jobs.)
Flags that matter: `--only-events` (a subset, marks the cache
`partial_split: true` so `train_gate.py` refuses it unless told),
`--pure-seeds-only`, and for the value builder `--targets-dir` plus
`--window-nsigma` to use re-propagated targets. The value builder needs
parquets with `is_ckf_selected` (the `_sel` directory, after
`scripts/patch_is_selected.py`), because "the hits accepted so far" is
otherwise not recoverable.

The builders stream the parquet in row groups of a million rows and never
hold an event in memory; the gate train cache for 24 envelope events is
243 M rows, 26 float32 columns, about 25 GB on disk.

### What a cache directory contains

| File | Gate cache | Value cache |
|---|---|---|
| `X.f32` | `(n_rows, 26)` float32, `GATE_FEATURES` order | `(n_rows, 11 or 12)` float32 |
| `y.u8` / `y.f32` | uint8 label ∈ {0, 1} | float32 soft target ∈ [0, 1] |
| `aux.f32` | `(n_rows, 3)`: `chi2_inc`, `n_window`, `eta`, unstandardised, for calibration strata and the χ² baseline | `(n_rows, 3)`: `vstar_t1`, `step_k`, `eta` |
| `ambiguous.u8` | merged-cluster flag | — |
| `meta.json` | `n_rows`, `n_positive`, `positive_fraction`, `n_features`, `feature_names`, `aux_columns`, `source_files`, `pure_seeds_only` (+ `partial_split`, `events_used`) | the same plus target statistics (`mean_vstar_*`, `marginal_fraction_0.2_0.8`, `n_tier_invariant_dropped`, `max_accepted_chi2`, …) and `window_nsigma` when windowed |
| `norm_stats.npz` | `mu`, `sigma`, `feature_names`; **train split only** | same |

`cckf.cache.load_cache(dir)` memmaps the arrays, so a training run reads
one batch at a time from disk. No row carries its event or branch id: if
you need to trace a row back, do it from the parquet (doc 06) with the
same filter.

## Choosing the features you train on

Both trainers take the full cache and select columns at load time; the
cache is never rebuilt for an ablation.

```
python scripts/train_gate.py  ... --feature-groups kalman,cluster_raw,cluster_norm,occupancy,context,sensor   # no history
python scripts/train_gate.py  ... --drop-features clus_sigma_uv,pathInX0_interval                             # single features
python scripts/train_value.py ... --drop-features x0_accumulated,min_gate_logodds
```

`--feature-groups` keeps whole groups of `GATE_GROUPS`; `--drop-features`
removes named features after that; the resolver
(`cckf.features.resolve_feature_columns`) always keeps the columns in the
canonical order, so the same ablation trains the same model however the
flags were typed. The checkpoint records `feature_names` (what it was
trained on) and `all_feature_names` (the cache's list); the exporter
(doc 10) pads dropped features back in with zero weights so the C++ never
changes. Read the ablation off the metrics file, not the run name:
`gate_metrics.json` carries `feature_names`.

Three ablations the spec asks for, and the flags that run them:

| Question | Command |
|---|---|
| Do the normalised cluster features add anything over the raw ones? (A4a) | `--feature-groups kalman,cluster_raw,occupancy,context,sensor,history` |
| Is the branch history doing the work, or the local evidence? (A4b) | `--feature-groups kalman,cluster_raw,cluster_norm,occupancy,context,sensor` |
| Does the value function need the accumulated evidence? | `train_value.py --drop-features sum_gate_logodds,min_gate_logodds` |

Compare on the val split with `scripts/eval_discrimination.py` (AUC-ROC,
AUC-PR) and on the cal split with `scripts/calibrate_and_audit.py` (ECE);
one variable per experiment, and the log entry names the dropped features.

## Tests

`tests/test_features.py` (every derived feature against hand-computed
values, the resolver), `test_labels.py`, `test_seed_purity.py`,
`test_splits.py`, `test_event_selection.py`, `test_cache.py` (batch
invariance, file-length consistency, the standardisation skip list),
`test_build_gate_cache.py`, `test_gate_cache_pure.py`,
`test_build_value_cache.py`, `test_value_cache_windowed.py`.
