# 06 — Expansion: from the CKF's log to training rows

Doc 05 showed what the CKF recorded: at each surface, the prediction and the
one measurement it took. To train a gate we also need the measurements it
*did not* take, with the same features, so that every decision becomes a set
of (candidate, label) rows. That is what `expansion.py` produces, one
parquet file per event. This document walks branch 128909 through it.

This replaces the older `data_schema.md`; the column list at the end is the
same information, condensed.

## What the expansion does

`run_expansion(trackstates_root, csv_dir, event_id, output_path, digi_config_path)`:

1. **Load the states** (`load_trackstates`): every track of the event, flat,
   one row per state, `state_idx` in propagation order (the reversal and the
   orientation check from doc 05 happen here). `sel_l0/sel_l1` are the
   accepted hit's `l_x_hit/l_y_hit`.
2. **Load the CSVs** of doc 02 (`load_measurements`, `load_cells`,
   `load_simhits`, `load_measurement_simhit_map`) and `predicted-cov.csv`
   (`load_predicted_cov`), clearing the endcap ring byte from every
   `geometry_id` on the way.
3. **Join states to candidates** (`expand_trackstates`): every measurement on
   the state's (volume, layer, module) is a candidate. For each, the residual
   r = measurement − prediction, S = V + P (measurement variance plus the
   projected predicted covariance), χ² = rᵀS⁻¹r. Keep the candidates inside
   the n·σ box (|r₀| ≤ n√S₀₀ and, for 2D sensors, |r₁| ≤ n√S₁₁; n = 10 for
   every collection run so far). A state with no candidate in the box gets
   one *hole row*.
4. **Mark the CKF's choice** (`is_ckf_selected`): the candidate whose
   `local0` equals `sel_l0` to 10⁻⁴ mm.
5. **Cluster features** (`compute_cluster_features_table`): from `cells.csv`,
   the cluster size in u and v, the total charge and the charge-weighted
   second moments of every candidate.
6. **Branch history** (`compute_branch_history`): for each state, how many
   hits and holes the branch had collected *before* it, in `step_k` order.
7. **The branch's particle** (`compute_branch_majority_pid`): the particle
   that owns at least two of the first three measurement states, i.e. the
   seed. Undefined when no particle does.
8. **Truth per candidate** (`compute_truth_labels`): the contributors to the
   candidate's cluster (`contrib_pids`, via the map and `simhits.csv`) and
   whether the branch's particle left a simhit on this surface at all
   (`majority_true_hit_on_surface`).
9. **Write** (`write_expanded_parquet`): 82 columns, `expanded_event000000004.parquet`.

The gate label is *not* stored; `cckf/labels.py::derive_labels` computes
`branch_majority_pid ∈ contrib_pids` when a cache is built, and drops rows
whose branch has no defined particle (about 90% of all rows: fake seeds
generate most branches).

## Branch 128909 in the parquet

`seed_id == 128909` selects 67 rows: 23 states, of which 11 are measurement
states with candidates (55 candidate rows) and 12 are holes or material
surfaces (12 hole rows, `cand_hit_id == −1`). Candidates per measurement
state: 27 on the seed surface (the 1 mm seed covariance, doc 03), 18 on
pixel layer 4, 2 on the outer long-strip disc, 1 everywhere else.

### The decision at pixel layer 4, module 361

All 18 candidates, sorted by χ²:

| cand_hit_id | residual l0 (mm) | residual l1 (mm) | χ² | contrib_pids | is_ckf_selected | label |
|---|---|---|---|---|---|---|
| 53230 | −0.0001 | 0.011 | 2e−5 | [281474977169414] | True | 1 |
| 53229 | −1.950 | −0.114 | 10.2 | [21392239964979754] | False | 0 |
| 53232 | −2.466 | 1.859 | 17.1 | [21110829413499187] | | 0 |
| 53228 | 2.700 | −8.682 | 34.3 | [57139420276064313] | | 0 |
| 53235 | −3.285 | 9.381 | 46.4 | [10977524093747511] | | 0 |
| 53222 | 1.600 | −15.855 | 53.5 | [56294995347767765] | | 0 |
| 53233 | 4.589 | 3.936 | 58.1 | [49539595906973785] | | 0 |
| 53231 | 4.680 | 0.952 | 58.5 | [3377699729637767] | | 0 |
| 53226 | −3.900 | −11.091 | 60.5 | [28991922605916231] | | 0 |
| 53220 | 2.338 | −16.520 | 65.9 | [52354345674408401] | | 0 |
| 53218 | −2.850 | −18.302 | 79.2 | [59672695070589048] | | 0 |
| 53216 | −1.930 | −20.571 | 84.0 | [16325548649677065] | | 0 |
| 53214 | 2.500 | −20.698 | 96.7 | [12103424006029425] | | 0 |
| 53224 | −5.050 | −13.454 | 97.1 | [9288674234531886] | | 0 |
| 53225 | 5.550 | −12.560 | 114.6 | [3377699735208159] | | 0 |
| 53212 | −3.663 | −22.591 | 123.3 | [3941001863103295] | | 0 |
| 53219 | 4.900 | −17.612 | 124.8 | [2533274791378958] | | 0 |
| 53215 | 4.518 | −20.671 | 136.6 | [35465847070982226] | | 0 |

Every row shares the state's columns: `S00` 0.3745, `S11` 5.560 (the box is
±6.1 mm by ±23.6 mm), `n_window` 18, `sel_l0` 0.325, `pitch_u` 0.05,
`is_pixel` 1, `branch_majority_pid` 281474977169414 (our particle, packed as
in doc 01), `majority_true_hit_on_surface` 1. The 18 rows differ in the
candidate columns: residuals, χ², `clus_s_u` (1 to 4 here), `clus_q_tot`,
`contrib_pids`. The label column, derived at cache time, is 1 for the first
row only; the χ² cut of 16.26 would also have passed the second.

### A hole row

Short-strip layer 8, module 86 (the real hole of doc 03): one row with
`cand_hit_id` −1, `n_window` 0, `action_taken` 1 (measurements existed on
the module, none inside the box; code 2 means the module had no
measurements at all), `S00` 0.0577 and `S11` 0.195 still filled from the
prediction, everything about a candidate NaN. The 11 material surfaces
(`module 0`, volume 20) look the same with `action_taken` 2.

## What the reversed `step_k` did to this branch

The example parquet was expanded before the 2026-09-08 fix, so its `step_k`
is the ROOT index. Three of the derived columns for branch 128909, as
stored versus as they will be after re-expansion (`step_k' = 22 − step_k`):

**History counters** at the pixel layer 4 state (`n_hits`, `n_holes`,
`n_seq_holes`, counted before the state):

| | stored (outermost-first) | correct (seed-first) |
|---|---|---|
| step_k | 19 | 3 |
| n_hits | 8 | 2 |
| n_holes | 11 | 1 |
| n_seq_holes | 1 | 1 |

The stored row says "this branch has 8 hits and 11 holes behind it" for a
state that is the third the CKF ever created. The C++ at inference counts
from the seed, so the trained gate and value function saw history features
from a distribution the deployed models never encounter.

**The branch's particle** was taken from the outermost three measurement
states (the wrong hit on 28/12 and the two strip hits on layer 6) instead of
the first three (pixel 110, 12, 361). Here both give particle 482, by 2 of 3
instead of 3 of 3; on branches that go wrong late, the stored label is the
wrong particle.

**The value target** (`cckf/value_target.py`; NaN in the parquet, computed
at cache time). At the pixel layer 4 state the correct target counts 2
correct hits behind, 0 wrong, and 8 findable ahead (all remaining surfaces
where the particle left a simhit), against N_total_true = 10 simhits:
completeness 10/10, purity 10/10, V = 1.0. The stored order puts the wrong
hit in the past (n_wrong 1) and the pixels in the future: V = min(10/10,
10/11) = 0.91. Mild here; a branch that picks up several wrong hits at the
end is scored as if it had started badly and recovered.

`n_holes` also counts the 11 material surfaces, in both orders. The C++
counts only `stateType == 2` holes, so the training history overstates
holes by about one per volume crossing. Doc 05 established that `stateType`
is in the ROOT file; reading it in `load_trackstates` and excluding
material states from `compute_branch_history` (or emitting no row for them)
is the fix. It is not in the code yet.

## The 82 columns

`expansion.SCHEMA_COLUMNS`, grouped. One row per (branch, state, candidate).

| Group | Columns | Notes |
|---|---|---|
| identity | `event_id`, `seed_id`, `branch_id`, `parent_branch_id`, `step_k`, `volume_id`, `layer_id`, `surface_id` | `seed_id == branch_id == track_nr`; `parent_branch_id` always −1; `surface_id` is the module number |
| predicted state | `state_l0, state_l1, state_phi, state_theta, state_qop, state_t`, `cov_00 … cov_20`, `pred_l0`, `pred_l1` | `cov_*` is the lower triangle of the 6×6 covariance; only the six diagonals (`cov_00, 06, 11, 15, 18, 20`) are filled |
| candidate | `cand_hit_id`, `residual_l0`, `residual_l1`, `S00`, `S01`, `S11`, `chi2_inc`, `var_local0`, `var_local1`, `is_1d`, `n_window`, `geometric_density` | `cand_hit_id` is the `measurement_id`; `residual_l1`, `S01`, `S11` NaN for 1D strips; `geometric_density` counts measurements within 5 mm |
| cluster | `clus_s_u`, `clus_s_v`, `clus_q_tot`, `clus_sigma_uu`, `clus_sigma_uv`, `clus_sigma_vv` | from `cells.csv`; sizes in channels, moments in channel² |
| incidence, sensor | `alpha_u`, `alpha_v`, `pitch_u`, `pitch_v`, `thickness`, `is_pixel`, `is_barrel`, `pathInX0_interval`, `dead_module_flag`, `layer_embed_idx` | `is_barrel` is wrong (derived from a constant that names the endcaps); `dead_module_flag` always 0 |
| history | `n_hits`, `n_holes`, `n_seq_holes` | counted before this state in `step_k` order; holes include material surfaces (above) |
| action | `action_taken`, `is_ckf_selected`, `sel_l0`, `sel_l1`, `prune_reason` | codes 0 candidate / 1 hole, measurements outside box / 2 hole, empty module; `prune_reason` null |
| truth | `contrib_pids`, `contrib_charge_frac`, `branch_majority_pid`, `majority_undefined`, `majority_true_hit_on_surface`, `truth_residual_l0`, `truth_residual_l1`, `vstar_soft` | `truth_residual_*` and `vstar_soft` are NaN placeholders |
| provenance | `env_config_hash` | SHA256 of the collection config |

Sizes: event 4 of the envelope run is 152 M rows, 41.5 M states, 1.69 M
branches, 5.8 GB. The tight config's events are a few percent of that.

## Reading it

```python
import pyarrow.parquet as pq
cols = ["seed_id", "step_k", "volume_id", "layer_id", "surface_id", "cand_hit_id",
        "residual_l0", "residual_l1", "chi2_inc", "n_window", "is_ckf_selected",
        "contrib_pids", "branch_majority_pid", "n_hits", "n_holes"]
d = pq.read_table("expanded_event000000004.parquet", columns=cols,
                  filters=[("seed_id", "==", 128909)]).to_pandas()
d = d.sort_values(["step_k", "chi2_inc"])
```

Filtering on `seed_id` through `pq.read_table` reads only the row groups
that contain it, so this is fast on a login node; reading whole columns for
the event needs a compute node. `scripts/trail_trackstates.py` (doc 05)
prints the branch table and the parquet rows together.

After any re-expansion, run the audit before anything downstream:

```
python scripts/audit_expansion.py --parquet expanded_event000000004.parquet \
    --trackstates trackstates_ckf.root --event 4 --json audit_4.json
```

It must print `AUDIT PASS` on nine checks (state order against ROOT,
surfaces, volumes present, hole fraction, candidate shares, history
counters, majority label, selected flag, target range). On the pre-fix
example file it fails two of them, on purpose.

## Downstream

`scripts/build_gate_cache.py` and `scripts/build_value_cache.py` turn the
parquets into memmapped feature caches per split (the feature lists are
`cckf/features.py::GATE_FEATURES`, 26 columns, and `VALUE_FEATURES`, 11);
`scripts/train_gate.py` and `scripts/train_value.py` train on them;
`scripts/export_weights.py` writes the binary blobs the C++ loads. The gate
learns, from rows like the 18 above, to tell the first row from the second.
