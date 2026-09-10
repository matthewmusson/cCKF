# Data Schema Reference: Stage 1 outputs and the expanded Parquet

Schema of every file produced by Stage 1 (digitisation + CKF with all writers
enabled), how the expansion pipeline joins them into the per-event Parquet,
and every convention that has bitten us. Revised 2026-09-10; supersedes the
2026-08 pilot version. Where a bug changed a definition, the entry says so
and names the log date.

**Worked example on NERSC (group-readable):**
`/global/cfs/cdirs/atlas/mussonm/cckf_handoff/examples/envelope_event4/` holds
one complete Stage 1 run directory (`stage1/`, events 4 and 5 of ColliderML
ttbar run 0, `configs/cckf_envelope.yaml`, `window_n=10`) and the expanded
Parquet for event 4. `examples/regen_tight/` and `examples/regen_fast/` are
smaller complete runs (8 events each) under the tight and fast operating
points, with their Parquets under `reexpanded/`.

---

## 0. Conventions that apply everywhere

| Convention | Rule | Where enforced | Log |
|---|---|---|---|
| **State order** | ROOT stores each track's states **outermost-first** (`RootTrackStatesWriter` iterates `trackStatesReversed()`). The Parquet's `step_k` is **propagation order**: 0 at the seed surface, largest at the outermost. Every "past / future along the branch" quantity assumes this. | `expansion.propagation_order_index` (the only conversion); `check_propagation_order` refuses a wrongly oriented file using `pathLength` | 2026-09-08 |
| **Geometry id** | ODD endcap volumes write the ring index (1-3) into the low `extra` byte of `geometry_id` in every CSV; the ROOT tree has no such field. All CSV loaders zero the byte at read time, so every join runs in `extra = 0` space. `(volume, layer, sensitive)` is unique across all 18,918 surfaces. | `expansion.normalize_geometry_id` | 2026-08-25 |
| **Selected hit** | Which candidate the CKF accepted is found by matching the candidate's measurement position to the ROOT `l_x_hit / l_y_hit` at the same `(track, state)`, never by residual coincidence. | `expand_trackstates` (inline, `SEL_MATCH_TOL = 1e-4` mm); `scripts/patch_is_selected.py` for older Parquets | 2026-08-17, 2026-08-25 |
| **Innovation covariance** | `S = V + H P Hᵀ` is built per candidate from the candidate's own variance `V` (measurements.csv) and the predicted local covariance `P` (predicted-cov.csv). The ROOT `S*_prt` branches are not used: `S11_prt` is corrupted on 34-35% of rows in events 0-3. | `expand_trackstates` | 2026-08-13 |
| **Identifiers** | `seed_id == branch_id == track_nr`: the Parquet's unit is one surviving CKF output track. `parent_branch_id` and `prune_reason` are unpopulated placeholders (-1 / null); branches pruned in flight left no states. | schema alias map | 2026-09-02 |
| **pathLength** | Not monotone along a branch: the CKF resets its stepper when it resumes a forked branch (82% of event-4 tracks carry a drop). Only the orientation (seed at 0, outermost above it) is an invariant. | `check_propagation_order` | 2026-09-08 |
| **Merges on ids** | Left-merges on int64 id columns coerce to float64 on misses and silently zero the low bytes of 19-digit ids. Use nullable `Int64` for id columns. | throughout | 2026-08-26 |

ODD volumes as they appear in these files:

| Volume | Sub-detector | Measurement |
|---|---|---|
| 16 / 17 / 18 | pixel endcap − / **barrel** / endcap + | 2D |
| 20 | beam pipe / passive material | none: every state here is a hole by construction (about one per branch) |
| 23 / 24 / 25 | short strip endcap − / **barrel** / endcap + | 2D (stereo pair) |
| 28 / 29 / 30 | long strip endcap − / **barrel** / endcap + | **1D** (`is_1d`) |

Layer ids increase with radius in barrels and along +z in endcaps, so on the
−z side the disc nearest the barrel has the *highest* layer id. Even layer ids
are sensitive layers, odd ones approach/representing surfaces.

---

## 1. trackstates_ckf.root

One TTree (`trackstates`), one entry per track, jagged per-state arrays.
Event 4 of the envelope run: 1,691,023 tracks, 41,506,348 states.

### Branches the pipeline uses

| Branch | Type | Description |
|---|---|---|
| `event_nr` | uint32 | event index (absent in single-event files: the loader then treats the whole file as the requested event) |
| `volume_id`, `layer_id`, `module_id` | uint32[] | surface of each state; `module_id = GeometryIdentifier::sensitive()` |
| `pathLength` | float[] | accumulated path length; **used only for the orientation guard** (see §0) |
| `predicted`, `filtered`, `smoothed` | bool[] | which parameter sets were stored |
| `eLOC0/eLOC1/ePHI/eTHETA/eQOP/eT_prt` | float[] | predicted state; `_flt` the filtered state (used by the tier-3 rollout worklist, predicted as fallback at holes) |
| `err_e*_prt` | float[] | σ of the predicted state (diagonal only) |
| `l_x_hit`, `l_y_hit` | float[] | local position of the **CKF-accepted** measurement; NaN at holes. All-state branches: index-parallel with `volume_id`. This is what marks `is_ckf_selected`. |
| `err_x_hit`, `err_y_hit` | float[] | σ of that measurement |
| `pathInX0_interval` | float[] | material since the previous surface (patched branch) |
| `alpha_u`, `alpha_v` | float[] | incidence angles atan2(dU,\|dN\|), atan2(dV,\|dN\|) (patched) |
| `clus_size_u/v`, `clus_qtot`, `clus_sigma_uu/uv/vv` | int/float[] | cluster shape of the accepted hit (patched); −1 / NaN at holes |
| `S00_prt`, `S01_prt`, `S11_prt` | float[] | innovation covariance at the accepted hit (patched). **Read for reference only; see §0.** |
| `particle_ids_{vertex_primary,vertex_secondary,particle,generation,sub_particle}` | uint32[][] | contributors to the accepted hit, per state |
| `chi2` | float[] | accumulated χ² at the accepted hit |

Measurement-only branches (`res_eLOC0_prt`, `pull_*`, …) have a different
jagged length from the all-state branches and must never be index-joined to
them. `res_eLOC0_prt` is **not** `l_x_hit − eLOC0_prt`.

Particle ids are packed as `(pv << 48) | (sv << 32) | (part << 16) | (gen << 8) | sub`
(`expansion.encode_particle_id`); the same packing is used by
`TruthRolloutSelector.hpp` so worklist ids compare directly.

The `stateType` of each state (measurement / hole / material / outlier) is
**not** persisted by the writer. Consequence: material and overlap surfaces
inside sensor volumes are indistinguishable from real holes in the Parquet
(median 10 such states per branch; 24-31 "sensor holes" on ≥ 9-hit branches).
Persisting it is on the handoff list.

### Per-state ordering (see §0)

ROOT index 0 is the outermost state; the last index is the seed surface
(`pathLength` ≈ 2500 mm at index 0, 0 at the last index, on 100% of event-4
tracks). `load_trackstates` returns `state_idx` in propagation order.

---

## 2. measurements.csv

All digitised measurements for one event (`CsvMeasurementWriter`).
`event{N:09d}-measurements.csv`, about 150k-200k rows per event.

| Column | Type | Description |
|---|---|---|
| `measurement_id` | int | 0-based index; this is `cand_hit_id` in the Parquet and the `IndexSourceLink` index in the C++ |
| `geometry_id` | int64 | packed GeometryIdentifier **with the ring byte set in endcaps**; normalised on read |
| `local_key` | int | which local coordinates are measured; 1D long-strip rows have no `local1` |
| `local0`, `local1` | float | measured local coordinates (mm) |
| `var_local0`, `var_local1` | float | measurement variance `V` (mm²); `var_local1` absent/NaN for 1D |

---

## 3. cells.csv

Per-channel activations behind every cluster (`CsvMeasurementWriter`).
`event{N:09d}-cells.csv`; several rows per `measurement_id`.

| Column | Type | Description |
|---|---|---|
| `geometry_id` | int64 | as above (normalised on read) |
| `measurement_id` | int | joins to measurements.csv |
| `channel0`, `channel1` | int | channel indices along local u / v |
| `timestamp` | int | unused (geometric digitisation) |
| `value` | float | cell charge |

Cluster features per measurement: `s_u = max(ch0) − min(ch0) + 1`, `s_v`
likewise, `Q_tot = Σ value`, and the charge-weighted second central moments
`σ²_uu, σ²_uv, σ²_vv` in channel units (`compute_cluster_features_table`).

---

## 4. simhits.csv

Geant4 truth hits (`CsvSimHitWriter`). `event{N:09d}-simhits.csv`.
The raw file has **five** particle-id columns (`particle_id_pv, _sv, _part,
_gen, _subpart`) plus `geometry_id, tx, ty, tz, tt, tpx, tpy, tpz, te,
deltapx/py/pz, deltae, index`. `expansion.load_simhits` returns
`hit_id, geometry_id, particle_id (packed), tx, ty, tz`; `hit_id` is the
0-based row number (verified equal to `index` on event 4, all 273,885 rows).
The particle's pT is `hypot(tpx, tpy)` from the raw file; `tt` orders a
particle's hits in time.

`N_total_true` (the value target's denominator) counts a particle's rows in
this file, i.e. **simhits**, not measurements.

---

## 5. measurement-simhit-map.csv

`measurement_id → hit_id`, many-to-many (charge sharing, merged clusters).
`event{N:09d}-measurement-simhit-map.csv`. The truth chain for a measurement
is map → simhits → packed `particle_id`; a candidate's `contrib_pids` is the
list of all contributors and `contrib_charge_frac` their charge shares.

---

## 6. predicted-cov.csv  (cCKF addition)

Written per state by `utils/predicted_cov_writer.PredictedCovWriter`
(`event{N:09d}-predicted-cov.csv`, about 2.5 GB per envelope event).

| Column | Type | Description |
|---|---|---|
| `track_nr` | int | = `seed_id` |
| `step_k` | int | **ROOT order** (the writer iterates `trackStatesReversed`, counting every state, predicted or not); converted to propagation order by `load_predicted_cov` using the ROOT per-track state count |
| `eLOC0_prt`, `eLOC1_prt` | float | predicted local position |
| `P00`, `P01`, `P11` | float | predicted covariance projected to local coordinates, i.e. the `H C Hᵀ` term |

States without a prediction have no row, which is why the CSV's own row
count cannot recover a track's length.

---

## 7. detectors.csv

Sensor geometry (`CsvTrackingGeometryWriter`, one file, not per event):
`geometry_id, geometry_type, volume_id, layer_id, module_id, cx, cy, cz,
rot_*, pitch_u, pitch_v, thickness`. Endcap ids here carry `extra = 0`, unlike
the per-event CSVs; the tier-3 worklist therefore resolves true surface ids
from `measurements.csv` first and falls back to this file.

---

## Geometry id encoding

```
Bit 63    56 55    48 47    36 35       8 7     0
 ┌────────┬────────┬─────────┬──────────┬───────┐
 │ volume │boundary│  layer  │sensitive │ extra │
 │ 8 bits │ 8 bits │ 12 bits │ 20 bits  │ 8 bits│
 └────────┴────────┴─────────┴──────────┴───────┘
```

`encode_geometry_id(vol, lay, mod) = vol<<56 | lay<<36 | mod<<8` (extra = 0).
CSV ids from endcaps have `extra ∈ {1,2,3}`; `normalize_geometry_id` clears it
before any join. Decoding: `vol = id>>56 & 0xFF`, `lay = id>>36 & 0xFFF`,
`mod = id>>8 & 0xFFFFF`.

---

## The expanded Parquet (82 columns, `expansion.SCHEMA_COLUMNS`)

One row per (branch, surface, in-window candidate); one hole row when a
surface has no in-window candidate. Event 4 of the envelope run: 152M rows,
41.5M states, 1.69M branches, 5.8 GB. Paths use `expanded_event{E:09d}.parquet`.

### Identity

| Column | Description |
|---|---|
| `event_id` | event index |
| `seed_id`, `branch_id` | both = ROOT `track_nr` (see §0) |
| `parent_branch_id` | placeholder, −1 |
| `step_k` | **propagation order**, 0 at the seed surface |
| `volume_id`, `layer_id`, `surface_id` | surface; `surface_id` is the ROOT `module_id` (sensitive field), not a packed id |

### Predicted state and covariance

`state_l0 … state_t` are the predicted parameters (`e*_prt`). `cov_00 … cov_20`
hold the lower triangle of the 6×6 predicted covariance; only the six diagonal
entries (`cov_00, 06, 11, 15, 18, 20`) are populated from the ROOT σ's, the
rest are NaN. `pred_l0`, `pred_l1` repeat the predicted local position used
for the window.

### Candidate measurement

| Column | Description |
|---|---|
| `cand_hit_id` | `measurement_id`; −1 on hole rows |
| `residual_l0`, `residual_l1` | candidate − prediction (mm); `residual_l1` NaN for 1D |
| `S00`, `S01`, `S11` | per-candidate innovation covariance `V + P` (see §0); `S11`, `S01` NaN for 1D |
| `chi2_inc` | `rᵀ S⁻¹ r` with the full 2×2 (1×1 for 1D) |
| `var_local0`, `var_local1` | the candidate's measurement variance |
| `is_1d` | long-strip (volumes 28/29/30) one-dimensional measurement |
| `n_window` | candidates inside the n·σ box on this surface (occupancy) |
| `geometric_density` | measurements within `r_geom_mm` of the prediction |

Window membership: `|r0| ≤ n·√S00` and (`is_1d` or `|r1| ≤ n·√S11`), plus the
geometric radius. `n = 10` for every collection run to date.

### Cluster, incidence, sensor

`clus_s_u, clus_s_v, clus_q_tot, clus_sigma_uu/uv/vv` from cells.csv for every
candidate (NaN on hole rows); `alpha_u, alpha_v` per surface;
`pitch_u, pitch_v, thickness` from the digitisation config; `is_pixel`
(volumes 16-18). **`is_barrel` is unreliable**: the constant it is derived
from names the endcap volumes; treat it as absent (task on the handoff list).
`pathInX0_interval`, `dead_module_flag` (always 0), `layer_embed_idx`.

### Branch history (counted **before** the current state, in `step_k` order)

`n_hits`, `n_holes`, `n_seq_holes`. A state is a hole when `n_window == 0`,
which **includes passive volume-20 crossings and material surfaces**; the C++
counts `isHole()` states only, so training and inference differ by about one
hole per branch past the beam pipe. Known, not yet reconciled (log 2026-09-04).

### Action and selection

| Column | Description |
|---|---|
| `action_taken` | 0 candidate row, 1 hole with measurements outside the window, 2 hole with no measurements on the surface |
| `is_ckf_selected` | this candidate is the one the CKF accepted (position match to `l_x_hit`, §0); at most one per state |
| `sel_l0`, `sel_l1` | the accepted hit's position from ROOT, repeated on every row of the state |
| `prune_reason` | placeholder, null |

### Truth

| Column | Description |
|---|---|
| `contrib_pids`, `contrib_charge_frac` | contributors to this candidate's cluster (packed ids) and their charge shares |
| `branch_majority_pid`, `majority_undefined` | mode of the accepted-hit primary particles over the branch's **first three measurement states in propagation order** (the seed); undefined when fewer than 2 of 3 agree. Before 2026-09-08 this was silently taken from the outermost three. |
| `majority_true_hit_on_surface` | the majority particle left a **simhit** on this surface (simhit-level, so material surfaces and overlap modules count) |
| `truth_residual_l0/l1` | NaN placeholders (need the global→local transform) |
| `vstar_soft` | NaN in the Parquet; value targets are computed by `scripts/build_value_cache.py` (`cckf.value_target`) and, for re-propagated targets, by `scripts/stitch_tier3.py` |
| `env_config_hash` | SHA256 of the collection config |

Gate labels (`label_same_particle`, `label_ambiguous`) are **not** stored;
`cckf.labels.derive_labels` computes them from `contrib_pids` against
`branch_majority_pid` at cache-build time.

---

## Pipeline data flow

```
trackstates_ckf.root ──load_trackstates──► states (propagation order; pathLength
        │                                   orientation guard)
        │                                   sel_l0/sel_l1 from l_x_hit/l_y_hit
predicted-cov.csv ──load_predicted_cov──► P00/P01/P11 (ROOT-order step_k converted
        │                                   with the per-track state count)
measurements.csv ─┐  normalize_geometry_id (endcap ring byte cleared)
cells.csv ────────┤
simhits.csv ──────┤
meas-simhit-map ──┘
        ▼
  expand_trackstates()      join states ↔ measurements on (vol, lay, sensitive)
                            S = V + P per candidate; window |r| ≤ n√S; chi2_inc
                            one row per candidate, one hole row otherwise
                            is_ckf_selected by position match
        ▼
  compute_branch_history()  n_hits / n_holes / n_seq_holes in step_k order
  compute_cluster_features_table()   cells → s_u, s_v, Q_tot, σ² moments
  compute_branch_majority_pid()      seed = innermost three measurement states
  compute_truth_labels()             contrib_pids, majority_true_hit_on_surface
        ▼
  expanded_event{E:09d}.parquet  (82 columns)
        ▼
  scripts/audit_expansion.py  ── must print AUDIT PASS (order, surfaces vs ROOT,
                                 volumes, hole fraction, shares, history,
                                 majority label, selected flag, target range)
```

Downstream consumers: `scripts/build_gate_cache.py` (gate features + labels),
`scripts/build_value_cache.py` (value features + `V^{π†}` targets),
`scripts/winfail_uncensored.py` (window/module failure), `cckf/tier3_walker.py`
(rollout worklists).
