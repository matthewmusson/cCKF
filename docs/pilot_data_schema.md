# Pilot Data Schema Reference

Schema of every file produced by Stage 1 (envelope CKF on 2 events with all
outputs enabled), and how the expansion pipeline joins them into the
post-expansion Parquet.

Source: pilot run at `/data/results/pilot_1786472672` on Modal (2 events,
`envelope.yaml` config, `window_n=10`, `ambi=False`).

---

## 1. trackstates_ckf.root

One ROOT TTree (`trackstates`) with **1,786,938 entries** across 2 events.
Each entry is one trackstate (a Kalman filter step on one track).

### Key branches used by the expansion pipeline

| Branch | Type | Example values | Description |
|--------|------|---------------|-------------|
| `event_nr` | uint32 | 0, 1 | Event index |
| `track_nr` | uint32 | 0, 1, 2, ... | Track index within event |
| `volume_id` | uint32 | 16, 17, 18, 23, 24, 25, 28 | ODD volume (16/17/18 = pixel, 23-28 = strip) |
| `layer_id` | uint32 | 2, 4, 6, 8, 10, 12, 14 | Layer within volume (even = sensitive, odd = approach) |
| `module_id` | uint32 | 1–2048 | Sensitive surface within layer (= `geoID.sensitive()`) |
| `predicted` | bool | true, false | Whether predicted state was stored |
| `filtered` | bool | true, false | Whether filtered state was stored |
| `smoothed` | bool | true, false | Whether smoothed state was stored |

### Predicted state vector

| Branch | Type | Example | Description |
|--------|------|---------|-------------|
| `eLOC0_prt` | float | -11.34 | Predicted local0 (mm) |
| `eLOC1_prt` | float | -62.47 | Predicted local1 (mm) |
| `ePHI_prt` | float | -2.372 | Predicted φ (rad) |
| `eTHETA_prt` | float | 2.998 | Predicted θ (rad) |
| `eQOP_prt` | float | -0.114 | Predicted q/p (1/GeV) |
| `eT_prt` | float | 1567.4 | Predicted time (ns) |

### Predicted errors (sqrt of covariance diagonal)

| Branch | Type | Example | Description |
|--------|------|---------|-------------|
| `err_eLOC0_prt` | float | 10.86 | σ(l₀) predicted (mm) |
| `err_eLOC1_prt` | float | 0.755 | σ(l₁) predicted (mm) |
| `err_ePHI_prt` | float | 0.0968 | σ(φ) predicted (rad) |
| `err_eTHETA_prt` | float | 6.36e-4 | σ(θ) predicted (rad) |
| `err_eQOP_prt` | float | 0.103 | σ(q/p) predicted (1/GeV) |
| `err_eT_prt` | float | 2997.9 | σ(t) predicted (ns) |

### Innovation covariance (patched branches)

These are S = H·C_{k+1|k}·H' + R evaluated at the CKF-selected hit.

| Branch | Type | Example | Description |
|--------|------|---------|-------------|
| `S00_prt` | float | 117.98 | S[0,0] — local0 innovation variance (mm²) |
| `S01_prt` | float | 1.605 | S[0,1] — local cross-covariance (mm²) |
| `S11_prt` | float | 0.591 | S[1,1] — local1 innovation variance (mm²) |

S includes measurement noise R. Window half-widths: Δl₀ = n·√S₀₀, Δl₁ = n·√S₁₁.

NaN when `predicted == false` (states with no prediction, e.g. the seed state).

### Cluster features of CKF-selected hit (patched branches)

| Branch | Type | Example | Description |
|--------|------|---------|-------------|
| `clus_size_u` | int | 1 | Cluster extent in local-u channels |
| `clus_size_v` | int | 1 | Cluster extent in local-v channels |
| `clus_qtot` | float | 0.202 | Total cluster charge (activation sum) |
| `clus_sigma_uu` | float | 3.16e-30 | Charge-weighted 2nd moment σ²_uu (mm²) |
| `clus_sigma_uv` | float | 0.0 | Cross moment σ²_uv (mm²) |
| `clus_sigma_vv` | float | 0.0 | σ²_vv (mm²). Zero when cluster size = 1. |

-1 for `clus_size_u/v` and NaN for others when no hit was selected (hole states).

### Incidence angles + material (patched branches)

| Branch | Type | Example | Description |
|--------|------|---------|-------------|
| `alpha_u` | float | 0.00309 | Incidence angle in local-u (rad), atan2(dU, |dN|) |
| `alpha_v` | float | 0.143 | Incidence angle in local-v (rad), atan2(dV, |dN|) |
| `pathInX0_interval` | float | 0.003–0.05 | Material traversed since last surface (X/X₀) |

### Hit position (CKF-selected measurement)

| Branch | Type | Example | Description |
|--------|------|---------|-------------|
| `l_x_hit` | float | 13.00 | Measurement local0 (mm) |
| `l_y_hit` | float | -63.69 | Measurement local1 (mm) |
| `err_x_hit` | float | 0.0231 | Measurement σ(local0) (mm) — sqrt(R[0,0]) |
| `err_y_hit` | float | 0.417 | Measurement σ(local1) (mm) — sqrt(R[1,1]) |

### Particle truth

| Branch | Type | Example | Description |
|--------|------|---------|-------------|
| `particle_ids_vertex_primary` | vector<uint32> | [0] | Primary vertex index |
| `particle_ids_vertex_secondary` | vector<uint32> | [0] | Secondary vertex index |
| `particle_ids_particle` | vector<uint32> | [213] | Particle number within vertex |
| `particle_ids_generation` | vector<uint32> | [0] | Generation (0=primary) |
| `particle_ids_sub_particle` | vector<uint32> | [0] | Sub-particle index |

These are per-state vectors of contributing particle IDs. Encoded as int64:
`(pv << 48) | (sv << 32) | (part << 16) | (gen << 8) | subpart`.

### Filtered/smoothed/unbiased state vectors

Same naming pattern with `_flt`, `_smt`, `_ubs` suffixes. Also `chi2` (accumulated
chi² at the selected hit), `eta_prt/flt/smt`, `pT_prt/flt/smt`, global positions
`g_x/y/z_prt/flt/smt/hit`, residuals `res_eLOC0/1_prt/flt/smt`, pulls
`pull_eLOC0/1_prt/flt/smt`, truth state `t_eLOC0/1`, `t_ePHI/eTHETA/eQOP/eT`.

Total: **191 branches** per entry.

---

## 2. measurements.csv

All digitized measurements for one event. Written by `CsvMeasurementWriter`.
One row per measurement (cluster centroid after digitization).

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `measurement_id` | int | 0, 1, 2, ... | Unique measurement index (0-based within event) |
| `geometry_id` | int64 | 1152922604118474752 | Packed ACTS GeometryIdentifier (see §Geometry ID below) |
| `local_key` | int | 1 | Measurement key (always 1 for 2D measurements) |
| `local0` | float | 12.999 | Local coordinate u (mm) |
| `local1` | float | -63.691 | Local coordinate v (mm) |
| `var_local0` | float | 5.33e-4 | Measurement variance σ²_u (mm²) = R[0,0] |
| `var_local1` | float | 0.1739 | Measurement variance σ²_v (mm²) = R[1,1] |

Filename pattern: `event{N:09d}-measurements.csv`

Typical count: ~150,000–200,000 measurements per event (μ=200 pileup).

---

## 3. cells.csv

Per-channel (pixel/strip) activation data for every cluster. Written by `CsvMeasurementWriter`.

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `geometry_id` | int64 | 1152922604118474752 | Same packed GeometryIdentifier as measurements.csv |
| `measurement_id` | int | 0 | Links to measurements.csv row |
| `channel0` | int | 127 | Channel index along local-u |
| `channel1` | int | 564 | Channel index along local-v |
| `timestamp` | int | 0 | Timestamp bin (unused in geometric digi) |
| `value` | float | 0.202 | Cell activation / charge deposit |

Filename pattern: `event{N:09d}-cells.csv`

Multiple rows per measurement_id (one per activated cell in the cluster).
Cluster features are computed from these:
- `s_u = max(channel0) - min(channel0) + 1`
- `s_v = max(channel1) - min(channel1) + 1`
- `Q_tot = sum(value)`
- `σ²_uu, σ²_uv, σ²_vv` = charge-weighted second central moments in channel space

---

## 4. simhits.csv

Geant4 simulated hits (truth-level energy deposits). Written by `CsvSimHitWriter`.

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `geometry_id` | int64 | 1152922604118474752 | Surface where the hit occurred |
| `particle_id` | int64 | 0 or encoded | Particle that produced this hit |
| `tx` | float | -32.45 | True position x (mm, global) |
| `ty` | float | 87.12 | True position y (mm, global) |
| `tz` | float | -210.3 | True position z (mm, global) |
| `tt` | float | 1.23 | True time (ns) |
| `tpx` | float | 0.342 | True momentum px (GeV) |
| `tpy` | float | -1.205 | True momentum py (GeV) |
| `tpz` | float | 3.456 | True momentum pz (GeV) |
| `te` | float | 3.78 | True energy (GeV) |
| `deltapx` | float | -0.001 | Momentum change Δpx (GeV) |
| `deltapy` | float | 0.002 | Momentum change Δpy (GeV) |
| `deltapz` | float | -0.003 | Momentum change Δpz (GeV) |
| `deltae` | float | 0.004 | Energy deposit ΔE (GeV) |
| `index` | int | 0, 1, 2, ... | Row index (= hit_id for the truth chain) |

Filename pattern: `event{N:09d}-simhits.csv`

The `index` column (or equivalently the 0-based row number) is the `hit_id`
referenced by `measurement-simhit-map.csv`.

---

## 5. measurement-simhit-map.csv

Links digitized measurements to their contributing simulated hits.
Written by `CsvMeasurementSimHitMapWriter`.

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `measurement_id` | int | 0, 1, 2, ... | Index into measurements.csv |
| `hit_id` | int | 0, 1, 2, ... | Row index into simhits.csv |

Filename pattern: `event{N:09d}-measurement-simhit-map.csv`

Many-to-many: one measurement can have multiple contributing simhits
(charge sharing, pile-up merging), and one simhit can contribute to
multiple measurements.

### Truth chain example

To find the true particle(s) for measurement 42:
1. `measurement-simhit-map.csv`: find rows with `measurement_id == 42` → get `hit_id`s
2. `simhits.csv`: look up those `hit_id`s (row indices) → get `particle_id` values

---

## 6. detectors.csv

Sensor properties for every sensitive module. Written by `CsvTrackingGeometryWriter`
with `writePerEvent=False` (single file, not per-event).

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `geometry_id` | int64 | 1152922604118474752 | Packed GeometryIdentifier |
| `geometry_type` | int | 4 | Surface type enum |
| `volume_id` | int | 16 | Volume |
| `layer_id` | int | 2 | Layer |
| `module_id` | int | 1 | Module |
| `cx` | float | -32.45 | Module center x (mm) |
| `cy` | float | 87.12 | Module center y (mm) |
| `cz` | float | -210.3 | Module center z (mm) |
| `rot_xu` | float | 0.98 | Rotation matrix element |
| `rot_xv` | float | -0.02 | Rotation matrix element |
| `rot_xw` | float | 0.01 | Rotation matrix element |
| `rot_yu` | float | 0.02 | Rotation matrix element |
| `rot_yv` | float | 0.99 | Rotation matrix element |
| `rot_yw` | float | -0.01 | Rotation matrix element |
| `rot_zu` | float | -0.01 | Rotation matrix element |
| `rot_zv` | float | 0.01 | Rotation matrix element |
| `rot_zw` | float | 0.99 | Rotation matrix element |
| `pitch_u` | float | 0.050 | Pixel/strip pitch in u (mm) |
| `pitch_v` | float | 0.050 | Pixel/strip pitch in v (mm) |
| `thickness` | float | 0.15 | Sensor thickness (mm) |

Filename: `detectors.csv` (not per-event)

---

## Geometry ID Encoding

All CSV files use a packed 64-bit `geometry_id`. The ACTS `GeometryIdentifier` bit layout:

```
Bit 63    56 55    48 47    36 35       8 7     0
 ┌────────┬────────┬─────────┬──────────┬───────┐
 │ volume │boundary│  layer  │sensitive │ extra │
 │ 8 bits │ 8 bits │ 12 bits │ 20 bits  │ 8 bits│
 └────────┴────────┴─────────┴──────────┴───────┘
```

Encoding from trackstates ROOT columns to match CSV geometry_id:
```python
geometry_id = (volume_id << 56) | (layer_id << 36) | (module_id << 8)
```

Decoding from CSV geometry_id:
```python
volume_id  = (geometry_id >> 56) & 0xFF
layer_id   = (geometry_id >> 36) & 0xFFF
module_id  = (geometry_id >>  8) & 0xFFFFF    # = sensitive field
```

In the trackstates ROOT, `module_id` = `geoID.sensitive()` (the 20-bit field).
Boundary and extra bits are zero for sensitive surfaces.

---

## Post-Expansion Parquet Schema

The expansion pipeline (`expansion.py`) joins trackstates with all
measurements in the search window and emits one row per
(branch, surface, candidate). The schema has **57 columns**.

### Identity

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | int | Event index |
| `seed_id` | int | Seed (track) index |
| `branch_id` | int | Branch within the track's CKF tree |
| `parent_branch_id` | int | Parent branch (for tree reconstruction) |
| `step_k` | int | Surface step index within the branch |
| `layer_id` | int | Encoded layer |
| `surface_id` | int64 | Packed geometry_id of the surface |

### Predicted state (at this surface, before update)

| Column | Type | Description |
|--------|------|-------------|
| `state_l0` | float | Predicted local0 (mm) = `eLOC0_prt` |
| `state_l1` | float | Predicted local1 (mm) = `eLOC1_prt` |
| `state_phi` | float | Predicted φ (rad) |
| `state_theta` | float | Predicted θ (rad) |
| `state_qop` | float | Predicted q/p (1/GeV) |
| `state_t` | float | Predicted time (ns) |

### Predicted covariance (21 lower-triangular values)

| Columns | Type | Description |
|---------|------|-------------|
| `cov_00` | float | C[0,0] = err_eLOC0_prt² |
| `cov_01` | float | C[1,0] — only S01 available, rest NaN |
| `cov_02`–`cov_05` | float | C[2..5, 0] — NaN (off-diagonal not in ROOT) |
| `cov_06` | float | C[1,1] = err_eLOC1_prt² |
| `cov_07`–`cov_10` | float | NaN |
| `cov_11` | float | C[2,2] = err_ePHI_prt² |
| `cov_12`–`cov_14` | float | NaN |
| `cov_15` | float | C[3,3] = err_eTHETA_prt² |
| `cov_16`–`cov_17` | float | NaN |
| `cov_18` | float | C[4,4] = err_eQOP_prt² |
| `cov_19` | float | NaN |
| `cov_20` | float | C[5,5] = err_eT_prt² |

Only the 6 diagonal elements are populated from ROOT. The off-diagonal
cross-covariances (e.g. loc0-phi, loc1-theta) are not in the ROOT tree.

### Candidate measurement

| Column | Type | Description |
|--------|------|-------------|
| `pred_l0` | float | Predicted local0 used for window (= `state_l0`) |
| `pred_l1` | float | Predicted local1 used for window (= `state_l1`) |
| `cand_hit_id` | int | measurement_id of this candidate. -1 for holes. |
| `residual_l0` | float | cand.local0 − pred_l0 (mm) |
| `residual_l1` | float | cand.local1 − pred_l1 (mm) |
| `chi2_inc` | float | r'S⁻¹r using full S (including S01) |

### Cluster features

| Column | Type | Description |
|--------|------|-------------|
| `clus_s_u` | int | Cluster size in u (channels) |
| `clus_s_v` | int | Cluster size in v (channels) |
| `clus_q_tot` | float | Total cluster charge |
| `clus_sigma_uu` | float | Charge-weighted σ²_uu (channel² or mm²) |
| `clus_sigma_uv` | float | Cross moment σ²_uv |
| `clus_sigma_vv` | float | σ²_vv |

For the CKF-selected hit: from ROOT branches. For other in-window candidates:
computed from cells.csv. NaN for hole rows.

### Incidence angles

| Column | Type | Description |
|--------|------|-------------|
| `alpha_u` | float | atan2(dU, |dN|) (rad) — incidence in u plane |
| `alpha_v` | float | atan2(dV, |dN|) (rad) — incidence in v plane |

Same for all candidates at a given surface (depends on track direction, not hit).

### Sensor properties

| Column | Type | Description |
|--------|------|-------------|
| `pitch_u` | float | Pixel/strip pitch in u (mm). From detectors.csv or digi config. |
| `pitch_v` | float | Pitch in v (mm) |
| `thickness` | float | Sensor thickness (mm) |
| `is_pixel` | bool | True if volume_id ∈ {16, 17, 18} |
| `is_barrel` | float | NaN (requires full ACTS geometry) |

### Occupancy and context

| Column | Type | Description |
|--------|------|-------------|
| `n_window` | int | Number of measurements inside the n-σ window |
| `geometric_density` | int | Measurements within fixed r_geom_mm radius |
| `pathInX0_interval` | float | Material traversed since last surface (X/X₀) |
| `dead_module_flag` | int | Always 0 (no data source yet) |
| `layer_embed_idx` | int | Ordinal layer index for embedding |

### Branch history

| Column | Type | Description |
|--------|------|-------------|
| `n_hits` | int | Accumulated hits so far on this branch |
| `n_holes` | int | Accumulated holes |
| `n_seq_holes` | int | Consecutive holes (resets on hit) |

### Action and pruning

| Column | Type | Description |
|--------|------|-------------|
| `action_taken` | int | 0 = candidate, 1 = hole (window failure), 2 = hole (no measurements) |
| `prune_reason` | str/null | Why the CKF pruned this branch (null if not pruned) |

### Truth labels

| Column | Type | Description |
|--------|------|-------------|
| `contrib_pids` | list[int64] | Particle IDs contributing to this candidate's cluster |
| `contrib_charge_frac` | list[float] | Charge fraction per contributor |
| `branch_majority_pid` | int64 | Majority particle of this branch (≥2/3 of seed hits) |
| `majority_undefined` | bool | True if no particle has ≥2/3 majority |
| `majority_true_hit_on_surface` | bool | Did the majority particle leave a simhit on this surface? |
| `truth_residual_l0` | float | NaN (needs global→local transform via ACTS geometry) |
| `truth_residual_l1` | float | NaN |

### Training targets and config

| Column | Type | Description |
|--------|------|-------------|
| `vstar_soft` | float | V^{π†} soft value target (computed downstream, NaN initially) |
| `env_config_hash` | str | SHA256 of the collection-envelope config |

---

## Expansion Pipeline Data Flow

```
trackstates_ckf.root         measurements.csv
  (per-track Kalman states)    (all cluster centroids)
         │                          │
         │   encode_geometry_id()   │   decode_geometry_id()
         │   vol<<56|lay<<36|mod<<8 │
         ▼                          ▼
    ┌──────────────────────────────────┐
    │  expand_trackstates()            │
    │  JOIN on geometry_id             │
    │  FILTER: |residual| ≤ n·√S      │
    │  EMIT: one row per candidate     │
    │  EMIT: hole row if zero matches  │
    └──────────────┬───────────────────┘
                   │
         cells.csv │   detectors.csv / digi config
              │    │         │
              ▼    ▼         ▼
    ┌──────────────────────────────────┐
    │  cluster features (cells→moments)│
    │  sensor properties (pitch, etc.) │
    │  chi2_inc (using full S matrix)  │
    └──────────────┬───────────────────┘
                   │
  simhits.csv      │   measurement-simhit-map.csv
       │           │         │
       ▼           ▼         ▼
    ┌──────────────────────────────────┐
    │  compute_truth_labels()          │
    │  branch majority PID             │
    │  per-candidate contributing PIDs │
    │  majority_true_hit_on_surface    │
    └──────────────┬───────────────────┘
                   │
                   ▼
         expanded_event{N}.parquet
         (57 columns, ~20-50M rows per event)
```
