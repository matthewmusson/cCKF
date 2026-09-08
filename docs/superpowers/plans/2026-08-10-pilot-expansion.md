# Pilot Expansion Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run envelope CKF on events 0–1 with all outputs, build the offline expansion pipeline that expands each trackstate to all candidates in the n=10 window, produce Parquet files per the §6.5 schema, and run pilot checks 1–5.

**Architecture:** Stage 1 runs ACTS CKF via our existing Modal pipeline with full CSV+ROOT outputs enabled. Stage 2 is a new Modal function that reads trackstates + measurement CSVs + simhit CSVs, computes windows from the innovation covariance (S₀₀, S₁₁), joins all measurements on each surface within the window, computes per-candidate features (residuals, chi2_inc, cluster features from cells), and writes Parquet. Pilot checks are standalone analysis on the outputs.

**Tech Stack:** Python 3.10+, Modal (remote compute), uproot/awkward (ROOT I/O), pandas/pyarrow (Parquet), numpy

## Global Constraints

- Events `[0, 32)` only. Events `[32, 64)` are SEALED — never touch.
- Pilot runs on events 0 and 1 only.
- Geometric digi required (not smearing) — cluster features need cells.
- Envelope config: `configs/envelope.yaml` — terminal cuts disabled, branch cap 5, n=10.
- No label/loss/training code changes. This is Tier 2/3 infrastructure.
- All fields marked **[IRREVERSIBLE]** in §6.5 must be logged — omission means re-running.

---

## File Structure

- **Modify:** `cCKF/modal_build_acts.py` — add `run_pilot_stage1()` and `run_pilot_expansion()` Modal functions
- **Modify:** `cCKF/digi_and_reco.py` — add simhit CSV output flag passthrough, ensure cluster data is available
- **Create:** `cCKF/expansion.py` — the offline expansion logic (window computation, measurement joining, feature extraction, Parquet writing)
- **Create:** `cCKF/pilot_checks.py` — pilot check 1–5 analysis functions (can be Modal functions or called locally on downloaded Parquet)

---

### Task 1: Stage 1 — Run envelope CKF with full outputs on events 0–1

Run the envelope config through ACTS on Modal, producing:
- `trackstates_ckf.root` (pre-ambi CKF candidates with S₀₀, S₀₁, S₁₁, cluster features, incidence angles, pathInX0)
- Measurement CSVs: `measurements.csv` (local0, local1, geometry_id per measurement)
- Cell CSVs: `cells.csv` (channel0, channel1, value per cell, linked by measurement_id) — needed to compute cluster features for non-selected hits
- SimHit CSVs: `simhits.csv` (particle_id, geometry_id, true position)
- Truth hit mapping: written by `addDigitization` as `measurement_particles_map` (in-memory) — need to also write `measurement-simhit-map.csv`
- Seed CSVs: seed-to-particle truth matching

**Files:**
- Modify: `cCKF/modal_build_acts.py` — add `run_pilot_stage1()` function

**Interfaces:**
- Produces: output directory on Modal volume at `/data/results/pilot_<timestamp>/` containing all ROOT and CSV files listed above

- [ ] **Step 1: Write the `run_pilot_stage1()` Modal function**

This function runs the envelope config with all outputs enabled. Key config flags:
```python
cfg = {
    "events": 2, "skip": 0, "threads": 8,
    "ckf": True, "ambi": False,
    "output_digi_csv": True,       # measurements.csv + cells.csv
    "output_simhits_csv": True,    # simhits.csv
    "output_seeds_csv": True,      # seeds.csv
    "output_measurements_root": True,  # measurements.root (cluster cell data)
    "write_track_states": True,    # trackstates_ckf.root with full predicted cov
    "write_predicted_cov": False,
    "ckf_finding_performance": False,
    "ambi_finding_performance": False,
    "write_track_summary": True,   # tracksummary_ckf.root
}
```

The function should:
1. Load envelope.yaml and override with the above
2. Run `setup_acts_reconstruction` + `s.run()`
3. List all output files with sizes
4. Verify `trackstates_ckf.root` exists and has the 12 new branches (S00, S01, S11, cluster features, etc.)
5. Verify `measurements.csv` and `cells.csv` exist
6. Print track counts per event
7. Persist output directory to Modal volume

- [ ] **Step 2: Run on Modal and verify outputs**

```bash
modal run modal_build_acts.py::run_pilot_stage1
```

Expected outputs:
- `trackstates_ckf.root` (~14 MB per event × 2 = ~28 MB)
- `event000000000-measurements.csv`, `event000000001-measurements.csv`
- `event000000000-cells.csv`, `event000000001-cells.csv`
- `event000000000-simhits.csv`, `event000000001-simhits.csv`

- [ ] **Step 3: Commit**

```bash
git add cCKF/modal_build_acts.py
git commit -m "feat: add pilot stage 1 — envelope CKF with full CSV+ROOT outputs"
```

---

### Task 2: Build the offline expansion pipeline

The core of the pilot. For each trackstate in `trackstates_ckf.root`:
1. Compute window bounds: Δl₀ = n·√S₀₀, Δl₁ = n·√S₁₁ (bounding box of the n-sigma ellipse)
2. Find all measurements on the same (volume, layer, module) with |local0 - pred_local0| ≤ Δl₀ AND |local1 - pred_local1| ≤ Δl₁
3. Compute per-candidate features: residual, chi2_inc (using full S including S₀₁), cluster features from cells
4. Emit one Parquet row per (branch, surface, candidate) + hole rows

**Files:**
- Create: `cCKF/expansion.py` — standalone expansion logic
- Modify: `cCKF/modal_build_acts.py` — add `run_pilot_expansion()` Modal function that calls expansion.py

**Interfaces:**
- Consumes: Stage 1 output directory (trackstates ROOT + CSV files)
- Produces: `expanded_event{N}.parquet` files in the same output directory

- [ ] **Step 1: Write `expansion.py` — data loading**

Load the three data sources into pandas:
```python
def load_trackstates(root_path: str, event_id: int) -> pd.DataFrame:
    """Load trackstates for one event. Returns flat DataFrame with one row per
    (track_nr, state_idx), including predicted local coords, innovation cov,
    geometry IDs, particle IDs, cluster features, incidence angles, pathInX0,
    branch history (n_hits, n_holes)."""

def load_measurements(csv_dir: str, event_id: int) -> pd.DataFrame:
    """Load measurements.csv for one event. Returns DataFrame with columns:
    measurement_id, geometry_id, local0, local1, var_local0, var_local1."""

def load_cells(csv_dir: str, event_id: int) -> pd.DataFrame:
    """Load cells.csv for one event. Returns DataFrame with columns:
    geometry_id, measurement_id, channel0, channel1, value."""

def load_simhits(csv_dir: str, event_id: int) -> pd.DataFrame:
    """Load simhits.csv for one event. Returns DataFrame with columns:
    geometry_id, particle_id (encoded int64), tx, ty, tz."""
```

Key details:
- `geometry_id` in the CSV encodes volume/layer/module as a single uint64 via ACTS `GeometryIdentifier`. Extract volume/layer/module: `volume = (geoid >> 28) & 0xf`, `layer = (geoid >> 16) & 0xfff`, `module = geoid & 0xffff` — but verify exact bit layout from ACTS source.
- Trackstates use separate `volume_id, layer_id, module_id` columns. Build composite key to match.
- Measurements CSV indexes states by `geometry_id` (surface-level). All measurements on the same surface share the same `geometry_id`.

- [ ] **Step 2: Write `expansion.py` — window computation and candidate matching**

```python
def compute_window_bounds(S00: np.ndarray, S11: np.ndarray, n: float = 10.0):
    """Compute window half-widths: delta_l0 = n * sqrt(S00), delta_l1 = n * sqrt(S11)."""

def expand_trackstates(
    states: pd.DataFrame,
    measurements: pd.DataFrame,
    n: float = 10.0,
) -> pd.DataFrame:
    """For each trackstate, find all measurements on the same surface within
    the n-sigma window. Returns one row per (track_nr, state_idx, candidate).
    
    Matching: same geometry_id AND |meas.local0 - state.pred_local0| <= delta_l0
                                AND |meas.local1 - state.pred_local1| <= delta_l1
    
    Also emits hole rows: trackstates with NO candidates in the window get a
    single row with cand_hit_id = -1 and null candidate features.
    
    Returns DataFrame with columns from the §6.5 schema.
    """
```

The join strategy (for ~1M trackstates × ~200k measurements):
- Group measurements by geometry_id (surface). Build a dict: geometry_id → DataFrame of measurements.
- For each trackstate, look up its geometry_id in the dict, then filter by window bounds.
- This is O(states × avg_measurements_per_surface), which is tractable since most surfaces have <100 measurements.

- [ ] **Step 3: Write `expansion.py` — per-candidate feature computation**

For each (trackstate, candidate) pair:
```python
def compute_candidate_features(state_row, cand_row, S00, S01, S11):
    """Compute features for one (state, candidate) pair.
    
    Returns dict with:
    - residual: [cand.local0 - state.pred_local0, cand.local1 - state.pred_local1]
    - chi2_inc: r^T S^{-1} r using FULL S (including S01)
      S = [[S00, S01], [S01, S11]]
      det = S00*S11 - S01^2
      chi2 = (S11*r0^2 - 2*S01*r0*r1 + S00*r1^2) / det
    - occupancy: n_window (count of candidates in this window)
    """
```

- [ ] **Step 4: Write `expansion.py` — cluster feature computation from cells**

For non-selected candidates, we need to compute cluster features from the cell data:
```python
def compute_cluster_features(cells_for_measurement: pd.DataFrame) -> dict:
    """Compute cluster size, charge, and moments from raw cell data.
    
    Returns: s_u, s_v, Q_tot, sigma_uu, sigma_uv, sigma_vv
    
    s_u = max(channel0) - min(channel0) + 1  (cluster size along u)
    s_v = max(channel1) - min(channel1) + 1  (cluster size along v)
    Q_tot = sum(value)
    Charge-weighted second central moments computed from channel positions
    weighted by activation values (same as clusterChargeMoments in C++).
    """
```

Note: cell positions are in bin/channel indices, not physical coordinates. The moments computed here will be in channel units. The conversion to physical units requires pitch, which we get from sensor_props. For consistency with the RootTrackStatesWriter (which uses `path2D` midpoints in physical coords), we should compute moments the same way. However, the CSV cell writer only has `channel0, channel1, value` — no path2D. We have two options:
- Compute moments in channel space and multiply by pitch² later
- Use the ROOT measurement file which has the full cluster data

Decision: compute moments in channel space. The normalized features (κ_u, κ_v, Q̃) from §8.2 normalize by expected values that also scale with geometry, so the convention just needs to be consistent. Document this.

- [ ] **Step 5: Write `expansion.py` — truth label computation**

```python
def compute_truth_labels(
    expanded: pd.DataFrame,
    simhits: pd.DataFrame, 
    measurement_particle_map: pd.DataFrame,
    branch_majority_pid: pd.Series,
) -> pd.DataFrame:
    """Add truth columns to the expanded DataFrame.
    
    - contrib_pids: list of particle IDs contributing to this cluster
    - contrib_charge_frac: charge fraction per contributor
    - majority_true_hit_on_surface: did the branch majority particle m leave
      a measurement on this surface? (requires simhit lookup)
    - truth_residual: h - h^true_m from SimTrackerHit
    """
```

The measurement-to-particle mapping comes from the `measurement_particles_map` produced by digitization. This is written to CSV only if we add a writer for it — check if ACTS has a `CsvMeasurementParticleMapWriter` or if we need to extract it from the digi output differently.

- [ ] **Step 6: Write `expansion.py` — Parquet writer**

```python
def write_expanded_parquet(
    expanded: pd.DataFrame,
    output_path: str,
    env_config_hash: str,
):
    """Write the expanded DataFrame to Parquet with the §6.5 schema.
    
    Columns match the spec schema:
    event_id, seed_id, branch_id, step_k, layer_id, surface_id,
    state (6 floats), cov_packed (21 floats), pred_local (2 floats),
    cand_hit_id, residual (2 floats), chi2_inc,
    cluster_feats (6 floats), incidence (2 floats),
    sensor_props (5 floats), occupancy_feats (n_window),
    context_feats (pathInX0, dead_module_flag),
    branch_feats (n_hits, n_holes, n_seq_holes),
    action_taken, prune_reason,
    contrib_pids (list), contrib_charge_frac (list),
    branch_majority_pid, majority_true_hit_on_surface,
    truth_residual (2 floats), env_config_hash
    """
```

- [ ] **Step 7: Commit**

```bash
git add cCKF/expansion.py
git commit -m "feat: add offline expansion pipeline for pilot data collection"
```

---

### Task 3: Wire expansion into Modal and run

**Files:**
- Modify: `cCKF/modal_build_acts.py` — add `run_pilot_expansion()` function

- [ ] **Step 1: Write `run_pilot_expansion()` Modal function**

```python
@app.function(image=image, volumes={...}, cpu=8, memory=65536, timeout=3600)
def run_pilot_expansion():
    """Stage 2: expand trackstates to all candidates in W_k(10)."""
    # 1. Find the pilot stage 1 output directory
    # 2. For each event (0, 1):
    #    a. Load trackstates, measurements, cells, simhits
    #    b. Run expansion
    #    c. Write Parquet
    #    d. Print summary stats (rows, positive fraction, file size)
```

- [ ] **Step 2: Run stage 1 on Modal**

```bash
modal run modal_build_acts.py::run_pilot_stage1
```

- [ ] **Step 3: Run expansion on Modal**

```bash
modal run modal_build_acts.py::run_pilot_expansion
```

- [ ] **Step 4: Commit**

```bash
git add cCKF/modal_build_acts.py
git commit -m "feat: wire expansion pipeline into Modal and run pilot"
```

---

### Task 4: Pilot checks 1–5

**Files:**
- Create: `cCKF/pilot_checks.py` — analysis functions
- Modify: `cCKF/modal_build_acts.py` — add `run_pilot_checks()` Modal function

- [ ] **Step 1: Check 1 — Cluster features populated**

Verify every measurement row in the digi CSV has non-null, non-degenerate s_u, s_v, Q_tot, σ_uu, σ_uv, σ_vv. Since cluster features aren't in the measurement CSV directly, verify them in the expanded Parquet (where we compute them from cells) AND verify they're present in `trackstates_ckf.root` for the selected hits.

Pass criterion: 100% of measurement rows have all six fields populated and finite.

- [ ] **Step 2: Check 2 — Seed recovery**

Already substantially done from prior work (100% middle-SP, 98.1% DM particle recovery). Re-verify with the new pilot run's data.

- [ ] **Step 3: Check 3 — Scale**

Report from the expansion output:
- Total tracks per event (before expansion)
- Total rows per event (after expansion) 
- Positive fraction (rows where majority particle contributed to the cluster)
- Parquet file size per event
- Wall time for Stage 1 and Stage 2 separately
- Extrapolation to 32 events

- [ ] **Step 4: Check 4 — Window-failure rate**

For each (branch, surface) where `majority_true_hit_on_surface = 1`, check whether the majority particle's measurement appears among the expanded candidates.

Report stratified by: η region, layer, accumulated X/X₀ quartiles, layer × η cross-tabulation.

Pass criterion: overall failure rate < 1% at n=10.

- [ ] **Step 5: Check 5 — Env hash stability**

Compute env_config_hash from the envelope config and verify it's identical across both events.

- [ ] **Step 6: Write summary and commit**

```bash
git add cCKF/pilot_checks.py cCKF/modal_build_acts.py
git commit -m "feat: pilot checks 1-5 for expansion pipeline"
```

---

## Key risks and mitigations

1. **Cluster features from cells:** The CSV cell data has channel indices, not physical coordinates. Moments in channel space ≠ moments in physical space. For the pilot we compute in channel space and document the convention. For production, we may need to use the ROOT measurement file which has path2D.

2. **geometry_id bit layout:** The CSV uses a packed uint64 for geometry_id. Need to verify the exact bit layout matches the ACTS GeometryIdentifier convention to correctly join measurements to trackstates.

3. **Memory for expansion:** With ~1M trackstates and ~200k measurements per event, the join could be memory-intensive. Process per-surface to keep memory bounded.

4. **measurement_particles_map:** The measurement-to-particle truth mapping is produced by digitization as an in-memory collection. Need to verify it's written to CSV by addDigitization's CSV writer, or add a writer for it.

5. **Occupancy computation:** n_window is simply the count of candidates within W_k(n) for each trackstate. This falls out naturally from the expansion join.
