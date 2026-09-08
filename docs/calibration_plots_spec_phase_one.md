# Spec: CKF Calibration Reliability Diagrams (Phase 1)

**Owner:** Matthew Musson
**Status:** draft
**Date:** 2026-08-03
**Depends on:** CKF benchmark pipeline (complete), RootTrackStatesWriter output

---

## 1. Goal

Produce stratified reliability diagrams that compare the χ²-implied probability Λ_χ² against empirical true-hit frequency for hits accepted by the CKF. This is the "before" picture (figure G3 in the master spec) that motivates the learned gate g_ψ.

**What the plot answers:** "Among CKF-accepted hits where the Gaussian model predicted probability Λ, what fraction were actually from the correct particle?"

**What it shows:** The χ² cut is a broken Neyman-Pearson test because it ignores the prior (1/n_window) and assumes uniform background. The reliability diagram quantifies *where* and *how badly* this breaks, stratified by occupancy and η.

---

## 2. Scope

**Phase 1 (this spec):** Post-processing of existing CKF output only. No ACTS instrumentation. Accepted hits only. Occupancy is an approximate proxy computed from accepted hits on the same module.

**Phase 2 (gate training spec, later):** Instrument MeasurementSelector to log all candidates (accepted and rejected), full feature vector including cluster features, exact occupancy from full hit set. The Phase 1 plots will be regenerated with exact data.

**Known limitations of Phase 1:**
- No rejected candidates (hits considered but not accepted). The reliability diagram is truncated below Λ ≈ exp(−χ²_max/2).
- Occupancy proxy underestimates true occupancy, especially in jet cores where many hits go unmatched. Stratification boundaries may shift in Phase 2.
- Missing off-diagonal of S_k (only diagonal written by RootTrackStatesWriter). Does not affect Λ computation (uses scalar χ²) but prevents full Mahalanobis window reconstruction.

---

## 3. Data Collection

### 3.1 Events

Use training split only: events [0, 32). Do NOT use events [32, 64) — those are held out for final evaluation (master spec §6.1).

### 3.2 CKF Configuration

Run with the **Medium operating point** (Optuna trial 70):

| Parameter | Value |
|---|---|
| maxSeedsPerSpM | 46 |
| seed minPt | 0.587 GeV |
| impactMax | 2.86 mm |
| sigmaScattering | 2.95 |
| chi2CutOffMeasurement | 12.04 |
| chi2CutOffOutlier | 35.75 |
| numMeasurementsCutOff | 2 |
| nMeasurementMin | 7 |
| maxHolesAndOutliers | 1 |
| CKF ptMin | 0.594 GeV |

Medium is chosen because it has the highest efficiency (ε_DM = 96.23%) and therefore produces the most accepted track states for analysis. With χ²_meas = 12.04, the accepted-only Λ range spans [exp(−12.04/2), 1] ≈ [0.0025, 1.0].

**Optional follow-up:** Run Tight and Fast operating points on the same events to show that χ² miscalibration is structural (independent of tuning), not an artifact of one configuration.

### 3.3 Required Output

Run the CKF with `writeTrackStates=True` to produce `trackstates_ckf.root`. This is the sole input for all downstream processing.

Verify the output contains per-track-state fields listed in §4.

---

## 4. Feature Extraction

Read `trackstates_ckf.root` with uproot. Filter to `stateType == 0` (measurement states only; exclude holes, outliers, material states).

### 4.1 Fields to Extract

| Field | Source column | Notes |
|---|---|---|
| event_id | `event_nr` | |
| track_id | `track_nr` | |
| chi2 | `chi2` | Per-hit χ² increment, scalar |
| meas_loc0 | `l_x_hit` | Measured local position |
| meas_loc1 | `l_y_hit` | Measured local position |
| pred_loc0 | `eLOC0_prt` | Predicted local position (= predicted measurement) |
| pred_loc1 | `eLOC1_prt` | Predicted local position |
| err_pred_loc0 | `err_eLOC0_prt` | √(C_predicted[0,0]) — predicted state uncertainty |
| err_pred_loc1 | `err_eLOC1_prt` | √(C_predicted[1,1]) |
| err_meas_loc0 | `err_x_hit` | √(R_k[0,0]) — measurement uncertainty |
| err_meas_loc1 | `err_y_hit` | √(R_k[1,1]) |
| eta | `eta_prt` | Pseudorapidity from predicted state |
| pT | `pT_prt` | Transverse momentum from predicted state |
| volume_id | `volume_id` | Detector geometry identifiers |
| layer_id | `layer_id` | |
| module_id | `module_id` | |
| truth_particle_id | `particle_ids_*` | Truth particle barcode(s) for this hit |
| res_loc0 | `res_x_hit` | Residual (meas − predicted) |
| res_loc1 | `res_y_hit` | |

### 4.2 Verification Checks on Extracted Data

Run these before proceeding. Fail loudly if any check fails.

1. `chi2` values are non-negative.
2. `chi2` ≤ `chi2CutOffMeasurement` (12.04) for all measurement states. Any above this indicates a mismatch between the config used and the output read.
3. `err_eLOC0_prt` and `err_eLOC1_prt` are positive for all rows.
4. Residuals are consistent: `res_x_hit ≈ l_x_hit − eLOC0_prt` within floating-point tolerance.
5. Row count is in the expected range: ~10^5 per event, ~3×10^6 total for 32 events.

---

## 5. Derived Quantities

### 5.1 Λ_χ² (χ²-implied probability)

```python
lambda_chi2 = np.exp(-chi2 / 2)
```

This is the survival function of the χ²₂ distribution: P(χ²₂ ≥ observed) = exp(−χ²/2). It maps χ² = 0 → Λ = 1, χ² → ∞ → Λ → 0. One line. No other computation needed.

### 5.2 Truth Label

A hit is "true" if the truth particle associated with the hit matches the branch's majority particle.

**Branch majority particle:** For each track, determine which truth particle contributes the most hits. This is the majority particle m for that track. Then for each hit on the track:

```python
label_same_particle = (hit_truth_particle_id == track_majority_particle_id)
```

**Tracks where no particle contributes ≥ 50% of hits are fake tracks.** All hits on fake tracks receive `label_same_particle = False`. Do not exclude fake tracks — their hits are real negatives that the CKF accepted incorrectly.

### 5.3 Occupancy Proxy

**Definition:** For a given TrackState at position (pred_loc0, pred_loc1) on module (volume_id, layer_id, module_id), count the number of accepted hits from ALL tracks on the same module that fall within the CKF's bounding box.

**Bounding box computation:**

```python
# Approximate S_k diagonal from available fields
S_00 = err_pred_loc0**2 + err_meas_loc0**2   # C_predicted[0,0] + R_k[0,0]
S_11 = err_pred_loc1**2 + err_meas_loc1**2

# Bounding box half-widths at the operating point's chi2 cut
chi2_max = 12.04  # Medium operating point
half_width_0 = np.sqrt(S_00 * chi2_max)
half_width_1 = np.sqrt(S_11 * chi2_max)

# Count hits on same module within bounding box
n_window = count of hits h on same (volume_id, layer_id, module_id) where:
    |h.meas_loc0 - pred_loc0| <= half_width_0 AND
    |h.meas_loc1 - pred_loc1| <= half_width_1
```

This counts accepted hits from all tracks, not just the current track. The current hit is included in its own count, so n_window ≥ 1 always.

**Known limitation:** This underestimates true occupancy because hits not accepted by any track are invisible. Phase 2 will recompute from the full hit set.

**Performance note:** This is an O(N²) computation per module per event. For efficiency, group hits by (event_id, volume_id, layer_id, module_id) first, then compute pairwise distances within each group. Most modules have few accepted hits, so this should be fast.

---

## 6. Reliability Diagram Specification

### 6.1 Binning

**Use equal-mass (quantile) bins, not equal-width.** Λ_χ² values pile up near 1.0 (most accepted hits have low χ²). Equal-width bins would leave high-Λ bins massively overpopulated and low-Λ bins nearly empty.

**Number of bins:** 15 for the overall plot. With ~3M total candidates, each bin contains ~200k candidates, giving binomial standard errors of ~0.001. For stratified plots (12 panels), reduce to 10 bins per panel if any stratum has fewer than 50k candidates.

**Per bin, compute:**
- `mean_predicted`: mean Λ_χ² in the bin (x-axis)
- `observed_fraction`: fraction of hits with `label_same_particle == True` (y-axis)
- `bin_count`: number of candidates in the bin (report alongside ECE)
- `ci_lower`, `ci_upper`: 95% Wilson binomial confidence interval

### 6.2 ECE Computation

Expected Calibration Error, per master spec §10.4:

```
ECE = Σ_b (|B_b| / N) × |observed_fraction_b − mean_predicted_b|
```

Report overall ECE and worst-stratum ECE.

### 6.3 Stratification

#### Occupancy Strata

Compute quartiles of the n_window distribution across all candidates. Report the actual quartile boundaries (we do not know them a priori).

| Stratum | Definition |
|---|---|
| Q1 (sparse) | n_window ≤ 25th percentile |
| Q2 | 25th–50th percentile |
| Q3 | 50th–75th percentile |
| Q4 (dense) | n_window > 75th percentile |

**Expectation:** Q4 (dense regions, jet cores) should show the largest deviation from the diagonal because the uniform background assumption fails most severely there.

#### η Strata

Three physically motivated regions of the ODD detector:

| Stratum | Range | Region |
|---|---|---|
| Central | \|η\| < 1.0 | Barrel |
| Transition | 1.0 ≤ \|η\| < 2.0 | Barrel-endcap transition |
| Forward | \|η\| ≥ 2.0 | Endcap |

#### Combined Stratification

4 occupancy × 3 η = 12 panels. If any panel has fewer than 10k candidates, merge with an adjacent panel and note the merge.

---

## 7. Output Files

### 7.1 Data

```
output/
  calibration_data.parquet     # Per-candidate: chi2, lambda, label, eta, n_window, event_id, track_id
  calibration_summary.json     # ECE values, bin boundaries, stratum boundaries, row counts
```

The Parquet file is the reusable artifact. All plots are regenerated from it. Schema:

| Column | Type | Description |
|---|---|---|
| event_id | int32 | |
| track_id | int32 | |
| chi2 | float32 | Per-hit χ² increment |
| lambda_chi2 | float32 | exp(−χ²/2) |
| label_same_particle | bool | Truth label |
| eta | float32 | Predicted η |
| pT | float32 | Predicted pT (GeV) |
| n_window | int32 | Occupancy proxy |
| volume_id | int32 | |
| layer_id | int32 | |
| module_id | int32 | |

### 7.2 Plots

All plots saved as PDF (vector, poster-quality) and PNG (quick inspection).

1. **Overall reliability diagram.** Λ_χ² vs observed fraction, 15 quantile bins, diagonal reference line, Wilson CIs, ECE in legend.

2. **Occupancy-stratified reliability diagram.** Four lines (Q1–Q4) on the same axes, color-coded, with ECE per stratum in legend. This is the candidate headline figure.

3. **η-stratified reliability diagram.** Three lines (central, transition, forward) on same axes.

4. **Combined panel plot.** 4×3 grid, one panel per (occupancy, η) stratum. Reduced to 10 bins per panel. Report bin counts per panel.

5. **Diagnostic: Λ_χ² histogram.** Distribution of predicted probabilities, log-scale y-axis. Confirms the pile-up near 1.0 and justifies quantile binning.

6. **Diagnostic: occupancy distribution.** Histogram of n_window, log-scale, with quartile boundaries marked. Informs Phase 2 stratification decisions.

### 7.3 Plot Style

- Matplotlib, no Seaborn
- Font size: 12pt body, 14pt titles
- Color palette: qualitatively distinct (e.g., tab10), colorblind-safe
- Diagonal reference line: black, dashed
- Confidence intervals: shaded bands, alpha=0.2
- ECE value in legend: "ECE = 0.042"
- Axis labels: "Predicted probability (Λ_χ²)" and "Observed fraction (true hits)"

---

## 8. Implementation Tasks

These are ordered. Each depends on the previous.

### Task 1: Run CKF on training events

Run the existing CKF pipeline with Medium operating point config on events [0, 32) with `writeTrackStates=True`. Output: `trackstates_ckf.root`.

**Test:** File exists, contains expected branches, row count ~10^5–10^6 per event.

### Task 2: Extract and validate features

Read ROOT file with uproot, filter to measurement states, extract fields per §4.1. Run all verification checks in §4.2. Save raw extraction as intermediate Parquet.

**Test:** All §4.2 checks pass. Row count reported.

### Task 3: Compute derived quantities

Compute Λ_χ², truth labels, and occupancy proxy per §5. Save as `calibration_data.parquet` per §7.1 schema.

**Test:**
- Λ_χ² values in [exp(−12.04/2), 1.0] ≈ [0.0025, 1.0]
- label_same_particle True fraction is between 0.5 and 0.99 (sanity: most accepted hits should be correct)
- n_window ≥ 1 for all rows
- No NaN or inf values in any column

### Task 4: Produce reliability diagrams

Generate all plots per §7.2 and compute ECE per §6.2. Save summary statistics to `calibration_summary.json`.

**Test:**
- All 6 plot files exist in both PDF and PNG
- ECE values are in [0, 1]
- Every bin in every plot has ≥ 100 candidates (if not, bins are too fine)
- Overall reliability diagram visually inspected: points should not all lie on the diagonal (if they do, either the diagnostic is uninformative or there is a labeling bug)

---

## 9. Extensibility

This spec is designed so Phase 2 replaces only the data source. The Parquet schema (§7.1) is a strict subset of the full decision log schema (master spec §6.5). When Phase 2 produces the full log:

- Add rejected candidates (extends Λ range below 0.0025)
- Replace occupancy proxy with exact n_window from MeasurementSelector
- Add cluster features, sensor properties, incidence angles (new columns, same rows)
- Regenerate all plots from the same plotting code
- Compare Phase 1 vs Phase 2 occupancy-stratified diagrams to assess proxy quality

No plotting code changes needed. Only the data generation pipeline changes.