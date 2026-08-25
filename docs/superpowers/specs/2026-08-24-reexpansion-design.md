# Re-expansion of the 32-event training set — design

**Date:** 2026-08-24
**Status:** Approved. Implementation starts with event 1.

## Problem

The offline gate audit (`experiments/LOG.md`, 2026-08-24) found three defects in
the expanded training Parquets:

1. **Long strips missing from 28 of 32 events.** `expand_trackstates` filters
   `states["S11"].notna()`. Volumes 28/29/30 are genuinely 1D, so
   `instrumentation.patch` writes `S11_prt = NaN` for them by design, and the
   filter drops every one. Long strips are 1.27% of the training set instead of
   their true share, and all surviving rows live in events 0-3.

   Consequence: the deployed gate accepts **0.000%** of long-strip candidates,
   on the surfaces with the *highest* base positive rate in the detector
   (18.9%, 37.0%, 18.3%). Long strips are the outermost layers, so every track
   loses its outer hits.

2. **Corrupted `S11` on 2D sensors in events 0-3.** 34-35% of trainable rows in
   those events have `S11`/`residual_l1` absent on volumes that digitise 2D,
   with `chi2_inc` recomputed as the 1D form. Gate features 1, 3, 4 and 5 are
   wrong on those rows. Gate purity on volume 17 is 89.5% on clean val events
   and 10.6% on events 0-3.

3. **Shared per-state `S`.** `expand_trackstates` reuses one innovation
   covariance -- the CKF-selected hit's -- for every candidate on a surface. It
   is a documented approximation, but the C++ computes `S` per candidate, so
   training and inference disagree.

## Approach

**Compute `S` per candidate from first principles instead of reading it.**

```
S00 = var_local0 + P00
S01 =              P01
S11 = var_local1 + P11
```

- `P00, P01, P11` -- the predicted covariance projected into local coordinates
  (the `H C H^T` term), from `event*-predicted-cov.csv`, keyed
  `(track_nr, step_k)`.
- `var_local0, var_local1` -- the per-measurement variance (`V`), from
  `event*-measurements.csv`, keyed `measurement_id`.

This one change addresses all three defects:

| defect | how this fixes it |
|--------|-------------------|
| corrupted `S11` | `S11_prt` is never read. The corrupted column drops out of the pipeline entirely. |
| shared per-state `S` | Each candidate contributes its own `V`, so `S` is per-candidate by construction. |
| 1D detection | A long strip has no `var_local1`. This is exact, and unlike `S11.isna()` it does **not** conflate genuine-1D with corrupted-2D. |

That last point is the reason to prefer this over patching in place: today
`S11 = NaN` means either "1D sensor" or "corruption", and nothing in the
Parquet distinguishes them. Deriving from `var_local1` removes the ambiguity at
the source.

## Inputs (verified present)

All in the `pilot_*` directories mirrored to `$SCRATCH/cckf/modal_backup/results`:

| file | size | provides |
|------|------|----------|
| `trackstates_ckf.root` | 11.5 GB | track states, `pathInX0`, cluster features |
| `event*-predicted-cov.csv` | ~2.5 GB | `P00, P01, P11` per `(track_nr, step_k)` |
| `event*-measurements.csv` | ~24 MB | `local0/1`, `var_local0/1`, `geometry_id` |
| `event*-cells.csv` | ~49 MB | cluster shape and charge |
| `event*-simhits.csv`, `event*-measurement-simhit-map.csv` | ~36 MB | truth |
| `event*-seed.csv` | ~28 MB | seeds |
| `detectors.csv` | 4 MB | geometry |

30/30 pilot directories carry `trackstates_ckf.root`.

## Changes to `expansion.py`

1. **`load_predicted_cov(csv_dir, event_id)`** -- new loader returning
   `track_nr, step_k, P00, P01, P11`.

2. **`expand_trackstates`**
   - Drop `states["S11"].notna()` from the `valid` mask. Keep
     `states["S00"].notna()`; `S00` exists for any measurement.
   - Join `P` on `(track_nr, step_k)`.
   - Carry `var_local0`, `var_local1` through the candidate merge (they are
     already read but currently dropped from the output schema).
   - Compute per-candidate `S00/S01/S11` from the formula above.
   - `is_1d := var_local1` is null.

3. **Dimension-aware window.** The n=10 axis-aligned box, as verified against
   259.9M rows:
   ```
   |r0| <= n*sqrt(S00)  and  (is_1d or |r1| <= n*sqrt(S11))
   ```

4. **Dimension-aware `chi2_inc`.**
   - 1D: `r0^2 / S00`
   - 2D: full 2x2 Mahalanobis using the per-candidate `S`

5. **Schema.** Keep `var_local0`, `var_local1`; add `is_1d`. 76 -> 79 columns.

**`is_1d` is a Parquet column for analysis and stratification, NOT a gate
feature.** `GATE_FEATURES` stays at 26, so `CckfFeatures.hpp`, the pybind
`ACTS_PYTHON_STRUCT` field list, the weight blob format and `export_weights.py`
are all untouched. This was a deliberate scope decision: with four days to the
poster, the blast radius stays inside `expansion.py` and the Parquets.

The gate can still distinguish 1D rows implicitly -- `chol_S_11 == 0` exactly
never occurs on a 2D row -- and with a correct 1D `chi2` those rows carry real
signal instead of the constant degenerate vector they carry today. If the
retrained gate still rejects long strips, adding an explicit `is_1d` feature is
the next step, and we will then know it was necessary.

## Validation gates (event 1, before scaling)

All must pass:

1. **Long strips present** -- volumes 28/29/30 row count > 0. Currently zero for
   events 4-31.
2. **No corrupted 2D rows** -- zero rows with null `S11` on a sensor whose
   `var_local1` exists.
3. **Window holds** -- zero violations of the box on candidate rows, matching
   the current audit result.
4. **`chi2_inc` self-consistent** -- reproduces from stored `S` and residuals to
   machine precision, separately for 1D and 2D rows.
5. **Row count and positive rate** compared against the existing Parquet, with
   the delta explained rather than merely observed. Expect *more* rows (long
   strips restored) and a *higher* positive rate (long strips are 18-37%
   positive).

## Scale-out

Measure wall time and peak memory on event 1, then decide the batching for the
remaining 31. The CPU allocation is 200 node-hours total (194 remaining), so
the per-event cost has to be known before committing.

## Out of scope

- The sealed `[32,64)` evaluation events. Untouched.
- Retraining. Separate step once the Parquets exist.
- The occupancy-conditional Platt refit and the (tau_g, tau_v) sweep. Running
  in parallel against the *current* weights to establish a baseline.
- Any C++ or gate-feature change.

## Risks

| risk | mitigation |
|------|------------|
| `predicted-cov.csv` join key does not cover all states | Validation gate 5 catches a row-count collapse; report join coverage explicitly. |
| 2.5 GB CSV per event drives memory up | Measure on event 1; read only the needed columns. |
| Re-expansion cost exceeds the remaining allocation | Measure on event 1 before committing to 32. |
| Schema 76 -> 79 breaks positional readers | Grep for positional Parquet access before scaling. |
