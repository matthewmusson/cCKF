# 05 — The track-states file: every decision the CKF made

`trackstates_ckf.root` is the CKF's own log. For every track candidate it
records every surface the propagation touched, what the filter predicted
there, which measurement it took, and (from our instrumentation) the
quantities the gate needs. The expansion (doc 06) reads it; so does every
diagnostic. This document shows what is in it, using track 128909 from
doc 03, and explains the one trap that cost this project a week.

## Shape

One ROOT file, one tree called `trackstates`, **one entry per track**. Most
branches are jagged: an array with one element per state of that track.
Event 4 of the envelope run has 1,691,023 tracks and 41.5 million states
(the example file also holds event 5, so `event_nr` selects). There are 188
branches; grouped:

| Group | Branches | Per |
|---|---|---|
| identity | `event_nr`, `track_nr`, `nStates`, `nMeasurements`, `nPredicted`, `nFiltered`, `nSmoothed`, `nUnbiased` | track |
| where | `volume_id`, `layer_id`, `module_id`, `pathLength`, `stateType`, `chi2` | state |
| truth at the state | `t_x, t_y, t_z, t_r, t_dx, t_dy, t_dz`, `t_eLOC0 … t_eQOP`, `t_eT`; `particle_ids_{vertex_primary, vertex_secondary, particle, generation, sub_particle}` | state (the `particle_ids_*` are doubly jagged: one list of contributors per state) |
| the accepted measurement | `l_x_hit`, `l_y_hit`, `g_x_hit`, `g_y_hit`, `g_z_hit` | state (NaN where no measurement) |
| measurement-only | `dim_hit`, `res_x_hit`, `res_y_hit`, `err_x_hit`, `err_y_hit`, `pull_x_hit`, `pull_y_hit` | **measurement**, not state: shorter arrays |
| predicted parameters | `predicted`, `eLOC0_prt … eT_prt`, `err_*_prt`, `res_*_prt`, `pull_*_prt`, `g_{x,y,z}_prt`, `p{x,y,z}_prt`, `eta_prt`, `pT_prt` | state |
| filtered, smoothed, unbiased | the same set with `_flt`, `_smt`, `_ubs` | state |
| our instrumentation | `S00_prt`, `S01_prt`, `S11_prt`, `pathInX0_interval`, `clus_size_u`, `clus_size_v`, `clus_qtot`, `clus_sigma_uu`, `clus_sigma_uv`, `clus_sigma_vv`, `alpha_u`, `alpha_v` | state |

The three parameter sets are the Kalman filter's three answers at a surface:
*predicted* before using the hit there, *filtered* after using it, and
*smoothed* after the backward pass has folded in every later hit. The gate
sees the predicted state, because that is what exists when the decision is
made.

`stateType` classifies each state: 0 measurement, 1 outlier (a hit the CKF
kept but did not use in the update), 2 hole (a sensor crossed with nothing
accepted), 3 material (a non-sensor surface the navigator stopped on: layer
approach surfaces, volume boundaries, the beam pipe). In a 20,000-track
sample of event 4 the split is 87k / 19k / 109k / 224k, and every type-3
state has `module_id == 0`. Until 2026-09-14 the project believed this
branch was not written; the expansion does not read it, which is why its
hole counters treat material states as holes (doc 06).

The instrumented branches come from `instrumentation.patch`;
`docs/instrumentation/trackstate_branch_reference.md` lists their sentinels
and caveats. `S11_prt` was found corrupted on a third of states in events 0
to 3, so the expansion recomputes S from `predicted-cov.csv` instead (below).

## Track 128909 as stored

The same 23 states as the table in doc 03, now in the order the file holds
them. `root` is the array index; `stateType` as above; `contrib` is the
`particle_ids_particle` list at that state (7 is our particle's `particle`
field; the other four fields are 1, 0, 0, 6).

| root | vol/lay/mod | stateType | path (mm) | l_x_hit | eLOC0_prt | err_eLOC0_prt | chi2 | pathInX0 | clus u×v | contrib |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 28/12/1 | 0 | 1532.6 | −49.188 | −46.571 | 0.659 | 15.746 | 0.083 | 1×1 | [29] |
| 1 | 29/0/0 | 3 | 1455.1 | | 766.446 | 0.529 | 0 | 0 | | [] |
| 2 | 30/0/0 | 3 | 1348.8 | | −53.452 | 0.369 | 0 | 0 | | [] |
| 3 | 24/8/86 | 2 | 1238.1 | | −15.870 | 0.240 | 0 | 0 | | [] |
| 4 | 24/8/0 | 3 | 1216.0 | | −56.625 | 0.211 | 0 | 0 | | [] |
| 5 | 24/6/1569 | 0 | 932.0 | 23.720 | 23.727 | 0.023 | 0.049 | 0 | 1×2 | [7] |
| 6 | 24/6/1590 | 0 | 919.2 | −17.720 | −17.752 | 0.228 | 0.019 | 0.029 | 1×2 | [7] |
| 7 | 24/6/0 | 3 | 897.8 | | −57.387 | 0.202 | 0 | 0 | | [] |
| 8 | 24/4/1088 | 0 | 643.0 | −10.840 | −11.220 | 0.151 | 6.291 | 0.031 | 1×2 | [7] |
| 9 | 24/4/0 | 3 | 620.4 | | −50.394 | 0.127 | 0 | 0 | | [] |
| 10 | 24/2/712 | 0 | 447.5 | −0.975 | −1.205 | 0.156 | 2.160 | 0.079 | 2×2 | [7] |
| 11 | 24/2/0 | 3 | 422.9 | | −40.516 | 0.124 | 0 | 0 | | [] |
| 12 | 20/2/0 | 3 | 335.0 | | −34.769 | 0.040 | 0 | 0 | | [] |
| 13 | 17/8/0 | 3 | 282.5 | | −31.022 | 0.008 | 0 | 0 | | [] |
| 14 | 17/8/1045 | 0 | 270.7 | −2.475 | −2.407 | 0.094 | 0.662 | 0.026 | 1×5 | [7] |
| 15 | 17/6/0 | 3 | 171.9 | | −22.343 | 0.007 | 0 | 0 | | [] |
| 16 | 17/6/668 | 0 | 161.5 | −7.375 | −7.371 | 0.007 | 0.167 | 0 | 1×5 | [7] |
| 17 | 17/6/654 | 0 | 158.1 | 6.525 | 6.525 | 0.254 | 0.023 | 0.032 | 1×5 | [7] |
| 18 | 17/4/0 | 3 | 81.3 | | −14.479 | 0.020 | 0 | 0 | | [] |
| 19 | 17/4/361 | 0 | 68.2 | 0.325 | 0.325 | 0.612 | 2e−5 | 0.025 | 1×5 | [7] |
| 20 | 17/2/0 | 3 | 11.2 | | −7.884 | 0.099 | 0 | 0 | | [] |
| 21 | 17/2/12 | 0 | 0.0 | −6.572 | −6.572 | 1.000 | 2e−13 | 0 | 2×5 | [7] |
| 22 | 17/2/110 | 0 | −3.3 | 6.325 | 6.338 | 0.006 | 2.075 | 0 | 1×5 | [7] |

Per-track values: `nStates` 23, `nMeasurements` 11, `nPredicted` 23,
`nFiltered` 23, `nSmoothed` 22. The measurement-only branches have length
11 here (`dim_hit` = [1, 2, 2, …]: the long-strip hit is one-dimensional,
the rest two-dimensional); never zip them against the 23-long branches.

Things to notice:

- **`l_x_hit` is the accepted measurement's position and `eLOC0_prt` the
  prediction.** Their difference is the residual the χ² is built from. The
  expansion marks a candidate as "the one the CKF took" by matching its
  `local0` to `l_x_hit` (`is_ckf_selected`, tolerance 10⁻⁴ mm).
- **`chi2` is the increment at that state**, zero at holes and material.
- **`pathInX0_interval`** is the material crossed since the previous
  measurement surface, in radiation lengths; the pixel-to-strip transition
  (root 10, 0.079) and the outermost step (root 0, 0.083) are the thickest.
- **`clus_size_u × v`** is the cluster shape, 1×5 for most pixel hits (the
  shallow crossing angle from doc 02), 1×2 for the strips.
- **The truth columns are per state.** `t_x, t_y, t_z` hold the simhit
  position of the accepted measurement's particle, NaN elsewhere, and
  `t_eQOP` its true q/p (−0.215 /GeV here). The parquet's
  `truth_residual_*` placeholders could be filled from these.

## The trap: the file is stored outermost-first

Look at `path` in the table: it runs from 1532.6 mm at index 0 down to
−3.3 mm at index 22. The ACTS writer (`RootTrackStatesWriter`) iterates
`trackStatesReversed()`, so **array index 0 is the last state the CKF
created and the last index is the seed**. Every "past" and "future" along
a branch is the other way round from the array.

Until 2026-09-08 the expansion took the array index as `step_k`. The value
target, the branch-history counters, the seed-majority particle and the
tier-3 walker were all computed on the mirrored branch. The loader now
reverses (`expansion.propagation_order_index`) and checks the orientation
(`expansion.check_propagation_order`): the seed state must have
`pathLength` 0, the outermost state a larger value, and the first step must
increase. `pathLength` is *not* monotone in between (the CKF resets its
stepper when it resumes a forked branch; 82% of tracks show at least one
drop), which is why the guard tests orientation and not monotonicity. The
tests are in `tests/test_state_order.py`; the audit
(`scripts/audit_expansion.py`) checks the same thing on a finished parquet.

Every parquet produced before that date, including the example in
`cckf_handoff/`, still carries the reversed `step_k` (doc 06 shows the
effect). Re-expansion is step 3 of `NEXT_STEPS.md`.

## The companion file: `predicted-cov.csv`

`event000000004-predicted-cov.csv` is written per state by
`utils/predicted_cov_writer.py`: `track_nr`, `step_k`, `eLOC0_prt`,
`eLOC1_prt`, `P00`, `P01`, `P11`, where P is the predicted covariance
projected onto the surface (the `H C Hᵀ` term). It exists because the
`S11_prt` branch above was unreliable; the expansion forms
S = V + P per candidate from this file and the candidate's own variance
(doc 06). Its `step_k` counts in ROOT order like the file above;
`expansion.load_predicted_cov` converts it using each track's state count.
About 2.5 GB per envelope event.

## How to read it

```python
import uproot, awkward as ak, numpy as np
t = uproot.open("trackstates_ckf.root")["trackstates"]
ev = t.arrays(["event_nr", "track_nr"], library="np")
i = int(np.where((ev["event_nr"] == 4) & (ev["track_nr"] == 128909))[0][0])
tr = t.arrays(["volume_id", "layer_id", "module_id", "stateType", "pathLength",
               "l_x_hit", "eLOC0_prt", "err_eLOC0_prt", "chi2",
               "particle_ids_particle"], entry_start=i, entry_stop=i + 1)[0]
n = len(tr["volume_id"])
for root in range(n - 1, -1, -1):          # propagation order
    print(n - 1 - root, tr["volume_id"][root], tr["layer_id"][root], tr["module_id"][root],
          tr["stateType"][root], tr["pathLength"][root])
```

Reading the whole event needs a compute node (the arrays for 1.7 M tracks
are tens of GB); `scripts/trail_trackstates.py` does that and prints the
per-track table above together with the parquet rows of doc 06:

```
python scripts/trail_trackstates.py <run_dir> <expanded.parquet> <event> pv sv part gen sub
```

`uproot`, `awkward`, `pyarrow` and this repository on `PYTHONPATH` are all
it needs (`NERSC.md` §6). `expansion.load_trackstates(root_path, event_id)`
is the production reader; it returns a flat DataFrame with `state_idx`
already in propagation order.
