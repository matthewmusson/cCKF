# 04 — Ambiguity resolution, and how a run is graded

Doc 03 ended with the CKF producing 1.69 million track candidates for event
4 under the envelope config, four of which hold most of particle 482's hits.
A particle should be reported once. This document covers the step that
picks the survivors, the truth matching that grades them, the files those
numbers live in, and the numbers this project reached.

## Ambiguity resolution

Candidates overlap because one particle is seeded many times (eight all-true
seeds for particle 482 in doc 03) and because branches fork. ACTS' greedy
resolver (`addAmbiguityResolution` in `digi_and_reco.py`, `ambi_solver:
greedy`) works on shared measurements:

1. Drop every candidate with fewer than `nMeasurementsMin` measurements
   (`ambi_nMeasurementsMin`; falls back to the CKF's `ckf_nMeasurementsMin`,
   or 7 when that cut is disabled).
2. Repeat: find the candidate with the largest number of measurements that
   are also on another candidate; if that number exceeds
   `maximumSharedHits` (3), remove the candidate. Stop when no candidate
   shares more than 3.

It has no notion of truth and no learned score. The score-based resolver
(`ambi_solver: scoring`, `configs/odd_ambi_scoring.json`) was tried on
2026-09-02 and lost about 20 points of efficiency at no fake-rate gain; the
set-packing resolver from the specification was never built.

### Our particle after resolution

In the envelope run the four candidates 128909, 128908, 564412 and 564413
all hold particle 482's hits, so at most one survives; the resolved
`tracksummary` was not written for that run. In the tight classical run
(`cckf_handoff/runs/runs_classical/motpe_tight`) the resolved
`tracksummary_ambi.root` for event 4 holds 1,044 tracks, exactly one of
which has particle 482 as its majority particle:

| field | value | meaning |
|---|---|---|
| `track_nr` | 206 | its index among the 1,044 |
| `nStates`, `nMeasurements`, `nOutliers`, `nHoles` | 23, 11, 0, 1 | same shape as track 128909 in doc 03: ten true hits, one wrong hit on disc 28/12, one hole on strip layer 8 |
| `nMajorityHits` | 10 | measurements from the majority particle |
| `nSharedHits` | 0 | after resolution nothing overlaps |
| `chi2Sum / NDF` | 24.1 / 21 | the fit quality |
| `majorityParticleId_*` | 1, 0, 7, 0, 6 | the barcode from doc 01 |
| `t_pT`, `t_eta` | 2.355, −1.303 | truth, copied in for convenience |
| `eQOP_fit ± err` | −0.328 ± 0.004 | the fitted charge over momentum at the perigee |
| `measurementVolume` | [28, 24, 24, 24, 24, 17, 17, 17, 17, 17, 17] | per measurement, outermost first (same order as doc 05) |

The `tracksummary_*.root` files have one entry per event with these
per-track arrays; `trackClassification` and the `t_*` truth columns come
from ACTS' truth matcher, which is also what the performance writer uses.

## Truth matching

A track and a particle are matched when both of these hold ("double
majority", DM, spec §3):

- **purity** ≥ 0.5: at least half the track's measurements come from the
  particle (`nMajorityHits / nMeasurements`; track 206: 10/11 = 0.91);
- **completeness** ≥ 0.5: the track holds at least half of the particle's
  measurements (10 of 10 here).

Only particles in the truth selection count (doc 01: charged, |η| < 3,
pT above the threshold, produced near the beam line, at least six
measurements with three in the pixels). Event 4 has 940 such particles at
pT > 1 GeV (`matchingdetails` in the performance file lists them with a
`matched` flag; particle (1, 0, 7, 0, 6) is row 5 and is matched). From the
matching, per run:

| Name in the file | Definition | What the project calls it |
|---|---|---|
| `eff_particles` | matched selected particles / selected particles | **efficiency** ε_DM |
| `fakeratio_tracks` | tracks matched to no selected particle at purity ≥ 0.5 / all tracks | **fake rate** f_DM (spec §3: this one, not `fakeratio_particles`) |
| `duplicateratio_tracks` | extra matched tracks beyond one per particle / all tracks | **duplicate rate** d_DM |
| `eff_tracks` | tracks matched to a selected particle / all tracks | not used; the remainder are tracks of real particles below the pT threshold |

The performance writer also stores these as functions of η, pT, φ, d₀, z₀
(`trackeff_vs_eta`, `fakeRatio_vs_pT`, …, ROOT `TEfficiency` and `TH1`
objects) and the per-track `nHoles_vs_*`, `purity_vs_*`,
`completeness_vs_*` profiles.

## The files

Every run directory has, per stage:

```
performance_finding_ckf.root     grading before ambiguity resolution   (ckf_finding_performance: true)
performance_finding_ambi.root    grading after                          (ambi_finding_performance: true)
tracksummary_ckf.root            per-track summary before               (write_track_summary: true)
tracksummary_ambi.root           per-track summary after
timing.tsv                       ACTS' per-algorithm wall clock (comma-separated despite the name)
timing.csv                       cCKF's own per-event gate/value timings (cckf runs only)
```

The scalars are ROOT `TVectorT<float>` objects. With `uproot`:

```python
import uproot, numpy as np
g = uproot.open("performance_finding_ambi.root")
eff  = float(np.asarray(g["eff_particles"].members["fElements"])[0])
fake = float(np.asarray(g["fakeratio_tracks"].members["fElements"])[0])
dup  = float(np.asarray(g["duplicateratio_tracks"].members["fElements"])[0])
```

`uproot` cannot deserialise the `TEfficiency` objects; for those use PyROOT
inside the container, as `scripts/nersc/collect_metrics.py` does.
`scripts/extract_metrics.py --output-dir <run>` writes everything, timing
included, to one JSON. The event-4 numbers for the tight classical run:

| stage | ε_DM (`eff_particles`) | f_DM (`fakeratio_tracks`) | d_DM |
|---|---|---|---|
| after the CKF | 90.2% | 0.15% | 36.7% |
| after ambiguity resolution | 90.2% | 0.00% | 0.00% |

The resolver removes every duplicate and the few fakes without losing a
matched particle: at this operating point the CKF's problem is not what it
finds but how many copies.

## Where the project got to (event 4, pT > 1 GeV, after resolution)

| Configuration | ε_DM | f_DM | source |
|---|---|---|---|
| classical, MOTPE tight (t79) | 90.3% | 0.00% | `experiments/LOG.md` 2026-09-02 |
| classical, MOTPE medium (t70) | 93.2% | 0.37% | same |
| classical, MOTPE fast (t331) | 91.6% | 0.00% | same |
| learned gate + value, 10σ window, best front point | 94.4% | 45.2% | `results/pareto/pareto_maj_dense.csv` |
| learned, 3σ window, knee (τ_g 0.64, τ_v 0.25) | 91.1% | 7.4% | `results/pareto/pareto_maj_n3_dense.csv` |

Every learned point sits far above the classical ones in fake rate. The
2026-09-08 diagnosis (doc 05, doc 06) explains part of that: the value
function was trained on inverted targets and history, and the gate on
inverted history features. What the learned pair does after a correct
re-expansion is the open question `NEXT_STEPS.md` is built around.

The sweep CSVs hold `tau_g, tau_v, efficiency, fake_rate,
duplicate_rate_pre_ambi, duplicate_rate_post_ambi, runtime_per_event_s,
gate_calls, value_calls`; `scripts/plot_pareto_overlay.py` draws them with
the classical points, and every plot footer states the pT threshold.

## Two caveats on the numbers

**The sealed events were used once, in July.** Events 32 to 63 were set
aside as the evaluation set. The Phase-1 classical tuning re-ran its whole
Optuna Pareto front on them and picked the tight / medium / fast operating
points from those results (`experiments/LOG.md` Part I, Stages 4 and 5).
Those three configurations are therefore selected on the evaluation set,
and their held-out numbers are optimistic by an unknown amount. Nothing
since August has read events 32 to 63; a fresh evaluation should use events
beyond 64 from the ColliderML runs (`NERSC.md` §4).

**Everything before 2026-09-14 was reconstructed in the wrong magnetic
field.** Writing this document exposed it: across the 848 resolved, fitted
tracks of the tight run on event 4, the fitted momentum was 0.668 times the
true momentum (16th to 84th percentile 0.661 to 0.674), with the charge
sign right on every track. `digi_and_reco.py` took the field from the ODD
detector description, a 2 T solenoid; ColliderML was simulated at 3 T (a
circle through particle 482's Geant4 hits has the 3 T radius, and the
dataset's own README says so). The fix is the `bfield_tesla` config key
(default 3.0); re-running the same configuration at 3 T gives a ratio of
1.000 (0.991 to 1.009) and a q/p pull of width 1.4 (`experiments/LOG.md`,
2026-09-14). Hit-based matching, and therefore the efficiencies and fake
rates above, never depended on the field, but every configuration in this
repository was tuned with a filter whose scattering covariance was 1.5×
too large and whose pT cuts acted on a 2/3 scale; at 3 T the tight point
gives 87.7% / 0.09% instead of 90.2% / 0.00% until it is re-tuned
(`NEXT_STEPS.md` step 1). Every dataset collected before this date shares
the same 2 T scale, so the trained models are internally consistent, but
re-collection must happen at 3 T.

## Reading `matchingdetails`

```python
import uproot
md = uproot.open("performance_finding_ambi.root")["matchingdetails"].arrays(library="pd")
ours = md[(md.particle_id_vertex_primary == 1) & (md.particle_id_particle == 7)
          & (md.particle_id_sub_particle == 6)]
print(ours)          # event_nr 4, matched True
```

`write_matching_details: true` in the config turns this tree on; it is what
`scripts/plot_eff_fake_vs_pt.py` uses to redo the efficiency at other pT
thresholds without re-running.
