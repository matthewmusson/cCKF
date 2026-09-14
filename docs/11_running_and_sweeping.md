# 11 — Running the learned CKF, one piece at a time, and sweeping thresholds

Docs 08 to 10 produce a gate blob and a value blob and describe where the
C++ calls them. This document is the operating manual: how to run the CKF
with the gate alone, the value function alone, both, or neither, on the
same event with the same cuts; how to sweep the two thresholds; where the
results land; and how to read them. Every command runs from the repository
root on NERSC (`NERSC.md`).

## One run

`scripts/nersc/run_p1_input.sbatch <config.yaml> <run_name> <edm4hep.root> [out_base]`
runs one event through `digi_and_reco.py` under the patched ACTS install
(debug queue, 30 minutes; one event takes one to ten minutes depending on
the thresholds). The config selects the event (`skip`) and everything
else. The output directory `out_base/run_name/` holds the performance
files of doc 04, `tracksummary_*.root`, `timing.tsv`, and for cCKF runs
`timing.csv` and `timing_traces.csv` (doc 10).

## The four modes

The two weight paths switch the two insertions independently
(`digi_and_reco.py::addCckfTracks`; `""` means off):

| Mode | `cckf` | `cckf_gate_weights` | `cckf_value_weights` | Hit selection | Branch stopping |
|---|---|---|---|---|---|
| none (classical) | false | | | χ² cuts | ACTS caps |
| gate only | true | gate.bin | `""` | **gate** | ACTS caps (`ClassicalBranchStopper`) |
| value only | true | `""` | value.bin | χ² cuts | **value function** |
| both | true | gate.bin | value.bin | gate | value function |

`configs/_ablation_{none,gate_only,value_only,both}.yaml` are the four,
differing in nothing else: the MOTPE tight seeding and CKF cuts, 3 T,
event 4, `cckf_gate_threshold 0.64`, `cckf_value_threshold 0.246`,
`cckf_gate_max_candidates 3`, `cckf_gate_window_nsigma 3.0`. Run them as

```
for v in none gate_only value_only both; do
  sbatch scripts/nersc/run_p1_input.sbatch _ablation_${v}.yaml abl_${v} \
      $SCRATCH/cckf/modal_backup/events/edm4hep.root $SCRATCH/cckf/runs_ablation
done
```

and read the four directories with `scripts/extract_metrics.py` (or the
`uproot` snippet in doc 04). What each comparison isolates:

- **gate only vs none**: the gate's hit selection against the χ² cut,
  with the classical hole caps holding the branch policy fixed. This is
  the cleanest test of the gate. It was not possible before 2026-09-14:
  the no-value fallback never stopped a branch (doc 10).
- **value only vs none**: the value function's branch policy against the
  caps, with the χ² selector holding hit selection fixed. The value
  features `sum/min_gate_logodds` are still fed, from the accepted hits'
  χ², so the value function sees the same inputs as with the gate.
- **both vs gate only**: what the value function adds on top of the gate.

The classical run in this set is also the reference for any classical
re-tune: change the cuts in `_ablation_none.yaml` and the other three
inherit them.

Two things to keep fixed across a comparison: the truth selection (pT > 1
GeV throughout; every plot footer states it) and the ambiguity resolver
(greedy). The gate's window, `cckf_gate_window_nsigma`, is part of the
operating point, not a constant: it was the single largest fake-rate lever
in the August sweeps (10σ to 3σ took the same thresholds from 45% to 14%
fake), and the training data were collected at 10σ, so a deployed window
narrower than 10σ is a distribution shift the gate was never shown.

### The four modes on event 4 (2026-09-14: weights_v3, 3 T, tight cuts, 3σ)

| Mode | ε_DM after resolution | f_DM | d_DM before resolution | tracks | CKF wall (s) | gate calls | value calls |
|---|---|---|---|---|---|---|---|
| none | 87.5% | 0.09% | 29.0% | | 11.9 | | |
| gate only | 63.1% | 0.00% | 16.2% | 925 | 23.6 | 458 k | 0 |
| value only | 85.2% | 0.00% | 27.7% | 1,677 | 53.2 | 0 | 4.5 M |
| both | 68.3% | 0.00% | 14.7% | 980 | 133.5 | 2.2 M | 498 k |

(Jobs 58314351 to 58314355; run directories `$SCRATCH/cckf/runs_ablation/abl_*`,
performance files copied to `cckf_handoff/runs/runs_ablation/`.) Read this
as a smoke test of the four modes, not as a result: the weights were
trained on the pre-fix data (reversed order, 2 T, the value inputs of
doc 09), the cuts were tuned at 2 T, and the gate runs in a 3σ window it
never saw in training. What it does establish:

- every mode runs, writes its performance files and its timing counters,
  and the modes are distinct (the call counts say which model ran);
- gate-only now stops branches: 925 tracks and a 16% duplicate rate,
  against unlimited growth before the classical stopper existed;
- with these weights the gate costs efficiency and the value function
  costs a little, and the fake rate is already zero in every learned mode
  at these thresholds, so the operating point is on the wrong side of the
  front: the thresholds are too strict for a 3σ window;
- the gate's hole counters show where the holes come from at 3σ: 453 k
  surfaces lost every candidate to the window itself, 71 k to the gate.
  Widen the window before blaming the gate.

The same four commands, on the regenerated data and retrained weights,
are the first learned-pair numbers worth reporting (`NEXT_STEPS.md`
step 6).

## Sweeping (τ_g, τ_v)

The two thresholds trade efficiency against fake rate. Three tools:

1. **Grid.** `./scripts/launch_sweep_nersc.sh <weights_dir> <event> [runs_dir]`
   submits the 4 × 3 grid `TAU_G="0.3 0.5 0.7 0.9"` × `TAU_V="0.1 0.2 0.4"`
   (override with the environment variables), each as its own job with a
   one-hour cap so a runaway low threshold becomes a data point rather
   than a stuck queue. It writes `configs/_<runs_dir>_sweep_gXpY_vXpY.yaml`
   from `configs/nersc_cckf_full_dm.yaml`; edit that base first (cuts,
   window, `bfield_tesla`).
2. **Collect.** `python scripts/pareto_sweep_nersc.py --runs-dir <runs_dir> --out results/pareto_<tag>.csv`
   reads every `sweep_g*_v*` directory into `tau_g, tau_v, efficiency,
   fake_rate, duplicate_rate_pre_ambi, duplicate_rate_post_ambi,
   runtime_per_event_s, gate_calls, value_calls, wall_seconds`.
   Efficiency is `eff_particles`, fake rate is `fakeratio_tracks`.
3. **Densify.** `python scripts/ehvi_sweep.py --pair maj --weights <weights_dir> --warm-csv results/pareto_<tag>.csv --runs-dir <runs_dir>_ehvi --out results/pareto_<tag>_dense.csv --rounds 5 --batch 8 --event 4 [--nsigma 3]`
   warm-starts a two-objective Bayesian optimiser (Optuna's BoTorch
   sampler, expected hypervolume improvement) from the grid, proposes a
   batch of (τ_g, τ_v) per round, submits them, waits, reads the results
   through PyROOT inside the container, and rewrites the CSV after every
   round (`source` = warm / ehvi_rN / fail). It runs on a login node under
   `nohup` for a few hours. A failed run is told (0, 1) so the optimiser
   steers away from it.

Then `python scripts/pareto_sweep.py --csv results/pareto_<tag>_dense.csv [--include-runtime --plot fig.png]`
prints the non-dominated front, and `python scripts/plot_pareto_overlay.py results/pareto --out figures/pareto/pareto_overlay_classical.png`
draws the fronts for the three windows with the classical points. The
committed CSVs in `results/pareto/` are the August sweeps.

## What the sweeps found, and why those numbers are provisional

Event 4, 936 truth particles at pT > 1 GeV, greedy resolver, weights_v3
(log 2026-08-26 to 29):

| Window | Front, best fake end | Front, best efficiency end | File |
|---|---|---|---|
| 10σ | 85.2% / 14.8% at (0.7, 0.4) | 94.4% / 45.2% at (0.33, 0.05) | `pareto_maj_dense.csv` |
| 5σ | 80.1% / 10.5% | 93.5% / 33.3% | `pareto_maj_n5_dense.csv` |
| 3σ | knee 91.1% / 7.4% at (0.640, 0.246) | | `pareto_maj_n3_dense.csv` |
| classical tight / medium / fast (2 T) | 90.3% / 0.00%, 93.2% / 0.37%, 91.6% / 0.00% | | log 2026-09-02 |

The fake rate tracked the window and barely moved with τ_v, which was the
first sign that the value function was not doing its job. Four things
found since make these numbers a record of the method, not a result:

1. the training parquets were stored outermost-first, inverting the value
   target and the history features (doc 05, doc 06);
2. reconstruction ran in a 2 T field on 3 T simulation, so every pT cut
   and scattering term was off by 1.5 (doc 04);
3. the value function trained with three of its eleven inputs constant or
   absent at inference (doc 09);
4. gate-only runs had no branch stopping at all (doc 10).

All four are fixed in the code as of 2026-09-14; none of the data, caches,
models or sweeps has been regenerated. That is `NEXT_STEPS.md`.

## Reading a run directory

```
abl_both/
    performance_finding_ckf.root    before ambiguity resolution: eff_particles, fakeratio_tracks,
                                    duplicateratio_tracks, the *_vs_eta/pT profiles, matchingdetails
    performance_finding_ambi.root   after
    tracksummary_ckf.root, tracksummary_ambi.root                 per-track summaries (doc 04)
    timing.tsv                      ACTS wall clock per algorithm (comma-separated)
    timing.csv                      cCKF counters: gate/value calls and ns, accept/reject, the three hole causes
    timing_traces.csv               first 100 tracks: every candidate's features and score
```

`python scripts/extract_metrics.py --output-dir abl_both --output abl_both.json`
puts all of it in one file. For a quick look at a single number, doc 04's
four-line `uproot` snippet is enough. Efficiency at a different pT
threshold without re-running: `write_matching_details: true` in the
config, then `scripts/plot_eff_fake_vs_pt.py`.

## Tests

`tests/test_winfail_uncensored.py` covers the window-failure accumulation
used to choose the window; the sweep scripts have no unit tests and are
validated by the committed CSVs they produced. `scripts/nersc/collect_metrics.py`
reads a key named `duplicationRatio_tracks` that does not exist
(`duplicateratio_tracks` is the real name), so its duplicate column is
always NaN; `pareto_sweep_nersc.py` reads the right key.
