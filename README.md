# cCKF — calibrated Combinatorial Kalman Filter

Research code from Matthew Musson's summer 2026 project (Stanford/SLAC,  
supervised by Lauren Tompkins, mentored by Rocky Garg) on replacing the  
hand-tuned decision heuristics inside the ACTS Combinatorial Kalman Filter  
(CKF) with learned, calibrated decision functions. 

The Kalman engine is not touched; learning is confined to three decisions the classical CKF  
makes with fixed cuts: whether a candidate hit belongs to the track (the  
**gate** g_ψ, replacing the χ² cut), whether a branch is worth continuing  
(the **value function** V_φ, replacing the hole and branch caps), and which candidates to keep at the end (a score q_ω plus set packing, replacing greedy ambiguity resolution; not built). 

Data are ColliderML ttbar events at 200 pileup on the Open Data Detector (ODD), simulated with Geant4, reconstructed with ACTS on NERSC Perlmutter. 

**State on 2026-09-08.** A trained gate and value function run inside ACTS
and produce a (τ_g, τ_v) Pareto front on one event, but the front sits an
order of magnitude above the classical operating points in fake rate.

The root cause found on the 8th was that every expanded training parquet
stores branch states outermost-first, so every "past/future along the
branch" quantity (the value target, the history counters, the seed-majority
label) was inverted at training time. 

The loader is fixed with new guards implemented; the data, caches, models, and sweeps have not been regenerated. 

`NEXT_STEPS.md` is the plan for doing that and for what comes after.

`experiments/LOG.md` is the dated record of every experiment. 

For the physics and the locked design decisions read
`docs/cCKF_specification.md` (the master spec, last revised 2026-07-28; the
original lives in the parent repo as an extensionless `cCKF_Specification`
file). For agent instructions read `CLAUDE.md`. For where every file on
NERSC lives, read `NERSC.md`.

## Phases

The directory map below tags each directory with the phase that produced it.
The phases, in order:


| Phase   | When        | Phase                                                                                                                                          | What came out of it                                                                                                                                                                                                          | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phase 1 | Jul 15 – 28 | Classical CKF tuning on Modal (seeding, then joint seeding+CKF optimization with Optuna)                                                       | the tight / medium / fast operating points (`configs/tight_t79.yaml`, `medium_t70.yaml`, `_motpe_*`), `experiments/joint_motpe/`                                                                                             | I began the Summer (after learning how the CKF works) by attempting to optimize it on ColliderML. My optimization included CKF parameters like (chi_2 cut off), unlike prior work which only touched seeding parameters `~/priorOptimizationPaper.pdf`                                                                                                                                                                                                                                                                                                                                                      |
| Phase 2 | Aug 3 – 12  | ACTS instrumentation (innovation covariance, X/X₀, cluster shape written into the track-states ROOT) and the pilot data collection + expansion | `instrumentation.patch`, `expansion.py`, the first parquets, the first window-failure plots                                                                                                                                  | I then instrumented ACTS to collect the additional data needed to train my gate function. This data was collected by rolling out the CKF with a loose yet optimized config. I then wrote the expansion script to take all the hits on a surface and construct (x, y) data pairs from them.                                                                                                                                                                                                                                                                                                                  |
| Phase 3 | Aug 13 – 18 | Gate and value training on the pilot data; calibration audit                                                                                   | `cckf/`, `scripts/train_*.py`, `figures/gate/`, `results/calib_maj/`                                                                                                                                                         | I then trained the gate and value function on this data.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Phase 4 | Aug 19 – 24 | The C++ integration into ACTS and the first real-weights runs (debugging SIGSEGV, the nσ pre-filter)                                           | `acts_patches/`, `scripts/build_cckf_nersc.sh`, `digi_and_reco.py`                                                                                                                                                           | I then wrote integrations into ACTS to run these new models within the Kalman Filter harness. There were some initial bugs where all training data was collected in a certain search window but during inference all hits on an entire surface were being considered.                                                                                                                                                                                                                                                                                                                                       |
| Phase 5 | Aug 25 – 29 | Move from modal to NERSC; endcap geometry-id fix and full re-expansion; weights_v3; grid and EHVI Pareto sweeps; gate-window scan              | `scripts/ehvi_sweep.py`; on NERSC (persistent, see `NERSC.md`): `/global/cfs/cdirs/atlas/mussonm/cckf_handoff/{weights_v3,models_v3,results}` and one complete example run + expanded parquet under `cckf_handoff/examples/` | I had been running all of these experiments on a separate cloud provider Modal and had to switch over to NERSC. I then found the first of a series of bugs where long strips were missing from the training data. I had to regenerate the training data and then ran some optimizations to find a pareto front for the gate and value function decision thresholds. (i.e. above what value do you accept a hit or keep a branch)                                                                                                                                                                            |
| Phase 6 | Sep 2 – 4   | Uncensored window-failure and module-failure plots; classical points through the harness; regeneration of tight/fast data                      | `scripts/winfail_uncensored.py`, `figures/winfail_*`, `scripts/nersc/regen_*`                                                                                                                                                | I had made plots early on of the window failure rate, how often a true hit falls out of a nsigma search window around the predicted measurement, and was concerned about how high the failure rate was. I then realized that this was a function of the CKF config you used, namely I was using a loose config from my initial data collection and found that by using a tighter config the failure rate dropped significantly.- `figures/winfail_uncensored/envelope/` — the loose collection config, 32 events- `figures/winfail_uncensored/tight_t79/` — regenerated with the tight config, 8 events |
| Phase 7 | Sep 4 – 8   | Tier-3 (re-propagated) value target with a χ² window; its acceptance gate failed; root cause = state order; handoff cleanup                    | `cckf/tier3_*`, `scripts/stitch_tier3.py`, `scripts/audit_expansion.py`, this README                                                                                                                                         | I found another bug where the hits within a branch were ordered backwards (outermost layer was number 0...) which messed up the training data for the gate and value function.                                                                                                                                                                                                                                                                                                                                                                                                                              |




## Directory map

Everything is in this repository plus the NERSC filesystem (`NERSC.md`). Phase tags refer to the table above. 

Ignored directories (not in git) are listed at the end because they exist on the working machines.

```
cCKF/
├── README.md                 this file
├── NEXT_STEPS.md             the handoff plan: what to do, in what order, with the requirements after each step
|
├── NERSC.md                  where every input, output, build and job lives on Perlmutter
|
├── CLAUDE.md                 instructions for coding agents
|
├── experiments/LOG.md        the dated experiment record: Part I the July classical tuning, Part II cCKF
│
├── expansion.py              [P2] ROOT track states + digi CSVs -> one expanded Parquet per event: every
│                             in-window candidate per CKF state with truth labels, cluster features,
│                             branch history, seed-majority particle. The propagation-order guard lives here.
├── digi_and_reco.py          [P1] the ACTS Python pipeline: digitisation -> seeding -> CKF (classical or cCKF)
│                             -> ambiguity resolution -> writers. Driven by a YAML config from configs/.
├── instrumentation.patch     [P2] git patch on the pinned ACTS commit: writes innovation covariance S,
│                             X/X0 per interval, cluster shape/charge and incidence angles into trackstates.root
├── setup_env.sh              [P1] shifter container + spack-Python environment on NERSC (source it)
├── requirements-train.txt    [P3] the Python training environment (torch, pyarrow, sklearn, wandb)
│
├── acts_patches/             [P4] the C++ that goes INTO ACTS at build time (copied by scripts/apply_cckf_integration.sh)
│   ├── cckf/                 header-only library: MLP inference, weight blob loader, the gate selector,
│   │                         the value branch-stopper, feature construction, sensor lookup, timers,
│   │                         and the truth-greedy rollout selector used for tier-3 targets
│   └── ActsExamples/TrackFinding/
│                             CckfTrackFindingAlgorithm (the CKF with our delegates wired in) and
│                             TruthRolloutAlgorithm (offline truth rollouts from logged states)
│
├── cckf/                     [P3, P7] the Python package: pure, unit-tested pieces of the training pipeline
│   ├── features.py           gate (26) and value (11 / 12 windowed) feature vectors from parquet columns
│   ├── labels.py             gate label: candidate's particle vs the branch's majority particle
│   ├── value_target.py       value targets V^{pi-dagger} from a branch's own log (no re-propagation)
│   ├── losses.py, models.py, train.py, calibration.py, metrics.py, curves.py, samplers.py
│   │                         BCE losses, the two MLPs, the training recipe, Platt fits, ECE/AUC, figure curves
│   ├── cache.py, splits.py, event_selection.py, seed_purity.py
│   │                         memmapped feature caches, the frozen event split, pure-seed classification
│   ├── stage1_map.py, trackstates_index.py
│   │                         which stage-1 run directory holds which event (do not "simplify" into a search)
│   └── tier3_walker.py, tier3_inputs.py, tier3_stitch.py
│                             [P7] classify branch states, emit rollout worklists, compose tier-3 targets
│
├── scripts/                  CLI entry points, one job each (flat on purpose: docs and sbatch files cite paths)
│   ├── build_cckf_nersc.sh, apply_cckf_integration.sh, build_patched_acts.sh
│   │                         [P4] build the patched ACTS (bootstrap once, then rsync + incremental make)
│   ├── run_cckf_nersc.sh     [P4] run digi_and_reco.py against the patched install on NERSC
│   ├── patch_is_selected.py  [P2] mark the CKF-accepted candidate per state (position match against ROOT)
│   ├── audit_expansion.py    [P7] fail-loudly checks on an expanded parquet vs its ROOT (run before a wave)
│   ├── build_gate_cache.py, build_value_cache.py, build_caches_nersc.py
│   │                         [P3, P5] parquets -> memmapped feature caches per split
│   ├── train_gate.py, train_value.py, export_weights.py, eval_discrimination.py, eval_value_cal.py,
│   │   calibrate_and_audit.py, export_gate_curves.py, plot_gate_figures.py
│   │                         [P3] training, calibration audit, weight export to the C++ blob, figures
│   ├── pareto_sweep_nersc.py, launch_sweep_nersc.sh, ehvi_sweep.py, pareto_sweep.py, plot_pareto_overlay.py
│   │                         [P5] (tau_g, tau_v) grid sweep, qEHVI densification, front extraction, overlay plot
│   ├── winfail_uncensored.py, plot_winfail_uncensored.py, winfail_unc.sbatch
│   │                         [P6] uncensored window/module-failure accumulation and the four plot families
│   ├── stitch_tier3.py (+.sbatch), tier3_rollout_n.sbatch (+_check.py), t3_window_smoke.sbatch (+_check.py),
│   │   value_window_monotonicity.py
│   │                         [P7] tier-3 rollout generation per window, stitching with the truth-suffix gate
│   ├── expand_pilot.py, seed_recovery.py, probe_cluster_features.py, expand_trackstates_to_chi2_rows.py
│   │                         [P2] the pilot-era expander and the spec 6.6 pilot checks (imported by modal_apps)
│   ├── plot_value_*.py, plot_purity_completeness.py, plot_eff_fake_vs_pt.py, analyze_value_targets.py,
│   │   analyze_traces.py, extract_metrics.py, subset_edm4hep.py,
│   │   sweep_parquet_purity.py, smoke_test_cckf.sh, train_gate_nersc.sbatch
│   │                         plotting, metrics extraction and one-off audits
│   ├── instrumentation/      [P2] verifies the instrumentation branches in a trackstates ROOT
│   └── nersc/                [P4-P7] every Slurm job and driver that ran the pipeline, by stage (own README)
│
├── configs/                  YAML for digi_and_reco.py. tight_t79 / medium_t70 / _motpe_* are the classical
│                             operating points [P1]; cckf_envelope is the loose data-collection config [P2];
│                             nersc_cckf_full_dm is the sweep base [P5]; _t3_* drive rollouts [P7];
│                             odd-digi-geometric-config.json is the ODD digitisation
├── utils/                    [P1] config/CLI parsing, timing logger, the predicted-covariance CSV writer,
│                             the decision-log parquet schema
├── tests/                    pytest suite (413) + two C++ parity tests; `python -m pytest tests -q`
│
├── docs/                     cCKF_specification.md (the master spec), implementation plans and design specs
│                             (superpowers/plans, superpowers/specs), the pilot data schema, the Phase-1
│                             calibration-plot spec, the trackstates branch reference
├── experiments/              LOG.md plus the July optimisation data [P1] (Optuna trials, MOTPE per-event
│                             results, evals) and the gate pilot report [P2]
├── results/                  small result files: gate curves [P3], calibration audit + figure G3 [P3],
│                             the first window-failure npz [P2], the Aug-12 analysis outputs
├── figures/                  gate figure set [P3], window-failure figures [P2, P6], value-target plots [P3]
├── output/                   the Aug-12 Modal-era calibration diagnostics (plots + summary) [P2]
├── slides/                   the Aug-27 project deck (tex + pdf + its figures)
├── modal_apps/               [P1-P4] LEGACY: the Modal cloud apps; volume gone; kept for the optimizer
│                             and the original build/expansion drivers (own README)
└── archive/                  superseded and one-off code, each entry indexed to its log entry (own README)

Ignored (exist on disk, never committed):
    data/                     ODD field map and material maps (from the ODD install)
    value_v0/                 first value-model predictions (46 MB)
    experiments/chi2_gate_calib/   12 GB of chi2 calibration rows
    results/winfail_{branches,rows}/  830 MB of window-failure intermediates
    output/*.root, output/gate.bin  ACTS run outputs
    .claude/, .superpowers/   agent worktrees and plan-execution ledgers
```


## Where the code is, how to test it, how to run it

Every command below is run from the repository root on NERSC
(`cd /global/cfs/cdirs/atlas/<user>/cCKF`) unless it says otherwise.
`python -m pytest tests -q` runs the whole Python suite (413 tests, about
15 s, no data needed).

### 1. Classical CKF optimisation
- **Code:** `modal_apps/modal_acts.py`, functions `run_seeding_optimizer`, `run_ckf_optimizer`, `run_joint_motpe_optimizer`, `validate_joint_motpe_eval` (Optuna; the TPE sampler stands in for MOTPE in Optuna 4). The resulting operating points are `configs/tight_t79.yaml`, `configs/medium_t70.yaml`, and the `configs/_motpe_*` harness configs; trials, evaluations and per-event results are in `experiments/joint_motpe/`; the full record is Part I of `experiments/LOG.md`.
- **Tests:** none. This was Modal-era code, validated by re-running the whole Pareto front on held-out events (see the caveat under "Conventions that bite").
- **Run:** the Modal path no longer runs (the volume is gone). To re-run a classical config through today's harness on one event: `sbatch scripts/nersc/run_p1_input.sbatch _motpe_tight_greedy.yaml motpe_tight $SCRATCH/cckf/modal_backup/events/edm4hep.root $SCRATCH/cckf/runs_classical`. Re-optimising on NERSC means porting the objective functions out of the Modal app; see `NEXT_STEPS.md` step 1.

### 2. The ACTS insertions
- **Code:** `instrumentation.patch` (extra branches in `trackstates_ckf.root`), `acts_patches/cckf/*.hpp` (MLP inference, weight blob, gate selector, value branch-stopper, features, sensor lookup, truth-rollout selector), `acts_patches/ActsExamples/TrackFinding/` (`CckfTrackFindingAlgorithm`, `TruthRolloutAlgorithm`). Applied onto a pinned ACTS commit by `scripts/apply_cckf_integration.sh`; the pybind block in that script must list every config field, or the field is silently ignored at runtime.
- **Tests:** two standalone C++ parity tests, no ACTS needed: `clang++ -std=c++17 -O2 -Iacts_patches/cckf tests/test_mlp_inference.cpp -o /tmp/t && /tmp/t`, likewise `tests/test_chi2_logodds.cpp`. Their fixtures are regenerated by `tests/test_export_weights.py` and `tests/test_features.py`. `scripts/instrumentation/check_trackstate_branches.py` verifies the patched branches in a real ROOT file.
- **Build:** once, `sbatch scripts/nersc/bootstrap_build.sbatch` (clone, patch, cmake, full build into `$SCRATCH/cckf/acts/{src,build,install}`, about an hour). After editing anything under `acts_patches/`: `salloc -N1 -C cpu -q interactive -t 00:45:00 -A atlas` then `./scripts/build_cckf_nersc.sh --install` (rsync the changed files, rebuild two targets, 1 to 2 minutes). The build-error table in `CLAUDE.md` is the first place to look when it fails.

### 3. Collecting a `trackstates_ckf.root` with the insertions
- **Code:** `digi_and_reco.py` (the ACTS Python pipeline) driven by a YAML from `configs/`; `scripts/run_cckf_nersc.sh` points it at the patched install.
- **Run:** `sbatch scripts/nersc/run_p1_input.sbatch <config.yaml> <run_name> <edm4hep.root> [out_base]`, one event per job (`events` and `skip` in the config choose which). For training data use `configs/cckf_envelope.yaml`, which turns on the CSV writers the expansion needs (`output_digi_csv`, `output_seeds_csv`, `write_track_states`) and disables the terminal cuts. The run directory then holds `trackstates_ckf.root`, `event*-{measurements,cells,simhits,measurement-simhit-map,predicted-cov}.csv`, and the performance files. A complete example is `cckf_handoff/examples/envelope_event4/stage1/` on CFS.
- **Input:** `$SCRATCH/cckf/modal_backup/events/edm4hep.root` is run 0 of ColliderML ttbar; events 0 to 31 are the working set, 32 to 63 are sealed. `NERSC.md` has the full data layout and the path of the original in the ColliderML area.

### 4. Expansion
- **Code:** `expansion.py`, entry `run_expansion(trackstates_root, csv_dir, event_id, output_path, digi_config_path=...)`, which joins ROOT states to every in-window candidate and computes labels, cluster features, branch history, and the seed-majority particle. State order is converted from ROOT's outermost-first to propagation order in `propagation_order_index` and checked by `check_propagation_order`. Then `scripts/patch_is_selected.py --parquet P --root R --event-id E --out P2` marks the CKF-accepted candidate per state.
- **Tests:** `tests/test_state_order.py`, `test_expand_inline_selected.py`, `test_patch_is_selected.py`, `test_geometry_id.py`, `test_labels.py`, `test_seed_purity.py`, `test_stage1_map.py`, `test_audit_expansion.py`.
- **Run:** `sbatch scripts/nersc/reexpand.sbatch <pilot_dir> <event> <out.parquet>` for one event (`reexpand_event.sbatch` for the events that need chunking; `chunk_driver.sh` for a whole wave). Before a wave, and again after: `python scripts/audit_expansion.py --parquet <new> --trackstates <root> --event E --reference <previous parquet> --json audit_E.json` must print `AUDIT PASS`. An expanded example is `cckf_handoff/examples/envelope_event4/expanded_event000000004.parquet`; small complete ones are under `cckf_handoff/examples/regen_tight/reexpanded/`.

### 5. The gate g_ψ
- **Definition:** `cckf/models.py::GateMLP` (raw logit head), features `cckf/features.py::build_gate_features` (26 columns, listed in `GATE_FEATURES`), label `cckf/labels.py::derive_labels`, loss `cckf/losses.py::bce_with_logits` (unweighted, by design), calibration `cckf/calibration.py::fit_platt_occupancy` (4-parameter, occupancy-conditional, fit on the cal split only).
- **Tests:** `tests/test_models.py`, `test_features.py`, `test_labels.py`, `test_losses.py`, `test_calibration.py`, `test_calibration_trace.py`, `test_metrics.py`, `test_curves.py`, `test_samplers.py`, `test_cache.py`, `test_splits.py`, `test_train.py`, `test_build_gate_cache.py`, `test_gate_cache_pure.py`, `test_export_weights.py`.
- **Train:** `python scripts/build_gate_cache.py --split {train,val,cal} --parquet-dir $SCRATCH/cckf/reexpanded --out-dir <cache>` per split (`scripts/nersc/build_caches.sbatch`); `python scripts/train_gate.py --train-cache … --val-cache … --out-dir … --sampler A` (`scripts/nersc/train_gate.sbatch`, GPU); `python scripts/calibrate_and_audit.py --model <ckpt> --cal-cache … --out-dir …`; `python scripts/eval_discrimination.py --checkpoint … --cache <val> --model-type gate --norm-stats … --out-json …`; `python scripts/export_weights.py --checkpoint … --standardization … --calibration … --output gate.bin --model-type gate`. The deployed blobs and their provenance are in `cckf_handoff/weights_v3/`.
- **Run in ACTS:** set in the YAML `cckf: true`, `cckf_gate_weights: <gate.bin>`, `cckf_gate_threshold: τ_g`, `cckf_gate_window_nsigma: 10.0` (the pre-filter; training data were collected inside this window), `cckf_gate_max_candidates: 6`. `configs/nersc_cckf_full_dm.yaml` is the reference. The C++ side is `acts_patches/cckf/CckfMeasurementSelector.hpp`, wired as the `measurementSelector` delegate in `CckfTrackFindingAlgorithm.cpp`.

### 6. The value function V_φ
- **Definition:** target `cckf/value_target.py::compute_value_targets` (V^{π†} = min(completeness, purity) from the branch's own log), model `cckf/models.py::ValueMLP`, features `VALUE_FEATURES` (11) and `VALUE_FEATURES_WINDOWED` (12, adds the search-window size). The re-propagated target: `cckf/tier3_walker.py` (classify states, emit worklists), `TruthRolloutAlgorithm` (C++ rollouts, enabled by `truth_rollout: true` configs such as `configs/_t3_full_ev1.yaml`), `cckf/tier3_stitch.py::compose_targets`, driver `scripts/stitch_tier3.py`.
- **Tests:** `tests/test_value_target.py`, `test_build_value_cache.py`, `test_value_cache_windowed.py`, `test_train_value_cache_loading.py`, `test_eval_value_cal_window.py`, `test_value_window_monotonicity.py`, `test_tier3_walker.py`, `test_tier3_inputs.py`, `test_tier3_stitch.py`, `test_stitch_tier3.py`.
- **Train:** `python scripts/build_value_cache.py --split … --parquet-dir … --csv-dir <stage-1 csvs> --out-dir …` (add `--targets-dir <stitched> --window-nsigma N` for re-propagated targets, `--pure-seeds-only` for the ablation); `python scripts/train_value.py --train-cache … --val-cache … --out-dir …`; `python scripts/eval_value_cal.py --model … --cal-cache … --out-dir …`; export with `--model-type value`. Re-propagated targets: `sbatch scripts/nersc/tier3_gen.sbatch` (array over events: walker + rollouts), `NSIG=3 sbatch scripts/tier3_rollout_n.sbatch` per window, `python scripts/stitch_tier3.py <event> <nsig> $SCRATCH/cckf` (the truth-suffix gate arms itself at nsig 10 and exits 1 on failure). One event's worklist and rollouts are in `cckf_handoff/tier3_event4/`.
- **Run in ACTS:** `cckf_value_weights: <value.bin>`, `cckf_value_threshold: τ_v`. C++ side `acts_patches/cckf/CckfBranchStopper.hpp`, the `branchStopper` delegate.

### 7. Sweeps and plots
- `./scripts/launch_sweep_nersc.sh <weights_dir> <event> [runs_dir]` runs the (τ_g, τ_v) grid; `python scripts/pareto_sweep_nersc.py --runs-dir … --out pareto.csv` collects it; `python scripts/ehvi_sweep.py --pair maj --weights … --warm-csv … --runs-dir … --out … --rounds 5 --batch 8 --event 4` densifies the front (usage in its docstring; runs on a login node under `nohup`); `python scripts/plot_pareto_overlay.py results/pareto --out figures/pareto/pareto_overlay_classical.png`. The sweep CSVs are committed in `results/pareto/`.
- Window failure: `python scripts/winfail_uncensored.py <event> $SCRATCH/cckf [--parquet-dir … --csv-dir … --out-dir …]` per event (`scripts/winfail_unc.sbatch`), then `python scripts/plot_winfail_uncensored.py <npz_dir> <out_base> [--footer-tag …]`. Rendered sets: `figures/winfail_uncensored/{envelope,tight_t79,fast_t331}/`; arrays on CFS under `cckf_handoff/`. Tests: `tests/test_winfail_uncensored.py`.
