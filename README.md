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


| Phase   | When        | Phase                                                                                                                                          | What came out of it                                                                                                              | Notes for Noe |
| ------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| Phase 1 | Jul 15 – 28 | Classical CKF tuning on Modal (seeding, then joint seeding+CKF optimization with Optuna)                                                       | the tight / medium / fast operating points (`configs/tight_t79.yaml`, `medium_t70.yaml`, `_motpe_*`), `experiments/joint_motpe/` | |
| Phase 2 | Aug 3 – 12  | ACTS instrumentation (innovation covariance, X/X₀, cluster shape written into the track-states ROOT) and the pilot data collection + expansion | `instrumentation.patch`, `expansion.py`, the first parquets, the first window-failure plots                                      | |
| Phase 3 | Aug 13 – 18 | Gate and value training on the pilot data; calibration audit                                                                                   | `cckf/`, `scripts/train_*.py`, `figures/gate/`, `results/calib_maj/`                                                             | |
| Phase 4 | Aug 19 – 24 | The C++ integration into ACTS and the first real-weights runs (debugging SIGSEGV, the nσ pre-filter)                                           | `acts_patches/`, `scripts/build_cckf_nersc.sh`, `digi_and_reco.py`                                                               | |
| Phase 5 | Aug 25 – 29 | Move from modal to NERSC; endcap geometry-id fix and full re-expansion; weights_v3; grid and EHVI Pareto sweeps; gate-window scan              | `scripts/ehvi_sweep.py`; on NERSC (persistent, see `NERSC.md`): `/global/cfs/cdirs/atlas/mussonm/cckf_handoff/{weights_v3,models_v3,results}` and one complete example run + expanded parquet under `cckf_handoff/examples/`                              | |
| Phase 6 | Sep 2 – 4   | Uncensored window-failure and module-failure plots; classical points through the harness; regeneration of tight/fast data                      | `scripts/winfail_uncensored.py`, `figures/winfail_*`, `scripts/nersc/regen_*`                                                    | |
| Phase 7 | Sep 4 – 8   | Tier-3 (re-propagated) value target with a χ² window; its acceptance gate failed; root cause = state order; handoff cleanup                    | `cckf/tier3_*`, `scripts/stitch_tier3.py`, `scripts/audit_expansion.py`, this README                                             | |




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

