# Next Steps for this project

Matthew's nine steps, in his words, each followed by what to read first,
the files involved, the commands, and the check that closes the step. Docs
are `docs/00` to `docs/11`; read `README.md` first for the map and
`NERSC.md` for where everything is.

## Before step 1: the state of the code on 2026-09-14

Fixed in code, but **nothing downstream has been regenerated** (no
parquet, cache, model or sweep after any of these):

| Fix | Where | Doc |
|---|---|---|
| parquet `step_k` was outermost-first; loader now reverses and checks orientation | `expansion.propagation_order_index`, `check_propagation_order`, `scripts/audit_expansion.py` | 05, 06 |
| reconstruction ran at 2 T on 3 T simulation; `bfield_tesla` (default 3.0) | `digi_and_reco.py` | 04 |
| gate-only runs had no branch stopping; `ClassicalBranchStopper` | `acts_patches/ActsExamples/TrackFinding/CckfTrackFindingAlgorithm.cpp` | 10, 11 |
| feature ablations: `--feature-groups` / `--drop-features`, padded export | `cckf/features.py`, `scripts/train_gate.py`, `scripts/train_value.py`, `scripts/export_weights.py` | 07, 10 |

Known and **not** fixed (each is a few lines; do them in step 3 or 4):

| Issue | Fix | Doc |
|---|---|---|
| hole counters count material surfaces; the ROOT file has `stateType` (0 meas, 1 outlier, 2 hole, 3 material) and the expansion never reads it | add `stateType` to `expansion._TRACKSTATE_SCALAR_BRANCHES`, exclude type 3 in `compute_branch_history` (or emit no row for it) | 05, 06 |
| `cov_00 … cov_20` are never filled, so the value features `sigma2_l0/l1` were 0 in training | in `expansion.expand_trackstates` set `cov_00 = err_eLOC0_prt²`, `cov_06 = err_eLOC1_prt²` (the rest may stay NaN) | 06, 09 |
| `x0_accumulated` is 0 at inference | implement in `CckfBranchStopperWrapper` (accumulate `pathInX0_interval`) or train with `--drop-features x0_accumulated` | 09, 10 |
| `is_barrel` column is wrong | fix the volume set in `expansion.py` (`BARREL_VOLUMES_DEFAULT` names the endcaps) | 06 |
| `scripts/eval_discrimination.py` reads `loaded.X` from a dict | `loaded["X"]`, `loaded["y"]` | 08 |
| `scripts/nersc/collect_metrics.py` reads `duplicationRatio_tracks` | key is `duplicateratio_tracks` | 11 |
| the `CLAUDE.md` blob-format section is stale | the real header is `n_features, n_hidden, n_layers` | 10 |
| the sealed events 32 to 63 were used to pick the July classical points | evaluate on events beyond 64 from the ColliderML runs | 04, NERSC.md §4 |

Code names for the value targets lag the locked naming (tier 2 =
re-propagated, tier 3 = logged branch); rename in one commit when the
value pipeline is next touched (`CLAUDE.md`).

## CKF

### 1. Potentially re-optimize the CKF config on a larger set of parameters and evaluate what kind of config we want to use to collect training data

The 3 T fix makes this mandatory, not optional: every config in
`configs/` was tuned against a filter with a 1.5× too large scattering
term and pT cuts on a 2/3 scale. The tight point went from 90.2% / 0.00%
to 87.7% / 0.09% just by fixing the field (log 2026-09-14).

- **Read:** docs 03 (the knobs), 04 (metrics and the sealed-set caveat),
  11 (the four-mode harness); `experiments/LOG.md` Part I for what the
  July optimisation searched and found; Chance's paper
  `~/priorOptimizationPaper.pdf` for the wider parameter set; the spec §6.
- **Files:** `configs/_ablation_none.yaml` (the classical reference at
  3 T), `configs/cckf_envelope.yaml` (the loose collection config),
  `modal_apps/modal_acts.py::run_joint_motpe_optimizer` (the July objective,
  to port), `scripts/ehvi_sweep.py` (a working EHVI driver over two
  parameters, to generalise), `scripts/nersc/run_p1_input.sbatch`.
- **Commands:** one classical run is
  `sbatch scripts/nersc/run_p1_input.sbatch _ablation_none.yaml <name> <edm4hep.root> $SCRATCH/cckf/runs_tune`;
  metrics via `python scripts/extract_metrics.py --output-dir <run> --output <run>.json`.
- **Done when:** a re-tuned tight point and a re-tuned loose collection
  config exist at 3 T, both logged with their event set, and the
  collection config is justified against the window-failure plots
  (`scripts/winfail_uncensored.py`, doc 11).

### 2. Use a larger set of events to collect training data, determine what nσ bounding box for data collection we should use

- **Read:** docs 01 (the input and the split), 02 (what stage 1 writes),
  06 (window membership), 11 (window as an operating point); `NERSC.md`
  §4 for reading the ColliderML runs directly.
- **Files:** `configs/event_split.json` and `cckf/splits.py` (extend the
  split; never by row), `cckf/stage1_map.py` (which run directory holds
  which event), `expansion.WINDOW_N_DEFAULT` (10 today), the collection
  config's `output_digi_csv`, `output_seeds_csv`, `write_track_states`,
  `write_predicted_cov`.
- **Commands:** `sbatch scripts/nersc/run_p1_input.sbatch <collection.yaml> ev<N> <run>/edm4hep.root $SCRATCH/cckf/stage1_v4`
  per event; `scripts/nersc/regen_stage1.sbatch` for a block of events.
- **Done when:** stage-1 directories exist for the new event set, the
  split file lists them, and the chosen window n is written into the
  collection config and the log with the reason.

### 3. Re-expand all of the events with much tighter audits on the output, i.e. take the hits selected by the CKF and construct data, label pairs from all of the hits the CKF could have chosen at a given layer

- **Read:** docs 05 (the file, the order trap, `stateType`), 06 (every
  step and column, the audit).
- **Files:** `expansion.py` (`run_expansion`, `load_trackstates`,
  `compute_branch_history`, `compute_branch_majority_pid`), the two
  unfixed items above (`stateType`, `cov_*`), `scripts/patch_is_selected.py`,
  `scripts/audit_expansion.py`, `scripts/nersc/reexpand.sbatch`,
  `reexpand_event.sbatch`, `chunk_driver.sh`, `scripts/trail_trackstates.py`
  (look at one branch before trusting a wave).
- **Commands:** `sbatch scripts/nersc/reexpand.sbatch <stage1_dir> <event> $SCRATCH/cckf/reexpanded_v4/expanded_event<E:09d>.parquet`;
  then `python scripts/audit_expansion.py --parquet <new> --trackstates <stage1>/trackstates_ckf.root --event E --json audit_E.json`
  must print `AUDIT PASS`; `python scripts/patch_is_selected.py ...` for
  the `_sel` copy the value cache needs.
- **Done when:** every event's audit passes all nine checks, one branch
  has been walked by hand with `trail_trackstates.py` and matches doc 05,
  and `n_holes` no longer counts material surfaces.

### 4. Re-propagate the CKF from wrong decisions to get the correct labels for the value function, audit these labels

- **Read:** doc 09 (tier 2 vs tier 3, the three pieces, the truth-suffix
  audit); `CLAUDE.md` for the code-name mapping.
- **Files:** `cckf/tier3_walker.py`, `acts_patches/ActsExamples/TrackFinding/TruthRolloutAlgorithm.cpp`,
  `cckf/tier3_stitch.py`, `scripts/stitch_tier3.py`, `configs/_t3_full_ev1.yaml`,
  `scripts/nersc/tier3_gen.sbatch`, `scripts/tier3_rollout_n.sbatch`.
- **Commands:** `sbatch --array=<events> scripts/nersc/tier3_gen.sbatch`
  (walker + rollout per event); `NSIG=3 sbatch scripts/tier3_rollout_n.sbatch`
  per window; `python scripts/stitch_tier3.py <event> 10 $SCRATCH/cckf`
  arms the truth-suffix gate.
- **Done when:** the truth-suffix check disagrees on under 1% of states
  at nsig 10 on every event (it was 44% on the reversed data; that number
  is the bug detector), and the tier-2 targets exist for the windows you
  will deploy.

### 5. Train both functions and audit the training curves, AUC-ROC, AUC-PR, etc. Here there are a lot of ablations you could run wherein you train on majority defined seeds or just pure seeds. You can also calibrate the output, which may be necessary if you subsample negative examples (which there are much more of in the training set - I got around this by using a very large minibatch).

- **Read:** docs 07 (features, labels, caches, the ablation flags), 08
  (gate: arms, calibration, the audit), 09 (value: targets, training).
- **Files:** `scripts/build_gate_cache.py`, `scripts/build_value_cache.py`,
  `scripts/train_gate.py` (`--sampler A`, `--feature-groups`,
  `--drop-features`, `--drop-ambiguous`), `scripts/train_value.py`
  (`--drop-features`, `--oversample-marginal`), `scripts/calibrate_and_audit.py`,
  `scripts/eval_discrimination.py` (fix its two lines first),
  `scripts/eval_value_cal.py`, `scripts/export_weights.py`; jobs
  `scripts/nersc/build_caches.sbatch`, `build_vcache.sbatch`,
  `train_gate.sbatch`, `train_value_v3.sbatch`.
- **Commands:** in doc 07 (caches), doc 08 (gate), doc 09 (value), doc 10
  (export). Until the `cov_*` and `x0` items are fixed, train the value
  function with `--drop-features sigma2_l0,sigma2_l1,x0_accumulated`.
- **Done when:** for each model a `*_metrics.json`, a calibration audit
  with ECE below 0.02 overall and 0.05 per stratum for the gate, an
  exported blob with `provenance.json`, and one log entry per ablation
  naming the dropped features and the AUCs.

### 6. Run reconstruction with just the gate function and just the value function, then both at once (step 7)

- **Read:** doc 11 (the four modes and what each comparison isolates),
  doc 10 (what runs when a weight path is empty).
- **Files:** `configs/_ablation_{none,gate_only,value_only,both}.yaml`
  (point the weight paths at the new blobs; set the window to the one the
  data were collected at).
- **Commands:** the four-line loop in doc 11; `scripts/extract_metrics.py`
  per directory.
- **Done when:** all four modes are logged on the same event with the same
  cuts and window, and gate-only against none is the headline gate number.

### 7. Run EHVI sweeps on the gate and value function to determine at what thresholds you see best efficiency, fake rate, and latency

- **Read:** doc 11 (grid, collect, densify, the provisional August fronts).
- **Files:** `configs/nersc_cckf_full_dm.yaml` (the sweep base: put the
  re-tuned cuts, `bfield_tesla`, and the window in it),
  `scripts/launch_sweep_nersc.sh`, `scripts/pareto_sweep_nersc.py`,
  `scripts/ehvi_sweep.py`, `scripts/pareto_sweep.py`,
  `scripts/plot_pareto_overlay.py`, `results/pareto/`.
- **Commands:** in doc 11 §Sweeping, in that order.
- **Done when:** a dense front per window is in `results/pareto/`, the
  overlay figure shows it against the re-tuned classical points, and the
  runtime column is reported alongside efficiency and fake rate.

### 8. Run the dataset augmentation loop I outlined with. Here we can sample different operating points from the pareto front, run the CKF with the gate and value function multiple times with each of the (tau_g, tau_v) pairs, then retrain the models on that data + the original CKF data

Not built. The pieces exist: a cCKF run with the CSV writers on produces
a stage-1 directory the expansion accepts (step 3), so a loop is
"collect at a front point → expand → append to the caches → retrain →
re-sweep". Spec §14 is the design (train loose, calibrate on-policy).

- **Read:** the spec §14; docs 06 and 07 for what the expansion and the
  cache builders need from a run.
- **Files:** `scripts/nersc/pipeline_driver.sh` (an earlier
  expansions-to-sweep driver to start from), `cckf/stage1_map.py` (it
  must learn about the new run directories), `cckf/splits.py`.
- **Done when:** one loop iteration has run end to end on a few events and
  the retrained pair's front is compared with the pre-loop front.

### 9. Try to fully parallelize seeding and the CKF on GPUs...

Out of scope for this codebase as it stands; `heptv2benchmarking/` in the
parent repository and Siqi's HEPTv2 work are the starting points.

## Habits that keep this honest

- One variable per experiment; commit before any run longer than five
  minutes; every run gets a line in `experiments/LOG.md` with the job id
  and the commit hash.
- Never open events 32 to 63 (the sealed set); evaluate on events beyond
  64 from the ColliderML runs.
- Every plot states its pT threshold in the footer.
- After any change to `acts_patches/`, rebuild (doc 10) and rerun the
  four-mode loop on one event before anything else.
