# Window-Conditioned Tier-3 Value Targets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the tier-3 value target V^{π†} a function of the deployment search window n - regenerate rollouts at n ∈ {3, 5, 10}, compose per-window targets, and train one value network with `window_nsigma` as a 12th input feature - eliminating the train/deploy window distribution shift (targets currently assume an UNBOUNDED π† search; deployment runs at n=3).

**Architecture:** The C++ `TruthRolloutSelector` gains a χ² window (accept the identity-selected true hit iff χ² < n², matching the deployed `cckf_gate_window_nsigma` pre-filter semantics `chi2 < nsigma²`); rollouts are regenerated per n reusing the existing worklists (the walker's divergence structure is window-independent - it compares logged actions to π† picks in the collection set). A per-event stitch driver assembles past counts + particle totals and calls the already-implemented `compose_targets` (Tier 1, Matthew's - NOT modified by this plan). The value cache gains the `window_nsigma` feature column; training reuses the existing recipe unchanged except input width. Deployment-side C++ feature building is the final, gated task.

**Tech Stack:** Python 3.10+ (pandas, numpy, pyarrow), C++ (ACTS patch tree, rebuilt via `scripts/build_cckf_nersc.sh` on NERSC), SLURM, W&B.

## Global Constraints

- Events `[32, 64)` are sealed. Only `[0, 32)` are read.
- Code sync local ↔ NERSC CFS: `git push` local → `git pull` on CFS. Never scp source.
- SSH: `ssh -i ~/.ssh/nersc -o IdentitiesOnly=yes mussonm@perlmutter.nersc.gov`.
- Tier system: `cckf/tier3_stitch.py::compose_targets` and the value LOSS/target definition are Tier 1 (Matthew's, already written, commit 8d3dce7) - this plan may CALL but never MODIFY them. All tasks here are Tier 2/3 wiring.
- The truth-suffix gate: on truth-suffix branches, |vstar_tier3(n=10) − vstar_t2| disagreement rate must be < 1% (spec §11). The stitch driver enforces it.
- Experiment discipline: one variable - the window feature - against the unbounded-target baseline; log all training runs to W&B; commit before any run > 5 min.
- Style: type hints, NumPy docstrings, Black 88. Mark the truth pT threshold on any figure produced.
- `weights_v3/` on scratch is written only by explicit promotion with provenance.json - training outputs land in run dirs, never directly there.

## Recorded facts (verified 2026-09-03, this session)

- **Rollout selector is unbounded**: `acts_patches/cckf/TruthRolloutSelector.hpp:165` - "Deliberately NO chi2 window"; the adapter in `acts_patches/ActsExamples/TrackFinding/TruthRolloutAlgorithm.cpp:50-79` picks the lowest-χ² truth candidate with `candidateChi2` (full S = V + HCHᵀ, `.eval()` discipline) and holes when none exist. Current tier-3 hits at `$SCRATCH/cckf/tier3/hits/` therefore encode an unbounded π†.
- `addTruthRollout` is `digi_and_reco.py:138`; config keys consumed at `digi_and_reco.py:683-693` (`truth_rollout, rollout_worklist_dir, rollout_output_dir, rollout_csv_dir, rollout_max`). Pybind for `TruthRolloutAlgorithm` is patched in by `scripts/apply_cckf_integration.sh` (binding block at ~line 184).
- Worklists: `$SCRATCH/cckf/tier3/worklists/` (32 events, from the August generation) - reusable across n.
- Per-event rollout configs `_t3_ev{0..31}.yaml` on CFS `configs/`; generation job pattern: `$SCRATCH/cckf/tier3_gen.sbatch` with `skip: $((E % 2))` (pilot dirs hold 2-event ranges - skip must be LOCAL) and `CCKF_STAGE1_BASE=$SCRATCH/cckf/modal_backup/results` for `cckf.stage1_map`.
- Value features: `cckf.features.VALUE_FEATURES` (11 names) consumed by `scripts/build_value_cache.py` (X is (n_states, 11); meta carries `n_features`/`feature_names`) and mirrored in C++ `acts_patches/cckf/CckfFeatures.hpp` (comment at :38 lists the vector). `WeightBlob` carries `input_dim` in its header, so the loader is already dimension-driven.
- The walker (`cckf/tier3_walker.py::classify_event`) classifies from the expanded parquet; `state_class(k)` describes the transition into k+1; π† pick rule = lowest χ² among truth candidates (ratified, two sites, guarded by `tests/test_tier3_walker.py::test_pick_rule_sites_agree`).
- `compose_targets` input contracts (from `cckf/tier3_stitch.py`): `states(seed_id, step_k, state_class, sel_hit)`, `futures(seed_id, step_k, n_findable)` via `rollout_futures(hits, worklist)`, `past(seed_id, step_k, n_correct, n_wrong)` cumulative INCLUDING the state, `n_total_true(seed_id, N_total_true)`. Failed PID joins and missing rollouts are dropped and counted by compose_targets itself.
- Suffix-gate validity: logged accepted hits have χ² < 16.26 (envelope selector) < 100 = 10², so the n=10-window rollout can take every logged hit → tier-2 equality on truth-suffix branches holds for n=10 targets (NOT for n=3: 9 < 16.26).

## Definitions

- **Window semantics (locked)**: the rollout accepts its identity-selected true hit iff `chi2 < windowNsigma * windowNsigma`; `windowNsigma <= 0` disables the window (exact current behavior). This matches the deployed gate pre-filter (`CckfMeasurementSelector` nSigma: `chi2 < nsigma²`), so V(n) conditions on exactly what the deployed chain can reach.
- **Windows generated**: n ∈ {3, 5, 10}. n=10 is regenerated (not reused from the unbounded set) so all three target sets share identical provenance; the unbounded `tier3/hits` stays on disk as the reference.
- **The 12th feature**: `window_nsigma`, float32, appended after the existing 11 `VALUE_FEATURES`. At training time it is the n of the target row; at inference it is the chain's configured `cckf_gate_window_nsigma`.

## File Structure

- Modify: `acts_patches/cckf/TruthRolloutSelector.hpp` (context gains windowNsigma), `acts_patches/ActsExamples/TrackFinding/TruthRolloutAlgorithm.{hpp,cpp}` (Config + gate), `scripts/apply_cckf_integration.sh` (pybind field), `digi_and_reco.py` (addTruthRollout param + config key `rollout_window_nsigma`).
- Create: `cckf/tier3_inputs.py` (past counts + N_total_true builders) + `tests/test_tier3_inputs.py`.
- Create: `scripts/stitch_tier3.py` (per-event, per-n driver: walker → futures → inputs → compose_targets → targets parquet; suffix gate) + `scripts/tier3_rollout_n.sbatch` (multi-window generation) + `scripts/stitch_tier3.sbatch`.
- Modify: `scripts/build_value_cache.py` (targets source + window feature column), `cckf/features.py` (VALUE_FEATURES_WINDOWED = VALUE_FEATURES + ["window_nsigma"]).
- Modify (Task 8, gated): `acts_patches/cckf/CckfFeatures.hpp` + `acts_patches/cckf/CckfBranchStopper.hpp` (12-feature vector, legacy-11 compat switch on blob input_dim), `scripts/export_weights.py` if feature-count assertions exist.

---

### Task 1: Recon - trainer entrypoint, tier-2 N_total convention, binding site

Three facts the later tasks must not guess. Record results in this file.

**Files:**
- Modify: this file (fill the "Recon results" block)

- [x] **Step 1: Locate the value trainer used for the current v3 weights** - `grep -rn "value" modal_train.py cckf/train.py scripts/*.py | grep -i "train"` locally, and on NERSC `ls $SCRATCH/cckf/ | grep -i train` + `sacct -u mussonm -S 2026-08-26 -E 2026-08-29 -X -n -o JobName%20 | sort -u` to identify the sbatch/entry that produced the deployed value weights (the tg_* training jobs). Record: entry script, config/hyperparams file, output layout, and the exact command to launch one training run.

- [x] **Step 2: Record tier-2's N_total_true convention** - read `cckf/value_target.py` and note whether its completeness denominator counts the majority particle's MEASUREMENTS or SIMHITS, and from which table. `cckf/tier3_inputs.py` (Task 3) MUST use the same convention or the truth-suffix gate compares different quantities.

- [x] **Step 3: Record the exact pybind block** for TruthRolloutAlgorithm in `scripts/apply_cckf_integration.sh` (line numbers + the ACTS_PYTHON_MEMBER pattern used) so Task 2 patches the right site.

- [x] **Step 4: Fill in below, commit** (`docs: record recon for window-conditioned tier3 plan`).

#### Recon results (Task 1)

- **Value trainer entry + launch command** (verified on NERSC: job `57623950`, `train_value_v3.sbatch`, COMPLETED 2026-08-26 12:26:43→14:31:24, produced the deployed `models_v3/value_t2_maj/` weights):
  - Entry script: `scripts/train_value.py`. Launched via `$SCRATCH/cckf/train_value_v3.sbatch` (1 node, `--constraint=cpu`, `--qos=regular`, `--account=atlas`, `--time=06:00:00`, `--cpus-per-task=128`, `--mem=0`), inside `shifter --image=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0` with venv `/global/cfs/cdirs/atlas/mussonm/venvs/modal`, repo `/global/cfs/cdirs/atlas/mussonm/cCKF`.
  - Exact command: `python scripts/train_value.py --train-cache $SCRATCH/cckf/vcache_v3/train --val-cache $SCRATCH/cckf/vcache_v3/val --out-dir $SCRATCH/cckf/models_v3/value_t2_maj --device cpu`.
  - **Input width is NOT read from cache meta** — `train_value.py:113-115` builds `models.ValueMLP(n_features=len(feat.VALUE_FEATURES), ...)`, hardcoding the module constant (currently 11). `_load()` (`train_value.py:28-45`) *does* read `meta["n_features"]` to shape the memmapped `X.f32`, so the cache itself is dimension-driven, but the model constructor is not — **Task 7 must change this call site** to use `tr["meta"]["n_features"]` (or `X_train.shape[1]`) instead of `len(feat.VALUE_FEATURES)`, or a 12-wide windowed cache will silently truncate/crash against an 11-wide model.
  - Hyperparameter source: plain argparse CLI defaults, no external hyperparam config file. Defaults: `--depth 2 --width 128 --max-epochs 50 --seed 0 --oversample-marginal 0.0`; `--device` defaults to cuda-if-available else cpu (the sbatch pins `--device cpu`). Any hyperparameter change is a CLI-flag change to the sbatch, not a config file edit.
  - Output layout in `--out-dir` (verified: `$SCRATCH/cckf/models_v3/value_t2_maj/`): `value_model.pt` (dict with `state_dict`, `n_features`, `width`, `depth`, `feature_names`, `mu`, `sigma` — this is the single checkpoint, not a directory of epoch checkpoints), `value_metrics.json` (train/val BCE, MSE, AUC-ROC, tier-gap, red flags, full loss history), `value_val_predictions.npz` (`pred`, `target`, `aux` arrays for the val split). A `calibration/` subdirectory also exists under `value_t2_maj/` (populated by a separate calibration step, not by `train_value.py` itself — not investigated further, out of scope for this recon).

- **Tier-2 `N_total_true` convention** (from `cckf/value_target.py`, verified against `cckf/labels.py` and `expansion.py`):
  - The denominator counts **SIMHITS**, not measurements. `particle_simhit_counts()` (`value_target.py:152-166`) does `simhits.groupby("particle_id").size()` on the output of `expansion.load_simhits(csv_dir, event_id)` — i.e. every simhit row in `simhits.csv` for that particle, with no measurement/clustering step involved.
  - `simhits.csv`'s `particle_id` column is produced by `expansion.encode_particle_id(...)` from the five `particle_id_{pv,sv,part,gen,subpart}` columns (`expansion.py:687-692`) — the project's custom-encoded particle id, **not** the raw ACTS/edm4hep barcode.
  - The majority-particle id used to look up this count is `branch_majority_pid` (`value_target.py:262-266`, `.map(particle_nhits)`), which is produced upstream by `expansion.py` (column listed at `expansion.py:209`) using the **same** `encode_particle_id` scheme, so the join is in a consistent id space. `cckf/tier3_inputs.py` (Task 3) must therefore compute `N_total_true` the same way: `expansion.load_simhits(csv_dir, event_id)` grouped by encoded `particle_id`, matched against `branch_majority_pid` — counting simhits, never measurements.
  - **Concern for Task 5** (not asked for by name, but load-bearing for the truth-suffix gate): the persisted tier-2 value **cache** (`scripts/build_value_cache.py::process_event`, `build_value_cache.py:148-282`) does NOT retain `seed_id`/`branch_id` in its on-disk arrays. `X.f32` is feature-only, `y.f32` is the bare `vstar_t2` float column, and `aux.f32` is only `(vstar_t1, step_k, eta)` (`build_value_cache.py:274-281`, meta `"aux_columns": ["vstar_t1", "step_k", "eta"]`) — `seed_id` is dropped at the `np.column_stack` step. The target column name in the pre-flatten per-event frame is `vstar_t2` (merged in at `build_value_cache.py:253-266`), but there is no on-disk cache file that can be joined back to `(seed_id, step_k)` for a specific event. **Task 5's `truth_suffix_check` against "the tier-2 target cache" cannot read `$SCRATCH/cckf/vcache_v3/*` directly** — it must instead recompute tier-2 targets per event by calling `value_target.build_step_table` + `value_target.compute_value_targets` directly on the same expanded parquet (exactly what `process_event` does internally, minus the final flatten-to-array step), to get a frame keyed on `(seed_id, branch_id, step_k)` to compare against tier-3's `(seed_id, step_k, vstar_tier3)`. Flag this explicitly when Task 5 is executed.
  - Cache file layout: one directory per `--split` (`train`/`val`/`cal`, e.g. `$SCRATCH/cckf/vcache_v3/{train,val}`), each containing `X.f32`, `y.f32`, `aux.f32`, `meta.json`; `norm_stats.npz` (`mu`, `sigma`, `feature_names`) is written only for `--split train` (`build_value_cache.py:451-463`). No further per-event sub-splitting on disk — `meta.json`'s `"events"` list records which event ids were concatenated into that one flat cache.

- **Pybind block location/pattern** for `TruthRolloutAlgorithm`, in `scripts/apply_cckf_integration.sh` (verified by line number):
  - Section header comment `# --- TruthRolloutAlgorithm binding (tier-3 rollout executor) ---` at **line 184**.
  - The binding is written by a Python heredoc (`PYEOF2`, opened line 185, closed line 214) that patches `${ACTS_SOURCE}/Python/Examples/src/TrackFinding.cpp`. Guard at line 190 (`if "TruthRolloutAlgorithm" in text: skip`) makes the patch idempotent.
  - Include anchor (line ~196-199): inserts `#include "ActsExamples/TrackFinding/TruthRolloutAlgorithm.hpp"` right after the existing `CckfTrackFindingAlgorithm.hpp` include.
  - Binding-block anchor (line ~200): matches the tail of the *previous* (`CckfTrackFindingAlgorithm`) binding block, `"        outputTimingPath, digiConfigPath);\n  }\n"`, and appends after it.
  - The exact appended block (lines 204-211 as constructed in the heredoc):
    ```cpp
    {
      using RAlg = TruthRolloutAlgorithm;
      auto [ralg, rc] = declareAlgorithm<RAlg, IAlgorithm>(mex, "TruthRolloutAlgorithm");
      ACTS_PYTHON_STRUCT(rc, inputMeasurements, outputTracks, worklistDir,
          outputDir, csvDir, trackingGeometry, magneticField, findTracks,
          maxSteps, maxRollouts);
    }
    ```
  - **This is the exact member list Task 2 Step 4 must extend** — add `windowNsigma` inside the `ACTS_PYTHON_STRUCT(rc, ...)` call (e.g. appended after `maxRollouts`), matching the pattern already used for the sibling `CckfTrackFindingAlgorithm` binding (`ACTS_PYTHON_STRUCT(c, ...)` at lines 163-171, preceded by `declareAlgorithm<Alg, IAlgorithm>` at line 152) which lists its `Config` fields the same way (comma-separated identifiers matching the C++ `Config` struct's member names verbatim — `ACTS_PYTHON_STRUCT` reads field names, not types).
  - Related, not to be confused: the file also patches `Examples/Algorithms/TrackFinding/CMakeLists.txt` (line 68) to add `TruthRolloutAlgorithm.cpp` to the build's source list — no change needed there for a config-field-only addition.

- **`cckf/features.py` `VALUE_FEATURES`** (verified, `features.py:108-120`): exactly 11 entries — `eta, state_qop, sigma2_l0, sigma2_l1, n_hits, n_holes, n_seq_holes, sum_gate_logodds, min_gate_logodds, step_k, x0_accumulated`. `NO_STANDARDIZE = {"n_hits", "n_holes", "n_seq_holes"}` (unaffected by the 12th feature, since `window_nsigma` is a genuine float meant to be standardized like the others — confirm this design choice in Task 6, not assumed here).

- **`scripts/export_weights.py` hardcodes a feature count**: yes — `export_weights.py:55` has `assert n_features == 11, f"Value expects 11 features, got {n_features}"` (and line 51 the analogous `== 26` assert for the gate). Both asserts read `n_features` from `weights[0].shape[1]` (the first-layer weight matrix, line 46) — the blob header packs `n_features` from that same value (line 94), so the binary format itself is dimension-agnostic (confirmed independently by the plan's "Recorded facts" note that `WeightBlob` carries `input_dim`). **Task 7 Step 3 must change the `== 11` assert to `in (11, 12)`** (or drop it in favor of a print) or the exporter will hard-fail on a 12-feature windowed checkpoint.

---

### Task 2: C++ rollout window

**Files:**
- Modify: `acts_patches/cckf/TruthRolloutSelector.hpp:169-181` (TruthRolloutContext)
- Modify: `acts_patches/ActsExamples/TrackFinding/TruthRolloutAlgorithm.hpp` (Config) and `.cpp` (adapter gate + ctx wiring)
- Modify: `scripts/apply_cckf_integration.sh` (pybind member), `digi_and_reco.py:138` (addTruthRollout) and `:683-693` (config key)

**Interfaces:**
- Produces: config key `rollout_window_nsigma` (float, default 0.0 = unbounded) threaded YAML → addTruthRollout → Config.windowNsigma → TruthRolloutContext.windowNsigma → the adapter's accept gate.

- [ ] **Step 1: Context member** - in `TruthRolloutContext` add below `majorityPid`:

```cpp
  /// Chi2 window for pi-dagger acceptance: the true hit is taken iff
  /// chi2 < windowNsigma^2. <= 0 disables the window (unbounded pi-dagger,
  /// the pre-2026-09 behavior). Matches the deployed gate pre-filter
  /// semantics (CckfMeasurementSelector nSigma: chi2 < nsigma^2), so
  /// V(n) conditions on exactly what the deployed chain can reach.
  double windowNsigma = 0.0;
```

Also update the "Deliberately NO chi2 window" doc comment: identity selection is unchanged; the window only gates ACCEPTANCE of the identified true hit.

- [ ] **Step 2: Adapter gate** - in `TruthRolloutAlgorithm.cpp`, after the candidate loop (the `if (best == candidates.size())` hole branch), extend the hole condition:

```cpp
    const double nsig = m_ctx->windowNsigma;
    if (best == candidates.size() ||
        (nsig > 0.0 && bestChi2 >= nsig * nsig)) {
      // No truth candidate, or the true hit sits outside the configured
      // search window: hole, continue propagating.
      return std::make_pair(candidates.begin(), candidates.begin());
    }
```

- [ ] **Step 3: Config + wiring** - `Config` gains `double windowNsigma = 0.0;`; where the algorithm sets `rolloutCtx.majorityPid` per rollout (near `.cpp:204`), also set `rolloutCtx.windowNsigma = m_cfg.windowNsigma;` once at context setup.

- [ ] **Step 4: Pybind + python plumbing** - add `windowNsigma` to the binding member list recorded in Task 1 Step 3; `addTruthRollout(..., window_nsigma: float = 0.0)` passes it to the Config; `digi_and_reco.py:692` block gains `window_nsigma=float(getattr(config, "rollout_window_nsigma", 0.0))`.

- [ ] **Step 5: Push, pull on CFS, rebuild** via `scripts/build_cckf_nersc.sh` (the same build that produced the runs_t3 binary). Expect a partial rebuild of the TrackFinding lib + bindings.

- [ ] **Step 6: Behavioral smoke** - run event 1's rollout config twice with `rollout_max: 200`: once `rollout_window_nsigma: 0`, once `3.0`, into scratch temp dirs. Verify with a short python read: per rollout_id, `n_findable(n=3) <= n_findable(n=0)`, strict decrease for at least some rollouts, and identical rollout_id sets. Paste counts into the commit message.

- [ ] **Step 7: Commit** (`feat: chi2 window for tier-3 rollout acceptance (rollout_window_nsigma)`).

---

### Task 3: tier3_inputs - past counts and particle totals

**Files:**
- Create: `cckf/tier3_inputs.py`
- Test: `tests/test_tier3_inputs.py`

**Interfaces:**
- Produces: `past_counts(parquet_path: str) -> pd.DataFrame[(seed_id, step_k, n_correct, n_wrong)]` - cumulative per branch in ascending step_k, INCLUDING the state's own accepted hit; membership test identical to the walker's (`branch_majority_pid in contrib_pids` of the `is_ckf_selected` row; `majority_undefined` branches excluded). `n_total_true(parquet_path: str, csv_dir: str, event_id: int) -> pd.DataFrame[(seed_id, N_total_true)]` - the majority particle's total hit count under the SAME convention tier-2 uses (Task 1 Step 2; if tier-2 counts measurements, join `expansion.load_measurement_simhit_map` + `load_simhits` and count measurements per majority pid).

- [ ] **Step 1: Failing tests** - synthetic candidate-row frames (reuse the fixture style of `tests/test_tier3_stitch.py`):

```python
def test_past_counts_cumulative_including_own():
    # branch: correct at 0, wrong at 1, hole at 2, correct at 3
    # expected n_correct: 1,1,1,2 ; n_wrong: 0,1,1,1
    ...


def test_past_counts_excludes_majority_undefined():
    ...


def test_n_total_true_counts_majority_measurements():
    # two particles; majority pid has 4 measurements -> N_total_true == 4
    ...
```

(Write the full frames in the test file; each is ~10 lines in the established `_rows` style. The expected values above are the contract.)

- [ ] **Step 2: RED**, **Step 3: implement** (vectorized: is_correct per selected row via the awkward/np.fromiter membership pattern already used in `winfail_uncensored.build_state_table`; per-branch `cumsum` over ascending step_k; holes contribute 0 to both), **Step 4: GREEN** (full suite), **Step 5: black + commit** (`feat: tier3 stitch input builders (past counts, particle totals)`).

---

### Task 4: Multi-window rollout generation on NERSC

**Files:**
- Create: `scripts/tier3_rollout_n.sbatch`

**Interfaces:**
- Produces: `$SCRATCH/cckf/tier3_nsig{3,5,10}/hits/event{E:09d}-rollout-hits.csv` for all 32 events, worklists REUSED from `$SCRATCH/cckf/tier3/worklists`.

- [ ] **Step 1: Job script** - clone the August pattern (per-event array, pilot resolution via `cckf.stage1_map`, LOCAL skip `$((E % 2))`), parameterized by `NSIG`; it seds a copy of `_t3_ev$E.yaml` adding `rollout_window_nsigma: $NSIG` and pointing `rollout_output_dir` at `$SCRATCH/cckf/tier3_nsig$NSIG/hits`. Fail-fast on missing worklist/pilot (reuse the August exit codes).

- [ ] **Step 2: Smoke one event per n** (debug QOS, events 0 only, all three n) and verify the three hit files exist with `n_findable` monotone non-increasing in n for matched rollout_ids.

- [ ] **Step 3: Full generation** - three array submissions (n=3, 5, 10), debug QOS. Record job ids. Verify 32 files per n and print per-n total findable-hit counts (must be monotone in n).

- [ ] **Step 4: Commit the sbatch** (`feat: multi-window tier-3 rollout generation`).

---

### Task 5: Stitch driver + truth-suffix gate

**Files:**
- Create: `scripts/stitch_tier3.py`, `scripts/stitch_tier3.sbatch`

**Interfaces:**
- Consumes: `tier3_walker.classify_event`, `tier3_stitch.rollout_futures/compose_targets/truth_suffix_check` (UNMODIFIED), `tier3_inputs.past_counts/n_total_true`, worklist CSVs, per-n hits.
- Produces: `$SCRATCH/cckf/tier3_targets/vstar_nsig{N}_event{E:09d}.parquet` with columns `(seed_id, step_k, vstar_tier3, window_nsigma)`; and for n=10 a printed `truth_suffix_check` report per event whose aggregate disagreement rate GATES the run (exit nonzero above 1%).

- [ ] **Step 1: CLI** `python3 scripts/stitch_tier3.py EVENT NSIG SCRATCH_BASE`: classify (or load a cached classification if one exists from the walker runs), build futures from `tier3_nsig{N}/hits` + worklist, build past/n_total via tier3_inputs, call compose_targets, add the constant `window_nsigma` column, write parquet, print compose_targets' loud counters plus row count.
- [ ] **Step 2: For NSIG == 10 additionally** load the tier-2 target cache for the event (path + column per Task 1 recon), run `truth_suffix_check(states, targets, tier2, tol=0.01)`, print the dict, and exit 1 if `disagree_rate >= 0.01`.
- [ ] **Step 3: Unit-light test** - the driver is thin; test only its `window_nsigma` column attachment and gate logic with monkeypatched compose/check functions (`tests/test_tier3_inputs.py` can host these two tests).
- [ ] **Step 4: Run event 4 for all three n on NERSC**; the n=10 run must pass the gate. THIS IS THE PLAN'S ACCEPTANCE MOMENT - if the gate fails, stop and report (it measures the diagonal-seeding bias; a failure is a finding, not a fix-in-place).
- [ ] **Step 5: Batch all 32 events × 3 n** (one sbatch, 8-way parallel like winfail_unc.sbatch). Verify 96 parquets.
- [ ] **Step 6: Commit** (`feat: tier-3 stitch driver with truth-suffix gate`).

---

### Task 6: Windowed value cache

**Files:**
- Modify: `cckf/features.py` (add `VALUE_FEATURES_WINDOWED: tuple = tuple(VALUE_FEATURES) + ("window_nsigma",)`), `scripts/build_value_cache.py`
- Test: extend the cache builder's existing tests (locate them; if none exist for the builder, add `tests/test_value_cache_windowed.py` covering the two changed behaviors below with synthetic frames)

**Interfaces:**
- Produces: cache files whose X is (n_states, 12) with the 12th column = window_nsigma; meta `n_features=12`, `feature_names=VALUE_FEATURES_WINDOWED`; y = `vstar_tier3` joined from `tier3_targets/vstar_nsig{N}_event{E}.parquet` on (seed_id, step_k). One cache pass per n; the training set is their concatenation (states repeat across n with different 12th column and different y - that is the design).

- [ ] Steps: failing test (12-wide X, correct constant column, y joined from the targets parquet, rows with no target dropped-and-counted) → RED → implement (`--targets-dir` + `--window-nsigma` flags; when absent, behavior byte-identical to today) → GREEN → run the real cache build for the 24 train events × 3 n on NERSC → commit (`feat: window-conditioned value cache (12th feature)`).

---

### Task 7: Train V3(n)

**Files:**
- Modify: the trainer entry recorded in Task 1 only as far as accepting `n_features` from the cache meta (it may already); training recipe, loss, splits are UNTOUCHED (Tier 1 / frozen).

- [ ] **Step 1:** Launch training on the concatenated windowed cache with the exact command recorded in Task 1, W&B-tagged `value_tier3_windowed`. Splits per `cckf/splits.py` (frozen).
- [ ] **Step 2:** Eval on the cal split PER WINDOW SLICE: reuse `scripts/eval_value_cal.py` filtered to each n; report ECE and reliability per n. Acceptance: monotonicity sanity - mean predicted V at fixed state features must be non-increasing as window_nsigma decreases (spot-check on 1k random cal states by re-scoring with the 12th feature swapped 10→3).
- [ ] **Step 3:** Export blobs with `scripts/export_weights.py` (12 features; blob header carries input_dim so the format needs no change - verify the exporter has no hardcoded 11). NO promotion to `weights_v3/` in this plan - that requires explicit user promotion with provenance.json.
- [ ] **Step 4:** Commit any wiring diffs (`feat: train window-conditioned tier-3 value function`).

---

### Task 8 (GATED - only after Task 7's eval is accepted by Matthew): deployment feature

**Files:**
- Modify: `acts_patches/cckf/CckfFeatures.hpp` (value vector builder appends the window feature), `acts_patches/cckf/CckfBranchStopper.hpp` (receives the configured nsigma; builds 12-vector iff the loaded blob's input_dim == 12, legacy 11-vector otherwise - the compat switch keys on the blob, so old and new weights both run), `CckfTrackFindingAlgorithm.cpp` (passes `cckf_gate_window_nsigma` into the stopper's feature context).

- [ ] Steps: implement with the blob-dim compat switch → push/pull → rebuild via `build_cckf_nersc.sh` → dummy-weight smoke (`generate_dummy_weights` equivalent with 12 features) → 1-event run with the Task 7 blob at n=3 confirming the value path executes and logs (stderr diag or timing counters) → commit (`feat: window feature in deployed value vector, blob-dim compat`).

---

## Self-Review

1. **Spec coverage**: option 2 = per-window targets + n as input feature: windows generated (T4), targets per n (T5), feature added train-side (T6) and deploy-side (T8), single network trained across n (T7). The user's "pass the search window size in as a feature" is T6/T7/T8. Unbounded→windowed C++ change (T2) is the prerequisite discovered in recon (selector currently has NO window). Suffix gate preserved and enforced at n=10 (T5), where its premise holds (16.26 < 100).
2. **Placeholder scan**: Task 3 Step 1 sketches test names with contracts and defers full frames to implementation per the established fixture style - the expected count sequences ARE stated. Task 1 is a recon task by design. No TBDs elsewhere.
3. **Type consistency**: `window_nsigma` (python/config, float) vs `windowNsigma` (C++) used consistently; targets parquet columns `(seed_id, step_k, vstar_tier3, window_nsigma)` consumed verbatim by T6; `VALUE_FEATURES_WINDOWED` name used in T6 only.

## Known risks

- The rebuild (T2, T8) touches the historically fragile ACTS patch tree - follow the Eigen `.eval()` rule and the build-error table in cCKF/CLAUDE.md; budget a fix cycle.
- Walker reuse across n is an approximation at divergence states: π†'s FIRST corrected action is whatever the windowed rollout takes, but the divergence/collapse classification itself was computed against collection-set picks. This is second-order (it can only relabel states whose true hit lies between n·σ and the collection window) and is bounded empirically by the per-n suffix distributions T5 prints; if the n=3 suffix distribution looks pathological, revisit before training.
- Training-set tripling (3 n values) changes the effective epoch size; keep the recorded hyperparameters and let early stopping handle it (no recipe edits - Tier 1).
