# scripts/nersc/ — Slurm jobs and drivers that ran the NERSC pipeline

Imported 2026-09-08 from the top of `$SCRATCH/cckf/` on Perlmutter, where
they had lived un-versioned. Every job here was submitted at least once; the
experiment log (`experiments/LOG.md`) names the job ids. Companion map of the
filesystem: `NERSC.md` at the repo root.

**Paths are hard-coded to one account.** Almost every file references
`/pscratch/sd/m/mussonm/cckf` (scratch root, `$SCRATCH/cckf`) and
`/global/cfs/cdirs/atlas/mussonm/cCKF` (the CFS checkout of this repo).
To run them under another account, replace those two roots, e.g.

    sed -i 's#/pscratch/sd/m/mussonm#/pscratch/sd/n/<user>#g; s#/global/cfs/cdirs/atlas/mussonm#/global/cfs/cdirs/atlas/<user>#g' scripts/nersc/*.sbatch scripts/nersc/*.sh

`$V` in the training and cache jobs is a user venv with torch and pyarrow
(`requirements-train.txt`); `$IMG` is the ODD software container.

Jobs are submitted from the CFS checkout (`cd .../cCKF && sbatch scripts/nersc/<job>`),
after copying the file to scratch if the job `cd`s there. Read the header of
each file before submitting: several take positional arguments.

## Pipeline order

| Stage | File | What it does |
|-------|------|--------------|
| build | `bootstrap_build.sbatch` | one-off: clone pinned ACTS into `$SCRATCH/cckf/acts/src`, apply the instrumentation patch and cCKF integration, cmake, full build, install (`scripts/build_cckf_nersc.sh --bootstrap --install`) |
| build | `apply_and_build.sbatch` | incremental: re-apply `acts_patches/` and rebuild the two affected targets (`build_cckf_nersc.sh --install`), used for the tier-3 window change |
| stage 1 (CKF runs) | `run_phase1.sbatch`, `run_p1_input.sbatch`, `run_p1_mem.sbatch` | one event through digi + CKF (+ cCKF) via `scripts/run_cckf_nersc.sh`; `_input` takes `CFG NAME DATA [OUT_BASE]` and is what the sweeps submit; `_mem` logs memory |
| stage 1 | `sweep_driver.sh` | drive the 12-point (τ_g, τ_v) grid through the debug queue, 5 at a time |
| stage 1 | `submit_all.sh` | the 32 pilot-run directory map (`pilot_<id>` → event) used by the expansion drivers |
| stage 1 | `regen_stage1.sbatch` | regenerate stage-1 output for a classical config (tight/fast) on 8 events, with CSV writers on |
| expansion | `reexpand.sbatch`, `reexpand_shared.sbatch` | expand one event's `trackstates_ckf.root` + CSVs into a parquet (`expansion.py` via `reexpand_ev.py`); `_shared` on the shared queue |
| expansion | `reexpand_chunked.sbatch`, `reexpand_event.sbatch`, `reexpand_chunk.py` | chunked expansion for the five events too large for one node's memory, then merge |
| expansion | `wave_driver.sh`, `chunk_driver.sh` | keep N expansions in flight until all 32 parquets exist; `chunk_driver` supersedes `wave_driver` |
| expansion | `pipeline_driver.sh` | expansions → caches → gate retrain → sweep, self-healing |
| expansion | `patch_sel.sbatch` | mark `is_ckf_selected` on an expanded parquet (`scripts/patch_is_selected.py`) |
| expansion | `regen_expand.sbatch` | expand the 8 regenerated tight/fast events |
| caches | `build_caches.sbatch`, `build_caches_split.sbatch` | gate caches for train/val/cal from `$SCRATCH/cckf/reexpanded` (`scripts/build_caches_nersc.py`); `_split` builds one split |
| caches | `build_vcache.sbatch` | value cache (`scripts/build_value_cache.py`) |
| caches | `build_pure_caches.sbatch` | gate + value caches with the pure-seed label |
| caches | `pilot_caches.sbatch` | single-event caches for smoke tests |
| training | `train_gate.sbatch` (GPU), `train_gate_cpu.sbatch`, `train_smoke.sbatch` | `scripts/train_gate.py`; the smoke variant is a few epochs on a staged cache |
| training | `train_value_v3.sbatch`, `train_value_pure.sbatch`, `train_value_int.sbatch`, `train_value_short.sbatch` | `scripts/train_value.py` variants: the promoted v3 run, pure-seed label, interactive-queue, short |
| tier 3 | `tier3_gen.sbatch` | array over 32 events: classify branches, emit rollout worklists, run the truth rollouts (`cckf/tier3_walker.py` + the rollout pipeline mode) |
| tier 3 | `t3_window_smoke_debug.sbatch` | the window smoke on the debug queue (the regular-queue copy is `scripts/t3_window_smoke.sbatch`) |
| tier 3 | `stitch_dryrun.sbatch` | stitch event 4 against the unbounded rollouts with the truth-suffix gate armed |
| window failure | `regen_winfail.sbatch`, `winfail_emu.sbatch`, `modfail.sbatch` | uncensored accumulation on regenerated data; the abandoned emulation run; module-failure analysis |
| data movement | `mirror_modal.sh`, `pull_parquets.sh`, `archive_to_cfs.sh` | the one-time mirror of the Modal volume into `$SCRATCH/cckf/modal_backup`, and the scratch → CFS archive |
| reporting | `collect_metrics.py` | Pareto fronts from `results/pareto_*.csv` and DM metrics from every run directory (PyROOT via shifter) |

## diagnostics/

One-off investigations, each tied to a log entry:

| File | Question it answered | Log |
|------|----------------------|-----|
| `audit_ev4_regular.sbatch` | template for `scripts/audit_expansion.py` on one event (90 min, regular queue) | 2026-09-08 |
| `rollout_hit_origin.sbatch`, `rollout_hit_origin2.sbatch`, `suffix_diag.sbatch` | where tier-3 rollout hits come from; found the step_k inversion | 2026-09-08 |
| `quantify_hole_mismatch_reexp.py` | n_holes / n_seq_holes train vs C++ inference mismatch | 2026-09-04 |
| `auc_dump.py`, `g3_dump.py`, `lin_reliability.py` | AUCs and reliability curves for the four gate estimators on the cal split (figure G3) | 2026-08-26 |
| `modfail_analysis.py` | module failure × occupancy × sensor (pre-uncensored version) | 2026-09-03 |
| `profile_expand.py`, `test_cache.py`, `verify_parquets.py` | expansion profiling, cache smoke, parquet row-count verification against the Modal originals | 2026-08-25 |
