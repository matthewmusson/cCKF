# archive/

Code that produced a logged result and is no longer on the active path.
Nothing here is imported by the pipeline or the tests. Each file is kept so
the corresponding `experiments/LOG.md` entry (or the parent repo's
`SURP/experiments/LOG.md`) stays reproducible. Git history has the full
provenance; these copies are for reading without checking out old commits.

## july_ckf_optimization/  (July 2026, classical CKF tuning on Modal/NERSC)

| File | What it was | Logged in |
|------|-------------|-----------|
| `analyze_sweep.py` | Post-processed the one-parameter CKF sweeps (χ², branch cap, nMeas) into ε/f tables | `SURP/experiments/LOG.md` Stage 1-3 |
| `optimizer.sbatch` | NERSC job wrapper for the Optuna optimizer (the optimizer itself was removed in 17b57ea) | `SURP/experiments/LOG.md` Stage 5 |
| `sweep.sbatch` | NERSC job wrapper for the config sweeps (`configs/sweep_*.yaml`) | same |
| `run_single_event.sh` | Phase-1 validation: one ttbar event through digi+reco | same |

The MOTPE operating points that came out of this work live on as
`configs/tight_t79.yaml`, `configs/medium_t70.yaml` and the `_motpe_*` configs.

## superseded/

| File | What it was | Replaced by |
|------|-------------|-------------|
| `patch_parquet.py` | Filled NaN columns in the first (pre-endcap-fix) expanded parquets | re-expansion with the fixed loaders (LOG 2026-08-25) |
| `window_failure.py` | The *censored* window-failure analysis: conditioned on the true hit lying inside the 10σ box, so its n=10 line was zero by construction | `scripts/winfail_uncensored.py` (LOG 2026-09-03); its two figures are kept in `figures/winfail_censored/` |

## diagnostics/  (one-off analyses, all logged)

| File | What it answered | Logged in |
|------|------------------|-----------|
| `analyze_chi2_gate_calibration.py` | Model-free calibration of the constant χ² gate (the baseline the learned gate is compared against) | LOG 2026-08-13/17 |
| `diagnose_chi2_diag_approx.py` | Diagonal vs full innovation-covariance χ² | LOG 2026-08-13 |
| `estimate_chi2_calib_storage.py` | Storage estimate for the χ² calibration rows (spec §6.6 check 3) | LOG 2026-08-13 |
| `plot_reliability_diagrams.py` | χ² reliability diagrams from the slim calibration rows; superseded by `scripts/plot_gate_figures.py` | LOG 2026-08-17 |
| `compare_pure_vs_all.py` | First pure-seed vs all-seed gate/value comparison (Modal era) | LOG 2026-08-19 |
| `score_gate_by_volume.py` | Found the gate rejecting 100% of long strips on events 0-3 | LOG 2026-08-24 |
| `generate_cckf_patch.sh` | Dev convenience for applying the cCKF integration; superseded by `scripts/apply_cckf_integration.sh` | — |

## analysis_aug12/  (2026-08-12/13 pilot window-failure and χ² studies)

Recovered 2026-09-08 from untracked files in the primary checkout. These
produced the first window-failure plots (`figures/winfail_aug12/`), the
value-target distribution plots (`figures/value_targets_aug19/`,
`results/value_targets_aug19/`) and `results/analysis/`. All are
**censored** analyses on the pilot parquets (conditioned on the true hit
lying inside the 10σ box) and predate the endcap fix; superseded by
`scripts/winfail_uncensored.py`.

| File | What it plotted |
|------|-----------------|
| `analyze_chi2_by_eta.py` | χ² distribution by sensor type and |η| |
| `analyze_material_vs_winfail.py` | window-failure rate vs η against accumulated X/X₀ |
| `analyze_posfrac_eta.py` | positive fraction vs occupancy per |η| bin, per window n |
| `analyze_training_dist.py` | expanded-parquet training distributions (Modal) |
| `analyze_winfail_signed_eta.py` | window-failure rate vs signed η |
| `analyze_winfail_stratified.py` | same, stratified by sensor type and purity |
| `analyze_with_errors.py` | the three headline figures with binomial error bars |
| `plot_winfail.py`, `winfail_eta_material.py` | v2 of the above (`analysis_v2/`) |
