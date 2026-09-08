# cCKF — calibrated Combinatorial Kalman Filter

Research code from Matthew Musson's summer 2026 project (Stanford/SLAC,
supervised by Lauren Tompkins, mentored by Rocky Garg) on replacing the
hand-tuned decision heuristics inside the ACTS Combinatorial Kalman Filter
(CKF) with learned, calibrated decision functions. The Kalman engine stays
frozen and exact; learning is confined to three decisions the classical CKF
makes with fixed cuts: whether a candidate hit belongs to the track (the
**gate** g_ψ, replacing the χ² cut), whether a branch is worth continuing
(the **value function** V_φ, replacing the hole and branch caps), and which
candidates to keep at the end (a score q_ω plus set packing, replacing greedy
ambiguity resolution; not built). Data are ColliderML ttbar events at 200
pileup on the Open Data Detector (ODD), simulated with Geant4, reconstructed
with ACTS on NERSC Perlmutter.

**State on 2026-09-08.** A trained gate and value function run inside ACTS
and produce a (τ_g, τ_v) Pareto front on one event, but the front sits an
order of magnitude above the classical operating points in fake rate. The
root cause found on the last day is that every expanded training parquet
stores branch states outermost-first, so every "past/future along the
branch" quantity (the value target, the history counters, the seed-majority
label) was inverted at training time. The loader is fixed and guarded; the
data, caches, models, and sweeps have not been regenerated. `NEXT_STEPS.md`
is the plan for doing that and for what comes after. `experiments/LOG.md`
is the dated record of every experiment; the classical-CKF tuning that
preceded this repo is logged in the parent repo's `SURP/experiments/LOG.md`.

For the physics and the locked design decisions read `cCKF_specification.md`
in the parent repo. For agent instructions read `CLAUDE.md`. For where every
file on NERSC lives, read `NERSC.md`.

## Timeline

| When | Phase | What came out of it |
|------|-------|---------------------|
| Jul 15 – 28 | Classical CKF tuning on Modal (seeding, then joint seeding+CKF with Optuna) | the tight / medium / fast operating points (`configs/tight_t79.yaml`, `medium_t70.yaml`, `_motpe_*`), `experiments/joint_motpe/` |
| Aug 3 – 12 | ACTS instrumentation (innovation covariance, X/X₀, cluster shape written into the track-states ROOT) and the pilot data collection + expansion | `instrumentation.patch`, `expansion.py`, the first parquets, the first window-failure plots |
| Aug 13 – 18 | Gate and value training on the pilot data; calibration audit | `cckf/`, `scripts/train_*.py`, `figures/gate/`, `results/calib_maj/` |
| Aug 19 – 24 | The C++ integration into ACTS and the first real-weights runs (three SIGSEGV cycles, the Eigen `.eval()` rule, the nσ pre-filter) | `acts_patches/`, `scripts/build_cckf_nersc.sh`, `digi_and_reco.py` |
| Aug 25 – 29 | Move to NERSC; endcap geometry-id fix and full re-expansion; weights_v3; grid and qEHVI Pareto sweeps; gate-window scan | `$SCRATCH/cckf/reexpanded`, `weights_v3/`, `results/pareto_*.csv` on NERSC, `scripts/ehvi_sweep.py` |
| Sep 2 – 4 | Uncensored window-failure and module-failure plots; classical points through the harness; regeneration of tight/fast data | `scripts/winfail_uncensored.py`, `figures/winfail_*`, `scripts/nersc/regen_*` |
| Sep 4 – 8 | Tier-3 (re-propagated) value target with a χ² window; its acceptance gate failed; root cause = state order; handoff cleanup | `cckf/tier3_*`, `scripts/stitch_tier3.py`, `scripts/audit_expansion.py`, this README |
