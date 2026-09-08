# Benchmark provenance — Stage 5 joint MO-TPE

Recorded 2026-07-24 for reproducing / comparing later RL or ML trackers
against the same CKF baseline operating points.

## Software commits

| Component | Commit / tag | Notes |
|-----------|--------------|-------|
| SURP (this repo) | `c1bd85f7d3f634f76a76cebc6c0bb041e91272c8` | `main` tip when Stage 5 finished; working tree had local uncommitted cCKF/LOG edits |
| KalmanML (submodule) | `d4b93df1ea2808b8c92910d197940aff0b349f1c` | `heads/main` |
| ACTS (submodule, reference checkout) | `4de1dcbbb` (`v0.07.03-6845-g4de1dcbbb`) | Local submodule; **runtime** ACTS is the Modal/spack build below |
| ODD | **v5.0.0** | Built from `https://github.com/OpenDataDetector/OpenDataDetector.git` branch `v5.0.0` inside `Dockerfile.modal` |

## Runtime stack (Modal CKF)

| Item | Value |
|------|-------|
| Base image | `ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0` |
| ACTS install (in image) | `/spack/opt/spack/linux-x86_64/acts-main-udwtnx3aoh5lh6s76slc2fzc5szhwe7y` |
| ODD install | `/opt/ODD_v5/install` |
| Material maps | Official GitLab LFS v5 `odd-material-maps.root` |
| Material maps SHA256 | `aa8c168f8046c1b252e41af030b53787c7cf59b86cfb3ab49ada656a97ec883a` |
| Material maps size | 13 984 935 bytes (~14 MB) |
| Material maps local path | `cCKF/data/odd-material-maps.root` |
| Material maps Modal path | `surp-acts-data:/odd-material-maps.root` |
| Optuna | 4.9.0 · `TPESampler(seed=42, n_startup_trials=20)` |
| Study name | `joint_seeding_ckf_motpe` |

## Input data (edm4hep)

| Field | Value |
|-------|-------|
| Physics | ColliderML **full_pileup ttbar** v1, μ≈200 |
| Modal volume path | `surp-acts-data:/events/edm4hep.root` |
| Size | **6 612 819 004** bytes (~6.16 GB) |
| SHA256 | `7656dca207dfc96bb37c67ac524a6966d1a3ad10ebaaa4a9d3c0d493a874996f` |
| Events used | **64** (Sequencer `events=32` with `skip=0` or `skip=32`) |
| Opt split | events **[0, 32)** |
| Eval / baseline split | events **[32, 64)** |

### Upstream source (NERSC / ColliderML)

Canonical full runs (do **not** re-copy whole runs to Modal):

```
/global/cfs/cdirs/m4958/data/ColliderML/simulation/full_pileup/ttbar/v1/runs/{N}/edm4hep.root
```

(~6.2 GB / 128 events per run). Subset/upload workflow: `cCKF/scripts/subset_edm4hep.py`
then `modal volume put surp-acts-data <slim.root> /events/edm4hep.root`.

**Note:** The Modal file reports ROOT `badread` warnings on some ECal baskets
(entries around 25–59). Reconstruction still completes; prefer a clean re-upload
before any paper-quality rebenchmark.

## Metric / matching contract (keep fixed for model comparisons)

| Item | Definition |
|------|------------|
| Matching | Double-majority, threshold **0.5** both sides |
| ε_DM | Particle efficiency: \|{t∈T : ∃ DM match}\| / \|T\| |
| f_DM | Track fake rate: `fakeratio_tracks` |
| d_DM | Track duplicate rate: `duplicateratio_tracks` |
| T (reconstructable) | pT > 1 GeV, \|η\| < 3, ≥6 measurements, ≥3 pixel hits, secondaries **included**, charged, vertex ρ<24 mm, \|z\|<1 m |
| R | Post **greedy** ambi tracks (`maximumSharedHits=3`, `nMeasurementsMin=6`) |
| loc0 cut | **Off** for Optuna / operating points (ACTS defaults baseline used ±4 mm separately) |

## Stage 5 result artifacts

| Artifact | Location |
|----------|----------|
| Optuna DB | `cCKF/experiments/joint_motpe/optuna.db` (+ Modal `/data/optimizer/joint_motpe/`) |
| Opt trials CSV | `cCKF/experiments/joint_motpe/trials.csv` |
| Eval Pareto CSV | `cCKF/experiments/joint_motpe/eval_pareto_4d.csv` |
| Per-event op points | `cCKF/experiments/joint_motpe/{tight_t79,medium_t70,fast_t331,acts_impact5}_per_event.json` |
| Profiler timing | `cCKF/experiments/joint_motpe/op_points_timing.json` |
| Plots | `cCKF/experiments/plots/joint_motpe/` |
| Interactive 3D | `cCKF/experiments/plots/joint_motpe/pareto_eff_fake_runtime_3d.html` |
| Plot script | `cCKF/scripts/plot_joint_motpe_pareto.py` |

## Recommended CKF baselines for future model comparisons (eval set)

Updated Jul 27, 2026 (Fast replaces Loose for primary reporting):

| Name | Trial | \(\varepsilon\) (mean±σ) | \(f\) (mean±σ) | \(t_{\mathrm{seed}\to\mathrm{trk}}\) | Key knobs |
|------|-------|--------------------------|----------------|--------------------------------------|-----------|
| Tight | 79 | \(92.34\pm1.03\)% | \(0.121\pm0.108\)% | 24.92 s/evt | seeds=16, branch=3, nMeas=9, χ²≈16.3/20.7, holes=1, ptMin≈0.46 |
| Medium | 70 | \(96.23\pm0.74\)% | \(0.773\pm0.362\)% | 48.42 s/evt | seeds=46, branch=2, nMeas=7, χ²≈12.0/35.8, holes=1, ptMin≈0.59 |
| Fast | 331 | \(93.14\pm0.93\)% | \(0.163\pm0.182\)% | 9.81 s/evt | seeds=25, branch=5, nMeas=8, χ²≈15.4/31.9, holes=1, impact≈0.58, ptMin≈0.62 |
| Loose (legacy) | 284 | ~97.1% (pooled) | ~5.0% (pooled) | — | max ε among \(f&lt;5\%\); demoted |

ε/\(f\): per-event mean±stdev on `[32,64)`, \(N=32\).
\(t_{\mathrm{seed}\to\mathrm{trk}}\): ACTS profiler sum of `time_perevent_s` (seed→ambi),
**not** the Optuna wall objective. See `experiments/LOG.md` Stage 5 runtime note.

ACTS ODD defaults on same eval set: **ε=37.3%, f=1.14%** (`impactMax=20`, `cCKF/configs/acts_odd_defaults.yaml`).
Fair-box ACTS reference (impactMax clipped to 5 mm, inside Stage 5 range): **ε=41.0%, f=1.28%**
(per-event: \(\varepsilon=40.98\pm2.26\)%, \(f=1.273\pm0.700\)%; `acts_odd_defaults_impact5.yaml`).

Full parameter vectors: `JOINT_OP_POINTS` in `cCKF/modal_acts.py`; also
`eval_pareto_4d.csv` / `*_per_event.json` / `op_points_timing.json`.

## Future NERSC re-run (ColliderML public ttbar)

Goal: re-optimize / re-validate on NERSC with edm4hep files aligned to the **public**
ColliderML full_pileup ttbar release (larger \(N\), clean I/O), using a
**parallelism-agnostic** runtime objective (profiler \(t_{\mathrm{seed}\to\mathrm{trk}}\)
or CKF-only `time_perevent_s`). Keep DM matching + T definition fixed so Modal Stage 5
operating points remain comparable.

Upstream (current CFS layout):
`/global/cfs/cdirs/m4958/data/ColliderML/simulation/full_pileup/ttbar/v1/runs/{N}/edm4hep.root`.

Do **not** assume the Modal 64-event subset (`SHA256 7656dca2…`) is a documented
prefix of a specific public run until event-index provenance is verified.