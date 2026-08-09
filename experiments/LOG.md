# cCKF Experiment Log

## 2026-07-20 — Seeding Bayesian optimization (Modal)

- **Stage:** seeding only (`num_seeds_per_spm`, `seed_minPt`, `seed_impactMax`); CKF frozen incl. `numMeasurementsCutOff=1`
- **Setup:** 8 events/trial, 8 ACTS threads, 4 parallel Modal trials; 8 random + 16 guided (24 total)
- **Score:** `-(eff - 1.0*fake)` post-ambi particle metrics
- **Data:** ColliderML full_pileup ttbar `edm4hep.root` on `surp-acts-data`
- **History:** `experiments/seeding_opt_history.csv` (also `/optimizer/seeding/` on volume)
- **Modal run:** https://modal.com/apps/musson28/main/ap-q8FhuhYV526jzlC8kG9SGt

### Best by score
| param | value |
|---|---|
| num_seeds_per_spm | ~37 |
| seed_minPt | ~0.43 GeV |
| seed_impactMax | 1.0 mm |
| efficiency | 89.7% |
| fakerate | 2.7% |
| duplicaterate | 0.0% |
| wall_time (8 evt) | ~126 s |

### Notes
- Highest efficiency trial reached ~90.3% eff / ~4.1% fake (more seeds/spm, slightly higher impactMax, slower).
- `seed_minPt` dominates: ≥~1 GeV collapses efficiency to ~20–60%.
- Baseline-ish (`seeds/spm≈40`, `minPt=0.5`) sits near the Pareto front; tight `impactMax=1` trades a bit of eff for lower fake.
- Next: freeze seeding at 3 Pareto points, jointly optimize CKF params (including `numMeasurementsCutOff`) via Optuna NSGA-II.

## 2026-07-21 — Seeding re-optimization v2 (Optuna NSGA-II, 4-param)

- **Motivation:** Stage 1 seeding used xopt with 3 params and per-particle fake rate (wrong metric). Re-running with correct DM-matched track-level metrics, and adding `sigmaScattering` which interacts with `maxSeedsPerSpM`.
- **Parameters:**
  | Param | Range | Type | Rationale |
  |-------|-------|------|-----------|
  | `num_seeds_per_spm` | [1, 80] | int | Controls how many seed candidates survive per middle SP |
  | `seed_minPt` | [0.3, 2.0] GeV | float | Minimum pT for seeds; below 0.3 unphysical, above 2.0 kills soft tracks |
  | `seed_impactMax` | [1.0, 10.0] mm | float | Max transverse impact; 1mm = primaries only, 10mm = includes secondaries |
  | `seed_sigmaScattering` | [2.0, 10.0] | float | Scattering window width (units of σ_MS). Interacts with `maxSeedsPerSpM`. |
- **CKF frozen at ACTS defaults:** `chi2Meas=15`, `chi2Out=25`, `branchCap=1`, `nMeasMin=6`, `holesOut=3`, `ptMin=0.7`
- **Objectives:** ε_DM↑ (particle eff), f_DM↓ (track fake), d_DM↓ (track dup), runtime↓
- **Setup:** 200 trials, 8 events/trial, 8 ACTS threads, 8 parallel Modal trials
- **pT threshold for T:** pT > 1 GeV, |η| < 3, ≥6 measurements, ≥3 pixel hits, charged only
- **History:** Optuna SQLite at `/optimizer/seeding_v2/optuna.db`, CSV at `trials.csv`; local at `experiments/seeding_v2_trials.csv`
- **Modal run:** https://modal.com/apps/musson28/main/ap-yckmpEkqaoBEYYtYqJobjW

### Results (200/200 trials, 0 failures)

- ε_DM range: [24.0%, 90.3%], f_DM range: [6.5%, 55.9%]
- Pareto front: 17 non-dominated points (ε↑, f↓)
- `sigmaScattering` interaction confirmed: Pareto-optimal points span σ ∈ [2.1, 8.7]

### Selected operating points (tercile on seeds/spm, eff > 80%)

| Point | seeds/spm | minPt | impactMax | σ_scat | ε_DM | f_DM | rt(8e) |
|-------|-----------|-------|-----------|--------|------|------|--------|
| Tight | 5 | 0.698 | 6.1 | 4.2 | 85.7% | 14.2% | 33s |
| Medium | 18 | 0.328 | 1.6 | 4.1 | 88.6% | 22.3% | 115s |
| Loose | 62 | 0.662 | 1.6 | 2.1 | 89.1% | 31.5% | 122s |

### Notes
- `seed_minPt` ∈ [0.3, 0.7] for all high-eff points; above ~0.7 GeV efficiency collapses
- Tight point originally selected at seeds=1 (81.3% eff), upgraded to seeds=5 for CKF compatibility
- These fake rates are under DM track-level matching (f_DM), much higher than v1's particle-level fakes

## 2026-07-22 — Joint CKF optimization (Optuna NSGA-II, per seeding point)

- **Stage:** CKF params optimized jointly: `chi2CutOffMeasurement`, `chi2CutOffOutlier` (constrained > chi2Meas), `numMeasurementsCutOff`, `nMeasurementsMin`, `maxHolesAndOutliers` (constrained + nMeasMin < 12), `ptMin`
- **Switched from xopt EI (scalar) to Optuna NSGA-II (true multi-objective):** 4 objectives: ε_DM↑, f_DM↓, d_DM↓, runtime↓
- **Metrics:** particle-level ε_DM, track-level f_DM and d_DM (standard DM matching)
- **pT threshold for T:** pT > 1 GeV (same as seeding stage)
- **Setup:** 100 trials/seeding point, 8 events/trial, 8 ACTS threads, 8 parallel Modal trials. 300 trials total.
- **Data:** ColliderML full_pileup ttbar pu200, 8 events per trial on `surp-acts-data`
- **History:** Optuna SQLite + CSV on volume at `/optimizer/ckf_{tight,medium,loose}/`; local at `experiments/ckf_{tight,medium,loose}_trials.csv`
- **Modal runs:** https://modal.com/apps/musson28/main/ap-nRU9qAi4WxFfpSLlj0hEkM (tight+loose), https://modal.com/apps/musson28/main/ap-u8KIffK3hirLmlZbjFqp9I (medium)

### Results

All 300 trials completed (tight 100/100, medium 94/100 valid, loose 89/100 valid).

#### Per-seeding Pareto highlights (pT > 1 GeV)

| Regime | Best ε | Best low-f (ε>85%) | Balanced (ε>85%, f<5%) |
|--------|--------|---------------------|------------------------|
| Tight (seeds=5) | 92.3% / 32.1% f (br=5) | 89.6% / 1.75% f (br=5) | 89.5% / 0.63% f (br=2) |
| Medium (seeds=18) | 95.8% / 21.6% f (br=2) | 95.0% / 1.67% f (br=3) | 93.9% / 0.77% f (br=1) |
| Loose (seeds=62) | 97.0% / 15.7% f (br=5) | 94.4% / 1.75% f (br=1) | 93.6% / 0.27% f (br=3) |

#### System Pareto envelope (14 points)

Dominated by loose seeding (10/14 points). Key operating points:

| ε_DM | f_DM | Seeding | Branch | χ²_meas | ptMin |
|------|------|---------|--------|---------|-------|
| 91.5% | 0.07% | loose | 3 | 6.7 | 0.39 |
| 93.0% | 0.22% | loose | 1 | 14.0 | 0.50 |
| 93.6% | 0.27% | loose | 3 | 12.1 | 0.35 |
| 95.0% | 1.67% | medium | 3 | 25.2 | 0.61 |
| 96.0% | 3.47% | loose | 5 | 7.8 | 0.44 |
| 97.0% | 15.7% | loose | 5 | 24.0 | 0.54 |

### Key findings

1. **Loose seeding dominates.** More seed candidates (62/spm) give the CKF enough input to achieve both high ε and low f simultaneously when paired with strict quality cuts.
2. **Branching gain is modest.** numMeasurementsCutOff contributes only +1–1.3% ε beyond branch=1. The Pareto front is populated by all branch values — no clear optimal.
3. **nMeasurementsMin=9 is the fake killer.** Nearly all sub-1% fake configs require ≥9 measurements per track. This aggressive quality cut eliminates fragmented fakes at mild efficiency cost.
4. **Tight seeding is bottlenecked.** Peak ε=92.3% vs loose's 97.0%. The CKF can't reconstruct particles that were never seeded.
5. **Best CKF baseline for RL comparison:** loose seeding, 93.6% ε / 0.27% f (br=3, χ²=12.1, nMeas=9).

### Known limitations

- All metrics at pT > 1 GeV threshold. Need to re-run with lower particle selector threshold and `writeMatchingDetails=True` for arbitrary-threshold reporting.
- 8 events per trial — small statistics. Pareto-optimal configs should be validated on larger event samples.
- Only ttbar pu200 evaluated. Generalization to other physics processes not tested.

## 2026-08-08 — ACTS instrumentation for cCKF gate features

Three additive patches to the ACTS fork so `trackstates.root`
carries the full gate feature vector (spec 8.2):

- **A** — full innovation covariance `S00_prt`/`S01_prt`/`S11_prt`
- **B** — `pathInX0_interval`, material X/X0 between measurement surfaces
- **C** — cluster shape/charge (`clus_*`) and incidence angles (`alpha_u/v`)

Verified numerically inert (track count, state counts and per-state chi2
bit-identical vs. the unpatched build on 1 event). No data collection run.
No offline join needed — clusters join in-memory by measurement index.

See `docs/instrumentation/trackstate_branch_reference.md`.
