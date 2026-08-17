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

## 2026-08-10 — Train-set expansion and patching (32 events, Modal)

### Pipeline

1. **Stage 1 (CKF runs):** 32 events in 16 batches of 2 on Modal (envelope config from §6.2). Each batch produces a ROOT trackstates file + per-event CSVs. Completed across 16 pilot dirs on `surp-acts-data` volume.

2. **Stage 2 (Expansion):** Offline join of ROOT trackstates + CSV measurements/seeds/simhits into per-event parquet files (72 columns, one row per branch-step-candidate). Stored at `results/train32/expanded/expanded_event{000..031}.parquet`. Row counts range from ~34M to ~200M per event.

3. **Stage 3 (Patching):** Fill NaN columns not populated during expansion:
   - **S00/S01/S11** (innovation covariance): joined from ROOT `S00_prt/S01_prt/S11_prt` on `(seed_id, step_k) = (track_nr, state_idx)`
   - **volume_id**: from ROOT, needed for sensor property lookup
   - **pitch_u/pitch_v/thickness/is_barrel**: from `configs/odd-digi-geometric-config.json` keyed by volume_id

### Status (2026-08-12, updated 2026-08-13)

**32/32 events fully patched** (76 columns, 100% fill on S00/pitch/volume_id).

The 11 previously corrupted events (5, 6, 10, 14, 15, 17, 18, 22, 28, 29, 30) have been re-expanded successfully. Verified 2026-08-13: all 32 files readable, 76 columns each, S00 present in all.

Row counts for the re-expanded events:
- Event 5: 132,084,368
- Event 6: 105,613,047
- Event 10: 94,272,826
- Event 14: 120,403,769
- Event 15: 113,533,892
- Event 17: 128,070,540
- Event 18: 117,759,145
- Event 22: 85,659,822
- Event 28: 86,887,680
- Event 29: 82,558,893
- Event 30: 66,556,584

All 32 events are now available for train/val/cal splits.

### Incident: concurrent volume commit corruption

First patch attempt used `starmap` (all 32 events in parallel). Each container called `data_vol.commit()` after writing. Concurrent commits corrupted 11 parquet files ("Parquet magic bytes not found in footer"). Fix: switched to sequential execution. The 21 successfully patched events are verified correct.

### Train/val/cal split (FROZEN 2026-08-13 in cckf/splits.py)

Per spec §6.1, events [0,32) split 24/4/4. All 32 events verified patched.

- **Train (24):** 0,1,2,3,5,6,8,9,10,11,13,14,16,17,18,19,21,22,24,25,26,27,29,30
- **Val (4):** 4,12,20,28
- **Cal (4):** 7,15,23,31
- **Test [32,64):** sealed, never touched

Val and cal picks are spread one-per-quarter across [0,32) and across distinct
Stage-1 generation batches (16 batches of 2), so no val event shares a batch
with a cal event. Frozen in `cckf/splits.py`; VAL and CAL must never change —
early-stopping and calibration numbers are only comparable across experiments
if the splits are identical. Enforced by `tests/test_splits.py`.

## 2026-08-17 — Gate g_ψ training, S1 sampling ablation (Modal)

First real-data gate training. Three sampling strategies (spec §2.5), one
variable each, trained then calibrated+audited **sequentially** — never in
parallel, per the concurrent-commit incident above.

- **Cache:** streaming Parquet → float32 memmap, read from `EXPANDED_DIR`
  (the gate needs no `is_ckf_selected` patch). Train 174,234,476 rows /
  858,316 pos (0.4926%); val 24,108,662 / 136,999 (0.5683%); cal
  44,372,735 / 164,094 (0.3698%). Total rows 1.77× the spec's ~137M estimate;
  positive fraction agrees with the health analysis (0.4926% vs 0.52%).
- **Model:** 26-dim gate features, 36,609 parameters, unweighted BCE.
- **Calibration:** cal split only. Reliability bins are **log-odds-uniform**
  (30 bins, p ∈ [1e-5, 0.99999]), not quantile — measured: quantile bins put
  *zero* populated bins in the decision region at a 0.4% base rate.
- **Runs:** W&B project `cckf-gate`, runs `gate_{B,C,A}_seed0`.

### Results (cal split, 44,372,735 rows; decision region p ∈ [0.01, 0.5])

| arm | sampler | AUC-ROC | AUC-PR | ECE | DR-ECE | MCE | worst stratum | wall |
|-----|---------|---------|--------|-----|--------|-----|---------------|------|
| A | none (batch 40,960) | 0.9931 | 0.7987 | 7.2e-05 | **0.00146** | 0.0126 | 0.00066 | ~76 min |
| B | uniform 1:5 (batch 4,096) | 0.9928 | 0.7710 | 1.0e-04 | 0.00240 | 0.0266 | 0.00081 | ~11 min |
| C | hard-neg ∝1/χ² (batch 4,096) | 0.9671 | 0.2922 | 2.1e-03 | 0.01589 | 0.99997 | 0.01310 | ~8 min |
| — | χ²_λ baseline | — | — | 0.0487 | 0.1246 | 0.9770 | — | — |

**Headline (figure G3):** arm A's calibrated gate reaches DR-ECE 0.00146
against χ²'s 0.1246 — **85× better** in the only region a τ sweep can reach.
All three arms set `headline_beats_chi2`.

### Key findings

1. **A > B >> C, matching the theory.** Uniform negative subsampling biases
   the logit by the constant `log(1/r)`: arm B's fitted Platt intercept is
   −3.7459 against the predicted −3.6988 (1.3% — not exact because a=0.913≠1
   makes slope and intercept trade off). Weighted sampling instead biases it
   by `log(w(x)/E[w])`, a *function of the covariate weighted on* — so C's
   intercept misses by 36% (−5.0436) with slope collapsed to 0.579, and
   Platt, being affine in the logit, cannot repair a χ²-dependent bias. C's
   DR-ECE is 6.6× B's and its MCE (0.99997) is worse than raw χ².
   Hard-negative mining is not viable as a primary.
2. **Unweighted BCE on the natural distribution is nearly calibrated out of
   the box.** Arm A *before* any Platt fit: ECE 1.8e-04, DR-ECE 0.0054, MCE
   0.031; Platt buys a further 3.7× only. Direct empirical support for spec
   §9.2's decision not to reweight.
3. **4-param Platt is not worth taking here.** It gains arm A nothing
   (0.00153 vs 0.00146) and arm B 6% — but gains arm C 2.3×, itself evidence
   for the covariate-bias account, since the 4-param form has occupancy-
   dependent slope/intercept to spend and C is the arm with covariate bias.
   Spec §10.3 locks 4-param as primary; the data prefers 2-param for the arms
   we would actually deploy. `recommend_4param_platt` is false for all three.

### Known limitations

- **Arm A is not demonstrably converged.** It ran all 50 epochs without
  early stopping firing (`stopped_epoch == max_epochs == 50`), so its numbers
  may be pessimistic and "A is the ceiling" is a weaker claim than it looks.
- **Raw `val BCE` is not comparable across arms** (A 0.0089, B 0.0399, C
  0.2861). B and C carry deliberate distribution-shift bias in their logits
  which Platt removes; BCE penalises them for something the deployed model
  does not do. Compare AUC (monotone-invariant) and post-Platt ECE only.
- **The AUC-PR red flag (<0.80) was miscalibrated and is retired.** It was
  set before the real base rate was known. Arm A — no subsampling, no shift —
  also lands below it at 0.7987, so the threshold, not the model, was wrong.
  At a 0.57% base rate a no-skill classifier scores 0.0057.
- **MCE is reported but not gated.** Arm C sets every pass-criterion true
  while carrying a ≥100-row bin (`MIN_BIN_COUNT`, so not sampling noise) that
  is maximally miscalibrated. The pass logic was deliberately left unchanged
  mid-ablation (one variable per experiment); it should include MCE before
  the ablation table is final.

## 2026-08-17 — Incident: `is_ckf_selected` patch wrote an all-False column

The first real patch run (events 0, 1) reported `frac_states_matched = 0.0`,
`n_selected = 0`, and **exited 0**, writing two Parquets whose
`is_ckf_selected` is uniformly False.

**Two independent defects.**

1. **The guard was off the executed path.** The `frac_states_matched >= 0.95`
   floor lived in `scripts/patch_is_selected.py::main()`, but
   `modal_train.patch_selected_all` imports `patch_event` directly, so the
   check never ran at scale — and `patch_event` had no tests of its own. A
   residual-join failure is invisible in the output schema (the column exists
   with the right name and dtype and is merely uniformly wrong), so a loud
   refusal is the only available detection mechanism. Fixed: the floor is
   enforced inside `patch_event`, *before* the write, covered by
   `tests/test_patch_event_write_guard.py`.
2. **The documented join key is wrong.** This log records
   `(seed_id, step_k) = (track_nr, state_idx)`. `seed_id` aligns exactly
   (both span [0, 979275]) but `step_k` does not — the expanded Parquet
   reaches 38, the ROOT trackstates only 17. Only 574,910 / 5,671,978 ROOT
   states (10.1%) share a key at all, and on the keys that *do* overlap the
   Parquet says hole (`cand_hit_id = -1`, NaN residual) where ROOT reports a
   finite measurement. Ruled out: **units** (median |residual| 3.63 mm
   Parquet vs 1.55 mm ROOT — same scale) and **sign convention**
   (opposite-sign agreement is better, 0.47% vs 0.009% within 1e-4, but
   nowhere near enough to explain total failure).

**Status:** value-function pipeline blocked pending resolution of what
`step_k` means on each side. Diagnose with
`modal run modal_train.py::diagnose_join --event-id 0` (read-only, never
commits). Do not "fix" this by widening `DEFAULT_TOL` or flipping a sign —
that manufactures agreement without establishing that the rows correspond.

**Cleanup still required:** delete
`results/train32/selected/expanded_event{000000000,000000001}.parquet`.
`patch_selected_all` skips events whose output already exists and its
integrity check validates only the Parquet footer, so it would silently
accept these two.

**Gate results are unaffected** — the gate cache reads `EXPANDED_DIR` and
never needs `is_ckf_selected`.
