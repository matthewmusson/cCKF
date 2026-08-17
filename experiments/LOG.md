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
- **AUC-PR came in under the spec's expected range.** Spec §9.5 expects
  0.85–0.95; arm A reaches 0.7987 and arm B 0.7710. This is the spec's number,
  not a threshold introduced here, so it is a missed expectation rather than a
  locally miscalibrated flag. It is not a debug-before-proceeding gate either —
  §9.5's only hard warning is AUC-ROC < 0.95, which every arm clears (A
  0.9931). Arm A carries no subsampling and no distribution shift, so the
  shortfall is a property of the feature set and architecture rather than of
  the sampling strategy. For scale, a no-skill classifier scores 0.0057 at
  this base rate, so 0.7987 is ~140× baseline.
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

## 2026-08-17 — Gate figure set, widened decision region, val-split audit

Figures for the S1 ablation, plus two methodology changes. **No retraining** —
inference only, from the frozen `gate_{A,B,C}/gate_model.pt` checkpoints.

- **Decision region widened** to `[0.01, 0.99]` (`metrics.DECISION_REGION`),
  symmetric in log-odds. Rationale: spec §10.1 consumes the gate as a branch
  score `sum_k log(g/(1-g))`, which integrates over both halves of the axis, so
  auditing only `[0.01, 0.5]` covered exactly the negative half. The old range
  is retained as `metrics.THRESHOLD_REGION` and still reported.
  **Provenance correction:** the plan attributed `[0.01, 0.5]` to a "spec §2.8 τ
  sweep range". No such section or range exists in the spec — τ there denotes
  the partial track state `τ_{0:k}` and the gate threshold is `g_min`. That
  bound was introduced by the plan, not locked by the spec.
- **Audited on the val split**, not cal. Platt is still fitted on cal only, so
  the fit and its evaluation are now on disjoint data, and the "before" AUCs are
  directly comparable to the training-time numbers.
- **Artifacts:** `results/curves/gate_{A,B,C}.{npz,json}` (~250 KB per arm),
  `figures/gate/F{1..6}*.{png,pdf}`.

### Results (val split, 24,108,662 rows, 0.5683% positive)

| arm | estimator | AUC-ROC | AUC-PR | ECE | DR-ECE [.01,.99] | ECE [.01,.5] | MCE |
|---|---|---|---|---|---|---|---|
| A | raw | 0.9931 | 0.7987 | 1.99e-04 | **3.39e-03** | 3.35e-03 | **0.0163** |
| A | + Platt-2 | 0.9931 | 0.7987 | 3.87e-04 | 9.89e-03 | 8.50e-03 | 0.0495 |
| A | + Platt-4 | 0.9931 | 0.7988 | 3.81e-04 | 9.75e-03 | 8.32e-03 | 0.0483 |
| B | raw | 0.9928 | 0.7710 | 2.37e-02 | 1.33e-01 | 8.42e-02 | 0.7218 |
| B | + Platt-2 | 0.9928 | 0.7710 | 4.65e-04 | 1.29e-02 | 1.17e-02 | 0.0708 |
| B | + Platt-4 | 0.9928 | 0.7709 | 4.52e-04 | 1.23e-02 | 1.09e-02 | 0.0705 |
| C | raw | 0.9671 | 0.2922 | 1.36e-01 | 2.89e-01 | 1.23e-01 | 0.9597 |
| C | + Platt-2 | 0.9671 | 0.2922 | 2.80e-03 | 2.48e-02 | 2.52e-02 | 0.9999 |
| C | + Platt-4 | 0.9598 | 0.3959 | 1.86e-03 | 1.50e-02 | 1.03e-02 | 0.9993 |
| — | χ²_λ | 0.8369 | 0.0315 | 5.50e-02 | 2.34e-01 | 1.22e-01 | 0.9692 |

Platt fits (on cal): A `a=0.9776 b=-0.1492`; B `a=0.9133 b=-3.7459`;
C `a=0.5789 b=-5.0436`. Final cal NLL, 2-param → 4-param: A 0.006395 →
0.006394; B 0.006847 → 0.006845; C 0.016871 → 0.014682.

### Key findings

1. **Arm A's raw gate is the best-calibrated estimator in the set** — DR-ECE
   3.39e-03 and MCE 0.0163 with *no calibrator at all*, 69× better than χ² on
   DR-ECE. Platt-2 makes arm A ~3× worse. The reliability curves show raw and
   Platt-2 essentially coincident along the diagonal, so this is a small
   localised difference, not a systematic shift. Strengthens finding 2 of the
   training entry: unweighted BCE on the natural distribution is self-calibrating,
   and for the deployed arm the Platt stage is optional at best.
2. **Platt is essential for B and useless for A**, which is exactly the
   distribution-shift account: B carries a +3.6988 logit shift for Platt to
   remove (DR-ECE 1.33e-01 → 1.29e-02, 10×), A carries none.
3. **χ²_λ is a p-value, not a posterior, and its reliability curve shows it.**
   `χ²_λ = exp(-χ²/2)` is exactly the χ²₂ survival function, i.e. P(residual
   this bad | hit is correct). The posterior needs the prior (0.57%) and the
   wrong-hit residual density, neither of which a p-value contains — and a valid
   p-value is Uniform(0,1) under H₀, so it *cannot* encode a base rate. Predicted
   consequence: flat near the base rate rather than diagonal. Confirmed across
   five decades in all three panels of F5, at AUC-ROC 0.8369 — a competent
   *ranker* being read as a probability.
4. **Arm C's reliability curve is non-monotone.** It rises above the diagonal,
   peaks near observed 0.70 at predicted ~0.35, then collapses toward p→1 (MCE
   0.9999). Hard-negative mining did not merely miscalibrate C, it *inverted*
   its confidence at the top end.
5. **Platt-4 measurably reorders arm C** (AUC-ROC 0.9671 → 0.9598, AUC-PR 0.2922
   → 0.3959, a 35% AUC-PR gain). This is *not* slope inversion — see below — but
   plain row-dependence: `a(x)`/`b(x)` vary with occupancy, so rows at different
   occupancy get different affine maps and legitimately swap order. Worth stating
   as a conceptual caveat on spec §10.3: an occupancy-conditional calibrator is
   not rank-preserving, so it is doing some *classification*, not only calibration.
6. **AUC invariance under 2-param Platt confirmed on real data.** Arm A raw and
   Platt-2 agree to 6 decimals on both AUCs (0.993063 / 0.798707). Residual
   differences of ~2e-6 (arm B's AUC-PR) come from float saturation creating
   ties at the extremes, not from the ordering changing; the invariance check
   tolerance is 1e-4 for that reason.

### Corrections to earlier claims in this log and session

- **Widening the region does not do the work claimed for it.** The wide/narrow
  DR-ECE ratio is 1.01–2.34 across estimators, and for arm C's calibrated gate
  it is **0.99** — no change at all. ECE is mass-weighted and almost no rows sit
  above p=0.5, so the upper half contributes negligibly however wrong it is. The
  conceptual argument for log-odds symmetry stands and the change is kept, but
  **MCE is what detects the top-end failure**, and MCE was always full-range.
  This reinforces rather than replaces the "MCE is not gated" gap.
- **A base-rate transport explanation for arm A's Platt degradation was
  proposed and does not hold.** cal is 0.3698% positive vs val 0.5683%, which
  predicts a −0.432 logit offset; the fitted intercept is −0.149, about a third
  of that, and the reliability curves show no uniform vertical offset. Direction
  right, magnitude over-predicted threefold. Likely because the gate's features
  already carry most of the occupancy/event information driving the base-rate
  difference, leaving little as a pure prior shift.

### The 4-param slope-inversion hazard: measured, and empty here

`a(x) = a0 + a1·log n_window` multiplies the logit, so `a1 < 0` makes `a(x)`
cross zero at `n_window = exp(-a0/a1)` and *invert* the model's ranking above
that occupancy. Arm C fits `a1 = -0.1408, a0 = 0.7007`, crossing at
**n_window = 144.95**.

Measured on all 24.1M val rows: **max n_window = 69**, `min_slope = 0.1045`,
**0 rows affected**. The hazard is real and does not trigger on this data, with
a 2.1× margin — thin enough that higher pileup or a looser window could cross
it. `calibration.platt_occupancy_slope_violations` now reports this every run.
