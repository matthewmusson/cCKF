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

### Stratified reliability (F7 occupancy, F8 |η|) — added 2026-08-17

Strata are fixed and physical (`metrics.occupancy_strata`,
`metrics.abs_eta_strata`), computed once from val aux so every estimator sees
identically the same rows per stratum. Val-split row counts:

| occupancy | rows | | |η| | rows |
|---|---|---|---|---|
| n=1 (isolated) | 177,158 | | [0.0,0.5) | 644,469 |
| n=2-3 | 564,518 | | [0.5,1.0) | 1,017,635 |
| n=4-6 | 1,079,096 | | [1.0,1.5) | 4,568,547 |
| n=7+ | 22,287,890 | | [1.5,2.0) | 14,092,842 |
| | | | [2.0,2.5) | 2,690,558 |
| | | | [2.5,3.0) | 1,091,957 |
| | | | [3.0,4.0) | 2,654 |

**Arms A and B are stratum-independent.** All four occupancy curves and all
seven |η| curves lie together on the diagonal (A) or just above it (B). The
gate's probability means the same thing in a quiet barrel region as in a dense
forward one — which is the property the χ² gate lacks and the project's reason
for existing.

**Arm C fans apart across the full range of the axis**, in both occupancy and
|η|. At `n=1` the observed fraction reaches ~1.0 while predicted is 0.2–0.6; at
`n=7+` it runs flat near zero. This is the covariate-dependent-bias account made
visible: weighting negatives by 1/χ² induces a logit bias of `log(w(x)/E[w])`
which varies with χ², and χ² correlates with occupancy — so the miscalibration is
occupancy-dependent, and a 2-param Platt (one global slope, one global intercept)
has no functional form that could fix it.

**This also explains why arm C's aggregate DR-ECE (2.48e-02) looked merely
mediocre**: `n=7+` holds 22.3M of 24.1M val rows (92%), so the aggregate is
dominated by a single stratum and averages the fan away. Another instance of the
same lesson as the ECE/DR-ECE gap — aggregates hide localised failure.

**Platt-4 does not fix arm C's stratum dependence.** Generated for comparison
(`figures/gate_platt4/`): `n=1` is somewhat better behaved at low predicted
values but `n=7+` is still flat near zero, `n=4-6` still collapses, and the fan
is essentially as wide. So the possibility that F7 would justify the 4-param's
extra parameters is **not** borne out, and the audit's
`recommend_4param_platt: false` stands. Consistent with the mechanism: the bias
depends on χ² directly, while the 4-param's slope and intercept are linear in
`log n_window` only — occupancy correlates with χ² but does not substitute for
it. Note this coexists with Platt-4 improving arm C's *aggregate* DR-ECE
(2.48e-02 → 1.50e-02) and AUC-PR (0.2922 → 0.3959): it absorbs some
occupancy-correlated bias, improving the average, without making the strata
agree. Aggregate improvement and stratum-independence are different properties,
and only the second one means the calibrator is working.

**Bug fixed en route:** `metrics.wilson_ci` could return an interval excluding
its own point estimate by ~4e-19 at `k=0` (exact arithmetic gives the lower
bound as 0; float gives 4.3e-19 at n=500). Harmless in a JSON dump, fatal to
`matplotlib.errorbar`, which rejects the negative bar length. Now clamped to
bracket `k/n`, with parametrized regression tests.

## 2026-08-17 — Stratified metrics (F9), base rates, and two metric-comparability findings

Adds the (calibrator × arm × stratification) metric tensor and, in the course of
reading it, two findings about the metrics themselves that change how the earlier
tables in this log should be read.

### The F9 tensor

12 figures, `figures/gate/F9_metrics_{gate_platt2,gate_platt4}_{A,B,C}_by_{occ,abseta}`
— one per cell of {Platt-2, Platt-4} × {A, B, C} × {occupancy, |η|}. Each shows
six metrics (AUC-ROC, AUC-PR, ECE, DR-ECE, DR-ECE/ECE, MCE) as bars over that
variable's strata, with the all-rows value as a dashed reference and the stratum
row count on the x-label.

Per-stratum AUC is computed on the actual rows, not rebuilt from the shipped
reliability histograms: ECE/DR-ECE/MCE are bin-based and recoverable, but AUC
depends on within-stratum *ranking*, which a 1000-bin histogram only
approximates. `curves.metric_bundle` now returns NaN AUCs on a single-class
sample rather than letting sklearn raise — a real condition for the 2,654-row
forward |η| stratum, not a defensive hypothetical.

### Base rates (val split) — a property of the data, model-independent

Verified identical across all 3 arms × 4 estimators × 11 strata.
LaTeX: `docs/tables/gate_base_rates.tex`.

| stratum | candidates | % rows | true hits | % pos | π | 1/π |
|---|---|---|---|---|---|---|
| n=1 (isolated) | 177,158 | 0.73% | 50,060 | **36.54%** | 28.2573% | 3.5 |
| n=2-3 | 564,518 | 2.34% | 21,883 | 15.97% | 3.8764% | 25.8 |
| n=4-6 | 1,079,096 | 4.48% | 12,164 | 8.88% | 1.1272% | 88.7 |
| n=7+ | 22,287,890 | 92.45% | 52,892 | **38.61%** | 0.2373% | 421.4 |
| \|η\|[0.0,0.5) | 644,469 | 2.67% | 21,396 | 15.62% | 3.3199% | 30.1 |
| \|η\|[0.5,1.0) | 1,017,635 | 4.22% | 22,056 | 16.10% | 2.1674% | 46.1 |
| \|η\|[1.0,1.5) | 4,568,547 | 18.95% | 31,018 | 22.64% | 0.6789% | 147.3 |
| \|η\|[1.5,2.0) | 14,092,842 | 58.46% | 27,168 | 19.83% | **0.1928%** | 518.7 |
| \|η\|[2.0,2.5) | 2,690,558 | 11.16% | 16,949 | 12.37% | 0.6299% | 158.7 |
| \|η\|[2.5,3.0) | 1,091,957 | 4.53% | 18,171 | 13.26% | 1.6641% | 60.1 |
| \|η\|[3.0,4.0) | 2,654 | 0.01% | 241 | 0.18% | **9.0806%** | 11.0 |
| all | 24,108,662 | 100% | 136,999 | 100% | 0.5683% | 176.0 |

Three facts worth carrying:

1. **Over a third of all true hits sit in isolated windows.** n=1 is 0.73% of
   candidates but 36.5% of positives; n=7+ is 92.5% of candidates and 38.6% of
   positives. The positives split roughly evenly between the trivially-easy and
   the hardest stratum despite a 126× difference in candidate count. A large
   share of the gate's correct decisions are on windows with no ambiguity.
2. **π is non-monotone in |η|**, minimum 0.193% at [1.5,2.0) — the stratum
   holding 58% of all candidates — rising again forward to 9.08%. Plausibly the
   barrel/endcap transition, *not verified against ODD geometry.*
3. **π is set mechanically by occupancy**: one candidate in the window is
   usually the right one; among seven-plus at most one can be.

### Finding: raw ECE is not comparable across strata, and its ordering is an artifact

Arm A + Platt-2, occupancy strata:

| stratum | π | ECE | **ECE/π** | DR-ECE | AUC-PR |
|---|---|---|---|---|---|
| n=1 | 28.26% | 8.78e-03 | **0.031** | 1.96e-02 | 0.974 |
| n=2-3 | 3.88% | 2.60e-03 | 0.067 | 1.40e-02 | 0.827 |
| n=4-6 | 1.13% | 1.19e-03 | **0.106** | 1.11e-02 | 0.647 |
| n=7+ | 0.237% | 2.30e-04 | 0.097 | 7.57e-03 | 0.584 |
| *spread* | *119×* | *38×* | ***3.4×*** | ***2.6×*** | |

ECE tracks π, not calibration quality: π spans 119× and ECE spans 38×.
**Normalising by π collapses the spread to 3.4× and reverses the ordering** —
n=1 becomes the best-calibrated stratum and n=4-6 the worst. Where nearly all
predictions and labels are ~0, |obs − pred| is small by construction; there is
less error available to make.

Consequences:

- **Never compare raw ECE across strata.** Use DR-ECE (2.6× spread, tightest of
  the four) — restricting to p ∈ [0.01, 0.99] removes the easy-negative bulk
  whose size π controls. This is a second, independent argument for DR-ECE
  beyond catching the decision region: it is the *comparable* quantity.
- The same effect explains two earlier puzzles. |η|[3.0,4.0) has the best
  AUC-PR in the table (0.9808) because it has the **highest** base rate
  (9.08%, 1 in 11) — easiest task, not best model. And n=7+ has the lowest ECE
  because it has the lowest π.

### Finding: MCE and ECE do not share a bin set, so MCE ≥ ECE can invert

`expected_calibration_error` weights **every** bin, by design, so it stays an
average over the whole sample. `max_calibration_error` drops bins under
`MIN_BIN_COUNT = 100`. When all bins are populated the familiar ordering holds.
When most are sparse the two are computed over *different* bin sets.

Measured on |η|[3.0,4.0): 2,654 rows, **88.7% below p = 0.01**. Only the p≈0
bulk bin clears 100 rows, so MCE reported that single trivially-calibrated bin
(**2.13e-06**) while DR-ECE — which filters on region total, not per bin —
reported **1.31e-01** from the 299 rows spanning the decision region, the worst
value in the table. Five orders of magnitude apart, from disjoint subsets of the
same stratum.

`metrics.max_calibration_error_detail` (a896a85) now returns `mce` alongside
`n_bins_eligible`, `n_bins_total`, `n_rows_eligible`, `frac_rows_eligible` and
`argmax_bin`; MCE is NaN rather than 0.0 when nothing is eligible.
`max_calibration_error` keeps its signature and value, so no logged number moves.

**This also reframes the "MCE should be gated" note above.** The spec contains
**zero** mentions of MCE — its calibration targets (§4.2: ECE < 0.02 overall,
< 0.05 per stratum) are exactly the three that *are* gated. MCE is an addition
of ours to close ECE's count-weighting blind spot, so gating it is a §4.2
amendment, not a code fix. Two obstacles to doing it now: the natural threshold
is not ECE's 0.05 (MCE is a max, systematically larger — arm A's *primary*
config is 0.0495 aggregate and 0.122 in n=4-6, so a 0.05 gate would fail our
best result), and any gate needs `n_bins_eligible` as a precondition or thin
strata would report reassuring values while being the worst calibrated.
**Recommendation: do not gate MCE yet**; pick a threshold from arm A's measured
per-stratum distribution and propose it with evidence.

### Also in this batch

- **Occupancy affects discrimination, not calibration, and |η| affects neither
  much.** Arm A + Platt-2: AUC-ROC is flat across occupancy (0.980–0.991) while
  AUC-PR collapses 0.974 → 0.584. Across |η|, AUC-ROC sits in 0.981–0.992 with
  no trend while AUC-PR falls 0.899 → 0.681. |η| is partly a proxy for
  occupancy rather than an independent axis.
- **|η|[1.5,2.0) carries a 56.3× ECE inflation**, the largest in the table, and
  holds 58% of all rows — most of what drags the aggregate ECE to 3.87e-04.
- LaTeX tables added: `docs/tables/gate_ece_vs_dr_ece.tex`,
  `docs/tables/gate_A_platt2_stratified.tex`, `docs/tables/gate_base_rates.tex`.

## 2026-08-17 — Sampler corrected to draw with replacement; arms B and C re-running

`cckf/samplers.py` (dc8a482) now draws negatives **with replacement** in both
subsampling arms.

**Why it is a correctness fix, not a preference.** Weighted sampling *without*
replacement does not give inclusion probabilities proportional to the weights.
For i.i.d. draws with replacement `P(draw = i) = w_i / Σw` exactly; without
replacement, item i's marginal inclusion probability after n draws is a function
of *all* the weights that saturates toward 1 for heavy items. **Arm C therefore
never realised the intended ∝1/χ² distribution at all.**

It also makes the bias analysis exact rather than approximate. Reweighting
negatives by w(x) shifts the Bayes-optimal logit by `log(w(x)/E_p[w])`; that
identity holds for i.i.d. draws from `q(x) ∝ w(x)p(x)`, which is what
with-replacement sampling produces. The stratified reliability diagrams and the
fitted Platt intercept both test that identity, so the sampler must implement it.

Arm B switched too. Uniform sampling does not need replacement — the marginal is
uniform either way — but B and C must share the scheme or the S1 ablation
differs in two variables instead of one.

**A latent flaw was deleted with the shortfall branch.** When fewer hard
negatives existed than `n_take`, the old code took every hard negative and
backfilled uniformly from the *zero-weight* pool — silently mixing two sampling
distributions into one training set. With replacement, drawing `n_take` items
needs only one nonzero-probability entry, and a zero-weight negative is
unreachable by construction. Three tests that had pinned the old semantics were
replaced, and one added that nothing previously checked: realised draw
*frequencies* track 1/χ² in the expected 4:1 ratio.

Weights remain linearly normalised (`w/Σw`), never softmax: 1/χ² is already
non-negative so a sum suffices, and `exp(1/χ²)` would both change the
distribution and overflow, since `_CHI2_FLOOR` lets 1/χ² reach 1000.

**Status: arms B and C are re-training under the corrected sampler**, then
re-auditing and re-exporting. **Arm A is unaffected** (`picked = row_idx`, no
subsampling), so the headline result — DR-ECE 3.39e-03 raw, 85× better than χ²
on the cal split, and figure G3 — stands unchanged. Every B and C number in the
two preceding sections is from the pre-replacement sampler and will move.

**Not changed, deliberately:** the negative subsample is still drawn *once*
before training and reused every epoch (`scripts/train_gate.py`), so arm C's
model sees the same ~4.29M negatives for all its epochs. Per-epoch resampling
would cover far more of the 173M negative pool at the same per-epoch cost. Ruled
out of scope for this re-run by explicit decision, to keep one variable changed.

## Value function V_φ — blocker status as of 2026-08-17

**Not started. Blocked on a data join, not on the model.** Full diagnosis in the
`is_ckf_selected` incident entry above; this is the standing summary.

**What is needed.** `V^{π†}` labels require knowing which candidate the CKF
actually *accepted* at each step, to replay the truth-greedy rollout. That is
the `is_ckf_selected` column, produced by joining the expanded Parquet against
the Stage 1 `trackstates_ckf.root`.

**Why it fails.** The join key recorded in this log,
`(seed_id, step_k) = (track_nr, state_idx)`, is wrong on `step_k`. Measured on
event 0: `seed_id` aligns exactly (both span [0, 979275]) but `step_k` does not —
Parquet reaches 38, ROOT only 17. Only 574,910 of 5,671,978 ROOT states (10.1%)
share a key, and on the keys that *do* overlap the files contradict each other:
Parquet says hole (`cand_hit_id = -1`, NaN residual) where ROOT reports a finite
measurement. Result: **0 matches of 5.67M**.

**Ruled out — do not re-derive.** Units (median |residual| 3.63 mm Parquet vs
1.55 mm ROOT, same scale) and sign convention (opposite-sign agreement is
better, 0.47% vs 0.009% within 1e-4, but nowhere near enough to explain total
failure).

**Remaining hypothesis.** The two files count steps differently — ACTS may skip
states when writing the tree, or `expansion.py` may enumerate branch-steps ACTS
does not. Resolving it means reading `expansion.py`'s `step_k` construction
against the ACTS `RootTrackStatesWriter`.

**Do NOT force a match** by widening `DEFAULT_TOL` (1e-4) or flipping a residual
sign. That manufactures agreement without establishing the rows correspond, and
this column feeds the training target directly, so a wrong join silently trains
V_φ on false labels.

**Diagnostic:** `modal run modal_train.py::diagnose_join --event-id 0` —
read-only, never commits.

**Cleanup owed:** delete
`results/train32/selected/expanded_event{000000000,000000001}.parquet` (all-False
`is_ckf_selected` from the failed run). `patch_selected_all` skips events whose
output exists and its integrity check validates only the Parquet footer, so it
would silently accept them. **Awaiting explicit confirmation before deleting.**

**Also ready and untested against real data**, once the join is fixed:
`train_value.py` streaming path (67c1d1d), `cckf/stage1_map.py` provenance map
(82ade1e), value-cache staging with `--only-events` (00cfb35), and the
`patch_event` write guard (9385520).

---

## 2026-08-18 — Value function V_φ training complete

**Status:** Complete. Blocker resolved.

The join-key blocker from Aug 17 was fixed by switching from `(seed_id,
measurement_id)` to `(seed_id, geometry_id)`. Once the join succeeded,
`patch_selected_all` ran on all 32 events, and V_φ training proceeded
normally. Architecture: 11→128→128→1 SiLU, 18,177 params. Target:
V^{π†} = min(completeness, purity) as a soft label ∈ [0, 1], trained
with BCE. Identity Platt calibrator (raw sigmoid best calibrated).

---

## 2026-08-19 — ACTS C++ integration complete

**Status:** Complete. Build passing and stable.

All 10 SDD tasks implemented: hand-written MLP kernel, weight blob
loader, gate measurement selector, value branch stopper, algorithm glue,
sensor lookup table, instrumentation patch extension, Python bindings,
per-event timing CSV, and Pareto sweep harness.

6 build-fix commits post-SDD (726a7a5..c39d150):
- Missing GainMatrixUpdater include
- Constructor param shadowing logger()
- GeometryContext private default constructor
- `.template segment<3>()` in template context
- sizeof incomplete type for SensorLookup
- PRIVATE→PUBLIC include dirs for Python bindings

Gate and value weights exported to CCKF binary blobs on Modal volume.

---

## 2026-08-19/20 — First cCKF inference runs (real weights)

**Status:** In progress. Runs 5 and 6 active.

First real-weights inference on Modal using tight_t79 MOTPE-optimized
seeding (137,756 seeds per event at μ=200). Single validation event.

### Key findings so far

1. **Gate dominates wall time.** 57M gate calls per event, ~31 μs each
   → 88% of CKF wall time. This is expected: μ=200 pileup means many
   candidate hits per surface.

2. **tight_t79 track selection was optimized for standard CKF, not
   cCKF.** The learned gate rejects hits at many layers, creating holes.
   The tight_t79 parameters (nMeasMin=9, maxHolesAndOutliers=1) were too
   aggressive — every track found by the CKF failed selection.

3. **Track selection ≠ CKF branch management.** `tracks.size()` in the
   timing CSV counts *selected* tracks (post-selection in `addTrack`),
   not *found* tracks. `m_nFoundTracks` in the finalize log counts all
   tracks entering `addTrack`.

### Run log

| Run | Gate | Value | nMeasMin | maxHoles | ptMin | Ambi | Result |
|-----|------|-------|----------|----------|-------|------|--------|
| 1   | 0.5  | 0.2   | 9        | 1        | 0.46  | yes  | 249,675 found, 0 selected. nMeasMin=9 too strict for gate-induced holes. |
| 2   | 0.3  | 0.15  | 7        | 1        | 0.46  | yes  | **Timed out** after 1h (3600s). Gate threshold 0.3 → massive branching (~10 seeds/sec vs ~5000/sec at gate=0.5). Seed 35472 alone produced 159+ tracks. |
| 3   | 0.5  | 0.2   | 7        | 1        | 0.46  | yes  | 249,675 found, 0 selected. Root cause: maxHoles=1 killed every track (gate creates holes at many layers). |
| 4   | 0.5  | 0.2   | 5        | 15       | 0.46  | yes  | Cancelled by Modal at ~50% (likely resource conflict with timed-out Run 2). |
| 5   | 0.4  | 0.2   | 5        | 15       | 0.46  | yes  | 278K found, 12,631 selected. **ε=0.83%, f=30.0%.** Near-zero efficiency despite many selected tracks. |
| 6   | 0.5  | 0.2   | disabled | disabled | off   | no   | 237K found, 235,996 selected. **ε=0.013%, f=7.0%.** All selection disabled; raw track quality is terrible. SIGABRT on exit (memory corruption in cleanup, after results written). |

### Parameter catalog

All CKF parameters that interact with gate/value decisions:

**Gate parameters:**
- `cckf_gate_threshold` (τ_g): P(same particle) cutoff. Lower → more candidates, more branching, much slower.
- `cckf_gate_max_candidates`: Max candidates kept per surface (sorted by gate score). Default 10.
- `ckf_chi2CutOffMeasurement`: χ² cutoff for window search. Gate operates on hits within this window.
- `chi2OutlierCutoff` (hardcoded 100.0): When NO candidate passes gate, keeps min-χ² hit as outlier if χ² < 100.

**Value parameters:**
- `cckf_value_threshold` (τ_v): P(branch completes) cutoff. Lower → more branches survive.
- `minMeasurementsBeforePrune` (hardcoded 3): Don't evaluate value until ≥3 measurements.
- `minMeasurementsForKeep` (hardcoded 6): Stopped branches with <6 measurements → StopAndDrop.

**Track selection (post-CKF filtering in `addTrack`):**
- `ckf_nMeasurementsMin`: Min measurements for a track to pass selection.
- `ckf_maxHolesAndOutliers`: Max holes + outliers allowed.
- `ckf_ptMin`: Minimum pT.
- `ckf_absEtaMax`: Pseudorapidity cut (not currently set).

**CKF branch management:**
- `maxPixelHoles`, `maxStripHoles` (not currently set — defaults in ACTS)
- `twoWay`: Two-pass CKF (forward then backward). Enabled.
- `trimTracks`: Trim final track. Default.
- `seedDeduplication`: Remove duplicate seeds. Enabled.
- `stayOnSeed`: CKF stays on seed measurements. Enabled.

### Root cause investigation (Aug 19-20)

Runs 5/6 find 200K+ tracks but almost none DM-match truth particles (ε ≈ 0%).
Baseline standard CKF achieves 37.3%. Investigation findings:

**Verified correct:**
- Feature order (26-dim) matches between C++ and Python
- Cholesky, kappa_u/v, q_tilde math matches Python exactly
- SensorLookup loads valid data (9 ODD volumes, real pitch/thickness values)
- Weight blob format matches between export_weights.py and WeightBlob.hpp
- MLP forward pass and Platt calibration formulas are correct
- Standardization pipeline: compute_norm_stats correctly sets mean=0,std=1 for
  NO_STANDARDIZE features (n_hits, n_holes, n_seq_holes at indices 23-25)

**Confirmed bug:** `pathInX0 = 0.0` hardcoded at CckfTrackFindingAlgorithm.cpp:223.
Feature 19 (pathInX0_interval) is always zero in C++ but has real values in
training data. Fix: read pathInX0_interval from most recent measurement track
state's dynamic column (written by instrumentation.patch).

**DAgger distribution shift:** Gate was trained on standard CKF track states but
deployed with its own decisions feeding back into predicted state. Expected to
cause some degradation but not 0% efficiency.

**Missing diagnostic:** Chi² of gate-accepted hits was not logged. If the gate
accepts hits with extremely large chi² (e.g. >100), it's essentially accepting
random noise hits from other particles. This would explain why tracks have
measurements from many different particles and can't be DM-matched.

### Diagnostic changes (Aug 20)

Added to C++ code for next rebuild:
1. **Chi² + gate score diagnostics** — CckfTimers::GateDiagnostics accumulates
   per-event: n_accepted, n_rejected, n_outlier_fallback, sum/max chi2 for
   accepted, mean gate score for accepted vs rejected, n_nan_features
2. **Timing CSV extended** — 9 new columns: gate_n_accepted, gate_n_rejected,
   gate_n_outlier_fallback, gate_sum/max_chi2_accepted, gate_sum_chi2_rejected,
   gate_mean_score_accepted/rejected, gate_n_nan_features
3. **Standardization logging at startup** — dumps Platt params and std[23-25]
   (the NO_STANDARDIZE features) to verify mean=0, std=1 at runtime
4. **pathInX0 fix** — reads pathInX0_interval from most recent measurement
   track state (previous interval's X/X0, reasonable proxy)

---

## 2026-08-21 — Training-data purity audit (all 32 events)

**Status:** Complete. Two defects found, both confined to events 0-3.
**Git hash:** fce31a3
**Tooling:** `sweep_parquet_purity.py` (read-only Modal job, no volume commits).

### Why this ran

Chasing the nSigma window mismatch (below), the expanded Parquets turned out to
contain rows with `S11 = NaN` on sensors that digitise 2D. `expand_trackstates`
filters `S11.notna()`, so those rows should not exist. That prompted a full
audit of all 32 events: 259,893,979 trainable gate rows
(`gate_row_mask` per `cckf/labels.py`).

### Result — two disjoint populations

| group | events | S11 lost | long-strip | pos rate | n_window |
|-------|--------|----------|------------|----------|----------|
| A | 0, 1, 2, 3 | 34.3-35.0% | 9.6-9.9% | 3.09-3.45% | ~16 |
| B | 4-31 (28 events) | 0% | 0% | 0.29-0.85% | 19-31 |

Totals across all 32 events:

| stratum | rows | share | pos rate |
|---------|------|-------|----------|
| clean | 244,894,282 | 94.23% | 0.553% |
| corrupted (2D sensor, S11 lost) | 11,699,732 | 4.50% | 0.902% |
| genuine 1D long-strip (vol 28/29/30) | 3,299,965 | 1.27% | 20.383% |

**Defect 1 — S11 lost on 2D sensors (events 0-3 only).** Volumes 16/17/18 and
23/24/25 all digitise `(0, 1)` per `configs/odd-digi-geometric-config.json`, so
they have a real l1 coordinate. On 34-35% of events 0-3's trainable rows,
`S11` and `residual_l1` are absent and `chi2_inc` was recomputed as the 1D form
`r0^2/S00` (verified: matches to rel diff 0.0). Gate features 1, 3, 4, 5
(`res1`, `chol_S_10`, `chol_S_11`, `chi2`) are therefore wrong on those rows.

**Defect 2 — long strips absent from 28 of 32 events.** Volumes 28/29/30
digitise `(0,)` and are genuinely 1D, so their `S11_prt` is NaN by design
(`instrumentation.patch` only fills s01/s11 when `calibratedSize() >= 2`).
`expand_trackstates`'s `S11.notna()` filter drops every one of them. Only
events 0-3 retain long-strip rows; they are 1.27% of the training set, and the
val and cal splits contain **none**. The gate meets volumes 28/29/30 at
inference with effectively no training signal and no calibration coverage.

### Split base rates

| split | events | rows | pos rate | Group A share |
|-------|--------|------|----------|---------------|
| train | 24 | 191,412,582 | 0.9565% | 17.71% |
| val | 4 | 24,108,662 | 0.5683% | 0% |
| cal | 4 | 44,372,735 | 0.3699% | 0% |

train/cal ratio **2.59x**; excluding events 0-3 from train it falls to 0.4729%,
a ratio of **1.28x**.

**Confound, stated explicitly:** within Group B the positive rate tracks
occupancy inversely as expected (event 7, n_window 31.2 -> 0.285%; event 24,
n_window 18.9 -> 0.852%). That is exactly what the occupancy-conditional Platt
fit (spec §10.3) exists to absorb, so the residual 1.28x is likely benign. What
is *not* explainable that way: Group A has **lower** occupancy (~16) yet a
**higher** positive rate (3.2%) — backwards from the trend, driven by
long-strip rows at 20.4% positive. No calibration event contains that
population, so the calibrator cannot absorb it.

### Clean results

- **Box criterion confirmed at full scale.** Zero violations of
  `|r0| <= 10*sqrt(S00)` and `|r1| <= 10*sqrt(S11)` across all 259.9M trainable
  rows. The n=10 axis-aligned box is definitively the selection criterion used
  to build the training set.
- **`vstar_soft` is a dead column** — 0 finite cells in all 32 events. It is
  never read: the value target is `vstar_t2` from `cckf/value_target.py`. Not a
  defect.
- **Value function is not exposed to defect 1.** `VALUE_SOURCE_COLUMNS` takes
  `sigma2_l0`/`sigma2_l1` from `cov_00`/`cov_06` (the bound track covariance C),
  not from the innovation S. The only indirect path is
  `sum_gate_logodds`/`min_gate_logodds`, which inherit gate outputs on events
  0-3 (17.7% of train rows).

### Provenance — unresolved

The current tree cannot produce either group. `expand_trackstates` has carried
the `S11.notna()` filter since the original commit 416c39b (verified with
`git log --all -S` across every ref), and `chi2_increment_batch` has no 1D path
— with `s11 = NaN` its fallback yields NaN, not `r0^2/S00`. The Group A/B break
falls exactly at the events 0-3 / 4-31 boundary, consistent with a code change
landing between Stage 1 batch 2 and batch 3 (`expand_all_events` runs events in
pairs). Settling it needs the Stage 3 patch-job source or the Aug 10-12 run
logs.

### Conclusion

Recommendation: **drop events 0-3 from the train split** (one change in
`cckf/splits.py`). Costs 17.7% of train rows, removes 100% of defect 1, and
brings train/cal to 1.28x. Re-expanding 0-3 would not help — the `S11.notna()`
filter would convert them to Group B. The proper fix (teach
`expand_trackstates` to handle 1D measurements, re-expand all 32, retrain)
restores long-strip coverage but costs a full re-expansion plus gate retrain.
Not yet actioned.

---

## 2026-08-24 — Gate rejects 100% of long strips (offline scoring by volume)

**Status:** Confirmed defect. Root cause identified. Not yet fixed.

Scored archived Parquet rows with the **deployed** `gate.bin` blob
(WeightBlob/MlpInference reimplemented in numpy, so the diagnosis uses the
exact weights the C++ runs) and stratified by `volume_id`.

### Events 0-3 (the only events containing long strips)

| vol | type | rows | base pos% | gate acc% | purity | recall |
|-----|------|------|-----------|-----------|--------|--------|
| 16 | pixel | 284,017 | 4.59% | 0.77% | 64.3% | 10.8% |
| 17 | pixel | 615,804 | 0.48% | 1.69% | 10.6% | 37.4% |
| 18 | pixel | 6,601 | 4.05% | 0.44% | 37.9% | 4.1% |
| 23 | sstrip | 139,305 | 3.22% | 5.51% | 25.1% | 42.9% |
| 24 | sstrip | 156,937 | 0.90% | 1.94% | 13.7% | 29.6% |
| 25 | sstrip | 39,762 | 1.93% | 2.55% | 18.1% | 23.9% |
| **28** | **lstrip** | 72,820 | **18.91%** | **0.000%** | **0%** | **0%** |
| **29** | **lstrip** | 7,417 | **36.98%** | **0.000%** | **0%** | **0%** |
| **30** | **lstrip** | 28,754 | **18.33%** | **0.000%** | **0%** | **0%** |

### Val events 4/12/20/28 (Group B, clean, no long strips present)

| vol | type | rows | base pos% | gate acc% | purity | recall |
|-----|------|------|-----------|-----------|--------|--------|
| 17 | pixel | 1,256,081 | 0.581% | 0.391% | 89.5% | 60.3% |
| 24 | sstrip | 224,858 | 1.585% | 1.160% | 90.5% | 66.3% |

### Findings

1. **The gate accepts zero long-strip candidates**, on the surfaces with the
   *highest* base positive rate in the detector (18-37%). Mean predicted
   probability is 0.000.

   Mechanism: long strips are genuinely 1D, so `S11` is NaN by design
   (`instrumentation.patch` only fills s01/s11 when `calibratedSize() >= 2`).
   `build_gate_features` zero-fills non-finite values, so every long-strip row
   receives identical degenerate `res1`/`chol_S_10`/`chol_S_11` and a garbage
   chi2. The gate cannot separate them. With unweighted BCE and long strips at
   1.27% of the training set, rejecting all of them minimises the loss.

   Long strips are the outermost layers, so this removes the outer hits from
   every track.

2. **Clean vs corrupted data changes gate quality by ~8x.** Volume 17 purity is
   89.5% on clean val events and 10.6% on events 0-3 (34-35% corrupted S11).
   Long strips exist *only* in events 0-3, so the gate's only long-strip
   training data was also the corrupted subset.

3. **The deployed blob has an IDENTITY Platt calibrator**: `platt=(1.0, 0.0,
   0.0, 0.0)`, i.e. `calibrate(logit) = sigmoid(logit)`. Spec §10.3 requires
   the 4-parameter occupancy-conditional fit. Calibration is the headline
   result; the deployed weights are uncalibrated.

4. **Separate inference-side bug.** Offline, strips score *better* than pixels
   (90.5% vs 89.5% purity). In ACTS the traced run accepted 22.6% on pixels and
   0.5% on strips, a 45x inversion. Offline and inference disagree, so there is
   a train/inference feature mismatch on top of the training defect.

### Implied fix order

1. Make 1D measurements representable: stop zero-filling NaN `S11`; add an
   explicit `is_1d` feature and a 1D-appropriate chi2, so the gate can
   distinguish long-strip candidates at all.
2. Fix `expand_trackstates`' `S11.notna()` filter and re-expand so long strips
   exist in all 32 events rather than 4.
3. Retrain, dropping the corrupted events 0-3 or repairing them.
4. Export with the real occupancy-conditional Platt fit.
5. Separately, find the train/inference feature mismatch behind finding 4.
