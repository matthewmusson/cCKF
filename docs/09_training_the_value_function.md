# 09 — Training the value function

The gate answers "is this hit mine?". The value function V_φ answers
"is this branch still worth propagating?", and replaces the CKF's hole and
branch caps. This document defines its target, shows the two ways of
computing it, and walks through training with `scripts/train_value.py`.

## The target: V^{π†}

A branch is graded by whether it will end as a double-majority matched
track (doc 04): purity ≥ 0.5 and completeness ≥ 0.5. At state k of a
branch, suppose a perfect decision-maker (π†, "pi dagger") took over from
here and picked the right hit at every remaining surface. The branch would
end with

```
n_shared = n_correct + n_findable          hits from the particle, past and future
n_track  = n_correct + n_wrong + n_findable
completeness = n_shared / N_total_true      N_total_true = the particle's simhits
purity       = n_shared / n_track
V^{π†}(k)    = min(completeness, purity)
```

`n_correct` and `n_wrong` count the hits the CKF accepted up to k
(`is_ckf_selected`, split by whether the branch's particle contributed);
`n_findable` counts the particle's hits still ahead. The `min` is the
point: a branch with twenty hits of which twelve are wrong has fine
completeness and hopeless purity; five perfect hits on a twenty-hit
particle is the reverse. The target stays soft in [0, 1]: 0.45 means
"nearly matchable", 0.05 means "hopeless", and a threshold would throw that
away. Locked in the specification (§11.1); the code is
`cckf/value_target.py::compute_value_targets` (Tier 1: the definition is
Matthew's, agents do not rewrite it).

On branch 128909 (doc 06) the past is always clean, so at every state
V = 1.0 until the last surface, where the wrong hit makes it 10/11 = 0.91.
A branch that took candidate 53229 at pixel layer 4 instead would have
n_wrong = 1 from there on; if it then found the seven remaining true hits,
V = min(9/10, 9/10) = 0.9; if the wrong hit steered it away and it found
none, V = min(2/10, 2/3) = 0.2.

## Two ways to count n_findable: tier 3 and tier 2

Everything above except `n_findable` is a fact about the log. How many of
the particle's remaining hits π† would find depends on where the branch
goes next, and there are two answers. The naming was locked on 2026-09-14
(`CLAUDE.md` has the mapping to the older code names):

**Tier 3, the logged-branch target.** Count the particle's simhits on the
surfaces the branch actually visited after k (`majority_true_hit_on_surface`
summed over later states). No re-propagation; computed entirely from the
parquet. This is `vstar_t2` in the code (`n_findable_t2`), the target the
deployed value function was trained on, and a lower bound: a branch that
went wrong at k visited the wrong surfaces afterwards, so its logged future
undercounts what π† would have reached. `vstar_t1` (`n_findable_t1`, every
remaining simhit of the particle wherever it is) is the matching upper
bound, kept as a diagnostic; `tier_invariant_violated` flags rows where the
bounds cross, which a correct expansion never produces.

**Tier 2, the re-propagated target.** Re-run the CKF from the last correct
state with a truth-greedy selector, and count what it finds. Three pieces:

1. `cckf/tier3_walker.py` classifies every state of every branch as
   `collapse` (the true hit was not in the window), `divergence` (it was,
   and the CKF took another), or `tip` (last state), and writes a worklist
   of states to roll out from: the filtered parameters and their variances
   at each anchor, with the majority particle id.
   `python cckf/tier3_walker.py --parquet P --trackstates-root R --event-id E --detectors-csv D --worklist-out W`
2. `TruthRolloutAlgorithm` (C++, `acts_patches/ActsExamples/TrackFinding/`)
   propagates from each worklist entry and at every surface accepts the
   majority particle's measurement if its χ² is inside `rollout_window_nsigma`
   (0 = unbounded). Enabled by `truth_rollout: true` in a config
   (`configs/_t3_full_ev1.yaml`); it reads the stage-1 CSVs directly and
   writes `event*-rollout-hits.csv`. `scripts/nersc/tier3_gen.sbatch` runs
   the walker and the rollout per event; `NSIG=3 sbatch scripts/tier3_rollout_n.sbatch`
   re-runs the rollout at a given window.
3. `cckf/tier3_stitch.py::compose_targets` joins rollout futures to past
   counts and N_total_true, giving `vstar_tier3` (code name) per
   (seed, step). `python scripts/stitch_tier3.py <event> <nsig> $SCRATCH/cckf`
   writes `tier2_targets/vstar_nsig{N}_event*.parquet`.

The window matters: with a 10σ window the rollout accepts almost anything
and the target is optimistic; with 3σ it is what a deployed 3σ CKF could
achieve. That is why the windowed cache carries `window_nsigma` as a 12th
feature (`VALUE_FEATURES_WINDOWED`) and why one model was meant to serve
several operating points.

**The audit that closes the loop.** On branches that never diverged (every
state `collapse` or `tip`), the logged future is exactly what π† would do,
so tier 2 and tier 3 must agree. `tier3_stitch.truth_suffix_check` compares
them; `stitch_tier3.py` arms it at nsig 10 and exits 1 above 1%
disagreement. On 2026-09-05 it failed at 44%, which is how the reversed
`step_k` (doc 05) was found: the walker had launched every rollout from
the seed end. Nothing tier 2 has been produced since the fix; the worklist
and hits of event 4 under `cckf_handoff/tier2_event4/` are the pre-fix
ones, kept as format examples.

## The cache and the training run

The value cache (doc 07) has one row per state: 11 features (12 windowed),
a float target, and `aux` = (`vstar_t1`, `step_k`, `eta`). Build it from
parquets that carry `is_ckf_selected`:

```
python scripts/build_value_cache.py --split train --parquet-dir $SCRATCH/cckf/reexpanded_sel \
    --csv-dir <stage-1 csvs> --out-dir $SCRATCH/cckf/vcache_v4/train            # tier 3 target
python scripts/build_value_cache.py --split train ... --out-dir $SCRATCH/cckf/vcache_v4w \
    --targets-dir $SCRATCH/cckf/tier2_targets --window-nsigma 3                 # tier 2 target, nested under nsig3/
```

`meta.json` reports the target's mean, the fraction in the marginal band
[0.2, 0.8], and how many rows the tier-invariant check dropped. Then:

```
python scripts/train_value.py --train-cache $SCRATCH/cckf/vcache_v4/train --val-cache .../val \
    --out-dir $SCRATCH/cckf/models_v4/value_t3_maj [--oversample-marginal 2] [--drop-features x0_accumulated]
```

(`scripts/nersc/train_value_v3.sbatch` is the GPU job.) The model is
`cckf.models.ValueMLP`: 11 → 128 → 128 → 1 with SiLU, 18,177 parameters,
a raw logit head. Loss is the same unweighted BCE as the gate, which with a
soft target in [0, 1] is the cross-entropy against a probability, so the
output is a probability estimate of "will this branch match". AdamW,
cosine schedule, batch 4096, early stopping on val BCE, patience 5. There
is no sampler flag; `--oversample-marginal N` repeats the marginal-band
rows N times so the region where the decision is close is not drowned by
the bimodal bulk. Pass `--train-cache` a parent holding `nsig*/`
subdirectories to train one model over several windows.

Outputs: `value_model.pt` (state dict, `feature_names`, `all_feature_names`,
`mu`, `sigma`), `value_metrics.json` (`train_bce`, `val_bce`, `val_mse`,
`auc_roc` against the target binarised at 0.5, `mean_target`,
`marginal_fraction`, `tier1_minus_tier2_mean`, `stopped_epoch`,
`red_flags`) and `value_val_predictions.npz` (`pred`, `target`, `aux`) for
plots. `scripts/eval_value_cal.py --model M --cal-cache C --out-dir O [--window-nsigma N]`
writes the reliability curve and the excess BCE over the entropy floor
(the part of the loss the model could still remove).

The promoted model (`cckf_handoff/models_v3/value_t2_maj`, trained
2026-08-26 on the tier-3 target with the **reversed** history): val AUC-ROC
0.822, AUC-PR 0.535, ECE 0.003. Those numbers are not a measure of the
target; they measure a model fitted to inverted past/future counts.

## Where the value function looks

The deployed model sees `sigma2_l0`, `sigma2_l1` (how uncertain the
prediction is), the three history counters, the accumulated and worst
log-odds of the accepted hits, `step_k`, η and q/p. Two of these carry
known train/inference mismatches that survive the 3 T and order fixes:

- `n_holes`, `n_seq_holes`: the expansion counts material surfaces as holes
  (doc 06); the C++ counts holes only. Reading `stateType` in the expansion
  removes it.
- `x0_accumulated`: 0 at inference (the C++ never fills it). Either
  implement it in `CckfBranchStopperWrapper` or train with
  `--drop-features x0_accumulated` (the exporter pads the column back with
  zero weight, doc 10) so the model cannot depend on it.
- `sigma2_l0`, `sigma2_l1`: **0 in training**. They are read from the
  parquet's `cov_00` and `cov_06`, and the expansion never fills any
  `cov_*` column (`expansion.SCHEMA_COLUMNS` lists them; nothing assigns
  them; every value is NaN, which the cache builder turns into 0). At
  inference the C++ fills them with the real predicted variances. Found
  2026-09-14. Until the expansion writes `err_eLOC0_prt²` and
  `err_eLOC1_prt²` into `cov_00` / `cov_06`, train with
  `--drop-features sigma2_l0,sigma2_l1` so training and inference agree.

So of the deployed model's 11 inputs, three were constant at training time
(`sigma2_l0`, `sigma2_l1`, `x0_accumulated`) and three were inverted
(`n_hits`, `n_holes`, `n_seq_holes`, plus `step_k`). The value function's
weak effect on the sweeps (doc 11) has more than one cause.

Deployment threshold: the C++ applies a sigmoid to the raw logit and stops
the branch when the probability is below `cckf_value_threshold` (τ_v); no
Platt fit is applied to the value function. `StopAndKeep` versus
`StopAndDrop` follows `minMeasurementsForKeep` (6, a C++ constant), and no
branch is judged before `minMeasurementsBeforePrune` (3).

## Tests

`tests/test_value_target.py` (hand-computed V at first and last step, the
min, the tier bounds, branch independence), `test_tier3_walker.py`,
`test_tier3_inputs.py`, `test_tier3_stitch.py` (a truth follower scores 1,
a wrong tip, the suffix check), `test_stitch_tier3.py`,
`test_build_value_cache.py`, `test_value_cache_windowed.py`,
`test_train_value_cache_loading.py`, `test_eval_value_cal_window.py`,
`test_value_window_monotonicity.py`.
