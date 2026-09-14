# 08 — Training the gate

The gate g_ψ replaces the χ² cut. Given the 26 features of doc 07 for one
candidate on one branch, it outputs a calibrated probability that the
candidate's cluster came from the branch's particle. This document walks
through the model, the loss, the imbalance strategy, the calibration, the
audit, and what the promoted gate says about the layer-4 decision of
branch 128909.

## The model

`cckf.models.GateMLP`: 26 → 128 → 128 → 128 → 1, SiLU activations, no
dropout, no normalisation layers, 36,609 parameters, and a **raw logit**
head. There is no sigmoid inside the model: the logit is what the loss,
the calibrator and the C++ consume, and a probability is only formed at
the very end. Depth and width are `--depth` and `--width` on the trainer;
the exporter asserts 128 × 3, so a different shape needs its assertion and
the C++ buffers changed together.

## The loss, and why it is not reweighted

`cckf.losses.bce_with_logits`: plain mean binary cross-entropy, no
`pos_weight`, even though positives are 0.5% of the rows. The reason
(spec §9.2) is that the gate must output a probability, not a score. A
positive weight w multiplies the positive term, and the minimiser of the
weighted loss is the true log-odds **shifted by log w**: the network then
learns a calibrated answer to a different question, and no monotone
recalibration undoes a shift that the occupancy-conditional Platt fit has
to fit on top of. The imbalance is handled by how batches are drawn.

## The three sampling arms

`cckf.samplers`, chosen with `--sampler`:

| Arm | What it does | Batch | Prior shift |
|---|---|---|---|
| **A** (primary) | every row once per epoch, shuffled, in large batches, so each batch still holds about 200 positives | 40,960 | 0 |
| B | keep all positives, draw `--neg-per-pos` (5) negatives per positive uniformly, with replacement | 4,096 | log(1/5) − log(natural odds) |
| C | as B, but negatives drawn with probability ∝ 1/max(χ², 10⁻³): hard negatives | 4,096 | as B, and the calibrator has to undo a non-uniform distortion |

The subsampled arms train faster and lose calibration; `prior_logit_shift`
is recorded in the checkpoint so it can be subtracted, and B and C exist
as ablations. Arm A was 85 times better calibrated than the χ² baseline on
the first data (log 2026-08-17) and is what `weights_v3` uses.

## Running it

```
python scripts/train_gate.py --train-cache $SCRATCH/cckf/caches_v4/train --val-cache $SCRATCH/cckf/caches_v4/val \
    --out-dir $SCRATCH/cckf/models_v4/gate_maj --sampler A --device cuda [--wandb-project cckf]
```

(`scripts/nersc/train_gate.sbatch <cache_parent> <out_dir> [arm]` on
`atlas_g`.) `cckf.train.train_model`: AdamW (lr 10⁻³, weight decay 10⁻²),
cosine schedule to 10⁻⁵ stepped per batch, gradient clipping at 1, up to
50 epochs, early stopping on validation BCE with patience 5, and the best
epoch's weights are returned, not the last. Validation is always on the
natural distribution. `torch.manual_seed(--seed)` and a re-initialisation
make two runs with the same flags identical. With `--wandb-project` set,
`train_loss`, `val_loss` and `lr` are logged per epoch.

The trainer refuses a cache marked `partial_split` unless
`--allow-partial-cache` is given, so a smoke cache can never become a real
model by accident. The two ablation flags, `--feature-groups` and
`--drop-features`, are in doc 07; `--drop-ambiguous` removes merged-cluster
rows (A8b).

Outputs in `--out-dir`:

- `gate_model.pt`: `state_dict`, `n_features`, `width`, `depth`,
  `feature_names`, `all_feature_names`, `mu`, `sigma` (the train split's
  standardisation, subset to the trained features), `prior_logit_shift`.
- `gate_metrics.json`: `train_bce`, `val_bce`, `auc_roc`, `auc_pr`,
  `stopped_epoch`, row counts, the per-epoch history, and `red_flags`
  (val BCE above 0.15, AUC-ROC below 0.95 or AUC-PR below 0.80 are
  reported as warnings).

The promoted gate (`cckf_handoff/models_v3/gate_maj`, 2026-08-26, arm A, 50
epochs, 243 M training rows): val BCE 0.0453, AUC-ROC 0.9887, AUC-PR
0.8524. The same recipe with the pure-seed cache (`gate_pure`): 0.9909 /
0.9602; cleaner labels, fewer rows.

## Calibration

The gate's raw logit is close to calibrated under arm A but not exactly,
and the miscalibration depends on occupancy: in a crowded window the same
evidence should mean less. The calibrator (spec §10.3) is a Platt fit with
occupancy-dependent slope and intercept,

```
p = σ(a(n)·z + b(n)),   a(n) = a0 + a1·log n_window,   b(n) = b0 + b1·log n_window
```

fitted on the **cal split only** (`cckf.calibration.fit_platt_occupancy`,
convex, L-BFGS-B on the exact negative log-likelihood). The four numbers
travel in the weight blob and the C++ applies the same formula with the
post-prefilter `n_window`. The calibration events are never used for
training or model selection; that is the whole reason they exist.

```
python scripts/calibrate_and_audit.py --model models_v4/gate_maj/gate_model.pt --cal-cache caches_v4/cal --out-dir models_v4/gate_maj/calibration
```

writes `platt_params.json` (`two_param` and `four_param`),
`calibration_audit.json` and the reliability figures. The audit compares
four estimators of P(same particle): the χ² likelihood ratio Λ = exp(−χ²/2)
(what the classical cut implicitly assumes), the raw gate, the two-parameter
Platt gate and the four-parameter one, each with its expected calibration
error overall, in the decision region, and in η and occupancy strata (a
stratum needs 1000 rows). Targets: ECE below 0.02 overall and 0.05 in every
stratum; `headline_beats_chi2` and `recommend_4param_platt` are the two
booleans to look at. Figure G3 (`figure_G3.png`) is the headline plot of the
project: reliability curves of the four estimators on one axis.
`scripts/eval_discrimination.py` gives AUC-ROC / AUC-PR on the val split
for any checkpoint; `scripts/plot_gate_figures.py` and
`scripts/export_gate_curves.py` produce the rest of the figure set.

## The layer-4 decision, scored

`scripts/diagnostics/score_branch.py` rebuilds the 26 features from the
parquet, standardises them with the checkpoint's `mu`/`sigma`, runs the
promoted gate and applies its four-parameter Platt fit, which is what the
C++ does at inference (doc 10):

```
python scripts/diagnostics/score_branch.py cckf_handoff/examples/envelope_event4/expanded_event000000004.parquet 128909 \
    --gate cckf_handoff/models_v3/gate_maj/gate_model.pt --platt cckf_handoff/models_v3/gate_maj/calibration/platt_params.json --step-k 19
```

The 18 candidates at pixel layer 4, module 361:

| cand_hit_id | χ² | cluster | label | logit | P(same particle) |
|---|---|---|---|---|---|
| 53230 | 2e−5 | 1×5 | 1 | +2.40 | **0.915** |
| 53229 | 10.2 | 1×2 | 0 | −11.40 | 1.0e−5 |
| 53232 | 17.1 | 2×1 | 0 | −12.34 | 4.0e−6 |
| 53228 | 34.3 | 1×3 | 0 | −11.08 | 1.4e−5 |
| … 13 more, χ² 46 to 125 | | | 0 | −11.6 to −15.0 | 3e−7 to 9e−6 |
| 53215 | 136.6 | 2×4 | 0 | −9.06 | 1.1e−4 |

Under the χ² cut (16.26), two candidates were compatible and the CKF
branched. Under the gate at τ_g = 0.64, one passes. The margin is not the
residual alone: the runner-up sits at χ² = 10, well inside the cut, and is
rejected by eleven logits. Note also the last row: the worst χ² in the
window gets the *second-highest* score, because its 2×4 cluster looks like
a real crossing; the gate has learnt that cluster shape is evidence,
which is exactly the point of the cluster features and exactly what a
χ² cut cannot use. And note the true hit's probability is 0.915, not
0.999: the branch history features it saw (`n_hits` 8, `n_holes` 11, the
reversed counts of doc 06) describe a branch that has been struggling,
and the calibrated gate discounts accordingly. After re-expansion those
read 2 and 1.

The standardised vector for the accepted candidate is printed by the same
command; the three history counters pass through unstandardised (8, 11, 1),
the rest sit within a unit or so of zero. That printout is the reference
for checking the C++ feature builder against the Python one
(`timing_traces.csv`, doc 10).

## Tests

`tests/test_models.py`, `test_losses.py` (no positive reweighting, soft
targets), `test_samplers.py` (arm coverage, with-replacement draws),
`test_train.py` (early stopping returns the best epoch, seeds, the lazy
standardised view and its column subset), `test_calibration.py` and
`test_calibration_trace.py` (the four-parameter fit recovers known
parameters and beats the two-parameter one), `test_metrics.py`,
`test_curves.py`, `test_train_gate_partial_cache.py`. A known gap:
`scripts/eval_discrimination.py` reads `loaded.X` from a dict and fails as
written; it needs `loaded["X"]`.
