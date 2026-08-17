# Gate Figure Set + Widened Decision Region — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the publication figure set for the gate S1 sampling ablation (loss curves, ROC/PR, calibration-fit convergence, reliability diagrams, before/after metric deltas) and widen the calibration decision region from `[0.01, 0.5]` to `[0.01, 0.99]`.

**Architecture:** No retraining. The three trained models are frozen artifacts at `/data/models/gate_{A,B,C}/gate_model.pt`, and per-epoch loss history is already in `gate_metrics.json`. One Modal function re-runs inference over the val and cal caches, fits Platt on cal with an NLL trace, applies it to val, and writes a **small** plot-data bundle (grid curves + scalar metrics, ~1 MB per arm) to the volume. Plotting then runs locally from those bundles — heavy compute stays where the data is, figure iteration stays fast.

**Tech Stack:** Python 3.10+, NumPy, PyTorch (inference only), scikit-learn (exact AUC), SciPy (L-BFGS-B Platt fit), matplotlib, Modal, pytest.

## Global Constraints

- Python 3.10+, type hints on all function signatures, NumPy-style docstrings, black formatting at 88 columns.
- **Events `[32, 64)` are sealed.** Never read them. `splits.assert_not_test` guards every event-resolving path.
- **Platt parameters are fitted on the calibration split only** (events 7, 15, 23, 31). Never on train or val.
- **Before/after metrics are evaluated on the val split** (events 4, 12, 20, 28), which never sees the Platt fit. This keeps the calibration fit and its evaluation on disjoint data, and makes the "before" AUCs directly comparable to the numbers already logged.
- **No retraining.** `gate_model.pt` files are read-only inputs. Any task that would retrain is out of scope.
- Reliability bin edges and ROC/PR threshold grids are **fixed constants, shared across all arms and estimators** — never data-derived. Quantile edges give each curve its own x-axis and make cross-arm comparison meaningless.
- Modal functions that write to the volume must **not** run concurrently with each other (`experiments/LOG.md`, 2026-08-12: concurrent `data_vol.commit()` corrupted 11 Parquets, and those containers were writing to disjoint paths).

---

### Task 1: Widen the decision region to [0.01, 0.99]

The current region `(0.01, 0.5)` audits only the negative half of the log-odds axis. Spec §10.1 defines the branch score as $\sum_k \log\frac{g}{1-g}$, so the gate output is consumed as accumulated log-odds and calibration error at high $g$ has a large effect on that sum. `[0.01, 0.99]` is the symmetric-in-log-odds region: $z \in [-4.595, +4.595]$.

The old region is retained as `THRESHOLD_REGION` and still reported, so the numbers already in `experiments/LOG.md` stay reproducible and comparable.

**Files:**
- Modify: `cckf/metrics.py:92`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `metrics.DECISION_REGION = (0.01, 0.99)`, `metrics.THRESHOLD_REGION = (0.01, 0.5)`. `metrics.decision_region_ece(pred, labels, region=..., n_bins=10) -> dict` keeps its existing signature and returns keys `ece`, `n_rows`, `positive_fraction`, `region`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metrics.py`:

```python
def test_decision_region_is_symmetric_in_log_odds():
    """The region must be symmetric in log-odds, not in probability.

    Spec §10.1's branch score sums log(g/(1-g)), so the audited region should
    cover equal magnitudes of that sum on both sides of zero. This test pins
    the *rationale*, so narrowing the upper bound back to 0.5 fails here
    rather than silently halving the audited range.
    """
    lo, hi = metrics.DECISION_REGION
    z_lo = np.log(lo / (1.0 - lo))
    z_hi = np.log(hi / (1.0 - hi))
    assert z_lo == pytest.approx(-z_hi)
    assert (lo, hi) == (0.01, 0.99)


def test_threshold_region_is_retained_for_continuity():
    assert metrics.THRESHOLD_REGION == (0.01, 0.5)


def test_wider_region_catches_high_confidence_miscalibration():
    """A confidently-wrong top-end bin is invisible to the narrow region.

    10,000 rows predicted 0.98 whose true rate is 0.5 (gap 0.48), plus
    10,000 rows predicted 0.02 whose true rate is 0.0 (gap 0.02). The narrow
    region sees only the second group; the wide region sees both, so its
    mass-weighted ECE is (0.48 + 0.02)/2 = 0.25.
    """
    rng = np.random.default_rng(0)
    pred = np.concatenate([np.full(10_000, 0.98), np.full(10_000, 0.02)])
    top_labels = rng.permutation(
        np.concatenate([np.ones(5_000), np.zeros(5_000)])
    ).astype(bool)
    labels = np.concatenate([top_labels, np.zeros(10_000, dtype=bool)])

    wide = metrics.decision_region_ece(pred, labels)
    narrow = metrics.decision_region_ece(
        pred, labels, region=metrics.THRESHOLD_REGION
    )

    assert wide["n_rows"] == 20_000
    assert narrow["n_rows"] == 10_000
    assert wide["ece"] == pytest.approx(0.25, abs=0.02)
    assert narrow["ece"] == pytest.approx(0.02, abs=0.01)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -k "region" -v`
Expected: FAIL — `test_decision_region_is_symmetric_in_log_odds` fails its `(0.01, 0.99)` assertion, and `test_threshold_region_is_retained_for_continuity` fails with `AttributeError: module 'cckf.metrics' has no attribute 'THRESHOLD_REGION'`.

- [ ] **Step 3: Change the constants**

In `cckf/metrics.py`, replace:

```python
DECISION_REGION: tuple[float, float] = (0.01, 0.5)
```

with:

```python
#: Region where calibration must hold, in probability units. Symmetric in
#: log-odds -- [0.01, 0.99] is z in [-4.595, +4.595] -- because spec §10.1
#: consumes the gate as an accumulated branch score sum_k log(g/(1-g)), not
#: only as an accept/reject threshold. A hit at g=0.99 contributes +4.60 to
#: that sum; if it is really 0.90 it should contribute +2.20, a 2.4x error in
#: one hit's weight, compounding over ~10 layers. Auditing only [0.01, 0.5]
#: covers exactly the negative half of the axis the score integrates over.
DECISION_REGION: tuple[float, float] = (0.01, 0.99)

#: The narrower accept/reject-threshold view: the range a g_min sweep would
#: plausibly cover for a loose, high-recall gate. Retained and still reported
#: so the numbers logged on 2026-08-17 stay reproducible and comparable.
THRESHOLD_REGION: tuple[float, float] = (0.01, 0.5)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS, all tests in the file.

- [ ] **Step 5: Run the full suite to catch dependants**

Run: `python -m pytest tests -q`
Expected: PASS. If any test asserted a DR-ECE value computed under the old region, update it to call `decision_region_ece(..., region=metrics.THRESHOLD_REGION)` — do **not** revert the constant.

- [ ] **Step 6: Commit**

```bash
git add cckf/metrics.py tests/test_metrics.py
git commit -m "Widen the calibration decision region to [0.01, 0.99]

The old (0.01, 0.5) audited only the negative half of the log-odds axis,
while spec 10.1 consumes the gate as a branch score sum_k log(g/(1-g))
that integrates over both halves symmetrically. The old range is retained
as THRESHOLD_REGION and still reported."
```

---

### Task 2: NLL trace and slope-inversion guard in the Platt fits

Two additions to `cckf/calibration.py`. The trace supplies the "calibration loss curve". The guard exists because the four-parameter form has a failure mode the two-parameter form cannot have: $a(x) = a_0 + a_1 \log n_{\text{window}}$ multiplies the logit, so if $a_1 < 0$ then $a(x)$ crosses zero at $n_{\text{window}} = e^{-a_0/a_1}$ and **inverts the model's ranking** above that occupancy. Arm C fitted $a_0 = 0.7007$, $a_1 = -0.1408$, which crosses at $n_{\text{window}} \approx 145$ — so this is a live concern, not a hypothetical.

**Files:**
- Modify: `cckf/calibration.py:66-120`
- Test: `tests/test_calibration_trace.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `calibration.fit_platt(logits, labels, trace: list[float] | None = None) -> tuple[float, float]`
  - `calibration.fit_platt_occupancy(logits, labels, n_window, trace: list[float] | None = None) -> tuple[float, float, float, float]`
  - `calibration.platt_occupancy_slope_violations(n_window, params) -> dict` with keys `n_window_at_slope_zero`, `n_rows_slope_nonpositive`, `frac_rows_slope_nonpositive`, `min_slope`, `max_n_window`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_calibration_trace.py`:

```python
"""Tests for the Platt NLL trace and the 4-param slope-inversion guard."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from cckf import calibration


def _synthetic(n: int = 20_000, seed: int = 0):
    """Miscalibrated logits with a known ground-truth link."""
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=2.0, size=n)
    labels = (rng.random(n) < expit(0.5 * z - 1.0)).astype(np.float64)
    return z, labels


def test_two_param_trace_decreases_and_ends_at_the_optimum():
    z, labels = _synthetic()
    trace: list[float] = []
    calibration.fit_platt(z, labels, trace=trace)

    assert len(trace) >= 2, "trace must record at least the start and the optimum"
    # Convex NLL under L-BFGS-B: no iterate may be worse than the start, and
    # the last entry must be the best seen.
    assert trace[-1] <= trace[0]
    assert trace[-1] == pytest.approx(min(trace), abs=1e-12)


def test_trace_is_optional_and_does_not_perturb_the_fit():
    z, labels = _synthetic()
    a_traced, b_traced = calibration.fit_platt(z, labels, trace=[])
    a_plain, b_plain = calibration.fit_platt(z, labels)
    assert a_traced == pytest.approx(a_plain, abs=1e-12)
    assert b_traced == pytest.approx(b_plain, abs=1e-12)


def test_four_param_trace_records_a_lower_optimum_than_two_param():
    """The 4-param family contains the 2-param family, so its NLL optimum
    cannot be worse. Checks the trace measures the objective it claims to."""
    z, labels = _synthetic()
    n_window = np.exp(np.linspace(0.0, 5.0, len(z)))

    t2: list[float] = []
    calibration.fit_platt(z, labels, trace=t2)
    t4: list[float] = []
    calibration.fit_platt_occupancy(z, labels, n_window, trace=t4)

    assert t4[-1] <= t2[-1] + 1e-9


def test_slope_violation_detects_an_inverting_calibrator():
    """a1 < 0 makes a(x) cross zero; above that occupancy the calibrator
    reverses the model's ranking, which no affine-in-logit map should do."""
    params = (0.7007, -0.1408, -3.6135, -0.7317)  # arm C's actual fit
    n_window = np.array([2.0, 10.0, 100.0, 200.0, 1000.0])

    report = calibration.platt_occupancy_slope_violations(n_window, params)

    assert report["n_window_at_slope_zero"] == pytest.approx(145.1, rel=0.01)
    assert report["n_rows_slope_nonpositive"] == 2  # 200 and 1000
    assert report["frac_rows_slope_nonpositive"] == pytest.approx(0.4)
    assert report["min_slope"] < 0.0


def test_slope_violation_is_empty_when_a1_is_positive():
    params = (0.9648, 0.0032878, -0.0989, -0.0283)  # arm A's actual fit
    n_window = np.array([1.0, 10.0, 1000.0, 100_000.0])

    report = calibration.platt_occupancy_slope_violations(n_window, params)

    assert report["n_rows_slope_nonpositive"] == 0
    assert report["frac_rows_slope_nonpositive"] == 0.0
    assert report["min_slope"] > 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_calibration_trace.py -v`
Expected: FAIL — `TypeError: fit_platt() got an unexpected keyword argument 'trace'` and `AttributeError: module 'cckf.calibration' has no attribute 'platt_occupancy_slope_violations'`.

- [ ] **Step 3: Add tracing to `_fit_logistic`**

In `cckf/calibration.py`, replace `_fit_logistic` with:

```python
def _fit_logistic(
    design: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    trace: list[float] | None = None,
) -> np.ndarray:
    """Fit a logistic model by L-BFGS on the convex NLL.

    Parameters
    ----------
    trace : list of float, optional
        If given, appended in place with the NLL at the initial guess, then
        once per L-BFGS-B iteration, then at the returned optimum. The final
        entry may duplicate the last iterate's value; that is harmless and
        keeps the invariant that ``trace[-1]`` is the NLL of the parameters
        actually returned. Tracing costs one extra NLL evaluation per
        iteration and does not affect the optimisation path.
    """
    y = np.asarray(y, dtype=np.float64)
    if not (np.any(y > 0.5) and np.any(y <= 0.5)):
        raise ValueError(
            "calibration split must contain both classes; cannot fit Platt on "
            "a single-class sample"
        )

    callback = None
    if trace is not None:
        trace.append(_nll(design, x0, y))

        def callback(xk: np.ndarray) -> None:
            trace.append(_nll(design, xk, y))

    result = minimize(
        lambda p: _nll(design, p, y),
        x0,
        jac=lambda p: _nll_grad(design, p, y),
        method="L-BFGS-B",
        callback=callback,
    )
    if trace is not None:
        trace.append(_nll(design, result.x, y))
    return result.x
```

- [ ] **Step 4: Thread `trace` through both public fits**

Replace the two public fit functions:

```python
def fit_platt(
    logits: np.ndarray, labels: np.ndarray, trace: list[float] | None = None
) -> tuple[float, float]:
    """Fit two-parameter Platt scaling on the calibration split.

    Parameters
    ----------
    trace : list of float, optional
        Appended in place with the per-iteration NLL; see ``_fit_logistic``.

    Returns
    -------
    tuple of float
        ``(a, b)``.
    """
    z = np.asarray(logits, dtype=np.float64)
    design = np.column_stack([z, np.ones_like(z)])
    a, b = _fit_logistic(design, labels, np.array([1.0, 0.0]), trace=trace)
    return float(a), float(b)
```

```python
def fit_platt_occupancy(
    logits: np.ndarray,
    labels: np.ndarray,
    n_window: np.ndarray,
    trace: list[float] | None = None,
) -> tuple[float, float, float, float]:
    """Fit four-parameter occupancy-conditional Platt scaling.

    Parameters
    ----------
    trace : list of float, optional
        Appended in place with the per-iteration NLL; see ``_fit_logistic``.

    Returns
    -------
    tuple of float
        ``(a0, a1, b0, b1)`` for ``a(x) = a0 + a1·log n_window`` and
        ``b(x) = b0 + b1·log n_window``.
    """
    z = np.asarray(logits, dtype=np.float64)
    log_nw = np.log(np.maximum(np.asarray(n_window, dtype=np.float64), _N_WINDOW_FLOOR))
    design = np.column_stack([z, z * log_nw, np.ones_like(z), log_nw])
    a0, a1, b0, b1 = _fit_logistic(
        design, labels, np.array([1.0, 0.0, 0.0, 0.0]), trace=trace
    )
    return float(a0), float(a1), float(b0), float(b1)
```

- [ ] **Step 5: Add the slope-inversion guard**

Append to `cckf/calibration.py`:

```python
def platt_occupancy_slope_violations(
    n_window: np.ndarray, params: tuple[float, float, float, float]
) -> dict:
    """Report rows where the 4-param calibrator's slope is non-positive.

    The occupancy-conditional form multiplies the logit by
    ``a(x) = a0 + a1·log n_window``. When ``a(x) <= 0`` the map is
    *decreasing* in the logit: a more confident model output becomes a lower
    calibrated probability, so the calibrator inverts the model's ranking for
    those rows. Two-parameter Platt cannot do this -- ``a`` is one fitted
    scalar and its sign is checkable once -- but the four-parameter form can,
    because ``a1 < 0`` makes ``a(x)`` cross zero at ``n_window =
    exp(-a0/a1)``.

    This is not hypothetical: the 2026-08-17 arm C fit gave
    ``a0 = 0.7007, a1 = -0.1408``, crossing at ``n_window ~= 145``.

    Parameters
    ----------
    n_window : numpy.ndarray
        Per-row occupancy, on the same rows the calibrator will be applied to.
    params : tuple of float
        ``(a0, a1, b0, b1)`` from :func:`fit_platt_occupancy`.

    Returns
    -------
    dict
        ``n_window_at_slope_zero`` (``inf`` when ``a1 == 0``; a value below 1
        means the slope never inverts for physical occupancies),
        ``n_rows_slope_nonpositive``, ``frac_rows_slope_nonpositive``,
        ``min_slope``, ``max_n_window``.
    """
    a0, a1, _, _ = params
    log_nw = np.log(np.maximum(np.asarray(n_window, dtype=np.float64), _N_WINDOW_FLOOR))
    slope = a0 + a1 * log_nw
    n_bad = int((slope <= 0.0).sum())
    crossing = float(np.exp(-a0 / a1)) if a1 != 0.0 else float("inf")
    return {
        "n_window_at_slope_zero": crossing,
        "n_rows_slope_nonpositive": n_bad,
        "frac_rows_slope_nonpositive": n_bad / max(len(log_nw), 1),
        "min_slope": float(slope.min()) if len(slope) else float("nan"),
        "max_n_window": float(np.exp(log_nw.max())) if len(log_nw) else float("nan"),
    }
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_calibration_trace.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest tests -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add cckf/calibration.py tests/test_calibration_trace.py
git commit -m "Add Platt NLL tracing and a 4-param slope-inversion guard

The trace supplies the calibration-fit convergence curve. The guard exists
because a(x) = a0 + a1*log n_window multiplies the logit, so a1 < 0 makes
a(x) cross zero and invert the model's ranking above that occupancy -- arm
C's fit crosses at n_window ~= 145."
```

---

### Task 3: Display-resolution ROC/PR curves and metric bundles

`sklearn.roc_curve` on 24.1M rows returns a point per distinct threshold — correct, but useless for a figure and heavy to move off the volume. This task computes the curve **exactly at a fixed grid of log-odds thresholds** by cumulative-summing histograms of the positive and negative score distributions: one $O(N)$ pass, $O(G)$ memory.

The AUCs are deliberately **not** taken from that grid. Trapezoidal integration over 4,000 points would bias a headline number, so they come from sklearn on the full data. Grid for display, sklearn for the numbers.

**Files:**
- Create: `cckf/curves.py`
- Test: `tests/test_curves.py`

**Interfaces:**
- Consumes: `metrics.DECISION_REGION`, `metrics.THRESHOLD_REGION`, `metrics.logit_bin_edges`, `metrics.expected_calibration_error`, `metrics.max_calibration_error`, `metrics.decision_region_ece` (Task 1).
- Produces:
  - `curves.GRID_LOGIT_RANGE = (-20.0, 20.0)`, `curves.GRID_POINTS = 4000`
  - `curves.grid_curves(logits, labels, n_points=..., logit_range=...) -> dict` with keys `threshold_logit`, `threshold_prob`, `tpr`, `fpr`, `precision`, `tp`, `fp`, `n_pos`, `n_neg`, `base_rate`
  - `curves.metric_bundle(prob, labels) -> dict` with keys `auc_roc`, `auc_pr`, `ece`, `mce`, `dr_ece`, `dr_n_rows`, `threshold_region_ece`, `threshold_region_n_rows`, `base_rate`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_curves.py`:

```python
"""Tests for grid-threshold ROC/PR curves and before/after metric bundles."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from cckf import curves


def _synthetic(n: int = 50_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=2.0, size=n)
    labels = rng.random(n) < expit(z - 4.0)  # ~2% positive
    return z, labels


def test_grid_curve_endpoints_are_the_trivial_classifiers():
    """At the lowest threshold everything is accepted; at the highest,
    nothing is."""
    z, labels = _synthetic()
    c = curves.grid_curves(z, labels)

    assert c["tpr"][0] == pytest.approx(1.0)
    assert c["fpr"][0] == pytest.approx(1.0)
    assert c["tp"][-1] == 0
    assert c["fp"][-1] == 0


def test_grid_counts_are_exact_at_grid_thresholds():
    """The grid subsamples which thresholds are displayed; it must not
    approximate the counts at those thresholds."""
    z, labels = _synthetic(n=10_000)
    c = curves.grid_curves(z, labels, n_points=50, logit_range=(-6.0, 6.0))
    z_clipped = np.clip(z, -6.0, 6.0)

    for j in (0, 7, 25, 49):
        t = c["threshold_logit"][j]
        assert c["tp"][j] == int(((z_clipped >= t) & labels).sum())
        assert c["fp"][j] == int(((z_clipped >= t) & ~labels).sum())


def test_rates_are_monotone_in_threshold():
    z, labels = _synthetic()
    c = curves.grid_curves(z, labels)
    assert np.all(np.diff(c["tpr"]) <= 1e-12)
    assert np.all(np.diff(c["fpr"]) <= 1e-12)


def test_precision_is_nan_where_nothing_is_accepted():
    """0/0 must not be reported as a precision value."""
    z, labels = _synthetic()
    c = curves.grid_curves(z, labels)
    empty = (c["tp"] + c["fp"]) == 0
    assert empty.any()
    assert np.all(np.isnan(c["precision"][empty]))


def test_metric_bundle_auc_matches_sklearn_on_full_data():
    """AUC must come from sklearn on all rows, not from the display grid."""
    z, labels = _synthetic()
    prob = expit(z)
    bundle = curves.metric_bundle(prob, labels)

    assert bundle["auc_roc"] == pytest.approx(roc_auc_score(labels, prob), abs=1e-12)
    assert bundle["auc_pr"] == pytest.approx(
        average_precision_score(labels, prob), abs=1e-12
    )


def test_two_param_platt_leaves_auc_invariant_but_moves_ece():
    """The central fact behind the before/after figure.

    2-param Platt is z' = a*z + b with a > 0, a strictly increasing map, so it
    cannot change the ranking -- ROC and PR curves and both AUCs are
    identical. ECE is a property of the values, so it does move. A figure that
    plots ROC 'before vs after' 2-param calibration draws two superimposed
    lines; this test is what keeps that from looking like a bug.
    """
    z, labels = _synthetic()
    before = curves.metric_bundle(expit(z), labels)
    after = curves.metric_bundle(expit(0.5 * z - 2.0), labels)

    assert after["auc_roc"] == pytest.approx(before["auc_roc"], abs=1e-12)
    assert after["auc_pr"] == pytest.approx(before["auc_pr"], abs=1e-12)
    assert abs(after["ece"] - before["ece"]) > 1e-4


def test_metric_bundle_reports_both_regions():
    z, labels = _synthetic()
    bundle = curves.metric_bundle(expit(z), labels)
    assert bundle["dr_n_rows"] >= bundle["threshold_region_n_rows"]
    assert set(bundle) >= {
        "auc_roc",
        "auc_pr",
        "ece",
        "mce",
        "dr_ece",
        "dr_n_rows",
        "threshold_region_ece",
        "threshold_region_n_rows",
        "base_rate",
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_curves.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cckf.curves'`.

- [ ] **Step 3: Create `cckf/curves.py`**

```python
"""Display-resolution ROC/PR curves and before/after metric bundles.

Why a threshold grid instead of ``sklearn.roc_curve``
-----------------------------------------------------
``roc_curve`` sorts every score and returns one point per distinct threshold.
On the 24.1M-row val split that is tens of millions of vertices to draw what
is visually a smooth line, and tens of megabytes to move off the Modal volume
per arm per estimator.

``TP(t)`` and ``FP(t)`` are just *counts above a threshold*, which are reverse
cumulative sums of histograms. So fixing a grid of thresholds and
histogramming the positive and negative score distributions over it gives the
exact counts at those thresholds in one O(N) pass and O(G) memory.

The grid is uniform in log-odds, which puts the displayed points densely in
the high-threshold / low-FPR region where the gate actually operates, instead
of densely where the curve is uninformative.

What is exact and what is not
-----------------------------
The counts are **exact** at the grid thresholds -- the grid subsamples which
thresholds are displayed, it does not approximate anything about them.

The AUCs are **not** computed from the grid. Trapezoidal integration over
4,000 points would put a small bias into a headline number, so
:func:`metric_bundle` takes them from scikit-learn on the full data. Grid for
display, sklearn for the numbers.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import metrics

#: Log-odds range spanned by the display grid. |z| = 20 is a probability
#: within 2e-9 of 0 or 1 -- past any resolution this project cares about.
#: Scores are clipped into this range rather than dropped, so no row silently
#: vanishes from a curve.
GRID_LOGIT_RANGE: tuple[float, float] = (-20.0, 20.0)

#: Number of grid intervals. 4,000 gives ~0.01 in log-odds per step, far finer
#: than any visible feature at figure resolution.
GRID_POINTS: int = 4000


def grid_curves(
    logits: np.ndarray,
    labels: np.ndarray,
    n_points: int = GRID_POINTS,
    logit_range: tuple[float, float] = GRID_LOGIT_RANGE,
) -> dict:
    """Exact ROC/PR quantities at a fixed grid of log-odds thresholds.

    Parameters
    ----------
    logits : numpy.ndarray
        Raw log-odds scores. Pass logits rather than probabilities so the grid
        spacing is uniform in the space the thresholds live in.
    labels : numpy.ndarray
        Binary truth, cast to bool.
    n_points : int
        Number of grid intervals; returned arrays have ``n_points + 1``
        entries, one per grid edge.
    logit_range : tuple of float
        Inclusive log-odds range of the grid.

    Returns
    -------
    dict
        ``threshold_logit``, ``threshold_prob``, ``tpr``, ``fpr``,
        ``precision`` (NaN where nothing is accepted), ``tp``, ``fp``,
        ``n_pos``, ``n_neg``, ``base_rate``. All arrays share length
        ``n_points + 1``, ordered from the lowest threshold (accept
        everything) to the highest (accept nothing).
    """
    lo, hi = logit_range
    z = np.clip(np.asarray(logits, dtype=np.float64), lo, hi)
    labels = np.asarray(labels).astype(bool)

    edges = np.linspace(lo, hi, n_points + 1)
    pos_hist, _ = np.histogram(z[labels], bins=edges)
    neg_hist, _ = np.histogram(z[~labels], bins=edges)

    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())

    # Counts at or above each edge. np.histogram's last bin is closed on the
    # right, so a score exactly at ``hi`` is counted in it and tp/fp reach 0 at
    # the final edge -- the "accept nothing" endpoint.
    tp = n_pos - np.concatenate([[0], np.cumsum(pos_hist)])
    fp = n_neg - np.concatenate([[0], np.cumsum(neg_hist)])

    accepted = tp + fp
    with np.errstate(invalid="ignore", divide="ignore"):
        tpr = tp / n_pos if n_pos else np.full(tp.shape, np.nan)
        fpr = fp / n_neg if n_neg else np.full(fp.shape, np.nan)
        # 0/0 is "no data", not "perfect precision".
        precision = np.where(accepted > 0, tp / np.maximum(accepted, 1), np.nan)

    total = n_pos + n_neg
    return {
        "threshold_logit": edges,
        "threshold_prob": 1.0 / (1.0 + np.exp(-edges)),
        "tpr": np.asarray(tpr, dtype=np.float64),
        "fpr": np.asarray(fpr, dtype=np.float64),
        "precision": np.asarray(precision, dtype=np.float64),
        "tp": tp.astype(np.int64),
        "fp": fp.astype(np.int64),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "base_rate": (n_pos / total) if total else float("nan"),
    }


def metric_bundle(prob: np.ndarray, labels: np.ndarray) -> dict:
    """Every scalar metric for one (estimator, split) pair.

    Discrimination (``auc_roc``, ``auc_pr``) depends only on the ranking of
    ``prob``; calibration (``ece``, ``mce``, ``dr_ece``) depends on its values.
    That split is why a monotone recalibration moves the second group and
    provably not the first -- see ``tests/test_curves.py::
    test_two_param_platt_leaves_auc_invariant_but_moves_ece``.

    Parameters
    ----------
    prob : numpy.ndarray
        Probabilities in [0, 1].
    labels : numpy.ndarray
        Binary truth, cast to bool.

    Returns
    -------
    dict
        ``auc_roc``, ``auc_pr``, ``ece``, ``mce``, ``dr_ece``, ``dr_n_rows``,
        ``threshold_region_ece``, ``threshold_region_n_rows``, ``base_rate``.
    """
    prob = np.asarray(prob, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    edges = metrics.logit_bin_edges(30)

    wide = metrics.decision_region_ece(prob, labels, region=metrics.DECISION_REGION)
    narrow = metrics.decision_region_ece(
        prob, labels, region=metrics.THRESHOLD_REGION
    )

    return {
        "auc_roc": float(roc_auc_score(labels, prob)),
        "auc_pr": float(average_precision_score(labels, prob)),
        "ece": float(metrics.expected_calibration_error(prob, labels, edges=edges)),
        "mce": float(metrics.max_calibration_error(prob, labels, edges=edges)),
        "dr_ece": float(wide["ece"]),
        "dr_n_rows": int(wide["n_rows"]),
        "threshold_region_ece": float(narrow["ece"]),
        "threshold_region_n_rows": int(narrow["n_rows"]),
        "base_rate": float(labels.mean()),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_curves.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Format and run the full suite**

Run: `black cckf/curves.py tests/test_curves.py && python -m pytest tests -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add cckf/curves.py tests/test_curves.py
git commit -m "Add grid-threshold ROC/PR curves and metric bundles

TP(t) and FP(t) are counts above a threshold, so histogramming the positive
and negative score distributions over a fixed log-odds grid gives exact
counts at those thresholds in one O(N) pass and O(G) memory -- instead of
sklearn's one-point-per-distinct-threshold output over 24.1M rows. AUCs still
come from sklearn on the full data, since trapezoidal integration over the
grid would bias a headline number."
```

---

### Task 4: Export the plot-data bundle from the volume

One Modal invocation handles all three arms sequentially (never concurrently — see Global Constraints). For each arm it loads the frozen checkpoint, runs inference on val and cal, fits both Platt forms on **cal** with NLL traces, applies them to **val**, and writes a compact `.npz` plus `.json` — roughly 1 MB per arm, so figure iteration happens locally without moving 68M inference scores.

**Files:**
- Create: `scripts/export_gate_curves.py`
- Modify: `modal_train.py` (add `export_gate_curves` function and `export_curves` entrypoint)

**Interfaces:**
- Consumes: `curves.grid_curves`, `curves.metric_bundle` (Task 3); `calibration.fit_platt`, `calibration.fit_platt_occupancy`, `calibration.apply_platt`, `calibration.apply_platt_occupancy`, `calibration.platt_occupancy_slope_violations` (Task 2); `cache.load_cache`, `train.StandardizedView`, `train.predict_logits`, `models.GateMLP`, `features.GATE_FEATURES`.
- Produces: `/data/results/curves/gate_{A,B,C}.npz` and `.json`.

- [ ] **Step 1: Create `scripts/export_gate_curves.py`**

```python
"""Export plot-data bundles for the gate figure set.

Reads a frozen ``gate_model.pt``, runs inference on the val and cal caches,
fits both Platt forms on **cal** (never val -- the calibration split is the
only split allowed to see a calibrator fit), applies them to **val**, and
writes a small bundle of grid curves plus scalar metrics.

Evaluating before/after on val rather than cal keeps the calibrator's fit and
its evaluation on disjoint data, and makes the "before" AUCs directly
comparable to the numbers already in experiments/LOG.md.

Usage
-----
    python scripts/export_gate_curves.py \\
        --model-dir /data/models/gate_A \\
        --val-cache /data/cache/gate/val \\
        --cal-cache /data/cache/gate/cal \\
        --out-dir /data/results/curves
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, logit

from cckf import cache, calibration, curves, features, models, train

#: aux.f32 column order written by cckf.cache.build_gate_cache.
AUX_CHI2, AUX_N_WINDOW, AUX_ETA = 0, 1, 2

#: Keep probabilities off the open interval's endpoints before taking logits,
#: so an exactly-saturated 0.0 or 1.0 does not become +-inf on a plot axis.
_P_EPS = 1e-12


def _load_model(model_dir: Path) -> dict:
    ckpt = torch.load(
        model_dir / "gate_model.pt", map_location="cpu", weights_only=False
    )
    model = models.GateMLP(
        n_features=ckpt["n_features"], width=ckpt["width"], depth=ckpt["depth"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {"model": model, "ckpt": ckpt}


def _logits_for(loaded: dict, cache_dir: str, device: str) -> tuple[np.ndarray, dict]:
    """Raw logits over one whole cache, streamed a batch at a time.

    Uses ``train.StandardizedView`` rather than materialising the standardised
    matrix: the cal cache is 44.4M x 26 float32, so an eager copy is ~4.6 GB
    and the transient standardisation doubles it.
    """
    ckpt = loaded["ckpt"]
    cached = cache.load_cache(cache_dir)
    col_idx = np.array(
        [features.GATE_FEATURES.index(n) for n in ckpt["feature_names"]]
    )
    # The checkpoint's mu/sigma are already subset to the model's own columns,
    # but StandardizedView indexes mu/sigma by *source* column before
    # narrowing, so scatter them back to full width first.
    mu = np.zeros(len(features.GATE_FEATURES), dtype=np.float32)
    sigma = np.ones(len(features.GATE_FEATURES), dtype=np.float32)
    mu[col_idx] = ckpt["mu"]
    sigma[col_idx] = ckpt["sigma"]

    view = train.StandardizedView(
        cached["X"], np.arange(len(cached["y"])), mu, sigma, col_idx
    )
    return train.predict_logits(loaded["model"], view, device=device), cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--cal-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    arm = model_dir.name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = _load_model(model_dir)
    training_metrics = json.loads((model_dir / "gate_metrics.json").read_text())

    cal_logits, cal = _logits_for(loaded, args.cal_cache, args.device)
    cal_labels = np.asarray(cal["y"]).astype(np.float64)
    cal_nw = np.asarray(cal["aux"][:, AUX_N_WINDOW], dtype=np.float64)

    # Platt is fitted on cal only.
    trace2: list[float] = []
    a, b = calibration.fit_platt(cal_logits, cal_labels, trace=trace2)
    trace4: list[float] = []
    p4 = calibration.fit_platt_occupancy(cal_logits, cal_labels, cal_nw, trace=trace4)

    val_logits, val = _logits_for(loaded, args.val_cache, args.device)
    val_labels = np.asarray(val["y"]).astype(bool)
    val_nw = np.asarray(val["aux"][:, AUX_N_WINDOW], dtype=np.float64)

    estimators = {
        # chi2-implied probability on the same val rows, for a shared axis.
        "chi2_lambda": np.exp(
            -0.5 * np.asarray(val["aux"][:, AUX_CHI2], dtype=np.float64)
        ),
        "gate_raw": expit(val_logits),
        "gate_platt2": calibration.apply_platt(val_logits, a, b),
        "gate_platt4": calibration.apply_platt_occupancy(val_logits, val_nw, p4),
    }

    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, object] = {
        "arm": arm,
        "platt_2param": {"a": a, "b": b},
        "platt_4param": dict(zip(("a0", "a1", "b0", "b1"), p4)),
        "prior_logit_shift": float(loaded["ckpt"].get("prior_logit_shift", 0.0)),
        "calibration_nll_trace_2param": [float(v) for v in trace2],
        "calibration_nll_trace_4param": [float(v) for v in trace4],
        "slope_violations_4param": calibration.platt_occupancy_slope_violations(
            val_nw, p4
        ),
        "training_history": training_metrics["history"],
        "n_val_rows": int(len(val_labels)),
        "n_cal_rows": int(len(cal_labels)),
        "metrics": {},
    }

    for name, prob in estimators.items():
        p = np.clip(prob, _P_EPS, 1.0 - _P_EPS)
        c = curves.grid_curves(logit(p), val_labels)
        for key in ("threshold_logit", "tpr", "fpr", "precision"):
            arrays[f"{name}__{key}"] = c[key].astype(np.float32)
        arrays[f"{name}__tp"] = c["tp"].astype(np.int64)
        arrays[f"{name}__fp"] = c["fp"].astype(np.int64)
        scalars["metrics"][name] = curves.metric_bundle(prob, val_labels)
        print(f"{arm} {name}: {scalars['metrics'][name]}")

    np.savez_compressed(out_dir / f"{arm}.npz", **arrays)
    (out_dir / f"{arm}.json").write_text(json.dumps(scalars, indent=2, default=str))
    print(f"wrote {out_dir / f'{arm}.npz'} and {arm}.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify imports resolve and the CLI parses**

Run: `PYTHONPATH=. python scripts/export_gate_curves.py --help`
Expected: usage text listing `--model-dir`, `--val-cache`, `--cal-cache`, `--out-dir`, `--device`.

- [ ] **Step 3: Add the Modal function and entrypoint**

In `modal_train.py`, add after `run_audit`:

```python
@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    gpu="A10G",
    cpu=8,
    memory=131072,
    timeout=14400,
)
def export_gate_curves(arms: str = "A,B,C") -> list[str]:
    """Export plot-data bundles for each gate arm, sequentially.

    Sequential by construction with a single ``data_vol.commit()`` after all
    arms: concurrent commits corrupted 11 Parquets on this volume once
    already (experiments/LOG.md, 2026-08-12), and these containers would all
    be writing to the same directory.
    """
    import sys

    written = []
    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        _run_script(
            [
                sys.executable,
                "/root/scripts/export_gate_curves.py",
                "--model-dir",
                f"{MODEL_DIR}/gate_{arm}",
                "--val-cache",
                f"{CACHE_DIR}/gate/val",
                "--cal-cache",
                f"{CACHE_DIR}/gate/cal",
                "--out-dir",
                f"{DATA_PATH}/results/curves",
            ]
        )
        written.append(f"{DATA_PATH}/results/curves/gate_{arm}.npz")
    data_vol.commit()
    return written


@app.local_entrypoint()
def export_curves(arms: str = "A,B,C") -> None:
    """Usage: modal run modal_train.py::export_curves --arms A,B,C"""
    print(export_gate_curves.remote(arms=arms))
```

- [ ] **Step 4: Verify `modal_train.py` still parses**

Run: `python -c "import ast; ast.parse(open('modal_train.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 5: Format, test, commit**

```bash
black scripts/export_gate_curves.py modal_train.py
python -m pytest tests -q
git add scripts/export_gate_curves.py modal_train.py
git commit -m "Export small plot-data bundles for the gate figure set

Runs inference on the frozen checkpoints, fits Platt on cal, applies to val,
and writes 4,001-point grid curves plus scalars (~1 MB per arm) so figure
iteration happens locally without moving 68M inference scores."
```

---

### Task 5: The figure set

Six figures from the bundles. Three of them carry a subtlety a reader would otherwise misread as a bug, so each states in its own caption what it does and does not show.

**Files:**
- Create: `scripts/plot_gate_figures.py`
- Test: manual — figures are visual artifacts, and every number they draw is already covered by Tasks 1–3.

**Interfaces:**
- Consumes: the `.npz`/`.json` bundles from Task 4; `metrics.logit_bin_edges`, `metrics.MIN_BIN_COUNT`, `metrics.DECISION_REGION`, `metrics.THRESHOLD_REGION`.
- Produces: `figures/gate/{F1_loss_curves,F2_roc,F3_pr,F4_calibration_nll,F5_reliability,F6_before_after}.{png,pdf}`

- [ ] **Step 1: Create `scripts/plot_gate_figures.py`**

```python
"""Figure set for the gate S1 sampling ablation.

Reads the bundles written by scripts/export_gate_curves.py and writes six
figures. Every panel uses fixed, shared axes so the three arms and the chi2
baseline are directly comparable.

Usage
-----
    python scripts/plot_gate_figures.py \\
        --bundle-dir results/curves --out-dir figures/gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cckf import metrics

ARMS = ("A", "B", "C")
ARM_LABEL = {
    "A": "A — no subsampling",
    "B": "B — uniform 1:5",
    "C": "C — hard-neg ∝1/χ²",
}
ARM_COLOR = {"A": "#1b5e9c", "B": "#2e8b57", "C": "#c0392b"}
EST_LABEL = {
    "chi2_lambda": "χ²_λ baseline",
    "gate_raw": "gate, raw",
    "gate_platt2": "gate + Platt-2",
    "gate_platt4": "gate + Platt-4",
}
EXTS = ("png", "pdf")


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in EXTS:
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def _load(bundle_dir: Path) -> dict:
    return {
        arm: {
            "arrays": np.load(bundle_dir / f"gate_{arm}.npz"),
            "scalars": json.loads((bundle_dir / f"gate_{arm}.json").read_text()),
        }
        for arm in ARMS
    }


def figure_loss_curves(data: dict, out_dir: Path) -> None:
    """F1: per-epoch train/val BCE for each arm."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, arm in zip(axes, ARMS):
        hist = data[arm]["scalars"]["training_history"]
        epochs = np.arange(1, len(hist["train_loss"]) + 1)
        ax.semilogy(epochs, hist["train_loss"], color=ARM_COLOR[arm], label="train BCE")
        ax.semilogy(
            epochs, hist["val_loss"], color=ARM_COLOR[arm], ls="--", label="val BCE"
        )
        best = int(np.argmin(hist["val_loss"]))
        ax.axvline(best + 1, color="0.5", lw=0.8, ls=":")
        ax.set_title(
            f"{ARM_LABEL[arm]}\nbest epoch {best + 1} of {len(epochs)}", fontsize=9
        )
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("BCE")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "F1  Gate training loss. Val BCE is NOT comparable across arms: B and C "
        "carry deliberate subsampling\nbias in their logits that Platt later "
        "removes, and BCE penalises exactly that bias. Compare AUC instead.",
        fontsize=9,
    )
    _save(fig, out_dir, "F1_loss_curves")


def figure_roc(data: dict, out_dir: Path) -> None:
    """F2: ROC per arm, full range plus a log-FPR view of the usable region."""
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12.5, 5))
    for arm in ARMS:
        arr, sc = data[arm]["arrays"], data[arm]["scalars"]
        auc = sc["metrics"]["gate_platt2"]["auc_roc"]
        for ax in (ax_full, ax_zoom):
            ax.plot(
                arr["gate_platt2__fpr"],
                arr["gate_platt2__tpr"],
                color=ARM_COLOR[arm],
                lw=1.6,
                label=f"{ARM_LABEL[arm]}  AUC {auc:.4f}",
            )
    arr = data["A"]["arrays"]
    auc_chi2 = data["A"]["scalars"]["metrics"]["chi2_lambda"]["auc_roc"]
    for ax in (ax_full, ax_zoom):
        ax.plot(
            arr["chi2_lambda__fpr"],
            arr["chi2_lambda__tpr"],
            color="0.35",
            lw=1.2,
            ls="-.",
            label=f"χ²_λ baseline  AUC {auc_chi2:.4f}",
        )
        ax.grid(alpha=0.3)
        ax.set_ylabel("true-positive rate (efficiency)")
    ax_full.plot([0, 1], [0, 1], color="0.7", lw=0.8, ls=":", label="chance")
    ax_full.set_xlabel("false-positive rate")
    ax_full.set_title("full range", fontsize=9)
    ax_zoom.set_xscale("log")
    ax_zoom.set_xlim(1e-6, 1.0)
    ax_zoom.set_xlabel("false-positive rate (log)")
    ax_zoom.set_title("usable region — FPR ≲ 1e-2", fontsize=9)
    for ax in (ax_full, ax_zoom):
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "F2  ROC on the val split. Curves depend only on score *ranking*, so "
        "2-param Platt cannot move them\n(a > 0 is strictly increasing) — raw "
        "and Platt-2 ROCs are identical by construction, not by coincidence.",
        fontsize=9,
    )
    _save(fig, out_dir, "F2_roc")


def figure_pr(data: dict, out_dir: Path) -> None:
    """F3: precision-recall per arm, with the base-rate floor drawn."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    base = data["A"]["scalars"]["metrics"]["gate_platt2"]["base_rate"]
    for arm in ARMS:
        arr, sc = data[arm]["arrays"], data[arm]["scalars"]
        rec, prec = arr["gate_platt2__tpr"], arr["gate_platt2__precision"]
        ok = np.isfinite(prec)
        ax.plot(
            rec[ok],
            prec[ok],
            color=ARM_COLOR[arm],
            lw=1.6,
            label=f"{ARM_LABEL[arm]}  AP {sc['metrics']['gate_platt2']['auc_pr']:.4f}",
        )
    arr = data["A"]["arrays"]
    prec = arr["chi2_lambda__precision"]
    ok = np.isfinite(prec)
    ax.plot(
        arr["chi2_lambda__tpr"][ok],
        prec[ok],
        color="0.35",
        lw=1.2,
        ls="-.",
        label=f"χ²_λ  AP {data['A']['scalars']['metrics']['chi2_lambda']['auc_pr']:.4f}",
    )
    ax.axhline(
        base, color="0.7", lw=1.0, ls=":", label=f"no-skill floor = {base:.4%}"
    )
    ax.set_yscale("log")
    ax.set_xlabel("recall (efficiency)")
    ax.set_ylabel("precision (purity, log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(
        f"F3  Precision-recall on the val split. At a {base:.2%} base rate, "
        "ROC's FPR denominator\n(~24M negatives) hides operational cost; PR's "
        "denominator is TP+FP, which is what a\nbranch budget actually pays.",
        fontsize=9,
    )
    _save(fig, out_dir, "F3_pr")


def figure_calibration_nll(data: dict, out_dir: Path) -> None:
    """F4: L-BFGS-B convergence of both Platt fits, per arm."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in ARMS:
        sc = data[arm]["scalars"]
        for key, ls, tag in (
            ("calibration_nll_trace_2param", "-", "Platt-2"),
            ("calibration_nll_trace_4param", "--", "Platt-4"),
        ):
            trace = np.asarray(sc[key], dtype=float)
            ax.plot(
                np.arange(len(trace)),
                trace,
                color=ARM_COLOR[arm],
                ls=ls,
                lw=1.5,
                marker="o",
                ms=3,
                label=f"{ARM_LABEL[arm]} — {tag}",
            )
    ax.set_yscale("log")
    ax.set_xlabel("L-BFGS-B iteration (0 = initial guess a=1, b=0)")
    ax.set_ylabel("mean NLL on the calibration split (log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title(
        "F4  Calibration fit convergence. The NLL is convex in the Platt "
        "parameters, so these traces are\nmonotone and the optimum is global — "
        "the curve shows cost, not risk of a bad local minimum.",
        fontsize=9,
    )
    _save(fig, out_dir, "F4_calibration_nll")


def _reliability_from_bundle(entry: dict, est: str, edges: np.ndarray) -> dict:
    """Per-bin observed vs predicted, reconstructed from the stored grid.

    The bundle ships tp/fp at 4,001 grid thresholds rather than 24M raw
    scores. Counts *within* a reliability bin are differences of those
    cumulative counts at the bin's two edges, so this is exact wherever a bin
    edge falls on a grid edge and off by at most one grid step (0.01 in
    log-odds) elsewhere. Good enough to draw; the ECE/MCE *values* quoted in
    the legend come from metric_bundle, computed on full data.
    """
    arr = entry["arrays"]
    z_grid = np.asarray(arr[f"{est}__threshold_logit"], dtype=np.float64)
    tp = np.asarray(arr[f"{est}__tp"], dtype=np.float64)
    fp = np.asarray(arr[f"{est}__fp"], dtype=np.float64)

    e = np.clip(np.asarray(edges, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    z_edges = np.log(e / (1.0 - e))
    idx = np.clip(np.searchsorted(z_grid, z_edges), 0, len(z_grid) - 1)

    obs, pred = [], []
    for j in range(len(idx) - 1):
        hi_i, lo_i = idx[j], idx[j + 1]
        pos_bin = tp[hi_i] - tp[lo_i]
        n_bin = pos_bin + (fp[hi_i] - fp[lo_i])
        if n_bin < metrics.MIN_BIN_COUNT:
            continue
        obs.append(pos_bin / n_bin)
        pred.append(0.5 * (edges[j] + edges[j + 1]))
    return {
        "observed_fraction": np.array(obs),
        "mean_predicted": np.array(pred),
    }


def figure_reliability(data: dict, out_dir: Path) -> None:
    """F5: reliability diagrams, one panel per arm, on shared logit axes."""
    edges = metrics.logit_bin_edges(30)
    lo, hi = metrics.DECISION_REGION
    t_lo, t_hi = metrics.THRESHOLD_REGION

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharex=True, sharey=True)
    for ax, arm in zip(axes, ARMS):
        sc = data[arm]["scalars"]
        for est, color, marker in (
            ("chi2_lambda", "0.35", "s"),
            ("gate_raw", "#e08a1e", "^"),
            ("gate_platt2", ARM_COLOR[arm], "o"),
        ):
            b = _reliability_from_bundle(data[arm], est, edges)
            if not len(b["mean_predicted"]):
                continue
            m = sc["metrics"][est]
            ax.plot(
                b["mean_predicted"],
                b["observed_fraction"],
                marker=marker,
                ms=4,
                lw=1.3,
                color=color,
                label=f"{EST_LABEL[est]}  ECE {m['ece']:.2e}, MCE {m['mce']:.3f}",
            )
        ax.plot([1e-5, 1 - 1e-5], [1e-5, 1 - 1e-5], color="0.7", lw=0.8, ls=":")
        ax.axvspan(lo, hi, color="#4a90d9", alpha=0.08)
        ax.axvspan(t_lo, t_hi, color="#4a90d9", alpha=0.08)
        ax.set_xscale("logit")
        ax.set_yscale("logit")
        ax.set_xlabel("mean predicted probability")
        ax.set_title(ARM_LABEL[arm], fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("observed positive fraction")
    fig.suptitle(
        "F5  Reliability on the val split, bins uniform in log-odds (30 bins, "
        f"p∈[1e-5, 0.99999]). Shaded: the audited\ndecision region [{lo}, {hi}] "
        f"and, darker, the narrower threshold view [{t_lo}, {t_hi}]. Bin edges "
        "are fixed constants\nshared by every curve — quantile edges would give "
        "each estimator its own x-axis.",
        fontsize=9,
    )
    _save(fig, out_dir, "F5_reliability")


def figure_before_after(data: dict, out_dir: Path) -> None:
    """F6: metric deltas from calibration, grouped bars."""
    keys = [
        ("auc_roc", "AUC-ROC"),
        ("auc_pr", "AUC-PR"),
        ("ece", "ECE"),
        ("dr_ece", f"DR-ECE {metrics.DECISION_REGION}"),
        ("mce", "MCE"),
    ]
    fig, axes = plt.subplots(1, len(keys), figsize=(18, 4.4))
    width = 0.26
    for ax, (key, title) in zip(axes, keys):
        for i, est in enumerate(("gate_raw", "gate_platt2", "gate_platt4")):
            vals = [data[arm]["scalars"]["metrics"][est][key] for arm in ARMS]
            ax.bar(
                np.arange(len(ARMS)) + (i - 1) * width,
                vals,
                width,
                label=EST_LABEL[est],
                color=["#bbbbbb", "#4a90d9", "#1b5e9c"][i],
            )
        chi2 = data["A"]["scalars"]["metrics"]["chi2_lambda"][key]
        ax.axhline(chi2, color="0.35", ls="-.", lw=1.2, label="χ²_λ")
        if key in ("ece", "dr_ece", "mce"):
            ax.set_yscale("log")
        ax.set_xticks(np.arange(len(ARMS)))
        ax.set_xticklabels(ARMS)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y", which="both")
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "F6  Before/after calibration on the val split. AUC-ROC and AUC-PR are "
        "IDENTICAL for raw and Platt-2 by\nconstruction (a > 0 is monotone, and "
        "AUC sees only ranking); they move only under Platt-4, whose slope\n"
        "a(x) = a0 + a1·log n_window is row-dependent and so is not a single "
        "monotone map.",
        fontsize=9,
    )
    _save(fig, out_dir, "F6_before_after")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", default="results/curves")
    parser.add_argument("--out-dir", default="figures/gate")
    args = parser.parse_args()

    data = _load(Path(args.bundle_dir))
    out_dir = Path(args.out_dir)
    figure_loss_curves(data, out_dir)
    figure_roc(data, out_dir)
    figure_pr(data, out_dir)
    figure_calibration_nll(data, out_dir)
    figure_reliability(data, out_dir)
    figure_before_after(data, out_dir)

    for arm in ARMS:
        sv = data[arm]["scalars"]["slope_violations_4param"]
        if sv["n_rows_slope_nonpositive"]:
            print(
                f"WARNING arm {arm}: Platt-4 slope <= 0 on "
                f"{sv['frac_rows_slope_nonpositive']:.4%} of val rows "
                f"(n_window >= {sv['n_window_at_slope_zero']:.0f}); the "
                "calibrator inverts the model's ranking there"
            )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the CLI parses**

Run: `PYTHONPATH=. python scripts/plot_gate_figures.py --help`
Expected: usage text listing `--bundle-dir` and `--out-dir`.

- [ ] **Step 3: Format and commit**

```bash
black scripts/plot_gate_figures.py
git add scripts/plot_gate_figures.py
git commit -m "Add the gate figure set

Six figures: loss curves, ROC (full + log-FPR), PR with the base-rate floor,
Platt NLL convergence, reliability on shared log-odds bins, and before/after
metric deltas. Captions state the two facts a reader would otherwise misread
as bugs: val BCE is not comparable across arms, and AUC is invariant under
2-param Platt."
```

---

### Task 6: Run it and record the results

**Files:**
- Modify: `experiments/LOG.md`
- Create: `results/curves/gate_{A,B,C}.{npz,json}` (downloaded), `figures/gate/*.{png,pdf}`

**Interfaces:**
- Consumes: everything above.
- Produces: the figure set and the log entry.

- [ ] **Step 1: Export the bundles**

Run: `modal run modal_train.py::export_curves --arms A,B,C`
Expected: three `wrote /data/results/curves/gate_*.npz and gate_*.json` lines. In the printed per-estimator metric dicts, `gate_platt2`'s `auc_roc` must equal `gate_raw`'s to ~1e-12 for every arm. If it does not, some Platt slope is non-positive and Task 2's guard should already have reported it.

- [ ] **Step 2: Download the bundles**

```bash
modal volume get surp-acts-data results/curves results/ --force
```
Expected: `results/curves/gate_{A,B,C}.npz` and `.json` present locally.

- [ ] **Step 3: Build the figures**

Run: `PYTHONPATH=. python scripts/plot_gate_figures.py --bundle-dir results/curves --out-dir figures/gate`
Expected: six `wrote F*` lines. Any `WARNING arm C: Platt-4 slope <= 0` line is a real finding, not a script error.

- [ ] **Step 4: Record the new numbers in `experiments/LOG.md`**

Append a `## 2026-08-17 — Gate figure set and widened decision region` section containing:
- The DR-ECE table under `[0.01, 0.99]` for all four estimators × three arms, **beside** the `[0.01, 0.5]` numbers already logged, so the region change is auditable rather than a silent restatement.
- The arm C Platt-4 slope-inversion result: crossing occupancy and the fraction of val rows above it.
- Confirmation that AUC-ROC and AUC-PR are unchanged by Platt-2, as predicted.

- [ ] **Step 5: Correct the AUC-PR entry in the earlier log section**

The 2026-08-17 gate-training section currently says the AUC-PR red flag "was miscalibrated and is retired". That is wrong: spec §9.5 sets the 0.85–0.95 expectation, so it is the spec's number, not a threshold introduced here. Replace that bullet with:

```markdown
- **AUC-PR came in under the spec's expected range.** Spec §9.5 expects
  0.85–0.95; arm A reaches 0.7987 and arm B 0.7710. This is a missed
  expectation, not a debug-before-proceeding gate — §9.5's only hard warning
  is AUC-ROC < 0.95, which all arms clear (A 0.9931). Arm A carries no
  subsampling and no distribution shift, so the shortfall is a property of
  the feature set and architecture rather than of the sampling strategy, and
  should be reported as such rather than treated as a bad threshold. For
  scale, a no-skill classifier scores 0.0057 at this base rate.
```

- [ ] **Step 6: Commit**

```bash
git add experiments/LOG.md figures/gate results/curves
git commit -m "Add gate figure set results and correct the AUC-PR log entry

Records DR-ECE under both the new [0.01, 0.99] region and the old [0.01, 0.5]
view, the arm C Platt-4 slope-inversion finding, and the confirmed AUC
invariance under 2-param Platt. Corrects the earlier claim that the AUC-PR
threshold was locally miscalibrated -- 0.85-0.95 is spec 9.5's expectation,
and we came in under it."
```

---

## Self-Review

**Spec coverage.** Five requests. F1 covers training loss curves; F2/F3 post-training ROC and PR; F4 calibration loss curves; F6 before/after deltas for all five named metrics; F5 the reliability plots. The `[0.01, 0.99]` DR-ECE change is Task 1 and flows into F6's axis label and the log entry. All covered.

**Placeholder scan.** No TBDs. Every code step carries literal code. The one prose step (Task 6 Step 4) names exactly the three items the log section must contain.

**Type consistency.** `grid_curves` returns `threshold_logit`/`tpr`/`fpr`/`precision`/`tp`/`fp`; Task 4 stores them as `f"{name}__{key}"`, which is exactly what Task 5's `_reliability_from_bundle` and the curve figures read. `metric_bundle`'s nine keys are the keys F6 indexes and F5 quotes. `platt_occupancy_slope_violations`'s five keys are those Task 4 stores and Task 5's warning reads.

**One known weakness, stated rather than hidden.** `_reliability_from_bundle` reconstructs per-bin counts from the 4,001-point cumulative grid instead of raw scores, so it is exact only where a reliability bin edge coincides with a grid edge and off by at most one grid step (0.01 in log-odds) elsewhere. That is fine for a diagram and wrong for a headline number — which is why F5's quoted ECE/MCE come from `metric_bundle` (full-data, Task 4) and never from the reconstruction. If the drawn points ever visibly disagree with the quoted ECE, ship raw val scores and recompute properly instead.
