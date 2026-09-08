# Window Failure Rate Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce two diagnostic plots showing how often the CKF's Mahalanobis search window fails to contain the truth particle's measurement, as a function of window multiplier n and pseudorapidity η.

**Architecture:** Load pilot data (trackstates ROOT + truth-chain CSVs) on Modal, compute per-state normalized distances d₀ = |true_l₀ − pred_l₀| / √S₀₀ and d₁ = |true_l₁ − pred_l₁| / √S₁₁ for every branch with a majority particle, aggregate failure rates, and produce publication-quality plots. Analysis functions are pure Python (no ACTS dependency); data loading reuses `expansion.py`.

**Tech Stack:** uproot, awkward, pandas, numpy, matplotlib (all on Modal)

## Background

The CKF searches for compatible hits inside a bounding-box window:

```
|local0 − pred_l0| ≤ n · √S₀₀   AND   |local1 − pred_l1| ≤ n · √S₁₁
```

where S = H·C_{k+1|k}·Hᵀ + R is the innovation covariance (includes measurement noise). A **window failure** occurs when the truth particle's measurement exists on the surface but falls outside this window.

The **normalized distance** d_max = max(|Δl₀|/√S₀₀, |Δl₁|/√S₁₁) characterizes how far the true hit is from the predicted center in units of √S. A window at threshold n fails iff d_max > n.

If the Kalman filter's covariance is well-calibrated, d₀ and d₁ are each marginally N(0,1), and the theoretical failure rate for the bounding-box window is:

```
P(fail | n) = 1 − [erf(n/√2)]²
```

Deviation from this curve indicates miscalibration of S (from unmodeled material, non-Gaussian tails, etc.).

**Plot 1 — Failure rate vs n:** Shows how quickly the window failure rate drops as n increases. Compare to the theoretical χ² curve.

**Plot 2 — Failure rate vs η at fixed n:** Shows where in the detector the window is most likely to fail. Expected to correlate with passive material X₀ vs η (peaks at |η| ≈ 1.5, 3.1 in the ODD).

## Global Constraints

- Events [0, 32) for train/cal, events [32, 64) are **SEALED** — never opened.
- Pilot data: 2 events (0, 1) at `/data/results/pilot_1786472672` on Modal volume `surp-acts-data`.
- Reuse `expansion.py` functions for data loading, geometry ID encoding, particle ID encoding, and majority-particle computation — do **not** reimplement these.
- Python 3.10+, Black formatting, 88 char line width.

---

### Task 1: Window failure analysis functions

**Files:**
- Create: `cCKF/window_failure.py`

**Interfaces:**
- Consumes: `expansion.load_trackstates()`, `expansion.load_measurements()`, `expansion.load_simhits()`, `expansion.load_measurement_simhit_map()`, `expansion.compute_branch_majority_pid()`, `expansion.encode_geometry_id()`
- Produces: `build_truth_measurement_lookup(measurements, simhits, meas_map) -> pd.DataFrame`, `compute_normalized_distances(states, truth_lookup, majority) -> pd.DataFrame`

- [ ] **Step 1: Write `build_truth_measurement_lookup`**

This function joins the truth chain to produce a table of (geometry_id, particle_id, local0, local1) — i.e., for every (surface, particle) pair, where is that particle's measurement?

```python
"""Window failure rate analysis for the cCKF pilot.

Computes normalized residual distances between the Kalman filter's
predicted state and the truth particle's actual measurement, for every
(branch, surface) pair where the branch has a majority particle and
that particle left a measurement on the surface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import erf

from expansion import (
    compute_branch_majority_pid,
    load_measurement_simhit_map,
    load_measurements,
    load_simhits,
    load_trackstates,
)


def build_truth_measurement_lookup(
    measurements: pd.DataFrame,
    simhits: pd.DataFrame,
    meas_map: pd.DataFrame,
) -> pd.DataFrame:
    """Build a (geometry_id, particle_id) → (local0, local1) lookup.

    Joins the truth chain: measurement-simhit-map → simhits → measurements.
    Returns one row per unique (measurement_id, particle_id), with the
    measurement's geometry_id and local coordinates.

    Parameters
    ----------
    measurements : pd.DataFrame
        From ``expansion.load_measurements``. Columns: measurement_id,
        geometry_id, local0, local1, var_local0, var_local1.
    simhits : pd.DataFrame
        From ``expansion.load_simhits``. Columns: hit_id, geometry_id,
        particle_id, tx, ty, tz.
    meas_map : pd.DataFrame
        From ``expansion.load_measurement_simhit_map``. Columns:
        measurement_id, hit_id.

    Returns
    -------
    pd.DataFrame
        Columns: measurement_id, particle_id, geometry_id, local0, local1.
        One row per unique (measurement_id, particle_id) pair.
    """
    # measurement_id → hit_id → particle_id
    joined = meas_map.merge(
        simhits[["hit_id", "particle_id"]], on="hit_id", how="inner"
    )
    # Deduplicate: one measurement can have multiple simhits from the same
    # particle (e.g. overlapping energy deposits). Keep unique pairs.
    joined = joined[["measurement_id", "particle_id"]].drop_duplicates()

    # Attach measurement coordinates
    joined = joined.merge(
        measurements[["measurement_id", "geometry_id", "local0", "local1"]],
        on="measurement_id",
        how="inner",
    )
    return joined
```

- [ ] **Step 2: Write `compute_normalized_distances`**

For each (branch, surface) with a majority particle, find the majority particle's true measurement on that surface and compute the normalized distance.

```python
def compute_normalized_distances(
    states: pd.DataFrame,
    truth_lookup: pd.DataFrame,
    majority: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-state normalized distances to the truth measurement.

    For each trackstate on a branch with a well-defined majority particle,
    finds the majority particle's measurement on the same surface and
    computes:

        d₀ = |true_l₀ − pred_l₀| / √S₀₀
        d₁ = |true_l₁ − pred_l₁| / √S₁₁
        d_max = max(d₀, d₁)

    States where the majority particle has no measurement on the surface
    (genuine holes) are excluded.

    Parameters
    ----------
    states : pd.DataFrame
        From ``expansion.load_trackstates``. Must include: geometry_id,
        pred_l0, pred_l1, S00, S11, eta, volume_id, event_id, track_nr,
        is_predicted.
    truth_lookup : pd.DataFrame
        From ``build_truth_measurement_lookup``. Columns: measurement_id,
        particle_id, geometry_id, local0, local1.
    majority : pd.DataFrame
        From ``expansion.compute_branch_majority_pid``. Columns: event_id,
        track_nr, branch_majority_pid, majority_undefined.

    Returns
    -------
    pd.DataFrame
        Columns: event_id, track_nr, state_idx, eta, volume_id, d0, d1,
        d_max. One row per (state, true measurement) pair that passed
        all filters. If a majority particle has multiple measurements on
        the same surface, only the closest (min d_max) is kept.
    """
    # Filter to predicted states with valid S
    df = states[
        states["is_predicted"]
        & states["S00"].notna()
        & (states["S00"] > 0)
        & states["S11"].notna()
        & (states["S11"] > 0)
    ].copy()

    # Attach majority particle
    df = df.merge(majority, on=["event_id", "track_nr"], how="inner")
    df = df[~df["majority_undefined"]]

    # Join with truth lookup on (geometry_id, majority_pid = particle_id)
    merged = df.merge(
        truth_lookup.rename(columns={
            "local0": "true_l0",
            "local1": "true_l1",
            "particle_id": "truth_pid",
        }),
        left_on=["geometry_id", "branch_majority_pid"],
        right_on=["geometry_id", "truth_pid"],
        how="inner",
    )

    # Compute normalized distances
    merged["d0"] = np.abs(merged["true_l0"] - merged["pred_l0"]) / np.sqrt(merged["S00"])
    merged["d1"] = np.abs(merged["true_l1"] - merged["pred_l1"]) / np.sqrt(merged["S11"])
    merged["d_max"] = np.maximum(merged["d0"], merged["d1"])

    # If multiple true measurements on same surface for same particle,
    # keep closest (min d_max)
    out = merged.sort_values("d_max").drop_duplicates(
        subset=["event_id", "track_nr", "state_idx"], keep="first"
    )

    return out[["event_id", "track_nr", "state_idx", "eta", "volume_id",
                "d0", "d1", "d_max"]].reset_index(drop=True)
```

- [ ] **Step 3: Write `run_analysis` orchestrator**

Loads all data for both events and returns the combined normalized-distance table.

```python
def run_analysis(
    root_path: str,
    csv_dir: str,
    event_ids: list[int] = (0, 1),
) -> pd.DataFrame:
    """Run the full window failure analysis on pilot data.

    Parameters
    ----------
    root_path : str
        Path to ``trackstates_ckf.root``.
    csv_dir : str
        Directory containing per-event CSVs.
    event_ids : list[int]
        Events to process.

    Returns
    -------
    pd.DataFrame
        Combined normalized-distance table with columns: event_id,
        track_nr, state_idx, eta, volume_id, d0, d1, d_max.
    """
    all_results = []

    for eid in event_ids:
        print(f"Processing event {eid}...")
        states = load_trackstates(root_path, eid)
        measurements = load_measurements(csv_dir, eid)
        simhits = load_simhits(csv_dir, eid)
        meas_map = load_measurement_simhit_map(csv_dir, eid)

        majority = compute_branch_majority_pid(states)
        truth_lookup = build_truth_measurement_lookup(measurements, simhits, meas_map)
        distances = compute_normalized_distances(states, truth_lookup, majority)
        all_results.append(distances)

    combined = pd.concat(all_results, ignore_index=True)
    print(
        f"Total states with truth hits: {len(combined)}, "
        f"median d_max: {combined['d_max'].median():.3f}"
    )
    return combined
```

- [ ] **Step 4: Commit**

```bash
git add cCKF/window_failure.py
git commit -m "feat: add window failure rate analysis functions"
```

---

### Task 2: Plotting functions

**Files:**
- Modify: `cCKF/window_failure.py` (append plotting functions)

**Interfaces:**
- Consumes: `compute_normalized_distances()` output (DataFrame with d_max, eta, volume_id columns)
- Produces: `plot_failure_vs_n(distances, output_path)`, `plot_failure_vs_eta(distances, output_path)`

- [ ] **Step 1: Write `plot_failure_vs_n`**

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PIXEL_VOLUMES = {16, 17, 18}


def plot_failure_vs_n(
    distances: pd.DataFrame,
    output_path: str,
    n_range: tuple[float, float] = (0.5, 20.0),
    n_points: int = 200,
) -> None:
    """Plot window failure rate vs n with theoretical comparison.

    Produces a semilog plot showing the fraction of truth hits that fall
    outside the bounding-box window as a function of the window multiplier n.
    Includes the theoretical curve 1 − [erf(n/√2)]² for comparison.

    Three curves: all sensors, pixel only, strip only.

    Parameters
    ----------
    distances : pd.DataFrame
        Output of ``compute_normalized_distances``. Must have columns:
        d_max, volume_id.
    output_path : str
        Where to save the PNG.
    """
    n_vals = np.linspace(n_range[0], n_range[1], n_points)

    d_all = distances["d_max"].to_numpy()
    is_pixel = distances["volume_id"].isin(PIXEL_VOLUMES).to_numpy()
    d_pixel = d_all[is_pixel]
    d_strip = d_all[~is_pixel]

    rate_all = np.array([np.mean(d_all > n) for n in n_vals])
    rate_pixel = np.array([np.mean(d_pixel > n) for n in n_vals])
    rate_strip = np.array([np.mean(d_strip > n) for n in n_vals])

    # Theoretical: independent N(0,1) marginals in a box window
    rate_theory = 1.0 - erf(n_vals / np.sqrt(2.0)) ** 2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(n_vals, rate_all, "k-", linewidth=2, label="All sensors")
    ax.semilogy(n_vals, rate_pixel, "r-", linewidth=1.5, label="Pixel")
    ax.semilogy(n_vals, rate_strip, "b-", linewidth=1.5, label="Strip")
    ax.semilogy(
        n_vals, rate_theory, "k--", linewidth=1, alpha=0.6,
        label=r"Theory: $1 - \mathrm{erf}(n/\sqrt{2})^2$",
    )

    ax.set_xlabel("Window multiplier $n$", fontsize=13)
    ax.set_ylabel("Window failure rate", fontsize=13)
    ax.set_title(
        "Fraction of truth hits outside bounding-box window vs $n$\n"
        "(ODD, ttbar $\\mu$=200, envelope CKF, 2 events)",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.set_xlim(n_range)
    ax.set_ylim(bottom=1e-5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
```

- [ ] **Step 2: Write `plot_failure_vs_eta`**

```python
def plot_failure_vs_eta(
    distances: pd.DataFrame,
    output_path: str,
    n_values: list[float] | None = None,
    eta_range: tuple[float, float] = (-4.0, 4.0),
    n_eta_bins: int = 40,
    min_count: int = 50,
) -> None:
    """Plot window failure rate vs η for multiple n values.

    Produces 10 curves (one per n) on the same axes, showing how the
    failure rate varies with pseudorapidity. Designed for comparison
    with the ODD passive material X₀ vs η plot.

    Parameters
    ----------
    distances : pd.DataFrame
        Output of ``compute_normalized_distances``. Must have columns:
        d_max, eta.
    output_path : str
        Where to save the PNG.
    n_values : list[float], optional
        Window multipliers to plot. Default: [2, 3, 4, 5, 6, 7, 8, 10, 12, 15].
    eta_range : tuple
        η axis limits.
    n_eta_bins : int
        Number of η bins.
    min_count : int
        Minimum states per η bin to plot a point (suppress noise).
    """
    if n_values is None:
        n_values = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]

    eta = distances["eta"].to_numpy()
    d_max = distances["d_max"].to_numpy()

    eta_edges = np.linspace(eta_range[0], eta_range[1], n_eta_bins + 1)
    eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    bin_idx = np.digitize(eta, eta_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_eta_bins - 1)

    cmap = plt.cm.viridis
    colors = [cmap(i / (len(n_values) - 1)) for i in range(len(n_values))]

    fig, ax = plt.subplots(figsize=(10, 6))

    for j, n in enumerate(n_values):
        rates = np.full(n_eta_bins, np.nan)
        for i in range(n_eta_bins):
            mask = bin_idx == i
            count = mask.sum()
            if count >= min_count:
                rates[i] = np.mean(d_max[mask] > n)

        valid = ~np.isnan(rates)
        ax.plot(
            eta_centers[valid], rates[valid],
            color=colors[j], linewidth=1.5, marker=".", markersize=4,
            label=f"$n = {n}$",
        )

    ax.set_xlabel("$\\eta$", fontsize=14)
    ax.set_ylabel("Window failure rate", fontsize=13)
    ax.set_title(
        "Window failure rate vs $\\eta$ at fixed $n$\n"
        "(ODD, ttbar $\\mu$=200, envelope CKF, 2 events)",
        fontsize=12,
    )
    ax.legend(fontsize=9, ncol=2, loc="upper center")
    ax.set_xlim(eta_range)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
```

- [ ] **Step 3: Commit**

```bash
git add cCKF/window_failure.py
git commit -m "feat: add window failure rate plots (vs n and vs eta)"
```

---

### Task 3: Modal function + run + download plots

**Files:**
- Modify: `cCKF/modal_build_acts.py` — add `matplotlib` to image pip_install, add `run_window_failure_analysis()` function, add `window_failure.py` to image

**Interfaces:**
- Consumes: `window_failure.run_analysis()`, `window_failure.plot_failure_vs_n()`, `window_failure.plot_failure_vs_eta()`
- Produces: Two PNG files saved to Modal volume at `/data/results/pilot_1786472672/plots/`

- [ ] **Step 1: Add matplotlib and window_failure.py to Modal image**

In `modal_build_acts.py`, modify the image definition:

```python
# Add "matplotlib" to the existing pip_install list:
.pip_install("uproot", "awkward", "pyyaml", "pandas", "pyarrow", "numpy", "jinja2", "matplotlib", "scipy")

# Add window_failure.py to the image:
.add_local_file("window_failure.py", remote_path="/app/window_failure.py")
```

Also add `expansion.py` to the image (window_failure.py imports from it):

```python
.add_local_file("expansion.py", remote_path="/app/expansion.py")
```

- [ ] **Step 2: Add `run_window_failure_analysis` Modal function**

```python
@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol, DATA_PATH: data_vol},
    timeout=600,
)
def run_window_failure_analysis():
    """Run window failure rate analysis on pilot data and save plots."""
    import sys
    sys.path.insert(0, "/app")
    from window_failure import run_analysis, plot_failure_vs_n, plot_failure_vs_eta

    data_vol.reload()
    pilot_dir = f"{DATA_PATH}/results/pilot_1786472672"
    root_path = f"{pilot_dir}/trackstates_ckf.root"
    csv_dir = pilot_dir

    # Run analysis
    distances = run_analysis(root_path, csv_dir, event_ids=[0, 1])

    # Save intermediate table
    import os
    plot_dir = f"{pilot_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)

    distances.to_parquet(f"{plot_dir}/window_distances.parquet", index=False)

    # Produce plots
    plot_failure_vs_n(distances, f"{plot_dir}/failure_rate_vs_n.png")
    plot_failure_vs_eta(distances, f"{plot_dir}/failure_rate_vs_eta.png")

    # Summary stats
    n_total = len(distances)
    n_fail_5 = (distances["d_max"] > 5).sum()
    n_fail_10 = (distances["d_max"] > 10).sum()
    summary = {
        "total_states_with_truth": n_total,
        "failure_rate_n5": float(n_fail_5 / n_total) if n_total > 0 else 0,
        "failure_rate_n10": float(n_fail_10 / n_total) if n_total > 0 else 0,
        "median_d_max": float(distances["d_max"].median()),
        "p99_d_max": float(distances["d_max"].quantile(0.99)),
    }
    print(f"\nSummary: {summary}")

    data_vol.commit()
    return summary
```

- [ ] **Step 3: Run on Modal**

```bash
cd /Users/matthewm/SURP/cCKF
modal run modal_build_acts.py::run_window_failure_analysis
```

Expected output: summary dict with failure rates. Plots saved to volume.

- [ ] **Step 4: Download plots**

Add a small helper or use `modal volume get` to retrieve the plots:

```bash
modal volume get surp-acts-data results/pilot_1786472672/plots/failure_rate_vs_n.png ./failure_rate_vs_n.png
modal volume get surp-acts-data results/pilot_1786472672/plots/failure_rate_vs_eta.png ./failure_rate_vs_eta.png
```

- [ ] **Step 5: Verify plots**

Open both PNGs and verify:
1. `failure_rate_vs_n.png`: should show failure rate dropping from ~50% at n=1 to <0.1% by n=10. The data curves should be ABOVE the theoretical curve (non-Gaussian tails from material). Pixel and strip curves may separate.
2. `failure_rate_vs_eta.png`: should show peaks at |η| ≈ 1.5 and |η| ≈ 3.1, correlating with the ODD passive material distribution. Higher n curves should be lower but with the same η shape.

If the data curve in Plot 1 is BELOW the theory curve, something is wrong (either S is over-estimated or the truth chain is incorrect). If Plot 2 shows no η dependence, the window failures may be dominated by something other than material (e.g., seed quality).

- [ ] **Step 6: Commit**

```bash
git add cCKF/modal_build_acts.py
git commit -m "feat: add Modal function for window failure rate analysis"
```
