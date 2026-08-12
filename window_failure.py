"""Window failure rate analysis for the cCKF pilot.

Computes normalized residual distances between the Kalman filter's
predicted state and the truth particle's actual measurement, for every
(branch, surface) pair where the branch has a majority particle and
that particle left a measurement on the surface.

Two diagnostic plots:
1. Failure rate vs n — how quickly window failures drop with n,
   compared to the theoretical 1 − [erf(n/√2)]² for calibrated S.
2. Failure rate vs η at fixed n — spatial structure of failures,
   expected to correlate with passive material X₀ vs η.

Uses a fast vectorized loader (awkward arrays, no Python loops) instead
of expansion.load_trackstates, which has a per-row loop that's too slow
for ~1M states.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erf

PIXEL_VOLUMES = {16, 17, 18}

_VOLUME_SHIFT = 56
_LAYER_SHIFT = 36
_SENSITIVE_SHIFT = 8


def _encode_geometry_id(
    volume_id: np.ndarray,
    layer_id: np.ndarray,
    module_id: np.ndarray,
) -> np.ndarray:
    v = np.asarray(volume_id, dtype=np.int64)
    l = np.asarray(layer_id, dtype=np.int64)
    m = np.asarray(module_id, dtype=np.int64)
    return (v << _VOLUME_SHIFT) | (l << _LAYER_SHIFT) | (m << _SENSITIVE_SHIFT)


def _encode_particle_id(pv, sv, part, gen, sub):
    a = np.asarray(pv, dtype=np.int64)
    b = np.asarray(sv, dtype=np.int64)
    c = np.asarray(part, dtype=np.int64)
    d = np.asarray(gen, dtype=np.int64)
    e = np.asarray(sub, dtype=np.int64)
    return (a << 48) | (b << 32) | (c << 16) | (d << 8) | e


# ---------------------------------------------------------------------------
# Fast data loading (vectorized, no per-row Python loops)
# ---------------------------------------------------------------------------


def load_states_fast(root_path: str, event_id: int) -> pd.DataFrame:
    """Load trackstates for one event — fast vectorized version.

    Reads only the branches needed for window failure analysis and
    computes geometry_id and particle IDs using numpy/awkward vectorized
    operations instead of the per-row Python loop in expansion.py.
    """
    import awkward as ak
    import uproot

    scalar_branches = [
        "volume_id", "layer_id", "module_id", "predicted",
        "eLOC0_prt", "eLOC1_prt", "S00_prt", "S01_prt", "S11_prt", "eta_prt",
        "pathInX0_interval",
    ]
    pid_branches = [
        "particle_ids_particle", "particle_ids_vertex_primary",
        "particle_ids_vertex_secondary", "particle_ids_generation",
        "particle_ids_sub_particle",
    ]

    with uproot.open(root_path) as f:
        tree = f["trackstates"]
        available = set(tree.keys())
        read_branches = [b for b in scalar_branches + pid_branches if b in available]
        if "event_nr" in available:
            read_branches.append("event_nr")

        arrays = tree.arrays(read_branches, library="ak")
        if "event_nr" in available:
            mask = ak.to_numpy(arrays["event_nr"]) == int(event_id)
            arrays = arrays[mask]

    n_tracks = len(arrays)
    if n_tracks == 0:
        return pd.DataFrame()

    n_states = ak.to_numpy(ak.num(arrays["volume_id"], axis=1))
    track_nr = np.repeat(np.arange(n_tracks, dtype=np.int64), n_states)
    state_idx = ak.to_numpy(
        ak.flatten(ak.local_index(arrays["volume_id"], axis=1))
    ).astype(np.int64)

    vol = ak.to_numpy(ak.flatten(arrays["volume_id"])).astype(np.int64)
    lay = ak.to_numpy(ak.flatten(arrays["layer_id"])).astype(np.int64)
    mod = ak.to_numpy(ak.flatten(arrays["module_id"])).astype(np.int64)

    out = pd.DataFrame({
        "event_id": event_id,
        "track_nr": track_nr,
        "state_idx": state_idx,
        "geometry_id": _encode_geometry_id(vol, lay, mod),
        "volume_id": vol,
        "is_predicted": ak.to_numpy(ak.flatten(arrays["predicted"])).astype(bool),
        "pred_l0": ak.to_numpy(ak.flatten(arrays["eLOC0_prt"])).astype(np.float64),
        "pred_l1": ak.to_numpy(ak.flatten(arrays["eLOC1_prt"])).astype(np.float64),
        "S00": ak.to_numpy(ak.flatten(arrays["S00_prt"])).astype(np.float64),
        "S01": ak.to_numpy(ak.flatten(arrays["S01_prt"])).astype(np.float64),
        "S11": ak.to_numpy(ak.flatten(arrays["S11_prt"])).astype(np.float64),
        "eta": ak.to_numpy(ak.flatten(arrays["eta_prt"])).astype(np.float64),
        "pathInX0": (
            ak.to_numpy(ak.flatten(arrays["pathInX0_interval"])).astype(np.float64)
            if "pathInX0_interval" in available else np.full(len(track_nr), np.nan)
        ),
    })

    # Vectorized primary particle ID per state.
    # particle_ids_* are jagged: [tracks][states][contributors].
    # Take the first contributor per state (the primary for single-hit states,
    # an arbitrary contributor for merged clusters — fine for majority vote).
    has_pids = all(b in available for b in pid_branches)
    if has_pids:
        part = arrays["particle_ids_particle"]   # [tracks][states][contribs]
        pv = arrays["particle_ids_vertex_primary"]
        sv = arrays["particle_ids_vertex_secondary"]
        gen = arrays["particle_ids_generation"]
        sub = arrays["particle_ids_sub_particle"]

        # Number of contributors per state (0 for holes)
        n_contribs = ak.num(part, axis=2)  # [tracks][states]

        # Pad to at least 1 contributor, fill missing with -1
        part_p = ak.fill_none(ak.firsts(part, axis=2), -1)
        pv_p = ak.fill_none(ak.firsts(pv, axis=2), 0)
        sv_p = ak.fill_none(ak.firsts(sv, axis=2), 0)
        gen_p = ak.fill_none(ak.firsts(gen, axis=2), 0)
        sub_p = ak.fill_none(ak.firsts(sub, axis=2), 0)

        part_flat = ak.to_numpy(ak.flatten(part_p)).astype(np.int64)
        pv_flat = ak.to_numpy(ak.flatten(pv_p)).astype(np.int64)
        sv_flat = ak.to_numpy(ak.flatten(sv_p)).astype(np.int64)
        gen_flat = ak.to_numpy(ak.flatten(gen_p)).astype(np.int64)
        sub_flat = ak.to_numpy(ak.flatten(sub_p)).astype(np.int64)

        primary_pid = np.where(
            part_flat >= 0,
            _encode_particle_id(pv_flat, sv_flat, part_flat, gen_flat, sub_flat),
            -1,
        )
        has_hit = ak.to_numpy(ak.flatten(n_contribs)).astype(np.int64) > 0
        out["primary_pid"] = primary_pid
        out["has_hit"] = has_hit
    else:
        out["primary_pid"] = np.int64(-1)
        out["has_hit"] = False

    print(f"  Loaded {len(out)} states for event {event_id} ({n_tracks} tracks)")
    return out


def compute_majority_fast(
    states: pd.DataFrame,
    min_agreement: int = 2,
) -> pd.DataFrame:
    """Compute seed-majority particle ID per track — vectorized.

    Uses the first 3 states with hits (has_hit == True) per track.
    min_agreement=2 gives >=2/3 majority; min_agreement=3 gives pure triplets.
    """
    hits = states[states["has_hit"]].copy()
    hits = hits.sort_values(["event_id", "track_nr", "state_idx"])

    # Take first 3 hit-states per track
    hits["rank"] = hits.groupby(["event_id", "track_nr"]).cumcount()
    seed_hits = hits[hits["rank"] < 3][["event_id", "track_nr", "primary_pid"]]

    # Count occurrences of each PID in the seed
    counts = (
        seed_hits.groupby(["event_id", "track_nr", "primary_pid"])
        .size()
        .reset_index(name="n")
    )
    # Total seed hits per track
    totals = (
        seed_hits.groupby(["event_id", "track_nr"])
        .size()
        .reset_index(name="total")
    )
    counts = counts.merge(totals, on=["event_id", "track_nr"])

    # Best PID per track (highest count)
    best_idx = counts.groupby(["event_id", "track_nr"])["n"].idxmax()
    best = counts.loc[best_idx].copy()

    best["majority_undefined"] = (best["n"] < min_agreement) | (best["total"] < 3)
    best = best.rename(columns={"primary_pid": "branch_majority_pid"})

    all_tracks = states[["event_id", "track_nr"]].drop_duplicates()
    result = all_tracks.merge(
        best[["event_id", "track_nr", "branch_majority_pid", "majority_undefined"]],
        on=["event_id", "track_nr"],
        how="left",
    )
    result["branch_majority_pid"] = result["branch_majority_pid"].fillna(-1).astype(np.int64)
    result["majority_undefined"] = result["majority_undefined"].fillna(True).astype(bool)

    n_def = (~result["majority_undefined"]).sum()
    print(f"  Majority defined for {n_def}/{len(result)} tracks")
    return result


def load_measurements_fast(csv_dir: str, event_id: int) -> pd.DataFrame:
    path = Path(csv_dir) / f"event{event_id:09d}-measurements.csv"
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    mid = (
        df["measurement_id"].to_numpy(dtype=np.int64)
        if "measurement_id" in df.columns
        else np.arange(len(df), dtype=np.int64)
    )
    return pd.DataFrame({
        "measurement_id": mid,
        "geometry_id": df["geometry_id"].to_numpy(dtype=np.int64),
        "local0": df["local0"].to_numpy(dtype=np.float64),
        "local1": df["local1"].to_numpy(dtype=np.float64),
    })


def load_simhits_fast(csv_dir: str, event_id: int) -> pd.DataFrame:
    path = Path(csv_dir) / f"event{event_id:09d}-simhits.csv"
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]

    def _col(name):
        return df[name].to_numpy(dtype=np.int64) if name in df.columns else np.zeros(len(df), dtype=np.int64)

    particle_id = _encode_particle_id(
        _col("particle_id_pv"), _col("particle_id_sv"),
        _col("particle_id_part"), _col("particle_id_gen"),
        _col("particle_id_subpart"),
    )
    return pd.DataFrame({
        "hit_id": np.arange(len(df), dtype=np.int64),
        "particle_id": particle_id,
    })


def load_meas_map_fast(csv_dir: str, event_id: int) -> pd.DataFrame:
    path = Path(csv_dir) / f"event{event_id:09d}-measurement-simhit-map.csv"
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    mcol = "measurement_id" if "measurement_id" in df.columns else df.columns[0]
    hcol = "hit_id" if "hit_id" in df.columns else df.columns[1]
    return pd.DataFrame({
        "measurement_id": df[mcol].to_numpy(dtype=np.int64),
        "hit_id": df[hcol].to_numpy(dtype=np.int64),
    })


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def build_truth_measurement_lookup(
    measurements: pd.DataFrame,
    simhits: pd.DataFrame,
    meas_map: pd.DataFrame,
) -> pd.DataFrame:
    """Build a (geometry_id, particle_id) -> (local0, local1) lookup.

    Joins the truth chain: measurement-simhit-map -> simhits -> measurements.
    Returns one row per unique (measurement_id, particle_id).
    """
    joined = meas_map.merge(
        simhits[["hit_id", "particle_id"]], on="hit_id", how="inner"
    )
    joined = joined[["measurement_id", "particle_id"]].drop_duplicates()
    joined = joined.merge(
        measurements[["measurement_id", "geometry_id", "local0", "local1"]],
        on="measurement_id",
        how="inner",
    )
    return joined


def compute_normalized_distances(
    states: pd.DataFrame,
    truth_lookup: pd.DataFrame,
    majority: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-state normalized distances to the truth measurement."""
    df = states[
        states["is_predicted"]
        & states["S00"].notna()
        & (states["S00"] > 0)
        & states["S11"].notna()
        & (states["S11"] > 0)
    ].copy()

    df = df.merge(majority, on=["event_id", "track_nr"], how="inner")
    df = df[~df["majority_undefined"]]

    # Cumulative material per branch (sum pathInX0 over all states up to this one)
    df = df.sort_values(["event_id", "track_nr", "state_idx"])
    df["cumX0"] = df.groupby(["event_id", "track_nr"])["pathInX0"].cumsum()

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

    r0 = (merged["true_l0"] - merged["pred_l0"]).to_numpy()
    r1 = (merged["true_l1"] - merged["pred_l1"]).to_numpy()
    s00 = merged["S00"].to_numpy()
    s01 = merged["S01"].fillna(0.0).to_numpy()
    s11 = merged["S11"].to_numpy()

    merged["d0"] = np.abs(r0) / np.sqrt(s00)
    merged["d1"] = np.abs(r1) / np.sqrt(s11)
    merged["d_max"] = np.maximum(merged["d0"], merged["d1"])

    # Full χ² = rᵀ S⁻¹ r using S₀₁
    det = s00 * s11 - s01 ** 2
    det_safe = np.where(np.abs(det) > 1e-30, det, 1e-30)
    merged["chi2"] = (s11 * r0**2 - 2 * s01 * r0 * r1 + s00 * r1**2) / det_safe

    out = merged.sort_values("d_max").drop_duplicates(
        subset=["event_id", "track_nr", "state_idx"], keep="first"
    )

    return out[
        [
            "event_id", "track_nr", "state_idx", "eta", "volume_id",
            "d0", "d1", "d_max", "chi2", "pathInX0", "cumX0",
        ]
    ].reset_index(drop=True)


def run_analysis(
    root_path: str,
    csv_dir: str,
    event_ids: list[int] = (0, 1),
    min_agreement: int = 2,
) -> pd.DataFrame:
    """Run the full window failure analysis on pilot data."""
    all_results = []

    for eid in event_ids:
        print(f"Processing event {eid}...")
        states = load_states_fast(root_path, eid)
        majority = compute_majority_fast(states, min_agreement=min_agreement)

        measurements = load_measurements_fast(csv_dir, eid)
        simhits = load_simhits_fast(csv_dir, eid)
        meas_map = load_meas_map_fast(csv_dir, eid)

        truth_lookup = build_truth_measurement_lookup(
            measurements, simhits, meas_map
        )
        distances = compute_normalized_distances(states, truth_lookup, majority)
        all_results.append(distances)
        print(f"  Event {eid}: {len(distances)} states with truth hits")

    combined = pd.concat(all_results, ignore_index=True)
    print(
        f"Total states with truth hits: {len(combined)}, "
        f"median d_max: {combined['d_max'].median():.3f}"
    )
    return combined


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_failure_vs_n(
    distances: pd.DataFrame,
    output_path: str,
    n_range: tuple[float, float] = (0.5, 20.0),
    n_points: int = 200,
    subtitle: str = "",
) -> None:
    """Plot window failure rate vs n with theoretical comparison.

    Semilog plot showing both the bounding-box cut (d_max > n) and the
    χ² elliptical cut (χ² > n²) failure rates as functions of the window
    multiplier n, each with its theoretical reference curve for
    calibrated S.

    Bounding-box theory: 1 - erf(n/√2)²
    χ² theory (2 dof): exp(-n²/2)  [i.e. 1 - CDF_χ²₂(n²)]
    """
    n_vals = np.linspace(n_range[0], n_range[1], n_points)

    d_all = distances["d_max"].to_numpy()
    chi2_all = distances["chi2"].to_numpy()
    is_pixel = distances["volume_id"].isin(PIXEL_VOLUMES).to_numpy()

    # Bounding-box failure rates
    rate_box_all = np.array([np.mean(d_all > n) for n in n_vals])
    rate_box_pixel = np.array([np.mean(d_all[is_pixel] > n) for n in n_vals])
    rate_box_strip = np.array([np.mean(d_all[~is_pixel] > n) for n in n_vals])

    # χ² elliptical failure rates: fail when χ² > n²
    rate_chi2_all = np.array([np.mean(chi2_all > n**2) for n in n_vals])
    rate_chi2_pixel = np.array([np.mean(chi2_all[is_pixel] > n**2) for n in n_vals])
    rate_chi2_strip = np.array([np.mean(chi2_all[~is_pixel] > n**2) for n in n_vals])

    # Theory curves
    theory_box = 1.0 - erf(n_vals / np.sqrt(2.0)) ** 2
    theory_chi2 = np.exp(-n_vals**2 / 2.0)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bounding-box curves (solid lines)
    ax.plot(n_vals, rate_box_all, "k-", linewidth=2, label="Box: all")
    ax.plot(n_vals, rate_box_pixel, "r-", linewidth=1.5, label="Box: pixel")
    ax.plot(n_vals, rate_box_strip, "b-", linewidth=1.5, label="Box: strip")

    # χ² curves (dashed lines, same colors)
    ax.plot(n_vals, rate_chi2_all, "k--", linewidth=2, label=r"$\chi^2$: all")
    ax.plot(n_vals, rate_chi2_pixel, "r--", linewidth=1.5, label=r"$\chi^2$: pixel")
    ax.plot(n_vals, rate_chi2_strip, "b--", linewidth=1.5, label=r"$\chi^2$: strip")

    # Theory reference curves (dotted, gray)
    ax.plot(
        n_vals, theory_box, color="gray", linestyle=":", linewidth=1.5,
        label=r"Theory (box): $1 - \mathrm{erf}(n/\sqrt{2})^2$",
    )
    ax.plot(
        n_vals, theory_chi2, color="gray", linestyle="-.", linewidth=1.5,
        label=r"Theory ($\chi^2_2$): $e^{-n^2/2}$",
    )

    ax.set_xlabel("Window multiplier $n$", fontsize=13)
    ax.set_ylabel("Window failure rate", fontsize=13)
    title_line2 = r"(ODD, $t\bar{t}$ $\mu$=200, envelope CKF, 2 events)"
    if subtitle:
        title_line2 = f"({subtitle}) — " + title_line2
    ax.set_title(
        "Truth-hit window failure rate vs $n$: bounding box vs $\\chi^2$ ellipse\n"
        + title_line2,
        fontsize=12,
    )
    ax.legend(fontsize=9, ncol=2)
    ax.set_xlim(n_range)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_failure_vs_eta(
    distances: pd.DataFrame,
    output_path: str,
    n_values: list[float] | None = None,
    eta_range: tuple[float, float] | None = None,
    n_eta_bins: int | None = None,
    n_bins: int | None = None,
    x_range: tuple[float, float] | None = None,
    min_count: int = 50,
    subtitle: str = "",
    eta_col: str = "eta",
    x_label: str | None = None,
    y_max: float | None = None,
) -> None:
    """Plot window failure rate vs a spatial variable, split pixel/strip.

    Two vertically stacked subplots (pixel on top, strip on bottom), each
    with 10 curves (one per n).
    """
    if n_values is None:
        n_values = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]

    xr = x_range or eta_range or (-4.0, 4.0)
    nb = n_bins or n_eta_bins or 40
    xl = x_label or r"$\eta$"
    var_name = xl.strip("$").strip()

    is_pixel = distances["volume_id"].isin(PIXEL_VOLUMES).to_numpy()
    subsets = [
        ("Pixel", distances[is_pixel]),
        ("Strip", distances[~is_pixel]),
    ]

    edges = np.linspace(xr[0], xr[1], nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    cmap = plt.cm.viridis
    colors = [cmap(i / (len(n_values) - 1)) for i in range(len(n_values))]

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    for ax, (label, sub) in zip(axes, subsets):
        x = sub[eta_col].to_numpy()
        d_max = sub["d_max"].to_numpy()
        bin_idx = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)

        for j, n in enumerate(n_values):
            rates = np.full(nb, np.nan)
            for i in range(nb):
                mask = bin_idx == i
                if mask.sum() >= min_count:
                    rates[i] = np.mean(d_max[mask] > n)

            valid = ~np.isnan(rates)
            ax.plot(
                centers[valid],
                rates[valid],
                color=colors[j],
                linewidth=1.5,
                marker=".",
                markersize=4,
                label=f"$n = {n}$",
            )

        ax.set_ylabel("Window failure rate", fontsize=12)
        ax.set_title(f"{label} — failure rate vs {xl}", fontsize=12)
        ax.legend(fontsize=8, ncol=5, loc="upper center")
        ax.set_xlim(xr)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel(xl, fontsize=14)
    title_line2 = r"(ODD, $t\bar{t}$ $\mu$=200, envelope CKF, 2 events)"
    if subtitle:
        title_line2 = f"({subtitle}) — " + title_line2
    fig.suptitle(
        f"Window failure rate vs {xl} at fixed $n$ — pixel vs strip"
        "\n" + title_line2,
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
