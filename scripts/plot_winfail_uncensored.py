"""Render the four uncensored window-failure / module-failure plot families
from `winfail_unc_event*.npz` tensors accumulated by `winfail_uncensored.py`.

Families (see task brief for the full spec):
  A. winfail_vs_eta_{pure,majority}.png   - 3 sensor panels, lines n=3/5/7/10
  B. winfail_vs_eta_occupancy_n{n}.png    - 3 sensor panels, lines = 5 occ strata
  C. modfail_vs_eta.png                   - 1 panel, 3 sensor lines
  D. modfail_vs_occupancy.png             - 1 panel, 3 sensor lines, errorbars
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

# One brief line: only the statistics needed to reconstruct the denominator.
FOOTER_WIN = ("all surviving branches (pre-ambi, envelope: chi2 16.26/35.75, "
              "cap 5, terminal cuts off) · majority defined · denom: majority "
              "simhit on surface · majority pT > {pt} GeV · uncensored "
              "(escaped = fail; incl. undigitized) · vol 20 excluded")
FOOTER_MOD = ("all surviving branches (pre-ambi, envelope: chi2 16.26/35.75, "
              "cap 5, terminal cuts off) · majority defined · majority pT > "
              "{pt} GeV · fail = no majority simhit on surface · vol 20 excluded")
SENSOR_LABELS = ["Pixel", "Short strip", "Long strip"]
OCC_LABELS = ["0-1", "2-4", "5-9", "10-19", "20+"]


def pt_slice(tensor: np.ndarray, threshold_gev: float) -> np.ndarray:
    """Sum the pT-bin axis (last axis) over bins at/above threshold_gev.

    threshold_gev must be an edge of PT_EDGES; raises ValueError otherwise
    so an unsupported threshold fails loudly.
    """
    from winfail_uncensored import PT_EDGES

    if threshold_gev not in PT_EDGES:
        raise ValueError(f"{threshold_gev} not in PT_EDGES {PT_EDGES}")
    i = PT_EDGES.index(threshold_gev)
    return tensor[..., i:].sum(axis=-1)


def _load_sums(npz_dir: str) -> dict[str, np.ndarray]:
    import glob

    files = sorted(glob.glob(f"{npz_dir}/winfail_unc_event*.npz"))
    assert files, f"no npz files in {npz_dir}"
    keys = ["mod_total", "mod_fail", "win_total", "win_fail"]
    acc: dict[str, np.ndarray | None] = {k: None for k in keys}
    for f in files:
        z = np.load(f)
        for k in keys:
            acc[k] = z[k] if acc[k] is None else acc[k] + z[k]
    z = np.load(files[0])
    acc["eta_bins"] = z["eta_bins"]
    acc["n_values"] = z["n_values"]
    return acc


def _rate_with_band(ax, x, k, n, label, color):
    from winfail_uncensored import wilson_interval

    with np.errstate(invalid="ignore", divide="ignore"):
        rate = k / n
    lo, hi = wilson_interval(k, n)
    ok = n > 0
    ax.plot(x[ok], rate[ok], "-", lw=1.2, color=color, label=label)
    ax.fill_between(x[ok], lo[ok], hi[ok], alpha=0.18, color=color, lw=0)


def _branch_class_suffix(branch_class: str) -> str:
    if branch_class == "ambi":
        return (" · ambi survivors (offline greedy replica, maxShared 3, "
                 "nMeasMin 7)")
    if branch_class == "all":
        return " · all CKF-output branches"
    raise ValueError(f"branch_class must be 'all' or 'ambi', got {branch_class!r}")


def _collapse_ambi(tensor: np.ndarray, branch_class: str) -> np.ndarray:
    """Collapse the trailing survived-ambi axis per branch_class."""
    if branch_class == "all":
        return tensor.sum(axis=-1)
    if branch_class == "ambi":
        return tensor[..., 1]
    raise ValueError(f"branch_class must be 'all' or 'ambi', got {branch_class!r}")


def _wrap_footer(text: str, width: int = 110) -> str:
    """Pack " · "-separated footer segments into lines <= width chars.

    Wrapping on the segment boundaries (rather than plain word-wrap) keeps
    each condition ("majority pT > 1.0 GeV", "vol 20 excluded", ...) intact
    on one line instead of splitting it mid-phrase.
    """
    parts = text.split(" · ")
    lines: list[str] = [parts[0]]
    for part in parts[1:]:
        candidate = f"{lines[-1]} · {part}"
        if len(candidate) <= width:
            lines[-1] = candidate
        else:
            lines.append(part)
    return "\n".join(lines)


def _footer(fig, template: str, threshold_gev: float, branch_class: str) -> None:
    fig.tight_layout()
    text = template.format(pt=threshold_gev) + _branch_class_suffix(branch_class)
    wrapped = _wrap_footer(text)
    fig.text(0.99, 0.005, wrapped, ha="right", va="bottom", fontsize=6.5,
              color="0.35", linespacing=1.3)
    # Reserve room below the axes so the (possibly multi-line) footer never
    # overlaps the x-axis label; tight_layout already set a sensible bottom,
    # this just pads it further, scaled by how many footer lines there are.
    n_lines = wrapped.count("\n") + 1
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + 0.03 * n_lines)


def _save(fig, out_dir: Path, name: str) -> str:
    path = out_dir / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _render_family_a(win_fail, win_total, x, n_values, purity_idx, name,
                      out_dir, threshold_gev, branch_class):
    """3 sensor panels; lines = the n values; one purity slice (pure/majority)."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_values)))
    for s, (ax, sensor_label) in enumerate(zip(axes, SENSOR_LABELS)):
        for ni, n in enumerate(n_values):
            k = win_fail[ni, :, s, purity_idx, :].sum(axis=-1)
            n_tot = win_total[:, s, purity_idx, :].sum(axis=-1)
            _rate_with_band(ax, x, k, n_tot, f"n = {int(n)}", colors[ni])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Window failure rate")
        ax.set_title(sensor_label, loc="left", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("η (branch state direction)")
    fig.suptitle(f"Window failure vs η, {name} branches (uncensored)")
    _footer(fig, FOOTER_WIN, threshold_gev, branch_class)
    return _save(fig, out_dir, f"winfail_vs_eta_{name}.png")


def _render_family_b(win_fail, win_total, x, n_values, ni,
                      out_dir, threshold_gev, branch_class):
    """3 sensor panels; lines = the 5 occupancy strata; one n slice (purities pooled)."""
    n = int(n_values[ni])
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(OCC_LABELS)))
    for s, (ax, sensor_label) in enumerate(zip(axes, SENSOR_LABELS)):
        for o, occ_label in enumerate(OCC_LABELS):
            k = win_fail[ni, :, s, :, o].sum(axis=1)
            n_tot = win_total[:, s, :, o].sum(axis=1)
            _rate_with_band(ax, x, k, n_tot, occ_label, colors[o])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Window failure rate")
        ax.set_title(sensor_label, loc="left", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, title="occupancy")
    axes[-1].set_xlabel("η (branch state direction)")
    fig.suptitle(
        f"Window failure vs η by occupancy, n = {n} (uncensored, purities pooled)"
    )
    _footer(fig, FOOTER_WIN, threshold_gev, branch_class)
    return _save(fig, out_dir, f"winfail_vs_eta_occupancy_n{n}.png")


def _render_family_c(mod_fail, mod_total, x, out_dir, threshold_gev, branch_class):
    """Single panel; 3 sensor lines, purities and occupancy pooled."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(SENSOR_LABELS)))
    for s, sensor_label in enumerate(SENSOR_LABELS):
        k = mod_fail[:, s].sum(axis=(1, 2))
        n_tot = mod_total[:, s].sum(axis=(1, 2))
        _rate_with_band(ax, x, k, n_tot, sensor_label, colors[s])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Module failure rate")
    ax.set_xlabel("η (branch state direction)")
    ax.set_title("Module failure vs η")
    ax.legend(loc="upper right", fontsize=8)
    _footer(fig, FOOTER_MOD, threshold_gev, branch_class)
    return _save(fig, out_dir, "modfail_vs_eta.png")


def _render_family_d(mod_fail, mod_total, out_dir, threshold_gev, branch_class):
    """Single panel; 3 sensor lines vs the 5 occupancy strata, Wilson errorbars."""
    from winfail_uncensored import wilson_interval

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(SENSOR_LABELS)))
    positions = np.arange(len(OCC_LABELS))
    offsets = np.linspace(-0.15, 0.15, len(SENSOR_LABELS))
    for s, sensor_label in enumerate(SENSOR_LABELS):
        k = mod_fail[:, s].sum(axis=(0, 1))
        n_tot = mod_total[:, s].sum(axis=(0, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = k / n_tot
        lo, hi = wilson_interval(k, n_tot)
        ok = n_tot > 0
        yerr = np.vstack([rate[ok] - lo[ok], hi[ok] - rate[ok]])
        ax.errorbar(
            positions[ok] + offsets[s], rate[ok], yerr=yerr, fmt="o",
            color=colors[s], label=sensor_label, capsize=3, markersize=5,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(OCC_LABELS)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Occupancy (n_window)")
    ax.set_ylabel("Module failure rate")
    ax.set_title("Module failure vs occupancy")
    ax.legend(loc="upper right", fontsize=8)
    _footer(fig, FOOTER_MOD, threshold_gev, branch_class)
    return _save(fig, out_dir, "modfail_vs_occupancy.png")


def render_all(
    npz_dir: str, out_dir: str, threshold_gev: float, branch_class: str
) -> list[str]:
    """Render all 8 figures for one (threshold_gev, branch_class) cell.

    Returns the list of written PNG paths.
    """
    if branch_class not in ("all", "ambi"):
        raise ValueError(f"branch_class must be 'all' or 'ambi', got {branch_class!r}")

    sums = _load_sums(npz_dir)
    eta_bins = sums["eta_bins"]
    n_values = sums["n_values"]
    x = 0.5 * (eta_bins[:-1] + eta_bins[1:])

    # Collapse the ambi axis first, then the pT axis (slicing rules, task brief).
    mod_total = pt_slice(_collapse_ambi(sums["mod_total"], branch_class), threshold_gev)
    mod_fail = pt_slice(_collapse_ambi(sums["mod_fail"], branch_class), threshold_gev)
    win_total = pt_slice(_collapse_ambi(sums["win_total"], branch_class), threshold_gev)
    win_fail = pt_slice(_collapse_ambi(sums["win_fail"], branch_class), threshold_gev)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    made = [
        _render_family_a(win_fail, win_total, x, n_values, 1, "pure",
                          out_path, threshold_gev, branch_class),
        _render_family_a(win_fail, win_total, x, n_values, 0, "majority",
                          out_path, threshold_gev, branch_class),
    ]
    for ni in range(len(n_values)):
        made.append(
            _render_family_b(win_fail, win_total, x, n_values, ni,
                              out_path, threshold_gev, branch_class)
        )
    made.append(_render_family_c(mod_fail, mod_total, x, out_path,
                                  threshold_gev, branch_class))
    made.append(_render_family_d(mod_fail, mod_total, out_path,
                                  threshold_gev, branch_class))
    return made


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Render uncensored winfail/modfail plot families from "
                     "accumulated npz tensors."
    )
    parser.add_argument("npz_dir", help="directory of winfail_unc_event*.npz files")
    parser.add_argument("out_base", help="base output directory")
    args = parser.parse_args()

    for threshold_gev in (1.0, 0.9):
        pt_tag = f"pt{threshold_gev}".replace(".", "p")
        for branch_class in ("all", "ambi"):
            out_dir = str(Path(args.out_base) / f"{pt_tag}_{branch_class}")
            made = render_all(args.npz_dir, out_dir, threshold_gev, branch_class)
            print(f"{out_dir}: wrote {len(made)} files")


if __name__ == "__main__":
    main()
