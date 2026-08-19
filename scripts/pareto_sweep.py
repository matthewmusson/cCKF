#!/usr/bin/env python3
"""Post-process a Pareto sweep CSV: identify the non-dominated front and plot it.

Task 10 (cCKF ACTS integration) — the sweep itself runs on Modal via
``modal run --detach modal_build_acts.py::pareto_sweep`` in the ``cCKF``
repo root, which writes ``/data/results/pareto_sweep.csv`` on the
``surp-acts-data`` volume with columns::

    tau_g, tau_v, efficiency, fake_rate, duplicate_rate_pre_ambi,
    duplicate_rate_post_ambi, runtime_per_event_s, gate_calls, value_calls,
    wall_seconds

Fetch that CSV locally first, e.g.::

    modal volume get surp-acts-data results/pareto_sweep.csv .

This script is the local-side reader: no Modal dependency, safe to run
against a plain CSV file. It identifies the Pareto front (non-dominated
points), prints a summary table, and optionally writes a scatter plot.

Usage
-----
    python scripts/pareto_sweep.py --csv results/pareto_sweep.csv
    python scripts/pareto_sweep.py --csv results/pareto_sweep.csv \\
        --include-runtime --plot figures/pareto_front.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Read the sweep CSV, coercing numeric columns and dropping failed rows.

    A grid point that failed on Modal writes ``None``/empty for its metric
    columns (see ``_extract_sweep_row`` in ``modal_build_acts.py``); those
    rows are kept for the printed summary but excluded from the Pareto front.
    """
    numeric_cols = [
        "tau_g", "tau_v", "efficiency", "fake_rate",
        "duplicate_rate_pre_ambi", "duplicate_rate_post_ambi",
        "runtime_per_event_s", "gate_calls", "value_calls", "wall_seconds",
    ]
    rows = []
    with open(csv_path, newline="") as f:
        for raw in csv.DictReader(f):
            row: dict[str, Any] = dict(raw)
            for col in numeric_cols:
                v = row.get(col)
                if v is None or v == "":
                    row[col] = None
                else:
                    try:
                        row[col] = float(v)
                    except ValueError:
                        row[col] = None
            rows.append(row)
    return rows


def is_complete(row: dict[str, Any], *, include_runtime: bool) -> bool:
    keys = ["efficiency", "fake_rate"] + (["runtime_per_event_s"] if include_runtime else [])
    return all(row.get(k) is not None for k in keys)


def pareto_front(
    rows: list[dict[str, Any]], *, include_runtime: bool = False
) -> list[dict[str, Any]]:
    """Non-dominated points on (efficiency MAX, fake_rate MIN[, runtime MIN]).

    A row is dominated if some other complete row is at least as good on
    every objective and strictly better on at least one.
    """
    complete = [r for r in rows if is_complete(r, include_runtime=include_runtime)]

    def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        better_or_equal = (
            a["efficiency"] >= b["efficiency"] and a["fake_rate"] <= b["fake_rate"]
        )
        strictly_better = a["efficiency"] > b["efficiency"] or a["fake_rate"] < b["fake_rate"]
        if include_runtime:
            better_or_equal = better_or_equal and a["runtime_per_event_s"] <= b["runtime_per_event_s"]
            strictly_better = strictly_better or a["runtime_per_event_s"] < b["runtime_per_event_s"]
        return better_or_equal and strictly_better

    front = [
        r for r in complete
        if not any(dominates(other, r) for other in complete if other is not r)
    ]
    front.sort(key=lambda r: -r["efficiency"])
    return front


def print_summary(rows: list[dict[str, Any]], front: list[dict[str, Any]], *, include_runtime: bool) -> None:
    n_total = len(rows)
    n_complete = len([r for r in rows if is_complete(r, include_runtime=include_runtime)])
    n_failed = n_total - n_complete
    print(f"Grid points: {n_total}  complete: {n_complete}  failed/incomplete: {n_failed}")
    print(f"Pareto front size: {len(front)}\n")

    header = f"{'tau_g':>7} {'tau_v':>7} {'eff%':>8} {'fake%':>8} {'dup_pre%':>9} {'dup_post%':>10} {'t/evt(s)':>9} {'wall(s)':>8}"
    print(header)
    print("-" * len(header))
    for r in front:
        print(
            f"{r['tau_g']:>7.3f} {r['tau_v']:>7.3f} "
            f"{r['efficiency']:>8.3f} {r['fake_rate']:>8.4f} "
            f"{(r.get('duplicate_rate_pre_ambi') or float('nan')):>9.4f} "
            f"{(r.get('duplicate_rate_post_ambi') or float('nan')):>10.4f} "
            f"{(r.get('runtime_per_event_s') or float('nan')):>9.3f} "
            f"{(r.get('wall_seconds') or float('nan')):>8.1f}"
        )


def make_plot(rows: list[dict[str, Any]], front: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complete = [r for r in rows if r.get("efficiency") is not None and r.get("fake_rate") is not None]
    front_keys = {(r["tau_g"], r["tau_v"]) for r in front}

    fig, ax = plt.subplots(figsize=(6, 5))
    dominated = [r for r in complete if (r["tau_g"], r["tau_v"]) not in front_keys]
    if dominated:
        ax.scatter(
            [r["fake_rate"] for r in dominated],
            [r["efficiency"] for r in dominated],
            c="lightgray", label="dominated", s=30,
        )
    front_sorted = sorted(front, key=lambda r: r["fake_rate"])
    ax.plot(
        [r["fake_rate"] for r in front_sorted],
        [r["efficiency"] for r in front_sorted],
        "o-", c="crimson", label="Pareto front",
    )
    ax.set_xlabel("Fake rate (%, DM, track-level)")
    ax.set_ylabel("Efficiency (%, DM, particle-level)")
    ax.set_title(r"cCKF Pareto front: $\tau_g \times \tau_v$ sweep")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--csv", type=Path, default=Path("results/pareto_sweep.csv"),
        help="Path to the sweep CSV produced by modal_build_acts.py::pareto_sweep.",
    )
    parser.add_argument(
        "--include-runtime", action="store_true",
        help="Also require runtime_per_event_s to be non-dominated (3-objective front). "
        "Default is the 2-objective (efficiency, fake_rate) front.",
    )
    parser.add_argument(
        "--plot", type=Path, default=None,
        help="Optional path to write a matplotlib scatter plot (fake_rate vs efficiency) to.",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"error: --csv {args.csv} not found", file=sys.stderr)
        raise SystemExit(1)

    rows = load_rows(args.csv)
    if not rows:
        print(f"error: {args.csv} has no data rows", file=sys.stderr)
        raise SystemExit(1)

    front = pareto_front(rows, include_runtime=args.include_runtime)
    print_summary(rows, front, include_runtime=args.include_runtime)

    if args.plot is not None:
        make_plot(rows, front, args.plot)


if __name__ == "__main__":
    main()
