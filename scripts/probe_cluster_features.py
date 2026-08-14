"""Pilot check 1 — cluster features populated and non-degenerate (spec §6.6.1).

Reads ACTS ``cells.csv`` / ``measurements.csv`` produced by geometric (or
smearing) digitization and reports whether ``s_u``, ``s_v``, ``Q_tot``, and
second-moment proxies are present and non-degenerate.

Cluster size is reconstructed per measurement as:
  s_u = 1 + max(channel0) - min(channel0)
  s_v = 1 + max(channel1) - min(channel1)
  Q_tot = sum(cell.value)   # activation / charge
  σ_uu, σ_uv, σ_vv from charge-weighted channel second moments
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_event_cells(digi_dir: Path, event: int) -> pd.DataFrame | None:
    path = digi_dir / f"event{event:09d}-cells.csv"
    if not path.exists():
        # ACTS sometimes writes without zero-padding width consistency
        candidates = sorted(digi_dir.glob(f"event*-cells.csv"))
        for c in candidates:
            # eventNNNNNNNNN-cells.csv
            stem = c.name.split("-")[0]  # event000000000
            try:
                ev = int(stem.replace("event", ""))
            except ValueError:
                continue
            if ev == event:
                path = c
                break
        else:
            return None
    return pd.read_csv(path, comment="#")


def summarize_clusters(cells: pd.DataFrame) -> dict[str, Any]:
    if cells.empty:
        return {
            "n_cells": 0,
            "n_measurements": 0,
            "pass": False,
            "reason": "empty cells.csv",
        }

    required = {"measurement_id", "channel0", "channel1", "value"}
    missing = required - set(cells.columns)
    if missing:
        return {
            "n_cells": int(len(cells)),
            "pass": False,
            "reason": f"cells.csv missing columns: {sorted(missing)}",
            "columns": list(cells.columns),
        }

    rows = []
    for mid, g in cells.groupby("measurement_id"):
        ch0 = g["channel0"].to_numpy(dtype=np.float64)
        ch1 = g["channel1"].to_numpy(dtype=np.float64)
        q = g["value"].to_numpy(dtype=np.float64)
        s_u = 1.0 + float(ch0.max() - ch0.min()) if len(ch0) else 0.0
        s_v = 1.0 + float(ch1.max() - ch1.min()) if len(ch1) else 0.0
        q_tot = float(q.sum())
        if q_tot > 0:
            w = q / q_tot
            mu0 = float((w * ch0).sum())
            mu1 = float((w * ch1).sum())
            sig_uu = float((w * (ch0 - mu0) ** 2).sum())
            sig_vv = float((w * (ch1 - mu1) ** 2).sum())
            sig_uv = float((w * (ch0 - mu0) * (ch1 - mu1)).sum())
        else:
            sig_uu = sig_vv = sig_uv = float("nan")
        rows.append((s_u, s_v, q_tot, sig_uu, sig_uv, sig_vv, len(g)))

    arr = np.asarray(rows, dtype=np.float64)
    s_u, s_v, q_tot, sig_uu, sig_uv, sig_vv, n_ch = arr.T

    def _stats(x: np.ndarray) -> dict[str, float]:
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "frac_gt1": float("nan"),
            }
        return {
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "min": float(x.min()),
            "max": float(x.max()),
            "frac_gt1": float((x > 1.0).mean()),
        }

    # Degeneracy: all sizes == 1 and all charges identical ⇒ smearing-like.
    size_degenerate = bool(np.nanmax(s_u) <= 1.0 and np.nanmax(s_v) <= 1.0)
    charge_degenerate = bool(
        np.nanstd(q_tot) < 1e-12 and np.nanmean(q_tot) <= 0.0
    ) or bool(np.all(~np.isfinite(q_tot)))
    moments_degenerate = (
        bool(np.nanmax(sig_uu) == 0.0 and np.nanmax(sig_vv) == 0.0) and size_degenerate
    )

    n_meas = int(len(rows))
    n_with_cells = int((n_ch > 0).sum())
    frac_with_cells = float(n_with_cells / n_meas) if n_meas else 0.0

    passed = (
        n_meas > 0
        and frac_with_cells > 0.5
        and not size_degenerate
        and not charge_degenerate
        and np.isfinite(q_tot).mean() > 0.9
    )

    reason = "ok"
    if not passed:
        if frac_with_cells <= 0.5:
            reason = (
                "most measurements have empty channel lists — digi hook missing "
                "or smearing digi (no geometric clusters)"
            )
        elif size_degenerate:
            reason = "cluster sizes are uniformly 1×1 (non-geometric / empty)"
        elif charge_degenerate:
            reason = "Q_tot degenerate (all zero/NaN or constant empty)"
        else:
            reason = "cluster features failed non-degeneracy checks"

    return {
        "n_cells": int(len(cells)),
        "n_measurements": n_meas,
        "frac_measurements_with_cells": float(frac_with_cells),
        "s_u": _stats(s_u),
        "s_v": _stats(s_v),
        "Q_tot": _stats(q_tot),
        "sig_uu": _stats(sig_uu),
        "sig_uv": _stats(sig_uv),
        "sig_vv": _stats(sig_vv),
        "size_degenerate": bool(size_degenerate),
        "charge_degenerate": bool(charge_degenerate),
        "moments_degenerate": bool(moments_degenerate),
        "pass": bool(passed),
        "reason": reason,
    }


def probe_digi_dir(digi_dir: Path, events: list[int]) -> dict[str, Any]:
    digi_dir = Path(digi_dir)
    per_event = {}
    for ev in events:
        cells = _load_event_cells(digi_dir, ev)
        if cells is None:
            per_event[str(ev)] = {
                "pass": False,
                "reason": f"cells.csv not found for event {ev} under {digi_dir}",
            }
        else:
            per_event[str(ev)] = summarize_clusters(cells)

    overall_pass = (
        all(bool(v.get("pass")) for v in per_event.values()) and len(per_event) > 0
    )
    return {
        "check": 1,
        "name": "cluster_features_populated",
        "digi_dir": str(digi_dir),
        "events": [int(e) for e in events],
        "per_event": per_event,
        "pass": bool(overall_pass),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("digi_dir", type=Path, help="Directory with event*-cells.csv")
    p.add_argument("--events", type=int, nargs="+", default=[0, 1])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    report = probe_digi_dir(args.digi_dir, args.events)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
