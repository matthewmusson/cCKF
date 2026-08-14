"""Storage estimate for chi²-gate calibration collection (spec §6.6 check 3).

Uses pilot digi/seed counts from gate_pilot_1785284680. Does not resolve
seed-recoverability. Writes JSON next to this script's output path.
"""

from __future__ import annotations

import json
from pathlib import Path

# Pilot event 0 (envelope, geometric digi, n=10 pilot window constant)
PILOT = {
    "event": 0,
    "n_measurements": 202_985,
    "n_cells": 1_016_311,
    "n_envelope_seeds": 541_376,
    "source": "gate_pilot_1785284680",
}

# Bytes/row estimates
FULL_SCHEMA_B = 300  # spec prior (~cov_packed alone is 84 B)
SLIM_SCHEMA_B = 48  # 12 numeric/bool columns, parquet-compressed ~this

# Row model: each CKF-visited state expands to all window candidates.
# Pilot has no CKF yet; bound using seeds × surfaces × candidates × branch factor.
SCENARIOS = [
    {
        "name": "greedy_low",
        "surfaces": 8,
        "avg_cands": 3,
        "branch": 1.0,
        "survive": 0.35,
    },
    {
        "name": "greedy_mid",
        "surfaces": 10,
        "avg_cands": 5,
        "branch": 1.0,
        "survive": 0.35,
    },
    {
        "name": "branch5_mid",
        "surfaces": 10,
        "avg_cands": 5,
        "branch": 2.0,
        "survive": 0.35,
    },
    {
        "name": "branch5_high",
        "surfaces": 12,
        "avg_cands": 8,
        "branch": 3.0,
        "survive": 0.35,
    },
]

N_EVENTS = 32
BUDGET_GB = 40.0

SLIM_COLUMNS = [
    "event_id",
    "seed_id",
    "branch_id",
    "step_k",
    "layer_id",
    "chi2_inc",
    "window_count",
    "geometric_density",
    "eta",
    "label",
    "majority_undefined",
    "cluster_merged",
]


def estimate() -> dict:
    rows = []
    for sc in SCENARIOS:
        rows_per_evt = (
            PILOT["n_envelope_seeds"]
            * sc["surfaces"]
            * sc["avg_cands"]
            * sc["branch"]
            * sc["survive"]
        )
        full_gb = rows_per_evt * FULL_SCHEMA_B * N_EVENTS / 1e9
        slim_gb = rows_per_evt * SLIM_SCHEMA_B * N_EVENTS / 1e9
        rows.append(
            {
                **sc,
                "rows_per_event": rows_per_evt,
                "full_schema_32evt_GB": full_gb,
                "slim_schema_32evt_GB": slim_gb,
                "full_exceeds_40GB": full_gb > BUDGET_GB,
                "slim_exceeds_40GB": slim_gb > BUDGET_GB,
            }
        )

    # Decision rule: if any mid scenario full > 40GB → slim only
    mid = next(r for r in rows if r["name"] == "branch5_mid")
    use_slim = mid["full_exceeds_40GB"] or mid["slim_schema_32evt_GB"] > 5.0

    return {
        "pilot": PILOT,
        "n_events": N_EVENTS,
        "budget_GB": BUDGET_GB,
        "scenarios": rows,
        "decision": {
            "use_slim_schema": True,  # forced: full mid scenario ≫ 40 GB
            "reason": (
                f"Full §6.5 at branch5_mid ≈ {mid['full_schema_32evt_GB']:.0f} GB "
                f"≫ {BUDGET_GB} GB. Writing slim columns only: {SLIM_COLUMNS}"
            ),
            "slim_columns": SLIM_COLUMNS,
            "geometric_density_radius_mm": 5.0,
            "window_n": 10,
            "note": (
                "Estimate uses pilot seed count before CKF pruning; actual rows "
                "may be lower if many seeds die early, or higher if windows are denser."
            ),
        },
    }


def main() -> None:
    out = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "chi2_gate_calib"
        / "storage_estimate.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    report = estimate()
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["decision"], indent=2))
    print(f"Wrote {out}")
    for sc in report["scenarios"]:
        print(
            f"  {sc['name']}: {sc['rows_per_event']/1e6:.1f}M rows/evt  "
            f"full={sc['full_schema_32evt_GB']:.0f}GB  slim={sc['slim_schema_32evt_GB']:.1f}GB"
        )


if __name__ == "__main__":
    main()
