#!/usr/bin/env python3
"""Generate CKF sweep config files for coordinate-wise parameter optimization.

Sweep strategy (from benchmark doc):
  1. Sweep chi2CutOffMeasurement at default holes/branches → find elbow
  2. Sweep numMeasurementsCutOff (branch cap) at best chi2
  3. Sweep nMeasurementsMin at best chi2 + best branches

Usage:
    python3 generate_sweep_configs.py
    python3 generate_sweep_configs.py --best-chi2 15 --best-branches 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).parent / "configs"

# Daniel's production defaults (from digitization_config.yaml)
PRODUCTION_DEFAULTS = {
    "events": 128,
    "threads": 1,
    "digi": True,
    "reco": True,
    "num_seeds_per_spm": 40,
    "seed_minPt": 0.5,
    "seed_impactMax": 3.0,
    "ckf_chi2CutOffMeasurement": 15.0,
    "ckf_chi2CutOffOutlier": 25.0,
    "ckf_numMeasurementsCutOff": 1,
    "ckf_nMeasurementsMin": 6,
    "ckf_maxHolesAndOutliers": 3,
    "ckf_ptMin": 0.7,
    "ckf_finding_performance": True,
    "ambi_finding_performance": True,
}


def write_config(name: str, overrides: dict) -> Path:
    cfg = {**PRODUCTION_DEFAULTS, **overrides}
    path = CONFIGS_DIR / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def generate_sweep_1_chi2() -> list[Path]:
    """Sweep chi2CutOffMeasurement at default everything else."""
    paths = []
    for chi2 in [5.0, 10.0, 15.0, 20.0, 30.0]:
        name = f"sweep_1_chi2_{int(chi2)}"
        p = write_config(name, {"ckf_chi2CutOffMeasurement": chi2})
        paths.append(p)
        print(f"  {name}: chi2CutOff={chi2}")
    return paths


def generate_sweep_2_branches(best_chi2: float) -> list[Path]:
    """Sweep numMeasurementsCutOff at best chi2."""
    paths = []
    for branches in [1, 3, 5, 10]:
        name = f"sweep_2_branches_{branches}"
        p = write_config(name, {
            "ckf_chi2CutOffMeasurement": best_chi2,
            "ckf_numMeasurementsCutOff": branches,
        })
        paths.append(p)
        print(f"  {name}: chi2={best_chi2}, branches={branches}")
    return paths


def generate_sweep_3_measurements(best_chi2: float, best_branches: int) -> list[Path]:
    """Sweep nMeasurementsMin at best chi2 + best branches."""
    paths = []
    for n_meas in [6, 7, 8]:
        name = f"sweep_3_nmeas_{n_meas}"
        p = write_config(name, {
            "ckf_chi2CutOffMeasurement": best_chi2,
            "ckf_numMeasurementsCutOff": best_branches,
            "ckf_nMeasurementsMin": n_meas,
        })
        paths.append(p)
        print(f"  {name}: chi2={best_chi2}, branches={best_branches}, nMeasMin={n_meas}")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-chi2", type=float, default=15.0,
                        help="Best chi2 from sweep 1 (default: 15, run sweep 1 first)")
    parser.add_argument("--best-branches", type=int, default=1,
                        help="Best branch cap from sweep 2 (default: 1, run sweep 2 first)")
    args = parser.parse_args()

    CONFIGS_DIR.mkdir(exist_ok=True)

    # Also write the production baseline for comparison
    write_config("baseline", {})
    print("Generated baseline config (Daniel's production defaults)")

    print("\n--- Sweep 1: chi2CutOffMeasurement ---")
    generate_sweep_1_chi2()

    print(f"\n--- Sweep 2: numMeasurementsCutOff (chi2={args.best_chi2}) ---")
    generate_sweep_2_branches(args.best_chi2)

    print(f"\n--- Sweep 3: nMeasurementsMin (chi2={args.best_chi2}, branches={args.best_branches}) ---")
    generate_sweep_3_measurements(args.best_chi2, args.best_branches)

    all_configs = sorted(CONFIGS_DIR.glob("sweep_*.yaml"))
    print(f"\nTotal: {len(all_configs)} sweep configs + 1 baseline")
    print(f"Update sweep.sbatch --array=0-{len(all_configs)-1}")


if __name__ == "__main__":
    main()
