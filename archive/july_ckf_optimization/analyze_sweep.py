#!/usr/bin/env python3
"""Extract performance metrics from CKF sweep results.

Reads the ACTS log output from each sweep config and produces a summary CSV
with efficiency, fake rate, duplicate rate, and timing per config.

Usage:
    python3 analyze_sweep.py                    # reads from $SCRATCH/cckf/sweep/
    python3 analyze_sweep.py --sweep-dir /path  # custom sweep directory
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import yaml


def parse_log_metrics(log_path: Path) -> dict | None:
    """Extract TrackFinderPerformance metrics from ACTS log output."""
    text = log_path.read_text()

    metrics = {}

    # Find CKF metrics (first set of TrackFinderP lines, before ambi)
    # and ambi metrics (second set)
    finder_blocks = re.findall(
        r"TrackFinderP\s+INFO\s+(Efficiency|Fake ratio|Duplicate ratio) "
        r"with (tracks|particles) \([^)]+\) = ([\d.]+)",
        text,
    )

    stage = "ckf"
    seen_efficiency_count = 0
    for metric_name, basis, value in finder_blocks:
        key_prefix = metric_name.lower().replace(" ", "_")
        key = f"{stage}_{key_prefix}_{basis}"
        metrics[key] = float(value)

        if metric_name == "Efficiency" and basis == "tracks":
            seen_efficiency_count += 1
            if seen_efficiency_count == 2:
                stage = "ckf_finding"
            elif seen_efficiency_count == 3:
                stage = "ambi"

    # Extract timing
    timing_match = re.search(r"Processed (\d+) events in ([\d.]+) s", text)
    if timing_match:
        metrics["n_events"] = int(timing_match.group(1))
        metrics["total_time_s"] = float(timing_match.group(2))
        metrics["time_per_event_ms"] = (
            float(timing_match.group(2)) / int(timing_match.group(1)) * 1000
        )

    # Extract TrackFinding stats
    seeds_match = re.search(r"total seeds: (\d+)", text)
    tracks_match = re.search(r"found tracks: (\d+)", text)
    selected_match = re.search(r"selected tracks: (\d+)", text)
    if seeds_match:
        metrics["total_seeds"] = int(seeds_match.group(1))
    if tracks_match:
        metrics["found_tracks"] = int(tracks_match.group(1))
    if selected_match:
        metrics["selected_tracks"] = int(selected_match.group(1))

    return metrics if metrics else None


def parse_config(config_path: Path) -> dict:
    """Extract CKF parameters from config YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    return {
        "chi2CutOff": cfg.get("ckf_chi2CutOffMeasurement", "?"),
        "numMeasCutOff": cfg.get("ckf_numMeasurementsCutOff", "?"),
        "nMeasMin": cfg.get("ckf_nMeasurementsMin", "?"),
        "maxHolesOutliers": cfg.get("ckf_maxHolesAndOutliers", "?"),
        "seedsPerSpM": cfg.get("num_seeds_per_spm", "?"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path,
                        default=Path.home() / "pscratch/sd" / "cckf/sweep")
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="Directory with Slurm log files (default: cCKF/logs/)")
    args = parser.parse_args()

    # Try $SCRATCH/cckf/sweep if default doesn't exist
    sweep_dir = args.sweep_dir
    if not sweep_dir.exists():
        import os
        scratch = os.environ.get("SCRATCH", "")
        if scratch:
            sweep_dir = Path(scratch) / "cckf" / "sweep"

    if not sweep_dir.exists():
        print(f"Sweep directory not found: {sweep_dir}")
        print("Run the sweep first, or pass --sweep-dir")
        sys.exit(1)

    results = []
    for config_dir in sorted(sweep_dir.iterdir()):
        if not config_dir.is_dir():
            continue

        config_name = config_dir.name
        config_file = config_dir / "config.yaml"
        log_files = list(config_dir.glob("results/*.out")) + list(config_dir.glob("*.out"))

        # Try to get metrics from log output in the results dir
        # If not found, try the Slurm logs
        metrics = None
        for log_candidate in [
            config_dir / "results" / "timing_report.txt",
            *sorted(config_dir.glob("**/*.out")),
        ]:
            if log_candidate.exists():
                metrics = parse_log_metrics(log_candidate)
                if metrics:
                    break

        # Also try Slurm log dir
        if not metrics and args.log_dir:
            for slurm_log in sorted(args.log_dir.glob(f"*{config_name}*")):
                metrics = parse_log_metrics(slurm_log)
                if metrics:
                    break

        config_params = parse_config(config_file) if config_file.exists() else {}

        row = {"config": config_name, **config_params}
        if metrics:
            row.update(metrics)
        else:
            row["status"] = "NO_METRICS"

        results.append(row)

    if not results:
        print("No results found.")
        sys.exit(1)

    # Print summary table
    print(f"\n{'Config':<30} {'χ²':>5} {'Branch':>6} {'nMin':>4} "
          f"{'Eff(trk)':>9} {'Fake':>7} {'Dup':>7} {'ms/evt':>7}")
    print("-" * 90)
    for r in results:
        eff = r.get("ckf_efficiency_tracks", r.get("ambi_efficiency_tracks", ""))
        fake = r.get("ckf_fake_ratio_tracks", r.get("ambi_fake_ratio_tracks", ""))
        dup = r.get("ckf_duplicate_ratio_tracks", r.get("ambi_duplicate_ratio_tracks", ""))
        ms = r.get("time_per_event_ms", "")
        print(f"{r['config']:<30} {r.get('chi2CutOff',''):>5} {r.get('numMeasCutOff',''):>6} "
              f"{r.get('nMeasMin',''):>4} "
              f"{eff:>9.4f}" if isinstance(eff, float) else f"{eff:>9}",
              end="")
        if isinstance(fake, float):
            print(f" {fake:>7.4f}", end="")
        if isinstance(dup, float):
            print(f" {dup:>7.4f}", end="")
        if isinstance(ms, float):
            print(f" {ms:>7.1f}", end="")
        print()

    # Write CSV
    output_csv = sweep_dir / "sweep_results.csv"
    fieldnames = sorted(set().union(*(r.keys() for r in results)))
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {output_csv}")


if __name__ == "__main__":
    main()
