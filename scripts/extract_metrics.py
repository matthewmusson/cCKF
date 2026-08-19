#!/usr/bin/env python3
"""Extract combined pre/post-ambi + timing metrics from a cCKF ACTS run.

Task 8 (per-event timing output and duplicate rate reporting) post-processing
step. Given a run's ``output_dir``, this script reads:

1. ``performance_finding_ckf.root`` — DM efficiency/fake/duplicate rate
   *before* ambiguity resolution, written by ``RootTrackFinderPerformanceWriter``
   when ``ckf_finding_performance: true`` (see ``digi_and_reco.py`` /
   ``_add_track_writers`` and ``configs/cckf_envelope.yaml``).
2. ``performance_finding_ambi.root`` — same metrics *after* ambiguity
   resolution (``ambi_finding_performance: true``).
3. The cCKF per-event timing CSV (``cckf_timing_output`` in the run config,
   default ``timing.csv``) — written by ``CckfTrackFindingAlgorithm::execute``
   via ``writeTimingCsv()``. One row per event with gate/value/feature-build
   sub-timings.
4. ``timing.tsv`` — ACTS ``Sequencer``'s own per-algorithm wall-clock timing
   (seeding, CKF, ambiguity resolution, ...), aggregated over the whole run
   (not per event). NOTE: despite the ``.tsv`` name, ACTS's
   ``Sequencer::storeTiming`` always writes comma-separated values
   (``identifier,time_total_s,time_perevent_s``) — this script sniffs the
   delimiter defensively rather than assuming either.

Both the pre/post-ambi ROOT summary scalars and the ACTS sequencer timing are
run-level aggregates (one number per run, not per event) — that is how
``RootTrackFinderPerformanceWriter`` and ``Sequencer::storeTiming`` write
them. Only the cCKF timing CSV is genuinely per-event. The output JSON
reflects that: a ``summary`` block for the aggregates, an
``acts_sequencer_timing`` list keyed by algorithm identifier, and a
``cckf_timing_per_event`` list keyed by event.

Any input that is missing (e.g. a run that only requested timing, with no
performance writers enabled) is reported as ``null`` in the corresponding
JSON field rather than raising — this script is meant to be safe to run
against partial output directories.

Usage
-----
    python scripts/extract_metrics.py --output-dir /path/to/results --output metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

try:
    import uproot
except ImportError:  # pragma: no cover - uproot is a project dependency
    uproot = None

# Scalar names written by ActsExamples::RootTrackFinderPerformanceWriter's
# finalize() as single-element TVectorF objects (see
# Examples/Io/Root/src/RootTrackFinderPerformanceWriter.cpp). These are all
# DM-matched (TrackTruthMatcher is configured with doubleMatching=True in
# digi_and_reco.py's addCckfTracks / addCKFTracks wiring), so "eff_tracks"
# etc. here are the DM efficiency/fake/duplicate rates the project tracks.
SUMMARY_SCALARS = [
    "eff_tracks",
    "fakeratio_tracks",
    "duplicateratio_tracks",
    "eff_particles",
    "fakeratio_particles",
    "duplicateratio_particles",
]


def read_perf_root(path: Path) -> dict[str, float] | None:
    """Read the summary scalars out of a RootTrackFinderPerformanceWriter file.

    Returns ``None`` if the file does not exist (e.g. the corresponding
    performance writer was disabled for this run) rather than raising.
    """
    if not path.exists():
        return None
    if uproot is None:
        raise RuntimeError(
            "uproot is required to read ROOT performance files "
            f"(missing while trying to read {path})"
        )

    out: dict[str, float] = {}
    with uproot.open(path) as f:
        available = {key.split(";")[0] for key in f.keys()}
        for name in SUMMARY_SCALARS:
            if name not in available:
                continue
            try:
                # RootTrackFinderPerformanceWriter writes these as
                # TVectorF(1) via WriteObject — a single-element vector.
                out[name] = float(f[name].member("fElements")[0])
            except Exception:
                continue
    return out or None


def _coerce(value: str) -> Any:
    """Best-effort int -> float -> str coercion for a CSV field."""
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def read_cckf_timing_csv(path: Path) -> list[dict[str, Any]] | None:
    """Parse the per-event cCKF timing CSV written by writeTimingCsv().

    Columns: event, n_seeds, n_tracks, gate_inference_ns, gate_inference_calls,
    gate_feature_ns, gate_feature_calls, value_inference_ns,
    value_inference_calls, meas_selection_ns, meas_selection_calls.
    """
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [
            {k: _coerce(v) for k, v in row.items() if k is not None}
            for row in reader
            if any((v or "").strip() for v in row.values())
        ]
    return rows or None


def read_acts_sequencer_timing(path: Path) -> list[dict[str, Any]] | None:
    """Parse ACTS Sequencer's per-algorithm timing file.

    ``Sequencer::storeTiming`` (Examples/Framework/src/Framework/Sequencer.cpp)
    always writes comma-separated rows
    ``identifier,time_total_s,time_perevent_s`` with a trailing blank line,
    regardless of the configured output filename. Sniff the delimiter so this
    still works if a future ACTS version (or a differently configured run)
    genuinely writes tabs.
    """
    if not path.exists():
        return None
    with open(path) as fh:
        first_line = fh.readline()
        delimiter = "\t" if "\t" in first_line and "," not in first_line else ","
        fh.seek(0)
        reader = csv.DictReader(fh, delimiter=delimiter)
        rows = []
        for row in reader:
            if not row or not any((v or "").strip() for v in row.values() if v is not None):
                continue
            rows.append({k: _coerce(v) for k, v in row.items() if k is not None})
    return rows or None


def build_metrics(
    output_dir: Path,
    *,
    cckf_timing_name: str = "timing.csv",
    acts_timing_name: str = "timing.tsv",
    pre_ambi_name: str = "performance_finding_ckf.root",
    post_ambi_name: str = "performance_finding_ambi.root",
) -> dict[str, Any]:
    pre_ambi = read_perf_root(output_dir / pre_ambi_name)
    post_ambi = read_perf_root(output_dir / post_ambi_name)

    return {
        "output_dir": str(output_dir),
        "summary": {
            "pre_ambi": pre_ambi,
            "post_ambi": post_ambi,
        },
        "cckf_timing_per_event": read_cckf_timing_csv(output_dir / cckf_timing_name),
        "acts_sequencer_timing": read_acts_sequencer_timing(output_dir / acts_timing_name),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Run output directory containing the ROOT/CSV/TSV files to read.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the combined metrics JSON to.",
    )
    parser.add_argument(
        "--cckf-timing-name",
        default="timing.csv",
        help="Filename of the cCKF per-event timing CSV within --output-dir "
        "(matches cckf_timing_output in the run's YAML config).",
    )
    parser.add_argument(
        "--acts-timing-name",
        default="timing.tsv",
        help="Filename of the ACTS Sequencer's per-algorithm timing file "
        "within --output-dir (matches outputTimingFile in setup_acts_reconstruction).",
    )
    parser.add_argument(
        "--pre-ambi-root",
        default="performance_finding_ckf.root",
        help="Filename of the pre-ambiguity-resolution performance ROOT file.",
    )
    parser.add_argument(
        "--post-ambi-root",
        default="performance_finding_ambi.root",
        help="Filename of the post-ambiguity-resolution performance ROOT file "
        "(the default 'ambi' name matches the greedy ambi_solver; pass e.g. "
        "'performance_finding_ambiML.root' for the ML solver).",
    )
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        print(f"error: --output-dir {args.output_dir} is not a directory", file=sys.stderr)
        raise SystemExit(1)

    metrics = build_metrics(
        args.output_dir,
        cckf_timing_name=args.cckf_timing_name,
        acts_timing_name=args.acts_timing_name,
        pre_ambi_name=args.pre_ambi_root,
        post_ambi_name=args.post_ambi_root,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(metrics, fh, indent=2)

    def _status(block: Any) -> str:
        return "found" if block else "missing"

    print(
        f"Wrote {args.output}  "
        f"(pre-ambi ROOT: {_status(metrics['summary']['pre_ambi'])}, "
        f"post-ambi ROOT: {_status(metrics['summary']['post_ambi'])}, "
        f"cCKF timing: {_status(metrics['cckf_timing_per_event'])}, "
        f"ACTS sequencer timing: {_status(metrics['acts_sequencer_timing'])})"
    )


if __name__ == "__main__":
    main()
