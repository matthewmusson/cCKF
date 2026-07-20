#!/usr/bin/env python3
"""Extract the first N events from an edm4hep/podio ROOT file.

Run via the cCKF shifter helper (loads ACTS + edm4hep dictionaries):

    cd ~/cCKF && source setup_env.sh
    IN=/global/cfs/cdirs/m4958/data/ColliderML/simulation/full_pileup/ttbar/v1/runs/0/edm4hep.root
    OUT=$SCRATCH/ttbar_pu_n32_edm4hep.root
    cckf_shifter_run "$(pwd)/scripts/subset_edm4hep.py" "$IN" "$OUT" -n 32

Uses ACTS PodioReader/PodioWriter (same stack as digi_and_reco.py). A plain
podio iterator fails here without edm4hep dictionaries loaded.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _subset_with_acts(
    input_path: Path, output_path: Path, n_events: int, category: str
) -> int:
    """Copy frames via ACTS Sequencer (reliable on the ColliderML shifter image)."""
    import acts
    import acts.examples
    from acts.examples import Sequencer
    from acts.examples.edm4hep import PodioReader, PodioWriter

    if output_path.exists():
        output_path.unlink()

    s = Sequencer(
        events=n_events,
        numThreads=1,
        logLevel=acts.logging.INFO,
    )
    s.addReader(
        PodioReader(
            level=acts.logging.INFO,
            inputPath=str(input_path),
            outputFrame="events",
            category=category,
        )
    )
    s.addWriter(
        PodioWriter(
            level=acts.logging.INFO,
            inputFrame="events",
            outputPath=str(output_path),
            category=category,
        )
    )
    s.run()
    return n_events


def _subset_with_podio(
    input_path: Path, output_path: Path, n_events: int, category: str
) -> int:
    """Fallback: direct podio I/O with edm4hep dictionaries preloaded."""
    # Dictionaries must be loaded before reading ColliderML ROOT files,
    # otherwise readNextEntry raises cppyy bad_function_call.
    import edm4hep  # noqa: F401

    try:
        from podio.reading import get_reader

        reader = get_reader(str(input_path))
    except Exception:
        from podio.root_io import Reader

        reader = Reader(str(input_path))

    from podio.root_io import Writer

    frames = reader.get(category)
    n_total = len(frames) if hasattr(frames, "__len__") else None
    if n_total is not None:
        print(f"Events in file: {n_total}")
        n_events = min(n_events, n_total)

    if output_path.exists():
        output_path.unlink()

    writer = Writer(str(output_path))
    written = 0
    # Index access avoids the broken readNextEntry iterator path.
    for i in range(n_events):
        frame = frames[i]
        writer.write_frame(frame, category)
        written += 1
        if written == 1 or written % 8 == 0:
            print(f"  wrote {written}/{n_events}")

    for method_name in ("finish", "close", "flush"):
        method = getattr(writer, method_name, None)
        if callable(method):
            method()
            break
    del writer
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source edm4hep.root")
    parser.add_argument("output", type=Path, help="Destination edm4hep.root")
    parser.add_argument(
        "-n",
        "--n-events",
        type=int,
        default=32,
        help="Number of events to copy (default: 32)",
    )
    parser.add_argument(
        "--category",
        default="events",
        help="Podio frame category (default: events)",
    )
    parser.add_argument(
        "--backend",
        choices=("acts", "podio"),
        default="acts",
        help="Copy backend (default: acts)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")
    if args.n_events < 1:
        raise SystemExit("--n-events must be >= 1")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {args.input} ({args.input.stat().st_size / 1e9:.2f} GB)")
    print(f"Output: {args.output}")
    print(f"Backend={args.backend}, n_events={args.n_events}, category={args.category}")
    print(f"ODD_PATH={os.environ.get('ODD_PATH', '<unset>')}")

    if args.backend == "acts":
        written = _subset_with_acts(
            args.input, args.output, args.n_events, args.category
        )
    else:
        written = _subset_with_podio(
            args.input, args.output, args.n_events, args.category
        )

    if not args.output.is_file():
        raise SystemExit(f"Output was not created: {args.output}")

    size_mb = args.output.stat().st_size / 1e6
    print(f"Done: {written} events, {size_mb:.1f} MB -> {args.output}")
    if size_mb < 1.0:
        raise SystemExit(
            f"Output looks too small ({size_mb:.1f} MB). Something went wrong."
        )


if __name__ == "__main__":
    main()
