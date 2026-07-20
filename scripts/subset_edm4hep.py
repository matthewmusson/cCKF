#!/usr/bin/env python3
"""Extract the first N events from an edm4hep/podio ROOT file.

Intended to run on NERSC inside the ColliderML/ODD shifter image, which
already has podio + ROOT:

    shifter --image=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0 \\
        -- python3 subset_edm4hep.py \\
        /global/cfs/cdirs/m4958/data/ColliderML/simulation/full_pileup/ttbar/v1/runs/0/edm4hep.root \\
        $SCRATCH/ttbar_pu_n32_edm4hep.root -n 32

A full_pileup ttbar file is ~6.2 GB / 128 events (~50 MB/event). Subsetting
to 8–32 events yields a Modal-friendly file without going through a laptop
for the heavy lifting (then `modal volume put` from NERSC or scp the slim file).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _open_reader(path: str):
    try:
        from podio.reading import get_reader

        return get_reader(path)
    except Exception:
        from podio.root_io import Reader

        return Reader(path)


def _open_writer(path: str):
    from podio.root_io import Writer

    return Writer(path)


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
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")
    if args.n_events < 1:
        raise SystemExit("--n-events must be >= 1")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    reader = _open_reader(str(args.input))
    frames = reader.get(args.category)
    n_total = len(frames) if hasattr(frames, "__len__") else None

    print(f"Input:  {args.input} ({args.input.stat().st_size / 1e9:.2f} GB)")
    if n_total is not None:
        print(f"Events in file: {n_total}")
        if args.n_events > n_total:
            print(f"Requested {args.n_events}; clamping to {n_total}")
            args.n_events = n_total
    print(f"Writing first {args.n_events} events -> {args.output}")

    writer = _open_writer(str(args.output))
    written = 0
    for i, frame in enumerate(frames):
        if i >= args.n_events:
            break
        writer.write_frame(frame, args.category)
        written += 1
        if written == 1 or written % 8 == 0:
            print(f"  wrote {written}/{args.n_events}")

    # Ensure buffers are flushed (API varies slightly across podio versions).
    for method_name in ("finish", "close", "flush"):
        method = getattr(writer, method_name, None)
        if callable(method):
            method()
            break
    del writer

    size_mb = args.output.stat().st_size / 1e6
    print(f"Done: {written} events, {size_mb:.1f} MB -> {args.output}")


if __name__ == "__main__":
    main()
