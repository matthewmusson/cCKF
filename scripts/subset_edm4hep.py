#!/usr/bin/env python3
"""Extract the first N events from an edm4hep/podio ROOT file.

Preferred path if this keeps failing: skip subsetting and upload the full
~6 GB file to Modal, then run_ckf with --events 8 (Sequencer only reads N
events). Subsetting is optional storage optimization.

Run via:
    cd ~/cCKF && source setup_env.sh
    cckf_shifter_run "$(pwd)/scripts/subset_edm4hep.py" "$IN" "$OUT" -n 32
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path


def _preload_edm4hep_dictionaries() -> None:
    """Load edm4hep/podio dict libs so Frame I/O can deserialize collections."""
    import ROOT

    spack_root = Path("/spack/opt/spack/linux-x86_64")
    include_dirs = []
    for pattern in ("edm4hep-*/include", "podio-*/include"):
        include_dirs.extend(str(p) for p in spack_root.glob(pattern))
    if include_dirs:
        existing = os.environ.get("ROOT_INCLUDE_PATH", "")
        os.environ["ROOT_INCLUDE_PATH"] = ":".join(
            include_dirs + ([existing] if existing else [])
        )
        print(f"ROOT_INCLUDE_PATH includes: {include_dirs}")

    lib_globs = [
        "/spack/opt/spack/linux-x86_64/edm4hep-*/lib/*.so*",
        "/spack/opt/spack/linux-x86_64/podio-*/lib/*.so*",
    ]
    loaded = []
    for pattern in lib_globs:
        for lib in sorted(glob.glob(pattern)):
            # Skip versioned duplicates like .so.1.2 when .so exists; Load is idempotent
            if ".so." in os.path.basename(lib) and not lib.endswith(".so"):
                continue
            rc = ROOT.gSystem.Load(lib)
            loaded.append((lib, rc))
    print(f"Loaded {len(loaded)} edm4hep/podio libraries")

    try:
        import edm4hep  # noqa: F401

        print("import edm4hep: OK")
    except Exception as exc:  # pragma: no cover
        print(f"import edm4hep failed: {exc}")


def _subset_with_acts(
    input_path: Path, output_path: Path, n_events: int, category: str
) -> int:
    """Copy frames via ACTS Sequencer."""
    _preload_edm4hep_dictionaries()

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
    # Empty collections list: write the frame as read from PodioReader.
    s.addWriter(
        PodioWriter(
            level=acts.logging.INFO,
            inputFrame="events",
            outputPath=str(output_path),
            category=category,
            collections=[],
        )
    )
    s.run()
    return n_events


def _subset_with_podio(
    input_path: Path, output_path: Path, n_events: int, category: str
) -> int:
    """Direct podio I/O with dictionaries preloaded."""
    _preload_edm4hep_dictionaries()

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


def _subset_with_root_clone(
    input_path: Path, output_path: Path, n_events: int, category: str
) -> int:
    """Byte-level TTree clone of first N entries (avoids edm4hep object model).

    Best-effort: works when podio stores events as TTrees. Metadata trees are
    copied in full. If the file uses RNTuple, this backend will fail clearly.
    """
    import ROOT

    if output_path.exists():
        output_path.unlink()

    fin = ROOT.TFile.Open(str(input_path))
    if not fin or fin.IsZombie():
        raise RuntimeError(f"Could not open {input_path}")

    fout = ROOT.TFile(str(output_path), "RECREATE")
    written = 0

    keys = list(fin.GetListOfKeys())
    print(f"ROOT keys ({len(keys)}): {[k.GetName() for k in keys[:20]]}")

    for key in keys:
        name = key.GetName()
        obj = key.ReadObj()
        if obj is None:
            continue
        fout.cd()
        if obj.InheritsFrom("TTree"):
            tree = obj
            n_entries = int(tree.GetEntries())
            # Event category trees get truncated; everything else copied whole.
            is_event = name == category or name.startswith(f"{category}/")
            n_copy = min(n_events, n_entries) if is_event else n_entries
            print(f"  cloning TTree '{name}': {n_copy}/{n_entries} entries")
            new_tree = tree.CloneTree(0)
            for i in range(n_copy):
                tree.GetEntry(i)
                new_tree.Fill()
            new_tree.Write()
            if is_event:
                written = max(written, n_copy)
        elif obj.InheritsFrom("TDirectory"):
            # Recurse one level for nested event directories
            fout.mkdir(name)
            fout.cd(name)
            for subkey in obj.GetListOfKeys():
                sub = subkey.ReadObj()
                if sub is not None and sub.InheritsFrom("TTree"):
                    n_entries = int(sub.GetEntries())
                    n_copy = min(n_events, n_entries)
                    print(
                        f"  cloning {name}/{subkey.GetName()}: "
                        f"{n_copy}/{n_entries} entries"
                    )
                    new_tree = sub.CloneTree(0)
                    for i in range(n_copy):
                        sub.GetEntry(i)
                        new_tree.Fill()
                    new_tree.Write()
                    written = max(written, n_copy)
            fout.cd()
        else:
            obj.Write()

    fout.Write()
    fout.Close()
    fin.Close()
    return written if written > 0 else n_events


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
        choices=("acts", "podio", "root-clone"),
        default="root-clone",
        help="Copy backend (default: root-clone; safest without dicts)",
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

    if args.backend == "acts":
        written = _subset_with_acts(
            args.input, args.output, args.n_events, args.category
        )
    elif args.backend == "podio":
        written = _subset_with_podio(
            args.input, args.output, args.n_events, args.category
        )
    else:
        written = _subset_with_root_clone(
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
