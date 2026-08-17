"""Modal entrypoints for the gate/value training pipeline.

The expanded Parquet lives on the ``surp-acts-data`` volume and is far too
large to move, so all cache building and training runs there.

Sequencing note: the ``is_ckf_selected`` patch and the cache builds write to the
volume and must run **sequentially**, one container at a time. A previous
parallel patch run corrupted 11 Parquet files through concurrent
``data_vol.commit()`` calls (see experiments/LOG.md, 2026-08-12). Do not
reintroduce ``starmap`` for any step that commits.

Usage
-----
    modal run modal_train.py::patch_selected
    modal run modal_train.py::build_caches
    modal run --detach modal_train.py::train_gate_all
    modal run modal_train.py::audit --model-dir /data/models/gate_A
"""

from __future__ import annotations

import modal

app = modal.App("cckf-train")
data_vol = modal.Volume.from_name("surp-acts-data", create_if_missing=False)
DATA_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        "numpy>=2.0",
        "pandas>=2.2",
        "pyarrow>=17.0",
        "scikit-learn>=1.5",
        "scipy>=1.14",
        "matplotlib>=3.9",
        "wandb>=0.17",
        "uproot>=5.3",
    )
    .add_local_dir("cckf", "/root/cckf")
    .add_local_dir("scripts", "/root/scripts")
    .add_local_file("expansion.py", "/root/expansion.py")
)

EXPANDED_DIR = f"{DATA_PATH}/results/train32/expanded"
SELECTED_DIR = f"{DATA_PATH}/results/train32/selected"
CACHE_DIR = f"{DATA_PATH}/cache"
MODEL_DIR = f"{DATA_PATH}/models"


def _run_script(cmd: list[str]) -> None:
    """Run one of the repo's CLI scripts inside the image.

    ``PYTHONPATH=/root`` is not optional. Python puts the *script's* own
    directory on ``sys.path``, not the working directory, and this image has no
    editable install — so ``/root/scripts/build_gate_cache.py`` cannot
    ``import cckf`` without it, even with ``cwd="/root"``. Discovered during
    Task 6 execution, where the CLI needed ``PYTHONPATH=.`` locally for exactly
    this reason.
    """
    import os
    import subprocess

    print(" ".join(cmd))
    subprocess.run(
        cmd,
        check=True,
        cwd="/root",
        env={**os.environ, "PYTHONPATH": "/root"},
    )


@app.function(
    image=image, volumes={DATA_PATH: data_vol}, cpu=8, memory=131072, timeout=86400
)
def patch_selected_all(only_events: str = "") -> list[dict]:
    """Add ``is_ckf_selected`` to every assigned event, sequentially.

    ``only_events``, when non-empty, restricts processing to a comma-
    separated subset of event ids (e.g. ``"0,1"``) instead of every assigned
    event. The subset is still validated against the sealed test guard and
    the assigned train/val/cal set -- see
    :func:`cckf.event_selection.resolve_requested_events`. Empty (the
    default) processes every assigned event, exactly as before.
    """
    import sys
    from pathlib import Path

    import pyarrow.parquet as pq

    sys.path.insert(0, "/root")
    from cckf import splits
    from cckf.event_selection import resolve_requested_events
    from scripts.patch_is_selected import patch_event

    reports = []
    assigned = (*splits.TRAIN_EVENTS, *splits.VAL_EVENTS, *splits.CAL_EVENTS)
    events = resolve_requested_events(only_events, assigned)
    splits.assert_not_test(events)  # guard applies to the resolved set too

    trackstates = _resolve_trackstates_paths(events)

    for event_id in sorted(events):
        src = f"{EXPANDED_DIR}/expanded_event{event_id:09d}.parquet"
        out = f"{SELECTED_DIR}/expanded_event{event_id:09d}.parquet"
        if Path(out).exists():
            # This volume has a documented history of Parquet corruption
            # (11 files, "magic bytes not found in footer" -- see
            # experiments/LOG.md) from a prior concurrent-commit incident. A
            # crash mid-commit is not provably excluded, so confirm the
            # existing output's footer is actually readable before trusting
            # it -- metadata-only, no row read.
            try:
                pq.ParquetFile(out)
            except Exception as exc:
                print(
                    f"event {event_id}: existing output at {out} failed "
                    f"integrity check ({exc!r}); re-patching"
                )
            else:
                print(f"event {event_id}: already patched, skipping")
                continue
        report = patch_event(src, trackstates[event_id], event_id, out)
        print(report)
        reports.append(report)
        data_vol.commit()  # one commit per event, never concurrent
    return reports


def _resolve_trackstates_paths(requested_events):
    """Map each requested event to its ``trackstates_ckf.root``.

    Returns ``{event_id: path}`` covering every event in
    ``requested_events``, or raises rather than returning a partial map.

    The explicit provenance map (``cckf.stage1_map``) is authoritative and is
    tried first; the directory scan (:func:`_build_trackstates_index` +
    ``find_trackstates``) is only a fallback for events the map cannot serve.
    That ordering is the whole point, and it is not an optimisation --
    scanning cannot answer this question correctly. ``/data/results`` holds
    over a thousand result directories, many containing a
    ``trackstates_ckf.root`` for the *same* event id produced by a
    *different* CKF configuration during the parameter scans. Nothing in the
    filename or the ``event_nr`` branch marks which one produced the training
    Parquets, so the scan resolves ties by sort order: on the first real run
    it picked ``calib_medium_1785797665`` (the Medium operating point) over
    the correct pilot file, which would have joined the envelope-config
    Parquets against Medium-config trackstates. The map records the actual
    provenance -- which Modal run wrote which output -- and so is the only
    source that can distinguish them.

    The scan is therefore built lazily, over the *unmapped* events only. That
    also narrows ``check_no_event_nr_fallback_is_safe`` to exactly the events
    that would actually rely on the no-``event_nr`` fallback, instead of
    tripping it over events the map already resolved unambiguously.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")
    from cckf import stage1_map

    resolved, unmapped = {}, []
    for event_id in sorted(requested_events):
        try:
            path = stage1_map.trackstates_path_for(event_id)
        except KeyError as exc:
            print(
                f"event {event_id}: not covered by cckf.stage1_map "
                f"({exc}); falling back to a directory scan"
            )
            unmapped.append(event_id)
            continue
        # Verify rather than trust: a map entry naming a path that is not on
        # this volume is a stale map, and silently scanning past it would
        # reintroduce the wrong-config join the map exists to prevent. Say so.
        if Path(path).exists():
            resolved[event_id] = path
        else:
            print(
                f"event {event_id}: cckf.stage1_map names {path}, which does "
                "not exist on this volume; falling back to a directory scan "
                "(check the map against modal_build_acts.py::expand_all_events)"
            )
            unmapped.append(event_id)

    if unmapped:
        from cckf.trackstates_index import find_trackstates

        print(f"scanning /data/results for {len(unmapped)} unmapped event(s)")
        index = _build_trackstates_index(unmapped)
        for event_id in unmapped:
            resolved[event_id] = find_trackstates(event_id, index)
    return resolved


def _build_trackstates_index(requested_events):
    """Scan ``/data/results`` once and record which event(s) each trackstates
    file covers, then validate the result against every event this run will
    resolve.

    Returns a ``cckf.trackstates_index.TrackstatesIndex``. That module holds
    the dataclass and the pure-logic lookup (:func:`cckf.trackstates_index.
    find_trackstates`) so both are importable and unit-testable without the
    ``modal`` package or a live ROOT file; this function is the only part
    that actually touches the filesystem.

    Stage 1 wrote 32 events across 16 batch directories, so the layout is not a
    single predictable path -- every ``trackstates_ckf.root`` under
    ``/data/results`` must be probed. Doing that scan once here, instead of
    re-globbing and re-opening every candidate per event (the original
    per-event search), turns up to ~512 file opens across 32 events into one
    open per file (~16).

    A file that fails to open, or whose ``trackstates`` tree can't be read, is
    recorded in ``unreadable`` with the exception -- never silently dropped.
    This volume has a documented history of exactly this kind of corruption
    (11 Parquet files with unreadable footers from a prior concurrent-commit
    incident, experiments/LOG.md), and a broken ROOT file that vanishes from
    the index with no trace would misdirect debugging toward "the file is
    missing" when the truth is "the file is broken".

    A file with no ``event_nr`` branch is *not* unreadable: per
    ``expansion.load_trackstates``, that is a legitimate single-event file
    from an older pipeline stage, and the whole file is treated as belonging
    to whichever event is requested. Such files go in ``no_event_nr``;
    :func:`cckf.trackstates_index.find_trackstates` decides whether that
    fallback is safe to use for a given lookup -- but that per-lookup guard
    can't see how many events the whole run is resolving, so it cannot alone
    stop a single-event file from being silently handed to every event in a
    multi-event run. ``requested_events`` lets
    :func:`cckf.trackstates_index.check_no_event_nr_fallback_is_safe` catch
    that configuration here, once, before the caller resolves (and commits)
    even the first event -- not partway through the run when the second
    event's lookup would otherwise reuse the same file.
    """
    import sys
    from pathlib import Path

    import uproot

    sys.path.insert(0, "/root")
    from cckf.trackstates_index import (
        TrackstatesIndex,
        check_no_event_nr_fallback_is_safe,
    )

    candidates = sorted(Path(f"{DATA_PATH}/results").rglob("trackstates_ckf.root"))
    if not candidates:
        raise FileNotFoundError("no trackstates_ckf.root anywhere under /data/results")

    index = TrackstatesIndex(n_candidates=len(candidates))
    for path in candidates:
        try:
            with uproot.open(path) as fh:
                tree = fh["trackstates"]
                if "event_nr" not in set(tree.keys()):
                    print(
                        f"[modal_train] WARNING: {path} has no 'event_nr' "
                        "branch -- treating it as a single-event file "
                        "(matches expansion.load_trackstates fallback)"
                    )
                    index.no_event_nr.append(str(path))
                    continue
                file_events = set(tree["event_nr"].array(library="np").tolist())
        except (OSError, KeyError, ValueError) as exc:
            print(f"[modal_train] WARNING: could not read {path}: {exc!r}")
            index.unreadable.append((str(path), repr(exc)))
            continue

        for event_id in file_events:
            event_id = int(event_id)
            if event_id in index.matched:
                print(
                    f"[modal_train] WARNING: event {event_id} found in both "
                    f"{index.matched[event_id]} and {path}; keeping the first"
                )
                continue
            index.matched[event_id] = str(path)

    check_no_event_nr_fallback_is_safe(index, requested_events)
    return index


@app.function(
    image=image, volumes={DATA_PATH: data_vol}, cpu=16, memory=262144, timeout=86400
)
def build_all_caches(
    csv_dir: str = "",
    gate_parquet_dir: str = EXPANDED_DIR,
    splits_to_build: str = "train,val,cal",
    only_events: str = "",
    skip_value: bool = False,
    skip_gate: bool = False,
) -> dict:
    """Build gate and value caches, sequentially.

    Parameters
    ----------
    csv_dir : str
        Overrides the per-event simhits directory for the *value* cache step.
        Empty (default) lets ``build_value_cache.py`` resolve each event's
        directory from ``cckf.stage1_map``, which is the correct behaviour:
        Stage 1 wrote each event's CSVs into its own batch directory, so no
        single directory holds them all. Pass a path only to override the map
        with one directory for every event. This no longer gates whether
        value caches are built -- use ``skip_value`` for that.
    gate_parquet_dir : str
        Directory the *gate* cache step reads expanded Parquet from.
        Defaults to ``EXPANDED_DIR`` -- unlike the value cache, the gate
        cache does not need the ``is_ckf_selected`` patch (verified:
        ``GATE_SOURCE_COLUMNS | LABEL_COLUMNS`` is a subset of the unpatched
        76-column schema, and ``is_ckf_selected`` is not among those 29
        columns), so it can read directly from the already-existing expanded
        Parquets instead of waiting on the patch step. Pass ``SELECTED_DIR``
        explicitly to read patched output instead.
    splits_to_build : str
        Comma-separated split names (subset of ``"train,val,cal"``) to build
        gate (and, unless ``skip_value``, value) caches for. Default is all
        three -- unchanged behaviour when omitted.
    only_events : str
        Forwarded verbatim to ``--only-events`` on *both* cache steps (e.g.
        ``"0,1"`` for a staged run over two events). Empty (default) omits
        the flag, so each step builds every event in each requested split.
        A staged run routes both caches to sibling ``*_staged`` roots so it
        cannot overwrite a completed full build.
    skip_value : bool
        If True, skip the value-cache half entirely. Default False.
    skip_gate : bool
        If True, skip the gate-cache half entirely. Default False. Useful
        once the gate caches are already built, so a value-cache run does
        not spend hours re-streaming 174M rows it would only overwrite with
        identical output.
    """
    import sys

    results = {}
    splits = [s.strip() for s in splits_to_build.split(",") if s.strip()]

    # A staged (``--only-events``) build must never write to the same
    # directory a full build does. ``partial_split``/``events_used`` in
    # meta.json are the only distinguisher between a 2-event smoke-test cache
    # and a real 24-event cache, and no consumer reads them (train_gate.py
    # now refuses a partial cache by default -- see its --allow-partial-cache
    # flag -- but that check exists precisely because nothing else would
    # notice). Routing staged output to a sibling ``gate_staged/`` root makes
    # a staged run physically unable to overwrite a completed full cache, in
    # either direction, regardless of run order.
    gate_cache_root = f"{CACHE_DIR}/gate_staged" if only_events else f"{CACHE_DIR}/gate"
    value_cache_root = (
        f"{CACHE_DIR}/value_staged" if only_events else f"{CACHE_DIR}/value"
    )

    for split in [] if skip_gate else splits:
        cmd = [
            sys.executable,
            "/root/scripts/build_gate_cache.py",
            "--split",
            split,
            "--parquet-dir",
            gate_parquet_dir,
            "--out-dir",
            f"{gate_cache_root}/{split}",
        ]
        if only_events:
            cmd += ["--only-events", only_events]
        _run_script(cmd)
        data_vol.commit()
        results[f"gate_{split}"] = "ok"

    if not skip_value:
        for split in splits:
            cmd = [
                sys.executable,
                "/root/scripts/build_value_cache.py",
                "--split",
                split,
                "--parquet-dir",
                SELECTED_DIR,
                "--out-dir",
                f"{value_cache_root}/{split}",
            ]
            # Omitted, not passed empty: an empty --csv-dir is what tells
            # build_value_cache.py to resolve each event's directory from
            # cckf.stage1_map, and passing "" explicitly would work only
            # because argparse's default happens to match.
            if csv_dir:
                cmd += ["--csv-dir", csv_dir]
            if only_events:
                cmd += ["--only-events", only_events]
            _run_script(cmd)
            data_vol.commit()
            results[f"value_{split}"] = "ok"
    return results


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    gpu="A10G",
    memory=131072,
    timeout=43200,
    secrets=[modal.Secret.from_name("wandb")],
)
def train_gate_sampler(sampler: str, wandb_project: str = "cckf-gate") -> dict:
    """Train the gate with one §2.5 sampling strategy (S1 ablation arm)."""
    import json
    import sys

    out_dir = f"{MODEL_DIR}/gate_{sampler}"
    cmd = [
        sys.executable,
        "/root/scripts/train_gate.py",
        "--train-cache",
        f"{CACHE_DIR}/gate/train",
        "--val-cache",
        f"{CACHE_DIR}/gate/val",
        "--out-dir",
        out_dir,
        "--sampler",
        sampler,
        "--device",
        "cuda",
        "--wandb-project",
        wandb_project,
    ]
    _run_script(cmd)
    data_vol.commit()
    with open(f"{out_dir}/gate_metrics.json") as fh:
        return json.load(fh)


@app.function(
    image=image, volumes={DATA_PATH: data_vol}, cpu=8, memory=65536, timeout=7200
)
def run_audit(model_dir: str, value_predictions: str = "") -> dict:
    """Fit Platt on the calibration split and produce the §4.2 audit.

    ``value_predictions``, when given, points at a ``value_val_predictions.npz``
    and additionally emits the V_φ-vs-Tier-2 reliability figure. Left empty for
    gate-only audits, which run before the value function is trained.
    """
    import json
    import sys

    out_dir = f"{model_dir}/audit"
    cmd = [
        sys.executable,
        "/root/scripts/calibrate_and_audit.py",
        "--model",
        f"{model_dir}/gate_model.pt",
        "--cal-cache",
        f"{CACHE_DIR}/gate/cal",
        "--out-dir",
        out_dir,
    ]
    if value_predictions:
        cmd += ["--value-predictions", value_predictions]
    _run_script(cmd)
    data_vol.commit()
    with open(f"{out_dir}/calibration_audit.json") as fh:
        return json.load(fh)


@app.local_entrypoint()
def patch_selected(only_events: str = "") -> None:
    print(patch_selected_all.remote(only_events=only_events))


@app.local_entrypoint()
def build_caches(csv_dir: str = "") -> None:
    print(build_all_caches.remote(csv_dir=csv_dir))


@app.local_entrypoint()
def build_gate_cache_staged(only_events: str, split: str = "train") -> None:
    """Staged gate-only cache build over a small event subset.

    For the first real-data run: build the gate cache for a couple of events
    (e.g. ``--only-events 0,1``) before committing to a full 24/4/4-event
    build. Reads from ``EXPANDED_DIR`` -- the gate cache does not need the
    ``is_ckf_selected`` patch, so this does not wait on ``patch_selected`` --
    and always skips the value half (which does need the patch). ``split``
    is a single split name (not a list: Modal local entrypoints only accept
    CLI-friendly scalars), default ``"train"``.

    Writes to ``{CACHE_DIR}/gate_staged/{split}``, never
    ``{CACHE_DIR}/gate/{split}`` -- a separate root from the full build
    (``build_caches``), so the two can never collide regardless of which
    runs first. ``scripts/train_gate.py`` additionally refuses to train on
    the resulting cache (``meta.json`` has ``partial_split: true``) unless
    given ``--allow-partial-cache``; it exists for smoke-testing the
    training wiring, not for real training runs.

    Usage
    -----
        modal run modal_train.py::build_gate_cache_staged \\
            --only-events 0,1 --split train
    """
    print(
        build_all_caches.remote(
            gate_parquet_dir=EXPANDED_DIR,
            splits_to_build=split,
            only_events=only_events,
            skip_value=True,
        )
    )


@app.local_entrypoint()
def build_value_cache_staged(only_events: str, split: str = "train") -> None:
    """Staged value-only cache build over a small event subset.

    The value cache, unlike the gate cache, reads ``SELECTED_DIR`` -- it needs
    ``is_ckf_selected``, so ``patch_selected`` must have run for these events
    first. Skips the gate half, which is already built.

    Writes to ``{CACHE_DIR}/value_staged/{split}``, never
    ``{CACHE_DIR}/value/{split}``, so it cannot collide with a full build.
    ``meta.json`` carries ``partial_split: true``.

    Usage
    -----
        modal run modal_train.py::build_value_cache_staged \\
            --only-events 0,1 --split train
    """
    print(
        build_all_caches.remote(
            splits_to_build=split,
            only_events=only_events,
            skip_gate=True,
        )
    )


@app.local_entrypoint()
def train_gate_all() -> None:
    """S1 ablation: train all three sampling strategies (spec §2.5)."""
    for sampler in ("A", "B", "C"):
        print(f"=== sampler {sampler} ===")
        print(train_gate_sampler.remote(sampler=sampler))


@app.local_entrypoint()
def audit(model_dir: str = f"{MODEL_DIR}/gate_B", value_predictions: str = "") -> None:
    print(run_audit.remote(model_dir=model_dir, value_predictions=value_predictions))
