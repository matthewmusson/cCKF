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
    from cckf.trackstates_index import find_trackstates
    from scripts.patch_is_selected import patch_event

    reports = []
    assigned = (*splits.TRAIN_EVENTS, *splits.VAL_EVENTS, *splits.CAL_EVENTS)
    events = resolve_requested_events(only_events, assigned)
    splits.assert_not_test(events)  # guard applies to the resolved set too

    index = _build_trackstates_index(events)  # one scan, not one per event

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
        root = find_trackstates(event_id, index)
        report = patch_event(src, root, event_id, out)
        print(report)
        reports.append(report)
        data_vol.commit()  # one commit per event, never concurrent
    return reports


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
def build_all_caches(csv_dir: str = "") -> dict:
    """Build gate and value caches for train/val/cal, sequentially."""
    import sys

    results = {}
    for split in ("train", "val", "cal"):
        cmd = [
            sys.executable,
            "/root/scripts/build_gate_cache.py",
            "--split",
            split,
            "--parquet-dir",
            SELECTED_DIR,
            "--out-dir",
            f"{CACHE_DIR}/gate/{split}",
        ]
        _run_script(cmd)
        data_vol.commit()
        results[f"gate_{split}"] = "ok"

    if csv_dir:
        for split in ("train", "val", "cal"):
            cmd = [
                sys.executable,
                "/root/scripts/build_value_cache.py",
                "--split",
                split,
                "--parquet-dir",
                SELECTED_DIR,
                "--csv-dir",
                csv_dir,
                "--out-dir",
                f"{CACHE_DIR}/value/{split}",
            ]
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
def train_gate_all() -> None:
    """S1 ablation: train all three sampling strategies (spec §2.5)."""
    for sampler in ("A", "B", "C"):
        print(f"=== sampler {sampler} ===")
        print(train_gate_sampler.remote(sampler=sampler))


@app.local_entrypoint()
def audit(model_dir: str = f"{MODEL_DIR}/gate_B", value_predictions: str = "") -> None:
    print(run_audit.remote(model_dir=model_dir, value_predictions=value_predictions))
