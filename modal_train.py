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
        "torch>=2.4", "numpy>=2.0", "pandas>=2.2", "pyarrow>=17.0",
        "scikit-learn>=1.5", "scipy>=1.14", "matplotlib>=3.9",
        "wandb>=0.17", "uproot>=5.3",
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


@app.function(image=image, volumes={DATA_PATH: data_vol}, cpu=8, memory=131072, timeout=86400)
def patch_selected_all() -> list[dict]:
    """Add ``is_ckf_selected`` to every assigned event, sequentially."""
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")
    from cckf import splits
    from scripts.patch_is_selected import patch_event

    reports = []
    events = [*splits.TRAIN_EVENTS, *splits.VAL_EVENTS, *splits.CAL_EVENTS]
    splits.assert_not_test(events)

    for event_id in sorted(events):
        src = f"{EXPANDED_DIR}/expanded_event{event_id:09d}.parquet"
        out = f"{SELECTED_DIR}/expanded_event{event_id:09d}.parquet"
        if Path(out).exists():
            print(f"event {event_id}: already patched, skipping")
            continue
        root = _find_trackstates(event_id)
        report = patch_event(src, root, event_id, out)
        print(report)
        reports.append(report)
        data_vol.commit()  # one commit per event, never concurrent
    return reports


def _find_trackstates(event_id: int) -> str:
    """Locate the trackstates ROOT file for one event on the volume.

    Stage 1 wrote 32 events across 16 batch directories, so the layout is not a
    single predictable path. Search rather than assume.
    """
    from pathlib import Path

    candidates = sorted(Path(f"{DATA_PATH}/results").rglob("trackstates_ckf.root"))
    if not candidates:
        raise FileNotFoundError("no trackstates_ckf.root anywhere under /data/results")

    import uproot

    for path in candidates:
        try:
            with uproot.open(path) as fh:
                events = set(fh["trackstates"]["event_nr"].array(library="np").tolist())
        except Exception:
            continue
        if event_id in events:
            print(f"event {event_id}: trackstates at {path}")
            return str(path)
    raise FileNotFoundError(f"no trackstates file contains event {event_id}")


@app.function(image=image, volumes={DATA_PATH: data_vol}, cpu=16, memory=262144, timeout=86400)
def build_all_caches(csv_dir: str = "") -> dict:
    """Build gate and value caches for train/val/cal, sequentially."""
    import subprocess
    import sys

    sys.path.insert(0, "/root")
    results = {}
    for split in ("train", "val", "cal"):
        cmd = [
            sys.executable, "/root/scripts/build_gate_cache.py",
            "--split", split,
            "--parquet-dir", SELECTED_DIR,
            "--out-dir", f"{CACHE_DIR}/gate/{split}",
        ]
        _run_script(cmd)
        data_vol.commit()
        results[f"gate_{split}"] = "ok"

    if csv_dir:
        for split in ("train", "val", "cal"):
            cmd = [
                sys.executable, "/root/scripts/build_value_cache.py",
                "--split", split,
                "--parquet-dir", SELECTED_DIR,
                "--csv-dir", csv_dir,
                "--out-dir", f"{CACHE_DIR}/value/{split}",
            ]
            _run_script(cmd)
            data_vol.commit()
            results[f"value_{split}"] = "ok"
    return results


@app.function(
    image=image, volumes={DATA_PATH: data_vol}, gpu="A10G",
    memory=131072, timeout=43200, secrets=[modal.Secret.from_name("wandb")],
)
def train_gate_sampler(sampler: str, wandb_project: str = "cckf-gate") -> dict:
    """Train the gate with one §2.5 sampling strategy (S1 ablation arm)."""
    import json
    import subprocess
    import sys

    out_dir = f"{MODEL_DIR}/gate_{sampler}"
    cmd = [
        sys.executable, "/root/scripts/train_gate.py",
        "--train-cache", f"{CACHE_DIR}/gate/train",
        "--val-cache", f"{CACHE_DIR}/gate/val",
        "--out-dir", out_dir,
        "--sampler", sampler,
        "--device", "cuda",
        "--wandb-project", wandb_project,
    ]
    _run_script(cmd)
    data_vol.commit()
    with open(f"{out_dir}/gate_metrics.json") as fh:
        return json.load(fh)


@app.function(image=image, volumes={DATA_PATH: data_vol}, cpu=8, memory=65536, timeout=7200)
def run_audit(model_dir: str, value_predictions: str = "") -> dict:
    """Fit Platt on the calibration split and produce the §4.2 audit.

    ``value_predictions``, when given, points at a ``value_val_predictions.npz``
    and additionally emits the V_φ-vs-Tier-2 reliability figure. Left empty for
    gate-only audits, which run before the value function is trained.
    """
    import json
    import subprocess
    import sys

    out_dir = f"{model_dir}/audit"
    cmd = [
        sys.executable, "/root/scripts/calibrate_and_audit.py",
        "--model", f"{model_dir}/gate_model.pt",
        "--cal-cache", f"{CACHE_DIR}/gate/cal",
        "--out-dir", out_dir,
    ]
    if value_predictions:
        cmd += ["--value-predictions", value_predictions]
    _run_script(cmd)
    data_vol.commit()
    with open(f"{out_dir}/calibration_audit.json") as fh:
        return json.load(fh)


@app.local_entrypoint()
def patch_selected() -> None:
    print(patch_selected_all.remote())


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
