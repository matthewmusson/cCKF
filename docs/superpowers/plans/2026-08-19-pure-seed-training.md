# Pure-Seed Training & Tier 3 Value Target Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train gate and value function on pure seeds only (3/3 seed hits from majority particle) with Tier 2 targets, and prepare infrastructure for Tier 3 value targets via ACTS oracle CKF.

**Architecture:** Add seed-purity filtering to existing cache builders via a `--pure-seeds-only` flag. Pure-seed caches go to separate directories (`gate_pure/`, `value_pure/`). The same training scripts run on the filtered caches. A shared `cckf/seed_purity.py` module extracts the purity computation from `analyze_value_targets.py`. Tier 3 oracle CKF is a separate Track blocked on the ACTS build landing.

**Tech Stack:** Python 3.10+, PyTorch, PyArrow, Modal (GPU compute), W&B

## Global Constraints

- Data split frozen per `cckf/splits.py`: events [0,32) for train/val/cal (24/4/4), events [32,64) sealed test set. Never touch test events.
- Norm stats computed on train split only, reused verbatim for val/cal.
- Gate loss: unweighted BCE (spec §9.2). No positive reweighting in primary.
- Value target: V^{π†} = min(completeness, purity) with DM matching (spec §11.1).
- Branch on `feat/pure-seed-training`, never commit to `main` directly.
- Gate reads from SELECTED_DIR (needs `is_ckf_selected` for seed purity computation).
- Value reads from SELECTED_DIR (already the default).
- Pure-seed caches write to `{CACHE_DIR}/gate_pure/{split}` and `{CACHE_DIR}/value_pure/{split}`, never overwriting existing caches.

---

## Track 1: Pure-Seed T2 Training (immediate, Python-only)

### Task 1: Create branch and extract seed purity module

**Files:**
- Create: `cckf/seed_purity.py`
- Modify: `scripts/analyze_value_targets.py` (import from shared module)
- Test: `tests/test_seed_purity.py`

**Interfaces:**
- Consumes: Parquet columns `seed_id`, `branch_id`, `step_k`, `is_ckf_selected`, `cand_hit_id`, `contrib_pids`, `branch_majority_pid`
- Produces: `compute_pure_seed_set(parquet_path) -> set[tuple[int, int]]` returning `{(seed_id, branch_id)}` pairs that are pure (3/3)

- [ ] **Step 1: Create branch**

```bash
git checkout -b feat/pure-seed-training main
```

- [ ] **Step 2: Write test for seed purity computation**

```python
# tests/test_seed_purity.py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from cckf.seed_purity import classify_seed_purity, compute_pure_seed_set


def _make_parquet(tmp_path, rows):
    """Write a minimal Parquet with the columns needed for seed purity."""
    df = pd.DataFrame(rows)
    # contrib_pids must be a list column
    table = pa.table({
        "seed_id": pa.array(df["seed_id"], pa.int64()),
        "branch_id": pa.array(df["branch_id"], pa.int64()),
        "step_k": pa.array(df["step_k"], pa.int64()),
        "is_ckf_selected": pa.array(df["is_ckf_selected"], pa.bool_()),
        "cand_hit_id": pa.array(df["cand_hit_id"], pa.int64()),
        "contrib_pids": pa.array(df["contrib_pids"], pa.list_(pa.int64())),
        "branch_majority_pid": pa.array(df["branch_majority_pid"], pa.int64()),
        "majority_undefined": pa.array([False] * len(df), pa.bool_()),
    })
    path = tmp_path / "test.parquet"
    pq.write_table(table, path)
    return path


def test_pure_seed_3_of_3(tmp_path):
    """All 3 seed hits from majority particle → pure."""
    rows = [
        # seed 0, branch 0: 3 selected measurement hits, all from pid=100
        {"seed_id": 0, "branch_id": 0, "step_k": 0, "is_ckf_selected": True,
         "cand_hit_id": 10, "contrib_pids": [100], "branch_majority_pid": 100},
        {"seed_id": 0, "branch_id": 0, "step_k": 1, "is_ckf_selected": True,
         "cand_hit_id": 11, "contrib_pids": [100], "branch_majority_pid": 100},
        {"seed_id": 0, "branch_id": 0, "step_k": 2, "is_ckf_selected": True,
         "cand_hit_id": 12, "contrib_pids": [100], "branch_majority_pid": 100},
    ]
    path = _make_parquet(tmp_path, rows)
    pure_set = compute_pure_seed_set(path)
    assert (0, 0) in pure_set


def test_majority_seed_2_of_3(tmp_path):
    """2 of 3 seed hits from majority → majority (not pure)."""
    rows = [
        {"seed_id": 0, "branch_id": 0, "step_k": 0, "is_ckf_selected": True,
         "cand_hit_id": 10, "contrib_pids": [100], "branch_majority_pid": 100},
        {"seed_id": 0, "branch_id": 0, "step_k": 1, "is_ckf_selected": True,
         "cand_hit_id": 11, "contrib_pids": [200], "branch_majority_pid": 100},
        {"seed_id": 0, "branch_id": 0, "step_k": 2, "is_ckf_selected": True,
         "cand_hit_id": 12, "contrib_pids": [100], "branch_majority_pid": 100},
    ]
    path = _make_parquet(tmp_path, rows)
    pure_set = compute_pure_seed_set(path)
    assert (0, 0) not in pure_set


def test_holes_skipped(tmp_path):
    """Hole rows (cand_hit_id == -1) should be skipped; purity computed from measurements only."""
    rows = [
        {"seed_id": 0, "branch_id": 0, "step_k": 0, "is_ckf_selected": True,
         "cand_hit_id": -1, "contrib_pids": [], "branch_majority_pid": 100},
        {"seed_id": 0, "branch_id": 0, "step_k": 1, "is_ckf_selected": True,
         "cand_hit_id": 11, "contrib_pids": [100], "branch_majority_pid": 100},
        {"seed_id": 0, "branch_id": 0, "step_k": 2, "is_ckf_selected": True,
         "cand_hit_id": 12, "contrib_pids": [100], "branch_majority_pid": 100},
        {"seed_id": 0, "branch_id": 0, "step_k": 3, "is_ckf_selected": True,
         "cand_hit_id": 13, "contrib_pids": [100], "branch_majority_pid": 100},
    ]
    path = _make_parquet(tmp_path, rows)
    pure_set = compute_pure_seed_set(path)
    assert (0, 0) in pure_set
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_seed_purity.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'cckf.seed_purity'`

- [ ] **Step 4: Implement `cckf/seed_purity.py`**

```python
"""Seed purity classification.

Classifies each (seed_id, branch_id) as 'pure' (3/3 seed hits from the
majority particle) or 'majority' (2/3). Extracted from
analyze_value_targets.py for reuse by the cache builders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import labels as lab

_PURITY_COLUMNS: tuple[str, ...] = (
    "seed_id",
    "branch_id",
    "step_k",
    "is_ckf_selected",
    "cand_hit_id",
    "contrib_pids",
    "branch_majority_pid",
    "majority_undefined",
)


def classify_seed_purity(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each (seed_id, branch_id) as 'pure' or 'majority'.

    Finds the first 3 CKF-selected measurement hits per branch (by step_k
    order), counts how many have label_same_particle == 1.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: seed_id, branch_id, step_k, is_ckf_selected,
        cand_hit_id, label_same_particle.

    Returns
    -------
    pd.DataFrame
        Columns: seed_id, branch_id, seed_purity ('pure' or 'majority').
    """
    sel = df["is_ckf_selected"].to_numpy(dtype=bool)
    hit = df["cand_hit_id"].to_numpy(dtype=np.int64) != -1
    meas = df.loc[
        sel & hit, ["seed_id", "branch_id", "step_k", "label_same_particle"]
    ].copy()
    meas = meas.sort_values(["seed_id", "branch_id", "step_k"])
    meas["rank"] = meas.groupby(["seed_id", "branch_id"]).cumcount()
    seed_hits = meas.loc[meas["rank"] < 3].copy()
    seed_hits["is_same"] = (
        seed_hits["label_same_particle"].to_numpy(dtype=np.int64) == 1
    ).astype(np.int64)

    per_branch = seed_hits.groupby(
        ["seed_id", "branch_id"], as_index=False
    ).agg(n_seed_same=("is_same", "sum"))
    per_branch["seed_purity"] = np.where(
        per_branch["n_seed_same"] >= 3, "pure", "majority"
    )
    return per_branch[["seed_id", "branch_id", "seed_purity"]]


def compute_pure_seed_set(parquet_path: str) -> set[tuple[int, int]]:
    """Return the set of (seed_id, branch_id) that are pure (3/3).

    Reads only the columns needed for purity computation, derives
    label_same_particle via labels.derive_labels, then classifies.

    Parameters
    ----------
    parquet_path : str or Path
        Path to an expanded Parquet file with is_ckf_selected column.

    Returns
    -------
    set of (int, int)
        Pure-seed (seed_id, branch_id) pairs.
    """
    table = pq.read_table(parquet_path, columns=list(_PURITY_COLUMNS))
    derived = lab.derive_labels(table)
    df = table.to_pandas()
    df["label_same_particle"] = derived["label_same_particle"]

    purity = classify_seed_purity(df)
    pure = purity.loc[purity["seed_purity"] == "pure"]
    return set(zip(pure["seed_id"].tolist(), pure["branch_id"].tolist()))
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_seed_purity.py -v
```
Expected: all 3 tests PASS

- [ ] **Step 6: Update `analyze_value_targets.py` to import from shared module**

Replace `_seed_purity` in `scripts/analyze_value_targets.py` with an import:

```python
from cckf.seed_purity import classify_seed_purity
```

Then change the call site from `_seed_purity(df)` to `classify_seed_purity(df)`.

- [ ] **Step 7: Commit**

```bash
git add cckf/seed_purity.py tests/test_seed_purity.py scripts/analyze_value_targets.py
git commit -m "feat: extract seed purity computation to shared module"
```

---

### Task 2: Add pure-seed filtering to gate cache builder

**Files:**
- Modify: `cckf/cache.py` — add `pure_seed_sets` parameter to `build_gate_cache`
- Modify: `scripts/build_gate_cache.py` — add `--pure-seeds-only` CLI flag
- Test: `tests/test_gate_cache_pure.py`

**Interfaces:**
- Consumes: `cckf.seed_purity.compute_pure_seed_set(path) -> set[tuple[int, int]]`
- Produces: Gate cache at `{out_dir}/` with only pure-seed rows; `meta.json` includes `"pure_seeds_only": true`

**Important:** When `--pure-seeds-only` is set, `--parquet-dir` must point to SELECTED_DIR (which has the `is_ckf_selected` column needed for seed purity computation). The script should validate this.

- [ ] **Step 1: Write test for pure-seed gate cache filtering**

```python
# tests/test_gate_cache_pure.py
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from cckf.cache import build_gate_cache


def _make_parquet_with_selected(tmp_path, n_pure=10, n_majority=30):
    """Create a minimal Parquet with both pure and majority seeds."""
    rows = []
    # Pure seed: seed_id=0, branch_id=0
    for step in range(n_pure):
        rows.append({
            "seed_id": 0, "branch_id": 0, "step_k": step,
            "is_ckf_selected": True, "cand_hit_id": step + 100,
            "contrib_pids": [1], "branch_majority_pid": 1,
            "majority_undefined": False, "n_window": 5,
            "residual_l0": 0.01, "residual_l1": 0.02,
            "S00": 0.1, "S01": 0.0, "S11": 0.1,
            "chi2_inc": 2.0, "state_theta": 1.0, "state_qop": 0.5,
            "clus_s_u": 0.01, "clus_s_v": 0.02, "clus_q_tot": 100.0,
            "clus_sigma_uu": 0.001, "clus_sigma_uv": 0.0, "clus_sigma_vv": 0.001,
            "alpha_u": 0.1, "alpha_v": 0.2,
            "pitch_u": 0.05, "pitch_v": 0.05, "thickness": 0.15,
            "is_pixel": True, "is_barrel": True,
            "n_hits": step + 1, "n_holes": 0, "n_seq_holes": 0,
            "pathInX0_interval": 0.01,
        })
    # Majority seed: seed_id=1, branch_id=0 (2/3 from pid=2, 1/3 from pid=3)
    for step in range(n_majority):
        pid = 2 if step != 1 else 3  # step 1 is from wrong particle
        rows.append({
            "seed_id": 1, "branch_id": 0, "step_k": step,
            "is_ckf_selected": True, "cand_hit_id": step + 200,
            "contrib_pids": [pid], "branch_majority_pid": 2,
            "majority_undefined": False, "n_window": 5,
            "residual_l0": 0.01, "residual_l1": 0.02,
            "S00": 0.1, "S01": 0.0, "S11": 0.1,
            "chi2_inc": 2.0, "state_theta": 1.0, "state_qop": 0.5,
            "clus_s_u": 0.01, "clus_s_v": 0.02, "clus_q_tot": 100.0,
            "clus_sigma_uu": 0.001, "clus_sigma_uv": 0.0, "clus_sigma_vv": 0.001,
            "alpha_u": 0.1, "alpha_v": 0.2,
            "pitch_u": 0.05, "pitch_v": 0.05, "thickness": 0.15,
            "is_pixel": True, "is_barrel": True,
            "n_hits": step + 1, "n_holes": 0, "n_seq_holes": 0,
            "pathInX0_interval": 0.01,
        })
    # Build Parquet with list columns for contrib_pids
    df = pd.DataFrame(rows)
    arrays = {}
    for col in df.columns:
        if col == "contrib_pids":
            arrays[col] = pa.array(df[col].tolist(), pa.list_(pa.int64()))
        elif col in ("is_ckf_selected", "majority_undefined", "is_pixel", "is_barrel"):
            arrays[col] = pa.array(df[col], pa.bool_())
        elif col in ("seed_id", "branch_id", "step_k", "cand_hit_id",
                      "branch_majority_pid", "n_window", "n_hits", "n_holes", "n_seq_holes"):
            arrays[col] = pa.array(df[col], pa.int64())
        else:
            arrays[col] = pa.array(df[col], pa.float64())
    table = pa.table(arrays)
    path = tmp_path / "test.parquet"
    pq.write_table(table, path)
    return path


def test_pure_seed_filter_reduces_rows(tmp_path):
    """With pure_seeds_only, cache should contain only pure-seed rows."""
    path = _make_parquet_with_selected(tmp_path, n_pure=10, n_majority=30)
    out_all = tmp_path / "cache_all"
    out_pure = tmp_path / "cache_pure"

    meta_all = build_gate_cache([path], out_all)
    from cckf.seed_purity import compute_pure_seed_set
    pure_set = {path: compute_pure_seed_set(str(path))}
    meta_pure = build_gate_cache([path], out_pure, pure_seed_sets=pure_set)

    assert meta_pure["n_rows"] < meta_all["n_rows"]
    assert meta_pure["n_rows"] == 10  # only pure seed's 10 rows
    assert meta_pure.get("pure_seeds_only") is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_gate_cache_pure.py -v
```
Expected: FAIL — `build_gate_cache` doesn't accept `pure_seed_sets` yet

- [ ] **Step 3: Modify `cckf/cache.py:build_gate_cache` to accept pure-seed filtering**

Add `pure_seed_sets: dict[Path, set[tuple[int, int]]] | None = None` parameter. When provided, after applying `gate_row_mask`, additionally filter rows to those whose `(seed_id, branch_id)` is in the set for the current file.

Key implementation detail: the filtering goes after `gate_row_mask` and before feature construction:

```python
if pure_seed_sets is not None:
    pure_set = pure_seed_sets.get(path)
    if pure_set is not None:
        sb_pairs = list(zip(
            df["seed_id"].to_numpy(dtype=np.int64),
            df["branch_id"].to_numpy(dtype=np.int64),
        ))
        pure_mask = np.array([(s, b) in pure_set for s, b in sb_pairs])
        df = df.loc[pure_mask].reset_index(drop=True)
        y = y[pure_mask]
        ambiguous = ambiguous[pure_mask]
        if len(df) == 0:
            continue
```

Also update `CacheWriter.close()` to accept and record `pure_seeds_only` in meta.json.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_gate_cache_pure.py -v
```
Expected: PASS

- [ ] **Step 5: Add `--pure-seeds-only` flag to `scripts/build_gate_cache.py`**

```python
parser.add_argument(
    "--pure-seeds-only",
    action="store_true",
    help="Filter to pure seeds (3/3 seed hits from majority particle). "
         "Requires --parquet-dir to point at selected Parquets "
         "(with is_ckf_selected column).",
)
```

When set:
1. Before the cache build loop, pre-compute `compute_pure_seed_set(path)` for each Parquet file
2. Pass the dict to `build_gate_cache(..., pure_seed_sets=sets)`
3. Mark `meta["pure_seeds_only"] = True`

- [ ] **Step 6: Commit**

```bash
git add cckf/cache.py scripts/build_gate_cache.py tests/test_gate_cache_pure.py
git commit -m "feat: add pure-seed filtering to gate cache builder"
```

---

### Task 3: Add pure-seed filtering to value cache builder

**Files:**
- Modify: `scripts/build_value_cache.py` — add `--pure-seeds-only` flag and filtering in `process_event()`
- Test: manual verification via cache meta.json row counts

**Interfaces:**
- Consumes: `cckf.seed_purity.compute_pure_seed_set(path) -> set[tuple[int, int]]`
- Produces: Value cache at `{out_dir}/` with only pure-seed rows; `meta.json` includes `"pure_seeds_only": true`

- [ ] **Step 1: Add `--pure-seeds-only` flag to `scripts/build_value_cache.py`**

```python
parser.add_argument(
    "--pure-seeds-only",
    action="store_true",
    help="Filter to pure seeds (3/3 seed hits from majority particle).",
)
```

- [ ] **Step 2: Add filtering in `process_event()`**

Accept `pure_seed_set: set[tuple[int, int]] | None` as an optional parameter. When set, after reading the Parquet and filtering `~majority_undefined`, additionally filter `df` to rows whose `(seed_id, branch_id)` is in the set:

```python
if pure_seed_set is not None:
    sb_pairs = list(zip(
        df["seed_id"].to_numpy(dtype=np.int64),
        df["branch_id"].to_numpy(dtype=np.int64),
    ))
    pure_mask = np.array([(s, b) in pure_seed_set for s, b in sb_pairs])
    df = df.loc[pure_mask].reset_index(drop=True)
```

- [ ] **Step 3: Wire the flag in `main()`**

Before the event loop, compute pure seed sets when the flag is set:

```python
if args.pure_seeds_only:
    from cckf.seed_purity import compute_pure_seed_set
    pure_sets = {}
    for event_id in events:
        path = Path(args.parquet_dir) / f"expanded_event{event_id:09d}.parquet"
        pure_sets[event_id] = compute_pure_seed_set(str(path))
        print(f"event {event_id}: {len(pure_sets[event_id]):,} pure branches")
```

Then pass `pure_seed_set=pure_sets.get(event_id)` to `process_event()`.

Add `"pure_seeds_only": True` to `meta` dict when the flag is set.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_value_cache.py
git commit -m "feat: add pure-seed filtering to value cache builder"
```

---

### Task 4: Add Modal entrypoints for pure-seed pipeline

**Files:**
- Modify: `modal_train.py` — add `build_pure_caches`, `train_gate_pure`, `train_value_pure` functions

**Interfaces:**
- Consumes: `build_gate_cache.py --pure-seeds-only`, `build_value_cache.py --pure-seeds-only`, `train_gate.py`, `train_value.py`
- Produces: Modal entrypoints callable via `modal run modal_train.py::build_pure_caches`, `modal run modal_train.py::train_pure_all`

- [ ] **Step 1: Add `build_pure_caches` function**

```python
@app.function(
    image=image, volumes={DATA_PATH: data_vol}, cpu=16, memory=262144, timeout=86400
)
def build_pure_caches(splits_to_build: str = "train,val,cal") -> dict:
    """Build gate and value caches filtered to pure seeds only."""
    import sys

    results = {}
    splits = [s.strip() for s in splits_to_build.split(",") if s.strip()]

    for split in splits:
        # Gate cache: reads from SELECTED_DIR (needs is_ckf_selected for purity)
        _run_script([
            sys.executable, "/root/scripts/build_gate_cache.py",
            "--split", split,
            "--parquet-dir", SELECTED_DIR,
            "--out-dir", f"{CACHE_DIR}/gate_pure/{split}",
            "--pure-seeds-only",
        ])
        data_vol.commit()
        results[f"gate_pure_{split}"] = "ok"

        # Value cache
        _run_script([
            sys.executable, "/root/scripts/build_value_cache.py",
            "--split", split,
            "--parquet-dir", SELECTED_DIR,
            "--out-dir", f"{CACHE_DIR}/value_pure/{split}",
            "--pure-seeds-only",
        ])
        data_vol.commit()
        results[f"value_pure_{split}"] = "ok"

    return results
```

- [ ] **Step 2: Add `train_gate_pure` function**

```python
@app.function(
    image=image, volumes={DATA_PATH: data_vol}, gpu="A10G",
    memory=131072, timeout=43200,
    secrets=[modal.Secret.from_name("wandb")],
)
def train_gate_pure(wandb_project: str = "cckf-gate-pure") -> dict:
    """Train gate on pure-seed cache with sampler A (no subsampling)."""
    import json, sys

    out_dir = f"{MODEL_DIR}/gate_pure_A"
    _run_script([
        sys.executable, "/root/scripts/train_gate.py",
        "--train-cache", f"{CACHE_DIR}/gate_pure/train",
        "--val-cache", f"{CACHE_DIR}/gate_pure/val",
        "--out-dir", out_dir,
        "--sampler", "A",
        "--device", "cuda",
        "--wandb-project", wandb_project,
    ])
    data_vol.commit()
    with open(f"{out_dir}/gate_metrics.json") as fh:
        return json.load(fh)
```

- [ ] **Step 3: Add `train_value_pure` function**

```python
@app.function(
    image=image, volumes={DATA_PATH: data_vol}, gpu="A10G",
    memory=131072, timeout=43200,
    secrets=[modal.Secret.from_name("wandb")],
)
def train_value_pure(wandb_project: str = "cckf-value-pure", seed: int = 0) -> dict:
    """Train value function V_φ on pure-seed cache with T2 targets."""
    import json, sys

    out_dir = f"{MODEL_DIR}/value_pure_v0"
    cmd = [
        sys.executable, "/root/scripts/train_value.py",
        "--train-cache", f"{CACHE_DIR}/value_pure/train",
        "--val-cache", f"{CACHE_DIR}/value_pure/val",
        "--out-dir", out_dir,
        "--device", "cuda",
        "--seed", str(seed),
    ]
    if wandb_project:
        cmd += ["--wandb-project", wandb_project]
    _run_script(cmd)
    data_vol.commit()
    with open(f"{out_dir}/value_metrics.json") as fh:
        return json.load(fh)
```

- [ ] **Step 4: Add local entrypoints**

```python
@app.local_entrypoint()
def build_pure(splits: str = "train,val,cal") -> None:
    """Build pure-seed caches. Usage: modal run --detach modal_train.py::build_pure"""
    import json
    print(json.dumps(build_pure_caches.remote(splits_to_build=splits), indent=2))


@app.local_entrypoint()
def train_pure_all(skip_cache: bool = False) -> None:
    """Full pure-seed pipeline: build caches → train gate → train value.
    Usage: modal run --detach modal_train.py::train_pure_all"""
    import json

    if not skip_cache:
        print("=== building pure-seed caches ===")
        print(json.dumps(build_pure_caches.remote(splits_to_build="train,val,cal"), indent=2))

    print("=== training gate (pure) ===")
    gate_metrics = train_gate_pure.remote()
    print(json.dumps(gate_metrics, indent=2))

    print("=== training value (pure, T2) ===")
    value_metrics = train_value_pure.remote()
    print(json.dumps(value_metrics, indent=2))
```

- [ ] **Step 5: Commit**

```bash
git add modal_train.py
git commit -m "feat: add Modal entrypoints for pure-seed training pipeline"
```

---

### Task 5: Run pure-seed training pipeline on Modal

**Files:** None (execution only, no code changes)

**Interfaces:**
- Consumes: All code from Tasks 1-4
- Produces: Trained models at `{MODEL_DIR}/gate_pure_A/` and `{MODEL_DIR}/value_pure_v0/` on Modal volume

- [ ] **Step 1: Build pure-seed caches**

```bash
modal run --detach modal_train.py::build_pure
```

Monitor progress. Expected output: cache meta.json with ~3.2% of the original row count (pure seeds are ~3.2% of all seeds).

- [ ] **Step 2: Train gate and value on pure-seed caches**

```bash
modal run --detach modal_train.py::train_pure_all --skip-cache
```

Or run the full pipeline:
```bash
modal run --detach modal_train.py::train_pure_all
```

- [ ] **Step 3: Run calibration audit on pure-seed gate**

```bash
modal run modal_train.py::audit --model-dir /data/models/gate_pure_A
```

- [ ] **Step 4: Log results to experiments/LOG.md**

Record: cache sizes, training metrics (BCE, AUC, MSE), calibration audit results, comparison to full-data gate.

---

### Task 6: Comparison evaluation plots

**Files:**
- Create: `scripts/compare_pure_vs_all.py` — comparison plot script
- Test: visual inspection of output plots

**Interfaces:**
- Consumes: Model predictions from `gate_pure_A/`, `value_pure_v0/`, `gate_A/`, `value_v0/`
- Produces: Comparison plots in `analysis/plots/`

- [ ] **Step 1: Write comparison script**

Produces side-by-side plots comparing:
1. Gate calibration: reliability diagram for pure-seed gate vs full gate
2. V_φ predictions vs eta/occupancy/step_k (using the formulation from `plot_value_distributions.py`)
3. Summary table of metrics

The script should:
- Load both models' predictions and calibration data
- Use the same binning and axes as the existing plots
- Output comparison figures

- [ ] **Step 2: Run comparison and generate plots**

```bash
python scripts/compare_pure_vs_all.py \
    --all-gate-dir /data/models/gate_A \
    --pure-gate-dir /data/models/gate_pure_A \
    --all-value-dir /data/models/value_v0 \
    --pure-value-dir /data/models/value_pure_v0 \
    --out-dir analysis/plots/pure_comparison
```

- [ ] **Step 3: Commit**

```bash
git add scripts/compare_pure_vs_all.py
git commit -m "feat: add pure-vs-all comparison evaluation plots"
```

---

## Track 2: Tier 3 Oracle CKF + Training (blocked on ACTS build)

**Status:** Blocked. Requires the ACTS build to pass with cCKF integration (being done by the other agent on `worktree-acts-instrumentation-plan`).

### Design summary

**Tier 3 value target** (V^{π†}_T3): For each (branch, step_k), inject the majority particle's truth hits from step_k onward and let the ACTS CKF repropagate to termination. Measure completeness and purity of the resulting track. V^{π†}_T3 = min(completeness, purity).

This breaks the chicken-and-egg problem: Tier 2 says "this branch is doomed because it visits different surfaces" → V_φ kills it → branch never recovers → label confirmed. Tier 3 actually tests whether recovery is possible.

**Implementation requires:**
1. `OracleMeasurementSelector` — C++ component that replaces `CckfMeasurementSelector` and injects truth hits instead of scoring candidates
2. `modal_oracle_ckf.py` — Modal entrypoint that runs the oracle CKF on all 32 events, collecting (seed_id, branch_id, step_k, vstar_t3)
3. `scripts/build_value_cache_t3.py` — Cache builder that reads T3 targets instead of T2

**Scope:** ~29.5M repropagations (one per (branch, step_k) pair across 32 events). Each is a partial CKF run from one surface to termination, so wall-clock is manageable with Modal parallelism.

**Prerequisite:** ACTS build with cCKF integration must pass. The other agent is working on this. Once it lands, this track produces:
- V_φ(all, T3) — value function trained on all seeds with T3 targets
- V_φ(pure, T3) — value function trained on pure seeds with T3 targets

### Tasks (to be detailed when ACTS build lands)

- Task T2.1: Implement OracleMeasurementSelector in C++
- Task T2.2: Add oracle CKF Modal pipeline
- Task T2.3: Compute Tier 3 labels for all 32 events
- Task T2.4: Build T3 value caches (all + pure)
- Task T2.5: Train V_φ(all, T3) and V_φ(pure, T3)
- Task T2.6: Full comparative evaluation (6 models)
