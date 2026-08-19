"""Streaming Parquet → memmap feature cache.

The expanded dataset is ~1.44B rows across all 32 events, of which ~137M (9.5%) are
trainable for the gate. At 26 float32 features that is ~14 GB — too large for
memory, but trivial to memmap. This module streams Parquet row groups, applies
the row-inclusion mask, derives features, and appends to flat binary files that
``numpy.memmap`` can index lazily during training.

Files written per cache directory
---------------------------------
``X.f32``
    Feature matrix, C-order ``(n_rows, n_features)`` float32.
``y.u8``
    Gate label, uint8 in {0, 1}.
``aux.f32``
    ``(n_rows, 3)`` float32 of ``chi2_inc``, ``n_window``, ``eta``. Kept
    separately from X because the calibration audit strata (§4.2) and the
    hard-negative sampler (§2.5 Approach C) need these *unstandardised*, while
    X gets z-scored. A NaN ``chi2_inc`` (e.g. a hole row with no residual) is
    mapped to ``+inf`` in this column, not left as NaN, so that
    ``cckf.samplers.hard_negative_subsample`` -- the sole consumer of
    ``aux[:, 0]`` -- can treat it as maximally easy (weight -> 0) with a plain
    ``np.isfinite`` check instead of also special-casing NaN.
``ambiguous.u8``
    Merged-cluster flag, for ablation A8b.
``meta.json``
    Row/positive counts, feature names, aux column names, source files.

Only one pass over the Parquet is ever needed: all three §2.5 sampling
strategies are index-level samplers over this single cache, not separate caches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import features as feat
from . import labels as lab

#: Columns the gate cache must read from Parquet.
_READ_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys((*feat.GATE_SOURCE_COLUMNS, *lab.LABEL_COLUMNS))
)

_AUX_COLUMNS: tuple[str, ...] = ("chi2_inc", "n_window", "eta")


class CacheWriter:
    """Append-only writer for the flat cache files."""

    def __init__(self, out_dir: Path | str, n_features: int) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.n_features = n_features
        self._fx = open(self.out_dir / "X.f32", "wb")
        self._fy = open(self.out_dir / "y.u8", "wb")
        self._fa = open(self.out_dir / "aux.f32", "wb")
        self._fm = open(self.out_dir / "ambiguous.u8", "wb")
        self.n_rows = 0
        self.n_positive = 0

    def append(
        self, X: np.ndarray, y: np.ndarray, aux: np.ndarray, ambiguous: np.ndarray
    ) -> None:
        """Append one batch. Arrays must share a leading dimension."""
        if X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X.shape[1]}")
        if not (len(X) == len(y) == len(aux) == len(ambiguous)):
            raise ValueError("batch arrays have mismatched lengths")
        self._fx.write(np.ascontiguousarray(X, dtype=np.float32).tobytes())
        self._fy.write(np.ascontiguousarray(y, dtype=np.uint8).tobytes())
        self._fa.write(np.ascontiguousarray(aux, dtype=np.float32).tobytes())
        self._fm.write(np.ascontiguousarray(ambiguous, dtype=np.uint8).tobytes())
        self.n_rows += len(X)
        self.n_positive += int(y.sum())

    def close(
        self, source_files: Sequence[str], pure_seeds_only: bool = False
    ) -> dict:
        """Flush, write ``meta.json``, and return the summary report.

        Parameters
        ----------
        source_files : sequence of str
            Parquet paths this cache was built from, recorded verbatim.
        pure_seeds_only : bool
            Whether rows were filtered to pure seeds (spec's pure-seed
            training ablation). Recorded in ``meta.json`` so a downstream
            consumer can tell a pure-seed cache apart from a full cache
            without re-deriving it from the source files.
        """
        for handle in (self._fx, self._fy, self._fa, self._fm):
            handle.close()
        meta = {
            "n_rows": self.n_rows,
            "n_positive": self.n_positive,
            "positive_fraction": (
                (self.n_positive / self.n_rows) if self.n_rows else 0.0
            ),
            "n_features": self.n_features,
            "feature_names": list(feat.GATE_FEATURES),
            "aux_columns": list(_AUX_COLUMNS),
            "source_files": [str(s) for s in source_files],
            "pure_seeds_only": pure_seeds_only,
        }
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta


def build_gate_cache(
    parquet_paths: Iterable[Path | str],
    out_dir: Path | str,
    batch_rows: int = 1_000_000,
    pure_seed_sets: dict[Path, set[tuple[int, int]]] | None = None,
) -> dict:
    """Stream Parquet files into a gate feature cache.

    Parameters
    ----------
    parquet_paths : iterable of path
        Expanded Parquet files, all belonging to the *same* split.
    out_dir : path
        Cache directory to create.
    batch_rows : int
        Rows per streaming batch. Memory use is roughly
        ``batch_rows × 26 × 8 bytes`` during derivation.
    pure_seed_sets : dict of Path to set of (int, int), optional
        When given, restricts the cache to rows whose ``(seed_id,
        branch_id)`` is a "pure" seed (spec's pure-seed training ablation
        -- see ``cckf.seed_purity.compute_pure_seed_set``). Keyed by the
        resolved Parquet path so each file is filtered against its own
        pure set; a file with no entry in the dict is left unfiltered
        (all its trainable rows pass through, same as ``pure_seed_sets
        is None``). Applied after ``gate_row_mask`` and before feature
        construction. ``None`` (the default) disables filtering entirely
        and behaves exactly as before.

    Returns
    -------
    dict
        The ``meta.json`` contents.
    """
    paths = [Path(p) for p in parquet_paths]
    writer = CacheWriter(out_dir, n_features=len(feat.GATE_FEATURES))

    # Pure-seed filtering keys on (seed_id, branch_id), which the gate
    # feature/label columns alone don't include -- pull them in too, but
    # only when the caller actually asked for filtering, so a plain (no
    # pure_seed_sets) run's column list -- and therefore its cache
    # contents -- is unchanged from before this parameter existed.
    read_columns = _READ_COLUMNS
    if pure_seed_sets is not None:
        read_columns = tuple(
            dict.fromkeys((*_READ_COLUMNS, "seed_id", "branch_id"))
        )

    for path in paths:
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        columns = [c for c in read_columns if c in available]
        missing = set(read_columns) - available
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
            table = batch if hasattr(batch, "num_rows") else None
            if table is None:  # pragma: no cover - defensive
                continue
            import pyarrow as pa

            tbl = pa.Table.from_batches([batch])
            derived = lab.derive_labels(tbl)
            mask = derived["gate_row_mask"]
            if not mask.any():
                continue

            df = tbl.to_pandas()
            df = df.loc[mask].reset_index(drop=True)
            y = derived["label_same_particle"][mask]
            ambiguous = derived["label_ambiguous"][mask]

            if pure_seed_sets is not None:
                pure_set = pure_seed_sets.get(path)
                if pure_set is not None:
                    # Vectorised membership test rather than a per-row
                    # `(s, b) in pure_set` loop: at ~1M rows/batch a Python
                    # loop dominates wall time, while a MultiIndex.isin is a
                    # single vectorised hash-join.
                    if pure_set:
                        pure_index = pd.MultiIndex.from_tuples(
                            pure_set, names=["seed_id", "branch_id"]
                        )
                        row_index = pd.MultiIndex.from_arrays(
                            [df["seed_id"].to_numpy(), df["branch_id"].to_numpy()]
                        )
                        pure_mask = row_index.isin(pure_index)
                    else:
                        pure_mask = np.zeros(len(df), dtype=bool)
                    df = df.loc[pure_mask].reset_index(drop=True)
                    y = y[pure_mask]
                    ambiguous = ambiguous[pure_mask]
                    if len(df) == 0:
                        continue

            X = feat.build_gate_features(df)
            aux = np.column_stack(
                [
                    # NaN -> +inf (see module docstring, "aux.f32"): a NaN chi2_inc
                    # means no residual to compute one, and +inf reads to the
                    # hard-negative sampler as maximally easy rather than as a
                    # missing value it would need to special-case.
                    np.nan_to_num(
                        df["chi2_inc"].to_numpy(dtype=np.float64), nan=np.inf
                    ),
                    df["n_window"].to_numpy(dtype=np.float64),
                    feat.eta_from_theta(df["state_theta"].to_numpy(dtype=np.float64)),
                ]
            ).astype(np.float32)

            writer.append(X, y, aux, ambiguous.astype(np.uint8))

    return writer.close(
        [str(p) for p in paths], pure_seeds_only=pure_seed_sets is not None
    )


def load_cache(out_dir: Path | str) -> dict:
    """Memmap a cache directory.

    Returns
    -------
    dict
        ``X`` (float32 memmap, ``(n, F)``), ``y`` (uint8 memmap),
        ``aux`` (float32 memmap, ``(n, 3)``), ``ambiguous`` (uint8 memmap),
        ``meta`` (dict).
    """
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text())
    n, f = meta["n_rows"], meta["n_features"]
    return {
        "X": np.memmap(out_dir / "X.f32", dtype=np.float32, mode="r", shape=(n, f)),
        "y": np.memmap(out_dir / "y.u8", dtype=np.uint8, mode="r", shape=(n,)),
        "aux": np.memmap(out_dir / "aux.f32", dtype=np.float32, mode="r", shape=(n, 3)),
        "ambiguous": np.memmap(
            out_dir / "ambiguous.u8", dtype=np.uint8, mode="r", shape=(n,)
        ),
        "meta": meta,
    }


def compute_norm_stats(
    X: np.ndarray,
    skip: Iterable[str],
    names: Sequence[str],
    chunk: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and standard deviation over the training cache.

    Computed in chunks with a streaming sum / sum-of-squares so the full matrix
    is never resident. Features named in ``skip`` get ``(mu, sigma) = (0, 1)``,
    i.e. pass-through — per spec §2.3 the integer branch counters are not
    standardised. Zero-variance features are floored to ``sigma = 1`` so a
    constant column becomes a constant zero rather than NaN.

    Returns
    -------
    tuple of numpy.ndarray
        ``(mu, sigma)``, each shape ``(n_features,)`` float32.
    """
    n_rows, n_feat = X.shape
    total = np.zeros(n_feat, dtype=np.float64)
    total_sq = np.zeros(n_feat, dtype=np.float64)

    for start in range(0, n_rows, chunk):
        block = np.asarray(X[start : start + chunk], dtype=np.float64)
        total += block.sum(axis=0)
        total_sq += (block * block).sum(axis=0)

    mu = total / max(n_rows, 1)
    var = np.maximum(total_sq / max(n_rows, 1) - mu * mu, 0.0)
    sigma = np.sqrt(var)
    sigma[sigma == 0.0] = 1.0

    skip_set = set(skip)
    for j, name in enumerate(names):
        if name in skip_set:
            mu[j] = 0.0
            sigma[j] = 1.0

    return mu.astype(np.float32), sigma.astype(np.float32)
