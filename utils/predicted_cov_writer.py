"""Dump per-state predicted loc0/loc1 covariance for χ²-gate rows.

RootTrackStatesWriter only stores √diag(P); the off-diagonal P₀₁ is required
to form the full innovation covariance S = HPHᵀ+V used by
MeasurementSelector::calculateChi2.

Compatible with both modern ACTS (ReadDataHandle) and older Modal spack ACTS
(WhiteBoard has no typed get — we fall back to iterating sequencer aliases).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import acts
import acts.examples


def _matrix_elem(cov: Any, i: int, j: int) -> float:
    try:
        return float(cov[i, j])
    except (TypeError, IndexError):
        pass
    try:
        return float(cov(i, j))
    except TypeError:
        pass
    arr = list(cov)
    # Flattened 6x6 row-major
    return float(arr[i * 6 + j])


class PredictedCovWriter(acts.examples.IAlgorithm):
    """Write ``eventXXXXX-predicted-cov.csv`` for the CKF track container."""

    def __init__(
        self,
        output_dir: Path | str,
        tracks: str = "tracks",
        level: acts.logging.Level = acts.logging.INFO,
    ) -> None:
        super().__init__("PredictedCovWriter", level)
        self._output_dir = Path(output_dir)
        self._tracks_name = tracks
        self._handle = None
        if hasattr(acts.examples, "ReadDataHandle") and hasattr(
            acts.examples, "ConstTrackContainer"
        ):
            self._handle = acts.examples.ReadDataHandle(
                self, acts.examples.ConstTrackContainer, "InputTracks"
            )
            self._handle.initialize(tracks)

    def _get_tracks(self, context: Any) -> Any:
        if self._handle is not None:
            return self._handle(context.eventStore)
        store = context.eventStore
        # Older ACTS: try common WhiteBoard accessors if present
        for attr in ("get", "getContainer", "__getitem__"):
            if hasattr(store, attr):
                try:
                    return getattr(store, attr)(self._tracks_name)
                except Exception:
                    continue
        raise RuntimeError(
            "Cannot read tracks from WhiteBoard: no ReadDataHandle and no "
            f"get() on eventStore (keys={getattr(store, 'keys', None)}). "
            "Upgrade ACTS Python bindings or patch RootTrackStatesWriter."
        )

    def execute(self, context: Any) -> acts.examples.ProcessCode:
        tracks = self._get_tracks(context)
        event_nr = int(context.eventNumber)
        out = self._output_dir / f"event{event_nr:09d}-predicted-cov.csv"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "track_nr",
                    "step_k",
                    "eLOC0_prt",
                    "eLOC1_prt",
                    "P00",
                    "P01",
                    "P11",
                ]
            )
            for track_nr, track in enumerate(tracks):
                step_k = 0
                for state in track.trackStatesReversed:
                    if not state.hasPredicted:
                        step_k += 1
                        continue
                    if not hasattr(state, "predictedCovariance"):
                        raise RuntimeError(
                            "TrackStateProxy lacks predictedCovariance on this ACTS build"
                        )
                    pred = state.predicted
                    cov = state.predictedCovariance
                    p00 = _matrix_elem(cov, 0, 0)
                    p01 = _matrix_elem(cov, 0, 1)
                    p11 = _matrix_elem(cov, 1, 1)
                    w.writerow(
                        [
                            track_nr,
                            step_k,
                            float(pred[0]),
                            float(pred[1]),
                            p00,
                            p01,
                            p11,
                        ]
                    )
                    step_k += 1

        return acts.examples.ProcessCode.SUCCESS
