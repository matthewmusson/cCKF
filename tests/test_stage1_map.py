"""Tests for the explicit event -> Stage 1 pilot-directory map.

These tests pin the map's *structure and contract*, not its contents against
the volume: the paths name directories on the Modal ``surp-acts-data`` volume,
which is not reachable from a local test run. The contents are verified
against ``modal_build_acts.py::expand_all_events`` by inspection (see the
module docstring of ``cckf.stage1_map``); what is worth automating is that the
map stays total over the training events, stays consistent between its two
accessors, and refuses to extrapolate.
"""

from __future__ import annotations

import pytest

from cckf import splits, stage1_map


def test_covers_every_non_test_event() -> None:
    """Every event the splits assign must be resolvable.

    If this fails, some split event has no provenance entry and the value
    cache would fall back to a directory scan for it -- which is exactly the
    wrong-config hazard the map exists to remove.
    """
    assigned = {*splits.TRAIN_EVENTS, *splits.VAL_EVENTS, *splits.CAL_EVENTS}
    assert assigned <= set(stage1_map.covered_events())


def test_covers_exactly_0_to_31() -> None:
    assert stage1_map.covered_events() == tuple(range(32))


def test_no_test_split_event_is_covered() -> None:
    """The map must not reach into the sealed [32, 64) test set."""
    assert not set(stage1_map.covered_events()) & set(splits.TEST_EVENTS)


def test_each_directory_covers_two_consecutive_events() -> None:
    """Stage 1 ran 16 batches of 2, so each directory owns an even/odd pair."""
    by_dir: dict[str, list[int]] = {}
    for event_id in stage1_map.covered_events():
        by_dir.setdefault(stage1_map.pilot_dir_for(event_id), []).append(event_id)

    assert len(by_dir) == 16
    for pilot_dir, events in by_dir.items():
        assert len(events) == 2, f"{pilot_dir} covers {events}"
        first, second = sorted(events)
        assert second == first + 1
        assert first % 2 == 0


def test_directories_are_distinct() -> None:
    """Two event pairs sharing a directory would mean a transcription slip."""
    dirs = [stage1_map.pilot_dir_for(e) for e in stage1_map.covered_events()]
    assert len(set(dirs)) == len(dirs) // 2


def test_events_30_31_use_the_corrected_directory() -> None:
    """Pin the one documented correction.

    ``modal_build_acts.py`` contains two disagreeing lists: the stale
    ``check_pilot_dirs`` says ``pilot_1786546373`` for events 30-31, while
    ``expand_all_events`` -- the function that actually wrote the expanded
    Parquets on the volume -- says ``pilot_1786547065``. The latter is
    correct and user-confirmed. This test exists so a future "cleanup" that
    reconciles the two lists toward the stale one fails loudly.
    """
    assert stage1_map.pilot_dir_for(30).endswith("pilot_1786547065")
    assert stage1_map.pilot_dir_for(31).endswith("pilot_1786547065")


def test_trackstates_path_is_under_the_pilot_dir() -> None:
    for event_id in (0, 15, 31):
        assert stage1_map.trackstates_path_for(event_id) == (
            f"{stage1_map.pilot_dir_for(event_id)}/trackstates_ckf.root"
        )


def test_csv_dir_matches_pilot_dir() -> None:
    """``expand_all_events`` sets ``csv_dir = pilot_dir`` -- same directory."""
    for event_id in stage1_map.covered_events():
        assert stage1_map.csv_dir_for(event_id) == stage1_map.pilot_dir_for(event_id)


@pytest.mark.parametrize("event_id", [-1, 32, 63, 1000])
def test_uncovered_event_raises_rather_than_extrapolating(event_id: int) -> None:
    """An uncovered event must not be guessed at by extending the pattern.

    Events 32-63 are the sealed test set; extrapolating "2 events per
    directory" past 31 would invent a path, and inventing one for a test
    event is worse than failing.
    """
    with pytest.raises(KeyError, match="not covered"):
        stage1_map.pilot_dir_for(event_id)
    with pytest.raises(KeyError):
        stage1_map.trackstates_path_for(event_id)
    with pytest.raises(KeyError):
        stage1_map.csv_dir_for(event_id)
