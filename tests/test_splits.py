"""Tests for the frozen event→split assignment."""

from __future__ import annotations

import pytest

from cckf import splits


def test_splits_partition_the_lower_range_exactly():
    train, val, cal = (
        set(splits.TRAIN_EVENTS),
        set(splits.VAL_EVENTS),
        set(splits.CAL_EVENTS),
    )
    assert train & val == set()
    assert train & cal == set()
    assert val & cal == set()
    # All 32 events in [0, 32) are patched and assigned — a true partition,
    # with nothing left over and nothing outside the range.
    assert train | val | cal == set(range(32))
    assert len(train) == 24 and len(val) == 4 and len(cal) == 4


def test_no_duplicate_events_within_a_split():
    for name in ("TRAIN_EVENTS", "VAL_EVENTS", "CAL_EVENTS"):
        events = getattr(splits, name)
        assert len(events) == len(set(events)), name


def test_val_and_cal_do_not_share_a_generation_batch():
    """Stage 1 produced events as 16 batches of 2: (0,1), (2,3), ..., (30,31).
    A batch-correlated artefact must not be confined to one split."""
    val_batches = {e // 2 for e in splits.VAL_EVENTS}
    cal_batches = {e // 2 for e in splits.CAL_EVENTS}
    assert val_batches & cal_batches == set()


def test_val_and_cal_are_spread_across_the_event_range():
    """One val and one cal pick per quarter of [0, 32), so neither split is
    concentrated at one end of the generation order."""
    for events in (splits.VAL_EVENTS, splits.CAL_EVENTS):
        quarters = {e // 8 for e in events}
        assert quarters == {0, 1, 2, 3}


def test_test_split_is_the_sealed_upper_range():
    assert splits.TEST_EVENTS == tuple(range(32, 64))
    assert set(splits.TEST_EVENTS) & set(splits.TRAIN_EVENTS) == set()


def test_split_of_classifies_each_range():
    assert splits.split_of(0) == "train"
    assert splits.split_of(4) == "val"
    assert splits.split_of(7) == "cal"
    assert splits.split_of(40) == "test"


def test_split_of_covers_every_event_in_zero_to_sixtyfour():
    for event_id in range(64):
        assert splits.split_of(event_id) in {"train", "val", "cal", "test"}


def test_split_of_rejects_an_out_of_range_event():
    # Must fail loudly rather than silently defaulting to "train".
    with pytest.raises(KeyError):
        splits.split_of(64)
    with pytest.raises(KeyError):
        splits.split_of(-1)


def test_assert_not_test_raises_on_sealed_event():
    splits.assert_not_test([0, 1, 4, 7])  # no raise
    with pytest.raises(ValueError, match="sealed"):
        splits.assert_not_test([0, 32])


def test_events_for_returns_the_matching_tuple():
    assert splits.events_for("train") == splits.TRAIN_EVENTS
    assert splits.events_for("val") == splits.VAL_EVENTS
    assert splits.events_for("cal") == splits.CAL_EVENTS
    assert splits.events_for("test") == splits.TEST_EVENTS


def test_events_for_rejects_an_unknown_split():
    with pytest.raises(ValueError):
        splits.events_for("bogus")


def test_schema_76_has_76_columns_in_expansion_order():
    assert len(splits.SCHEMA_76) == 76
    assert splits.SCHEMA_76[0] == "event_id"
    assert splits.SCHEMA_76[-1] == "env_config_hash"
    assert "S00" in splits.SCHEMA_76 and "eta" not in splits.SCHEMA_76
