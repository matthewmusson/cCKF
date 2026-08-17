"""Tests for ``scripts.build_gate_cache.resolve_split_events``.

``resolve_requested_events`` (``cckf.event_selection``) already owns and is
tested for the parsing/validation logic itself (12 tests in
``tests/test_event_selection.py``): malformed tokens, dedup, the sealed-test
guard, the "not in the assigned set" guard. These tests instead cover the
call-site wiring in ``resolve_split_events`` -- specifically, that it scopes
the "assigned" set to the *requested split's own events*, not the full
train+val+cal union ``resolve_requested_events`` would default to. Getting
that wrong (e.g. passing the union) would let ``--split train --only-events
4`` silently succeed even though 4 is a validation event, mixing splits.
"""

from __future__ import annotations

import pytest

from cckf import splits
from scripts.build_gate_cache import resolve_split_events


def test_omitting_only_events_returns_the_full_split():
    assert resolve_split_events("val", "") == tuple(sorted(splits.VAL_EVENTS))
    assert resolve_split_events("train", "") == tuple(sorted(splits.TRAIN_EVENTS))


def test_valid_subset_of_the_requested_split_is_accepted():
    a, b = splits.TRAIN_EVENTS[0], splits.TRAIN_EVENTS[1]
    assert resolve_split_events("train", f"{a},{b}") == tuple(sorted((a, b)))


def test_event_from_a_different_split_is_rejected():
    # 4 is a VAL event, not a TRAIN event -- must not silently mix splits.
    val_event = splits.VAL_EVENTS[0]
    assert val_event not in splits.TRAIN_EVENTS
    with pytest.raises(ValueError, match="not in the assigned"):
        resolve_split_events("train", str(val_event))


def test_event_from_the_sealed_test_range_is_rejected():
    test_event = splits.TEST_EVENTS[0]
    with pytest.raises(ValueError, match="sealed"):
        resolve_split_events("train", str(test_event))
