"""Tests for the ``--only-events`` CLI-string parsing/validation used by
``modal_train.py::patch_selected_all``.

``cckf.event_selection`` holds this logic (not ``cckf.trackstates_index``,
which is pure trackstates-file lookup with no event-selection concerns)
specifically so it is importable and unit-testable without the ``modal``
package, which is not installed in this environment.
"""

from __future__ import annotations

import pytest

from cckf.event_selection import resolve_requested_events

ASSIGNED = (0, 1, 2, 3, 4)


def test_empty_string_means_every_assigned_event():
    assert resolve_requested_events("", ASSIGNED) == tuple(sorted(ASSIGNED))


def test_whitespace_only_string_means_every_assigned_event():
    assert resolve_requested_events("   ", ASSIGNED) == tuple(sorted(ASSIGNED))


def test_valid_subset_is_parsed_and_sorted():
    assert resolve_requested_events("3,1", ASSIGNED) == (1, 3)


def test_valid_subset_tolerates_surrounding_whitespace():
    assert resolve_requested_events(" 1 , 3 ", ASSIGNED) == (1, 3)


def test_duplicate_ids_are_deduplicated():
    assert resolve_requested_events("1,1,3", ASSIGNED) == (1, 3)


def test_unassigned_id_raises_and_names_it():
    with pytest.raises(ValueError, match=r"\[99\]"):
        resolve_requested_events("1,99", ASSIGNED)


def test_sealed_test_id_raises_the_sealed_test_message_not_unassigned():
    # 40 is in cckf.splits.TEST_EVENTS -- must hit assert_not_test's own
    # "sealed" message, not the generic "not in the assigned" message, even
    # though 40 is also not a member of ASSIGNED.
    with pytest.raises(ValueError, match="sealed"):
        resolve_requested_events("40", ASSIGNED)


def test_sealed_test_id_message_does_not_say_unassigned():
    with pytest.raises(ValueError) as excinfo:
        resolve_requested_events("40", ASSIGNED)
    assert "not in the assigned" not in str(excinfo.value)


def test_mixed_sealed_and_unassigned_ids_prefer_the_sealed_message():
    # 32 is sealed-test; 99 is merely unassigned. The more specific sealed
    # guard must win regardless of token order.
    with pytest.raises(ValueError, match="sealed"):
        resolve_requested_events("99,32", ASSIGNED)


def test_malformed_token_raises_clear_error_naming_it():
    with pytest.raises(ValueError, match="foo"):
        resolve_requested_events("0,foo", ASSIGNED)


def test_empty_token_between_commas_raises_clear_error():
    with pytest.raises(ValueError):
        resolve_requested_events("0,,1", ASSIGNED)


def test_default_assigned_events_pulled_from_splits_when_omitted():
    from cckf import splits

    full = tuple(sorted((*splits.TRAIN_EVENTS, *splits.VAL_EVENTS, *splits.CAL_EVENTS)))
    assert resolve_requested_events("") == full
    # A valid id from the real assignment resolves without error.
    assert resolve_requested_events(str(splits.TRAIN_EVENTS[0])) == (
        splits.TRAIN_EVENTS[0],
    )
