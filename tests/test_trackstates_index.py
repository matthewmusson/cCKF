"""Tests for the pure-logic trackstates lookup used by ``modal_train.py``.

``cckf.trackstates_index`` holds only the index dataclass and the lookup
function -- no ``modal`` import, no ROOT/uproot I/O -- specifically so this
logic can be exercised with hand-built ``TrackstatesIndex`` instances and no
live data. The I/O that populates the index (``modal_train._build_trackstates_
index``) lives in ``modal_train.py`` and is not importable here since the
``modal`` package is not installed in this environment; that function is
exercised only by inspection (see task-15-report.md), not by these tests.
"""
from __future__ import annotations

import pytest

from cckf.trackstates_index import (
    TrackstatesIndex,
    check_no_event_nr_fallback_is_safe,
    find_trackstates,
)


def test_exact_match_returns_the_mapped_path():
    index = TrackstatesIndex(
        n_candidates=2,
        matched={0: "/data/results/batch0/trackstates_ckf.root", 1: "/data/results/batch0/trackstates_ckf.root"},
    )
    assert find_trackstates(0, index) == "/data/results/batch0/trackstates_ckf.root"
    assert find_trackstates(1, index) == "/data/results/batch0/trackstates_ckf.root"


def test_sole_no_event_nr_candidate_is_used_as_fallback():
    """The unambiguous case: exactly one candidate file total, and it has no
    event_nr branch -- the genuine "older pipeline, single-event file"
    scenario expansion.load_trackstates documents."""
    index = TrackstatesIndex(n_candidates=1, no_event_nr=["/data/results/legacy/trackstates_ckf.root"])
    assert find_trackstates(7, index) == "/data/results/legacy/trackstates_ckf.root"


def test_multiple_no_event_nr_candidates_raise_and_name_both_paths():
    index = TrackstatesIndex(
        n_candidates=2,
        no_event_nr=[
            "/data/results/a/trackstates_ckf.root",
            "/data/results/b/trackstates_ckf.root",
        ],
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        find_trackstates(3, index)
    message = str(excinfo.value)
    assert "/data/results/a/trackstates_ckf.root" in message
    assert "/data/results/b/trackstates_ckf.root" in message
    assert "3" in message


def test_one_no_event_nr_candidate_alongside_other_matches_raises():
    """Even a single no-event_nr file cannot be trusted as the fallback for a
    different, unmatched event if other (matched) candidate files exist --
    their presence means this volume follows the normal per-batch layout, so
    the no-event_nr file is an outlier of unknown provenance, not proof it
    belongs to the event being looked up."""
    index = TrackstatesIndex(
        n_candidates=2,
        matched={0: "/data/results/batch0/trackstates_ckf.root"},
        no_event_nr=["/data/results/orphan/trackstates_ckf.root"],
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        find_trackstates(5, index)
    message = str(excinfo.value)
    assert "/data/results/orphan/trackstates_ckf.root" in message
    assert "5" in message


def test_no_candidates_at_all_raises_all_readable_form():
    index = TrackstatesIndex(n_candidates=3, matched={0: "/data/results/x/trackstates_ckf.root"})
    with pytest.raises(FileNotFoundError) as excinfo:
        find_trackstates(9, index)
    message = str(excinfo.value)
    assert "all readable" in message
    assert "9" in message
    assert "3" in message


def test_unreadable_candidates_are_named_in_the_error():
    index = TrackstatesIndex(
        n_candidates=2,
        unreadable=[
            ("/data/results/broken/trackstates_ckf.root", "OSError('truncated file')"),
        ],
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        find_trackstates(11, index)
    message = str(excinfo.value)
    assert "/data/results/broken/trackstates_ckf.root" in message
    assert "1 unreadable" in message
    assert "11" in message


# --- check_no_event_nr_fallback_is_safe: build-time multi-event guard -----
#
# find_trackstates's per-lookup guard (n_candidates == 1) cannot see how many
# events the whole run intends to resolve, so by itself it would let a
# single no-event_nr file be silently reused for every event in a
# multi-event run -- each call individually looks like the sanctioned
# single-file case. check_no_event_nr_fallback_is_safe closes that gap by
# validating the full requested-event set once, before any lookups happen.


def test_single_requested_event_with_sole_no_event_nr_candidate_is_permitted():
    """One event, one no-event_nr file: the genuine single-event-file case --
    must not raise, and find_trackstates must still resolve it afterwards."""
    index = TrackstatesIndex(n_candidates=1, no_event_nr=["/data/results/legacy/trackstates_ckf.root"])
    check_no_event_nr_fallback_is_safe(index, [7])  # must not raise
    assert find_trackstates(7, index) == "/data/results/legacy/trackstates_ckf.root"


def test_multiple_requested_events_with_sole_no_event_nr_candidate_raises():
    """Two or more requested events against a single no-event_nr file: the
    file can supply only one event's data, so honoring every request would
    silently attribute the same contents to multiple distinct events. This
    must be caught here, at index-build time, before any event is resolved
    (find_trackstates alone -- see the module docstring's reproduction --
    would let every one of these calls through with no exception)."""
    index = TrackstatesIndex(n_candidates=1, no_event_nr=["/data/results/legacy/trackstates_ckf.root"])
    with pytest.raises(FileNotFoundError) as excinfo:
        check_no_event_nr_fallback_is_safe(index, [5, 6])
    message = str(excinfo.value)
    assert "/data/results/legacy/trackstates_ckf.root" in message
    assert "2" in message


def test_check_is_a_noop_when_fallback_would_not_be_used():
    """The guard only concerns the sole-no-event_nr-file configuration; any
    index with an exact match available, or with no no-event_nr file at all,
    must never raise here regardless of how many events are requested."""
    index = TrackstatesIndex(n_candidates=1, matched={0: "/data/results/x/trackstates_ckf.root"})
    check_no_event_nr_fallback_is_safe(index, [0, 1, 2, 3])  # must not raise
