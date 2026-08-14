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

from cckf.trackstates_index import TrackstatesIndex, find_trackstates


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
