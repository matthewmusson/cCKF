"""Explicit event -> Stage 1 pilot-directory map.

Why this map exists (read before "simplifying" it back into a search)
-----------------------------------------------------------------------
The ``surp-acts-data`` volume holds 1,347 result directories. Many of them
contain a ``trackstates_ckf.root`` for the *same* event id, produced by
*different* CKF configurations (different chi2 cuts, seed counts, etc.) run
during parameter scans -- nothing in a directory's filename or an event's
``event_nr`` branch distinguishes "the pilot run whose Parquets became the
32-event training set" from "some other config that happens to cover the
same event id". A directory-scanning search (``cckf.trackstates_index`` /
``modal_train._build_trackstates_index``) has no way to prefer one over the
other; on the first real run it picked
``/data/results/calib_medium_1785797665/...`` -- the *Medium* operating
point (chi2=12.04, nMeasMin=7, 46 seeds/spm) -- over the correct pilot file,
purely because that path sorted first alphabetically. The training Parquets
in ``results/train32/expanded`` were built from ``configs/envelope.yaml``,
so joining them against the Medium-config trackstates would have produced a
residual mismatch on every row.

The *correct* pairing is not discoverable by scanning -- it is a fact about
which Modal run produced which output, recorded only in
``modal_build_acts.py::expand_all_events``, the function that actually wrote
``results/train32/expanded/``. That function's ``pilot_dirs`` list is the
provenance record for both the trackstates ROOT file *and* the per-event
simhits CSVs (it sets ``csv_dir = pilot_dir`` -- the same directory supplies
both), and each directory covers exactly 2 consecutive events. This module
copies that list, once, as the single source of truth for both consumers
(``modal_train.py`` for trackstates, ``scripts/build_value_cache.py`` for
simhits), so neither has to re-derive or re-guess it.

One entry is a **documented correction**, not a transcription of the
original list: ``modal_build_acts.py`` also contains a now-stale helper,
``check_pilot_dirs``, whose hardcoded list gives events 30-31 as
``pilot_1786546373``. That is a leftover from an earlier, superseded pilot
run. ``expand_all_events`` -- the function whose output is actually on disk
as ``results/train32/expanded`` -- uses ``pilot_1786547065`` for events
30-31, and the user has confirmed that is the correct directory. Trust
``expand_all_events``, not ``check_pilot_dirs``, if the two ever disagree
again.

Consequence of getting this wrong: a bad pairing does not corrupt data
silently. ``expansion.load_trackstates`` / the patch step's
``frac_states_matched >= 0.95`` gate compares the trackstates file's hits
against the expanded Parquet's own hits and refuses a low match fraction, so
a wrong pairing fails loudly at that gate rather than training on a subtly
mismatched join. That gate is a safety net, not a substitute for using the
right map -- it catches gross mismatches, not a config that happens to
overlap enough hits to clear 95%.
"""

from __future__ import annotations

#: (pilot_dir, first_event_id) for each Stage 1 batch. Copied verbatim from
#: ``modal_build_acts.py::expand_all_events`` (lines ~2795-2812), the
#: function that actually produced ``results/train32/expanded/``, with the
#: events-30-31 directory corrected per the user-confirmed provenance note
#: above (``pilot_1786547065``, not the stale ``pilot_1786546373`` found in
#: ``modal_build_acts.py::check_pilot_dirs``). Each directory covers exactly
#: 2 consecutive event ids: ``first_event_id`` and ``first_event_id + 1``.
_PILOT_DIRS: tuple[tuple[str, int], ...] = (
    ("/data/results/pilot_1786524194", 0),  # events 0-1
    ("/data/results/pilot_1786524971", 2),  # events 2-3
    ("/data/results/pilot_1786525888", 4),  # events 4-5
    ("/data/results/pilot_1786527639", 6),  # events 6-7
    ("/data/results/pilot_1786529841", 8),  # events 8-9
    ("/data/results/pilot_1786531084", 10),  # events 10-11
    ("/data/results/pilot_1786532999", 12),  # events 12-13
    ("/data/results/pilot_1786534329", 14),  # events 14-15
    ("/data/results/pilot_1786536577", 16),  # events 16-17
    ("/data/results/pilot_1786538022", 18),  # events 18-19
    ("/data/results/pilot_1786539360", 20),  # events 20-21
    ("/data/results/pilot_1786540365", 22),  # events 22-23
    ("/data/results/pilot_1786541815", 24),  # events 24-25
    ("/data/results/pilot_1786542563", 26),  # events 26-27
    ("/data/results/pilot_1786543789", 28),  # events 28-29
    # events 30-31 -- the corrected entry; see the module docstring
    ("/data/results/pilot_1786547065", 30),
)

#: Number of consecutive events each pilot directory covers.
_EVENTS_PER_DIR = 2

#: event_id -> pilot_dir, expanded once from ``_PILOT_DIRS`` at import time.
_EVENT_TO_DIR: dict[int, str] = {
    first_event + offset: pilot_dir
    for pilot_dir, first_event in _PILOT_DIRS
    for offset in range(_EVENTS_PER_DIR)
}


def pilot_dir_for(event_id: int) -> str:
    """Return the Stage 1 pilot directory that produced ``event_id``.

    Parameters
    ----------
    event_id : int
        The event id to look up.

    Returns
    -------
    str
        The pilot directory path, e.g. ``"/data/results/pilot_1786524194"``.
        Supplies both the trackstates ROOT file
        (:func:`trackstates_path_for`) and the simhits CSVs
        (:func:`csv_dir_for`) for this event.

    Raises
    ------
    KeyError
        If ``event_id`` is not covered by the map (see module docstring for
        why an uncovered event is not silently guessed at).
    """
    try:
        return _EVENT_TO_DIR[event_id]
    except KeyError:
        raise KeyError(
            f"event {event_id} is not covered by the stage1 pilot-directory "
            f"map (covered: {min(_EVENT_TO_DIR)}-{max(_EVENT_TO_DIR)}); this "
            "map is the explicit provenance record copied from "
            "modal_build_acts.py::expand_all_events and is not meant to be "
            "extrapolated -- add a new (pilot_dir, first_event_id) entry to "
            "cckf.stage1_map._PILOT_DIRS instead of guessing a directory for "
            f"event {event_id}"
        ) from None


def trackstates_path_for(event_id: int) -> str:
    """Return the ``trackstates_ckf.root`` path for ``event_id``.

    Parameters
    ----------
    event_id : int
        The event id to look up.

    Returns
    -------
    str
        ``{pilot_dir_for(event_id)}/trackstates_ckf.root``.

    Raises
    ------
    KeyError
        Propagated from :func:`pilot_dir_for` if ``event_id`` is uncovered.
    """
    return f"{pilot_dir_for(event_id)}/trackstates_ckf.root"


def csv_dir_for(event_id: int) -> str:
    """Return the simhits CSV directory for ``event_id``.

    This is the pilot directory itself: ``expand_all_events`` sets
    ``csv_dir = pilot_dir``, i.e. the same Stage 1 batch directory that
    holds the trackstates ROOT file also holds
    ``event{event_id:09d}-simhits.csv`` and the other per-event CSVs
    (``expansion.load_simhits`` reads directly from this directory).

    Parameters
    ----------
    event_id : int
        The event id to look up.

    Returns
    -------
    str
        The pilot directory path.

    Raises
    ------
    KeyError
        Propagated from :func:`pilot_dir_for` if ``event_id`` is uncovered.
    """
    return pilot_dir_for(event_id)


def covered_events() -> tuple[int, ...]:
    """Return every event id covered by the map, sorted ascending.

    Callers can use this to test membership (``event_id in
    stage1_map.covered_events()``) without relying on exception handling.

    Returns
    -------
    tuple[int, ...]
        Sorted event ids covered by the map. Currently ``0`` through ``31``
        inclusive, with no gaps or duplicates.
    """
    return tuple(sorted(_EVENT_TO_DIR))
