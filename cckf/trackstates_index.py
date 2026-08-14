"""Pure-logic lookup of a trackstates ROOT file for a given event.

Split out of ``modal_train.py`` so this logic is importable and unit-testable
without depending on the ``modal`` package (not installed in the local dev
environment) or on live ROOT files. The I/O -- scanning ``/data/results`` and
opening each candidate with ``uproot`` -- stays in
``modal_train._build_trackstates_index``, which constructs a
``TrackstatesIndex`` from what it finds and hands it to
:func:`find_trackstates`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrackstatesIndex:
    """Result of one scan of the candidate trackstates files.

    ``matched`` maps ``event_id -> path`` for files with an exact
    ``event_nr`` hit. ``no_event_nr`` lists files that lack an ``event_nr``
    branch entirely -- per ``expansion.load_trackstates``, a legitimate
    single-event file from an older pipeline stage, treated by that function
    as belonging to whatever event is requested of it. ``unreadable`` lists
    ``(path, repr(exc))`` for files that failed to open or whose
    ``trackstates`` tree could not be read.
    """

    n_candidates: int = 0
    matched: dict[int, str] = field(default_factory=dict)
    no_event_nr: list[str] = field(default_factory=list)
    unreadable: list[tuple[str, str]] = field(default_factory=list)


def find_trackstates(event_id: int, index: TrackstatesIndex) -> str:
    """Look up the trackstates file for one event from a pre-built index.

    Prefers an exact ``event_nr`` match.

    Falls back to a no-``event_nr`` file only when that fallback is
    *unambiguous* -- i.e. it is the sole candidate file found on the whole
    volume (``index.n_candidates == 1``), matching the single-event-file
    scenario ``expansion.load_trackstates`` documents (its own fallback is a
    1:1, per-call assumption: "this file *is* the requested event's file").

    Any other no-``event_nr`` configuration is refused rather than guessed:

    - **More than one** no-``event_nr`` file: there is no way to know which
      one holds which event.
    - **Exactly one** no-``event_nr`` file **alongside other candidates**
      (matched and/or unreadable): the presence of other files means this
      volume is following the normal per-batch layout with proper
      ``event_nr`` branches, so a lone no-``event_nr`` file is an outlier --
      it is not safe to assume it belongs to *this particular* unmatched
      event rather than some other event, or no event in this split at all.

    ``TrackstatesIndex`` is shared across every event in a run, so unlike
    ``expansion.load_trackstates``'s single-call fallback, guessing here
    could attribute the *same* file to two different events, or the *wrong*
    file to one event -- silently mislabelling ``is_ckf_selected`` with no
    visible signal that anything went wrong. That corrupts a training target
    quietly, which is worse than the loud failure this raises instead.
    """
    if event_id in index.matched:
        path = index.matched[event_id]
        print(f"event {event_id}: trackstates at {path}")
        return path

    if index.no_event_nr:
        if len(index.no_event_nr) == 1 and index.n_candidates == 1:
            path = index.no_event_nr[0]
            print(
                f"event {event_id}: no exact event_nr match, falling back "
                f"to the sole candidate file (no event_nr branch) {path}"
            )
            return path
        raise FileNotFoundError(
            f"cannot resolve trackstates for event {event_id}: "
            f"{len(index.no_event_nr)} file(s) without an 'event_nr' branch "
            f"present ({', '.join(index.no_event_nr)}) alongside "
            f"{index.n_candidates} total candidate file(s) on the volume. "
            "The no-event_nr fallback is only safe when such a file is the "
            "sole candidate found (a genuine single-event file from an "
            "older pipeline stage) -- with other candidates present there "
            "is no way to know which event a no-event_nr file belongs to. "
            "Inspect these file(s) by hand, or re-run Stage 1 so every "
            "file carries an 'event_nr' branch."
        )

    n_unreadable = len(index.unreadable)
    if n_unreadable:
        bad = ", ".join(p for p, _ in index.unreadable)
        raise FileNotFoundError(
            f"no trackstates file contains event {event_id}; probed "
            f"{index.n_candidates} file(s), {n_unreadable} unreadable "
            f"({bad}), the rest readable but none matched"
        )
    raise FileNotFoundError(
        f"no trackstates file contains event {event_id}; probed "
        f"{index.n_candidates} file(s), all readable, none contained "
        f"event {event_id}"
    )
