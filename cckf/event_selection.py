"""Parsing and validation for a CLI-supplied subset of events.

Split out of ``modal_train.py`` so the ``--only-events`` parsing/validation
logic is importable and unit-testable without the ``modal`` package -- not
installed in this local dev/test environment, unlike ``cckf.trackstates_index``
(pure lookup logic with no event-selection concerns) this module is the wrong
home for parsing a CLI scalar into a validated event subset.

Modal local entrypoints take CLI-friendly scalars, not lists, so the
``--only-events`` flag arrives as a single comma-separated string (e.g.
``"0,1"``) rather than a Python list. :func:`resolve_requested_events` turns
that string into the validated, sorted tuple of event ids a run should
actually process.
"""

from __future__ import annotations

from typing import Iterable

from cckf import splits


def resolve_requested_events(
    only_events: str, assigned_events: Iterable[int] | None = None
) -> tuple[int, ...]:
    """Resolve the ``--only-events`` CLI string into a validated event tuple.

    Parameters
    ----------
    only_events : str
        Comma-separated event ids, e.g. ``"0,1"``. Empty, or all whitespace,
        means "use every assigned event" -- the caller's default behavior.
    assigned_events : Iterable[int], optional
        The events a run is permitted to touch. Defaults to
        ``TRAIN_EVENTS + VAL_EVENTS + CAL_EVENTS`` from :mod:`cckf.splits`.

    Returns
    -------
    tuple[int, ...]
        Sorted, de-duplicated event ids to process.

    Raises
    ------
    ValueError
        If ``only_events`` contains a token that does not parse as an
        integer (e.g. ``"0,foo"``), or if any parsed id is not a member of
        ``assigned_events``. The error names the offending token/id.
    ValueError
        (raised by :func:`cckf.splits.assert_not_test`) if any parsed id
        falls in the sealed test range [32, 64) -- the same guard the
        default event set is already subject to, so a subset can never
        bypass it. This check runs *before* the "unassigned" check below,
        so a sealed-test id gets that guard's specific message rather than
        a generic "not assigned" one.
    """
    if assigned_events is None:
        assigned = (*splits.TRAIN_EVENTS, *splits.VAL_EVENTS, *splits.CAL_EVENTS)
    else:
        assigned = tuple(assigned_events)

    stripped = only_events.strip()
    if not stripped:
        return tuple(sorted(assigned))

    ids: list[int] = []
    for raw_token in stripped.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(
                f"--only-events: empty event id in {only_events!r}; expected "
                "a comma-separated list of integers like '0,1', with no "
                "empty entries"
            )
        try:
            ids.append(int(token))
        except ValueError:
            raise ValueError(
                f"--only-events: could not parse {token!r} as an integer "
                f"event id (from input {only_events!r}); expected a "
                "comma-separated list like '0,1'"
            ) from None

    # The sealed-test guard applies to a requested subset exactly as it
    # applies to the default set -- check this first so an id in [32, 64)
    # raises with the guard's own "sealed test" message, not the generic
    # "unassigned" message from the check below.
    splits.assert_not_test(ids)

    unassigned = sorted(set(ids) - set(assigned))
    if unassigned:
        raise ValueError(
            f"--only-events: event id(s) {unassigned} are not in the "
            f"assigned train/val/cal set {sorted(assigned)}; refusing to "
            "silently process an unassigned event"
        )

    return tuple(sorted(set(ids)))
