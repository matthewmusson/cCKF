"""Tests for ``scripts.train_gate._refuse_partial_cache``.

The hazard this guards against: ``modal_train.py::build_gate_cache_staged``
builds a gate cache from a small event subset (e.g. 2 of 24 train events) for
smoke-testing, marking the result ``partial_split: true`` /
``events_used: [...]`` in ``meta.json`` (``scripts/build_gate_cache.py``).
Nothing else reads that flag -- a staged cache is byte-for-byte
indistinguishable from a full one to every other consumer. These tests build
only a ``meta.json`` fixture (no ``X.f32``/``y.u8``/cache arrays at all),
since the check reads ``meta.json`` directly and must work without a real
cache existing.
"""

from __future__ import annotations

import json

import pytest

from scripts.train_gate import _refuse_partial_cache

_FULL_EVENTS = tuple(range(24))  # stand-in for splits.TRAIN_EVENTS; only len() matters


def _write_meta(tmp_path, meta: dict):
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_partial_split_true_is_refused_by_default(tmp_path):
    cache_dir = _write_meta(tmp_path, {"partial_split": True, "events_used": [0, 1]})
    with pytest.raises(SystemExit) as exc_info:
        _refuse_partial_cache(str(cache_dir), _FULL_EVENTS, allow_partial=False)
    message = str(exc_info.value)
    assert str(cache_dir) in message
    assert "2/24" in message  # events covered vs. the split's full count
    assert "[0, 1]" in message
    assert "--only-events" in message  # tells the caller how to rebuild


def test_allow_partial_cache_opt_in_bypasses_the_refusal(tmp_path):
    cache_dir = _write_meta(tmp_path, {"partial_split": True, "events_used": [0, 1]})
    # Must not raise, and must not even require the rest of meta.json to be
    # well-formed for a real cache -- the whole point of the opt-in is to
    # skip the check.
    _refuse_partial_cache(str(cache_dir), _FULL_EVENTS, allow_partial=True)


def test_partial_split_absent_is_unaffected(tmp_path):
    cache_dir = _write_meta(tmp_path, {"n_rows": 1000})
    _refuse_partial_cache(str(cache_dir), _FULL_EVENTS, allow_partial=False)


def test_partial_split_false_is_unaffected(tmp_path):
    cache_dir = _write_meta(tmp_path, {"partial_split": False})
    _refuse_partial_cache(str(cache_dir), _FULL_EVENTS, allow_partial=False)
