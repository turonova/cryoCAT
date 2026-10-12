"""Tests for cryocat.app.session — the append-only JSONL event stream."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import cryocat.app.session as session
from cryocat.app.logger import dash_logger


@pytest.fixture(autouse=True)
def _reset(tmp_path):
    """Ensure each test starts with a clean session and an empty pane buffer."""
    session.close_session()
    dash_logger.buffer.clear()
    yield
    session.close_session()
    dash_logger.buffer.clear()


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_round_trip(tmp_path):
    path = session.start_session(tmp_path)
    session.emit({"kind": "call", "fn": "some.fn", "status": "ok"})
    session.emit({"kind": "message", "text": "hello", "level": "info"})

    loaded = session.events()
    assert len(loaded) == 3  # session event + 2 emitted

    assert loaded[0]["kind"] == "session"
    assert loaded[0]["seq"] == 0

    assert loaded[1]["kind"] == "call"
    assert loaded[1]["seq"] == 1
    assert loaded[1]["fn"] == "some.fn"

    assert loaded[2]["kind"] == "message"
    assert loaded[2]["seq"] == 2
    assert "t" in loaded[2]

    assert session.session_path() == path


def test_round_trip_by_explicit_path(tmp_path):
    path = session.start_session(tmp_path)
    session.emit({"kind": "pool", "action": "rename"})

    loaded = session.events(path)
    assert len(loaded) == 2
    assert loaded[1]["kind"] == "pool"


def test_events_returns_empty_with_no_session():
    assert session.events() == []


def test_events_returns_empty_for_missing_file(tmp_path):
    assert session.events(tmp_path / "nonexistent.jsonl") == []


# ---------------------------------------------------------------------------
# Truncated final line (crash safety)
# ---------------------------------------------------------------------------

def test_truncated_final_line_skipped(tmp_path):
    good_events = [
        {"kind": "session", "seq": 0, "t": "2026-01-01T00:00:00.000"},
        {"kind": "call", "seq": 1, "t": "2026-01-01T00:00:01.000", "fn": "a.b"},
    ]
    truncated_file = tmp_path / "truncated.jsonl"
    with open(truncated_file, "w", encoding="utf-8") as f:
        for ev in good_events:
            f.write(json.dumps(ev) + "\n")
        # Write a truncated (invalid) JSON line — simulates a hard kill mid-write
        f.write('{"kind": "call", "seq": 2, "t": "')

    loaded = session.events(truncated_file)
    assert len(loaded) == 2
    assert loaded[0]["kind"] == "session"
    assert loaded[1]["fn"] == "a.b"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_start_session_idempotent(tmp_path):
    path_a = session.start_session(tmp_path / "dir_a")
    path_b = session.start_session(tmp_path / "dir_b")  # second call — must no-op

    assert path_a == path_b
    assert session.session_path() == path_a
    # Only the first session event should exist — dir_b should be empty
    assert not (tmp_path / "dir_b").exists()


def test_start_session_returns_path(tmp_path):
    path = session.start_session(tmp_path)
    assert path.exists()
    assert path.suffix == ".jsonl"


# ---------------------------------------------------------------------------
# emit with no open session warns exactly once
# ---------------------------------------------------------------------------

def test_emit_without_session_warns_once():
    # No session open — first emit should warn once to dash_logger.buffer
    session.emit({"kind": "message", "text": "dropped"})
    session.emit({"kind": "message", "text": "also dropped"})
    session.emit({"kind": "message", "text": "still dropped"})

    error_entries = [(msg, src) for msg, src in dash_logger.buffer if src == "error"]
    assert len(error_entries) == 1
    assert "start_session" in error_entries[0][0]
