"""Append-only JSONL event stream for one GUI session.

One file per session; ``emit`` assigns ``seq`` and ``t`` and flushes after
each write, so a hard kill leaves at most one truncated line — which
``events()`` silently skips.

Called from ``server.py`` only, before either Dash app is imported::

    from cryocat.app import session
    session.start_session()

Public API
----------
* :func:`start_session` — open the stream (idempotent).
* :func:`emit`          — append one event.
* :func:`events`        — read a stream back; tolerates truncated last line.
* :func:`session_path`  — the currently-open file, or ``None``.
* :func:`close_session` — flush and close (also registered with ``atexit``).
"""
from __future__ import annotations

import atexit
import datetime as _dt
import json
import platform
import sys
from pathlib import Path


_session_path: Path | None = None
_fh = None
_seq: int = 0
_warned_no_session: bool = False


def start_session(log_dir: str | Path | None = None) -> Path:
    """Open the event stream. Idempotent — second call returns existing path.

    Default directory is ``~/.cryocat/sessions``. Emits the ``session`` event.
    Registers :func:`close_session` with ``atexit``.
    """
    global _session_path, _fh, _seq, _warned_no_session
    if _session_path is not None:
        return _session_path

    if log_dir is None:
        log_dir = Path.home() / ".cryocat" / "sessions"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _session_path = log_dir / f"cryocat_session_{ts}.jsonl"
    _seq = 0
    _warned_no_session = False
    _fh = open(_session_path, "a", encoding="utf-8")
    atexit.register(close_session)

    try:
        from cryocat import __version__ as _cv
    except Exception:
        _cv = "unknown"

    emit({
        "kind": "session",
        "cryocat_version": _cv,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "session_id": ts,
    })

    return _session_path


def emit(event: dict) -> None:
    """Append one event as a JSON line, assigning ``seq`` and ``t``. Flushes.

    If called before :func:`start_session`, emits one ``source="error"``
    warning to the pane and silently drops all subsequent events.
    """
    global _seq, _warned_no_session
    if _fh is None:
        if not _warned_no_session:
            _warned_no_session = True
            from cryocat.app.logger import dash_logger
            dash_logger.write(
                "session.emit() called before start_session() — events are "
                "dropped. Call cryocat.app.session.start_session() at startup.",
                source="error",
            )
        return
    record = {
        "seq": _seq,
        "t": _dt.datetime.now().isoformat(timespec="milliseconds"),
        **event,
    }
    _seq += 1
    _fh.write(json.dumps(record) + "\n")
    _fh.flush()


def events(path: Path | None = None) -> list[dict]:
    """Read a JSONL event stream back as a list of dicts.

    Uses the currently-open file when *path* is omitted.  Lines that fail
    JSON parsing (e.g. a truncated final line after a hard kill) are silently
    skipped.
    """
    target = path or _session_path
    if target is None or not Path(target).exists():
        return []
    result: list[dict] = []
    with open(target, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result


def session_path() -> Path | None:
    """Return the path to the currently-open event stream, or ``None``."""
    return _session_path


def last_seq() -> int:
    """Return the seq number of the most recently emitted event.

    Returns ``-1`` if no events have been emitted this session.
    """
    return _seq - 1 if _seq > 0 else -1


def close_session() -> None:
    """Flush and close the event stream; reset all module-level state."""
    global _fh, _session_path, _seq, _warned_no_session
    if _fh is not None:
        try:
            _fh.close()
        except Exception:
            pass
        _fh = None
    _session_path = None
    _seq = 0
    _warned_no_session = False
