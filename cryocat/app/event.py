"""Event construction and validation for the cryoCAT session stream.

Four event kinds are defined here. ``seq`` and ``t`` are assigned by
``session.emit``, not here, so builder functions return plain dicts without
those keys.

Public API
----------
* :data:`RESULT_BUDGET` — byte budget for a JSON-encoded result summary.
* :func:`describe` — build a bounded, structured result summary (preferred).
* :func:`summarise_result` — thin legacy summary without pool context.
* :func:`validate_result` — raise ``ValueError`` if a result summary breaks rules.
* :func:`call_event` — build a ``kind == "call"`` event dict.
* :func:`pool_event` — build a ``kind == "pool"`` event dict.
* :func:`message_event` — build a ``kind == "message"`` event dict.

Schema rules (enforced by :func:`validate_result`):
- ``result`` must be JSON-serializable (no numpy arrays, DataFrames, raw objects).
- JSON-encoded size must not exceed :data:`RESULT_BUDGET` bytes.
- Unknown top-level keys on any event are tolerated by all projections.
- Every optional key may be absent; projections must not assume presence.
"""
from __future__ import annotations

import json as _json

RESULT_BUDGET = 4096  # bytes (JSON-encoded)


def _cap_text(text: str, max_lines: int = 5, max_chars: int = 200) -> str:
    """Cap *text* to *max_lines* lines, each at most *max_chars* characters."""
    lines = text.splitlines()
    capped = [
        (ln[:max_chars] + "…" if len(ln) > max_chars else ln)
        for ln in lines[:max_lines]
    ]
    if len(lines) > max_lines:
        capped.append(f"… ({len(lines) - max_lines} more lines)")
    return "\n".join(capped)


def describe(
    obj,
    *,
    pool_id: str | None = None,
    label: str | None = None,
    source: str | None = None,
    before: dict | None = None,
) -> dict:
    """Bounded structured summary of a call result. Pure. Never calls format_arg.

    Parameters
    ----------
    obj:
        The live return value to summarise.
    pool_id:
        Pool id of the result entry (e.g. ``"motl_1"``); set by
        :func:`~cryocat.app.apputils.run_operation_to_pool`.
    label:
        Human-readable label of the pool entry.
    source:
        Input file path for load operations.
    before:
        ``{"n_rows": int}`` captured before the call; produces a ``delta``
        field when the result also has ``n_rows``.

    Returns
    -------
    dict
        JSON-serializable summary within :data:`RESULT_BUDGET` bytes.
    """
    if obj is None:
        d: dict = {"type": "None"}
        if pool_id is not None:
            d["pool_id"] = pool_id
        return d

    d = {"type": type(obj).__name__}

    # Rich text from str() — skipped for primitives whose value field suffices
    if not isinstance(obj, (int, float, bool)):
        try:
            d["text"] = _cap_text(str(obj))
        except Exception:
            pass

    # Motl-specific structured fields
    if hasattr(obj, "df"):
        try:
            cols = list(obj.df.columns)
            d["n_rows"] = int(len(obj.df))
            d["n_columns"] = len(cols)
            d["columns"] = cols[:15]
        except Exception:
            pass
    elif hasattr(obj, "shape"):
        try:
            d["shape"] = [int(x) for x in obj.shape]
        except Exception:
            pass
    elif isinstance(obj, str):
        d["value"] = obj
    elif isinstance(obj, (int, float, bool)):
        d["value"] = obj

    # Pool identity
    if pool_id is not None:
        d["pool_id"] = pool_id
    if label is not None:
        d["label"] = label
    if source is not None:
        d["source"] = source

    # Delta (row-count change; only when receiver rows were captured before the call)
    if before is not None and "n_rows" in d:
        before_n = before.get("n_rows")
        if before_n is not None:
            d["delta"] = {"n_rows_before": int(before_n), "n_rows_after": d["n_rows"]}

    # Enforce byte budget — truncate aggressively before raising
    try:
        raw = _json.dumps(d)
    except (TypeError, ValueError):
        return {"type": type(obj).__name__}
    if len(raw.encode()) > RESULT_BUDGET:
        if "text" in d:
            try:
                d["text"] = _cap_text(str(obj), max_lines=2, max_chars=80)
            except Exception:
                d.pop("text", None)
        if len(_json.dumps(d, default=str).encode()) > RESULT_BUDGET:
            d.pop("text", None)
        if len(_json.dumps(d, default=str).encode()) > RESULT_BUDGET:
            d.pop("columns", None)
        if len(_json.dumps(d, default=str).encode()) > RESULT_BUDGET:
            d.pop("delta", None)

    return d


def summarise_result(value) -> dict | None:
    """Thin result summary — legacy helper for callers without pool context.

    Prefer :func:`describe` when pool identity (``pool_id``, ``label``) is
    available.  Kept for the console execute path which handles bare
    expressions that are not pool operations.
    """
    if value is None:
        return None
    summary: dict = {"type": type(value).__name__}
    if hasattr(value, "df"):
        try:
            summary["n_rows"] = int(len(value.df))
        except Exception:
            pass
    elif hasattr(value, "shape"):
        try:
            summary["shape"] = [int(d) for d in value.shape]
        except Exception:
            pass
    elif isinstance(value, str):
        summary["value"] = value
    elif isinstance(value, (int, float, bool)):
        summary["value"] = value
    return summary


def validate_result(d: dict | None) -> None:
    """Raise ``ValueError`` if *d* breaks the result summary rules.

    Rules:
    - Must be JSON-serializable (no numpy arrays, DataFrames, raw objects).
    - JSON-encoded size must not exceed :data:`RESULT_BUDGET` bytes.

    ``None`` (void result) is always valid.
    """
    if d is None:
        return
    try:
        raw = _json.dumps(d)
    except (TypeError, ValueError) as e:
        raise ValueError(f"result contains non-JSON-serializable value: {e}") from e
    byte_len = len(raw.encode("utf-8"))
    if byte_len > RESULT_BUDGET:
        raise ValueError(
            f"result is {byte_len} bytes; budget is {RESULT_BUDGET}"
        )


def call_event(
    fn_name: str,
    kwargs_src: dict[str, str],
    *,
    status: str,
    imports: list[list[str]] | None = None,
    receiver: str | None = None,
    assign_to: str | None = None,
    result: dict | None = None,
    error: dict | None = None,
    duration_s: float | None = None,
    source: str = "gui",
    tool: str | None = None,
    command_src: str | None = None,
) -> dict:
    """Build a ``kind == "call"`` event dict.

    Parameters
    ----------
    fn_name : str
        Fully qualified function name, e.g.
        ``"cryocat.core.cryomotl.Motl.clean_by_distance"``.
    kwargs_src : dict[str, str]
        Per-argument Python source strings, as returned by
        ``_render_value`` for each kwarg.
    status : str
        ``"ok"`` or ``"error"``.
    imports : list of [short, statement] pairs, optional
        Import statements needed to execute the call.
    receiver : str or None
        Variable name the method is called on (``"motl_0"``), or ``None``
        for free functions.
    assign_to : str or None
        Variable name the result is bound to, or ``None``.
    result : dict or None
        Result summary (success path only); must pass :func:`validate_result`.
    error : dict or None
        Error info (error path only): ``{"type", "msg", "traceback"}``.
    duration_s : float or None
        Wall-clock seconds for the call.
    source : str
        ``"gui"`` (default) or ``"console"``.
    tool : str or None
        GUI tool name, e.g. ``"editor"``.
    command_src : str or None
        For console events: the desugared Python source of the command.
        The script projection uses this verbatim instead of reconstructing
        the call from ``fn`` / ``kwargs_src``.
    """
    event: dict = {
        "kind": "call",
        "status": status,
        "source": source,
        "fn": fn_name,
        "imports": imports or [],
        "receiver": receiver,
        "assign_to": assign_to,
        "kwargs_src": kwargs_src,
    }
    if tool is not None:
        event["tool"] = tool
    if command_src is not None:
        event["command_src"] = command_src
    if duration_s is not None:
        event["duration_s"] = round(duration_s, 3)
    if status == "ok" and result is not None:
        event["result"] = result
    if status == "error" and error is not None:
        event["error"] = error
    return event


def pool_event(
    action: str,
    motl_id: str,
    *,
    detail: dict | None = None,
) -> dict:
    """Build a ``kind == "pool"`` event dict.

    Parameters
    ----------
    action : str
        ``"rename"``, ``"remove"``, or ``"set_active"``.
    motl_id : str
        The pool entry affected, e.g. ``"motl_3"``.
    detail : dict, optional
        Action-specific payload (e.g. new label for a rename).
    """
    return {
        "kind": "pool",
        "action": action,
        "motl_id": motl_id,
        "detail": detail or {},
    }


def message_event(text: str, *, level: str = "info") -> dict:
    """Build a ``kind == "message"`` event dict.

    Parameters
    ----------
    text : str
        The user-visible message (replaces ``print_dash`` output).
    level : str
        ``"info"`` (default) or ``"error"``.
    """
    return {"kind": "message", "text": text, "level": level}
