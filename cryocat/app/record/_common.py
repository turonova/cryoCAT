"""Shared helpers for the projection modules.

All functions are pure: no I/O, no imports from cryocat.app at module level.
"""
from __future__ import annotations


def _callable_expr(ev: dict) -> str:
    """Build the callable part (without args) from a call event.

    Returns either ``"receiver.method_name"`` (when ``receiver`` is set) or
    ``"short_module.ClassName.method"`` (reconstructed from imports + fn).
    """
    fn = ev.get("fn", "unknown")
    receiver = ev.get("receiver")
    if receiver:
        return f"{receiver}.{fn.rsplit('.', 1)[-1]}"

    imports = ev.get("imports") or []
    if imports:
        short = imports[0][0]
        # fn == "{full_module}.{qualname}"; find the boundary via the short name.
        needle = f".{short}."
        idx = fn.find(needle)
        if idx >= 0:
            return f"{short}.{fn[idx + len(needle):]}"
        # Module == short itself  ("short.QualName")
        prefix = f"{short}."
        if fn.startswith(prefix):
            return fn

    # Fallback: last two dot-separated components.
    parts = fn.rsplit(".", 2)
    if len(parts) >= 3:
        return f"{parts[-2]}.{parts[-1]}"
    if len(parts) == 2:
        return f"{parts[0]}.{parts[1]}"
    return fn


def _kwargs_str(ev: dict) -> str:
    """Render the kwargs as a comma-separated ``k=v`` string."""
    ks = ev.get("kwargs_src") or {}
    parts = []
    for k, v in ks.items():
        parts.append(f"{k}=None" if v is None else f"{k}={v}")
    return ", ".join(parts)


def call_expr(ev: dict) -> str:
    """Full call expression, e.g. ``motl_0.clean_by_distance(distance=20)``."""
    return f"{_callable_expr(ev)}({_kwargs_str(ev)})"


def format_result(r: dict | None) -> str:
    """Human-readable summary of a result dict.

    Returns e.g. ``'motl_1 "run1" · Motl, 26 rows, 20 columns'`` when pool
    identity fields are present, or ``'Motl, 1204 rows'`` for legacy results.
    """
    if not r:
        return ""
    typ = r.get("type", "")

    # Pool identity prefix
    pool_id = r.get("pool_id")
    label = r.get("label")
    if pool_id:
        var = pool_id.replace("-", "_")
        id_str = f'{var} "{label}"' if label else var
    else:
        id_str = ""

    # Type + structured fields
    type_parts: list[str] = [typ] if typ else []
    if "n_rows" in r:
        row_s = f"{r['n_rows']} rows"
        if r.get("dropped"):
            row_s += f" ({r['dropped']} removed)"
        type_parts.append(row_s)
        if "n_columns" in r:
            type_parts.append(f"{r['n_columns']} columns")
    elif "shape" in r:
        type_parts.append(f"shape {tuple(r['shape'])}")
    if "value" in r:
        type_parts.append(str(r["value"]))
    if "path" in r:
        type_parts.append(str(r["path"]))

    type_str = ", ".join(type_parts) if type_parts else typ

    if id_str and type_str:
        return f"{id_str} · {type_str}"
    return id_str or type_str


def ts_from(ev: dict) -> str:
    """Extract ``HH:MM:SS`` from a ``t`` field (ISO timestamp)."""
    t = ev.get("t", "")
    if "T" in t:
        return t.split("T")[1][:8]
    return t[:8] if len(t) >= 8 else (t or "??:??:??")
