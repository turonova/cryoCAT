"""Server-side working copy for the Tables tab (W1, W4).

Held in module-level dicts keyed by source entry id — never in a dcc.Store.
Lazy: the copy is created on the first operation, not when the source is opened.
One copy at a time per source id; switching source does not discard the copy
(W3 — working copies are never silently lost).
"""
from __future__ import annotations
import pandas as pd

# ── Server-side storage ───────────────────────────────────────────────────────
_wc_data: dict[str, pd.DataFrame] = {}
_wc_meta: dict[str, dict] = {}
# meta keys per source_id:
#   ops_count    int    operations applied since init (0 = clean)
#   source_n_rows int   n_rows of source at init time (revision proxy — W3)
#   source_kind  str    "motl" | "data"
#   source_reader str   reader key from dp_registry (e.g. "nn", "twist", "table_op")


# ── Key extraction ────────────────────────────────────────────────────────────

def source_id_for_ref(ref: dict | None) -> str | None:
    """Extract pool key from a source ref (motl_id or data_id)."""
    if not ref:
        return None
    return ref.get("motl_id") or ref.get("data_id")


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def has_copy(source_id: str) -> bool:
    return source_id in _wc_data


def get_copy(source_id: str) -> pd.DataFrame | None:
    return _wc_data.get(source_id)


def init_copy(
    source_id: str,
    df: pd.DataFrame,
    *,
    source_n_rows: int,
    source_kind: str,
    source_reader: str,
) -> None:
    """Initialise working copy from source df — called lazily on first op (W1)."""
    _wc_data[source_id] = df.copy()
    _wc_meta[source_id] = {
        "ops_count": 0,
        "source_n_rows": source_n_rows,
        "source_kind": source_kind,
        "source_reader": source_reader,
    }


def apply_op(source_id: str, result_df: pd.DataFrame) -> int:
    """Store operation result; return new ops_count."""
    _wc_data[source_id] = result_df.copy()
    meta = _wc_meta.setdefault(source_id, {"ops_count": 0})
    meta["ops_count"] = meta.get("ops_count", 0) + 1
    return meta["ops_count"]


def clear(source_id: str) -> None:
    """Remove the working copy after commit or explicit discard."""
    _wc_data.pop(source_id, None)
    _wc_meta.pop(source_id, None)


def clear_all() -> None:
    """Drop all working copies — for tests and hot-reload only."""
    _wc_data.clear()
    _wc_meta.clear()


# ── State accessors ───────────────────────────────────────────────────────────

def get_ops_count(source_id: str) -> int:
    return _wc_meta.get(source_id, {}).get("ops_count", 0)


def get_meta(source_id: str) -> dict:
    return dict(_wc_meta.get(source_id, {}))


def indicator_text(source_id: str) -> str:
    """Human-readable unsaved-changes label; empty string when clean."""
    n = get_ops_count(source_id)
    if n == 0:
        return ""
    return f"{n} operation{'s' if n != 1 else ''} pending — not yet applied"


def source_changed(source_id: str, current_n_rows: int) -> bool:
    """True if source n_rows changed since copy was initialised (W3 revision proxy)."""
    stored = _wc_meta.get(source_id, {}).get("source_n_rows")
    return stored is not None and stored != current_n_rows


# ── Validation (W4) ──────────────────────────────────────────────────────────

def validate_for_apply(
    df: pd.DataFrame,
    source_kind: str,
    source_reader: str = "",
) -> tuple[bool, str]:
    """Return (ok, reason) before applying working copy back to source (W4).

    Reuses the same column-check functions used at load time — not reimplemented.

    =====================  ================================
    source_kind / reader   check used
    =====================  ================================
    "motl"                 satisfies_motl_schema
    "data" + "nn"          NearestNeighbors.check_nn_columns
    "data" + "twist"       TwistDescriptor.check_twist_columns
    "data" + "nn_twist"    TwistDescriptor.check_twist_columns
    "data" + other         no constraint (plain table)
    =====================  ================================
    """
    if source_kind == "motl":
        from cryocat.app.components._tableops import satisfies_motl_schema
        ok, missing = satisfies_motl_schema(df)
        if not ok:
            cols = ", ".join(missing[:5]) + ("…" if len(missing) > 5 else "")
            return False, f"Missing motl columns: {cols}"
        return True, ""
    if source_reader == "nn":
        from cryocat.analysis.nnana import NearestNeighbors
        missing = NearestNeighbors.check_nn_columns(df)
        if missing:
            cols = ", ".join(missing[:5]) + ("…" if len(missing) > 5 else "")
            return False, f"Missing NN columns: {cols}"
        return True, ""
    if source_reader in ("twist", "nn_twist"):
        from cryocat.analysis.tango import TwistDescriptor
        missing = TwistDescriptor.check_twist_columns(df)
        if missing:
            cols = ", ".join(missing[:5]) + ("…" if len(missing) > 5 else "")
            return False, f"Missing twist columns: {cols}"
        return True, ""
    return True, ""
