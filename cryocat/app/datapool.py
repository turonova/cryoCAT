"""Data pool — server-side payload store and pure state reducers.

Mirrors the motl pool (pool.py) for heterogeneous loaded data:
DataFrames, numpy arrays, dicts, and volumes.  Browser stores carry only
lightweight DataEntry handles; actual objects live in _payloads (server-side).

IDs are "data_1", "data_2", … using DATA_POOL_NEXT_ID counter; never reused.

Bridging to the motl pool viewer
----------------------------------
For DataFrame/2D-array entries, set_view_df() writes into pool._payloads
under the reserved key "dp-view" so pool.get_rows("dp-view") works and the
existing tablegrid component can display data pool entries without changes.
Call clear_view_df() when nothing is selected.

Table pool (non-motl tables)
-----------------------------
A second, simpler store (``_table_payloads``) holds DataFrames for non-motl
table components (NN analysis, tango twist, tango descriptors).  These refs
carry ``data_id`` values with the ``"tab_"`` prefix so they are distinct from
the file-pool ``"data-N"`` ids.  Use ``insert`` / ``resolve_df`` /
``resolve_n_rows`` / ``id_column_for`` for this lighter API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

_MAX_COLUMNS = 50  # cap columns list in the handle so the store stays lightweight


# ── Server-side payload store ─────────────────────────────────────────────────
# Process-local; clears on hot-reload / restart.  Keyed by data_id.

_payloads: dict[str, Any] = {}


def clear_payloads() -> None:
    """Drop all server-side payloads.  For tests and hot-reload only."""
    _payloads.clear()


# ── Exception ─────────────────────────────────────────────────────────────────

class DataPayloadMissing(Exception):
    """Raised when the server-side payload for a data_id is absent.

    Includes data_id, label, and source_path in the message so the user knows
    which file to reload.
    """


# ── Handle dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DataEntry:
    """Lightweight, JSON-round-trip-safe handle for one data pool entry.

    All fields survive serialisation through a dcc.Store; no large arrays are
    stored here.
    """
    data_id:     str
    label:       str
    kind:        str               # "dataframe" | "array" | "volume" | "dict"
    reader:      str               # GuiEntry key used to load the entry
    source_path: str               # original file path
    n_rows:      int | None        # row count (DataFrame / 2D-array / dict keys)
    columns:     list[str] | None  # column names, capped at _MAX_COLUMNS
    shape:       tuple | None      # array / volume shape
    dtype:       str | None        # array dtype string
    id_column:   str | None = None                   # identity column name (for row-level edit operations)
    motl_links:  dict[str, list[str]] | None = None  # role → list of motl ids


# ── Pool state ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DataPoolState:
    """Immutable snapshot of the two browser-side data pool stores.

    ``registry``      — ``{ data_id: asdict(DataEntry) }``
    ``next_id``       — monotone counter; never reused.
    ``kind_counters`` — per-kind insertion counters; ``{"nn": 2, "desc": 1}`` means
                        two NN entries and one descriptor entry have been created.
                        Stored inside the registry dict under ``"__kind_counters__"``
                        to avoid a third dcc.Store; stripped out in ``from_stores``
                        so ``registry`` is always clean (no reserved key visible).
    """
    registry:      dict
    next_id:       int
    kind_counters: dict = field(default_factory=dict)

    @classmethod
    def from_stores(cls, registry, next_id) -> DataPoolState:
        """Construct from store values.  Accepts ``None`` for empty state."""
        reg = dict(registry or {})
        kind_counters = dict(reg.pop("__kind_counters__", None) or {})
        return cls(
            registry=reg,
            next_id=int(next_id or 0),
            kind_counters=kind_counters,
        )

    def to_stores(self) -> tuple:
        """Return ``(registry, next_id)`` for unpacking into Dash Outputs.

        The kind_counters dict is embedded in the returned registry under the
        reserved key ``"__kind_counters__"`` so it survives the dcc.Store
        round-trip without requiring a third store.
        """
        reg = {**self.registry, "__kind_counters__": self.kind_counters}
        return reg, self.next_id


# ── Kind detection ────────────────────────────────────────────────────────────

def _detect_kind(payload: Any) -> str:
    """Infer the data kind from the payload's Python type."""
    if isinstance(payload, pd.DataFrame):
        return "dataframe"
    if isinstance(payload, np.ndarray):
        return "volume" if payload.ndim == 3 else "array"
    if isinstance(payload, dict):
        return "dict"
    # Fallback: treat as dataframe so the pool viewer can try to display it.
    return "dataframe"


# ── Entry factory ─────────────────────────────────────────────────────────────

def _make_entry(
    payload: Any,
    data_id: str,
    label: str,
    reader_key: str,
    source_path: str,
    motl_links: dict[str, list[str]] | None = None,
) -> DataEntry:
    """Build a DataEntry handle from a payload.  Pure function; no side effects."""
    kind: str = _detect_kind(payload)
    n_rows:  int | None       = None
    columns: list[str] | None = None
    shape:   tuple | None     = None
    dtype:   str | None       = None

    if kind == "dataframe":
        df: pd.DataFrame = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
        n_rows  = len(df)
        columns = list(df.columns)[:_MAX_COLUMNS]
        shape   = tuple(df.shape)

    elif kind in ("volume", "array"):
        arr: np.ndarray = payload
        shape = tuple(arr.shape)
        dtype = str(arr.dtype)
        if arr.ndim == 1:
            n_rows = len(arr)
        elif arr.ndim == 2:
            n_rows  = arr.shape[0]
            columns = [f"col_{i}" for i in range(min(arr.shape[1], _MAX_COLUMNS))]

    elif kind == "dict":
        n_rows = len(payload)

    return DataEntry(
        data_id=data_id,
        label=label,
        kind=kind,
        reader=reader_key,
        source_path=source_path,
        n_rows=n_rows,
        columns=columns,
        shape=shape,
        dtype=dtype,
        motl_links=motl_links,
    )


# ── Pure reducers ─────────────────────────────────────────────────────────────

def insert_entry(
    state: DataPoolState,
    payload: Any,
    *,
    label: str,
    reader: str,
    source_path: str,
    motl_links: dict[str, list[str]] | None = None,
    data_id: str | None = None,
    entry_kind: str | None = None,
) -> tuple[DataPoolState, str]:
    """Store *payload* server-side; return (new_state, data_id).

    ID resolution order:
    1. *data_id* provided → used as-is (legacy / explicit override).
    2. *entry_kind* provided → ``f"{entry_kind}_{counter}"`` where *counter* is
       the per-kind count (1-based, never reused).  Example: first ``"nn"`` entry
       gets ``"nn_1"`` regardless of how many other entries exist.
    3. Neither → ``f"data_{state.next_id + 1}"`` (global fallback).

    The generic ``next_id`` counter increments unconditionally so future
    ``data_N`` ids stay unique even when *entry_kind* is used.
    """
    kind_counters = dict(state.kind_counters)
    if data_id is None:
        if entry_kind is not None:
            count = kind_counters.get(entry_kind, 0) + 1
            kind_counters[entry_kind] = count
            data_id = f"{entry_kind}_{count}"
        else:
            data_id = f"data_{state.next_id + 1}"
    _payloads[data_id] = payload
    entry = _make_entry(payload, data_id, label, reader, source_path, motl_links)
    return DataPoolState(
        registry={**state.registry, data_id: asdict(entry)},
        next_id=state.next_id + 1,
        kind_counters=kind_counters,
    ), data_id


def remove_entry(state: DataPoolState, data_id: str) -> DataPoolState:
    """Remove an entry from stores and drop its server-side payload.

    No-op if *data_id* is not in the registry.  ``next_id`` is unchanged.
    """
    if data_id not in state.registry:
        return state
    _payloads.pop(data_id, None)
    return DataPoolState(
        registry={k: v for k, v in state.registry.items() if k != data_id},
        next_id=state.next_id,
        kind_counters=state.kind_counters,
    )


def replace_entry(state: DataPoolState, data_id: str, payload: Any) -> DataPoolState:
    """Replace an entry's payload in-place, re-deriving all computed fields.

    The entry keeps its label, reader, source_path, motl_links, and id_column.
    ``next_id`` is unchanged so existing slot assignments referencing *data_id*
    are unaffected.  No-op if *data_id* is not in the registry.
    """
    if data_id not in state.registry:
        return state
    old = state.registry[data_id]
    _payloads[data_id] = payload
    entry = _make_entry(
        payload, data_id,
        old["label"], old["reader"], old.get("source_path", ""),
        old.get("motl_links"),
    )
    entry_dict = asdict(entry)
    entry_dict["id_column"] = old.get("id_column")
    return DataPoolState(
        registry={**state.registry, data_id: entry_dict},
        next_id=state.next_id,
        kind_counters=state.kind_counters,
    )



def get_payload(data_id: str, state: DataPoolState | None = None) -> Any:
    """Return the server-side payload for *data_id*.

    Parameters
    ----------
    data_id:
        Pool entry id (e.g. ``"data_3"``).
    state:
        Optional :class:`DataPoolState`; used only to include ``label`` and
        ``source_path`` in the error message when the payload has been evicted.

    Raises
    ------
    DataPayloadMissing
        If the payload is not in the server-side store.
    """
    payload = _payloads.get(data_id)
    if payload is None:
        label = ""
        source_path = ""
        if state is not None:
            entry_dict = state.registry.get(data_id, {})
            label = entry_dict.get("label", "")
            source_path = entry_dict.get("source_path", "")
        msg = (
            f"Payload for {data_id!r} (label={label!r}, source={source_path!r}) "
            f"is no longer in memory — the app may have restarted; reload the file."
        )
        raise DataPayloadMissing(msg)
    return payload


# ── Pool viewer bridge ────────────────────────────────────────────────────────

def set_view_df(data_id: str, state: DataPoolState | None = None) -> None:
    """Write a DataFrame view of *data_id* into pool._payloads["dp-view"].

    This allows the existing tablegrid component to display data pool entries
    by reading from pool.get_rows("dp-view").

    * DataFrame payloads are written as-is.
    * 2D ndarray payloads are converted to a DataFrame with ``col_0`` … ``col_N``
      column names.
    * All other kinds are no-ops (volume, dict, 1D array).
    """
    from cryocat.app.pool import _payloads as _pool_payloads, PoolPayload

    payload = _payloads.get(data_id)
    if payload is None:
        return

    kind = _detect_kind(payload)

    if kind == "dataframe":
        df = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
    elif kind == "array" and isinstance(payload, np.ndarray) and payload.ndim == 2:
        cols = [f"col_{i}" for i in range(payload.shape[1])]
        df = pd.DataFrame(payload, columns=cols)
    else:
        return  # volumes, dicts, and 1D arrays are not tabular

    _pool_payloads["dp-view"] = PoolPayload(rows=df, extra=None)


def clear_view_df() -> None:
    """Remove the dp-view bridge entry from pool._payloads.

    Call when no data pool entry is selected so the tablegrid is empty.
    """
    from cryocat.app.pool import _payloads as _pool_payloads
    _pool_payloads.pop("dp-view", None)


def set_view_df_direct(df: pd.DataFrame) -> None:
    """Bridge *df* directly into pool._payloads['dp-view'] for working-copy display.

    Used by the working-copy path in tableeditor so the tablegrid can show the
    working copy without creating a data pool entry.
    """
    from cryocat.app.pool import _payloads as _pool_payloads, PoolPayload
    _pool_payloads["dp-view"] = PoolPayload(rows=df.copy(), extra=None)


def replace_payload(
    state: "DataPoolState",
    data_id: str,
    df: pd.DataFrame,
) -> "DataPoolState":
    """Replace the DataFrame payload for an existing data pool entry.

    Updates n_rows and columns in the registry handle; leaves all other
    metadata (label, reader, source_path, etc.) unchanged.  No-op if
    *data_id* is not in the registry.
    """
    if data_id not in state.registry:
        return state
    _payloads[data_id] = df.copy()
    old = dict(state.registry[data_id])
    old["n_rows"] = len(df)
    old["columns"] = list(df.columns)[:_MAX_COLUMNS]
    return DataPoolState(
        registry={**state.registry, data_id: old},
        next_id=state.next_id,
        kind_counters=state.kind_counters,
    )


# ── Table pool (non-motl tables) ──────────────────────────────────────────────
# Simpler store for NN analysis, tango twist, and tango descriptor DataFrames.
# IDs use the "table_N" format; file-pool uses "data_N".
# The ref that flows through dcc.Store is:
#   {"table_id": "table_1", "n_rows": N, "id_column": str|None, "label": str}

_table_payloads: dict[str, pd.DataFrame] = {}
_table_counter: list[int] = [0]


def insert(
    df: pd.DataFrame,
    *,
    label: str,
    id_column: str | None,
    source: str = "",
    motl_links: dict[str, list[str]] | None = None,
) -> dict:
    """Store *df* and return a ref dict for dcc.Store."""
    _table_counter[0] += 1
    table_id = f"table_{_table_counter[0]}"
    _table_payloads[table_id] = df.copy()
    return {
        "table_id": table_id,
        "n_rows": len(df),
        "id_column": id_column,
        "label": label,
        "source": source,
        "motl_links": motl_links,
    }


def resolve_df(ref: dict | None) -> pd.DataFrame | None:
    """Return the DataFrame for a table-pool ref, or ``None``."""
    if not isinstance(ref, dict):
        return None
    table_id = ref.get("table_id")
    if not table_id:
        return None
    return _table_payloads.get(table_id)


def resolve_payload_df(ref: dict | None) -> pd.DataFrame | None:
    """Return the DataFrame for a data-pool ref ``{"data_id": "data_N"}``, or ``None``."""
    if not isinstance(ref, dict):
        return None
    data_id = ref.get("data_id")
    if not data_id:
        return None
    payload = _payloads.get(data_id)
    if not isinstance(payload, pd.DataFrame):
        return None
    return payload


def resolve_n_rows(ref: dict | None) -> int:
    """Return the row count recorded in *ref*, or 0."""
    if not isinstance(ref, dict):
        return 0
    return ref.get("n_rows", 0)


def id_column_for(ref: dict | None) -> str | None:
    """Return the identity column embedded in *ref*, or ``None``."""
    if not isinstance(ref, dict):
        return None
    return ref.get("id_column")
