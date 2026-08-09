"""Pool state model and pure reducers.

Three browser-side stores (`pool-registry`, `pool-meta`, `pool-next-id`) carry
lightweight handles and metadata.  Row data lives in the server-side
``_payloads`` dict, keyed by ``motl_id``.  No module outside this file may
hand-build store contents or reference id strings (use :mod:`cryocat.app.ids`).

§5 guarantees:
- ``motl_id`` is always ``f"motl-{next_id}"``, counter only increases, ids
  are never reused or renumbered after removal.
- :func:`insert_motl` stores a :class:`PoolPayload` server-side and returns a
  handle; row data never appears in a ``dcc.Store``.
- :func:`get_rows` / :func:`get_extra` raise :exc:`PoolPayloadMissing` when
  the server-side payload has been evicted (hot-reload / restart, §12).
- :meth:`PoolState.to_stores` output is JSON-round-trip safe and contains no
  row data.
- Every mutation bumps ``revision`` on the handle so dependent callbacks
  watching ``pool-registry`` re-fire (D4 of POOL_SERVER_SIDE_STORAGE.md).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


# ── Server-side payload store ────────────────────────────────────────────────────
# Process-local (§12).  Keyed by motl_id ("motl-1", "motl-2", …).
# Clears on hot-reload / restart; handles in the browser then point to missing
# payloads, which :func:`get_rows` surfaces as :exc:`PoolPayloadMissing`.

@dataclass
class PoolPayload:
    """Server-side container for a motl's DataFrames."""
    rows: pd.DataFrame
    extra: pd.DataFrame | None


_payloads: dict[str, PoolPayload] = {}
_snapshots: dict[str, "pd.DataFrame"] = {}


def clear_payloads() -> None:
    """Drop all server-side payloads.  For tests and hot-reload only."""
    _payloads.clear()
    _snapshots.clear()


# ── Exception ───────────────────────────────────────────────────────────────────

class PoolPayloadMissing(Exception):
    """Raised when the server-side payload for a motl_id is absent.

    This happens after a hot-reload or restart: the browser still holds the
    handle but the process-local ``_payloads`` dict is empty.
    """


# ── Handle ────────────────────────────────────────────────────────────────────────

_MAX_COLUMNS = 50  # cap so the handle stays small in the browser store


@dataclass(frozen=True)
class PoolEntry:
    label: str
    type: str
    n_rows: int
    n_columns: int
    columns: list[str]       # capped at _MAX_COLUMNS; for pickers/column dropdowns
    active: bool
    source_path: str | None  # original file path; shown in PoolPayloadMissing message
    revision: int            # bumped on every mutation so store-watchers re-fire
    has_tab: bool = True     # False for batch-loaded group members (hidden from individual rows)
    numeric_columns: list[str] = field(default_factory=list)       # numeric cols; for dropdowns
    column_ranges: dict[str, list[float]] = field(default_factory=dict)  # {col: [min, max]}
    tomo_ids: list[int] = field(default_factory=list)              # sorted unique tomo IDs


# ── Pool state ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PoolState:
    """Immutable snapshot of the three browser-side pool stores."""

    registry: dict   # { motl_id: asdict(PoolEntry) }
    meta: dict       # { motl_id: small dict | None }   (relion params, data_type, …)
    next_id: int

    @classmethod
    def from_stores(cls, registry, *args) -> PoolState:
        """Construct from store values.

        Accepts the new 3-arg form ``(registry, meta, next_id)`` *and* the
        legacy 5-arg form ``(registry, motls, extra, meta, next_id)`` — in the
        latter case ``motls`` and ``extra`` are silently discarded so existing
        call-sites (including tests) keep working without modification.
        """
        if len(args) == 2:
            meta, next_id = args
        elif len(args) == 4:
            _motls, _extra, meta, next_id = args  # legacy; row data ignored
        elif len(args) == 0:
            meta, next_id = {}, 0
        else:
            raise TypeError(
                f"PoolState.from_stores() expects 3 or 5 positional args, "
                f"got {1 + len(args)}"
            )
        return cls(
            registry=dict(registry or {}),
            meta=dict(meta or {}),
            next_id=int(next_id or 0),
        )

    def to_stores(self) -> tuple:
        """Return ``(registry, meta, next_id)`` for unpacking into Dash Outputs."""
        return self.registry, self.meta, self.next_id


def default_label(next_id: int) -> str:
    return f"Motl {next_id + 1}"


# ── Metadata helper ──────────────────────────────────────────────────────────────

def _compute_entry_metadata(df: pd.DataFrame) -> tuple[list[str], dict[str, list[float]], list[int]]:
    """Compute lightweight metadata from a DataFrame. Pure.

    Returns ``(numeric_columns, column_ranges, tomo_ids)``.  All values are
    JSON-serializable Python types so they survive a dcc.Store round-trip.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    ranges: dict[str, list[float]] = {}
    for col in num_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if pd.isna(col_min) or pd.isna(col_max) or col_min == col_max:
            continue
        mn, mx = float(col_min), float(col_max)
        step = 1.0 if pd.api.types.is_integer_dtype(df[col]) else ((mx - mn) / 100 or 1.0)
        ranges[col] = [mn, mx, step]  # [min, max, step]

    tomo_ids: list[int] = []
    if "tomo_id" in df.columns:
        tomo_ids = sorted(int(t) for t in df["tomo_id"].dropna().unique())

    return num_cols, ranges, tomo_ids


# ── Pure reducers ────────────────────────────────────────────────────────────────

def insert_motl(
    state: PoolState,
    rows: pd.DataFrame | list,
    *,
    label: str | None = None,
    motl_type: str = "emmotl",
    extra: pd.DataFrame | list | None = None,
    meta: dict | None = None,
    source_path: str | None = None,
    has_tab: bool = True,
) -> tuple[PoolState, str]:
    """Append one motl; store its payload server-side; return (new_state, motl_id)."""
    motl_id = f"motl-{state.next_id + 1}"

    df = _to_df(rows)
    extra_df = _to_df(extra) if extra is not None else None

    _payloads[motl_id] = PoolPayload(rows=df, extra=extra_df)

    cols = list(df.columns)[:_MAX_COLUMNS]
    num_cols, ranges, tids = _compute_entry_metadata(df)
    entry = PoolEntry(
        label=label or default_label(state.next_id),
        type=motl_type,
        n_rows=len(df),
        n_columns=len(df.columns),
        columns=cols,
        active=True,
        source_path=source_path,
        revision=0,
        has_tab=has_tab,
        numeric_columns=num_cols,
        column_ranges=ranges,
        tomo_ids=tids,
    )
    return PoolState(
        registry={**state.registry, motl_id: asdict(entry)},
        meta={**state.meta, motl_id: meta},
        next_id=state.next_id + 1,
    ), motl_id


def remove_motl(state: PoolState, motl_id: str) -> PoolState:
    """Remove a motl from all stores and the server-side payload. No-op if absent."""
    if motl_id not in state.registry:
        return state
    _payloads.pop(motl_id, None)
    return PoolState(
        registry={k: v for k, v in state.registry.items() if k != motl_id},
        meta={k: v for k, v in state.meta.items() if k != motl_id},
        next_id=state.next_id,
    )


def set_active(state: PoolState, motl_id: str, active: bool) -> PoolState:
    """Toggle the active flag for one entry. No-op if the id is unknown."""
    if motl_id not in state.registry:
        return state
    entry = {
        **state.registry[motl_id],
        "active": active,
        "revision": state.registry[motl_id].get("revision", 0) + 1,
    }
    return PoolState(
        registry={**state.registry, motl_id: entry},
        meta=state.meta,
        next_id=state.next_id,
    )


def replace_motl_rows(
    state: PoolState,
    motl_id: str,
    rows: pd.DataFrame | list,
    *,
    label: str | None = None,
    motl_type: str | None = None,
) -> PoolState:
    """Update rows for an existing entry and bump its ``revision``.

    A no-op if *motl_id* is not in the registry.  ``next_id`` is unchanged.
    ``extra`` is preserved from the existing payload; only rows change.
    """
    if motl_id not in state.registry:
        return state

    df = _to_df(rows)
    existing = _payloads.get(motl_id)
    _payloads[motl_id] = PoolPayload(rows=df, extra=existing.extra if existing else None)

    num_cols, ranges, tids = _compute_entry_metadata(df)
    entry = dict(state.registry[motl_id])
    entry["n_rows"] = len(df)
    entry["n_columns"] = len(df.columns)
    entry["columns"] = list(df.columns)[:_MAX_COLUMNS]
    entry["numeric_columns"] = num_cols
    entry["column_ranges"] = ranges
    entry["tomo_ids"] = tids
    entry["revision"] = entry.get("revision", 0) + 1
    if label is not None:
        entry["label"] = label
    if motl_type is not None:
        entry["type"] = motl_type
    return PoolState(
        registry={**state.registry, motl_id: entry},
        meta=state.meta,
        next_id=state.next_id,
    )


def get_rows(motl_id: str, *, state: PoolState | None = None) -> pd.DataFrame:
    """Return the :class:`~pandas.DataFrame` for *motl_id*.

    Parameters
    ----------
    motl_id:
        Pool entry id (e.g. ``"motl-3"``).
    state:
        Optional :class:`PoolState`; used only to include ``source_path`` in
        the error message when the payload has been evicted.

    Raises
    ------
    PoolPayloadMissing
        If the payload is not in the server-side store (e.g. after restart).
    """
    payload = _payloads.get(motl_id)
    if payload is None:
        source = None
        if state is not None:
            source = state.registry.get(motl_id, {}).get("source_path")
        msg = f"'{motl_id}' is no longer in memory (the app restarted)"
        if source:
            msg += f"; reload it from {source!r}"
        raise PoolPayloadMissing(msg)
    return payload.rows


def get_extra(motl_id: str) -> pd.DataFrame | None:
    """Return the extra :class:`~pandas.DataFrame` for *motl_id*, or ``None``."""
    payload = _payloads.get(motl_id)
    return payload.extra if payload is not None else None


def save_snapshot(motl_id: str, df: "pd.DataFrame") -> None:
    """Save a pre-operation snapshot for pool-aware undo.  Server-side only."""
    import pandas as _pd
    _snapshots[motl_id] = df.copy()


def restore_snapshot(motl_id: str) -> "pd.DataFrame | None":
    """Return and remove the undo snapshot for *motl_id*, or None if absent."""
    return _snapshots.pop(motl_id, None)


def active_ids(state: PoolState) -> list[str]:
    """Return motl ids whose active flag is True, in insertion order."""
    return [mid for mid, entry in state.registry.items() if entry.get("active", True)]


def set_has_tab(state: PoolState, motl_id: str, has_tab: bool) -> PoolState:
    """Toggle the has_tab flag for one entry and bump revision. No-op if id unknown."""
    if motl_id not in state.registry:
        return state
    entry = {
        **state.registry[motl_id],
        "has_tab": has_tab,
        "revision": state.registry[motl_id].get("revision", 0) + 1,
    }
    return PoolState(
        registry={**state.registry, motl_id: entry},
        meta=state.meta,
        next_id=state.next_id,
    )


# ── Group state ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PoolGroup:
    """Lightweight handle for a named, ordered collection of pool entries."""
    label: str
    members: list[str]   # ordered motl_ids; order is semantically meaningful (D3)


@dataclass(frozen=True)
class GroupState:
    """Snapshot of the pool-groups store.  Separate from PoolState so existing
    callbacks that write pool stores need no changes to their Output lists."""

    groups: dict   # { group_id: asdict(PoolGroup) }
    next_id: int   # incrementing counter; never reused (mirrors PoolState.next_id)

    @classmethod
    def from_store(cls, data: dict | None) -> GroupState:
        data = data or {}
        return cls(
            groups=dict(data.get("groups", {})),
            next_id=int(data.get("next_id", 0)),
        )

    def to_store(self) -> dict:
        return {"groups": self.groups, "next_id": self.next_id}


def default_group_label(next_id: int) -> str:
    return f"Group {next_id + 1}"


# ── Group pure reducers ───────────────────────────────────────────────────────────

def _evict_from_other_groups(gstate: GroupState, motl_id: str, except_gid: str | None = None) -> GroupState:
    """Remove motl_id from every group except except_gid.  Enforces one-group-per-motl (R2)."""
    new_groups = {}
    for gid, g in gstate.groups.items():
        if gid == except_gid:
            new_groups[gid] = g
        else:
            members = [m for m in g.get("members", []) if m != motl_id]
            new_groups[gid] = {**g, "members": members}
    return GroupState(groups=new_groups, next_id=gstate.next_id)


def create_group(
    gstate: GroupState,
    member_ids: list[str],
    *,
    label: str | None = None,
) -> tuple[GroupState, str]:
    """Append a new group; return (new_state, group_id).  Order of member_ids is preserved.

    R2: each member is removed from any previous group first — adding to a group is a *move*.
    """
    for mid in member_ids:
        gstate = _evict_from_other_groups(gstate, mid, except_gid=None)
    group_id = f"group-{gstate.next_id + 1}"
    group = PoolGroup(label=label or default_group_label(gstate.next_id), members=list(member_ids))
    return GroupState(
        groups={**gstate.groups, group_id: asdict(group)},
        next_id=gstate.next_id + 1,
    ), group_id


def rename_group(gstate: GroupState, group_id: str, label: str) -> GroupState:
    """Rename a group.  No-op if group_id is unknown."""
    if group_id not in gstate.groups:
        return gstate
    g = {**gstate.groups[group_id], "label": label}
    return GroupState(groups={**gstate.groups, group_id: g}, next_id=gstate.next_id)


def add_to_group(gstate: GroupState, group_id: str, motl_id: str) -> GroupState:
    """Append motl_id to the end of a group.

    R2: motl_id is removed from any previous group first — at most one group per motl.
    No-op if already present in this group or group is unknown.
    """
    if group_id not in gstate.groups:
        return gstate
    gstate = _evict_from_other_groups(gstate, motl_id, except_gid=group_id)
    g = gstate.groups[group_id]
    if motl_id in g.get("members", []):
        return gstate
    members = list(g.get("members", [])) + [motl_id]
    return GroupState(groups={**gstate.groups, group_id: {**g, "members": members}}, next_id=gstate.next_id)


def remove_from_group(gstate: GroupState, group_id: str, motl_id: str) -> GroupState:
    """Remove motl_id from a group.  No-op if absent or group unknown."""
    if group_id not in gstate.groups:
        return gstate
    g = gstate.groups[group_id]
    members = [m for m in g.get("members", []) if m != motl_id]
    return GroupState(groups={**gstate.groups, group_id: {**g, "members": members}}, next_id=gstate.next_id)


def reorder_group(gstate: GroupState, group_id: str, new_order: list[str]) -> GroupState:
    """Replace a group's member list with new_order.  No-op if group unknown."""
    if group_id not in gstate.groups:
        return gstate
    g = {**gstate.groups[group_id], "members": list(new_order)}
    return GroupState(groups={**gstate.groups, group_id: g}, next_id=gstate.next_id)


def delete_group(gstate: GroupState, group_id: str) -> GroupState:
    """Delete a group.  Does NOT delete the member motls from the pool.  No-op if unknown."""
    if group_id not in gstate.groups:
        return gstate
    return GroupState(
        groups={k: v for k, v in gstate.groups.items() if k != group_id},
        next_id=gstate.next_id,
    )


def group_members(gstate: GroupState, group_id: str) -> list[str]:
    """Return the ordered member motl_ids for a group (empty list if unknown)."""
    return list((gstate.groups.get(group_id) or {}).get("members", []))


def purge_motl_from_groups(gstate: GroupState, motl_id: str) -> GroupState:
    """Remove motl_id from every group that contains it.  Called when a motl is deleted."""
    new_groups = {}
    for gid, g in gstate.groups.items():
        members = [m for m in g.get("members", []) if m != motl_id]
        new_groups[gid] = {**g, "members": members}
    return GroupState(groups=new_groups, next_id=gstate.next_id)


def natural_sort_key(s: str) -> list:
    """Key function for natural sorting (run_2 before run_10).  Use in sorted(...)."""
    import re
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


# ── Internal helpers ──────────────────────────────────────────────────────────────

def _to_df(data: pd.DataFrame | list | None) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data) if data else pd.DataFrame()
