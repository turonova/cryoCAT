"""Pool state model and pure reducers.

The five pool stores (`pool-registry`, `pool-motls`, `pool-extra`, `pool-meta`,
`pool-next-id`) form one logical unit. No module outside this file may hand-build
their contents or reference their id strings (use :mod:`cryocat.app.ids`).

§5 guarantees:
- `motl_id` is always ``f"motl-{next_id}"``, counter only increases, ids are
  never reused or renumbered after removal.
- :func:`insert_motl` writes **all four data stores**. Passing no `extra`/`meta`
  stores an explicit ``None``, so a consumer can rely on the key being present.
- Registry values are ``dataclasses.asdict(PoolEntry(...))``, never hand-built.
- :meth:`PoolState.to_stores` output is JSON-round-trip safe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PoolEntry:
    label: str
    type: str
    n_rows: int
    active: bool = True


@dataclass(frozen=True)
class PoolState:
    """Immutable snapshot of the five pool stores."""

    registry: dict
    motls: dict
    extra: dict
    meta: dict
    next_id: int

    @classmethod
    def from_stores(cls, registry, motls, extra, meta, next_id) -> PoolState:
        return cls(
            registry=dict(registry or {}),
            motls=dict(motls or {}),
            extra=dict(extra or {}),
            meta=dict(meta or {}),
            next_id=int(next_id or 0),
        )

    def to_stores(self) -> tuple:
        return self.registry, self.motls, self.extra, self.meta, self.next_id


def default_label(next_id: int) -> str:
    return f"Motl {next_id + 1}"


def insert_motl(
    state: PoolState,
    rows: list,
    *,
    label: str | None = None,
    motl_type: str = "emmotl",
    extra: list | None = None,
    meta: dict | None = None,
) -> tuple[PoolState, str]:
    """Append one motl; returns the new state and the fresh ``motl_id``."""
    motl_id = f"motl-{state.next_id + 1}"
    entry = PoolEntry(
        label=label or default_label(state.next_id),
        type=motl_type,
        n_rows=len(rows) if rows is not None else 0,
        active=True,
    )
    return PoolState(
        registry={**state.registry, motl_id: asdict(entry)},
        motls={**state.motls, motl_id: rows},
        extra={**state.extra, motl_id: extra},
        meta={**state.meta, motl_id: meta},
        next_id=state.next_id + 1,
    ), motl_id


def remove_motl(state: PoolState, motl_id: str) -> PoolState:
    """Remove a motl from all stores. No-op if the id is unknown."""
    if motl_id not in state.registry:
        return state
    return PoolState(
        registry={k: v for k, v in state.registry.items() if k != motl_id},
        motls={k: v for k, v in state.motls.items() if k != motl_id},
        extra={k: v for k, v in state.extra.items() if k != motl_id},
        meta={k: v for k, v in state.meta.items() if k != motl_id},
        next_id=state.next_id,
    )


def set_active(state: PoolState, motl_id: str, active: bool) -> PoolState:
    """Toggle the active flag for one entry. No-op if the id is unknown."""
    if motl_id not in state.registry:
        return state
    return PoolState(
        registry={**state.registry, motl_id: {**state.registry[motl_id], "active": active}},
        motls=state.motls,
        extra=state.extra,
        meta=state.meta,
        next_id=state.next_id,
    )


def replace_motl_rows(
    state: PoolState,
    motl_id: str,
    rows: list,
    *,
    label: str | None = None,
    motl_type: str | None = None,
) -> PoolState:
    """Update the rows (and optionally label / type) for an existing entry.

    A no-op if *motl_id* is not in the registry.  ``next_id`` is unchanged
    (the entry keeps its original counter slot).
    """
    if motl_id not in state.registry:
        return state
    entry = dict(state.registry[motl_id])
    entry["n_rows"] = len(rows) if rows is not None else 0
    if label is not None:
        entry["label"] = label
    if motl_type is not None:
        entry["type"] = motl_type
    return PoolState(
        registry={**state.registry, motl_id: entry},
        motls={**state.motls, motl_id: rows},
        extra=state.extra,
        meta=state.meta,
        next_id=state.next_id,
    )


def get_rows(state: PoolState, motl_id: str) -> list:
    """Return the row list for ``motl_id``, or an empty list if absent."""
    return state.motls.get(motl_id) or []


def active_ids(state: PoolState) -> list[str]:
    """Return motl ids whose active flag is True, in insertion order."""
    return [mid for mid, entry in state.registry.items() if entry.get("active", True)]
