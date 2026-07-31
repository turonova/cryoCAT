"""Provenance table keyed by pool entry id.

Replaces the unsound ``id()``-based ``StreamToList._motl_sources`` (A5).
Each pool entry owns a stable string id (``motl-3``); provenance is keyed
by that id so it survives serialisation across callbacks and is never
invalidated by CPython's id-reuse after GC.

Call :func:`record` when a pool entry is first produced (P6:
``run_operation_to_pool``), :func:`forget` when it is removed from the pool,
and :func:`clear` at session end.

Public API
----------
* :func:`bind`     — canonical script variable name for a pool entry.
* :func:`record`   — note which event (by ``seq``) produced an entry.
* :func:`producer` — seq of the producing event, or ``None``.
* :func:`var_for`  — variable name for a recorded entry, or ``None``.
* :func:`forget`   — remove a single entry (pool deletion).
* :func:`clear`    — reset all state.
"""
from __future__ import annotations

# motl_id  →  seq of the event that produced it
_producers: dict[str, int] = {}

# counter for descriptor variable names ("desc-0", "desc-1", …)
_desc_counter: int = 0


def next_desc_id() -> str:
    """Allocate the next unique descriptor id (``'desc-0'``, ``'desc-1'``, …)."""
    global _desc_counter
    desc_id = f"desc-{_desc_counter}"
    _desc_counter += 1
    return desc_id


def bind(motl_id: str) -> str:
    """Return the canonical script variable name for a pool entry.

    The mapping is a pure function: ``'motl-3'`` → ``'motl_3'``.
    Deterministic and injective — distinct ids always produce distinct names,
    and the same id always produces the same name.
    """
    return motl_id.replace("-", "_")


def record(motl_id: str, seq: int) -> None:
    """Note that pool entry *motl_id* was produced by the event at *seq*.

    Overwrites any prior record for the same id (a pool entry that was
    replaced in-place by a new operation still has one producer at a time).
    """
    _producers[motl_id] = seq


def producer(motl_id: str) -> int | None:
    """Return the seq of the event that produced *motl_id*, or ``None``."""
    return _producers.get(motl_id)


def var_for(motl_id: str) -> str | None:
    """Return the script variable name for a recorded entry, or ``None``.

    Returns ``None`` if *motl_id* has never been passed to :func:`record` —
    meaning its producer is unknown and it cannot be referenced in a script.
    """
    if motl_id not in _producers:
        return None
    return bind(motl_id)


def forget(motl_id: str) -> None:
    """Remove *motl_id* from the provenance table (called on pool deletion)."""
    _producers.pop(motl_id, None)


def clear() -> None:
    """Reset all provenance state (called on session close or test teardown)."""
    global _desc_counter
    _producers.clear()
    _desc_counter = 0
