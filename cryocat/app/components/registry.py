"""Generic server-side object registry for live, non-JSON-serializable data.

Single-process only — matches the same constraint as
:data:`cryocat.app.logger.dash_logger`. Keys are stable for the process
lifetime and are never reused (counter never decrements).
"""
from __future__ import annotations

from itertools import count


class Registry[T]:
    """Server-side store for live objects too heavy for a dcc.Store.

    Parameters
    ----------
    prefix : str
        Key prefix; generated keys are ``f"{prefix}-{n}"``.
    max_items : int | None
        When set, the registry holds at most this many items. Adding beyond
        the limit evicts the oldest entry first (FIFO).
    """

    def __init__(self, prefix: str, *, max_items: int | None = None, start: int = 0) -> None:
        self._prefix = prefix
        self._max_items = max_items
        self._store: dict[str, T] = {}
        self._counter = count(start)

    def add(self, obj: T) -> str:
        """Store *obj* and return a fresh, stable key."""
        if self._max_items is not None:
            while len(self._store) >= self._max_items:
                oldest = next(iter(self._store))
                del self._store[oldest]
        key = f"{self._prefix}-{next(self._counter)}"
        self._store[key] = obj
        return key

    def get(self, key: str) -> T | None:
        """Return the object for *key*, or ``None`` if absent."""
        return self._store.get(key)

    def replace(self, key: str, obj: T) -> None:
        """Replace the object behind an existing *key*.

        Raises
        ------
        KeyError
            If *key* is not registered.
        """
        if key not in self._store:
            raise KeyError(key)
        self._store[key] = obj

    def remove(self, key: str) -> None:
        """Drop *key* from the registry. No-op if absent."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Empty the registry (intended for tests / hot-reload)."""
        self._store.clear()

    def keys(self) -> list[str]:
        """Return a snapshot list of current keys in insertion order."""
        return list(self._store)

    def __len__(self) -> int:
        return len(self._store)
