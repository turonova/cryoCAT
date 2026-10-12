"""Synthetic data builders for GUI app tests.

All builders are tiny by design — 10 rows, 16³ volumes.  Any test that
needs a larger fixture is doing real work and should be rethought.
"""
from __future__ import annotations

import itertools
import types
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from cryocat.core.cryomotl import Motl
    from cryocat.app.pool import PoolState


# ── Motl rows ────────────────────────────────────────────────────────────────

_MOTL_COLS = [
    "score", "geom1", "geom2", "subtomo_id", "tomo_id", "object_id",
    "subtomo_mean", "x", "y", "z",
    "shift_x", "shift_y", "shift_z",
    "geom3", "geom4", "geom5",
    "phi", "psi", "theta",
    "class",
]


def motl_rows(
    n: int = 10,
    *,
    tomo_ids: tuple[int, ...] = (1, 2),
    with_score: bool = True,
) -> list[dict]:
    """Pool-shaped rows carrying the standard motl columns."""
    rng = np.random.default_rng(42)
    tid_cycle = itertools.cycle(tomo_ids)
    rows = []
    for i in range(n):
        row = {c: 0.0 for c in _MOTL_COLS}
        row["subtomo_id"] = float(i + 1)
        row["tomo_id"] = float(next(tid_cycle))
        row["x"] = float(rng.integers(1, 100))
        row["y"] = float(rng.integers(1, 100))
        row["z"] = float(rng.integers(1, 100))
        row["phi"] = float(rng.uniform(0, 360))
        row["psi"] = float(rng.uniform(0, 360))
        row["theta"] = float(rng.uniform(0, 180))
        if with_score:
            row["score"] = float(rng.uniform(0, 1))
        rows.append(row)
    return rows


def motl(n: int = 10, **kw) -> "Motl":
    """A live Motl object with *n* rows."""
    from cryocat.core.cryomotl import Motl
    return Motl(pd.DataFrame(motl_rows(n, **kw)))


def pool_state(n_motls: int = 2) -> "PoolState":
    """A PoolState with *n_motls* entries."""
    from cryocat.app.pool import PoolState, insert_motl
    state = PoolState(registry={}, motls={}, extra={}, meta={}, next_id=0)
    for i in range(n_motls):
        rows = motl_rows(10, tomo_ids=(i + 1,))
        state = insert_motl(state, rows, label=f"Motl {i + 1}", motl_type="emmotl")
    return state


# ── Event streams ─────────────────────────────────────────────────────────────

def event_stream(*, n_ok: int = 2, n_error: int = 1) -> list[dict]:
    """A valid event stream list for projection tests."""
    from cryocat.app.event import call_event

    events: list[dict] = []
    seq = 0

    fn = "cryocat.core.cryomotl.Motl.clean_by_tomo"
    for _ in range(n_ok):
        ev = call_event(
            fn, {"tomo_ids": repr([1, 2])},
            status="ok",
            imports=[["cryomotl", "from cryocat.core import cryomotl"]],
            receiver="motl_0",
            assign_to=None,
            duration_s=0.01,
            result={"type": "Motl", "n_rows": 5},
        )
        ev["seq"] = seq
        seq += 1
        events.append(ev)

    for _ in range(n_error):
        ev = call_event(
            fn, {"tomo_ids": repr([99])},
            status="error",
            imports=[["cryomotl", "from cryocat.core import cryomotl"]],
            receiver="motl_0",
            assign_to=None,
            duration_s=0.001,
            error={"type": "KeyError", "msg": "tomo 99 not found", "traceback": ""},
        )
        ev["seq"] = seq
        seq += 1
        events.append(ev)

    return events


# ── NumPy volumes ─────────────────────────────────────────────────────────────

def volume(shape: tuple[int, ...] = (16, 16, 16), *, blob: bool = True) -> np.ndarray:
    """A small float32 volume, optionally with a Gaussian blob at the centre."""
    arr = np.zeros(shape, dtype=np.float32)
    if blob:
        c = [s // 2 for s in shape]
        grid = np.indices(shape)
        r2 = sum((grid[i] - c[i]) ** 2 for i in range(len(shape)))
        arr = np.exp(-r2 / (2 * (min(shape) // 4) ** 2)).astype(np.float32)
    return arr
