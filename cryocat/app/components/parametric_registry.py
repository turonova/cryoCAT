"""Server-side **active fit** slot for :class:`ParametricSurface`.

Phase 3 of the Structure page holds at most one fitted parametric surface at
a time. The fit is live state (a :class:`QuadricsM` with open3d/numpy buffers)
so it lives in this server-side slot, while the page's
``dcc.Store(id="parametric-active")`` carries only a lightweight handle dict.

This mirrors :mod:`cryocat.app.components.surface_registry` (Phase 1) but
with a single slot instead of an id-keyed dict — configured via
``max_items=1`` on the :class:`~cryocat.app.components.registry.Registry`.

Public API
----------
* :data:`registry` — ``Registry(max_items=1)`` instance; use ``.add()`` to
  store a new fit (auto-evicts the old one), ``.keys()`` + ``.get()`` to
  retrieve it, ``.clear()`` for tests.
* :func:`make_handle` — build the handle dict the page store carries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from cryocat.app.components.registry import Registry

if TYPE_CHECKING:
    from cryocat.analysis.structure import ParametricSurface


registry: Registry["ParametricSurface"] = Registry("parametric", max_items=1)


@dataclass
class ParametricHandle:
    """Lightweight handle stored in ``dcc.Store(id="parametric-active")``."""

    column_name: str
    surface_type: str
    n_quadrics: int
    source: str


def make_handle(psurf: "ParametricSurface", source: str) -> dict:
    """Build the lightweight handle the page's ``parametric-active`` store carries.

    The schema is defined by :class:`ParametricHandle`; ``dataclasses.fields``
    is the single source of truth.

    Parameters
    ----------
    psurf : ParametricSurface
        The live fit (used to derive ``column_name``, the count of fitted
        quadrics, and the surface type label).
    source : str
        Provenance string forwarded verbatim onto the handle.
    """
    qd = getattr(psurf.quadrics, "dict", {}) or {}
    return asdict(ParametricHandle(
        column_name=str(getattr(psurf, "column_name", "object_id")),
        surface_type="ellipsoid",
        n_quadrics=int(len(qd)),
        source=str(source),
    ))
