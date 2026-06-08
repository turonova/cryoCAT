"""Server-side **active fit** slot for :class:`ParametricSurface`.

Phase 3 of the Structure page holds at most one fitted parametric surface
at a time. The fit is live state (a :class:`QuadricsM` with open3d/numpy
buffers) so it lives in this server-side slot, while the page's
``dcc.Store(id="parametric-active")`` carries only a tiny handle dict.

This mirrors :mod:`cryocat.app.components.surface_registry` (Phase 1) but
with a single slot instead of an id-keyed dict -- the workflow assumes one
active fit, swapped wholesale when the user fits / loads a new one.

Public API
----------
* :func:`set_active_fit` -- store a fit and return its handle.
* :func:`get_active_fit` -- look up the live :class:`ParametricSurface`
  (``None`` when no fit is active).
* :func:`clear_active_fit` -- drop the slot (e.g. for tests).
* :func:`make_handle` -- build the handle dict the page store carries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryocat.analysis.structure import ParametricSurface


# Module-level singleton slot. Single Dash worker => no lock needed.
_ACTIVE: dict = {"surface": None}


def set_active_fit(psurf: "ParametricSurface", source: str) -> dict:
    """Install ``psurf`` as the active fit and return its handle dict.

    Parameters
    ----------
    psurf : ParametricSurface
        The fitted parametric surface to store. Replaces any previously
        active fit.
    source : str
        Human-readable provenance string (e.g. ``"motl:my_motl"`` or
        ``"csv:/path/to/file.csv"``); shown in the handle so the user can
        tell at a glance how this fit was produced.

    Returns
    -------
    dict
        The handle dict the page's ``parametric-active`` store should
        carry. See :func:`make_handle` for the schema.
    """
    _ACTIVE["surface"] = psurf
    return make_handle(psurf, source=source)


def get_active_fit() -> "ParametricSurface | None":
    """Return the active fit, or ``None`` when no fit is installed."""
    return _ACTIVE["surface"]


def clear_active_fit() -> None:
    """Drop the active fit slot. Intended for tests / hot-reload safety."""
    _ACTIVE["surface"] = None


def make_handle(psurf: "ParametricSurface", source: str) -> dict:
    """Build the lightweight handle the page's ``parametric-active`` store carries.

    Parameters
    ----------
    psurf : ParametricSurface
        The live fit (used to derive ``column_name``, the count of fitted
        quadrics, and the surface type label).
    source : str
        Provenance string forwarded verbatim onto the handle.

    Returns
    -------
    dict
        Keys: ``"column_name"`` (the grouping column the fit was built on),
        ``"surface_type"`` (currently always ``"ellipsoid"``),
        ``"n_quadrics"`` (number of fitted surfaces, i.e.
        ``len(psurf.quadrics.dict)``), ``"source"``.
    """
    qd = getattr(psurf.quadrics, "dict", {}) or {}
    return {
        "column_name": str(getattr(psurf, "column_name", "object_id")),
        "surface_type": "ellipsoid",
        "n_quadrics": int(len(qd)),
        "source": str(source),
    }
