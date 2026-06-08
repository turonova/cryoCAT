"""Server-side registry for live memthick analysis results.

A loaded membrane carries a :class:`MembraneData` object plus two analysis
result objects (:class:`ThicknessAnalysisResults` and
:class:`IntensityProfileAnalysisResults`) and the raw intensity profile
arrays — all far too large for a ``dcc.Store``. The Analyze section
therefore keeps the heavy payload in this module-level dict and threads
only lightweight **handles** through Dash stores (one per membrane: name,
n_rows, boundary-mode counts, pixel size).

Same single-process / single-worker pattern as
:mod:`cryocat.app.components.surface_registry`.

Public API
----------
* :func:`register_results` — store a ``MembraneResults`` bundle, return the id.
* :func:`get_results`      — look up by id; ``None`` if it was evicted.
* :func:`remove_results`   — drop one membrane.
* :func:`clear_registry`   — empty the registry (test fixture).
* :func:`list_ids`         — read-only snapshot of current ids.
* :func:`make_handle`      — build the lightweight handle the page stores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid heavy imports at module load time
    pass


@dataclass
class MembraneResults:
    """Bundle of one membrane's loaded data + analysis results.

    Attributes
    ----------
    membrane : str
        Human-readable name (matches what the user typed into the form).
    membrane_data : MembraneData
        The :class:`memthick_analyze_plot.MembraneData` instance.
    thickness_results : ThicknessAnalysisResults
        Output of :func:`memthick_analyze_plot.analyze_membrane_thickness`.
    profile_results : IntensityProfileAnalysisResults
        Output of :func:`memthick_analyze_plot.analyze_intensity_profiles`.
    boundary_info : dict
        Output of :func:`memthick_analyze_plot.return_boundary_info`.
    thickness_csv : str
        Path the membrane was loaded from (used by motl export).
    pixel_size_nm : float | None
        Resolved pixel size (None if it couldn't be determined).
    extra : dict
        Free-form bag for callers — not used by the registry itself.
    """

    membrane: str
    membrane_data: Any
    thickness_results: Any
    profile_results: Any
    boundary_info: dict
    thickness_csv: str
    pixel_size_nm: float | None = None
    extra: dict = field(default_factory=dict)


_REGISTRY: dict[str, MembraneResults] = {}
_ID_COUNTER = count(0)


def register_results(results: MembraneResults) -> str:
    """Store ``results`` and return a fresh, stable id."""
    rid = f"memthick-{next(_ID_COUNTER)}"
    _REGISTRY[rid] = results
    return rid


def get_results(rid: str) -> MembraneResults | None:
    """Return the bundle for ``rid``, or ``None`` if not present."""
    return _REGISTRY.get(rid)


def remove_results(rid: str) -> None:
    """Drop ``rid`` from the registry. No-op if it wasn't there."""
    _REGISTRY.pop(rid, None)


def clear_registry() -> None:
    """Empty the registry. Intended for tests / hot-reload safety."""
    _REGISTRY.clear()


def list_ids() -> list[str]:
    """Return a snapshot list of currently-registered ids (stable order)."""
    return list(_REGISTRY)


def make_handle(results: MembraneResults) -> dict:
    """Build the dcc.Store-safe handle for a registered bundle.

    The handle is JSON-serialisable; it carries only the fields the
    sidebar / dropdowns / status text need. The page round-trips it
    through ``dcc.Store(id="memthick-results-handles")`` and looks the
    heavy object back up here via :func:`get_results`.

    Parameters
    ----------
    results : MembraneResults
        The bundle whose handle to build.

    Returns
    -------
    dict
        Keys: ``"membrane"``, ``"n_rows"``, ``"n_resolved"``,
        ``"n_unresolved"``, ``"by_detection_mode"``, ``"pixel_size_nm"``,
        ``"thickness_csv"``.
    """
    info = results.boundary_info or {}
    md = results.membrane_data
    try:
        n_rows = int(len(md.thickness_df))
    except Exception:
        n_rows = 0
    return {
        "membrane": results.membrane,
        "n_rows": n_rows,
        "n_resolved": int(info.get("n_resolved", 0)),
        "n_unresolved": int(info.get("n_unresolved", 0)),
        "n_finite_inflection_thickness_nm": int(info.get("n_finite_inflection_thickness_nm", 0)),
        "by_detection_mode": {
            str(k): int(v) for k, v in (info.get("by_detection_mode") or {}).items()
        },
        "pixel_size_nm": (
            float(results.pixel_size_nm) if results.pixel_size_nm is not None else None
        ),
        "thickness_csv": str(results.thickness_csv),
    }
