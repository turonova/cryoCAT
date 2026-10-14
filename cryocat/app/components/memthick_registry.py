"""Server-side registry for live memthick analysis results.

A loaded membrane carries a :class:`MembraneData` object plus two analysis
result objects (:class:`ThicknessAnalysisResults` and
:class:`IntensityProfileAnalysisResults`) and the raw intensity profile
arrays — all far too large for a ``dcc.Store``. The Analyze section therefore
keeps the heavy payload in this module-level registry and threads only
lightweight **handles** through Dash stores (one per membrane: name, n_rows,
boundary-mode counts, pixel size).

Same single-process / single-worker pattern as
:mod:`cryocat.app.components.surface_registry`.

Public API
----------
* :data:`registry` — ``Registry`` instance; use ``.add()``, ``.get()``,
  ``.remove()``, ``.clear()`` directly.
* :func:`make_handle` — build the lightweight handle the page stores.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cryocat.app.components.registry import Registry


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


registry: Registry[MembraneResults] = Registry("memthick")


@dataclass
class MembraneHandle:
    """Lightweight handle stored in ``dcc.Store(id="memthick-results-handles")``.

    The schema is defined here; ``dataclasses.fields`` is the single source of
    truth — docs cannot drift from the code.
    """

    membrane: str
    n_rows: int
    n_resolved: int
    n_unresolved: int
    n_finite_inflection_thickness_nm: int
    by_detection_mode: dict
    pixel_size_nm: float | None
    thickness_csv: str


def make_handle(results: MembraneResults) -> dict:
    """Build the dcc.Store-safe handle for a registered bundle.

    The handle is JSON-serialisable; it carries only the fields the
    sidebar / dropdowns / status text need. The page round-trips it through
    ``dcc.Store(id="memthick-results-handles")`` and looks the heavy object
    back up via :data:`registry`.

    The exact key set is defined by :class:`MembraneHandle`.
    """
    info = results.boundary_info or {}
    md = results.membrane_data
    try:
        n_rows = int(len(md.thickness_df))
    except Exception:
        n_rows = 0
    return asdict(MembraneHandle(
        membrane=results.membrane,
        n_rows=n_rows,
        n_resolved=int(info.get("n_resolved", 0)),
        n_unresolved=int(info.get("n_unresolved", 0)),
        n_finite_inflection_thickness_nm=int(info.get("n_finite_inflection_thickness_nm", 0)),
        by_detection_mode={
            str(k): int(v) for k, v in (info.get("by_detection_mode") or {}).items()
        },
        pixel_size_nm=(
            float(results.pixel_size_nm) if results.pixel_size_nm is not None else None
        ),
        thickness_csv=str(results.thickness_csv),
    ))
