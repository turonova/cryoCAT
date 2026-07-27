"""Server-side registry for live :class:`PleomorphicSurface` objects.

Live surfaces hold open3d-backed buffers and are not JSON-serializable, so
they cannot live in a ``dcc.Store``. The Structure page instead stores only
lightweight **handles** in ``dcc.Store(id="structure-pool")`` and looks the
heavy object back up via :data:`registry` when it needs to render or operate.

The registry is a module-level :class:`~cryocat.app.components.registry.Registry`
instance. This matches the single-process session model that
:data:`cryocat.app.logger.dash_logger` already relies on; spawning workers /
hot-reload will reset it, which is the intended scope.

Public API
----------
* :data:`registry` — ``Registry`` instance; call ``.add()``, ``.get()``,
  ``.replace()``, ``.remove()``, ``.clear()`` directly.
* :func:`make_handle` — build the dcc.Store-safe handle dict for a surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from cryocat.app.components.registry import Registry

if TYPE_CHECKING:
    from cryocat.analysis.structure import PleomorphicSurface


registry: Registry["PleomorphicSurface"] = Registry("surface")


@dataclass
class SurfaceHandle:
    """Lightweight handle stored in ``dcc.Store(id="structure-pool")``."""

    label: str
    representation: str
    n_elements: int
    parent_id: str | None
    visible: bool
    has_curvatures: bool


def _mesh_has_curvatures(mesh) -> bool:
    """True iff per-vertex curvature fields are populated on ``mesh``.

    :meth:`Mesh.compute_curvatures` and :meth:`Mesh.read_curvatures` both
    populate the private ``_mean_curvature`` / ``_principal_curvatures``
    attributes; we check the cheaper of the two.
    """
    return getattr(mesh, "_mean_curvature", None) is not None


def make_handle(
    surface: "PleomorphicSurface",
    label: str,
    parent_id: str | None = None,
    visible: bool = True,
) -> dict:
    """Build the lightweight handle the page's pool store carries.

    The schema is defined by :class:`SurfaceHandle`; ``dataclasses.fields``
    is the single source of truth.

    Parameters
    ----------
    surface : PleomorphicSurface
        The live surface (used to derive ``representation``, ``n_elements``,
        and ``has_curvatures``).
    label : str
        Human-readable label shown in the surfaces list.
    parent_id : str, optional
        ``surface_id`` of the surface this one was derived from, or ``None``
        for top-level loads.
    visible : bool, default=True
        Initial visibility flag.
    """
    has_curvatures = False
    if surface.is_mesh:
        representation = "mesh"
        n_elements = len(surface.surface.vertices) if surface.surface.vertices is not None else 0
        has_curvatures = _mesh_has_curvatures(surface.surface)
    elif surface.is_point_cloud:
        representation = "point_cloud"
        n_elements = len(surface.surface.vertices) if surface.surface.vertices is not None else 0
    else:
        representation = "unknown"
        n_elements = 0
    return asdict(SurfaceHandle(
        label=label,
        representation=representation,
        n_elements=int(n_elements),
        parent_id=parent_id,
        visible=bool(visible),
        has_curvatures=bool(has_curvatures),
    ))
