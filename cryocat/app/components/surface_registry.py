"""Server-side registry for live :class:`PleomorphicSurface` objects.

Live surfaces (``PleomorphicSurface`` wrapping a ``Mesh`` or
``OrientedPointCloud``) hold open3d-backed buffers and are not
JSON-serializable, so they cannot live in a ``dcc.Store``. The Structure
page instead stores only lightweight **handles** in
``dcc.Store(id="structure-pool")`` --
``{surface_id: {label, representation, n_elements, parent_id, visible}}`` --
and looks the heavy object back up here when it needs to render or operate
on it.

The registry is a module-level dict keyed by ``surface_id``. This matches the
single-process session model that :data:`cryocat.app.logger.dash_logger`
already relies on; spawning workers / hot-reload will reset it, which is the
intended scope.

Public API
----------
* :func:`register_surface` -- store a surface and get back a fresh handle.
* :func:`get_surface` -- look up a surface by id (``None`` if it was evicted).
* :func:`update_surface` -- replace the live object behind an existing id
  (used by in-place ops that swap the wrapped backend).
* :func:`remove_surface` -- drop a surface entirely.
* :func:`clear_registry` -- empty the registry (test fixture).
* :func:`list_surface_ids` -- read-only snapshot of current ids.
"""

from __future__ import annotations

from itertools import count
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid heavy import at module load
    from cryocat.analysis.structure import PleomorphicSurface


# Module-level registry. Single Dash worker process => single dict, no lock.
_REGISTRY: dict[str, "PleomorphicSurface"] = {}
_ID_COUNTER = count(0)


def register_surface(surface: "PleomorphicSurface") -> str:
    """Store ``surface`` and return a fresh, stable ``surface_id``.

    The id is opaque to callers (currently ``"surface-<n>"`` with ``n`` an
    incrementing counter), but it is stable for the lifetime of the running
    process: two consecutive ``register_surface`` calls always yield distinct
    ids.

    Parameters
    ----------
    surface : PleomorphicSurface
        Live surface to register. The registry keeps a reference -- the caller
        keeps the id.

    Returns
    -------
    str
        The fresh ``surface_id``.
    """
    sid = f"surface-{next(_ID_COUNTER)}"
    _REGISTRY[sid] = surface
    return sid


def get_surface(surface_id: str) -> "PleomorphicSurface | None":
    """Return the live surface for ``surface_id``, or ``None`` if not present.

    Callers must handle ``None`` -- after a hot reload or a worker restart the
    registry is empty even if the page's handle store still references the
    id."""
    return _REGISTRY.get(surface_id)


def update_surface(surface_id: str, surface: "PleomorphicSurface") -> None:
    """Replace the live object behind an existing id.

    Used by in-place ops that swap the wrapped backend (e.g.
    ``Mesh.oversample`` returns an :class:`OrientedPointCloud` -- the same
    handle in the page's pool now refers to a point cloud).

    Raises
    ------
    KeyError
        If ``surface_id`` is not registered.
    """
    if surface_id not in _REGISTRY:
        raise KeyError(f"Unknown surface_id: {surface_id!r}")
    _REGISTRY[surface_id] = surface


def remove_surface(surface_id: str) -> None:
    """Drop ``surface_id`` from the registry. No-op if it wasn't there."""
    _REGISTRY.pop(surface_id, None)


def clear_registry() -> None:
    """Empty the registry. Intended for tests / hot-reload safety."""
    _REGISTRY.clear()


def list_surface_ids() -> list[str]:
    """Return a snapshot list of currently-registered ids (stable order)."""
    return list(_REGISTRY)


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

    The page's ``dcc.Store(id="structure-pool")`` is a
    ``{surface_id: handle_dict}`` mapping. This helper assembles one such
    dict for a surface freshly added to the registry.

    Parameters
    ----------
    surface : PleomorphicSurface
        The live surface (used to derive ``representation``, ``n_elements``,
        and ``has_curvatures``).
    label : str
        Human-readable label shown in the surfaces list.
    parent_id : str, optional
        ``surface_id`` of the surface this one was derived from (e.g. inner
        and outer branches keep their parent's id), or ``None`` for top-level
        loads.
    visible : bool, default=True
        Initial visibility flag (the viewer renders visible handles only).

    Returns
    -------
    dict
        Keys: ``"label"``, ``"representation"``, ``"n_elements"``,
        ``"parent_id"``, ``"visible"``, ``"has_curvatures"``. The last is
        always ``False`` for point clouds; for meshes it reflects whether
        :meth:`cryocat.core.surface.Mesh.compute_curvatures` /
        :meth:`Mesh.read_curvatures` has populated the per-vertex fields.
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
    return {
        "label": label,
        "representation": representation,
        "n_elements": int(n_elements),
        "parent_id": parent_id,
        "visible": bool(visible),
        "has_curvatures": bool(has_curvatures),
    }
