"""Shared alpha-shape tetra cache and rendering helpers.

Keyed by *source_key*, which can be any string identifier — a data-pool motl
id (Utilities tab) or a surface-registry id (Structure tab).  Both call sites
write into the same dict so the expensive Delaunay step is never repeated for
the same source across the two tabs.
"""
from __future__ import annotations

import math
import time

import numpy as np
import plotly.graph_objects as go

from cryocat.app.components.graphsettings import styled_figure, error_figure


# source_key -> (tetra_mesh, pt_map)  — Open3D objects, NOT JSON-serialisable.
alpha_tetra_cache: dict[str, tuple] = {}


def compute_tetra(source_key: str, coords: np.ndarray) -> dict | None:
    """Compute and cache the Delaunay tetrahedralisation for *source_key*.

    Returns a JSON-serialisable dict (suitable for dcc.Store) with the alpha
    range and point count, or None when fewer than 4 points or coplanar.
    The Open3D objects go into ``alpha_tetra_cache`` keyed by *source_key*.
    """
    from cryocat.core.surface import Mesh
    if coords is None or len(coords) < 4:
        return None
    try:
        tetra_mesh, pt_map = Mesh.alpha_shape_tetra(coords)
    except (ValueError, Exception):
        return None
    alpha_tetra_cache[source_key] = (tetra_mesh, pt_map)
    lo, hi = Mesh.suggest_alpha_range(coords)
    return {
        "source_key": source_key,
        "log_min": math.log10(lo),
        "log_max": math.log10(hi),
        "n_points": int(len(coords)),
    }


def slider_to_alpha(slider_val: float, tetra_info: dict) -> float:
    """Map a [0, 1] slider value to a linear alpha via the stored log range."""
    log_min = tetra_info["log_min"]
    log_max = tetra_info["log_max"]
    return 10.0 ** (log_min + float(slider_val) * (log_max - log_min))


def render_alpha_figure(
    alpha: float,
    source_key: str,
    coords: np.ndarray,
    show_pts: bool,
    gs: dict,
) -> tuple[go.Figure, str]:
    """Build the Mesh3d preview figure and a stats string for *alpha*.

    Reads the cached tetra from ``alpha_tetra_cache``; accepts *coords*
    directly rather than fetching from any pool, so it works from any tab.
    """
    from cryocat.core.surface import Mesh

    tetra_mesh, pt_map = alpha_tetra_cache.get(source_key, (None, None))
    if tetra_mesh is None:
        return (
            error_figure("Reselect the source to rebuild the tetrahedralisation."),
            "",
        )
    t0 = time.perf_counter()
    try:
        mesh = Mesh.from_alpha_shape(coords, alpha, tetra_mesh, pt_map)
    except Exception as exc:
        return error_figure(f"Alpha shape error: {exc}"), ""
    elapsed = time.perf_counter() - t0
    n_verts = int(len(mesh.vertices)) if mesh.vertices is not None else 0
    n_faces = int(len(mesh.faces)) if mesh.faces is not None else 0
    traces: list = []
    if n_verts > 0 and n_faces > 0:
        traces.append(go.Mesh3d(
            x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
            opacity=0.7, color="lightblue", name="Alpha shape",
        ))
    if show_pts and coords is not None:
        n = len(coords)
        if n > 30_000:
            step = max(1, n // 30_000)
            pts = coords[::step]
            hint = f" (1:{step} decimated from {n:,})"
        else:
            pts = coords
            hint = ""
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker={"size": 2, "color": "orange", "opacity": 0.5},
            name=f"Input points{hint}",
        ))
    fig = styled_figure(
        go.Figure(traces), gs, uirevision=f"alpha-{source_key}"
    )
    n_comp = mesh.get_connected_component_count() if n_faces > 0 else 0
    watertight = mesh.is_watertight() if n_faces > 0 else False
    stats = (
        f"Vertices: {n_verts:,} · Triangles: {n_faces:,} · "
        f"Components: {n_comp} · Watertight: {'yes' if watertight else 'no'} · "
        f"{elapsed * 1000:.0f} ms"
    )
    return fig, stats
