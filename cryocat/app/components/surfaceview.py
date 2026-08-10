"""Shared 3D surface viewer (meshes + oriented point clouds).

The viewer takes a *handles* store (id chosen by the embedding page) keyed by
``surface_id``, looks each live surface up in
:mod:`cryocat.app.components.surface_registry`, and renders the visible ones
in a single 3-D figure: meshes as :class:`plotly.graph_objects.Mesh3d` from
``vertices``/``faces``, point clouds as :class:`plotly.graph_objects.Scatter3d`
plus :class:`plotly.graph_objects.Cone` arrows for the normals (the trace set
that :func:`cryocat.analysis.visplot.plot_points_with_normals` would have
produced — we extract them and add them to the combined figure).

Every figure is passed through
:func:`cryocat.app.components.graphsettings.apply_settings_to_figure` so the
viewer honours the suite's global graph-settings store.

Contract
--------
* :func:`get_surface_view(prefix)` -- layout fragment: a single ``dcc.Graph``
  with id ``f"{prefix}-graph"``.
* :func:`register_surface_view_callbacks(app, prefix, pool_store_id, *,
  selected_store_id=None)` -- wires the redraw callback.

The embedding page owns the handles store (``pool_store_id``) and updates it
whenever a surface is added / removed / has its visibility toggled. The
viewer reacts to *that* store plus ``graph-settings-store``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State

from cryocat.app import ids
from cryocat.app.components.surface_registry import registry as _surface_registry
from cryocat.app.components.graphsettings import styled_figure
from cryocat.analysis.visplot import resolve_palette as _resolve_palette
from cryocat.app.formgen import make_dropdown


# Curvature color-by options. Values match
# :func:`cryocat.analysis.visplot.plot_vtp_mesh` for cross-tool parity; the
# trace builder maps each value to the matching PleomorphicSurface getter.
COLOR_BY_OPTIONS = [
    {"label": "No color-by", "value": "none"},
    {"label": "Mean curvature", "value": "mean_curvature"},
    {"label": "Gaussian curvature", "value": "gaussian_curvature"},
    {"label": "k1 (principal #1)", "value": "k1"},
    {"label": "k2 (principal #2)", "value": "k2"},
    {"label": "Curvature anisotropy", "value": "curvature_anisotropy"},
]


def _vertex_field(psurf, color_by: str) -> np.ndarray | None:
    """Pull a per-vertex curvature field off a Mesh-backed PleomorphicSurface.

    Returns ``None`` when the surface has no curvatures populated, or the
    requested ``color_by`` is ``"none"`` / unrecognised. Field naming matches
    :func:`cryocat.analysis.visplot.plot_vtp_mesh(color_by=)` so the two
    code paths render the same fields.

    Parameters
    ----------
    psurf : PleomorphicSurface
        Source surface; must wrap a :class:`cryocat.core.surface.Mesh` for
        any non-``"none"`` field to be returned.
    color_by : str
        One of the values in :data:`COLOR_BY_OPTIONS`.

    Returns
    -------
    numpy.ndarray or None
        Shape ``(N,)`` per-vertex field, or ``None`` when unavailable.
    """
    if color_by in (None, "none") or not psurf.is_mesh:
        return None
    if not _registry._mesh_has_curvatures(psurf.surface):
        return None
    if color_by == "mean_curvature":
        return psurf.get_mean_curvature()
    if color_by == "gaussian_curvature":
        return psurf.get_gaussian_curvature()
    if color_by in ("k1", "k2"):
        pk = psurf.get_principal_curvatures()
        return pk[:, 0] if color_by == "k1" else pk[:, 1]
    if color_by == "curvature_anisotropy":
        pk = psurf.get_principal_curvatures()
        k1, k2 = pk[:, 0], pk[:, 1]
        denom = np.abs(k1) + np.abs(k2) + 1e-12
        return np.abs(k1 - k2) / denom
    return None


# ── Trace builders ────────────────────────────────────────────────────────────

def _mesh_traces(
    surface, color: str, name: str, selected: bool,
    *, intensity: np.ndarray | None = None, colorscale: str = "RdBu_r",
) -> list:
    """Build the Plotly traces for a Mesh-backed :class:`PleomorphicSurface`.

    Parameters
    ----------
    surface : PleomorphicSurface
        Must wrap a :class:`cryocat.core.surface.Mesh`.
    color : str
        Hex / named color applied uniformly to the mesh surface when no
        ``intensity`` is provided.
    name : str
        Trace name (shown in legend / hover labels).
    selected : bool
        When True, render the mesh slightly more opaque to mark it as selected.
    intensity : numpy.ndarray, optional
        Per-vertex scalar field for color-by. Shape ``(N,)`` matching the
        mesh's vertex count; renders via ``go.Mesh3d(intensity=...,
        intensitymode="vertex", colorscale=...)``. Falls back to the flat
        ``color`` when None or shape-mismatched.
    colorscale : str, default="RdBu_r"
        Plotly colorscale name used when ``intensity`` is provided.
    """
    mesh = surface.surface
    if mesh.vertices is None or mesh.faces is None:
        return []
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    if v.size == 0 or f.size == 0:
        return []

    kw: dict = dict(
        x=v[:, 0].tolist(), y=v[:, 1].tolist(), z=v[:, 2].tolist(),
        i=f[:, 0].tolist(), j=f[:, 1].tolist(), k=f[:, 2].tolist(),
        opacity=0.85 if selected else 0.55,
        name=name,
        hoverinfo="name",
        flatshading=True,
    )
    if intensity is not None and len(intensity) == v.shape[0]:
        finite = np.isfinite(intensity)
        if finite.any():
            vmin = float(np.percentile(intensity[finite], 2))
            vmax = float(np.percentile(intensity[finite], 98))
            if vmin == vmax:
                vmax = vmin + 1.0
            kw.update(
                intensity=intensity.tolist(),
                intensitymode="vertex",
                colorscale=colorscale,
                cmin=vmin, cmax=vmax,
                showscale=True,
                colorbar=dict(title=name, thickness=12),
            )
        else:
            kw["color"] = color
    else:
        kw["color"] = color

    return [go.Mesh3d(**kw)]


def _point_cloud_traces(
    surface, color: str, name: str, selected: bool,
    *, show_normals: bool = True, normal_scale: float = 5.0,
    max_normal_arrows: int = 2000, marker_size: int = 3,
) -> list:
    """Build the Plotly traces for an :class:`OrientedPointCloud`-backed surface.

    Always emits a :class:`plotly.graph_objects.Scatter3d` of the points. When
    ``show_normals`` is True and normals are available, additionally emits a
    :class:`plotly.graph_objects.Cone` trace (sub-sampled to at most
    ``max_normal_arrows`` arrows so dense clouds stay responsive).
    """
    opc = surface.surface
    if opc.vertices is None:
        return []
    pts = np.asarray(opc.vertices)
    if pts.size == 0:
        return []

    traces = [
        go.Scatter3d(
            x=pts[:, 0].tolist(), y=pts[:, 1].tolist(), z=pts[:, 2].tolist(),
            mode="markers",
            marker=dict(size=marker_size, color=color, opacity=0.95 if selected else 0.75),
            name=name,
            hoverinfo="name",
        )
    ]

    if show_normals and opc.normals is not None:
        nrm = np.asarray(opc.normals)
        if nrm.shape == pts.shape and nrm.size > 0:
            # Down-sample dense clouds so the cone trace stays cheap.
            n = pts.shape[0]
            if n > max_normal_arrows:
                idx = np.linspace(0, n - 1, max_normal_arrows).astype(int)
                p_s, n_s = pts[idx], nrm[idx]
            else:
                p_s, n_s = pts, nrm
            traces.append(
                go.Cone(
                    x=p_s[:, 0].tolist(), y=p_s[:, 1].tolist(), z=p_s[:, 2].tolist(),
                    u=(n_s[:, 0] * normal_scale).tolist(),
                    v=(n_s[:, 1] * normal_scale).tolist(),
                    w=(n_s[:, 2] * normal_scale).tolist(),
                    showscale=False,
                    colorscale=[[0, color], [1, color]],
                    sizemode="absolute",
                    sizeref=normal_scale,
                    anchor="tail",
                    name=f"{name} normals",
                    hoverinfo="skip",
                )
            )

    return traces


def _build_figure(
    handles: dict | None,
    selected_id: str | None,
    gs: dict | None,
    color_by: str | None = None,
) -> go.Figure:
    """Assemble the combined figure for every visible handle in ``handles``.

    Parameters
    ----------
    handles : dict
        ``{surface_id: handle_dict}`` from the page's pool store. The
        ``visible`` flag gates rendering; missing surfaces (e.g. registry was
        cleared) are silently skipped.
    selected_id : str, optional
        ``surface_id`` of the currently-selected surface (rendered more
        prominently).
    gs : dict, optional
        Contents of ``graph-settings-store`` to feed to
        :func:`apply_settings_to_figure`.
    color_by : str, optional
        Per-vertex curvature field name to color the *selected* mesh by; one
        of the values in :data:`COLOR_BY_OPTIONS` (defaults to ``"none"`` /
        no color-by). Applies only to the selected surface, and only when it
        is a mesh with curvatures populated. Other surfaces continue to use
        flat palette colors.
    """
    # W3: early-return an empty figure when nothing is visible.
    # Trigger: pool_store_id fires at page mount (prevent_initial_call=False) with
    # surfaces-pool={} — the structure page has no surfaces during a motl load.
    # Without this guard, styled_figure + palette resolution runs for 125 ms on
    # every load even though there is nothing to draw.
    handles = handles or {}
    if not any(h.get("visible", True) for h in handles.values()):
        return go.Figure()

    palette = _resolve_palette((gs or {}).get("discrete_palette"))
    traces: list = []
    for i, (sid, h) in enumerate(handles.items()):
        if not h.get("visible", True):
            continue
        psurf = _surface_registry.get(sid)
        if psurf is None:
            continue
        color = palette[i % len(palette)]
        label = h.get("label", sid)
        is_sel = (selected_id is not None and sid == selected_id)
        rep = h.get("representation")
        if rep == "mesh":
            # Color-by only applies to the selected mesh with curvatures.
            intensity = None
            if is_sel and color_by and color_by != "none" and h.get("has_curvatures"):
                intensity = _vertex_field(psurf, color_by)
            traces.extend(_mesh_traces(psurf, color, label, is_sel, intensity=intensity))
        elif rep == "point_cloud":
            traces.extend(_point_cloud_traces(psurf, color, label, is_sel))
        # Unknown representations are skipped (handle is informational only).

    fig = go.Figure(data=traces)
    return styled_figure(
        fig, gs or {},
        uirevision="surface-view",
        height=620,
        margin={"t": 0, "b": 0, "l": 0, "r": 0},
        scene={"xaxis": {"title": "x"}, "yaxis": {"title": "y"}, "zaxis": {"title": "z"}, "aspectmode": "data"},
    )


# ── Public API ────────────────────────────────────────────────────────────────

def get_surface_view(prefix: str):
    """Layout fragment for the surface viewer.

    Renders a ``Color by`` dropdown above the 3D graph. The dropdown only
    has effect when the currently-selected handle is a mesh with curvatures
    populated (``handle["has_curvatures"] is True``); otherwise it is a
    no-op and the mesh renders with its flat palette color.

    Parameters
    ----------
    prefix : str
        Used to namespace the graph id (``f"{prefix}-graph"``) and the
        color-by selector (``f"{prefix}-color-by"``).

    Returns
    -------
    dash.html.Div
        A ``Div`` wrapping the color-by selector + a single ``dcc.Graph``.
        Embed it in the page's main column.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Label(
                        "Color by",
                        style={"marginRight": "0.5rem"},
                    ),
                    make_dropdown(
                        f"{prefix}-color-by",
                        COLOR_BY_OPTIONS,
                        "none",
                        clearable=False,
                        style={"width": "260px"},
                    ),
                ],
                style={**{"display": "flex", "alignItems": "center", "gap": "0.5rem"}, "marginBottom": "0.4rem"},
            ),
            dcc.Graph(
                id={"type": "styled-graph", "owner": prefix, "name": "graph"},
                style={"height": "620px"},
                config={"scrollZoom": True},
            ),
        ]
    )


def register_surface_view_callbacks(
    app,
    prefix: str,
    pool_store_id: str,
    *,
    selected_store_id: str | None = None,
):
    """Register the redraw callback.

    The viewer reacts to changes in the page's handles store
    (``pool_store_id``), the optional selected-id store, and
    ``graph-settings-store``. It pulls live surfaces out of
    :mod:`cryocat.app.components.surface_registry` and draws every handle whose
    ``visible`` flag is True.

    Parameters
    ----------
    app : dash.Dash
        The Dash app to register against.
    prefix : str
        Same prefix used in :func:`get_surface_view`.
    pool_store_id : str
        Id of the ``dcc.Store`` that holds ``{surface_id: handle_dict}`` for
        the page; the page is responsible for keeping it in sync with the
        registry.
    selected_store_id : str, optional
        Id of an optional ``dcc.Store`` carrying the selected ``surface_id``
        (a scalar string). When provided, the selected surface renders more
        prominently.
    """

    color_by_id = f"{prefix}-color-by"

    if selected_store_id is None:
        @app.callback(
            Output({"type": "styled-graph", "owner": prefix, "name": "graph"}, "figure"),
            Input(pool_store_id, "data"),
            Input(color_by_id, "value"),
            State(ids.GRAPH_SETTINGS_STORE, "data"),
            prevent_initial_call=False,
        )
        def _draw(handles, color_by, gs):
            return _build_figure(handles, None, gs, color_by=color_by)
    else:
        @app.callback(
            Output({"type": "styled-graph", "owner": prefix, "name": "graph"}, "figure"),
            Input(pool_store_id, "data"),
            Input(selected_store_id, "data"),
            Input(color_by_id, "value"),
            State(ids.GRAPH_SETTINGS_STORE, "data"),
            prevent_initial_call=False,
        )
        def _draw(handles, selected_id, color_by, gs):
            return _build_figure(handles, selected_id, gs, color_by=color_by)
