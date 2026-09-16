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
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State

from cryocat.app import ids
import cryocat.app.datapool as _datapool
from cryocat.app.components.surface_registry import registry as _surface_registry, _mesh_has_curvatures
from cryocat.app.components.graphsettings import styled_figure
from cryocat.app.components.customel import customel_graph
from cryocat.analysis.visplot import resolve_palette as _resolve_palette, resolve_colorscale as _resolve_colorscale
from cryocat.app.formgen import make_dropdown
from cryocat.app.components.paletteloader import get_palette_loader, register_palette_loader_callbacks


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
    {"label": "Hit counts", "value": "hit_counts"},
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
    if color_by in (None, "none"):
        return None
    if color_by == "hit_counts":
        counts = getattr(psurf.surface, "_hit_counts", None)
        return np.asarray(counts, dtype=np.float32) if counts is not None else None
    if not psurf.is_mesh:
        return None
    if not _mesh_has_curvatures(psurf.surface):
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
    opacity: float = 0.5,
    cmin: float | None = None,
    cmax: float | None = None,
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
        opacity=opacity,
        name=name,
        hoverinfo="name",
        flatshading=True,
    )
    if intensity is not None and len(intensity) == v.shape[0]:
        finite = np.isfinite(intensity)
        if finite.any():
            if cmin is None:
                vmin = float(np.percentile(intensity[finite], 2))
            else:
                vmin = float(cmin)
            if cmax is None:
                vmax = float(np.percentile(intensity[finite], 98))
            else:
                vmax = float(cmax)
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
    opacity: float = 0.5,
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
            marker=dict(size=marker_size, color=color, opacity=opacity),
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
    isect_df: "pd.DataFrame | None" = None,
    surface_opacity: float = 0.5,
    hit_color_by: str = "t_hit",
    hit_marker_size: int = 5,
    hit_coord_prefix: str = "hit_points",
    surf_palette: str = "",
    hit_colorscale: str = "",
    cmin: float | None = None,
    cmax: float | None = None,
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
    isect_df : pandas.DataFrame, optional
        Full per-ray DataFrame from the data pool (one row per ray, hits and
        misses, with a ``hit`` boolean column).  When present, hit rows are
        drawn as a ``go.Scatter3d`` layer on top of all surfaces.  Miss count
        is derived from ``(~df["hit"]).sum()``.
    surface_opacity : float, default 0.5
        Opacity applied to every surface trace (mesh or point cloud).
    hit_color_by : str, default "t_hit"
        Column from *isect_df* used to colour the hit-point scatter.  Falls
        back to ``"t_hit"`` when the column is absent.
    hit_marker_size : int, default 5
        Marker size for the hit-point scatter.
    hit_coord_prefix : str, default "hit_points"
        Prefix for the three coordinate columns (``{prefix}_x/y/z``) in
        *isect_df*.  Use ``"hit_points"`` for ray-cast results and
        ``"closest_points"`` for distance-to-points results.
    """
    import pandas as _pd
    handles = handles or {}
    has_visible = any(h.get("visible", True) for h in handles.values())
    _coord_x = f"{hit_coord_prefix}_x"
    has_hits = (
        isect_df is not None
        and not isect_df.empty
        and _coord_x in isect_df.columns
        and "hit" in isect_df.columns
        and isect_df["hit"].any()
    ) if isect_df is not None else False

    if not has_visible and not has_hits:
        fig = go.Figure()
        fig.update_layout(
            annotations=[{
                "text": "No visible surfaces", "showarrow": False,
                "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
                "font": {"size": 14},
            }]
        )
        return fig

    palette = _resolve_palette(surf_palette or (gs or {}).get("discrete_palette"))
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
            if is_sel and color_by and color_by != "none" and (
                h.get("has_curvatures") or h.get("has_hit_counts")
            ):
                intensity = _vertex_field(psurf, color_by)
            surf_opacity = min(1.0, surface_opacity + 0.30) if is_sel else surface_opacity
            traces.extend(_mesh_traces(psurf, color, label, is_sel, intensity=intensity,
                                       opacity=surf_opacity, cmin=cmin, cmax=cmax))
        elif rep == "point_cloud":
            surf_opacity = min(1.0, surface_opacity + 0.30) if is_sel else surface_opacity
            traces.extend(_point_cloud_traces(psurf, color, label, is_sel,
                                              opacity=surf_opacity))
        # Unknown representations are skipped (handle is informational only).

    if has_hits:
        hit_rows = isect_df[isect_df["hit"]]
        n_hits = len(hit_rows)
        n_miss = int((~isect_df["hit"]).sum())
        miss_str = f", {n_miss} misses" if n_miss else ""
        _col = hit_color_by if hit_color_by in hit_rows.columns else next(
            (c for c in hit_rows.columns if hit_rows[c].dtype.kind in "fiu" and c not in ("hit",)),
            hit_color_by,
        )
        _color_vals = hit_rows[_col].tolist() if _col in hit_rows.columns else None
        _cs_raw = hit_colorscale or (gs or {}).get("continuous_palette") or "Viridis"
        try:
            _cs = [[p, c] for p, c in _resolve_colorscale(_cs_raw)]
        except Exception:
            _cs = "Viridis"
        traces.append(go.Scatter3d(
            x=hit_rows[f"{hit_coord_prefix}_x"].tolist(),
            y=hit_rows[f"{hit_coord_prefix}_y"].tolist(),
            z=hit_rows[f"{hit_coord_prefix}_z"].tolist(),
            mode="markers",
            marker=dict(
                size=int(hit_marker_size) if hit_marker_size else 5,
                color=_color_vals,
                colorscale=_cs,
                showscale=True,
                colorbar=dict(title=_col, thickness=12, len=0.6),
                opacity=0.9,
            ),
            name=f"Hit points ({n_hits}{miss_str})",
        ))

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
    _row = {"display": "flex", "alignItems": "center", "gap": "0.5rem", "marginBottom": "0.4rem", "flexWrap": "wrap"}
    _num_style = {"width": "6rem"}
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Color by", style={"marginRight": "0.25rem", "flexShrink": 0}),
                    make_dropdown(
                        f"{prefix}-color-by",
                        COLOR_BY_OPTIONS,
                        "none",
                        clearable=False,
                        style={"width": "200px"},
                    ),
                    html.Label("cmin", style={"marginLeft": "0.5rem", "marginRight": "0.25rem", "flexShrink": 0}),
                    dbc.Input(
                        id=f"{prefix}-cmin",
                        type="number",
                        placeholder="auto",
                        debounce=True,
                        size="sm",
                        style=_num_style,
                    ),
                    html.Label("cmax", style={"marginLeft": "0.25rem", "marginRight": "0.25rem", "flexShrink": 0}),
                    dbc.Input(
                        id=f"{prefix}-cmax",
                        type="number",
                        placeholder="auto",
                        debounce=True,
                        size="sm",
                        style=_num_style,
                    ),
                    dbc.Button(
                        "Widest range",
                        id=f"{prefix}-widest-range-btn",
                        size="sm",
                        color="secondary",
                        style={"flexShrink": 0},
                    ),
                    html.Span(
                        "",
                        id=f"{prefix}-color-range-display",
                        style={"fontSize": "0.8rem", "color": "#888", "marginLeft": "0.25rem"},
                    ),
                    html.Label("Opacity", style={"marginLeft": "0.75rem", "marginRight": "0.25rem", "flexShrink": 0}),
                    html.Div(
                        dcc.Slider(
                            id=f"{prefix}-surface-opacity",
                            min=0.0, max=1.0, step=0.05, value=0.5,
                            marks=None,
                            tooltip=None,
                        ),
                        style={"width": "180px"},
                    ),
                    html.Label("Surface palette", style={"marginLeft": "0.75rem", "marginRight": "0.25rem", "flexShrink": 0}),
                    get_palette_loader(f"{prefix}-surf-pal", mode="discrete", allow_auto=True, swatch_inline=True),
                    dbc.Button(
                        "Update graph",
                        id=f"{prefix}-update-surf-btn",
                        size="sm",
                        color="primary",
                        style={"flexShrink": 0, "marginLeft": "0.25rem"},
                    ),
                ],
                style=_row,
            ),
            html.Div(
                [
                    html.Label("Hit colour by", style={"marginRight": "0.25rem", "flexShrink": 0}),
                    make_dropdown(
                        f"{prefix}-hit-color-by",
                        [{"label": "t_hit (distance)", "value": "t_hit"}],
                        "t_hit",
                        clearable=False,
                        style={"width": "180px"},
                    ),
                    html.Label("Marker size", style={"marginLeft": "0.75rem", "marginRight": "0.25rem", "flexShrink": 0}),
                    dbc.Input(
                        id=f"{prefix}-hit-marker-size",
                        type="number", value=5, min=1, max=20, step=1,
                        debounce=True,
                        size="sm",
                        style={"width": "4rem"},
                    ),
                    html.Label("Hit palette", style={"marginLeft": "0.75rem", "marginRight": "0.25rem", "flexShrink": 0}),
                    get_palette_loader(f"{prefix}-hit-pal", mode="continuous", allow_auto=True, swatch_inline=True),
                    dbc.Button(
                        "Update graph",
                        id=f"{prefix}-update-hit-btn",
                        size="sm",
                        color="primary",
                        style={"flexShrink": 0, "marginLeft": "0.25rem"},
                    ),
                ],
                style=_row,
            ),
            dbc.Spinner(
                customel_graph(prefix, "graph",
                    dcc.Graph(
                        id={"type": "styled-graph", "owner": prefix, "name": "graph"},
                        style={"height": "620px"},
                        config={"scrollZoom": True},
                    )),
            ),
        ]
    )


def register_surface_view_callbacks(
    app,
    prefix: str,
    pool_store_id: str,
    *,
    selected_store_id: str | None = None,
    isect_pool_id_store_id: str | None = None,
    isect_coord_prefix_store_id: str | None = None,
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
    isect_pool_id_store_id : str, optional
        Id of a ``dcc.Store`` carrying the data pool entry id (a short string
        like ``"isect_1"``) written by the page's adoption callback after a
        ``ray_intersections`` run.  When provided, the viewer fetches the full
        per-ray DataFrame from :mod:`cryocat.app.datapool` server-side and
        draws hit points as a ``Scatter3d`` layer coloured by ``t_hit``.
        Misses (``hit == False``) are excluded from the trace but counted.
    """

    color_by_id = f"{prefix}-color-by"
    surface_opacity_id = f"{prefix}-surface-opacity"
    hit_color_by_id = f"{prefix}-hit-color-by"
    hit_marker_size_id = f"{prefix}-hit-marker-size"
    surf_pal_id = f"{prefix}-surf-pal-value"
    hit_pal_id = f"{prefix}-hit-pal-value"
    cmin_id = f"{prefix}-cmin"
    cmax_id = f"{prefix}-cmax"
    range_display_id = f"{prefix}-color-range-display"

    register_palette_loader_callbacks(app, f"{prefix}-surf-pal", mode="discrete",
                                      settings_store_id=ids.GRAPH_SETTINGS_STORE)
    register_palette_loader_callbacks(app, f"{prefix}-hit-pal", mode="continuous",
                                      settings_store_id=ids.GRAPH_SETTINGS_STORE)

    def _resolve_isect_df(pool_data_id: str | None):
        if not pool_data_id:
            return None
        try:
            return _datapool.get_payload(pool_data_id)
        except Exception:
            return None

    _button_inputs = [
        Input(f"{prefix}-update-surf-btn", "n_clicks"),
        Input(f"{prefix}-update-hit-btn", "n_clicks"),
    ]
    _setting_states = [
        State(color_by_id, "value"),
        State(surface_opacity_id, "value"),
        State(hit_color_by_id, "value"),
        State(hit_marker_size_id, "value"),
        State(surf_pal_id, "data"),
        State(hit_pal_id, "data"),
        State(cmin_id, "value"),
        State(cmax_id, "value"),
    ]

    # isect_pool_id_store_id is an Input so the graph redraws automatically when
    # a computation completes (without the user having to press "Update graph").
    extra_inputs = []
    extra_state = []
    if isect_pool_id_store_id is not None:
        extra_inputs.append(Input(isect_pool_id_store_id, "data"))
    if isect_coord_prefix_store_id is not None:
        extra_state.append(State(isect_coord_prefix_store_id, "data"))

    if isect_pool_id_store_id is not None:
        # Populate hit-colour-by dropdown options from the intersection DataFrame columns,
        # and auto-select the first valid column so the scatter uses meaningful coloring.
        @app.callback(
            Output(hit_color_by_id, "options"),
            Output(hit_color_by_id, "value"),
            Input(isect_pool_id_store_id, "data"),
            prevent_initial_call=True,
        )
        def _update_hit_color_options(pool_data_id):
            df = _resolve_isect_df(pool_data_id)
            fallback = [{"label": "t_hit (distance)", "value": "t_hit"}]
            if df is None or df.empty:
                return fallback, "t_hit"
            _coord_cols = {c for c in df.columns if c.endswith(("_x", "_y", "_z"))}
            _exclude = {"hit", "primitive_ids", "geometry_ids"} | _coord_cols
            options = []
            for c in df.columns:
                if c in _exclude:
                    continue
                if df[c].dtype.kind not in "fiuc":
                    continue
                label = "t_hit (distance)" if c == "t_hit" else c
                options.append({"label": label, "value": c})
            if not options:
                return fallback, "t_hit"
            return options, options[0]["value"]

    _has_selected = selected_store_id is not None
    _has_pool = isect_pool_id_store_id is not None
    _has_prefix = isect_coord_prefix_store_id is not None

    def _build_kw(surface_opacity, hit_color_by, hit_marker_size, surf_palette, hit_colorscale, cmin_val=None, cmax_val=None, pool_data_id=None, coord_prefix=None):
        return dict(
            surface_opacity=float(surface_opacity) if surface_opacity is not None else 0.5,
            hit_color_by=hit_color_by or "t_hit",
            hit_marker_size=int(hit_marker_size) if hit_marker_size else 5,
            surf_palette=surf_palette or "",
            hit_colorscale=hit_colorscale or "",
            isect_df=_resolve_isect_df(pool_data_id),
            hit_coord_prefix=coord_prefix or "hit_points",
            cmin=float(cmin_val) if cmin_val is not None else None,
            cmax=float(cmax_val) if cmax_val is not None else None,
        )

    # One registration. Settings are States so they only apply when the user
    # clicks "Update graph" or when the pool/selected store changes.
    # To add a new optional store: add its id parameter above, append
    # State(...) to extra_state when not None, add a _has_<name> flag, and
    # read args[_i] in _draw. No new @app.callback block.
    _all_inputs = [
        Input(pool_store_id, "data"),
        *([] if not _has_selected else [Input(selected_store_id, "data")]),
        *_button_inputs,
        *_setting_states,
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        *extra_inputs,
        *extra_state,
    ]

    # "Widest range" button: compute global min/max of the current color-by
    # field across all visible mesh surfaces and populate cmin/cmax inputs.
    @app.callback(
        Output(cmin_id, "value"),
        Output(cmax_id, "value"),
        Output(range_display_id, "children"),
        Input(f"{prefix}-widest-range-btn", "n_clicks"),
        State(pool_store_id, "data"),
        State(color_by_id, "value"),
        prevent_initial_call=True,
    )
    def _widest_range(_, handles, color_by):
        handles = handles or {}
        all_vals: list[np.ndarray] = []
        for sid, h in handles.items():
            if not h.get("visible", True):
                continue
            if h.get("representation") != "mesh":
                continue
            psurf = _surface_registry.get(sid)
            if psurf is None:
                continue
            field = _vertex_field(psurf, color_by)
            if field is not None:
                finite = field[np.isfinite(field)]
                if len(finite):
                    all_vals.append(finite)
        if not all_vals:
            return None, None, ""
        combined = np.concatenate(all_vals)
        lo = float(combined.min())
        hi = float(combined.max())
        txt = f"range [{lo:.3g}, {hi:.3g}]"
        return lo, hi, txt

    @app.callback(
        Output({"type": "styled-graph", "owner": prefix, "name": "graph"}, "figure"),
        *_all_inputs,
        prevent_initial_call=False,
    )
    def _draw(*args):
        _i = 0
        handles = args[_i]; _i += 1
        selected_id = args[_i] if _has_selected else None
        if _has_selected: _i += 1
        _i += 2  # skip update-surf-btn and update-hit-btn n_clicks
        color_by = args[_i]; _i += 1
        surface_opacity = args[_i]; _i += 1
        hit_color_by = args[_i]; _i += 1
        hit_marker_size = args[_i]; _i += 1
        surf_palette = args[_i]; _i += 1
        hit_colorscale = args[_i]; _i += 1
        cmin_val = args[_i]; _i += 1
        cmax_val = args[_i]; _i += 1
        gs = args[_i]; _i += 1
        pool_data_id = args[_i] if _has_pool else None
        if _has_pool: _i += 1
        coord_prefix = args[_i] if _has_prefix else None
        return _build_figure(handles, selected_id, gs, color_by=color_by,
                             **_build_kw(surface_opacity, hit_color_by, hit_marker_size,
                                         surf_palette, hit_colorscale, cmin_val, cmax_val,
                                         pool_data_id, coord_prefix))
