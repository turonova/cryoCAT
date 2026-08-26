"""Utilities page — standalone builder tools discovered via ``@gui_exposed``.

Every function decorated with ``@gui_exposed(category="builder", standalone=True)``
appears here as its own panel.  Adding a new standalone builder requires only the
decorator on the function; the page discovers it automatically via
:func:`~cryocat.app.discovery.standalone_builders`.

Layout mirrors the other suite pages: a sticky sidebar on the left holds the
form controls for each tool; the main column on the right shows the corresponding
visualisation(s).

Contract: exposes ``layout`` (attribute) and ``register_callbacks(app)``.
"""

import math

import numpy as np
import plotly.graph_objects as go

import dash
from dash import html, dcc, Input, Output, State, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from cryocat.app import formgen, ids, styles, discovery
from cryocat.app.components.pathfield import get_path_field
from cryocat.app.apputils import generate_kwargs, run_operation
from cryocat.app.components.anglesbuilder import (
    get_angles_builder_sidebar_content,
    register_angles_builder_callbacks,
    inplane_figure,
    _ID_TYPE as _ANGLES_ID_TYPE,
)
from cryocat.app.components.graphsettings import styled_figure, error_figure
from cryocat.app.components.wedgepreview import wedge_xz_figure
from cryocat.app.components.poolpicker import get_pool_picker, register_pool_picker_callbacks
from cryocat.app.components.orientpicker import (
    get_orientation_picker_controls,
    get_orientation_picker_graph,
    register_orientation_picker_callbacks,
)
from cryocat.utils.geom import generate_angles
from cryocat.app.pageshell import page_shell, sidebar_accordion
from cryocat.utils.classutils import GuiEntry


_OUTPUT_AREA_ID = "util-output-area"
_WEDGE_ID_TYPE = "wedge-util-param"

# ── Alpha-shape tool constants ─────────────────────────────────────────────────

_ALPHA_PREFIX = "util-alpha"
_ALPHA_GRAPH_ID = "util-alpha-graph"
_ALPHA_STATS_ID = "util-alpha-stats"
_ALPHA_PANEL_ID = "util-alpha-panel"

# ── Orientation-picker constants ───────────────────────────────────────────────

_ORIENT_PREFIX = "util-orient"
_ORIENT_GRAPH_ID = "util-orient-graph"
_ORIENT_PANEL_ID = "util-orient-panel"

# Server-side tetra cache: motl_id → (tetra_mesh, pt_map).  Open3D objects
# cannot be serialised into dcc.Store; they live here for the process lifetime.
_alpha_tetra_cache: dict[str, tuple] = {}


# ── Alpha-shape module-level helpers ──────────────────────────────────────────


def _get_motl_coords(motl_id: str) -> np.ndarray | None:
    """Return (N, 3) coordinate array for *motl_id*, or None on failure."""
    from cryocat.app import pool
    from cryocat.core.cryomotl import Motl
    try:
        return Motl(pool.get_rows(motl_id)).get_coordinates()
    except Exception:
        return None


def _compute_tetra(motl_id: str) -> dict | None:
    """Compute (and cache) the Delaunay tetrahedralisation for *motl_id*.

    Returns a plain JSON-serialisable dict with the alpha range and point count,
    so it can be stored in a dcc.Store.  The open3d objects go into
    ``_alpha_tetra_cache`` keyed by *motl_id*.
    """
    from cryocat.core.surface import Mesh
    coords = _get_motl_coords(motl_id)
    if coords is None or len(coords) < 4:
        return None
    try:
        tetra_mesh, pt_map = Mesh.alpha_shape_tetra(coords)
    except ValueError:
        return None
    _alpha_tetra_cache[motl_id] = (tetra_mesh, pt_map)
    lo, hi = Mesh.suggest_alpha_range(coords)
    return {
        "motl_id": motl_id,
        "log_min": math.log10(lo),
        "log_max": math.log10(hi),
        "n_points": int(len(coords)),
    }


def _slider_to_alpha(slider_val: float, tetra_info: dict) -> float:
    """Map a [0, 1] slider value to a linear alpha via the stored log range."""
    log_min = tetra_info["log_min"]
    log_max = tetra_info["log_max"]
    return 10.0 ** (log_min + float(slider_val) * (log_max - log_min))


def _render_alpha_shape(
    alpha: float, tetra_info: dict, show_pts: bool, gs: dict
) -> tuple[go.Figure, str]:
    """Build the Mesh3d figure and stats string for the given alpha.

    Reads the tetra from ``_alpha_tetra_cache``; re-fetches coordinates from
    the pool for the point overlay.
    """
    import time
    from cryocat.core.surface import Mesh
    motl_id = tetra_info["motl_id"]
    tetra_mesh, pt_map = _alpha_tetra_cache.get(motl_id, (None, None))
    if tetra_mesh is None:
        return error_figure("Reload the motl to rebuild the tetrahedralisation."), ""
    coords = _get_motl_coords(motl_id)
    if coords is None:
        return error_figure("Motl data not found in pool."), ""
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
    if show_pts:
        n = len(coords)
        if n > 30000:
            step = max(1, n // 30000)
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
    fig = styled_figure(go.Figure(traces), gs, uirevision=f"alpha-{motl_id}")
    n_comp = mesh.get_connected_component_count() if n_faces > 0 else 0
    watertight = mesh.is_watertight() if n_faces > 0 else False
    stats = (
        f"Vertices: {n_verts:,} · Triangles: {n_faces:,} · "
        f"Components: {n_comp} · Watertight: {'yes' if watertight else 'no'} · "
        f"{elapsed * 1000:.0f} ms"
    )
    return fig, stats


def _do_alpha_save(
    alpha: float | None, tetra_info: dict | None, path: str | None
) -> str:
    """Save the alpha-shape mesh to *path* via Mesh.save."""
    if not tetra_info or not path or not str(path).strip():
        return "Set a file path first."
    if alpha is None or alpha <= 0:
        return "No valid alpha set. Move the slider first."
    from cryocat.core.surface import Mesh
    motl_id = tetra_info["motl_id"]
    tetra_mesh, pt_map = _alpha_tetra_cache.get(motl_id, (None, None))
    if tetra_mesh is None:
        return "Reload the motl to recompute the tetrahedralisation."
    coords = _get_motl_coords(motl_id)
    if coords is None:
        return "Motl data not found."
    try:
        mesh = Mesh.from_alpha_shape(coords, alpha, tetra_mesh, pt_map)
        run_operation(mesh.save, {"output_path": path})
        return f"Saved → {path}  (alpha={alpha:.6g}, source={motl_id})"
    except Exception as exc:
        return f"Error: {exc}"


def _do_alpha_register(alpha: float | None, name: str | None) -> str:
    """Bind *alpha* to *name* in the console locals."""
    if alpha is None or alpha <= 0:
        return "No alpha set. Move the slider first."
    if not name or not name.strip():
        return "Provide a variable name."
    name = name.strip()
    if not name.isidentifier():
        return f"{name!r} is not a valid Python identifier."
    from cryocat.app.console.execute import _CONSOLE_LOCALS
    _CONSOLE_LOCALS[name] = float(alpha)
    return f"Registered @{name} = {alpha:.6g}."


# ── Alpha-shape sidebar layout ─────────────────────────────────────────────────


def _alpha_shape_sidebar_content() -> html.Div:
    p = _ALPHA_PREFIX
    return html.Div([
        get_pool_picker(f"{p}-src", label="Motl source"),
        html.Hr(style={"margin": "0.5rem 0"}),
        formgen.form_row(
            "alpha",
            dcc.Slider(
                id=f"{p}-slider",
                min=0.0, max=1.0, step=0.001, value=0.5,
                tooltip={"placement": "bottom", "always_visible": False},
                marks=None,
            ),
            "Log-scaled alpha. Low values give fine surface detail; high values approach the convex hull.",
            label_text="Alpha",
        ),
        formgen.form_row(
            "alpha_val",
            dbc.Input(id=f"{p}-display", type="number", size="sm", debounce=True),
            "Current alpha value (linear scale). Type a precise value; press Enter to apply.",
            label_text="Alpha value",
        ),
        formgen.form_row(
            "show_points",
            dbc.Switch(id=f"{p}-points-sw", value=True, label=""),
            "Overlay the input point cloud on the surface to show where alpha is too small.",
            label_text="Show points",
        ),
        html.Hr(style={"margin": "0.5rem 0"}),
        formgen.form_row(
            "save_path",
            get_path_field(
                f"{p}-save-path", mode="save", extensions=(".ply", ".vtp"),
                placeholder="e.g. /data/mesh.ply",
            ),
            "Write path for the mesh. Format inferred from the extension (.ply or .vtp).",
            label_text="Save path",
        ),
        dbc.Button(
            "Save mesh",
            id=f"{p}-save-btn",
            color=styles.BTN_PRIMARY,
            size="sm",
            style={"width": "100%"},
        ),
        html.Div(id=f"{p}-save-status", style=styles.HINT),
        html.Hr(style={"margin": "0.5rem 0"}),
        formgen.form_row(
            "reg_name",
            dbc.Input(
                id=f"{p}-reg-name", type="text", size="sm",
                debounce=True, placeholder="my_alpha",
            ),
            "Python identifier. The current alpha value will be accessible as @name in the console.",
            label_text="@name",
        ),
        dbc.Button(
            "Register as @name",
            id=f"{p}-reg-btn",
            color=styles.BTN_SECONDARY,
            size="sm",
            style={"width": "100%"},
        ),
        html.Div(id=f"{p}-reg-status", style=styles.HINT),
        dcc.Store(id=f"{p}-tetra-info"),
    ])


# ── Sidebar helpers ────────────────────────────────────────────────────────────


def _sidebar_content(builder: GuiEntry) -> html.Div:
    prefix = f"util-{builder.fn.__name__}"
    if builder.fn.__name__ == "generate_angles":
        return get_angles_builder_sidebar_content(prefix, preview_btn=True)
    if builder.fn.__name__ == "generate_wedge_mask":
        return _wedge_mask_sidebar_content(prefix)
    return html.Div("Controls not yet implemented.", style={"color": "grey"})


def _wedge_mask_sidebar_content(prefix: str) -> html.Div:
    entry = discovery.get("wedgeutils.generate_wedge_mask")
    form_rows = formgen.build_form(
        entry,
        id_type=_WEDGE_ID_TYPE,
        id_extra={"owner": prefix},
    )
    return html.Div(
        [
            html.Div(form_rows, style={"marginBottom": "0.75rem"}),
            html.Div(get_path_field(f"{prefix}-output-path", mode="save",
                                    extensions=(".em",),
                                    placeholder="Output path (e.g. /path/to/wedge_mask.em)"),
                     style={"marginBottom": "0.4rem"}),
            dbc.Button(
                "Preview (middle XZ slice)",
                id=f"{prefix}-preview-btn",
                color="secondary",
                size="sm",
                style={"width": "100%", "marginBottom": "0.4rem"},
            ),
            dbc.Button(
                "Generate wedge mask",
                id=f"{prefix}-generate",
                color="primary",
                size="sm",
                style={"width": "100%", "marginBottom": "0.4rem"},
            ),
            html.Div(
                id=f"{prefix}-status",
                style={**styles.HINT, "marginTop": "0.25rem", "wordBreak": "break-word"},
            ),
            dcc.Store(id=f"{prefix}-params"),
        ],
    )


# The main area is a single shared output: whichever builder ran most recently
# writes its figure(s) here, replacing whatever was previously displayed. We
# never pre-allocate per-builder graphs.


# ── Layout builders ────────────────────────────────────────────────────────────


def _sidebar(builders: list[GuiEntry]) -> list:
    items = [
        dbc.AccordionItem(
            _sidebar_content(b),
            title=b.label,
            item_id=f"util-acc-{b.fn.__name__}",
        )
        for b in builders
    ]
    items.append(
        dbc.AccordionItem(
            _alpha_shape_sidebar_content(),
            title="Alpha Shape Tuning",
            item_id="util-acc-alpha-shape",
        )
    )
    items.append(
        dbc.AccordionItem(
            get_orientation_picker_controls(_ORIENT_PREFIX, mode=None, show_structure=True),
            title="Orientation Picker",
            item_id="util-acc-orient",
        )
    )
    return [
        sidebar_accordion(items, active_item=[f"util-acc-{b.fn.__name__}" for b in builders]),
    ]


def _main(builders: list[GuiEntry]) -> list:
    builder_area = html.Div(
        id=_OUTPUT_AREA_ID,
        children=html.P(
            "Run a builder from the sidebar to display its result here.",
            style={"color": "grey"},
        ) if builders else None,
    )
    alpha_panel = html.Div(
        [
            dcc.Graph(id=_ALPHA_GRAPH_ID, style={"height": "500px"}),
            html.Div(id=_ALPHA_STATS_ID, style=styles.HINT),
        ],
        id=_ALPHA_PANEL_ID,
    )
    orient_panel = html.Div(
        [get_orientation_picker_graph(_ORIENT_PREFIX, height="500px")],
        id=_ORIENT_PANEL_ID,
    )
    return [builder_area, alpha_panel, orient_panel]


def _build_layout() -> html.Div:
    builders = discovery.standalone_builders()
    return html.Div(
        [
            page_shell(_sidebar(builders), _main(builders)),
        ],
        style={"margin": "0", "padding": "0"},
    )


layout = _build_layout()


# ── Callbacks ──────────────────────────────────────────────────────────────────


def _err_panel(msg: str) -> html.Div:
    """Plain text panel for cases where we want a message rather than a figure
    in the shared output area."""
    return html.Div(msg, style={"color": "grey", "padding": "0.5rem"})


def _register_alpha_shape_callbacks(app) -> None:
    p = _ALPHA_PREFIX

    register_pool_picker_callbacks(app, f"{p}-src")

    @app.callback(
        Output(f"{p}-tetra-info", "data"),
        Input(f"{p}-src-value", "data"),
        prevent_initial_call=True,
    )
    def _on_motl_changed(motl_ids):
        if not motl_ids:
            raise PreventUpdate
        motl_id = motl_ids[0]
        info = _compute_tetra(motl_id)
        return info

    @app.callback(
        Output(_ALPHA_GRAPH_ID, "figure"),
        Output(_ALPHA_STATS_ID, "children"),
        Output(f"{p}-display", "value"),
        Input(f"{p}-slider", "value"),
        Input(f"{p}-points-sw", "value"),
        State(f"{p}-tetra-info", "data"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        prevent_initial_call=True,
    )
    def _on_slider(slider_val, show_pts, tetra_info, gs):
        if not tetra_info:                                                      # 1
            raise PreventUpdate
        alpha = _slider_to_alpha(float(slider_val or 0.5), tetra_info)         # 2
        figure, stats = _render_alpha_shape(alpha, tetra_info, bool(show_pts), gs or {})  # 3
        return figure, stats, round(alpha, 6)                                   # 4

    @app.callback(
        Output(_ALPHA_GRAPH_ID, "figure", allow_duplicate=True),
        Output(_ALPHA_STATS_ID, "children", allow_duplicate=True),
        Input(f"{p}-display", "value"),
        State(f"{p}-points-sw", "value"),
        State(f"{p}-tetra-info", "data"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        prevent_initial_call=True,
    )
    def _on_display_typed(display_val, show_pts, tetra_info, gs):
        if display_val is None or not tetra_info:                               # 1
            raise PreventUpdate
        alpha = float(display_val)                                              # 2
        if alpha <= 0:                                                          # 3
            raise PreventUpdate
        figure, stats = _render_alpha_shape(alpha, tetra_info, bool(show_pts), gs or {})  # 4
        return figure, stats                                                    # 5

    @app.callback(
        Output(f"{p}-save-status", "children"),
        Input(f"{p}-save-btn", "n_clicks"),
        State(f"{p}-display", "value"),
        State(f"{p}-tetra-info", "data"),
        State({"type": "path-input", "owner": f"{p}-save-path"}, "value"),
        prevent_initial_call=True,
    )
    def _on_save(n_clicks, display_val, tetra_info, path):
        if not n_clicks:
            raise PreventUpdate
        alpha = float(display_val) if display_val is not None else None
        return _do_alpha_save(alpha, tetra_info, path)

    @app.callback(
        Output(f"{p}-reg-status", "children"),
        Input(f"{p}-reg-btn", "n_clicks"),
        State(f"{p}-display", "value"),
        State(f"{p}-reg-name", "value"),
        prevent_initial_call=True,
    )
    def _on_register(n_clicks, display_val, name):
        if not n_clicks:
            raise PreventUpdate
        alpha = float(display_val) if display_val is not None else None
        return _do_alpha_register(alpha, name)


def register_callbacks(app) -> None:
    from cryocat.analysis import visplot

    _register_alpha_shape_callbacks(app)
    register_orientation_picker_callbacks(app, _ORIENT_PREFIX, mode=None, show_structure=True)

    for b in discovery.standalone_builders():
        prefix = f"util-{b.fn.__name__}"
        if b.fn.__name__ == "generate_angles":
            # Register _collect_params and _create from anglesbuilder;
            # skip the built-in single-graph preview so we own both outputs.
            register_angles_builder_callbacks(app, prefix, with_graphs=False)

            @app.callback(
                Output(f"{prefix}-angles", "data"),
                Output(_OUTPUT_AREA_ID, "children", allow_duplicate=True),
                Input(f"{prefix}-preview-btn", "n_clicks"),
                State({"type": _ANGLES_ID_TYPE, "owner": prefix, "param": ALL, "tag": ALL}, "value"),
                State({"type": _ANGLES_ID_TYPE, "owner": prefix, "param": ALL, "tag": ALL}, "id"),
                State(ids.GRAPH_SETTINGS_STORE, "data"),
                prevent_initial_call=True,
            )
            def _preview(n_clicks, values, ids, gs, _prefix=prefix):
                if not n_clicks:
                    raise PreventUpdate

                params = generate_kwargs(ids, values) if (values and ids) else {}
                if params.get("cone_angle") is None or params.get("cone_sampling") is None:
                    return dash.no_update, _err_panel("Set cone_angle and cone_sampling first.")

                try:
                    kwargs = {k: v for k, v in params.items() if v is not None}
                    angles = generate_angles(**kwargs)
                except Exception as exc:
                    return dash.no_update, _err_panel(f"Error generating angles: {exc}")

                angles_list = angles.tolist()

                n_phi = len(np.unique(np.round(angles[:, 0], 8)))
                n_cone = len(angles) // n_phi if n_phi > 0 else len(angles)

                try:
                    fig1 = visplot.plot_rotation_normals(angles)
                    sphere_fig = styled_figure(
                        fig1, gs or {},
                        uirevision=f"{_prefix}-preview",
                        title={"text": f"Cone sampling — {n_cone} angles", "font": {"size": 12}},
                        margin={"l": 0, "r": 0, "t": 40, "b": 0},
                    )
                except Exception as exc:
                    sphere_fig = error_figure(f"Sphere plot error: {exc}")

                try:
                    inplane_fig = inplane_figure(angles, gs)
                except Exception as exc:
                    inplane_fig = error_figure(f"Inplane plot error: {exc}")

                output = dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=sphere_fig, style={"height": "460px"}), width=6),
                        dbc.Col(dcc.Graph(figure=inplane_fig, style={"height": "460px"}), width=6),
                    ],
                    className="g-1",
                )
                return angles_list, output

        elif b.fn.__name__ == "generate_wedge_mask":
            _register_wedge_mask_callbacks(app, prefix)


def _register_wedge_mask_callbacks(app, prefix: str) -> None:
    @app.callback(
        Output(f"{prefix}-params", "data"),
        Input({"type": _WEDGE_ID_TYPE, "owner": prefix, "param": ALL, "tag": ALL}, "value"),
        State({"type": _WEDGE_ID_TYPE, "owner": prefix, "param": ALL, "tag": ALL}, "id"),
    )
    def _collect_params(values, ids):
        if not values or not ids:
            raise PreventUpdate
        return generate_kwargs(ids, values)

    @app.callback(
        Output(_OUTPUT_AREA_ID, "children", allow_duplicate=True),
        Output(f"{prefix}-status", "children", allow_duplicate=True),
        Input(f"{prefix}-preview-btn", "n_clicks"),
        State(f"{prefix}-params", "data"),
        prevent_initial_call=True,
    )
    def _preview(n_clicks, params):
        if not n_clicks:
            raise PreventUpdate
        if not params:
            return _err_panel("Fill in the form parameters first."), "Preview needs the form filled."
        required = ["map_size", "wedgelist", "tomo_number"]
        missing = [r for r in required if not params.get(r)]
        if missing:
            msg = f"Missing required fields: {', '.join(missing)}."
            return _err_panel(msg), msg
        try:
            # In-memory only: drop any output_path the user typed for the actual generate.
            kwargs = {k: v for k, v in params.items() if v is not None and k != "output_path"}
            _wedge_fn = discovery.get("wedgeutils.generate_wedge_mask").fn
            result = _wedge_fn(**kwargs)
            mask = result["mask"] if isinstance(result, dict) else result
            output = dcc.Graph(
                figure=wedge_xz_figure(mask),
                style={"height": "520px", "width": "520px", "maxWidth": "100%"},
            )
            return output, f"Preview rendered (mask shape {mask.shape})."
        except Exception as exc:
            msg = f"Preview error: {exc}"
            return _err_panel(msg), msg

    @app.callback(
        Output(f"{prefix}-status", "children"),
        Input(f"{prefix}-generate", "n_clicks"),
        State(f"{prefix}-params", "data"),
        State({"type": "path-input", "owner": f"{prefix}-output-path"}, "value"),
        prevent_initial_call=True,
    )
    def _generate(n_clicks, params, out_path):
        if not n_clicks:
            raise PreventUpdate
        if not params:
            return "Fill in the form parameters first."
        required = ["map_size", "wedgelist", "tomo_number"]
        missing = [r for r in required if not params.get(r)]
        if missing:
            return f"Missing required fields: {', '.join(missing)}."
        try:
            kwargs = {k: v for k, v in params.items() if v is not None}
            if out_path and str(out_path).strip():
                kwargs["output_path"] = out_path
            run_operation(discovery.get("wedgeutils.generate_wedge_mask").fn, kwargs)
            msg = f"Wedge mask generated"
            if out_path:
                msg += f" → {out_path}"
            return msg
        except Exception as exc:
            return f"Error: {exc}"
