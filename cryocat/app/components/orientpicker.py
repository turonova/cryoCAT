"""Orientation-picker component — reusable across Utilities and modals.

Three layout factories (choose one per use-site):

  get_orientation_picker_controls(prefix, *, mode=None, show_structure=True)
      Flat controls div for embedding inside an accordion item or modal.

  get_orientation_picker_graph(prefix, *, height="400px")
      dcc.Graph for embedding in the main area alongside the controls.

  get_orientation_picker_panel(prefix, *, mode=None, show_structure=True, height="400px")
      Self-contained two-column layout (controls left, graph right) for
      embedding in a modal body.

One callback factory:

  register_orientation_picker_callbacks(app, prefix, *, mode=None, show_structure=True)
      Registers all callbacks for one picker instance.  May be called any
      number of times with distinct *prefix* values.
      Raises RuntimeError if called twice with the same (app, prefix) pair.

Public contract
---------------
  {prefix}-value  dcc.Store
      direction mode → [x, y, z]  (unit vector, float list)
      rotation mode  → [phi, theta, psi]  (zxz degrees, float list)

Pure helpers are importable for testing:
  _normalize, _compute_angles, _nearest_sphere_point, _rotate_mesh_verts, _SPHERE_PTS
"""
from __future__ import annotations

import pathlib

import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as srot

from dash import html, dcc, Input, Output, State, no_update, callback_context as ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from cryocat.app import ids, styles, formgen
from cryocat.app.apputils import run_operation
from cryocat.app.components.graphsettings import styled_figure
from cryocat.app.components.pathfield import get_path_field
from cryocat.utils.geom import sample_sphere

# ── Sphere sample (computed once at import) ────────────────────────────────────

_N_SPHERE_PTS = 2000
_SPHERE_PTS: np.ndarray = sample_sphere(_N_SPHERE_PTS)

# ── Registration guard ─────────────────────────────────────────────────────────

_REGISTERED: dict[int, set[str]] = {}

# ── Pure helpers ───────────────────────────────────────────────────────────────


def _normalize(direction) -> np.ndarray:
    """Return unit vector; raise ValueError for a zero or near-zero input."""
    d = np.asarray(direction, dtype=float).ravel()
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        raise ValueError("Direction vector must not be zero.")
    return d / n


def _compute_angles(direction, phi: float) -> np.ndarray:
    """Return zxz Euler angles [phi, theta, psi] for direction + in-plane phi.

    theta and psi are derived directly from the direction using the ZXZ formula:
    direction = (sin(theta)*sin(psi), -sin(theta)*cos(psi), cos(theta)).
    """
    d = _normalize(direction)
    theta = float(np.degrees(np.arctan2(np.sqrt(d[0] ** 2 + d[1] ** 2), d[2])))
    psi = float(np.degrees(np.arctan2(d[0], -d[1])))
    return np.array([float(phi), theta, psi])


def _nearest_sphere_point(sphere_pts: np.ndarray, direction) -> np.ndarray:
    """Return the pre-sampled sphere point with the highest dot product with direction."""
    d = _normalize(direction)
    dots = sphere_pts @ d
    return sphere_pts[int(np.argmax(dots))]


def _rotate_mesh_verts(verts: np.ndarray, phi: float, theta: float, psi: float) -> np.ndarray:
    """Apply zxz Euler rotation (degrees) to an (N, 3) vertex array."""
    R = srot.from_euler("zxz", [phi, theta, psi], degrees=True).as_matrix()
    return (R @ verts.T).T


def _publish_fn(name: str, value) -> str:
    from cryocat.app.console.execute import _CONSOLE_LOCALS
    _CONSOLE_LOCALS[name] = value
    return f"Published @{name}"


def _struct_load_fn(path: str, level: float | None = None, binning: int = 1) -> dict:
    import cryocat.cryomap as cryomap
    from cryocat.app.components.volumeview import mesh_at

    volume = np.array(cryomap.read(path))
    bin_f = max(1, int(binning or 1))
    if bin_f > 1:
        from scipy.ndimage import zoom
        volume = zoom(volume, 1.0 / bin_f, order=1)
    lvl = float(level) if level is not None else float(np.mean(volume))
    mesh = mesh_at(volume, lvl)
    if mesh is None:
        raise ValueError(f"No surface found at level {lvl:.4g} — try a different value.")
    verts = np.column_stack([mesh["x"], mesh["y"], mesh["z"]])
    verts -= verts.mean(axis=0)
    max_dist = float(np.max(np.linalg.norm(verts, axis=1)))
    if max_dist < 1e-12:
        raise ValueError("Mesh has zero spatial extent.")
    verts /= max_dist
    return {
        "x": verts[:, 0].tolist(), "y": verts[:, 1].tolist(), "z": verts[:, 2].tolist(),
        "i": mesh["i"], "j": mesh["j"], "k": mesh["k"],
        "n_verts": len(verts),
    }


# ── Plotly trace builders ──────────────────────────────────────────────────────

def _sphere_surface() -> go.Surface:
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x, y=y, z=z, opacity=0.08,
        colorscale=[[0, "#CCCCCC"], [1, "#CCCCCC"]],
        showscale=False, hoverinfo="none", name="sphere", showlegend=False,
    )


def _sphere_scatter() -> go.Scatter3d:
    return go.Scatter3d(
        x=_SPHERE_PTS[:, 0], y=_SPHERE_PTS[:, 1], z=_SPHERE_PTS[:, 2],
        mode="markers", marker={"size": 3, "color": "#AAAAAA", "opacity": 0.25},
        name="pick", hoverinfo="none", showlegend=False,
    )


def _direction_trace(d: np.ndarray) -> go.Scatter3d:
    return go.Scatter3d(
        x=[0.0, float(d[0])], y=[0.0, float(d[1])], z=[0.0, float(d[2])],
        mode="lines+markers",
        line={"color": "#3498DB", "width": 5},
        marker={"size": [0, 8], "color": "#3498DB"},
        name="Direction", hoverinfo="skip", showlegend=False,
    )


def _reference_trace(visible: bool) -> go.Scatter3d:
    return go.Scatter3d(
        x=[0.0, 0.0], y=[0.0, 0.0], z=[0.0, 1.0],
        mode="lines+markers",
        line={"color": "#E74C3C", "width": 3, "dash": "dash"},
        marker={"size": [0, 6], "color": "#E74C3C"},
        name="Reference (z)", visible=visible, hoverinfo="skip", showlegend=False,
    )


def _mesh_trace(mesh_data: dict, mode: str, d: np.ndarray, phi: float) -> go.Mesh3d:
    if mode == "rotation":
        angles = _compute_angles(d, phi)
        verts = np.column_stack([mesh_data["x"], mesh_data["y"], mesh_data["z"]])
        verts = _rotate_mesh_verts(verts, angles[0], angles[1], angles[2])
        mx, my, mz = verts[:, 0].tolist(), verts[:, 1].tolist(), verts[:, 2].tolist()
    else:
        mx, my, mz = mesh_data["x"], mesh_data["y"], mesh_data["z"]
    return go.Mesh3d(
        x=mx, y=my, z=mz, i=mesh_data["i"], j=mesh_data["j"], k=mesh_data["k"],
        opacity=0.45, color="#85C1E9", flatshading=True,
        hoverinfo="skip", name="Structure", showlegend=False,
    )


def _build_sphere_fig(dir_data, mode, phi, ref_val, mesh_data, gs, prefix: str) -> go.Figure:
    try:
        d = _normalize(dir_data or [0.0, 0.0, 1.0])
    except ValueError:
        d = np.array([0.0, 0.0, 1.0])
    phi = float(phi or 0.0)
    mode = mode or "direction"
    show_ref = bool(ref_val)
    gs = gs or {}
    traces: list = [
        _sphere_surface(), _sphere_scatter(), _direction_trace(d), _reference_trace(show_ref),
    ]
    if mesh_data and mesh_data.get("x"):
        traces.append(_mesh_trace(mesh_data, mode, d, phi))
    else:
        traces.append(go.Mesh3d(x=[], y=[], z=[], name="Structure", showlegend=False))
    fig = go.Figure(data=traces)
    return styled_figure(
        fig, gs,
        uirevision=f"orient-{prefix}",
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        scene={
            "aspectmode": "cube",
            "xaxis": {"range": [-1.2, 1.2], "title": "x"},
            "yaxis": {"range": [-1.2, 1.2], "title": "y"},
            "zaxis": {"range": [-1.2, 1.2], "title": "z"},
        },
    )


# ── Layout factories ───────────────────────────────────────────────────────────


def get_orientation_picker_controls(
    prefix: str, *, mode: str | None = None, show_structure: bool = True
) -> html.Div:
    """Flat controls div; embed inside an accordion item or modal column.

    *mode* — ``"direction"``, ``"rotation"``, or ``None`` (switchable).
    When fixed, the mode radio is hidden but still in the DOM so callbacks fire.
    """
    p = prefix
    mode_initial = mode if mode is not None else "direction"
    mode_radio_style = {"display": "none"} if mode is not None else {}
    phi_initial = {} if mode == "rotation" else {"display": "none"}

    mode_row = html.Div(
        formgen.form_row(
            "mode",
            dbc.RadioItems(
                id=f"{p}-mode-radio",
                options=[
                    {"label": "Direction", "value": "direction"},
                    {"label": "Rotation", "value": "rotation"},
                ],
                value=mode_initial,
                inline=True,
                inputStyle=styles.RADIO_INLINE_INPUT,
                labelStyle=styles.RADIO_INLINE_LABEL,
            ),
            "Direction: output a unit vector. "
            "Rotation: output zxz Euler angles (phi, theta, psi).",
            label_text="Mode",
            label_id=f"{p}-mode-lbl",
        ),
        style=mode_radio_style,
    )

    dir_section = [
        formgen.form_row(
            "dir_x",
            dbc.Input(id=f"{p}-dir-x", type="number", value=0.0, step=0.001, debounce=True,
                      style=styles.FORM_COMPACT_INPUT),
            "x component of the direction vector.",
            label_text="x", label_id=f"{p}-dir-x-lbl",
        ),
        formgen.form_row(
            "dir_y",
            dbc.Input(id=f"{p}-dir-y", type="number", value=0.0, step=0.001, debounce=True,
                      style=styles.FORM_COMPACT_INPUT),
            "y component of the direction vector.",
            label_text="y", label_id=f"{p}-dir-y-lbl",
        ),
        formgen.form_row(
            "dir_z",
            dbc.Input(id=f"{p}-dir-z", type="number", value=1.0, step=0.001, debounce=True,
                      style=styles.FORM_COMPACT_INPUT),
            "z component of the direction vector.",
            label_text="z", label_id=f"{p}-dir-z-lbl",
        ),
        html.Div(id=f"{p}-dir-status", style=styles.HINT),
        html.Div(
            [
                html.Hr(style={"margin": "0.4rem 0"}),
                formgen.form_row(
                    "inplane",
                    dcc.Slider(
                        id=f"{p}-inplane",
                        min=0, max=360, step=1, value=0,
                        marks={0: "0°", 90: "90°", 180: "180°", 270: "270°", 360: "360°"},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    "In-plane rotation φ in degrees. Governs how the particle frame is"
                    " rotated around its own z-axis; does not move the picked direction.",
                    label_text="φ (in-plane)", label_id=f"{p}-inplane-lbl",
                ),
            ],
            id=f"{p}-inplane-wrap",
            style=phi_initial,
        ),
    ]

    if show_structure:
        struct_section = [
            html.Hr(style={"margin": "0.5rem 0"}),
            get_path_field(
                f"{p}-struct-path", mode="open", kind="volume",
                extensions=(".em", ".mrc", ".rec", ".map"),
                placeholder="Select volume file…",
            ),
            formgen.form_row(
                "struct_level",
                dbc.Input(id=f"{p}-struct-level", type="number", placeholder="auto",
                          debounce=True, style=styles.FORM_COMPACT_INPUT),
                "Isosurface level for marching cubes. Leave blank to use the volume mean.",
                label_text="Level", label_id=f"{p}-struct-level-lbl", truly_optional=True,
            ),
            formgen.form_row(
                "struct_binning",
                dbc.Input(id=f"{p}-struct-binning", type="number", value=1, min=1, step=1,
                          debounce=True, style=styles.FORM_COMPACT_INPUT),
                "Bin the volume before meshing to reduce memory and speed up rendering.",
                label_text="Binning", label_id=f"{p}-struct-binning-lbl",
            ),
            dbc.Button(
                "Load structure", id=f"{p}-struct-load-btn",
                color=styles.BTN_PRIMARY, size="sm", style={"width": "100%"},
            ),
            html.Div(id=f"{p}-struct-status", style=styles.HINT),
        ]
    else:
        struct_section = []

    options_section = [
        html.Hr(style={"margin": "0.5rem 0"}),
        formgen.form_row(
            "ref_toggle",
            dbc.Checklist(
                id=f"{p}-ref-toggle",
                options=[{"label": "", "value": "show"}],
                value=["show"],
                switch=True,
                inputStyle={"cursor": "pointer"},
            ),
            "Show a reference z-axis vector for orientation comparison.",
            label_text="Reference", label_id=f"{p}-ref-lbl",
        ),
    ]

    result_section = [
        html.Hr(style={"margin": "0.5rem 0"}),
        dcc.Textarea(
            id=f"{p}-result-text",
            readOnly=True,
            value="[0.0000, 0.0000, 1.0000]",
            style={
                "width": "100%", "height": "48px",
                "fontFamily": "monospace", "fontSize": styles.FONT_SM,
                "resize": "none", "border": "1px solid var(--bs-border-color)",
                "borderRadius": "4px", "padding": "4px 6px",
                "background": "transparent", "color": "inherit",
            },
        ),
        dcc.Clipboard(
            target_id=f"{p}-result-text", id=f"{p}-copy-btn",
            title="Copy result to clipboard",
            style={"display": "block", "textAlign": "right", "fontSize": styles.FONT_SM,
                   "marginBottom": "0.4rem"},
        ),
        formgen.form_row(
            "publish_name",
            dbc.Input(id=f"{p}-publish-name", type="text", placeholder="my_direction",
                      debounce=True, style=styles.FORM_COMPACT_INPUT),
            "Variable name under which to publish the result to the console.",
            label_text="Name", label_id=f"{p}-publish-name-lbl",
        ),
        dbc.Button(
            "Publish", id=f"{p}-publish-btn",
            color=styles.BTN_SECONDARY, size="sm", style={"width": "100%"},
        ),
        html.Div(id=f"{p}-publish-status", style=styles.HINT),
    ]

    stores = [
        dcc.Store(id=f"{p}-value", data=[0.0, 0.0, 1.0]),
        dcc.Store(id=f"{p}-dir-store", data=[0.0, 0.0, 1.0]),
        dcc.Store(id=f"{p}-dir-click-data", data=None),
        dcc.Store(id=f"{p}-dir-type-data", data=None),
        dcc.Store(id=f"{p}-struct-mesh-store", data=None),
    ]

    return html.Div([
        *stores,
        mode_row,
        *dir_section,
        *struct_section,
        *options_section,
        *result_section,
    ])


def get_orientation_picker_graph(prefix: str, *, height: str = "400px") -> dcc.Graph:
    """dcc.Graph for the sphere visualisation; embed in the main area."""
    return dcc.Graph(
        id=f"{prefix}-sphere-graph",
        style={"height": height},
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )


def get_orientation_picker_panel(
    prefix: str, *,
    mode: str | None = None,
    show_structure: bool = True,
    height: str = "400px",
) -> dbc.Row:
    """Self-contained two-column panel; embed in a modal body."""
    return dbc.Row(
        [
            dbc.Col(
                get_orientation_picker_controls(prefix, mode=mode, show_structure=show_structure),
                width=4,
                style={"overflowY": "auto", "maxHeight": height},
            ),
            dbc.Col(get_orientation_picker_graph(prefix, height=height), width=8),
        ],
        className="g-0",
    )


# ── Callback factory ───────────────────────────────────────────────────────────


def register_orientation_picker_callbacks(
    app, prefix: str, *, mode: str | None = None, show_structure: bool = True
) -> None:
    """Register all callbacks for one picker instance at *prefix*.

    Raises RuntimeError if called twice for the same (app, prefix) pair.
    """
    app_key = id(app)
    registered = _REGISTERED.setdefault(app_key, set())
    if prefix in registered:
        raise RuntimeError(
            f"register_orientation_picker_callbacks already called for "
            f"prefix={prefix!r} on this app instance."
        )
    registered.add(prefix)
    p = prefix

    @app.callback(
        Output(f"{p}-inplane-wrap", "style"),
        Input(f"{p}-mode-radio", "value"),
    )
    def _toggle_inplane(m):
        return {} if m == "rotation" else {"display": "none"}

    @app.callback(
        Output(f"{p}-ref-toggle", "value"),
        Input(f"{p}-mode-radio", "value"),
    )
    def _set_ref_default(m):
        return ["show"] if m != "rotation" else []

    @app.callback(
        Output(f"{p}-dir-click-data", "data"),
        Output(f"{p}-dir-x", "value"),
        Output(f"{p}-dir-y", "value"),
        Output(f"{p}-dir-z", "value"),
        Input(f"{p}-sphere-graph", "clickData"),
        prevent_initial_call=True,
    )
    def _click_dir(click_data):
        if not click_data:                                                      # 1
            raise PreventUpdate
        pts = click_data.get("points", [])                                      # 2
        if not pts or pts[0].get("curveNumber") not in {0, 1}:                  # 3
            raise PreventUpdate
        pt = pts[0]                                                             # 4
        raw = [pt.get("x", 0.0), pt.get("y", 0.0), pt.get("z", 0.0)]          # 5
        try:                                                                    # 6
            nearest = _nearest_sphere_point(_SPHERE_PTS, raw)
        except ValueError:
            return {"dir": None, "status": "Click rejected: zero vector."}, no_update, no_update, no_update
        return (                                                                 # 7
            {"dir": nearest.tolist(), "status": ""},
            round(float(nearest[0]), 6),
            round(float(nearest[1]), 6),
            round(float(nearest[2]), 6),
        )

    @app.callback(
        Output(f"{p}-dir-type-data", "data"),
        Input(f"{p}-dir-x", "value"),
        Input(f"{p}-dir-y", "value"),
        Input(f"{p}-dir-z", "value"),
        prevent_initial_call=True,
    )
    def _type_dir(x, y, z):
        try:                                                                    # 1
            d = _normalize([x or 0.0, y or 0.0, z or 0.0])
        except ValueError:
            return {"dir": None, "status": "Direction must not be zero."}
        return {"dir": d.tolist(), "status": ""}                                # 2

    @app.callback(
        Output(f"{p}-dir-store", "data"),
        Output(f"{p}-dir-status", "children"),
        Input(f"{p}-dir-click-data", "data"),
        Input(f"{p}-dir-type-data", "data"),
        prevent_initial_call=True,
    )
    def _merge_dir(click_data, type_data):
        trig = ctx.triggered_id                                                 # 1
        source = click_data if trig == f"{p}-dir-click-data" else type_data    # 2
        if source is None or source.get("dir") is None:                        # 3
            raise PreventUpdate
        return source["dir"], source.get("status", "")                         # 4

    if show_structure:
        @app.callback(
            Output(f"{p}-struct-mesh-store", "data"),
            Output(f"{p}-struct-status", "children"),
            Input(f"{p}-struct-load-btn", "n_clicks"),
            State({"type": "path-input", "owner": f"{p}-struct-path"}, "value"),
            State(f"{p}-struct-level", "value"),
            State(f"{p}-struct-binning", "value"),
            prevent_initial_call=True,
        )
        def _load_struct(_n, path, level, binning):
            if not path:                                                        # 1
                return no_update, "No file selected."
            try:                                                                # 2
                result = run_operation(
                    _struct_load_fn, {"path": path, "level": level, "binning": binning}
                )
            except Exception as exc:
                return no_update, f"Error: {exc}"
            if result is None:                                                  # 3
                return no_update, "Load failed."
            return result, f"Loaded {pathlib.Path(path).name}: {result['n_verts']:,} vertices."  # 4

    @app.callback(
        Output(f"{p}-sphere-graph", "figure"),
        Input(f"{p}-dir-store", "data"),
        Input(f"{p}-mode-radio", "value"),
        Input(f"{p}-inplane", "value"),
        Input(f"{p}-ref-toggle", "value"),
        Input(f"{p}-struct-mesh-store", "data"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
    )
    def _render_figure(dir_data, m, phi, ref_val, mesh_data, gs):
        return _build_sphere_fig(dir_data, m, phi, ref_val, mesh_data, gs, p)

    @app.callback(
        Output(f"{p}-value", "data"),
        Input(f"{p}-dir-store", "data"),
        Input(f"{p}-mode-radio", "value"),
        Input(f"{p}-inplane", "value"),
    )
    def _update_value(dir_data, m, phi):
        try:                                                                    # 1
            d = _normalize(dir_data or [0.0, 0.0, 1.0])
        except ValueError:
            return [0.0, 0.0, 1.0]
        if (m or "direction") == "rotation":                                    # 2
            return _compute_angles(d, float(phi or 0.0)).tolist()
        return d.tolist()                                                       # 3

    @app.callback(
        Output(f"{p}-result-text", "value"),
        Input(f"{p}-dir-store", "data"),
        Input(f"{p}-mode-radio", "value"),
        Input(f"{p}-inplane", "value"),
    )
    def _update_result(dir_data, m, phi):
        try:                                                                    # 1
            d = _normalize(dir_data or [0.0, 0.0, 1.0])
        except ValueError:
            return "Error: zero vector"
        if (m or "direction") == "rotation":                                    # 2
            angles = _compute_angles(d, float(phi or 0.0))
            return f"[{angles[0]:.4f}, {angles[1]:.4f}, {angles[2]:.4f}]"
        return f"[{d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f}]"                        # 3

    @app.callback(
        Output(f"{p}-publish-status", "children"),
        Input(f"{p}-publish-btn", "n_clicks"),
        State(f"{p}-dir-store", "data"),
        State(f"{p}-mode-radio", "value"),
        State(f"{p}-inplane", "value"),
        State(f"{p}-publish-name", "value"),
        prevent_initial_call=True,
    )
    def _publish(_n, dir_data, m, phi, name):
        if not name or not name.strip():                                        # 1
            return "Enter a variable name."
        if not dir_data:                                                        # 2
            return "Pick a direction first."
        name = name.strip()                                                     # 3
        try:                                                                    # 4
            d = _normalize(dir_data)
        except ValueError:
            return "Invalid direction stored."
        value = (                                                               # 5
            _compute_angles(d, float(phi or 0.0)).tolist()
            if (m or "direction") == "rotation" else d.tolist()
        )
        try:                                                                    # 6
            run_operation(_publish_fn, {"name": name, "value": value})
        except Exception as exc:
            return f"Error: {exc}"
        return f"Published @{name}."                                            # 7
