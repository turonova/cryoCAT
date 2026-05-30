"""Reusable 'rotation builder' panel component.

Lets the user specify a rotation as Euler angles, a quaternion or a rotation
matrix, previews the result as an orientational-distribution point (where the
z-axis points after rotation), and exposes the result as a comma-separated
Euler-angle string (phi,theta,psi, zxz convention, degrees) that
:func:`cryocat.utils.geom.as_rotation` can consume directly.

Public API
----------
get_rotation_builder_sidebar_content(prefix)
    Controls-only layout (type selector, value inputs, stores).  Graph lives
    separately in the main area — use for sidebar/main split layouts.
get_rotation_builder_panel(prefix)
    Self-contained layout: controls left, preview graph right.  Use inside
    modal bodies or any place where a single contained widget is needed.
register_rotation_builder_callbacks(app, prefix)
    Register all callbacks.  Works with both layout variants.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as srot

import dash
from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from cryocat.analysis import visplot
from cryocat.app.components.graphsettings import apply_settings_to_figure


_HINT_STYLE = {"fontSize": "0.85rem", "color": "var(--color9)"}

_ROT_TYPES = [
    {"label": "Euler angles (degrees)", "value": "euler"},
    {"label": "Quaternion (x, y, z, w)", "value": "quaternion"},
    {"label": "Rotation matrix (row-major)", "value": "matrix"},
]

_EULER_ORDERS = ["zxz", "zyz", "xyz", "zxy", "zyx", "xyx", "xzx", "yxy", "yzy"]


# ── Shared sub-widgets ────────────────────────────────────────────────────────

def _num_row(label: str, id_: str, default: float) -> html.Div:
    return html.Div(
        [
            html.Label(
                label,
                style={"width": "70px", "fontSize": "0.82rem", "flexShrink": 0},
            ),
            dcc.Input(
                id=id_,
                type="number",
                value=default,
                debounce=True,
                style={"width": "110px"},
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": "4px"},
    )


def _euler_block(prefix: str) -> html.Div:
    return html.Div(
        [
            _num_row("phi (φ)", f"{prefix}-euler-phi", 0.0),
            _num_row("theta (θ)", f"{prefix}-euler-theta", 0.0),
            _num_row("psi (ψ)", f"{prefix}-euler-psi", 0.0),
        ],
        id=f"{prefix}-euler-block",
    )


def _quat_block(prefix: str) -> html.Div:
    return html.Div(
        [
            _num_row("x", f"{prefix}-quat-x", 0.0),
            _num_row("y", f"{prefix}-quat-y", 0.0),
            _num_row("z", f"{prefix}-quat-z", 0.0),
            _num_row("w", f"{prefix}-quat-w", 1.0),
        ],
        id=f"{prefix}-quat-block",
        style={"display": "none"},
    )


def _matrix_block(prefix: str) -> html.Div:
    """3 × 3 grid of number inputs, initialised to the identity matrix."""
    rows = []
    for i in range(3):
        cells = []
        for j in range(3):
            cells.append(
                dcc.Input(
                    id=f"{prefix}-mat-{i}{j}",
                    type="number",
                    value=1.0 if i == j else 0.0,
                    debounce=True,
                    style={"width": "70px", "marginRight": "4px", "fontSize": "0.82rem"},
                )
            )
        rows.append(html.Div(cells, style={"display": "flex", "marginBottom": "4px"}))
    return html.Div(rows, id=f"{prefix}-matrix-block", style={"display": "none"})


def _controls(prefix: str) -> list:
    lbl = {"fontWeight": "bold", "marginBottom": "0.3rem", "fontSize": "0.85rem"}
    return [
        html.Label("Input format:", style=lbl),
        dcc.Dropdown(
            id=f"{prefix}-rot-type",
            options=_ROT_TYPES,
            value="euler",
            clearable=False,
            searchable=False,
            style={"marginBottom": "0.5rem"},
        ),
        html.Div(
            [
                html.Label("Euler convention:", style={**lbl, "marginRight": "8px", "marginBottom": 0}),
                dcc.Dropdown(
                    id=f"{prefix}-euler-order",
                    options=_EULER_ORDERS,
                    value="zxz",
                    clearable=False,
                    searchable=False,
                    style={"width": "100px"},
                ),
            ],
            id=f"{prefix}-euler-order-row",
            style={"display": "flex", "alignItems": "center", "marginBottom": "0.5rem"},
        ),
        _euler_block(prefix),
        _quat_block(prefix),
        _matrix_block(prefix),
        html.Div(
            id=f"{prefix}-rot-status",
            style={**_HINT_STYLE, "marginTop": "0.5rem", "wordBreak": "break-word"},
        ),
        dcc.Store(id=f"{prefix}-rot-euler-store"),
    ]


# ── Public layout functions ───────────────────────────────────────────────────

def get_rotation_builder_sidebar_content(prefix: str) -> html.Div:
    """Controls-only layout: type selector, value inputs, stores.

    The preview graph is NOT included — place :func:`get_rotation_builder_graph`
    in the main content area separately.

    Parameters
    ----------
    prefix : str
        Unique ID prefix.  Must match the prefix passed to
        :func:`register_rotation_builder_callbacks`.
    """
    return html.Div(_controls(prefix))


def get_rotation_builder_graph(prefix: str) -> dcc.Graph:
    """Preview graph only — companion to :func:`get_rotation_builder_sidebar_content`."""
    return dcc.Graph(id=f"{prefix}-rot-preview", style={"height": "400px"})


def get_rotation_builder_panel(prefix: str) -> html.Div:
    """Self-contained panel: controls (left) and preview graph (right).

    Suitable for embedding in a modal body or any standalone context.

    Parameters
    ----------
    prefix : str
        Unique ID prefix.
    """
    return html.Div(
        dbc.Row(
            [
                dbc.Col(html.Div(_controls(prefix)), width=5),
                dbc.Col(
                    dcc.Graph(id=f"{prefix}-rot-preview", style={"height": "340px"}),
                    width=7,
                ),
            ],
            className="g-2",
        ),
        id=f"{prefix}-rot-panel",
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_rotation_builder_callbacks(app: dash.Dash, prefix: str) -> None:
    """Register all callbacks for the rotation-builder identified by *prefix*.

    Parameters
    ----------
    app : dash.Dash
        The Dash application instance.
    prefix : str
        Must match the prefix passed to the layout function.
    """

    @app.callback(
        Output(f"{prefix}-euler-block", "style"),
        Output(f"{prefix}-quat-block", "style"),
        Output(f"{prefix}-matrix-block", "style"),
        Output(f"{prefix}-euler-order-row", "style"),
        Input(f"{prefix}-rot-type", "value"),
    )
    def _toggle_blocks(rot_type):
        show = {"display": "block"}
        hide = {"display": "none"}
        euler_row_show = {"display": "flex", "alignItems": "center", "marginBottom": "0.5rem"}
        return (
            show if rot_type == "euler" else hide,
            show if rot_type == "quaternion" else hide,
            show if rot_type == "matrix" else hide,
            euler_row_show if rot_type == "euler" else hide,
        )

    mat_inputs = [Input(f"{prefix}-mat-{i}{j}", "value") for i in range(3) for j in range(3)]

    @app.callback(
        Output(f"{prefix}-rot-euler-store", "data"),
        Output(f"{prefix}-rot-preview", "figure"),
        Output(f"{prefix}-rot-status", "children"),
        Input(f"{prefix}-rot-type", "value"),
        Input(f"{prefix}-euler-phi", "value"),
        Input(f"{prefix}-euler-theta", "value"),
        Input(f"{prefix}-euler-psi", "value"),
        Input(f"{prefix}-quat-x", "value"),
        Input(f"{prefix}-quat-y", "value"),
        Input(f"{prefix}-quat-z", "value"),
        Input(f"{prefix}-quat-w", "value"),
        *mat_inputs,
        Input(f"{prefix}-euler-order", "value"),
        State("graph-settings-store", "data"),
        prevent_initial_call=True,
    )
    def _update(
        rot_type,
        phi, theta, psi,
        qx, qy, qz, qw,
        m00, m01, m02, m10, m11, m12, m20, m21, m22,
        euler_order, gs,
    ):
        try:
            order = euler_order or "zxz"
            if rot_type == "euler":
                if any(v is None for v in (phi, theta, psi)):
                    raise PreventUpdate
                rot = srot.from_euler(order, [phi, theta, psi], degrees=True)
            elif rot_type == "quaternion":
                if any(v is None for v in (qx, qy, qz, qw)):
                    raise PreventUpdate
                rot = srot.from_quat([qx, qy, qz, qw])
            else:
                mvals = [m00, m01, m02, m10, m11, m12, m20, m21, m22]
                if any(v is None for v in mvals):
                    raise PreventUpdate
                rot = srot.from_matrix(np.array(mvals, dtype=float).reshape(3, 3))

            euler_out = rot.as_euler("zxz", degrees=True)
            euler_str = f"{euler_out[0]:.4f},{euler_out[1]:.4f},{euler_out[2]:.4f}"

            # Show where the z-axis maps to after rotation
            direction = rot.apply(np.array([[0.0, 0.0, 1.0]]))
            fig = visplot.plot_orientational_distribution(
                coordinates=direction, projection="stereo"
            )
            fig_dict = fig.to_plotly_json()
            if gs:
                fig_dict = apply_settings_to_figure(fig_dict, gs)
            fig_dict.setdefault("layout", {}).update({
                "margin": {"l": 10, "r": 10, "t": 30, "b": 10},
                "uirevision": f"{prefix}-rot-preview",
            })
            return euler_str, go.Figure(fig_dict), f"Euler zxz (deg): {euler_str}"

        except PreventUpdate:
            raise
        except Exception as exc:
            return no_update, go.Figure(), f"Error: {exc}"
