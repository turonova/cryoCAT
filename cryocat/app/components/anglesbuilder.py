"""Reusable 'angles builder' panel component.

A self-contained Dash panel that exposes ``cryocat.utils.geom.generate_angles``
via a ``formgen``-generated form, renders a live orientational-distribution
preview, and writes the result to a file on demand.

The panel has no knowledge of where it lives (modal or page).  After a
successful write the ``{prefix}-created-path`` store is updated; the host wires
its own callback to react to that event.

Public API
----------
get_angles_builder_sidebar_content(prefix)
    Controls-only layout (form, path, create button, stores). Graph lives separately.
get_angles_builder_panel(prefix)
    Self-contained layout including the preview graph.  Used in modal contexts.
register_angles_builder_callbacks(app, prefix)
    Register all callbacks for the panel (works with both layout variants).
"""

import numpy as np
import plotly.graph_objects as go

import dash
from dash import html, dcc, Input, Output, State, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from cryocat.utils.geom import generate_angles
from cryocat.app import formgen
from cryocat.app.apputils import generate_kwargs, run_operation
from cryocat.app.components.graphsettings import apply_settings_to_figure


_HINT_STYLE = {"fontSize": "0.85rem", "color": "var(--color9)"}

_ID_TYPE = "angles-param"


def _inplane_figure(angles: np.ndarray, gs) -> go.Figure:
    """Polar scatter of inplane (phi) angles + optional graph settings."""
    phi = np.unique(np.round(angles[:, 0], 8))
    fig = go.Figure(
        go.Scatterpolar(
            r=[1.0] * len(phi),
            theta=phi,
            mode="markers",
            marker=dict(size=6, opacity=0.8),
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1.3]),
            angularaxis=dict(
                direction="counterclockwise",
                tickmode="array",
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
            ),
        ),
        showlegend=False,
        title=dict(text=f"Inplane sampling (φ) — {len(phi)} angles", font=dict(size=12)),
        margin=dict(l=50, r=50, t=40, b=50),
    )
    if gs:
        from cryocat.app.components.graphsettings import apply_settings_to_figure
        fig_dict = fig.to_plotly_json()
        fig_dict = apply_settings_to_figure(fig_dict, gs)
        return go.Figure(fig_dict)
    return fig


def get_angles_builder_sidebar_content(prefix: str, preview_btn: bool = False) -> html.Div:
    """Form controls, output path, create button and stores — no graph.

    Designed for split sidebar/main layouts where the preview graph lives
    in the main content area.  IDs are identical to those in
    :func:`get_angles_builder_panel`, so :func:`register_angles_builder_callbacks`
    works unchanged with either layout.

    Parameters
    ----------
    prefix : str
        Same prefix that will be passed to :func:`register_angles_builder_callbacks`.
    preview_btn : bool, default=False
        When True, a "Preview" button (``{prefix}-preview-btn``) is added
        between the form and the output-path field so the user can trigger
        the visualisation manually.
    """
    form_rows = formgen.build_form(
        generate_angles,
        id_type=_ID_TYPE,
        id_extra={"builder": prefix},
    )
    preview_button = (
        [
            dbc.Button(
                "Preview",
                id=f"{prefix}-preview-btn",
                color="info",
                size="sm",
                style={"width": "100%", "marginBottom": "0.5rem"},
            )
        ]
        if preview_btn
        else []
    )
    return html.Div(
        [
            html.Div(form_rows, id=f"{prefix}-form", style={"marginBottom": "0.75rem"}),
            *preview_button,
            dbc.Input(
                id=f"{prefix}-output-path",
                type="text",
                placeholder="Output path (e.g. /path/to/angles.csv)",
                size="sm",
                style={"marginBottom": "0.4rem"},
            ),
            dbc.Button(
                "Create angle list",
                id=f"{prefix}-create",
                color="primary",
                size="sm",
                style={"width": "100%", "marginBottom": "0.4rem"},
            ),
            html.Div(
                id=f"{prefix}-status",
                style={
                    **_HINT_STYLE,
                    "marginTop": "0.25rem",
                    "wordBreak": "break-word",
                },
            ),
            dcc.Store(id=f"{prefix}-params"),
            dcc.Store(id=f"{prefix}-angles"),
            dcc.Store(id=f"{prefix}-created-path"),
        ],
    )


def get_angles_builder_panel(prefix: str) -> html.Div:
    """Build the angles-builder panel layout.

    The form rows are generated from :func:`~cryocat.utils.geom.generate_angles`
    via :func:`~cryocat.app.formgen.build_form`.  Every control's id includes
    ``{"builder": prefix}`` so pattern-matching callbacks can scope their inputs
    to this specific panel instance.

    Parameters
    ----------
    prefix : str
        Unique string prefix for all component IDs (e.g. ``"pana-angles-build"``
        or ``"util-generate_angles"``).

    Returns
    -------
    dash.html.Div
        Self-contained layout tree.
    """
    form_rows = formgen.build_form(
        generate_angles,
        id_type=_ID_TYPE,
        id_extra={"builder": prefix},
    )
    return html.Div(
        [
            # Controls come first so they stay visible when graphs are rendered
            html.Div(form_rows, id=f"{prefix}-form", style={"marginBottom": "0.5rem"}),
            dbc.Input(
                id=f"{prefix}-output-path",
                type="text",
                placeholder="Output path (e.g. /path/to/angles.csv)",
                size="sm",
                style={"marginBottom": "0.4rem"},
            ),
            html.Div(
                [
                    dbc.Button(
                        "Create angle list",
                        id=f"{prefix}-create",
                        color="primary",
                        size="sm",
                    ),
                    html.Span(
                        id=f"{prefix}-status",
                        style={
                            "marginLeft": "0.75rem",
                            "fontSize": "0.85rem",
                            "color": "grey",
                            "alignSelf": "flex-end",
                        },
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "0.5rem"},
            ),
            # Two graphs side-by-side (same as Utilities page)
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id=f"{prefix}-preview", style={"height": "320px"}), width=6),
                    dbc.Col(dcc.Graph(id=f"{prefix}-inplane-preview", style={"height": "320px"}), width=6),
                ],
                className="g-1",
            ),
            dcc.Store(id=f"{prefix}-params"),
            dcc.Store(id=f"{prefix}-angles"),
            dcc.Store(id=f"{prefix}-created-path"),
        ],
        id=f"{prefix}-panel",
    )


def register_angles_builder_callbacks(app: dash.Dash, prefix: str, skip_preview: bool = False) -> None:
    """Register all callbacks for the angles-builder panel identified by ``prefix``.

    Parameters
    ----------
    app : dash.Dash
        The Dash application instance.
    prefix : str
        Must match the ``prefix`` passed to :func:`get_angles_builder_panel`.
    skip_preview : bool, default=False
        When True the ``_preview`` callback (which writes to ``{prefix}-preview``
        and ``{prefix}-angles``) is not registered.  Use this when the host page
        registers its own preview callback (e.g. a two-graph layout).
    """

    @app.callback(
        Output(f"{prefix}-params", "data"),
        Input({"type": _ID_TYPE, "builder": prefix, "param": ALL, "tag": ALL}, "value"),
        State({"type": _ID_TYPE, "builder": prefix, "param": ALL, "tag": ALL}, "id"),
    )
    def _collect_params(values, ids):
        if not values or not ids:
            raise PreventUpdate
        return generate_kwargs(ids, values)

    if not skip_preview:
        @app.callback(
            Output(f"{prefix}-angles", "data"),
            Output(f"{prefix}-preview", "figure"),
            Output(f"{prefix}-inplane-preview", "figure"),
            Input(f"{prefix}-params", "data"),
            State("graph-settings-store", "data"),
            prevent_initial_call=True,
        )
        def _preview(params, gs):
            if not params:
                raise PreventUpdate
            if params.get("cone_angle") is None or params.get("cone_sampling") is None:
                raise PreventUpdate

            def _err(msg):
                f = go.Figure()
                f.update_layout(annotations=[{"text": msg, "showarrow": False, "xref": "paper", "yref": "paper"}])
                return f

            try:
                kwargs = {k: v for k, v in params.items() if v is not None}
                angles = generate_angles(**kwargs)
            except Exception as exc:
                err = _err(f"Error: {exc}")
                return dash.no_update, err, err

            from cryocat.analysis import visplot
            n_phi = len(np.unique(np.round(angles[:, 0], 8)))
            n_cone = len(angles) // n_phi if n_phi > 0 else len(angles)

            try:
                fig1 = visplot.plot_rotation_normals(angles)
                fig1_dict = fig1.to_plotly_json()
                fig1_dict = apply_settings_to_figure(fig1_dict, gs)
                fig1_dict.setdefault("layout", {}).update({
                    "uirevision": f"{prefix}-preview",
                    "title": {"text": f"Cone sampling — {n_cone} angles", "font": {"size": 12}},
                    "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
                })
                sphere_fig = go.Figure(fig1_dict)
            except Exception as exc:
                sphere_fig = _err(f"Sphere plot error: {exc}")

            try:
                inplane_fig = _inplane_figure(angles, gs)
            except Exception as exc:
                inplane_fig = _err(f"Inplane plot error: {exc}")

            return angles.tolist(), sphere_fig, inplane_fig

    @app.callback(
        Output(f"{prefix}-status", "children"),
        Output(f"{prefix}-created-path", "data"),
        Input(f"{prefix}-create", "n_clicks"),
        State(f"{prefix}-params", "data"),
        State(f"{prefix}-output-path", "value"),
        prevent_initial_call=True,
    )
    def _create(_n, params, out_path):
        if not out_path:
            return "Specify an output path.", dash.no_update
        if not params or params.get("cone_angle") is None or params.get("cone_sampling") is None:
            return "Set cone_angle and cone_sampling first.", dash.no_update
        try:
            kwargs = {k: v for k, v in params.items() if v is not None}
            angles = run_operation(generate_angles, {**kwargs, "output_path": out_path})
        except Exception as exc:
            return f"Error: {exc}", dash.no_update
        return f"Saved {len(angles)} angles → {out_path}", out_path
