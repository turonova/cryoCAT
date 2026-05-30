"""Utilities page — standalone builder tools discovered via ``@gui_exposed``.

Every function decorated with ``@gui_exposed(category="builder", standalone=True)``
appears here as its own panel.  Adding a new standalone builder requires only the
decorator on the function; the page discovers it automatically via
:func:`~cryocat.app.apputils.iter_standalone_builders`.

Layout mirrors the other suite pages: a sticky sidebar on the left holds the
form controls for each tool; the main column on the right shows the corresponding
visualisation(s).

Contract: exposes ``layout`` (attribute) and ``register_callbacks(app)``.
"""

import numpy as np
import plotly.graph_objects as go

import dash
from dash import html, dcc, Input, Output, State, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from cryocat.app import formgen
from cryocat.app.apputils import iter_standalone_builders, generate_kwargs, run_operation
from cryocat.app.components.anglesbuilder import (
    get_angles_builder_sidebar_content,
    register_angles_builder_callbacks,
    _inplane_figure,
    _ID_TYPE as _ANGLES_ID_TYPE,
)
from cryocat.app.components.graphsettings import apply_settings_to_figure
from cryocat.utils.geom import generate_angles
from cryocat.utils.wedgeutils import generate_wedge_mask


_WEDGE_ID_TYPE = "wedge-util-param"

_WEDGE_BUILDER = {
    "id": "generate_wedge_mask",
    "label": "Generate wedge mask",
    "fn": generate_wedge_mask,
    "preview": None,
}


def _all_builders() -> list:
    return list(iter_standalone_builders()) + [_WEDGE_BUILDER]


# ── Sidebar helpers ────────────────────────────────────────────────────────────

_HINT_STYLE = {"fontSize": "0.85rem", "color": "var(--color9)"}
_LABEL_STYLE = {"fontSize": "0.85rem", "flex": "0 0 45%", "marginBottom": "0"}
_ROW_STYLE = {"display": "flex", "alignItems": "center", "gap": "0.5rem", "marginBottom": "0.35rem"}
_INPUT_WRAPPER = {"flex": "1 1 auto", "minWidth": "0"}


def _sidebar_content(builder: dict) -> html.Div:
    prefix = f"util-{builder['id']}"
    if builder["id"] == "generate_angles":
        return get_angles_builder_sidebar_content(prefix, preview_btn=True)
    if builder["id"] == "generate_wedge_mask":
        return _wedge_mask_sidebar_content(prefix)
    return html.Div("Controls not yet implemented.", style={"color": "grey"})


def _wedge_mask_sidebar_content(prefix: str) -> html.Div:
    form_rows = formgen.build_form(
        generate_wedge_mask,
        id_type=_WEDGE_ID_TYPE,
        id_extra={"builder": prefix},
    )
    return html.Div(
        [
            html.Div(form_rows, style={"marginBottom": "0.75rem"}),
            dbc.Input(
                id=f"{prefix}-output-path",
                type="text",
                placeholder="Output path (e.g. /path/to/wedge_mask.em)",
                size="sm",
                style={"marginBottom": "0.4rem"},
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
                style={**_HINT_STYLE, "marginTop": "0.25rem", "wordBreak": "break-word"},
            ),
            dcc.Store(id=f"{prefix}-params"),
        ],
    )


def _main_content(builder: dict) -> html.Div:
    prefix = f"util-{builder['id']}"
    if builder["id"] == "generate_angles":
        graphs = dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id=f"{prefix}-preview", style={"height": "460px"}),
                    width=6,
                ),
                dbc.Col(
                    dcc.Graph(id=f"{prefix}-inplane-preview", style={"height": "460px"}),
                    width=6,
                ),
            ],
            className="g-1",
        )
    else:
        graphs = html.Div()
    return html.Div(graphs, id=f"util-main-{builder['id']}")


# ── Layout builders ────────────────────────────────────────────────────────────


def _sidebar(builders: list) -> dbc.Col:
    if not builders:
        items = [
            html.P(
                "No standalone builder tools registered.",
                style={"color": "grey", "padding": "0.5rem"},
            )
        ]
    else:
        items = [
            dbc.AccordionItem(
                _sidebar_content(b),
                title=b["label"],
                item_id=f"util-acc-{b['id']}",
            )
            for b in builders
        ]

    return dbc.Col(
        html.Div(
            [
                dbc.Accordion(
                    items,
                    always_open=True,
                    active_item=[f"util-acc-{b['id']}" for b in builders],
                )
                if builders
                else items[0],
            ],
            className="sidebar",
            style={
                "padding": "0.5rem",
                "overflowY": "auto",
                "height": "100vh",
                "display": "flex",
                "flexDirection": "column",
            },
        ),
        width=3,
        style={
            "margin": "0",
            "padding": "0",
            "height": "100vh",
            "position": "sticky",
            "top": "0px",
        },
    )


def _main(builders: list) -> dbc.Col:
    if not builders:
        body = html.P(
            "No standalone builder tools registered.",
            style={"color": "grey"},
        )
    else:
        body = html.Div([_main_content(b) for b in builders])

    return dbc.Col(
        html.Div(
            [
                body,
            ],
            style={"padding": "0.5rem"},
        ),
        width=9,
        style={"margin": "0", "padding": "0"},
    )


def _build_layout() -> html.Div:
    builders = _all_builders()
    return html.Div(
        [
            dbc.Row(
                [_sidebar(builders), _main(builders)],
                className="g-0",
                style={"margin": "0", "padding": "0"},
            )
        ],
        style={"margin": "0", "padding": "0"},
    )


layout = _build_layout()


# ── Callbacks ──────────────────────────────────────────────────────────────────


def _err_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(annotations=[{"text": msg, "showarrow": False, "xref": "paper", "yref": "paper"}])
    return fig


def register_callbacks(app) -> None:
    from cryocat.analysis import visplot

    for b in _all_builders():
        prefix = f"util-{b['id']}"
        if b["id"] == "generate_angles":
            # Register _collect_params and _create from anglesbuilder;
            # skip the built-in single-graph preview so we own both outputs.
            register_angles_builder_callbacks(app, prefix, skip_preview=True)

            @app.callback(
                Output(f"{prefix}-angles", "data"),
                Output(f"{prefix}-preview", "figure"),
                Output(f"{prefix}-inplane-preview", "figure"),
                Input(f"{prefix}-preview-btn", "n_clicks"),
                State({"type": _ANGLES_ID_TYPE, "builder": prefix, "param": ALL, "tag": ALL}, "value"),
                State({"type": _ANGLES_ID_TYPE, "builder": prefix, "param": ALL, "tag": ALL}, "id"),
                State("graph-settings-store", "data"),
                prevent_initial_call=True,
            )
            def _preview(n_clicks, values, ids, gs, _prefix=prefix):
                if not n_clicks:
                    raise PreventUpdate

                params = generate_kwargs(ids, values) if (values and ids) else {}
                if params.get("cone_angle") is None or params.get("cone_sampling") is None:
                    msg = "Set cone_angle and cone_sampling first."
                    return dash.no_update, _err_fig(msg), _err_fig(msg)

                try:
                    kwargs = {k: v for k, v in params.items() if v is not None}
                    angles = generate_angles(**kwargs)
                except Exception as exc:
                    msg = f"Error generating angles: {exc}"
                    return dash.no_update, _err_fig(msg), _err_fig(msg)

                angles_list = angles.tolist()

                n_phi = len(np.unique(np.round(angles[:, 0], 8)))
                n_cone = len(angles) // n_phi if n_phi > 0 else len(angles)

                try:
                    fig1 = visplot.plot_rotation_normals(angles)
                    fig1_dict = fig1.to_plotly_json()
                    if gs:
                        fig1_dict = apply_settings_to_figure(fig1_dict, gs)
                    fig1_dict.setdefault("layout", {}).update({
                        "uirevision": f"{_prefix}-preview",
                        "title": {"text": f"Cone sampling — {n_cone} angles", "font": {"size": 12}},
                        "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
                    })
                    sphere_fig = go.Figure(fig1_dict)
                except Exception as exc:
                    sphere_fig = _err_fig(f"Sphere plot error: {exc}")

                try:
                    inplane_fig = _inplane_figure(angles, gs)
                except Exception as exc:
                    inplane_fig = _err_fig(f"Inplane plot error: {exc}")

                return angles_list, sphere_fig, inplane_fig

        elif b["id"] == "generate_wedge_mask":
            _register_wedge_mask_callbacks(app, prefix)


def _register_wedge_mask_callbacks(app, prefix: str) -> None:
    @app.callback(
        Output(f"{prefix}-params", "data"),
        Input({"type": _WEDGE_ID_TYPE, "builder": prefix, "param": ALL, "tag": ALL}, "value"),
        State({"type": _WEDGE_ID_TYPE, "builder": prefix, "param": ALL, "tag": ALL}, "id"),
    )
    def _collect_params(values, ids):
        if not values or not ids:
            raise PreventUpdate
        return generate_kwargs(ids, values)

    @app.callback(
        Output(f"{prefix}-status", "children"),
        Input(f"{prefix}-generate", "n_clicks"),
        State(f"{prefix}-params", "data"),
        State(f"{prefix}-output-path", "value"),
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
            run_operation(generate_wedge_mask, kwargs)
            msg = f"Wedge mask generated"
            if out_path:
                msg += f" → {out_path}"
            return msg
        except Exception as exc:
            return f"Error: {exc}"
