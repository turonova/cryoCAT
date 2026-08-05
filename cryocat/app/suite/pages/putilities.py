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
from cryocat.utils.geom import generate_angles
from cryocat.app.pageshell import page_shell, sidebar_accordion
from cryocat.utils.classutils import GuiEntry


_OUTPUT_AREA_ID = "util-output-area"

_WEDGE_ID_TYPE = "wedge-util-param"


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
                title=b.label,
                item_id=f"util-acc-{b.fn.__name__}",
            )
            for b in builders
        ]

    return [
        sidebar_accordion(items, active_item=[f"util-acc-{b.fn.__name__}" for b in builders])
        if builders
        else items[0],
    ]


def _main(builders: list[GuiEntry]) -> list:
    if not builders:
        body = html.P(
            "No standalone builder tools registered.",
            style={"color": "grey"},
        )
    else:
        body = html.Div(
            id=_OUTPUT_AREA_ID,
            children=html.P(
                "Run a builder from the sidebar to display its result here.",
                style={"color": "grey"},
            ),
        )

    return [body]


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


def register_callbacks(app) -> None:
    from cryocat.analysis import visplot

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
