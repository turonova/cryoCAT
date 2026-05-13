from dash import html, dcc, callback, Input, Output, State, no_update, ctx, ALL
import dash_bootstrap_components as dbc

GRAPH_SETTINGS_DEFAULTS = {
    "font_family": "Arial",
    "font_size": 12,
    "marker_size": 6,
    "line_width": 2,
    "line_dash": "solid",
    "discrete_palette": "Monet",
    "continuous_palette": "Viridis",
    "bg_color": "white",
}

_FONT_FAMILIES = ["Arial", "Helvetica", "Courier New", "Times New Roman", "Verdana"]
_LINE_DASHES = [
    {"label": "Solid", "value": "solid"},
    {"label": "Dashed", "value": "dash"},
    {"label": "Dotted", "value": "dot"},
    {"label": "Dash-dot", "value": "dashdot"},
    {"label": "Long dash", "value": "longdash"},
]
_DISCRETE_PALETTES = ["Monet", "Plotly", "D3", "G10", "Vivid", "Bold", "Pastel", "Safe"]
_CONTINUOUS_PALETTES = ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Jet", "Hot", "Blues", "RdBu"]
_BG_COLORS = [
    {"label": "White", "value": "white"},
    {"label": "Light grey", "value": "#f5f5f5"},
    {"label": "Dark", "value": "#1e1e1e"},
]


def _setting_row(label, control):
    return dbc.Row(
        [
            dbc.Col(html.Label(label, style={"fontSize": "0.85rem"}), width=5,
                    className="d-flex align-items-center"),
            dbc.Col(control, width=7),
        ],
        className="mb-2",
    )


def get_graph_settings_components():
    """Global store + modal dialog; add to app.layout."""
    return [
        dcc.Store(id="graph-settings-store", data=GRAPH_SETTINGS_DEFAULTS),
        dbc.Modal(
            id="graph-settings-modal",
            is_open=False,
            centered=True,
            size="md",
            class_name="graph-settings-modal",
            children=[
                dbc.ModalHeader(dbc.ModalTitle("Graph Settings")),
                dbc.ModalBody([
                    html.P(
                        "Font, background, marker and line settings apply immediately to all existing graphs. "
                        "Color palette is used for new graphs.",
                        style={"fontSize": "0.8rem", "color": "grey", "marginBottom": "1rem"},
                    ),
                    _setting_row("Font family", dcc.Dropdown(
                        id="gs-font-family",
                        options=_FONT_FAMILIES,
                        value=GRAPH_SETTINGS_DEFAULTS["font_family"],
                        clearable=False,
                    )),
                    _setting_row("Font size", dbc.Input(
                        id="gs-font-size",
                        type="number",
                        value=GRAPH_SETTINGS_DEFAULTS["font_size"],
                        min=6, max=30, step=1,
                    )),
                    _setting_row("Marker size", dbc.Input(
                        id="gs-marker-size",
                        type="number",
                        value=GRAPH_SETTINGS_DEFAULTS["marker_size"],
                        min=1, max=30, step=1,
                    )),
                    _setting_row("Line width", dbc.Input(
                        id="gs-line-width",
                        type="number",
                        value=GRAPH_SETTINGS_DEFAULTS["line_width"],
                        min=0.5, max=10, step=0.5,
                    )),
                    _setting_row("Line style", dcc.Dropdown(
                        id="gs-line-dash",
                        options=_LINE_DASHES,
                        value=GRAPH_SETTINGS_DEFAULTS["line_dash"],
                        clearable=False,
                    )),
                    _setting_row("Discrete palette", dcc.Dropdown(
                        id="gs-discrete-palette",
                        options=_DISCRETE_PALETTES,
                        value=GRAPH_SETTINGS_DEFAULTS["discrete_palette"],
                        clearable=False,
                    )),
                    _setting_row("Continuous palette", dcc.Dropdown(
                        id="gs-continuous-palette",
                        options=_CONTINUOUS_PALETTES,
                        value=GRAPH_SETTINGS_DEFAULTS["continuous_palette"],
                        clearable=False,
                    )),
                    _setting_row("Background", dcc.Dropdown(
                        id="gs-bg-color",
                        options=_BG_COLORS,
                        value=GRAPH_SETTINGS_DEFAULTS["bg_color"],
                        clearable=False,
                    )),
                ]),
                dbc.ModalFooter([
                    html.Span(id="gs-status",
                              style={"fontSize": "0.8rem", "color": "grey", "marginRight": "auto"}),
                    dbc.Button("Apply Changes", id="gs-apply-btn", color="primary",
                               className="me-2", n_clicks=0),
                    dbc.Button("Close", id="gs-close-btn", color="secondary", n_clicks=0),
                ]),
            ],
        ),
    ]


def get_graph_settings_button(prefix: str):
    """Button to embed in the plot panel for a given prefix."""
    return dbc.Button(
        "Graph Settings",
        id={"type": "open-graph-settings-btn", "index": prefix},
        color="light",
        style={"width": "100%"},
    )


def register_graph_settings_callbacks():
    @callback(
        Output("graph-settings-modal", "is_open"),
        Input({"type": "open-graph-settings-btn", "index": ALL}, "n_clicks"),
        Input("gs-close-btn", "n_clicks"),
        State("graph-settings-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_modal(open_clicks_list, close_clicks, is_open):
        triggered = ctx.triggered_id
        if isinstance(triggered, dict) and triggered.get("type") == "open-graph-settings-btn":
            # Callback also fires when a new matching component appears in the DOM (n_clicks=0).
            # Only open on an actual user click (value > 0).
            triggered_value = ctx.triggered[0]["value"] if ctx.triggered else 0
            return True if triggered_value else no_update
        if triggered == "gs-close-btn":
            return False
        return no_update

    @callback(
        Output("graph-settings-store", "data"),
        Output("gs-status", "children"),
        Input("gs-apply-btn", "n_clicks"),
        State("gs-font-family", "value"),
        State("gs-font-size", "value"),
        State("gs-marker-size", "value"),
        State("gs-line-width", "value"),
        State("gs-line-dash", "value"),
        State("gs-discrete-palette", "value"),
        State("gs-continuous-palette", "value"),
        State("gs-bg-color", "value"),
        prevent_initial_call=True,
    )
    def apply_settings(_, font_family, font_size, marker_size, line_width, line_dash,
                       discrete_palette, continuous_palette, bg_color):
        settings = {
            "font_family": font_family or GRAPH_SETTINGS_DEFAULTS["font_family"],
            "font_size": font_size or GRAPH_SETTINGS_DEFAULTS["font_size"],
            "marker_size": marker_size or GRAPH_SETTINGS_DEFAULTS["marker_size"],
            "line_width": line_width or GRAPH_SETTINGS_DEFAULTS["line_width"],
            "line_dash": line_dash or GRAPH_SETTINGS_DEFAULTS["line_dash"],
            "discrete_palette": discrete_palette or GRAPH_SETTINGS_DEFAULTS["discrete_palette"],
            "continuous_palette": continuous_palette or GRAPH_SETTINGS_DEFAULTS["continuous_palette"],
            "bg_color": bg_color or GRAPH_SETTINGS_DEFAULTS["bg_color"],
        }
        return settings, "Applied."


_DISCRETE_TRACE_TYPES = {"scatter", "scattergl", "scatter3d", "bar", "histogram", "violin", "box"}
_CONTINUOUS_TRACE_TYPES = {"heatmap", "contour", "surface", "densitymapbox"}


def apply_settings_to_figure(fig_dict, settings):
    """Apply settings to a Plotly figure dict in-place. Returns the dict."""
    if not settings or not isinstance(fig_dict, dict):
        return fig_dict

    from cryocat.analysis.visplot import resolve_palette

    layout = fig_dict.setdefault("layout", {})

    if settings.get("font_family") or settings.get("font_size"):
        font = layout.setdefault("font", {})
        if settings.get("font_family"):
            font["family"] = settings["font_family"]
        if settings.get("font_size"):
            font["size"] = settings["font_size"]

    if settings.get("bg_color"):
        layout["paper_bgcolor"] = settings["bg_color"]
        layout["plot_bgcolor"] = settings["bg_color"]

    if settings.get("discrete_palette"):
        palette = resolve_palette(settings["discrete_palette"])
        layout["colorway"] = palette
        discrete_traces = [t for t in fig_dict.get("data", [])
                           if t.get("type", "scatter") in _DISCRETE_TRACE_TYPES]
        for i, trace in enumerate(discrete_traces):
            color = palette[i % len(palette)]
            marker = trace.get("marker", {})
            if not isinstance(marker.get("color"), list):
                trace.setdefault("marker", {})["color"] = color
            line = trace.get("line", {})
            if not isinstance(line.get("color"), list):
                trace.setdefault("line", {})["color"] = color

    if settings.get("continuous_palette"):
        scale = settings["continuous_palette"]
        layout.setdefault("coloraxis", {})["colorscale"] = scale
        for trace in fig_dict.get("data", []):
            if trace.get("type") in _CONTINUOUS_TRACE_TYPES:
                trace["colorscale"] = scale

    marker_size = settings.get("marker_size")
    line_width = settings.get("line_width")
    line_dash = settings.get("line_dash")

    for trace in fig_dict.get("data", []):
        trace_type = trace.get("type", "")
        if trace_type in ("scatter", "scattergl", "scatter3d") and marker_size:
            marker = trace.setdefault("marker", {})
            if isinstance(marker.get("size"), (int, float, type(None))):
                marker["size"] = marker_size
        if trace_type in ("scatter", "scattergl", "scatter3d"):
            if line_width or line_dash:
                line = trace.setdefault("line", {})
                if line_width:
                    line["width"] = line_width
                if line_dash:
                    line["dash"] = line_dash

    return fig_dict
