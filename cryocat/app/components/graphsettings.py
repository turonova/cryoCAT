import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State, no_update, ctx, ALL
import dash_bootstrap_components as dbc

from cryocat.app import ids
from cryocat.app.components.paletteloader import get_palette_loader, register_palette_loader_callbacks
from cryocat.app.formgen import make_dropdown

GRAPH_SETTINGS_DEFAULTS = {
    "font_family": "Arial",
    "font_size": 12,
    "marker_size": 6,
    "line_width": 2,
    "line_dash": "solid",
    "discrete_palette": "StarryNight",
    "continuous_palette": "StarryNight",
    "bg_color": "white",
    "palette_is_user_set": False,
}

_FONT_FAMILIES = ["Arial", "Helvetica", "Courier New", "Times New Roman", "Verdana"]
_LINE_DASHES = [
    {"label": "Solid", "value": "solid"},
    {"label": "Dashed", "value": "dash"},
    {"label": "Dotted", "value": "dot"},
    {"label": "Dash-dot", "value": "dashdot"},
    {"label": "Long dash", "value": "longdash"},
]
_BG_COLORS = [
    {"label": "White", "value": "white"},
    {"label": "Light grey", "value": "#f5f5f5"},
    {"label": "Dark", "value": "#1e1e1e"},
]


def _setting_row(label, control):
    return dbc.Row(
        [
            dbc.Col(html.Label(label), width=5,
                    className="d-flex align-items-center"),
            dbc.Col(control, width=7),
        ],
        className="mb-2",
    )


def get_graph_settings_components():
    """Global store + modal dialog; add to app.layout."""
    return [
        dcc.Store(id=ids.GRAPH_SETTINGS_STORE, data=GRAPH_SETTINGS_DEFAULTS),
        dcc.Store(id=ids.GRAPH_PALETTE_SIGNAL, data={
            "discrete_palette": GRAPH_SETTINGS_DEFAULTS["discrete_palette"],
            "continuous_palette": GRAPH_SETTINGS_DEFAULTS["continuous_palette"],
        }),
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
                        style={"color": "grey", "marginBottom": "1rem"},
                    ),
                    _setting_row("Font family", make_dropdown(
                        "gs-font-family",
                        _FONT_FAMILIES,
                        GRAPH_SETTINGS_DEFAULTS["font_family"],
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
                    _setting_row("Line style", make_dropdown(
                        "gs-line-dash",
                        _LINE_DASHES,
                        GRAPH_SETTINGS_DEFAULTS["line_dash"],
                        clearable=False,
                    )),
                    html.Div([
                        html.Label("Discrete palette", style={"marginBottom": "0.2rem"}),
                        get_palette_loader(
                            "gs-discrete-pal", mode="discrete",
                            default=GRAPH_SETTINGS_DEFAULTS["discrete_palette"],
                        ),
                    ], style={"marginBottom": "0.75rem"}),
                    html.Div([
                        html.Label("Continuous palette", style={"marginBottom": "0.2rem"}),
                        get_palette_loader(
                            "gs-continuous-pal", mode="continuous",
                            default=GRAPH_SETTINGS_DEFAULTS["continuous_palette"],
                        ),
                    ], style={"marginBottom": "0.75rem"}),
                    _setting_row("Background", make_dropdown(
                        "gs-bg-color",
                        _BG_COLORS,
                        GRAPH_SETTINGS_DEFAULTS["bg_color"],
                        clearable=False,
                    )),
                ]),
                dbc.ModalFooter([
                    html.Span(id="gs-status",
                              style={"color": "grey", "marginRight": "auto"}),
                    dbc.Button("Apply Changes", id="gs-apply-btn", color="primary",
                               className="me-2", n_clicks=0),
                    dbc.Button("Apply to existing", id="gs-apply-existing-btn", color="secondary",
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
        id={"type": "open-graph-settings-btn", "owner": prefix},
        color="light",
        style={"width": "100%"},
    )


def register_graph_settings_callbacks(app):
    register_palette_loader_callbacks(app, "gs-discrete-pal", mode="discrete")
    register_palette_loader_callbacks(app, "gs-continuous-pal", mode="continuous")

    @app.callback(
        Output(ids.GRAPH_PALETTE_SIGNAL, "data"),
        Input(ids.GRAPH_SETTINGS_STORE, "data"),
        State(ids.GRAPH_PALETTE_SIGNAL, "data"),
        prevent_initial_call=True,
    )
    def _sync_palette_signal(settings, prev_signal):
        s = settings or GRAPH_SETTINGS_DEFAULTS
        new_dis = s.get("discrete_palette", GRAPH_SETTINGS_DEFAULTS["discrete_palette"])
        new_con = s.get("continuous_palette", GRAPH_SETTINGS_DEFAULTS["continuous_palette"])
        prev = prev_signal or {}
        if prev.get("discrete_palette") == new_dis and prev.get("continuous_palette") == new_con:
            return no_update
        return {"discrete_palette": new_dis, "continuous_palette": new_con}

    @app.callback(
        Output("graph-settings-modal", "is_open"),
        Output("gs-font-family", "value"),
        Output("gs-font-size", "value"),
        Output("gs-marker-size", "value"),
        Output("gs-line-width", "value"),
        Output("gs-line-dash", "value"),
        Output("gs-bg-color", "value"),
        Input({"type": "open-graph-settings-btn", "owner": ALL}, "n_clicks"),
        Input("gs-close-btn", "n_clicks"),
        State("graph-settings-modal", "is_open"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        prevent_initial_call=True,
    )
    def toggle_modal(open_clicks_list, close_clicks, is_open, settings):
        _nu = (no_update,) * 6
        triggered = ctx.triggered_id
        if isinstance(triggered, dict) and triggered.get("type") == "open-graph-settings-btn":
            # Callback also fires when a new matching component appears in the DOM (n_clicks=0).
            # Only open on an actual user click (value > 0).
            triggered_value = ctx.triggered[0]["value"] if ctx.triggered else 0
            if not triggered_value:
                return (no_update,) + _nu
            s = settings or GRAPH_SETTINGS_DEFAULTS
            return (
                True,
                s.get("font_family", GRAPH_SETTINGS_DEFAULTS["font_family"]),
                s.get("font_size", GRAPH_SETTINGS_DEFAULTS["font_size"]),
                s.get("marker_size", GRAPH_SETTINGS_DEFAULTS["marker_size"]),
                s.get("line_width", GRAPH_SETTINGS_DEFAULTS["line_width"]),
                s.get("line_dash", GRAPH_SETTINGS_DEFAULTS["line_dash"]),
                s.get("bg_color", GRAPH_SETTINGS_DEFAULTS["bg_color"]),
            )
        if triggered == "gs-close-btn":
            return (False,) + _nu
        return (no_update,) + _nu

    @app.callback(
        Output(ids.GRAPH_SETTINGS_STORE, "data"),
        Output("gs-status", "children"),
        Input("gs-apply-btn", "n_clicks"),
        State("gs-font-family", "value"),
        State("gs-font-size", "value"),
        State("gs-marker-size", "value"),
        State("gs-line-width", "value"),
        State("gs-line-dash", "value"),
        State("gs-discrete-pal-value", "data"),
        State("gs-continuous-pal-value", "data"),
        State("gs-bg-color", "value"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        prevent_initial_call=True,
    )
    def apply_settings(_, font_family, font_size, marker_size, line_width, line_dash,
                       discrete_palette, continuous_palette, bg_color, prev_settings):
        prev = prev_settings or GRAPH_SETTINGS_DEFAULTS
        new_dis = discrete_palette if discrete_palette else prev.get("discrete_palette", GRAPH_SETTINGS_DEFAULTS["discrete_palette"])
        new_con = continuous_palette if continuous_palette else prev.get("continuous_palette", GRAPH_SETTINGS_DEFAULTS["continuous_palette"])
        palette_is_user_set = True if (discrete_palette or continuous_palette) else prev.get("palette_is_user_set", False)
        return {
            "font_family": font_family or GRAPH_SETTINGS_DEFAULTS["font_family"],
            "font_size": font_size or GRAPH_SETTINGS_DEFAULTS["font_size"],
            "marker_size": marker_size or GRAPH_SETTINGS_DEFAULTS["marker_size"],
            "line_width": line_width or GRAPH_SETTINGS_DEFAULTS["line_width"],
            "line_dash": line_dash or GRAPH_SETTINGS_DEFAULTS["line_dash"],
            "discrete_palette": new_dis,
            "continuous_palette": new_con,
            "bg_color": bg_color or GRAPH_SETTINGS_DEFAULTS["bg_color"],
            "palette_is_user_set": palette_is_user_set,
        }, "Applied."

    @app.callback(
        Output(ids.GRAPH_SETTINGS_STORE, "data", allow_duplicate=True),
        Input(ids.SUITE_URL, "pathname"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        prevent_initial_call=True,
    )
    def _set_tab_palette(pathname, settings):
        tool = (pathname or "").lstrip("/").split("/")[0] or ""
        settings = dict(settings or GRAPH_SETTINGS_DEFAULTS)
        if settings.get("palette_is_user_set"):
            return no_update
        target = "Monet" if tool == "tango" else "StarryNight"
        if settings.get("discrete_palette") == target:
            return no_update
        return {**settings, "discrete_palette": target, "continuous_palette": target}


_DISCRETE_TRACE_TYPES = {"scatter", "scattergl", "scatter3d", "bar", "histogram", "violin", "box"}
_CONTINUOUS_TRACE_TYPES = {
    "heatmap", "contour", "surface", "densitymapbox",
    "mesh3d", "isosurface", "volume", "histogram2d",
}


def _is_dark(color: str) -> bool:
    """Return True when the color has approximate relative luminance < 0.35."""
    _NAMES = {"white": "#ffffff", "black": "#000000"}
    c = _NAMES.get(color.lower(), color)
    if not c.startswith("#") or len(c) not in (4, 7):
        return False
    if len(c) == 4:
        c = f"#{c[1]*2}{c[2]*2}{c[3]*2}"
    r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) < (0.35 * 255)


def apply_settings_to_figure(fig_dict: dict, settings: dict, override: bool = False) -> dict:
    """Apply settings to a Plotly figure dict in-place. Returns the dict.

    When *override* is False (default), ``marker.size``, ``line.width``, and
    ``line.dash`` are only written when the trace does not already carry an
    explicit scalar value.  Pass ``override=True`` to unconditionally overwrite.
    """
    if not settings or not isinstance(fig_dict, dict):
        return fig_dict

    from cryocat.analysis.visplot import resolve_palette, resolve_colorscale

    layout = fig_dict.setdefault("layout", {})

    if settings.get("font_family") or settings.get("font_size"):
        font = layout.setdefault("font", {})
        if settings.get("font_family"):
            font["family"] = settings["font_family"]
        if settings.get("font_size"):
            font["size"] = settings["font_size"]

    if settings.get("bg_color"):
        bg = settings["bg_color"]
        layout["paper_bgcolor"] = bg
        layout["plot_bgcolor"] = bg
        if _is_dark(bg):
            text_color = "#e0e0e0"
            layout.setdefault("font", {}).setdefault("color", text_color)
            for axis_key in ("xaxis", "yaxis", "zaxis"):
                ax = layout.setdefault(axis_key, {})
                ax.setdefault("gridcolor", "#444444")
                ax.setdefault("tickfont", {}).setdefault("color", text_color)

    # W1: palette_is_user_set=True  → clear existing scalar string colours first so the
    #     chosen palette wins over Express-assigned per-trace colours.
    # W1: palette_is_user_set=False → fill-only: skip traces that already carry any
    #     explicit string colour (they were deliberately coloured by the figure builder).
    palette_is_user_set = bool(settings.get("palette_is_user_set"))

    if settings.get("discrete_palette"):
        palette = resolve_palette(settings["discrete_palette"])
        layout["colorway"] = palette
        discrete_traces = [t for t in fig_dict.get("data", [])
                           if t.get("type", "scatter") in _DISCRETE_TRACE_TYPES]
        for i, trace in enumerate(discrete_traces):
            color = palette[i % len(palette)]
            marker = trace.get("marker", {})
            existing_mc = marker.get("color")

            if isinstance(existing_mc, list):
                # Per-point data array — never overwrite in either mode.
                continue

            if not palette_is_user_set and isinstance(existing_mc, str):
                # Fill-only: trace already has an explicit colour — leave it.
                continue

            # User-set mode: clear scalar string colours so the palette wins.
            if palette_is_user_set and isinstance(existing_mc, str):
                trace.setdefault("marker", {}).pop("color", None)
                existing_lc = trace.get("line", {}).get("color")
                if isinstance(existing_lc, str):
                    trace.setdefault("line", {}).pop("color", None)

            trace.setdefault("marker", {})["color"] = color
            if trace.get("type") in ("histogram", "violin", "box", "bar"):
                # bar/histogram/violin/box carry line styling inside marker.line
                mline = trace.setdefault("marker", {}).setdefault("line", {})
                if not isinstance(mline.get("color"), list):
                    mline["color"] = color
            else:
                line = trace.get("line", {})
                if not isinstance(line.get("color"), list):
                    trace.setdefault("line", {})["color"] = color

    if settings.get("continuous_palette"):
        scale = resolve_colorscale(settings["continuous_palette"])
        # W2: coloraxis-bound traces (marker.coloraxis set) read the palette from
        # layout.coloraxis.colorscale — that is the correct path for Express scatter
        # with color="continuous_column".  Standalone heatmap/contour/surface traces
        # keep their own trace-level colorscale and need it set directly.
        layout.setdefault("coloraxis", {})["colorscale"] = scale
        for trace in fig_dict.get("data", []):
            if trace.get("type") in _CONTINUOUS_TRACE_TYPES:
                if not trace.get("marker", {}).get("coloraxis"):
                    trace["colorscale"] = scale

    marker_size = settings.get("marker_size")
    line_width = settings.get("line_width")
    line_dash = settings.get("line_dash")

    for trace in fig_dict.get("data", []):
        trace_type = trace.get("type", "")
        if trace_type in ("scatter", "scattergl", "scatter3d") and marker_size:
            marker = trace.setdefault("marker", {})
            if override or marker.get("size") is None:
                marker["size"] = marker_size
        if trace_type in ("scatter", "scattergl", "scatter3d"):
            if line_width or line_dash:
                line = trace.setdefault("line", {})
                if line_width and (override or line.get("width") is None):
                    line["width"] = line_width
                if line_dash and (override or line.get("dash") is None):
                    line["dash"] = line_dash

    return fig_dict


def style_figure(fig: go.Figure, settings: dict) -> dict:
    """Serialise *fig* and apply settings. Returns a mutable dict.

    Thin front for apply_settings_to_figure — the to_plotly_json call lives
    here so callers outside graphsettings.py do not need to call it directly.
    Does not stamp uirevision; use styled_figure when that is needed.
    """
    return apply_settings_to_figure(fig.to_plotly_json(), settings or {})


def figure_to_dict(fig: go.Figure) -> dict:
    """Serialise an already-styled figure to a Plotly JSON dict."""
    return fig.to_plotly_json()


def styled_figure(
    fig: go.Figure,
    settings: dict,
    *,
    uirevision: str,
    title: dict | str | None = None,
    margin: dict | None = None,
    scene: dict | None = None,
    height: int | None = None,
) -> go.Figure:
    """Apply settings to *fig* and stamp invariant layout keys. Returns a new Figure.

    The sole entry point for turning a go.Figure into a display-ready figure.
    *uirevision* is required so Dash preserves interactive state (zoom,
    selection) across data updates.
    """
    fig_dict = apply_settings_to_figure(fig.to_plotly_json(), settings or {})
    layout = fig_dict.setdefault("layout", {})
    layout["uirevision"] = uirevision
    if title is not None:
        layout["title"] = title if isinstance(title, dict) else {"text": title}
    if margin is not None:
        layout["margin"] = margin
    if scene is not None:
        layout["scene"] = scene
    if height is not None:
        layout["height"] = height
    return go.Figure(fig_dict)


def register_figure_in_pool(
    state: "GraphPoolState",
    fig: go.Figure,
    *,
    label: str,
    kind: str,
) -> "tuple[GraphPoolState, str]":
    """Serialise *fig* here (allowed) and insert into the graph pool.

    The only entry point for adding a go.Figure to the graph pool from outside
    graphsettings.py.  Callers pass a Figure; the to_plotly_json call stays
    inside this module.
    """
    from cryocat.app import graphpool as _graphpool
    payload = fig.to_plotly_json()
    return _graphpool.insert_graph_entry(state, payload, label=label, kind=kind)


def error_figure(msg: str) -> go.Figure:
    """Return a zero-trace figure with a centred annotation. Use for error states."""
    return go.Figure(layout={
        "annotations": [{
            "text": msg, "showarrow": False,
            "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
        }],
        "xaxis": {"visible": False},
        "yaxis": {"visible": False},
    })
