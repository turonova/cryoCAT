from cryocat.app.logger import dash_logger

import dash
from dash import html, dcc
from dash import Input, Output, State, exceptions, callback_context, ctx, ALL
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from cryocat.core import cryomotl
from cryocat.app.apputils import make_axis_trace
from cryocat.analysis import visplot
from cryocat.app.formgen import make_dropdown
from cryocat.app import ids
from cryocat.app.components.tableplot import _CONTINUOUS as _PALETTE_DISCRETE


def hover_template(columns, hover_info) -> str:
    if hover_info == "full":
        parts = [f"{col}: %{{customdata[{i}]}}" for i, col in enumerate(columns)]
    else:
        cols_list = list(columns)
        parts = [
            f"{col}: %{{customdata[{cols_list.index(col)}]}}"
            for col in hover_info
            if col in cols_list
        ]
    return "<br>".join(parts) + "<extra></extra>"


def aspect_ratio_dict(coords) -> dict:
    ranges = [coords[:, i].max() - coords[:, i].min() for i in range(3)]
    max_r = max(ranges) or 1
    return {"x": ranges[0] / max_r, "y": ranges[1] / max_r, "z": ranges[2] / max_r}


def tomo_figure(df: pd.DataFrame, index, color_col, colorscale, marker_size, hover_info, show_dual_graph) -> tuple:
    motl = cryomotl.Motl(df[cryomotl.Motl.motl_columns])
    tomo_ids = sorted(motl.df["tomo_id"].unique())
    tomo = tomo_ids[index]
    tm = motl.get_motl_subset(tomo)
    coords = tm.get_coordinates()
    color_vals = tm.df[color_col] if color_col in tm.df else tm.df["score"]
    hovertemplate = hover_template(tm.df.columns, hover_info)
    scale = visplot.resolve_colorscale(colorscale)
    fig = go.Figure(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        customdata=tm.df.values, hovertemplate=hovertemplate,
        mode="markers",
        marker=dict(size=marker_size or 5, opacity=0.8, color=color_vals, colorscale=scale),
    ))
    fig.update_layout(
        height=500, margin=dict(t=0, b=0, l=0, r=0),
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                   aspectmode="manual", aspectratio=aspect_ratio_dict(coords)),
    )
    graph_width = 6 if show_dual_graph else 12
    return fig, graph_width, {"display": "block", "marginTop": "1rem"}, f"Tomo ID: {tomo}"


def detail_figure(clickData, df: pd.DataFrame, twist_data, radius, tomo_index=0) -> go.Figure | None:
    point = clickData["points"][0]
    point_number = point.get("pointNumber")
    if point_number is None:
        return None
    motl = cryomotl.Motl(df[cryomotl.Motl.motl_columns])
    tomo_ids = sorted(motl.df["tomo_id"].unique())
    tomo_index = tomo_index or 0
    if tomo_index >= len(tomo_ids):
        tomo_index = 0
    tm = motl.get_motl_subset(tomo_ids[tomo_index])
    if point_number >= len(tm.df):
        return None
    subtomo_id = tm.df.iloc[point_number]["subtomo_id"]
    qp_df = pd.DataFrame(twist_data)
    qp_df = qp_df[qp_df["qp_id"] == subtomo_id]
    if qp_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=qp_df["twist_x"], y=qp_df["twist_y"], z=qp_df["twist_z"],
        mode="markers", marker=dict(size=3, color="#83BA99", opacity=0.8),
        name="Neighbor Points",
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers",
        marker=dict(size=6, color="orange"), name="Clicked Point",
    ))
    make_axis_trace(fig, length=5)
    fig.update_layout(
        height=500, margin=dict(t=0, b=0, l=0, r=0),
        scene=dict(
            xaxis=dict(title="X", range=[-radius, radius]),
            yaxis=dict(title="Y", range=[-radius, radius]),
            zaxis=dict(title="Z", range=[-radius, radius]),
            aspectmode="cube",
        ),
        showlegend=False,
    )
    return fig


def color_options_from_handle(handle: dict) -> list[dict]:
    """Return color-dropdown options from a PoolEntry registry dict.

    Raises TypeError when passed anything other than a PoolEntry handle
    (e.g. list[dict] store data from the old pre-pool pattern).
    """
    if not isinstance(handle, dict):
        raise TypeError(
            f"color_options_from_handle expects a pool registry entry (dict), "
            f"got {type(handle).__name__}"
        )
    if "numeric_columns" not in handle:
        raise TypeError(
            "color_options_from_handle expects a pool registry entry with "
            "'numeric_columns' key — pass a PoolEntry dict, not row data"
        )
    return [
        {"label": col, "value": col}
        for col in handle["numeric_columns"]
        if col != "tomo_id"
    ]


def tomo_items_from_handle(handle: dict, prefix: str) -> list:
    """Return tomo DropdownMenuItem list from a PoolEntry registry dict.

    Raises TypeError when passed anything other than a PoolEntry handle.
    """
    if not isinstance(handle, dict):
        raise TypeError(
            f"tomo_items_from_handle expects a pool registry entry (dict), "
            f"got {type(handle).__name__}"
        )
    if "tomo_ids" not in handle:
        raise TypeError(
            "tomo_items_from_handle expects a pool registry entry with "
            "'tomo_ids' key — pass a PoolEntry dict, not row data"
        )
    return [
        dbc.DropdownMenuItem(
            f"Tomo {tid}",
            id={"type": "tomo-menu-item", "owner": prefix, "index": f"{tid}"},
            n_clicks=0,
        )
        for tid in handle["tomo_ids"]
    ]


def get_viewer_component(prefix: str):
    return html.Div(
        id=f"{prefix}-container",
        style={"marginTop": "1rem"},
        children=[
            html.Div(
                id=f"{prefix}-graph-menu",
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "flexDirection": "row",
                            "gap": "1rem",
                            "alignItems": "center",
                            "width": "auto",  # don't stretch full width
                            "maxWidth": "100%",  # safety for responsiveness
                            "flexWrap": "wrap",  # optional if it might wrap on small screens
                        },
                        children=[
                            html.Div(
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button("Previous", id=f"{prefix}-prev", n_clicks=0),
                                        dbc.DropdownMenu(
                                            id=f"{prefix}-tomo-selector",
                                            label="Tomo ID",
                                            children=[],
                                            group=True,
                                            className="scrollable-dropdown",
                                        ),
                                        dbc.Button("Next", id=f"{prefix}-next", n_clicks=0),
                                    ]
                                ),
                                style={"flex": "0 0 auto"},
                            ),
                            html.Div(
                                make_dropdown(
                                    f"{prefix}-color-dropdown",
                                    [],
                                    None,
                                    placeholder="Color by",
                                    style={"width": "150px"},
                                ),
                                style={"flex": "0 0 auto"},
                            ),
                            html.Div(
                                make_dropdown(
                                    f"{prefix}-colorscale-dropdown",
                                    [{"label": s, "value": s} for s in _PALETTE_DISCRETE],
                                    None,
                                    placeholder="Auto (palette)",
                                    style={"width": "150px"},
                                ),
                                style={"flex": "0 0 auto"},
                            ),
                            html.Div(
                                style={
                                    "display": "flex",
                                    "flexDirection": "row",
                                    "alignItems": "center",  # vertically align label + slider
                                    "flex": "0 0 auto",  # prevent it from growing/shrinking
                                    "gap": "0.1rem",
                                },
                                children=[
                                    html.H5("Marker size:", style={"margin": 0}),
                                    html.Div(
                                        dcc.Slider(
                                            id=f"{prefix}-marker-size",
                                            min=1,
                                            max=20,
                                            step=1,
                                            value=5,
                                            tooltip={"placement": "right"},
                                            marks=None,
                                        ),
                                        style={
                                            "width": "150px",
                                            "flex": "0 0 auto",
                                        },
                                    ),
                                ],
                            ),
                        ],
                    )
                ],
                style={**{"display": "flex", "alignItems": "center"}, "marginBottom": "0.5rem"},
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div([dcc.Graph(id=f"{prefix}-graph")]), id=f"{prefix}-graph1-col", width=6),
                    dbc.Col(html.Div(id=f"{prefix}-graph2-container"), id=f"{prefix}-graph2-col", width=6),
                ],
                id=f"{prefix}-graph-row",
            ),
        ],
    )


def register_viewer_callbacks(app, prefix: str, show_dual_graph=False, hover_info="full", detailed_table=None, tabs_id=None, tab_value: str | None = None, visible_on_tabs=("motl-tab", "twist-tab", "nn-motl-tab", "cluster-tab"), radius_store_id=None, resolve_detail_df=None):

    def _motl_df(data):
        """Return a DataFrame from a pool reference dict or a legacy list[dict]."""
        if isinstance(data, dict) and "motl_id" in data:
            from cryocat.app.pool import get_rows, PoolPayloadMissing
            try:
                return get_rows(data["motl_id"])
            except PoolPayloadMissing:
                return pd.DataFrame()
        if isinstance(data, list) and data:
            return pd.DataFrame.from_records(data)
        return pd.DataFrame()

    if detailed_table == None:
        detailed_table = f"{prefix}-data"

    if tabs_id is not None:
        @app.callback(
            Output(f"{prefix}-graph-menu", "style"),
            Input(tabs_id, "active_tab"),
            State(f"{prefix}-graph-menu", "style"),
        )
        def toggle_visibility(active_tab, current_style):
            if current_style is None:
                current_style = {}

            updated_style = current_style.copy()

            if active_tab in visible_on_tabs:
                updated_style["display"] = "flex"
            else:
                updated_style["display"] = "none"

            return updated_style

    @app.callback(
        Output(f"{prefix}-index", "data"),
        Input(f"{prefix}-prev", "n_clicks"),
        Input(f"{prefix}-next", "n_clicks"),
        Input(f"{prefix}-data", "data"),
        State(f"{prefix}-index", "data"),
        State(ids.POOL_REGISTRY, "data"),
        prevent_initial_call=True,
    )
    def update_index(prev, next_, data, current_index, registry):
        if not data:
            raise exceptions.PreventUpdate
        if isinstance(data, dict) and "motl_id" in data:
            entry = (registry or {}).get(data["motl_id"], {})
            tomo_ids = entry.get("tomo_ids", [])
        elif isinstance(data, list) and data:
            tomo_ids = sorted({row["tomo_id"] for row in data})
        else:
            raise exceptions.PreventUpdate
        n = len(tomo_ids)
        if not n:
            raise exceptions.PreventUpdate
        ctx_id = callback_context.triggered_id
        if ctx_id == f"{prefix}-prev":
            return (current_index - 1) % n
        if ctx_id == f"{prefix}-next":
            return (current_index + 1) % n
        return 0

    @app.callback(
        Output(f"{prefix}-color-dropdown", "options"),
        Output(f"{prefix}-color-dropdown", "value"),
        Input(f"{prefix}-data", "data"),
        State(ids.POOL_REGISTRY, "data"),
        prevent_initial_call=True,
    )
    def update_color_options(data, registry):
        if not data:
            raise exceptions.PreventUpdate
        if isinstance(data, dict) and "motl_id" in data:
            entry = (registry or {}).get(data["motl_id"], {})
            opts = color_options_from_handle(entry)
        else:
            df = _motl_df(data)
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            opts = [{"label": col, "value": col} for col in numeric_cols if col not in ["tomo_id"]]
        values = [o["value"] for o in opts]
        default = "score" if "score" in values else (values[0] if values else None)
        return opts, default

    _tab_state = [State(tabs_id, "active_tab")] if (tabs_id and tab_value) else []

    @app.callback(
        Output(f"{prefix}-graph", "figure"),
        Output(f"{prefix}-graph1-col", "width"),
        Output(f"{prefix}-container", "style", allow_duplicate=True),
        Output(f"{prefix}-tomo-selector", "label"),
        Input(f"{prefix}-index", "data"),
        Input(f"{prefix}-color-dropdown", "value"),
        Input(f"{prefix}-colorscale-dropdown", "value"),
        Input(f"{prefix}-marker-size", "value"),
        Input(f"{prefix}-data", "data"),
        *_tab_state,
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        prevent_initial_call=True,
    )
    def update_plot(index, color_col, colorscale, marker_size, data, *rest):
        # rest is (active_tab, settings) when tab guard active, else (settings,)
        if tab_value:
            active_tab, settings = rest[0], rest[1]
            if active_tab != tab_value:
                raise exceptions.PreventUpdate
        else:
            settings = rest[0]
        if not data:
            raise exceptions.PreventUpdate
        effective = colorscale or (settings or {}).get("discrete_palette", "StarryNight")
        return tomo_figure(_motl_df(data), index, color_col, effective, marker_size, hover_info, show_dual_graph)

    @app.callback(
        Output(f"{prefix}-tomo-selector", "children"),
        Input(f"{prefix}-data", "data"),
        State(ids.POOL_REGISTRY, "data"),
        prevent_initial_call=True,
    )
    def populate_tomo_dropdown(data, registry):
        if not data:
            raise exceptions.PreventUpdate
        if isinstance(data, dict) and "motl_id" in data:
            entry = (registry or {}).get(data["motl_id"], {})
            return tomo_items_from_handle(entry, prefix)
        df = _motl_df(data)
        tomo_ids = sorted(df["tomo_id"].unique())
        return [
            dbc.DropdownMenuItem(
                f"Tomo {tid}",
                id={"type": "tomo-menu-item", "owner": prefix, "index": f"{tid}"},
                n_clicks=0,
            )
            for tid in tomo_ids
        ]

    @app.callback(
        Output(f"{prefix}-index", "data", allow_duplicate=True),
        Input({"type": "tomo-menu-item", "owner": prefix, "index": ALL}, "n_clicks"),
        State(f"{prefix}-data", "data"),
        prevent_initial_call=True,
    )
    def on_tomo_selected(n_clicks_list, data):
        if not data or not any(n_clicks_list):
            raise exceptions.PreventUpdate

        # Find which menu item was clicked
        triggered = ctx.triggered_id
        if triggered and isinstance(triggered, dict) and "index" in triggered:
            selected_tomo = int(triggered["index"])
            df = _motl_df(data)
            tomo_ids = sorted(df["tomo_id"].unique())
            try:
                return tomo_ids.index(selected_tomo)
            except ValueError:
                raise exceptions.PreventUpdate

        raise exceptions.PreventUpdate

    if show_dual_graph:
        if not radius_store_id:
            raise ValueError(
                "radius_store_id is required when show_dual_graph=True; "
                "pass the id of the store that holds the NN radius."
            )

        @app.callback(
            Output(f"{prefix}-graph2-container", "children"),
            Input(f"{prefix}-graph", "clickData"),
            State(f"{prefix}-data", "data"),
            State(f"{detailed_table}", "data"),
            State(radius_store_id, "data"),
            State(f"{prefix}-index", "data"),
            prevent_initial_call=True,
        )
        def show_detail_on_click(clickData, data, twist_data, radius_data, tomo_index):
            if not clickData or not data:
                raise dash.exceptions.PreventUpdate
            if isinstance(radius_data, dict):
                radius = radius_data.get("nn_radius")
            else:
                radius = radius_data
            if radius is None:
                raise exceptions.PreventUpdate
            if resolve_detail_df is not None:
                twist_records = resolve_detail_df(twist_data)
                if twist_records is None:
                    raise exceptions.PreventUpdate
            elif isinstance(twist_data, dict) and "motl_id" in twist_data:
                try:
                    from cryocat.app.pool import get_rows, PoolPayloadMissing
                    twist_records = get_rows(twist_data["motl_id"])
                except Exception:
                    raise exceptions.PreventUpdate
            else:
                twist_records = twist_data
            fig = detail_figure(clickData, _motl_df(data), twist_records, radius, tomo_index)
            if fig is None:
                return html.Div(
                    "No twist neighbours for this particle.",
                    style={"color": "#888", "padding": "1rem", "textAlign": "center"},
                )
            return dcc.Graph(id=f"{prefix}-detail-graph", figure=fig)
