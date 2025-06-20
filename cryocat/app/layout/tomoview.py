from cryocat.app.logger import dash_logger

import dash
from dash import html, dcc
from dash import Input, Output, State, callback, exceptions, callback_context, ctx, ALL
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from cryocat import cryomotl
from cryocat.app.apputils import make_axis_trace


def get_colorscale(colorscale_name):
    if colorscale_name == "Monet":
        return [
            [0.0, "#AEC684"],
            [0.25, "#4EACB6"],
            [0.5, "#C0A3BA"],
            [0.75, "#7D82AB"],
            [1.0, "#865B96"]
        ]
    else:
        return colorscale_name  # use Plotly built-in name


def get_viewer_component(prefix: str):
    return html.Div(
        id=f"{prefix}-container",
        style={"marginTop": "1rem"},
        children=[
            html.Div(
                id=f"{prefix}-graph-menu",
                children=[
                    dbc.Row(
                        children=[
                            dbc.Col(
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
                                width=2,
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id=f"{prefix}-color-dropdown",
                                    placeholder="Color by",
                                ),
                                width=1,
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id=f"{prefix}-colorscale-dropdown",
                                    placeholder="Color scale",
                                    options=[
                                        {"label": s, "value": s} for s in ["Monet", "Viridis", "Cividis", "Plasma", "Jet", "Hot"]
                                    ],
                                    value="Monet",
                                ),
                                width=1,
                            ),
                            dbc.Col(
                                dcc.Slider(
                                    id=f"{prefix}-marker-size",
                                    min=1,
                                    max=20,
                                    step=1,
                                    value=3,
                                ),
                                width=4,
                            ),
                        ],
                        className="g-2",  # Optional gutter spacing
                    ),
                ],
                style={
                    "marginBottom": "0.5rem",
                    "display": "flex",
                    "alignItems": "center",
                },
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(id=f"{prefix}-graph1-container"), id=f"{prefix}-graph1-col", width=6),
                    dbc.Col(html.Div(id=f"{prefix}-graph2-container"), id=f"{prefix}-graph2-col", width=6),
                ],
                id=f"{prefix}-graph-row",
            ),
        ],
    )


def register_viewer_callbacks(prefix: str, show_dual_graph=False, hover_info="full", detailed_table=None):

    if detailed_table == None:
        detailed_table = f"{prefix}-data"

    @callback(
        Output(f"{prefix}-graph-menu", "style"),
        Input("table-tabs", "active_tab"),
        State(f"{prefix}-graph-menu", "style"),
    )
    def toggle_visibility(active_tab, current_style):
        if current_style is None:
            current_style = {}

        updated_style = current_style.copy()

        if active_tab == "motl-tab":
            updated_style["display"] = "block"
        else:
            updated_style["display"] = "none"

        return updated_style

    @callback(
        Output(f"{prefix}-index", "data"),
        Input(f"{prefix}-prev", "n_clicks"),
        Input(f"{prefix}-next", "n_clicks"),
        Input(f"motl-data-store", "data"),
        State(f"{prefix}-index", "data"),
        State(f"{prefix}-data", "data"),
        prevent_initial_call=True,
    )
    def update_index(prev, next_, motl_data, current_index, data):
        print(data)
        if not data:
            raise exceptions.PreventUpdate

        tomo_ids = sorted({row["tomo_id"] for row in data})
        n = len(tomo_ids)

        ctx = callback_context.triggered_id
        if ctx == f"{prefix}-prev":
            return (current_index - 1) % n
        if ctx == f"{prefix}-next":
            return (current_index + 1) % n
        return current_index

    @callback(
        Output(f"{prefix}-color-dropdown", "options"),
        Input(f"{prefix}-data", "data"),
        prevent_initial_call=True,
    )
    def update_color_options(data):
        if not data:
            raise exceptions.PreventUpdate
        df = pd.DataFrame(data)
        # Limit to numeric columns only
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        return [{"label": col, "value": col} for col in numeric_cols if col not in ["tomo_id"]]

    @callback(
        # Output(f"{prefix}-graph", "figure"),
        Output(f"{prefix}-graph1-container", "children"),
        Output(f"{prefix}-graph1-col", "width"),
        Output(f"{prefix}-container", "style", allow_duplicate=True),
        Output(f"{prefix}-tomo-selector", "label"),
        Input(f"{prefix}-index", "data"),
        Input(f"{prefix}-color-dropdown", "value"),
        Input(f"{prefix}-colorscale-dropdown", "value"),
        Input(f"{prefix}-marker-size", "value"),
        Input(f"{prefix}-data", "data"),
        prevent_initial_call=True,
    )
    def update_plot(index, color_col, colorscale, marker_size, data):
        if not data:
            raise exceptions.PreventUpdate
        motl_df = pd.DataFrame(data)
        motl = cryomotl.Motl(motl_df[cryomotl.Motl.motl_columns])
        tomo_ids = motl.get_unique_values("tomo_id")
        if tomo_ids is None:
            raise exceptions.PreventUpdate

        tomo = tomo_ids[index]
        tm = motl.get_motl_subset(tomo)

        coords = tm.get_coordinates()

        color_vals = tm.df[color_col] if color_col in tm.df else tm.df["score"]

        # Convert full rows to a 2D array for customdata
        customdata = tm.df.values
        columns = tm.df.columns  # to generate hovertemplate dynamically

        if hover_info == "full":
            hover_lines = [f"{col}: %{{customdata[{i}]}}" for i, col in enumerate(columns)]
            hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"
        else:
            hover_lines = [
                f"{col}: %{{customdata[{list(columns).index(col)}]}}" for col in hover_info if col in columns
            ]
            hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

        fig = go.Figure(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                customdata=customdata,
                hovertemplate=hovertemplate,
                mode="markers",
                marker=dict(size=marker_size or 3, opacity=0.8, color=color_vals, colorscale=get_colorscale(colorscale) or get_colorscale("Monet")),
            )
        )

        fig.update_layout(
            # title=f"Tomo ID: {tomo} • Color by: {color_col or 'z'}",
            height=500,
            margin=dict(t=20, b=20, l=0, r=0),
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        )

        graph = dcc.Graph(id=f"{prefix}-graph", figure=fig)

        graph_width = 12
        if show_dual_graph:
            graph_width = 6

        return graph, graph_width, {"display": "block", "marginTop": "1rem"}, f"Tomo ID: {tomo}"

    @callback(
        Output(f"{prefix}-tomo-selector", "children"),
        Input(f"{prefix}-data", "data"),
        prevent_initial_call=True,
    )
    def populate_tomo_dropdown(data):
        if not data:
            raise exceptions.PreventUpdate
        df = pd.DataFrame(data)
        tomo_ids = sorted(df["tomo_id"].unique())
        return [
            dbc.DropdownMenuItem(f"Tomo {tid}", id={"type": f"{prefix}-menu-item", "index": f"{tid}"}, n_clicks=0)
            for tid in tomo_ids
        ]

    @callback(
        Output(f"{prefix}-index", "data", allow_duplicate=True),
        Input({"type": f"{prefix}-menu-item", "index": ALL}, "n_clicks"),
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
            df = pd.DataFrame(data)
            tomo_ids = sorted(df["tomo_id"].unique())
            try:
                return tomo_ids.index(selected_tomo)
            except ValueError:
                raise exceptions.PreventUpdate

        raise exceptions.PreventUpdate

    @callback(
        Output(f"{prefix}-graph2-container", "children"),
        Input(f"{prefix}-graph", "clickData"),
        State(f"{prefix}-data", "data"),
        State(f"{detailed_table}", "data"),
        prevent_initial_call=True,
    )
    def show_detail_on_click(clickData, data, twist_data):
        if not show_dual_graph or not clickData or not data:
            raise dash.exceptions.PreventUpdate

        # Get index of clicked point from customdata
        clicked_row = clickData["points"][0]["customdata"]
        clicked_df = pd.DataFrame([clicked_row], columns=pd.DataFrame(data).columns)

        subtomo_index = clicked_df.columns.get_loc("subtomo_id")  # Find its position
        subtomo_id = clicked_row[subtomo_index]  # Get the actual value

        qp_df = pd.DataFrame(twist_data)
        qp_df = qp_df[qp_df["qp_id"] == subtomo_id]

        if qp_df.empty:
            raise exceptions.PreventUpdate

        ax_limit = qp_df[["twist_x", "twist_y", "twist_z"]].abs().to_numpy().max()

        fig = go.Figure()

        # Add all related points (with uniform color)
        fig.add_trace(
            go.Scatter3d(
                x=qp_df["twist_x"],
                y=qp_df["twist_y"],
                z=qp_df["twist_z"],
                mode="markers",
                marker=dict(
                    size=3,
                    color="#83BA99",  # Uniform color for all data points
                    opacity=0.8,
                ),
                name="Neighbor Points",
            )
        )

        # Add the central point at origin (0,0,0) with a different color
        fig.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode="markers",
                marker=dict(
                    size=6,
                    color="orange",  # Central clicked point
                ),
                name="Clicked Point",
            )
        )

        make_axis_trace(fig, length=5)

        fig.update_layout(
            margin=dict(t=10, b=10),
            scene=dict(
                xaxis=dict(title="X", range=[-ax_limit, ax_limit]),
                yaxis=dict(title="Y", range=[-ax_limit, ax_limit]),
                zaxis=dict(title="Z", range=[-ax_limit, ax_limit]),
            ),
            showlegend=False,
        )

        return dcc.Graph(id=f"{prefix}-detail-graph", figure=fig)
