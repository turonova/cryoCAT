"""Range-slider filtering for tableview — pure functions + callbacks."""
from __future__ import annotations

import pandas as pd
from dash import html, dcc, Input, Output, State, ALL, MATCH, exceptions, no_update, ctx
import dash_bootstrap_components as dbc


# ── Pure functions ─────────────────────────────────────────────────────────────


def slider_specs(rows: list[dict]) -> list[dict]:
    """Numeric columns → [{column, min, max, step}]. Pure."""
    if not rows:
        return []
    df = pd.DataFrame(rows)
    specs = []
    for col in df.select_dtypes(include="number").columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min == col_max or pd.isna(col_min) or pd.isna(col_max):
            continue
        min_val = float(col_min)
        max_val = float(col_max)
        step = 1.0 if pd.api.types.is_integer_dtype(df[col]) else ((max_val - min_val) / 100 or 1.0)
        specs.append({"column": col, "min": min_val, "max": max_val, "step": step})
    return specs


def apply_filters(rows: list[dict], filters: dict[str, tuple[float, float]]) -> list[dict]:
    """Filter rows to those within (lo, hi) per column. Pure."""
    if not rows or not filters:
        return rows
    df = pd.DataFrame(rows)
    for col, (lo, hi) in filters.items():
        if col in df.columns:
            df = df[df[col].between(lo, hi)]
    return df.to_dict("records")


def sync_bounds(
    slider_val: list[float],
    min_input: float | None,
    max_input: float | None,
    slider_min: float,
    slider_max: float,
    triggered_type_key: str,
) -> tuple:
    """Reconcile slider and input-box for a single range. Pure.

    Returns (slider_out, min_out, max_out) using no_update as a sentinel where
    the control that fired should not receive its own value back.
    """
    if f"-filter-slider" in triggered_type_key:
        return no_update, slider_val[0], slider_val[1]
    if f"-filter-min" in triggered_type_key:
        if min_input is None:
            return no_update, no_update, no_update
        clamped = float(max(slider_min, min(min_input, slider_val[1])))
        return [clamped, slider_val[1]], clamped, no_update
    if f"-filter-max" in triggered_type_key:
        if max_input is None:
            return no_update, no_update, no_update
        clamped = float(min(slider_max, max(max_input, slider_val[0])))
        return [slider_val[0], clamped], no_update, clamped
    return no_update, no_update, no_update


# ── Callbacks ──────────────────────────────────────────────────────────────────


def register_tablefilter_callbacks(app, prefix: str) -> None:
    @app.callback(
        Output(f"{prefix}-filters-container", "children"),
        Input(f"{prefix}-global-data-store", "data"),
    )
    def build_range_sliders(data):
        if not data:
            raise exceptions.PreventUpdate
        specs = slider_specs(data)
        divFlex = {"display": "flex", "flexDirection": "row", "alignItems": "center"}
        labelFlex = {
            "flexShrink": 0,
            "fontSize": "10px",
            "fontWeight": "600",
            "color": "var(--color10)",
            "whiteSpace": "nowrap",
            "marginRight": "4px",
            "marginLeft": "6px",
            "marginTop": "-10px",
        }
        slideFlex = {"flex": "1", "minWidth": "0", "padding": "0", "marginTop": "-10px"}
        inputStyle = {"width": "52px", "flexShrink": 0, "marginTop": "-10px"}

        col_components = [
            dbc.Col(
                html.Div([
                    html.Div(f"{s['column']}:", style=labelFlex),
                    dcc.Input(
                        id={"type": f"{prefix}-filter-min", "column": s["column"]},
                        type="number",
                        value=s["min"],
                        debounce=True,
                        className="filter-range-input",
                        style=inputStyle,
                    ),
                    html.Div(
                        dcc.RangeSlider(
                            id={"type": f"{prefix}-filter-slider", "column": s["column"]},
                            min=s["min"],
                            max=s["max"],
                            step=s["step"],
                            value=[s["min"], s["max"]],
                            tooltip={"placement": "top"},
                            marks=None,
                            allowCross=False,
                        ),
                        style=slideFlex,
                        className="filter-slider-wrapper",
                    ),
                    dcc.Input(
                        id={"type": f"{prefix}-filter-max", "column": s["column"]},
                        type="number",
                        value=s["max"],
                        debounce=True,
                        className="filter-range-input",
                        style=inputStyle,
                    ),
                ], style=divFlex),
                width=3,
            )
            for s in specs
        ]
        return [dbc.Row(col_components, className="gx-1 gy-0")]

    @app.callback(
        Output({"type": f"{prefix}-filter-slider", "column": MATCH}, "value", allow_duplicate=True),
        Output({"type": f"{prefix}-filter-min", "column": MATCH}, "value", allow_duplicate=True),
        Output({"type": f"{prefix}-filter-max", "column": MATCH}, "value", allow_duplicate=True),
        Input({"type": f"{prefix}-filter-slider", "column": MATCH}, "value"),
        Input({"type": f"{prefix}-filter-min", "column": MATCH}, "value"),
        Input({"type": f"{prefix}-filter-max", "column": MATCH}, "value"),
        State({"type": f"{prefix}-filter-slider", "column": MATCH}, "min"),
        State({"type": f"{prefix}-filter-slider", "column": MATCH}, "max"),
        prevent_initial_call=True,
    )
    def sync_filter_controls(slider_val, min_input, max_input, slider_min, slider_max):
        triggered = ctx.triggered_id
        if triggered is None or slider_val is None:
            raise exceptions.PreventUpdate
        ttype = triggered.get("type", "") if isinstance(triggered, dict) else ""
        result = sync_bounds(slider_val, min_input, max_input, slider_min, slider_max, ttype)
        if result == (no_update, no_update, no_update):
            raise exceptions.PreventUpdate
        return result

    @app.callback(
        Output(f"{prefix}-grid", "rowData", allow_duplicate=True),
        Input(f"{prefix}-global-data-store", "data"),
        Input({"type": f"{prefix}-filter-slider", "column": ALL}, "value"),
        State({"type": f"{prefix}-filter-slider", "column": ALL}, "id"),
        prevent_initial_call=True,
    )
    def filter_data_by_sliders(global_data, slider_values, slider_ids):
        if not global_data:
            raise exceptions.PreventUpdate
        filters = {
            sid.get("column"): (lo, hi)
            for sid, (lo, hi) in zip(slider_ids, slider_values)
        }
        return apply_filters(global_data, filters)

    @app.callback(
        Output(f"{prefix}-global-data-store", "data", allow_duplicate=True),
        Output(f"{prefix}-grid", "rowData", allow_duplicate=True),
        Input(f"{prefix}-apply-btn", "n_clicks"),
        State(f"{prefix}-global-data-store", "data"),
        State({"type": f"{prefix}-filter-slider", "column": ALL}, "value"),
        State({"type": f"{prefix}-filter-slider", "column": ALL}, "id"),
        prevent_initial_call=True,
    )
    def apply_filters_btn(_, global_data, slider_values, slider_ids):
        if not global_data:
            raise exceptions.PreventUpdate
        filters = {
            sid.get("column"): (lo, hi)
            for sid, (lo, hi) in zip(slider_ids, slider_values)
        }
        filtered = apply_filters(global_data, filters)
        return filtered, filtered
