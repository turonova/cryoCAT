"""Range-slider filtering for tableview — pure functions + callbacks.

The global-data-store for pool-aware tables holds a reference dict
``{"motl_id": str, "rev": int}`` instead of row data.  All callbacks here
read rows directly from the server-side pool and write results back there.
"""

from __future__ import annotations

import pandas as pd
from dash import html, dcc, Input, Output, State, ALL, MATCH, exceptions, no_update, ctx
import dash_bootstrap_components as dbc

from cryocat.app import ids
from cryocat.app.pool import (
    _compute_entry_metadata,
    get_rows,
    replace_motl_rows,
    PoolPayloadMissing,
    PoolState,
)


# ── Pure functions ─────────────────────────────────────────────────────────────

_LABEL_FLEX = {
    "flexShrink": 0,
    "fontWeight": "600",
    "color": "var(--color10)",
    "whiteSpace": "nowrap",
    "marginRight": "4px",
    "marginLeft": "6px",
    "marginTop": "-10px",
}
_SLIDE_FLEX = {"flex": "1", "minWidth": "0", "padding": "0", "marginTop": "-10px"}
_INPUT_STYLE = {"width": "52px", "flexShrink": 0, "marginTop": "-10px"}
_DIV_FLEX = {"display": "flex", "flexDirection": "row", "alignItems": "center"}


def slider_specs(column_ranges: dict[str, list[float]]) -> list[dict]:
    """column_ranges → [{column, min, max, step}].  Pure.

    Each value in *column_ranges* is ``[min, max, step]`` as produced by
    :func:`~cryocat.app.pool._compute_entry_metadata`.

    Raises TypeError when passed anything other than a dict — the caller must
    supply pre-computed range metadata, not raw row data.  (T1/W1)
    """
    if not isinstance(column_ranges, dict):
        raise TypeError(
            f"slider_specs expects a column_ranges dict, got {type(column_ranges).__name__} "
            "— pass entry['column_ranges'] from the pool registry, not row data"
        )
    return [
        {"column": col, "min": rng[0], "max": rng[1], "step": rng[2]}
        for col, rng in column_ranges.items()
    ]


def apply_filters(df: pd.DataFrame, filters: dict[str, tuple[float, float]]) -> list[dict]:
    """Filter *df* to rows within (lo, hi) per column.  Returns ``list[dict]`` for the grid."""
    if df.empty or not filters:
        return df.to_dict("records")
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
    """Reconcile slider and input-box for a single range.  Pure.

    Returns ``(slider_out, min_out, max_out)`` using ``no_update`` where the
    control that fired should not receive its own value back.
    """
    if "-filter-slider" in triggered_type_key:
        return no_update, slider_val[0], slider_val[1]
    if "-filter-min" in triggered_type_key:
        if min_input is None:
            return no_update, no_update, no_update
        clamped = float(max(slider_min, min(min_input, slider_val[1])))
        return [clamped, slider_val[1]], clamped, no_update
    if "-filter-max" in triggered_type_key:
        if max_input is None:
            return no_update, no_update, no_update
        clamped = float(min(slider_max, max(max_input, slider_val[0])))
        return [slider_val[0], clamped], no_update, clamped
    return no_update, no_update, no_update


def _slider_col(s: dict, prefix: str) -> dbc.Col:
    """Build one dbc.Col slider widget for a single column spec.  Pure."""
    return dbc.Col(
        html.Div(
            [
                html.Div(f"{s['column']}:", style=_LABEL_FLEX),
                dcc.Input(
                    id={"type": f"{prefix}-filter-min", "column": s["column"]},
                    type="number",
                    value=s["min"],
                    debounce=True,
                    className="filter-range-input",
                    style=_INPUT_STYLE,
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
                    style=_SLIDE_FLEX,
                    className="filter-slider-wrapper",
                ),
                dcc.Input(
                    id={"type": f"{prefix}-filter-max", "column": s["column"]},
                    type="number",
                    value=s["max"],
                    debounce=True,
                    className="filter-range-input",
                    style=_INPUT_STYLE,
                ),
            ],
            style=_DIV_FLEX,
        ),
        width=3,
    )


def _commit_filter(
    ref: dict,
    slider_ids: list,
    slider_values: list,
    registry,
    pool_meta,
    next_id,
) -> tuple:
    """Pure: apply slider filters to pool rows, commit to pool, return (state, motl_id, filtered_df)."""
    motl_id = ref.get("motl_id")
    df = get_rows(motl_id)  # raises PoolPayloadMissing if absent
    filters = {sid.get("column"): (lo, hi) for sid, (lo, hi) in zip(slider_ids, slider_values)}
    filtered_df = df.copy()
    for col, (lo, hi) in filters.items():
        if col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[col].between(lo, hi)]
    state = PoolState.from_stores(registry, pool_meta, next_id)
    return replace_motl_rows(state, motl_id, filtered_df), motl_id, filtered_df


# ── Callbacks ──────────────────────────────────────────────────────────────────


def register_tablefilter_callbacks(app, prefix: str) -> None:
    @app.callback(
        Output(f"{prefix}-filters-container", "children"),
        Input(f"{prefix}-global-data-store", "data"),
        State(ids.POOL_REGISTRY, "data"),
        prevent_initial_call=True,
    )
    def build_range_sliders(ref, registry):
        if not ref or not isinstance(ref, dict) or not ref.get("motl_id"):
            raise exceptions.PreventUpdate
        motl_id = ref["motl_id"]
        entry = (registry or {}).get(motl_id, {})
        col_ranges = entry.get("column_ranges", {})
        if not col_ranges:
            try:
                df = get_rows(motl_id)
            except PoolPayloadMissing:
                raise exceptions.PreventUpdate
            _, col_ranges, _ = _compute_entry_metadata(df)
        cols = [_slider_col(s, prefix) for s in slider_specs(col_ranges)]
        return [dbc.Row(cols, className="gx-1 gy-0")]

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
    def filter_data_by_sliders(ref, slider_values, slider_ids):
        if not ref or not isinstance(ref, dict) or not ref.get("motl_id"):
            raise exceptions.PreventUpdate
        try:
            df = get_rows(ref["motl_id"])
        except PoolPayloadMissing:
            raise exceptions.PreventUpdate
        filters = {sid.get("column"): (lo, hi) for sid, (lo, hi) in zip(slider_ids, slider_values)}
        return apply_filters(df, filters)

    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output(f"{prefix}-global-data-store", "data", allow_duplicate=True),
        Output(f"{prefix}-grid", "rowData", allow_duplicate=True),
        Input(f"{prefix}-apply-btn", "n_clicks"),
        State(f"{prefix}-global-data-store", "data"),
        State({"type": f"{prefix}-filter-slider", "column": ALL}, "value"),
        State({"type": f"{prefix}-filter-slider", "column": ALL}, "id"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def apply_filters_btn(_, ref, slider_values, slider_ids, registry, pool_meta, next_id):
        if not ref or not isinstance(ref, dict) or not ref.get("motl_id"):
            raise exceptions.PreventUpdate
        try:
            state, motl_id, _ = _commit_filter(ref, slider_ids, slider_values, registry, pool_meta, next_id)
        except PoolPayloadMissing:
            raise exceptions.PreventUpdate
        new_rev = state.registry[motl_id]["revision"]
        # Grid rowData is left to filter_data_by_sliders (triggered by global-data-store change).
        # Returning no_update here avoids serialising the full filtered DataFrame. (W2/T3)
        return *state.to_stores(), {"motl_id": motl_id, "rev": new_rev}, no_update
