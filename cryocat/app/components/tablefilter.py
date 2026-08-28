"""Range-slider filtering for tableview — pure functions + callbacks.

Sliders are live view filters: moving a slider encodes the active range into
the grid's filterModel (as AG Grid inRange conditions), which causes the grid
to purge its infinite-model cache and re-request rows from the server with the
updated filter.  The "Apply Changes" button commits the currently filtered
subset back to the pool permanently.

W2 invariants:
- Sliders at their full column range contribute nothing to filterModel — no
  no-op filtering and no refresh on initial slider build (prevent_initial_call).
- Slider filters and AG Grid column header filters live in one filterModel;
  apply_filter_model (tablegrid) handles both with logical AND.
- Active filter count is shown in the grid area.
"""

from __future__ import annotations

from dash import html, dcc, Input, Output, State, ALL, MATCH, exceptions, no_update, ctx
import dash_bootstrap_components as dbc

from cryocat.app import ids, pool as _pool
from cryocat.app.components.tablegrid import apply_filter_model


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


# ── Callbacks ──────────────────────────────────────────────────────────────────


def register_tablefilter_callbacks(app, prefix: str, resolve_df=None) -> None:
    @app.callback(
        Output(f"{prefix}-filters-container", "children"),
        Input(f"{prefix}-global-data-store", "data"),
        State(ids.POOL_REGISTRY, "data"),
        prevent_initial_call=True,
    )
    def build_range_sliders(ref, registry):
        if not ref or not isinstance(ref, dict):
            raise exceptions.PreventUpdate
        col_ranges = _pool.get_column_ranges_for_ref(ref, registry, resolve_df)
        if not col_ranges:
            raise exceptions.PreventUpdate
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
        Output(f"{prefix}-slider-filters-store", "data"),
        Input({"type": f"{prefix}-filter-slider", "column": ALL}, "value"),
        State({"type": f"{prefix}-filter-slider", "column": ALL}, "id"),
        State({"type": f"{prefix}-filter-slider", "column": ALL}, "min"),
        State({"type": f"{prefix}-filter-slider", "column": ALL}, "max"),
        prevent_initial_call=True,
    )
    def _on_slider_change(slider_values, slider_ids, slider_mins, slider_maxs):
        """Write active slider ranges to slider-filters-store to trigger server-side re-filter.

        Full-range sliders contribute nothing (W2 — no no-op filtering).
        """
        slider_filters = {}
        for val, sid, mn, mx in zip(slider_values, slider_ids, slider_mins, slider_maxs):
            lo, hi = val[0], val[1]
            if lo != mn or hi != mx:
                slider_filters[sid["column"]] = [lo, hi]
        return slider_filters

    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output(f"{prefix}-global-data-store", "data", allow_duplicate=True),
        Output(f"{prefix}-grid", "filterModel", allow_duplicate=True),
        Output(f"{prefix}-slider-filters-store", "data", allow_duplicate=True),
        Input(f"{prefix}-apply-btn", "n_clicks"),
        State(f"{prefix}-global-data-store", "data"),
        State(f"{prefix}-grid", "filterModel"),
        State(f"{prefix}-slider-filters-store", "data"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def apply_filters_btn(_, ref, filter_model, slider_filters, registry, pool_meta, next_id):
        """Commit the currently filtered subset back to the pool (permanent filter).

        Reads column-header filters (filterModel) and slider filters (slider-filters-store),
        applies both to the table entry, and commits the result.  Clears both stores
        afterward so the grid reflects the new baseline.
        """
        if not ref or not isinstance(ref, dict):
            raise exceptions.PreventUpdate
        df = _pool.get_table_df(ref)
        if df is None:
            raise exceptions.PreventUpdate
        filtered_df = apply_filter_model(df, filter_model or {}, slider_filters or {})
        return (*_pool.commit_rows(ref, filtered_df, registry, pool_meta, next_id), {}, {})
