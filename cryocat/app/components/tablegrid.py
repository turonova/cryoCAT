"""AG Grid with infinite row model — server-side sort, filter, and selection.

The grid never receives the full DataFrame. On each scroll or filter event
AG Grid sends getRowsRequest; the _rows callback applies sort + filter to the
server-side pool entry and returns one 1000-row block via getRowsResponse.

Pure functions (apply_sort_model, apply_filter_model, slice_block,
resolve_select_all_ids) are testable without Dash. Block serialisation goes
through pool.block_to_records so the T3 AST guard is not triggered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Input, Output, State, ctx, exceptions, html, dcc, no_update
import dash_ag_grid as dag

from cryocat.app import ids, pool as _pool
from cryocat.app.pool import _CACHE_BLOCK_SIZE, block_to_records

CACHE_BLOCK_SIZE = _CACHE_BLOCK_SIZE  # re-exported so test can import from here

_BASE_GRID_OPTIONS = {"cacheBlockSize": CACHE_BLOCK_SIZE, "maxBlocksInCache": 20, "rowSelection": {"mode": "multiRow"}}


# ── Pure functions ─────────────────────────────────────────────────────────────


def apply_sort_model(df: pd.DataFrame, sort_model: list[dict]) -> pd.DataFrame:
    """Apply AG Grid sortModel to *df*.  Pure, stable sort.

    *sort_model* is ``[{"colId": col, "sort": "asc"|"desc"}, ...]`` in priority order.
    """
    if not sort_model:
        return df
    cols = [s["colId"] for s in sort_model]
    asc = [s.get("sort", "asc") == "asc" for s in sort_model]
    return df.sort_values(cols, ascending=asc, kind="stable")


def apply_filter_model(
    df: pd.DataFrame,
    filter_model: dict,
    slider_filters: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Apply AG Grid filterModel plus slider range filters to *df*.  Pure.

    Both sources combine with logical AND.  Recognised filterModel types:
    - ``"number"`` with ops ``equals``, ``greaterThan``, ``lessThan``, ``inRange``
    - ``"text"`` with op ``contains``

    Columns prefixed with ``_`` are ignored (internal grid markers).
    """
    for col, spec in (filter_model or {}).items():
        if col not in df.columns or col.startswith("_"):
            continue
        ftype = spec.get("filterType", "number")
        op = spec.get("type", "equals")
        val = spec.get("filter")
        val2 = spec.get("filterTo")
        if ftype == "number":
            if op == "equals" and val is not None:
                df = df[df[col] == val]
            elif op == "greaterThan" and val is not None:
                df = df[df[col] > val]
            elif op == "lessThan" and val is not None:
                df = df[df[col] < val]
            elif op == "inRange" and val is not None and val2 is not None:
                df = df[df[col].between(val, val2)]
        elif ftype == "text":
            if op == "contains" and val:
                df = df[df[col].astype(str).str.contains(str(val), case=False, na=False)]
    for col, (lo, hi) in (slider_filters or {}).items():
        if col in df.columns:
            df = df[df[col].between(lo, hi)]
    return df


def slice_block(df: pd.DataFrame, start_row: int, end_row: int) -> pd.DataFrame:
    """Return ``df.iloc[start_row:end_row]``.  No error if end_row > len(df)."""
    return df.iloc[start_row:end_row]


def rows_response(
    request: dict | None,
    df: pd.DataFrame | None,
    n_rows_hint: int = 0,
    *,
    slider_filters: dict | None = None,
) -> dict:
    """Pure getRowsResponse handler.  Never raises.

    When df is absent and n_rows_hint > 0, return that count so the grid does
    not see rowCount=0 while the pool payload is temporarily missing (hot-reload).
    n_rows_hint comes from the pool registry; it is 0 when no motl is loaded.
    """
    if request is None:
        return {"rowData": [], "rowCount": n_rows_hint}
    if df is None:
        return {"rowData": [], "rowCount": n_rows_hint}
    filter_model = request.get("filterModel") or {}
    sort_model = request.get("sortModel") or []
    start_row = request.get("startRow", 0)
    end_row = request.get("endRow", CACHE_BLOCK_SIZE)
    filtered = apply_filter_model(df, filter_model, slider_filters or {})
    sorted_df = apply_sort_model(filtered, sort_model)
    block = slice_block(sorted_df, start_row, end_row)
    return {
        "rowData": block_to_records(block, max_rows=CACHE_BLOCK_SIZE),
        "rowCount": len(filtered),
    }


def resolve_select_all_ids(
    df: pd.DataFrame,
    filter_model: dict,
    slider_filters: dict[str, tuple[float, float]],
    *,
    id_column: str | None = None,
) -> list:
    """Return identity-column values for all rows that pass the current filters.

    Uses *id_column* when given; probes common identity columns otherwise;
    falls back to the integer row index when none is found.
    """
    filtered = apply_filter_model(df, filter_model, slider_filters)
    if id_column and id_column in filtered.columns:
        return filtered[id_column].tolist()
    for candidate in ("subtomo_id", "qp_id", "qp_subtomo_id"):
        if candidate in filtered.columns:
            return filtered[candidate].tolist()
    return list(range(len(filtered)))


def resolve_filtered_ids(
    df: pd.DataFrame,
    filter_model: dict,
    slider_filters: dict[str, tuple[float, float]],
) -> list:
    """Return *subtomo_id* values for every row passing all current filters.

    Raises ``ValueError`` if the DataFrame has no *subtomo_id* column — the
    caller must check before inserting a subset into the pool.
    """
    if "subtomo_id" not in df.columns:
        raise ValueError(
            "resolve_filtered_ids: DataFrame has no 'subtomo_id' column."
        )
    filtered = apply_filter_model(df, filter_model, slider_filters)
    return filtered["subtomo_id"].tolist()


def subset_motl_rows(
    df: pd.DataFrame,
    ids: list,
    *,
    store_column: str | None = None,
    values: dict | None = None,
) -> pd.DataFrame:
    """Return rows where *subtomo_id* is in *ids*, original order, no duplicates.

    Parameters
    ----------
    df:
        Source DataFrame — must contain *subtomo_id*.
    ids:
        Sequence of subtomo_id values to keep.
    store_column:
        Optional column name to write per-row values into.
    values:
        Mapping ``{subtomo_id: value}`` written to *store_column* when set.

    Raises ``ValueError`` if the DataFrame has no *subtomo_id* column.
    """
    if "subtomo_id" not in df.columns:
        raise ValueError("subset_motl_rows: DataFrame has no 'subtomo_id' column.")
    id_set = set(ids)
    result = df[df["subtomo_id"].isin(id_set)].copy()
    if store_column is not None and values:
        result[store_column] = result["subtomo_id"].map(values)
    return result


# ── Layout helpers ─────────────────────────────────────────────────────────────


_DEFAULT_COL_DEF = {
    "sortable": True,
    "filter": True,
    "editable": False,
    "resizable": True,
    "flex": 1,  # share available width equally between columns
    "minWidth": 90,  # never narrower — the grid scrolls instead
}

_GRID_STYLE = {"height": "300px", "width": "100%"}

# Client-side row model: rows above this threshold are not sent to the browser.
# Measured with IF1 compression: 150 000 rows → 52 MB, 200 000 rows → ~70 MB,
# 240 000+ rows → 83+ MB which starts causing noticeable browser lag.
# 150 000 rows (a normal working size) is always below the limit.
_MOTL_TABLE_MAX_ROWS = 200_000


def _build_motl_col_defs() -> list[dict]:
    from cryocat.core.cryomotl import Motl
    return [
        {
            "field": col,
            "headerName": col,
            "headerTooltip": col,
            "filter": "agNumberColumnFilter",
            "floatingFilter": False,
            "valueFormatter": {"function": "(params.value != null) ? params.value.toFixed(3) : ''"},
        }
        for col in Motl.motl_columns
    ]


# All 20 motl columns are float64 — built once at import time from Motl.motl_columns.
MOTL_COL_DEFS: list[dict] = _build_motl_col_defs()


def _motl_precision_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-only copy of *df* with per-column precision reduction.

    The stored payload is never touched — only the copy passed to ``to_dict("records")``
    is affected.  No per-row decisions are made: the classification (integer / decimals /
    significant figures) is determined once per column from vectorised stats.

    Rule (applied to every float column):
    - all values whole (arr % 1 == 0) → ``int64``; exact, no rounding artefact
    - otherwise, find the column's smallest non-zero |value|:
        - ``min_abs < 0.001`` → **significant figures** (4 sig figs); enough decimal places
          so that even the tiniest non-zero value in the column is not rounded to zero.
          Cut-off of 0.001: at 3 fixed decimals, 0.0009 rounds to 0.000, so any column
          whose smallest value is below 0.001 would lose information.
        - ``min_abs ≥ 0.001`` → **3 decimal places**; adequate for coordinates, angles,
          and CCC scores in the normal range.
    """
    _SAMPLE = 1_000  # rows checked before any full-array work
    cols: dict = {}
    for col in df.columns:
        arr = df[col].to_numpy()
        # Opt-1: integer-dtype column → nothing to test.
        if arr.dtype.kind in ("i", "u"):
            cols[col] = arr
            continue
        if arr.dtype.kind != "f":
            cols[col] = arr
            continue
        # Opt-2: sample the raw slice first — no isfinite pass yet.
        # nan % 1 is nan, nan != 0 is True, so a NaN in the sample routes the
        # column to the float path immediately, which is the safe outcome.
        sample = arr[:_SAMPLE]
        if (sample % 1 != 0).any():
            # Non-integer column confirmed without touching the full array.
            # Now one O(n) pass for min_abs (isfinite needed for correctness).
            nz = arr[np.isfinite(arr) & (arr != 0)]
            min_abs = float(np.abs(nz).min()) if len(nz) > 0 else 1.0
            if min_abs < 0.001:
                min_mag = int(np.floor(np.log10(min_abs)))
                n_dec = max(0, min(6, 3 - min_mag))
                cols[col] = np.round(arr, n_dec)
            else:
                cols[col] = np.round(arr, 3)
        else:
            # Sample is all integral — run isfinite + full check.
            finite = arr[np.isfinite(arr)]
            if len(finite) == 0 or bool(np.all(finite % 1 == 0)):
                # Keep as float if any NaN/Inf present (int64 cannot represent NaN).
                cols[col] = arr.astype(np.int64) if len(finite) == len(arr) else arr
            else:
                nz = finite[finite != 0]
                min_abs = float(np.abs(nz).min()) if len(nz) > 0 else 1.0
                if min_abs < 0.001:
                    min_mag = int(np.floor(np.log10(min_abs)))
                    n_dec = max(0, min(6, 3 - min_mag))
                    cols[col] = np.round(arr, n_dec)
                else:
                    cols[col] = np.round(arr, 3)
    return pd.DataFrame(cols, index=df.index)


def col_defs_from_df(df: pd.DataFrame) -> list[dict]:
    """Build AG Grid columnDefs from a DataFrame.  Pure."""
    col_defs = []
    for col in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        is_float = pd.api.types.is_float_dtype(df[col])
        col_def = {
            "field": col,
            "headerName": col,
            "headerTooltip": col,
            "filter": "agNumberColumnFilter" if is_numeric else True,
            "floatingFilter": False,
        }
        if is_float:
            col_def["valueFormatter"] = {"function": "(params.value != null) ? params.value.toFixed(3) : ''"}
        col_defs.append(col_def)
    return col_defs


def get_grid_container(prefix: str, *, motl_table: bool = False) -> html.Div:
    """Container holding the grid (built once) and a hidden sink for the purge callback."""
    children: list = [
        html.Div(id=f"{prefix}-purge-sink", style={"display": "none"}),
        dcc.Store(id=f"{prefix}-slider-filters-store", data={}),
        dcc.Store(id=f"{prefix}-col-ref-store"),
    ]
    if motl_table:
        # Notice div receives a message when the row count exceeds _MOTL_TABLE_MAX_ROWS.
        children.append(
            html.Div(
                id=f"{prefix}-rowdata-notice",
                style={
                    "color": "var(--bs-warning, orange)",
                    "fontSize": "0.85rem",
                    "padding": "4px 0",
                },
            )
        )
    children.append(get_grid(prefix, motl_table=motl_table))
    return html.Div(children, id=f"{prefix}-grid-container")


def get_grid(prefix: str, *, motl_table: bool = False) -> dag.AgGrid:
    """Return a static AgGrid.

    When *motl_table=True*, returns a client-side row model grid with static
    MOTL_COL_DEFS already set.  Otherwise returns the standard infinite-scroll
    grid with empty columnDefs.

    Used both by get_grid_container() and by unit tests that register callbacks
    without a real data load.
    """
    if motl_table:
        return dag.AgGrid(
            id=f"{prefix}-grid",
            columnDefs=MOTL_COL_DEFS,
            rowData=[],
            dashGridOptions={"rowSelection": {"mode": "multiRow"}},
            defaultColDef=_DEFAULT_COL_DEF,
            style=_GRID_STYLE,
            className="ag-theme-balham",
        )
    return dag.AgGrid(
        id=f"{prefix}-grid",
        columnDefs=[],
        rowModelType="infinite",
        dashGridOptions={
            "cacheBlockSize": CACHE_BLOCK_SIZE,
            "maxBlocksInCache": 20,
        },
        defaultColDef=_DEFAULT_COL_DEF,
        style=_GRID_STYLE,
        className="ag-theme-balham",
    )


# ── Callbacks ──────────────────────────────────────────────────────────────────


def register_tablegrid_callbacks(
    app,
    prefix: str,
    *,
    resolve_df,
    resolve_n_rows,
    tabs_id: str | None = None,
    tab_value: str | None = None,
    data_pool: bool = False,
    motl_table: bool = False,
) -> None:
    """Register grid callbacks for *prefix*.

    When *motl_table=True*, uses the client-side row model: columnDefs are
    already set at layout time (MOTL_COL_DEFS) so the _cols callback and the
    entire purge chain are skipped; a single _set_rows callback pushes all rows
    to rowData whenever the data store or slider filters change.

    When *motl_table=False* (default), uses the infinite scroll model: _cols
    sets columnDefs when data loads, purgeInfiniteCache fires on column changes,
    and _rows answers each getRowsRequest block.

    Parameters
    ----------
    resolve_df:
        ``(ref) -> pd.DataFrame | None`` — resolves the store reference.
    resolve_n_rows:
        Accepted for API compatibility; not used.
    tabs_id, tab_value:
        Accepted for API compatibility; not used.
    motl_table:
        When True, enables client-side row model with static MOTL_COL_DEFS.
    """

    if motl_table:
        # Client-side row model: all rows pushed once on data load; AG Grid
        # filters/sorts entirely in-browser.  Slider filters are translated to
        # filterModel inRange conditions via a clientside callback so slider
        # interaction costs zero server round-trips.
        @app.callback(
            Output(f"{prefix}-grid", "rowData"),
            Output(f"{prefix}-rowdata-notice", "children"),
            Input(f"{prefix}-global-data-store", "data"),
        )
        def _set_rows(ref):
            if not ref:
                return [], ""
            df = resolve_df(ref)
            if df is None:
                return [], ""
            if len(df) > _MOTL_TABLE_MAX_ROWS:
                msg = (
                    f"This motl has {len(df):,} rows — above the "
                    f"{_MOTL_TABLE_MAX_ROWS:,}-row limit for the in-browser table. "
                    f"Create a subset via the pool (select / filter on the server-side "
                    f"table, then 'Create from filtered') to view a slice here."
                )
                return [], msg
            return _motl_precision_df(df).to_dict("records"), ""

        # Translate slider-filters-store → AG Grid filterModel as inRange conditions.
        # Previous slider columns tracked in window._cryocat_slider_cols[prefix] so
        # that clearing a slider removes its inRange condition from the filterModel.
        app.clientside_callback(
            f"""function(sliderFilters, currentFM) {{
                var fm = Object.assign({{}}, currentFM || {{}});
                window._cryocat_slider_cols = window._cryocat_slider_cols || {{}};
                var prev = window._cryocat_slider_cols["{prefix}"] || [];
                var newCols = Object.keys(sliderFilters || {{}});
                prev.forEach(function(col) {{
                    if (newCols.indexOf(col) < 0) delete fm[col];
                }});
                window._cryocat_slider_cols["{prefix}"] = newCols;
                Object.entries(sliderFilters || {{}}).forEach(function(kv) {{
                    fm[kv[0]] = {{filterType: "number", type: "inRange",
                                  filter: kv[1][0], filterTo: kv[1][1]}};
                }});
                return fm;
            }}""",
            Output(f"{prefix}-grid", "filterModel"),
            Input(f"{prefix}-slider-filters-store", "data"),
            State(f"{prefix}-grid", "filterModel"),
            prevent_initial_call=True,
        )

        # Shared callbacks registered below.

    if not motl_table:
        _col_inputs = [Input(f"{prefix}-global-data-store", "data")]
        if data_pool:
            _col_inputs.append(Input(ids.DATA_POOL_REVS, "data"))
        @app.callback(
            Output(f"{prefix}-grid", "columnDefs"),
            Output(f"{prefix}-grid", "dashGridOptions"),
            Output(f"{prefix}-col-ref-store", "data"),
            *_col_inputs,
            State(f"{prefix}-col-ref-store", "data"),
        )
        def _cols(*cb_args):
            if data_pool:
                ref, revs_data, stored = cb_args
            else:
                ref, stored = cb_args
                revs_data = {}
            stored = stored or {}
            prev_ref = stored.get("ref")
            prev_fields = stored.get("fields", [])
            prev_rev = stored.get("rev", -1)

            if ctx.triggered_id == ids.DATA_POOL_REVS:
                ref = prev_ref
                if not ref:
                    raise exceptions.PreventUpdate
                data_id = ref.get("data_id") if isinstance(ref, dict) else None
                if data_id is None:
                    raise exceptions.PreventUpdate
                live_rev = (revs_data or {}).get(data_id, 0)
                if live_rev == prev_rev:
                    raise exceptions.PreventUpdate
            else:
                if ref == prev_ref:
                    raise exceptions.PreventUpdate

            def _live_rev(r):
                did = r.get("data_id") if isinstance(r, dict) else None
                return (revs_data or {}).get(did, 0) if did else 0

            df = resolve_df(ref)
            if df is None:
                if ref is not None:
                    new_stored = {"ref": ref, "fields": [], "rev": _live_rev(ref)}
                    return [], {
                        **_BASE_GRID_OPTIONS,
                        "overlayNoRowsTemplate": (
                            "<span style='padding:8px;color:var(--bs-warning,orange)'>"
                            "Table reference is stale — this can happen after a server "
                            "restart or an interrupted save. Please reload the data."
                            "</span>"
                        ),
                    }, new_stored
                return no_update, no_update, {"ref": None, "fields": [], "rev": -1}
            new_cols = col_defs_from_df(df)
            new_fields = [c["field"] for c in new_cols]
            new_stored = {"ref": ref, "fields": new_fields, "rev": _live_rev(ref)}
            new_options = {**_BASE_GRID_OPTIONS, "infiniteInitialRowCount": len(df)}
            if new_fields == prev_fields:
                return no_update, no_update, new_stored
            return new_cols, new_options, new_stored

        app.clientside_callback(
            "function(c){if(c===undefined||c===null)return window.dash_clientside.no_update;"
            f'window.dash_ag_grid.getApiAsync("{prefix}-grid")'
            ".then(function(a){if(a)a.purgeInfiniteCache()});"
            "return window.dash_clientside.no_update;}",
            Output(f"{prefix}-purge-sink", "children"),
            Input(f"{prefix}-col-ref-store", "data"),
        )

        app.clientside_callback(
            "function(f){"
            f'window.dash_ag_grid.getApiAsync("{prefix}-grid")'
            ".then(function(a){if(a)a.purgeInfiniteCache()});"
            "return window.dash_clientside.no_update;}",
            Output(f"{prefix}-purge-sink", "children", allow_duplicate=True),
            Input(f"{prefix}-slider-filters-store", "data"),
            prevent_initial_call=True,
        )

        # Empty-slot guard: when the slot has no data, respond to AG Grid's
        # getRowsRequest in the browser (zero server round-trip).  The server
        # _rows callback below guards against ref=None so it never fires for
        # empty slots.
        app.clientside_callback(
            "function(req, ref) {"
            "  if (ref || req === null || req === undefined)"
            "    return window.dash_clientside.no_update;"
            '  return {"rowData": [], "rowCount": 0};'
            "}",
            Output(f"{prefix}-grid", "getRowsResponse", allow_duplicate=True),
            Input(f"{prefix}-grid", "getRowsRequest"),
            State(f"{prefix}-global-data-store", "data"),
            prevent_initial_call=True,
        )

        @app.callback(
            Output(f"{prefix}-grid", "getRowsResponse", allow_duplicate=True),
            Input(f"{prefix}-grid", "getRowsRequest"),
            State(f"{prefix}-global-data-store", "data"),
            State(f"{prefix}-slider-filters-store", "data"),
            prevent_initial_call=True,
        )
        def _rows(request, ref, slider_filters):
            if not ref:
                return no_update  # empty-slot response handled by client-side callback
            if request is None:
                return no_update
            df = resolve_df(ref)
            if df is None:
                return no_update
            return rows_response(request, df, len(df), slider_filters=slider_filters)

    @app.callback(
        Output(f"{prefix}-selection-ids-store", "data", allow_duplicate=True),
        Input(f"{prefix}-grid", "selectedRows"),
        State(f"{prefix}-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def _track_grid_selection(selected_rows, ref):
        if not selected_rows:
            return []
        id_col = _pool.get_id_column(ref)
        if not id_col:
            first = selected_rows[0] or {}
            for candidate in ("subtomo_id", "qp_id", "qp_subtomo_id"):
                if candidate in first:
                    id_col = candidate
                    break
        if not id_col:
            return []
        return [row[id_col] for row in selected_rows if id_col in row]

    @app.callback(
        Output(f"{prefix}-selection-ids-store", "data", allow_duplicate=True),
        Output(f"{prefix}-select-all-btn", "children"),
        Input(f"{prefix}-select-all-btn", "n_clicks"),
        State(f"{prefix}-global-data-store", "data"),
        State(f"{prefix}-grid", "filterModel"),
        State(f"{prefix}-selection-ids-store", "data"),
        State(f"{prefix}-slider-filters-store", "data"),
        prevent_initial_call=True,
    )
    def _toggle_select_all(n_clicks, ref, filter_model, current_ids, slider_filters):
        """Select all filtered rows by identity column, or deselect if already all selected."""
        if not n_clicks:
            raise exceptions.PreventUpdate
        if current_ids:
            return [], "Select All Filtered"
        df = resolve_df(ref)
        if df is None:
            raise exceptions.PreventUpdate
        id_col = _pool.get_id_column(ref)
        # motl_table: slider conditions are already in filter_model as inRange entries
        effective_slider = {} if motl_table else (slider_filters or {})
        ids_list = resolve_select_all_ids(df, filter_model or {}, effective_slider, id_column=id_col)
        total = len(df)
        n = len(ids_list)
        label = f"Deselect All ({n:,})" if n < total else "Deselect All"
        return ids_list, label

    @app.callback(
        Output(f"{prefix}-active-filter-count", "children"),
        Input(f"{prefix}-grid", "filterModel"),
        Input(f"{prefix}-slider-filters-store", "data"),
        State(f"{prefix}-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def _update_filter_count(filter_model, slider_filters, ref):
        effective_slider = {} if motl_table else (slider_filters or {})
        df = resolve_df(ref)
        active_filters = (
            sum(1 for v in (filter_model or {}).values() if v) + len(effective_slider)
        )
        if df is None:
            return f"{active_filters} active filter{'s' if active_filters != 1 else ''}" if active_filters else ""
        total = len(df)
        filtered_df = apply_filter_model(df, filter_model or {}, effective_slider)
        n_filtered = len(filtered_df)
        if active_filters == 0:
            return f"{total:,} rows"
        return f"{n_filtered:,} of {total:,} rows"

    @app.callback(
        Output(f"{prefix}-pool-from-filtered-btn", "children"),
        Output(f"{prefix}-pool-from-filtered-btn", "disabled"),
        Output(f"{prefix}-pool-from-filtered-btn", "style"),
        Input(f"{prefix}-grid", "filterModel"),
        Input(f"{prefix}-slider-filters-store", "data"),
        State(f"{prefix}-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def _update_filtered_btn(filter_model, slider_filters, ref):
        effective_slider = {} if motl_table else (slider_filters or {})
        df = resolve_df(ref)
        if df is None:
            return "Create from filtered", True, {"display": "none"}
        id_col = _pool.get_id_column(ref)
        if not id_col or id_col not in df.columns:
            return "Create from filtered", True, {"display": "none"}
        filtered_df = apply_filter_model(df, filter_model or {}, effective_slider)
        n = len(filtered_df)
        return f"Create from filtered ({n:,})", n == 0, {}

    @app.callback(
        Output(f"{prefix}-pool-from-selected-btn", "children"),
        Output(f"{prefix}-pool-from-selected-btn", "disabled"),
        Output(f"{prefix}-pool-from-selected-btn", "style"),
        Input(f"{prefix}-selection-ids-store", "data"),
        State(f"{prefix}-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def _update_selected_btn(selected_ids, ref):
        df = resolve_df(ref)
        if df is None:
            return "Create from selected", True, {"display": "none"}
        id_col = _pool.get_id_column(ref)
        if not id_col or id_col not in df.columns:
            return "Create from selected", True, {"display": "none"}
        n = len(selected_ids or [])
        return f"Create from selected ({n:,})", n == 0, {}

    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output(f"{prefix}-global-data-store", "data", allow_duplicate=True),
        Input(f"{prefix}-pool-from-filtered-btn", "n_clicks"),
        State(f"{prefix}-global-data-store", "data"),
        State(f"{prefix}-grid", "filterModel"),
        State(f"{prefix}-slider-filters-store", "data"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def _create_from_filtered(n_clicks, ref, filter_model, slider_filters, registry, pool_meta, next_id):
        if not n_clicks:
            raise exceptions.PreventUpdate
        df = resolve_df(ref)
        if df is None:
            raise exceptions.PreventUpdate
        effective_slider = {} if motl_table else (slider_filters or {})
        filtered_df = apply_filter_model(df, filter_model or {}, effective_slider)
        if filtered_df.empty:
            raise exceptions.PreventUpdate
        active_filters = sum(1 for v in (filter_model or {}).values() if v) + len(effective_slider)
        label = f"{_pool.get_entry_label(ref, registry)} {'filtered' if active_filters else 'subset'}"
        return _pool.create_pool_entry(ref, filtered_df, registry, pool_meta, next_id, label=label)

    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output(f"{prefix}-global-data-store", "data", allow_duplicate=True),
        Input(f"{prefix}-pool-from-selected-btn", "n_clicks"),
        State(f"{prefix}-selection-ids-store", "data"),
        State(f"{prefix}-global-data-store", "data"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def _create_from_selected(n_clicks, selected_ids, ref, registry, pool_meta, next_id):
        if not n_clicks or not selected_ids:
            raise exceptions.PreventUpdate
        df = resolve_df(ref)
        if df is None:
            raise exceptions.PreventUpdate
        id_col = _pool.get_id_column(ref)
        if not id_col or id_col not in df.columns:
            raise exceptions.PreventUpdate
        id_set = set(selected_ids)
        subset_df = df[df[id_col].isin(id_set)]
        if subset_df.empty:
            raise exceptions.PreventUpdate
        label = f"{_pool.get_entry_label(ref, registry)} selection"
        return _pool.create_pool_entry(ref, subset_df, registry, pool_meta, next_id, label=label)

    @app.callback(
        Output(f"{prefix}-selection-count", "children"),
        Input(f"{prefix}-selection-ids-store", "data"),
        State(f"{prefix}-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def _update_selection_count(ids_list, ref):
        n_sel = len(ids_list or [])
        if n_sel == 0:
            return ""
        df = resolve_df(ref)
        if df is not None:
            total = len(df)
            if n_sel < total:
                return f"{n_sel:,} rows selected (of {total:,})"
        return f"{n_sel:,} rows selected"
