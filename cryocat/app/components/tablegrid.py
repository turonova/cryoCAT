"""AG Grid with infinite row model — server-side sort, filter, and selection.

The grid never receives the full DataFrame. On each scroll or filter event
AG Grid sends getRowsRequest; the _rows callback applies sort + filter to the
server-side pool entry and returns one 100-row block via getRowsResponse.

Pure functions (apply_sort_model, apply_filter_model, slice_block,
resolve_select_all_ids) are testable without Dash. Block serialisation goes
through pool.block_to_records so the T3 AST guard is not triggered.
"""

from __future__ import annotations

import pandas as pd
from dash import Input, Output, State, exceptions, html, dcc, no_update
import dash_ag_grid as dag

from cryocat.app import ids
from cryocat.app.pool import (
    _CACHE_BLOCK_SIZE,
    block_to_records,
    insert_motl,
    PoolState,
)

CACHE_BLOCK_SIZE = _CACHE_BLOCK_SIZE  # re-exported so test can import from here

_BASE_GRID_OPTIONS = {"cacheBlockSize": CACHE_BLOCK_SIZE, "maxBlocksInCache": 4, "rowSelection": "multiple"}


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
    print(f"_ROWS filter_in={len(df)} filterModel={filter_model!r} slider_filters={slider_filters!r}")
    filtered = apply_filter_model(df, filter_model, slider_filters or {})
    print(f"_ROWS filter_out={len(filtered)} applied={list(filter_model.keys())!r}")
    sorted_df = apply_sort_model(filtered, sort_model)
    block = slice_block(sorted_df, start_row, end_row)
    resp = {
        "rowData": block_to_records(block, max_rows=CACHE_BLOCK_SIZE),
        "rowCount": len(filtered),
    }
    print(f"_ROWS resp_n={len(resp['rowData'])} rowCount={resp['rowCount']}")
    return resp


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
            "resolve_filtered_ids: DataFrame has no 'subtomo_id' column; " "cannot create a stable motl subset."
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


def col_defs_from_df(df: pd.DataFrame) -> list[dict]:
    """Build AG Grid columnDefs from a DataFrame.  Pure."""
    col_defs = []
    for col in df.columns:
        is_float = pd.api.types.is_float_dtype(df[col])
        col_def = {
            "field": col,
            "headerName": col,
            "headerTooltip": col,
            "filter": True,
            "floatingFilter": False,
            # "minWidth": 20 if is_float else 30,
        }
        if is_float:
            col_def["valueFormatter"] = {"function": "(params.value != null) ? params.value.toFixed(3) : ''"}
        col_defs.append(col_def)
    return col_defs


def get_grid_container(prefix: str) -> html.Div:
    """Container holding the grid (built once) and a hidden sink for the purge callback."""
    return html.Div(
        [
            html.Div(id=f"{prefix}-purge-sink", style={"display": "none"}),
            dcc.Store(id=f"{prefix}-slider-filters-store", data={}),
            get_grid(prefix),
        ],
        id=f"{prefix}-grid-container",
    )


def get_grid(prefix: str) -> dag.AgGrid:
    """Return a static AgGrid with empty columnDefs and no columnSize.

    Used both by get_grid_container() (to pre-render an inert placeholder grid)
    and by unit tests that register callbacks without a real data load.
    """
    return dag.AgGrid(
        id=f"{prefix}-grid",
        columnDefs=[],
        rowModelType="infinite",
        dashGridOptions={
            "cacheBlockSize": CACHE_BLOCK_SIZE,
            "maxBlocksInCache": 4,
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
) -> None:
    """Register grid callbacks for *prefix*.

    The grid lives in the layout from startup (via get_grid_container) with
    columnDefs=[].  Two callbacks drive it: _cols sets real columns when data
    loads; a clientside purgeInfiniteCache fires on the column change so AG Grid
    issues a fresh getRowsRequest; _rows answers that request from the pool.

    Parameters
    ----------
    resolve_df:
        ``(ref) -> pd.DataFrame | None`` — resolves the store reference.
    resolve_n_rows:
        Accepted for API compatibility; not used.
    tabs_id, tab_value:
        Accepted for API compatibility; not used.
    """

    @app.callback(
        Output(f"{prefix}-grid", "columnDefs"),
        Output(f"{prefix}-grid", "dashGridOptions"),
        Input(f"{prefix}-global-data-store", "data"),
    )
    def _cols(ref):
        df = resolve_df(ref)
        if df is None:
            return no_update, no_update
        return col_defs_from_df(df), {**_BASE_GRID_OPTIONS, "infiniteInitialRowCount": len(df)}

    app.clientside_callback(
        "function(c){if(!c||!c.length)return window.dash_clientside.no_update;"
        f'window.dash_ag_grid.getApiAsync("{prefix}-grid")'
        ".then(function(a){if(a)a.purgeInfiniteCache()});"
        "return window.dash_clientside.no_update;}",
        Output(f"{prefix}-purge-sink", "children"),
        Input(f"{prefix}-grid", "columnDefs"),
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

    @app.callback(
        Output(f"{prefix}-grid", "getRowsResponse"),
        Input(f"{prefix}-grid", "getRowsRequest"),
        State(f"{prefix}-global-data-store", "data"),
        State(f"{prefix}-slider-filters-store", "data"),
    )
    def _rows(request, ref, slider_filters):
        print(f"_ROWS prefix={prefix} fired request={'present' if request else 'None'} "
              f"filterModel={request.get('filterModel') if request else 'N/A'!r} "
              f"slider_filters={slider_filters!r}")
        if request is None or not ref:
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
        id_col = (ref or {}).get("id_column") if isinstance(ref, dict) else None
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
        id_col = (ref or {}).get("id_column") if isinstance(ref, dict) else None
        ids_list = resolve_select_all_ids(df, filter_model or {}, slider_filters or {}, id_column=id_col)
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
        df = resolve_df(ref)
        active_filters = (
            sum(1 for v in (filter_model or {}).values() if v) + len(slider_filters or {})
        )
        if df is None:
            return f"{active_filters} active filter{'s' if active_filters != 1 else ''}" if active_filters else ""
        total = len(df)
        filtered_df = apply_filter_model(df, filter_model or {}, slider_filters or {})
        n_filtered = len(filtered_df)
        if active_filters == 0:
            return f"{total:,} rows"
        return f"{n_filtered:,} of {total:,} rows"

    @app.callback(
        Output(f"{prefix}-pool-from-filtered-btn", "children"),
        Output(f"{prefix}-pool-from-filtered-btn", "disabled"),
        Output(f"{prefix}-pool-from-filtered-btn", "style"),
        Input(f"{prefix}-grid", "filterModel"),
        Input(f"{prefix}-global-data-store", "data"),
        Input(f"{prefix}-slider-filters-store", "data"),
    )
    def _update_filtered_btn(filter_model, ref, slider_filters):
        df = resolve_df(ref)
        if df is None:
            return "Create from filtered", True, {"display": "none"}
        id_col = (ref or {}).get("id_column") if isinstance(ref, dict) else None
        if not id_col and not any(c in df.columns for c in ("subtomo_id", "qp_id", "qp_subtomo_id")):
            return "Create from filtered", True, {"display": "none"}
        filtered_df = apply_filter_model(df, filter_model or {}, slider_filters or {})
        n = len(filtered_df)
        return f"Create from filtered ({n:,})", n == 0, {}

    @app.callback(
        Output(f"{prefix}-pool-from-selected-btn", "children"),
        Output(f"{prefix}-pool-from-selected-btn", "disabled"),
        Output(f"{prefix}-pool-from-selected-btn", "style"),
        Input(f"{prefix}-selection-ids-store", "data"),
        Input(f"{prefix}-global-data-store", "data"),
    )
    def _update_selected_btn(selected_ids, ref):
        df = resolve_df(ref)
        if df is None:
            return "Create from selected", True, {"display": "none"}
        id_col = (ref or {}).get("id_column") if isinstance(ref, dict) else None
        if not id_col and not any(c in df.columns for c in ("subtomo_id", "qp_id", "qp_subtomo_id")):
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
        active_filters = (
            sum(1 for v in (filter_model or {}).values() if v) + len(slider_filters or {})
        )
        if isinstance(ref, dict) and ref.get("data_id"):
            from cryocat.app import datapool as _datapool
            id_col = ref.get("id_column")
            if not id_col or id_col not in df.columns:
                raise exceptions.PreventUpdate
            filtered_df = apply_filter_model(df, filter_model or {}, slider_filters or {})
            if filtered_df.empty:
                raise exceptions.PreventUpdate
            label = f"{ref.get('label', 'Data')} filtered" if active_filters else f"{ref.get('label', 'Data')} subset"
            new_ref = _datapool.insert(filtered_df, label=label, id_column=id_col)
            return no_update, no_update, no_update, new_ref
        try:
            filtered_ids = resolve_filtered_ids(df, filter_model or {}, slider_filters or {})
        except ValueError:
            raise exceptions.PreventUpdate
        if not filtered_ids:
            raise exceptions.PreventUpdate
        subset_df = subset_motl_rows(df, filtered_ids)
        state = PoolState.from_stores(registry, pool_meta, next_id)
        motl_id = (ref or {}).get("motl_id") if isinstance(ref, dict) else None
        source_label = (registry or {}).get(motl_id, {}).get("label", "Data") if motl_id else "Data"
        label = f"{source_label} filtered" if active_filters else f"{source_label} subset"
        new_state, _ = insert_motl(state, subset_df, label=label)
        return *new_state.to_stores(), no_update

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
        if isinstance(ref, dict) and ref.get("data_id"):
            from cryocat.app import datapool as _datapool
            id_col = ref.get("id_column")
            if not id_col or id_col not in df.columns:
                raise exceptions.PreventUpdate
            id_set = set(selected_ids)
            subset_df = df[df[id_col].isin(id_set)]
            if subset_df.empty:
                raise exceptions.PreventUpdate
            new_ref = _datapool.insert(subset_df, label=f"{ref.get('label', 'Data')} selection", id_column=id_col)
            return no_update, no_update, no_update, new_ref
        subset_df = subset_motl_rows(df, selected_ids)
        if subset_df.empty:
            raise exceptions.PreventUpdate
        state = PoolState.from_stores(registry, pool_meta, next_id)
        motl_id = (ref or {}).get("motl_id") if isinstance(ref, dict) else None
        source_label = (registry or {}).get(motl_id, {}).get("label", "Data") if motl_id else "Data"
        new_state, _ = insert_motl(state, subset_df, label=f"{source_label} selection")
        return *new_state.to_stores(), no_update

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
