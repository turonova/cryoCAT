"""AG Grid with infinite row model — server-side sort, filter, and selection.

The grid never receives the full DataFrame.  On each scroll or filter event
AG Grid sends getRowsRequest; the _rows callback applies sort + filter to the
server-side pool entry and returns one 100-row block via getRowsResponse.

Pure functions (apply_sort_model, apply_filter_model, slice_block,
resolve_select_all_ids) are testable without Dash.  Block serialisation goes
through pool.block_to_records so the T3 AST guard is not triggered.
"""
from __future__ import annotations

import pandas as pd
from dash import Input, Output, State, exceptions, no_update
import dash_ag_grid as dag

from cryocat.app import ids
from cryocat.app.pool import (
    _CACHE_BLOCK_SIZE,
    block_to_records,
    get_rows,
    PoolPayloadMissing,
)

CACHE_BLOCK_SIZE = _CACHE_BLOCK_SIZE  # re-exported so test can import from here


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


def rows_response(request: dict | None, df: pd.DataFrame | None) -> dict:
    """Pure getRowsResponse handler.  Never raises; returns empty block when data absent.

    W1 fix: the first ``getRowsRequest`` fires before the pool payload is ready.
    Raising PreventUpdate here leaves the grid holding a permanently empty block
    (the grid does not retry after a no-response).  Returning
    ``{"rowData": [], "rowCount": 0}`` lets the grid proceed; when ``_on_load``
    later sets ``infiniteInitialRowCount`` to the real row count the grid issues
    a fresh block request and the table populates without any user interaction.
    """
    if request is None or df is None:
        return {"rowData": [], "rowCount": 0}
    filter_model = request.get("filterModel") or {}
    sort_model = request.get("sortModel") or []
    start_row = request.get("startRow", 0)
    end_row = request.get("endRow", CACHE_BLOCK_SIZE)
    filtered = apply_filter_model(df, filter_model, {})
    sorted_df = apply_sort_model(filtered, sort_model)
    block = slice_block(sorted_df, start_row, end_row)
    return {
        "rowData": block_to_records(block, max_rows=CACHE_BLOCK_SIZE),
        "rowCount": len(filtered),
    }


def initial_grid_options(n_rows: int) -> dict:
    """Return the ``dashGridOptions`` dict to emit when a motl loads.

    Setting ``infiniteInitialRowCount`` to the real row count signals AG Grid
    that rows exist and triggers an immediate block request (cache-refresh
    signal, W1 wiring).  Must be a full options dict because Dash replaces the
    entire ``dashGridOptions`` property on each write.
    """
    return {
        "cacheBlockSize": CACHE_BLOCK_SIZE,
        "maxBlocksInCache": 4,
        "rowBuffer": 0,
        "infiniteInitialRowCount": n_rows,
        "rowSelection": "multiple",
        "suppressRowClickSelection": True,
    }


def resolve_select_all_ids(
    df: pd.DataFrame,
    filter_model: dict,
    slider_filters: dict[str, tuple[float, float]],
) -> list:
    """Return the *subtomo_id* values for all rows that pass the current filters.

    Used for server-side "select all filtered" (W3).  If *subtomo_id* is absent,
    falls back to the integer row index.
    """
    filtered = apply_filter_model(df, filter_model, slider_filters)
    if "subtomo_id" in filtered.columns:
        return filtered["subtomo_id"].tolist()
    return list(range(len(filtered)))


# ── Layout helpers ─────────────────────────────────────────────────────────────


def col_defs_from_df(df: pd.DataFrame) -> list[dict]:
    """Build AG Grid columnDefs from a DataFrame.  Pure."""
    col_defs = []
    for i, col in enumerate(df.columns):
        col_def = {
            "field": col,
            "headerName": col,
            "headerTooltip": col,
            "checkboxSelection": i == 0,
            "filter": True,
            "floatingFilter": False,
            "minWidth": 80 if i == 0 else None,
        }
        if pd.api.types.is_float_dtype(df[col]):
            col_def["valueFormatter"] = {
                "function": "(params.value != null) ? params.value.toFixed(3) : ''"
            }
        col_defs.append(col_def)
    return col_defs


def get_grid(prefix: str) -> dag.AgGrid:
    """Return an AG Grid component configured for the infinite row model."""
    return dag.AgGrid(
        id=f"{prefix}-grid",
        columnDefs=[],
        rowModelType="infinite",
        dashGridOptions={
            "cacheBlockSize": CACHE_BLOCK_SIZE,
            "maxBlocksInCache": 4,
            "rowBuffer": 0,
            "infiniteInitialRowCount": 0,
            "rowSelection": "multiple",
            "suppressRowClickSelection": True,
        },
        defaultColDef={
            "sortable": True,
            "filter": True,
            "editable": False,
            "resizable": True,
        },
        style={"height": "300px", "width": "100%"},
        className="ag-theme-balham",
        columnSizeOptions={"skipHeader": False},
    )


# ── Callbacks ──────────────────────────────────────────────────────────────────


def register_tablegrid_callbacks(app, prefix: str) -> None:
    @app.callback(
        Output(f"{prefix}-grid", "columnDefs", allow_duplicate=True),
        Output(f"{prefix}-grid", "dashGridOptions", allow_duplicate=True),
        Output(f"{prefix}-grid", "filterModel", allow_duplicate=True),
        Input(f"{prefix}-global-data-store", "data"),
        State(ids.POOL_REGISTRY, "data"),
        prevent_initial_call=True,
    )
    def _on_load(ref, registry):
        """Set column defs + row count when a new motl ref lands; clear filters.

        Diagnosis (W1 — cause (a)): the first getRowsRequest fires at grid
        mount, before this callback has run.  The old _rows raised PreventUpdate,
        so the grid cached an empty block and never retried.  Fix: _rows now
        returns {"rowData": [], "rowCount": 0} instead of raising; this callback
        then updates dashGridOptions.infiniteInitialRowCount to the real count,
        which causes AG Grid to issue a fresh block request immediately.
        """
        if not ref or not isinstance(ref, dict) or not ref.get("motl_id"):
            raise exceptions.PreventUpdate
        motl_id = ref["motl_id"]
        entry = (registry or {}).get(motl_id, {})
        n_rows = entry.get("n_rows", 0)
        try:
            df = get_rows(motl_id)
            n_rows = len(df)
        except PoolPayloadMissing:
            raise exceptions.PreventUpdate
        return (
            col_defs_from_df(df),
            initial_grid_options(n_rows),
            {},  # clear stale filters; filter change also acts as cache-purge signal
        )

    @app.callback(
        Output(f"{prefix}-grid", "getRowsResponse"),
        Input(f"{prefix}-grid", "getRowsRequest"),
        State(f"{prefix}-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def _rows(request, ref):
        """Respond to AG Grid's infinite-model row request with one sorted+filtered block.

        W1 fix: never raises PreventUpdate — returns empty response when data is
        not yet available.  See rows_response() for the contract.
        """
        motl_id = (ref or {}).get("motl_id") if isinstance(ref, dict) else None
        df = None
        if motl_id:
            try:
                df = get_rows(motl_id)
            except PoolPayloadMissing:
                pass
        return rows_response(request, df)

    @app.callback(
        Output(f"{prefix}-grid", "columnSize"),
        Input(f"{prefix}-grid", "columnDefs"),
        prevent_initial_call=True,
    )
    def adapt_column_size(_col):
        return "sizeToFit"

    @app.callback(
        Output(f"{prefix}-selection-ids-store", "data", allow_duplicate=True),
        Output(f"{prefix}-select-all-btn", "children"),
        Input(f"{prefix}-select-all-btn", "n_clicks"),
        State(f"{prefix}-global-data-store", "data"),
        State(f"{prefix}-grid", "filterModel"),
        State(f"{prefix}-selection-ids-store", "data"),
        prevent_initial_call=True,
    )
    def _toggle_select_all(n_clicks, ref, filter_model, current_ids):
        """Select all filtered rows by subtomo_id, or deselect if already all selected."""
        if not n_clicks:
            raise exceptions.PreventUpdate
        motl_id = (ref or {}).get("motl_id")
        if not motl_id:
            raise exceptions.PreventUpdate

        if current_ids:
            return [], "Select All Filtered"

        try:
            df = get_rows(motl_id)
        except PoolPayloadMissing:
            raise exceptions.PreventUpdate

        ids_list = resolve_select_all_ids(df, filter_model or {}, {})
        total = len(df)
        n = len(ids_list)
        label = f"Deselect All ({n:,})" if n < total else "Deselect All"
        return ids_list, label

    @app.callback(
        Output(f"{prefix}-active-filter-count", "children"),
        Input(f"{prefix}-grid", "filterModel"),
        prevent_initial_call=True,
    )
    def _update_filter_count(filter_model):
        n = sum(1 for v in (filter_model or {}).values() if v)
        if n == 0:
            return ""
        return f"{n} active filter{'s' if n != 1 else ''}"

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
        motl_id = (ref or {}).get("motl_id")
        if motl_id:
            try:
                total = len(get_rows(motl_id))
                if n_sel < total:
                    return f"{n_sel:,} rows selected (of {total:,})"
            except PoolPayloadMissing:
                pass
        return f"{n_sel:,} rows selected"
