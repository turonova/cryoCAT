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
from dash import html, Input, Output, State, exceptions, no_update
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


def rows_response(request: dict | None, df: pd.DataFrame | None, n_rows_hint: int = 0) -> dict:
    """Pure getRowsResponse handler.  Never raises.

    W2: when df is absent but the handle says n_rows > 0, return that count
    instead of 0.  rowCount=0 is permanent poison — AG Grid concludes the
    dataset is empty and never requests again.  n_rows_hint comes from the
    pool registry entry; it is 0 only when no motl is loaded (correct) or
    when the entry itself has n_rows=0 (also correct: motl is genuinely empty).

    The normal W1 path (grid remounted with data already present) means
    n_rows_hint is only exercised on hot-reload, when the payload was evicted
    but the handle survives in the browser store.
    """
    if request is None:
        return {"rowData": [], "rowCount": n_rows_hint}
    if df is None:
        return {"rowData": [], "rowCount": n_rows_hint}
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


_DEFAULT_COL_DEF = {
    "sortable": True,
    "filter": True,
    "editable": False,
    "resizable": True,
}

_GRID_STYLE = {"height": "300px", "width": "100%"}


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


def _tab_active(active_tab: str | None, tab_value: str | None) -> bool:
    """Return True when this grid's tab is active (or no tab tracking is configured)."""
    return tab_value is None or active_tab == tab_value


def _build_grid(prefix: str, df: pd.DataFrame) -> dag.AgGrid:
    """Build a fresh AgGrid configured for df.  Pure; no Dash calls.

    W1: returned by _mount_grid as the container's children so AG Grid mounts
    with correct columnDefs and infiniteInitialRowCount before its first
    getRowsRequest fires.  The first request therefore always finds pool data
    present and is answered with real rows.

    columnSize="sizeToFit" is set at construction time so columns fit the page
    width immediately when the grid is mounted into a visible container.
    """
    return dag.AgGrid(
        id=f"{prefix}-grid",
        columnDefs=col_defs_from_df(df),
        rowModelType="infinite",
        dashGridOptions=initial_grid_options(len(df)),
        defaultColDef=_DEFAULT_COL_DEF,
        style=_GRID_STYLE,
        className="ag-theme-balham",
        columnSizeOptions={"skipHeader": False},
        columnSize="sizeToFit",
    )


def get_grid(prefix: str) -> html.Div:
    """Return a container Div with an empty placeholder AgGrid.

    W1: the placeholder has columnDefs=[] and infiniteInitialRowCount=0 so the
    grid makes no initial getRowsRequest (zero rows, nothing to ask for).
    _mount_grid replaces it with a fully-configured grid when a motl loads;
    the new grid's first request arrives after the pool payload is present.
    The grid id must be in the static layout so callbacks registered against it
    are accepted (suppress_callback_exceptions=True still requires the id at
    callback-registration time to avoid the ID-resolution test failures).
    """
    return html.Div(
        dag.AgGrid(
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
            defaultColDef=_DEFAULT_COL_DEF,
            style=_GRID_STYLE,
            className="ag-theme-balham",
            columnSizeOptions={"skipHeader": False},
        ),
        id=f"{prefix}-grid-container",
    )


# ── Callbacks ──────────────────────────────────────────────────────────────────


def register_tablegrid_callbacks(
    app,
    prefix: str,
    *,
    tabs_id: str | None = None,
    tab_value: str | None = None,
) -> None:
    """Register all AG Grid callbacks for *prefix*.

    When *tabs_id* and *tab_value* are supplied, ``_mount_grid`` is also
    triggered by tab-activation events and skips mounting when the grid's tab
    is not the active one.  This prevents AG Grid from being constructed into
    a ``display: none`` container (inactive ``dbc.Tab`` pane), which would
    suppress the first ``getRowsRequest`` and break column fitting.
    """
    _mount_inputs = [Input(f"{prefix}-global-data-store", "data")]
    if tabs_id:
        _mount_inputs.append(Input(tabs_id, "active_tab"))

    @app.callback(
        Output(f"{prefix}-grid-container", "children"),
        *_mount_inputs,
        State(ids.POOL_REGISTRY, "data"),
        prevent_initial_call=True,
    )
    def _mount_grid(*args):
        """W1: mount the grid only when its tab is active; skip for invisible tabs.

        Without tab awareness _mount_grid fires for all slots on every pool
        change, including slots whose dbc.Tab is hidden (display: none).
        AG Grid in a zero-size viewport never fires getRowsRequest and cannot
        run sizeColumnsToFit, explaining both the missing rows and the wide
        column regression.

        With tabs_id/tab_value: the callback also fires when the user switches
        to this grid's tab (Input on active_tab), and raises PreventUpdate when
        a different tab is active so the container stays empty.  When the tab
        becomes active with pool data already loaded, _mount_grid builds the
        grid into a now-visible container → getRowsRequest fires immediately.
        """
        if tabs_id:
            ref, active_tab, registry = args
            if not _tab_active(active_tab, tab_value):
                raise exceptions.PreventUpdate
        else:
            ref, registry = args
        if not ref or not isinstance(ref, dict) or not ref.get("motl_id"):
            raise exceptions.PreventUpdate
        motl_id = ref["motl_id"]
        try:
            df = get_rows(motl_id)
        except PoolPayloadMissing:
            raise exceptions.PreventUpdate
        return _build_grid(prefix, df)

    @app.callback(
        Output(f"{prefix}-grid", "getRowsResponse"),
        Input(f"{prefix}-grid", "getRowsRequest"),
        State(f"{prefix}-global-data-store", "data"),
        State(ids.POOL_REGISTRY, "data"),
        prevent_initial_call=True,
    )
    def _rows(request, ref, registry):
        """Respond to AG Grid's infinite-model row request with one sorted+filtered block.

        W2: rowCount comes from the handle when df is absent (hot-reload edge
        case) so the grid never sees rowCount=0 for a non-empty motl.
        """
        motl_id = (ref or {}).get("motl_id") if isinstance(ref, dict) else None
        df = None
        n_rows_hint = 0
        if motl_id:
            n_rows_hint = (registry or {}).get(motl_id, {}).get("n_rows", 0)
            try:
                df = get_rows(motl_id)
            except PoolPayloadMissing:
                pass
        return rows_response(request, df, n_rows_hint)

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
