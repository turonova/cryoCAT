"""S1 scratch app — prove that updating columnDefs in-place triggers a fresh getRowsRequest.

Two dataframes with different columns.  A button switches the active one.
A single _cols callback updates columnDefs (and dashGridOptions) when the store changes.
A single _rows callback answers getRowsRequest using the current store value as State.

Questions to answer in the browser:
  1. Do rows appear initially (df_a: id / name, 60 000 rows)?
  2. After pressing "Switch dataframe", do the columns change to alpha / beta / gamma?
  3. Does a new REQUEST line print in the terminal, and do the rows change to df_b values?

If (3) is no: uncomment the dashGridOptions output line in _cols and retest.
"""

import dash_ag_grid as dag
from dash import Dash, Input, Output, State, html, dcc, no_update, callback
import pandas as pd

df_a = pd.DataFrame({
    "id":   range(60_000),
    "name": [f"row {i}" for i in range(60_000)],
})
df_b = pd.DataFrame({
    "alpha": range(1_000),
    "beta":  [f"b{i}" for i in range(1_000)],
    "gamma": [i * 1.5 for i in range(1_000)],
})
FRAMES = {"a": df_a, "b": df_b}

_BASE_OPTIONS = {"cacheBlockSize": 100, "maxBlocksInCache": 4}


def _col_defs(df: pd.DataFrame) -> list[dict]:
    return [{"field": col, "minWidth": 100} for col in df.columns]


app = Dash(__name__)
app.layout = html.Div([
    html.H4("column-switch test — infinite row model"),
    dcc.Store(id="active-store", data="a"),
    html.Button("Switch dataframe", id="switch-btn", n_clicks=0),
    html.P(id="status-label", children="Active: a  (60,000 rows, columns: id / name)"),
    html.Div(id="purge-sink", style={"display": "none"}),
    dag.AgGrid(
        id="grid",
        columnDefs=[],                          # filled by _cols on first store fire
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        rowModelType="infinite",
        dashGridOptions={**_BASE_OPTIONS, "infiniteInitialRowCount": len(df_a)},
        style={"height": "400px", "width": "100%"},
    ),
])


@callback(
    Output("active-store", "data"),
    Output("status-label", "children"),
    Input("switch-btn", "n_clicks"),
    State("active-store", "data"),
    prevent_initial_call=True,
)
def switch(_, current):
    nxt = "b" if current == "a" else "a"
    df = FRAMES[nxt]
    return nxt, f"Active: {nxt}  ({len(df):,} rows, columns: {' / '.join(df.columns)})"


@callback(
    Output("grid", "columnDefs"),
    Output("grid", "dashGridOptions"),
    Input("active-store", "data"),
)
def _cols(which):
    if not which:
        return no_update, no_update
    df = FRAMES[which]
    return _col_defs(df), {**_BASE_OPTIONS, "infiniteInitialRowCount": len(df)}


@callback(
    Output("grid", "getRowsResponse"),
    Input("grid", "getRowsRequest"),
    State("active-store", "data"),
)
def _rows(request, which):
    print(f"REQUEST {request}")
    if request is None:
        return no_update
    df = FRAMES.get(which or "a")
    partial = df.iloc[request["startRow"]:request["endRow"]]
    return {"rowData": partial.to_dict("records"), "rowCount": len(df)}


# Purge the infinite-row-model cache whenever columnDefs change so AG Grid
# fires a fresh getRowsRequest for the new dataset.
app.clientside_callback(
    """
    function(col_defs) {
        if (!col_defs || col_defs.length === 0) return window.dash_clientside.no_update;
        window.dash_ag_grid.getApiAsync("grid").then(function(api) {
            if (api) api.purgeInfiniteCache();
        });
        return window.dash_clientside.no_update;
    }
    """,
    Output("purge-sink", "children"),
    Input("grid", "columnDefs"),
)


if __name__ == "__main__":
    app.run(debug=False, port=8051)
