import dash_ag_grid as dag
from dash import Dash, Input, Output, html, no_update, callback
import pandas as pd

df = pd.DataFrame({"id": range(60000), "name": [f"row {i}" for i in range(60000)]})

app = Dash()
app.layout = html.Div(
    [
        html.H4("infinite row model test"),
        dag.AgGrid(
            id="grid",
            columnDefs=[{"field": "id"}, {"field": "name"}],
            defaultColDef={"sortable": True, "filter": True, "resizable": True, "minWidth": 100},
            rowModelType="infinite",
            columnSize="sizeToFit",
            style={"height": "400px", "width": "100%"},
        ),
    ]
)


@callback(Output("grid", "getRowsResponse"), Input("grid", "getRowsRequest"))
def infinite_scroll(request):
    print("REQUEST:", request)
    if request is None:
        return no_update
    partial = df.iloc[request["startRow"] : request["endRow"]]
    return {"rowData": partial.to_dict("records"), "rowCount": len(df.index)}


if __name__ == "__main__":
    app.run(debug=False, port=8051)
