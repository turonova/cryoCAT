"""Per-object dataframe results slot: static table, Save CSV, Send to editor.

One slot per object (surface or complex).  Switching the active object shows
that object's result; running again on the same object replaces it; other
objects' results are untouched.

``{prefix}-obj-results`` — dict store ``{object_id: slot_dict}`` owned by the
page; written by the page's run callback.
``{prefix}-slot-data``   — derived active-object slot, driven by this module.

Static table rendering pattern
    from ``passembly._records_table``
    (``cryocat/app/suite/pages/passembly.py``).
Send-to-editor control pattern
    from ``motlsink.get_send_to_editor_button``
    (``cryocat/app/components/motlsink.py``),
    routed to the data pool instead of the motl pool.
"""
from __future__ import annotations

import pandas as pd
from dash import html, dcc, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from cryocat.app import ids
import cryocat.app.datapool as _datapool


# ── Layout ─────────────────────────────────────────────────────────────────────

def get_results_slot(prefix: str) -> html.Div:
    """Slot layout: static table + Save CSV + Send to editor.

    The page callback writes ``{object_id: {"records": list[dict], "label": str,
    "n_rows": int, "pool_label": str}}`` to ``f"{prefix}-obj-results"``.
    This module projects the active object's entry to ``f"{prefix}-slot-data"``
    which drives display and TTM gating.

    Parameters
    ----------
    prefix:
        ID prefix for all elements in this component.
    """
    return html.Div([
        dcc.Store(id=f"{prefix}-obj-results", data={}),
        dcc.Store(id=f"{prefix}-slot-data"),
        dcc.Download(id=f"{prefix}-slot-download"),
        html.Div(id=f"{prefix}-slot-display"),
        html.Div(
            [
                dbc.Button(
                    "Save CSV",
                    id=f"{prefix}-slot-save-btn",
                    size="sm",
                    color="secondary",
                    className="me-1",
                ),
                dbc.Input(
                    id=f"{prefix}-slot-label",
                    placeholder="variable name",
                    size="sm",
                    style={"width": "120px", "display": "inline-block"},
                    className="me-1",
                ),
                dbc.Button(
                    "Send to editor",
                    id=f"{prefix}-slot-send-btn",
                    size="sm",
                    color="primary",
                    className="me-1",
                ),
                html.Span(
                    id=f"{prefix}-slot-status",
                    style={"fontSize": "0.78rem", "color": "#555"},
                ),
            ],
            style={"display": "flex", "alignItems": "center", "flexWrap": "wrap",
                   "gap": "2px", "marginTop": "6px"},
        ),
    ])


# ── Rendering helper ───────────────────────────────────────────────────────────

def _render_display(slot_data: dict | None) -> list:
    if not slot_data or not slot_data.get("records"):
        return [html.Small("No results yet.", style={"color": "#888"})]

    records: list[dict] = slot_data["records"]
    label: str = slot_data.get("label", "Results")
    n_rows: int = slot_data.get("n_rows", len(records))
    cols = list(records[0].keys()) if records else []

    header_cells = [
        html.Th(c, style={"padding": "2px 8px", "fontWeight": "bold",
                           "borderBottom": "1px solid #dee2e6", "whiteSpace": "nowrap"})
        for c in cols
    ]
    body_rows = [
        html.Tr([
            html.Td(
                str(r.get(c, "")),
                style={"padding": "2px 8px", "fontSize": "0.8rem", "whiteSpace": "nowrap"},
            )
            for c in cols
        ])
        for r in records
    ]

    table = html.Div(
        html.Table(
            [html.Thead(html.Tr(header_cells)), html.Tbody(body_rows)],
            style={"borderCollapse": "collapse", "width": "100%"},
        ),
        style={
            "maxHeight": "340px",
            "overflowY": "auto",
            "overflowX": "auto",
            "border": "1px solid #dee2e6",
            "borderRadius": "4px",
            "padding": "4px",
        },
    )

    shown = len(records)
    suffix = f" (showing {shown})" if shown < n_rows else ""
    header_div = html.Div(
        f"{label} — {n_rows} rows{suffix}",
        style={"fontWeight": "600", "fontSize": "0.85rem", "marginBottom": "4px"},
    )
    return [header_div, table]


# ── Callbacks ──────────────────────────────────────────────────────────────────

def register_results_slot_callbacks(
    app,
    prefix: str,
    active_id_store_id: str,
) -> None:
    """Register display, CSV-download, and send-to-editor callbacks for *prefix*.

    Parameters
    ----------
    active_id_store_id:
        ID of the dcc.Store that holds the currently-active object id (e.g.
        ``"surfaces-selected"`` or ``_CPX_SEL``).  Switching the active object
        immediately updates the display to that object's result.
    """

    @app.callback(
        Output(f"{prefix}-slot-data", "data"),
        Input(f"{prefix}-obj-results", "data"),
        Input(active_id_store_id, "data"),
    )
    def _project_active(obj_results, active_id):
        if not active_id or not obj_results:
            return None
        return (obj_results or {}).get(active_id)

    @app.callback(
        Output(f"{prefix}-slot-display", "children"),
        Input(f"{prefix}-slot-data", "data"),
    )
    def _update_display(slot_data):
        return _render_display(slot_data)

    @app.callback(
        Output(f"{prefix}-slot-download", "data"),
        Input(f"{prefix}-slot-save-btn", "n_clicks"),
        State(f"{prefix}-slot-data", "data"),
        prevent_initial_call=True,
    )
    def _download_csv(_, slot_data):
        if not slot_data or not slot_data.get("records"):
            raise PreventUpdate
        df = pd.DataFrame(slot_data["records"])
        label = slot_data.get("label", "results").replace(" ", "_").replace("/", "_")
        return dcc.send_data_frame(df.to_csv, f"{label}.csv", index=False)

    @app.callback(
        Output(ids.DATA_POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.DATA_POOL_NEXT_ID, "data", allow_duplicate=True),
        Output(f"{prefix}-slot-status", "children"),
        Input(f"{prefix}-slot-send-btn", "n_clicks"),
        State(f"{prefix}-slot-data", "data"),
        State(f"{prefix}-slot-label", "value"),
        State(ids.DATA_POOL_REGISTRY, "data"),
        State(ids.DATA_POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def _send_to_editor(_, slot_data, label_val, dp_registry, dp_next_id):
        if not slot_data or not slot_data.get("records"):
            return no_update, no_update, "No results to send."
        df = pd.DataFrame(slot_data["records"])
        entry_label = (
            (label_val or "").strip()
            or slot_data.get("pool_label")
            or slot_data.get("label", "results")
        )
        state = _datapool.DataPoolState.from_stores(dp_registry, dp_next_id)
        state, entry_id = _datapool.insert_entry(
            state, df,
            label=entry_label,
            reader="dataframe",
            source_path="",
        )
        new_reg, new_next = state.to_stores()
        return new_reg, new_next, f"Sent '{entry_label}' → {entry_id}."
