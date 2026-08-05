"""Global variable-picker modal for @-variable references in form fields (Part F).

One modal is mounted at app level in the suite app (D1 — one modal, not one
per field).  Every text-field ``@`` button opens it by writing to
:data:`~cryocat.app.ids.VAR_PICKER_REQUEST`.  Selecting a variable writes
``{"owner": <json-cid>, "value": "@name"}`` to
:data:`~cryocat.app.ids.VAR_PICKER_RESULT`; per-form write-back callbacks
(registered via :func:`cryocat.app.formgen.register_var_picker_writeback`)
route the result to the correct text input.

Layout
------
* ``dcc.Store(id=VAR_PICKER_REQUEST)`` — input store; written by ``@`` buttons.
* ``dcc.Store(id=VAR_PICKER_RESULT)``  — output store; written on selection.
* ``dbc.Modal(id="var-picker-modal")`` — the picker UI.

Callbacks (registered by :func:`register_var_picker_callbacks`)
--------------------------------------------------------------
``_open_picker``
    Triggered by any ``{"type": "var-picker-btn"}`` click.  Reads pool stores,
    builds the variable list, opens the modal.

``_select_var``
    Triggered by clicking a variable item or the Cancel button.  Writes result,
    closes modal.
"""
from __future__ import annotations

import dash
from dash import html, dcc, Input, Output, State, ALL, no_update, ctx
import dash_bootstrap_components as dbc

from cryocat.app import ids as _ids


def get_var_picker_modal() -> list:
    """Return the app-level components for the variable picker.

    Mount the returned list in the app layout alongside the file browser and
    rotation modal.
    """
    return [
        dcc.Store(id=_ids.VAR_PICKER_REQUEST, data=None),
        dcc.Store(id=_ids.VAR_PICKER_RESULT, data=None),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Session variables")),
                dbc.ModalBody(
                    id="var-picker-body",
                    children=[html.Div("Loading…")],
                    style={"maxHeight": "60vh", "overflowY": "auto"},
                ),
                dbc.ModalFooter(
                    dbc.Button("Cancel", id="var-picker-cancel", color="secondary", size="sm")
                ),
            ],
            id="var-picker-modal",
            is_open=False,
            size="md",
            scrollable=True,
        ),
    ]


def _build_picker_body(bound: dict, target_owner: str):
    """Build the modal body for the variable picker.

    Extracted from the callback so the ``for`` loop doesn't violate §3.
    """
    if not bound:
        return html.Div(
            [
                html.P("No session variables bound yet.", style={"marginBottom": "0.25rem"}),
                html.Small(
                    "Load a motl through the pool, or assign a variable in the console.",
                    style={"color": "var(--color9)"},
                ),
            ]
        )
    rows = []
    for name in sorted(bound.keys()):
        type_name = type(bound[name]).__name__
        rows.append(
            dbc.Button(
                [
                    html.Code(f"@{name}", style={"marginRight": "0.5rem"}),
                    html.Small(type_name, style={"color": "var(--color9)"}),
                ],
                id={"type": "var-picker-item", "name": name, "owner": target_owner},
                color="link",
                style={
                    "textAlign": "left",
                    "width": "100%",
                    "padding": "0.25rem 0.5rem",
                    "borderBottom": "1px solid var(--bs-border-color)",
                },
            )
        )
    return html.Div(rows)


def register_var_picker_callbacks(app) -> None:
    """Register the variable-picker open / select callbacks.

    Parameters
    ----------
    app:
        The Dash app instance (suite app).
    """

    @app.callback(
        Output("var-picker-modal", "is_open"),
        Output("var-picker-body", "children"),
        Output(_ids.VAR_PICKER_REQUEST, "data"),
        Input({"type": "var-picker-btn", "owner": ALL}, "n_clicks"),
        State(_ids.POOL_REGISTRY, "data"),
        State(_ids.POOL_META, "data"),
        State(_ids.POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def _open_picker(n_clicks_list, registry, pool_meta, next_id):
        trigger = ctx.triggered_id
        if not trigger or not any(n for n in n_clicks_list if n):
            raise dash.exceptions.PreventUpdate
        target_owner = trigger.get("owner", "")
        from cryocat.app.pool import PoolState
        from cryocat.app.apputils import _bound_session_vars
        pool_state = PoolState.from_stores(registry or {}, pool_meta or {}, next_id or 0)
        bound = _bound_session_vars(pool_state)
        return True, _build_picker_body(bound, target_owner), target_owner

    @app.callback(
        Output(_ids.VAR_PICKER_RESULT, "data"),
        Output("var-picker-modal", "is_open", allow_duplicate=True),
        Input({"type": "var-picker-item", "name": ALL, "owner": ALL}, "n_clicks"),
        Input("var-picker-cancel", "n_clicks"),
        prevent_initial_call=True,
    )
    def _select_var(item_clicks, cancel_clicks):
        trigger = ctx.triggered_id

        if not trigger:
            raise dash.exceptions.PreventUpdate

        if trigger == "var-picker-cancel":
            return no_update, False

        if isinstance(trigger, dict) and trigger.get("type") == "var-picker-item":
            if not any(n for n in item_clicks if n):
                raise dash.exceptions.PreventUpdate
            name = trigger.get("name", "")
            owner = trigger.get("owner", "")
            return {"owner": owner, "value": f"@{name}"}, False

        raise dash.exceptions.PreventUpdate
