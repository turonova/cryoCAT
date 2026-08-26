"""App-level orientation-picker modal.

One instance per app — mounted once in app.py.  Any number of direction
fields can open it by firing an ``orient-pick-btn`` pattern id.  The modal
writes the chosen unit vector back to the requesting TripletLike control via
a pattern-matching Output that matches only 4-key TripletLike ids.

Mount point:  get_orient_modal()           → app layout
Callbacks  :  register_orient_modal_callbacks(app)  → called once per app

W3 — orientation picker modal (ORIENTATION_PICKER_PLACEMENT.md)
"""
from __future__ import annotations

from dash import html, dcc, Input, Output, State, ALL, no_update, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from cryocat.app import ids
from cryocat.app.components.orientpicker import (
    get_orientation_picker_panel,
    register_orientation_picker_callbacks,
    _normalize,
)

_INNER = "orient-modal-inner"
_REGISTERED_APPS: set[int] = set()


def get_orient_modal() -> html.Div:
    """Single app-level orientation-picker modal.  Mount once in the app layout."""
    return html.Div([
        dcc.Store(id=ids.ORIENT_REQUEST),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Pick direction")),
                dbc.ModalBody(
                    get_orientation_picker_panel(
                        _INNER, mode="direction", show_structure=True, height="440px"
                    )
                ),
                dbc.ModalFooter([
                    dbc.Button(
                        "Use this direction",
                        id="orient-modal-use-btn",
                        color="primary",
                        className="me-2",
                    ),
                    dbc.Button("Close", id="orient-modal-close-btn", color="secondary"),
                ]),
            ],
            id="orient-modal",
            size="xl",
            is_open=False,
            centered=True,
        ),
    ])


def register_orient_modal_callbacks(app) -> None:
    """Register open/close and write-back callbacks.  Call exactly once per app."""
    app_key = id(app)
    if app_key in _REGISTERED_APPS:
        raise RuntimeError(
            "register_orient_modal_callbacks already called for this app instance. "
            f"Mount and register the orientation modal exactly once per app. "
            f"(inner prefix: {_INNER!r})"
        )
    _REGISTERED_APPS.add(app_key)
    register_orientation_picker_callbacks(app, _INNER, mode="direction", show_structure=True)

    @app.callback(
        Output("orient-modal", "is_open"),
        Output(ids.ORIENT_REQUEST, "data"),
        Input({"type": "orient-pick-btn", "owner": ALL, "param": ALL}, "n_clicks"),
        Input("orient-modal-close-btn", "n_clicks"),
        Input("orient-modal-use-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _open_modal(pick_clicks, _close, _use):
        tid = ctx.triggered_id
        if isinstance(tid, dict) and tid.get("type") == "orient-pick-btn":
            return True, {"target": tid}
        return False, no_update

    @app.callback(
        Output({"type": ALL, "owner": ALL, "param": ALL, "tag": "TripletLike"}, "value"),
        Input("orient-modal-use-btn", "n_clicks"),
        State(f"{_INNER}-value", "data"),
        State(ids.ORIENT_REQUEST, "data"),
        prevent_initial_call=True,
    )
    def _write_back(n_use, dir_value, request):
        if not n_use:                                                           # 1
            raise PreventUpdate
        target = (request or {}).get("target") or {}                           # 2
        owner = target.get("owner", "")                                        # 3
        param = target.get("param", "")                                        # 4
        try:                                                                    # 5
            d = _normalize(dir_value or [0.0, 0.0, 1.0])
            value = f"{d[0]:.6g},{d[1]:.6g},{d[2]:.6g}"
        except ValueError:
            value = no_update
        return [                                                                # 6
            value if (e["id"].get("owner") == owner and e["id"].get("param") == param)
            else no_update
            for e in ctx.outputs_list
        ]
