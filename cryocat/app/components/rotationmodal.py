"""App-level rotation-builder modal (Phase 11 R2).

One instance per app — mounted once in app.py / tango/layout.py.  Any number
of rotation fields can open it by firing a ``rotation-build-btn`` pattern id.
The modal writes the chosen Euler string back to the requesting control via a
pattern-matching Output that matches only 4-key RotationLike ids.

Mount point:  get_rotation_modal()  →  app layout
Callbacks  :  register_rotation_modal_callbacks(app)  →  called once per app
"""
from __future__ import annotations

from dash import html, dcc, Input, Output, State, ALL, no_update, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from cryocat.app import ids
from cryocat.app.components.rotationbuilder import (
    get_rotation_builder_panel,
    register_rotation_builder_callbacks,
)

_INNER = "rotation-modal-inner"
_REGISTERED_APPS: set[int] = set()


def get_rotation_modal() -> html.Div:
    """Single app-level rotation-builder modal.  Mount once in the app layout."""
    return html.Div([
        dcc.Store(id=ids.ROTATION_REQUEST),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Build rotation")),
                dbc.ModalBody(get_rotation_builder_panel(_INNER)),
                dbc.ModalFooter([
                    dbc.Button(
                        "Use this rotation",
                        id="rotation-use-btn",
                        color="primary",
                        className="me-2",
                    ),
                    dbc.Button("Close", id="rotation-close-btn", color="secondary"),
                ]),
            ],
            id="rotation-modal",
            size="lg",
            is_open=False,
            centered=True,
        ),
    ])


def register_rotation_modal_callbacks(app) -> None:
    """Register open/close and write-back callbacks.  Call exactly once per app."""
    app_key = id(app)
    if app_key in _REGISTERED_APPS:
        raise RuntimeError(
            "register_rotation_modal_callbacks already called for this app instance. "
            "Mount and register the rotation modal exactly once per app."
        )
    _REGISTERED_APPS.add(app_key)
    register_rotation_builder_callbacks(app, _INNER)

    @app.callback(
        Output("rotation-modal", "is_open"),
        Output(ids.ROTATION_REQUEST, "data"),
        Input({"type": "rotation-build-btn", "owner": ALL, "param": ALL}, "n_clicks"),
        Input("rotation-close-btn", "n_clicks"),
        Input("rotation-use-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _open_modal(build_clicks, _close, _use):
        tid = ctx.triggered_id
        if isinstance(tid, dict) and tid.get("type") == "rotation-build-btn":
            return True, {"target": tid}
        return False, no_update

    @app.callback(
        Output({"type": ALL, "owner": ALL, "param": ALL, "tag": "RotationLike"}, "value"),
        Input("rotation-use-btn", "n_clicks"),
        State(f"{_INNER}-value", "data"),
        State(ids.ROTATION_REQUEST, "data"),
        prevent_initial_call=True,
    )
    def _write_back(n_use, euler_str, request):
        if not n_use:
            raise PreventUpdate
        target = (request or {}).get("target") or {}
        owner, param = target.get("owner", ""), target.get("param", "")
        return [
            euler_str if (e["id"].get("owner") == owner and e["id"].get("param") == param)
            else no_update
            for e in ctx.outputs_list
        ]
