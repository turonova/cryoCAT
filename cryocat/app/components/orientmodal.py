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

from cryocat.app import ids
from cryocat.app.components.orientpicker import (
    get_orientation_picker_panel,
    register_orientation_picker_callbacks,
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

    app.clientside_callback(
        """
        function(n_use, dir_value, request) {
            if (!n_use) return window.dash_clientside.no_update;
            var target = ((request || {}).target) || {};
            var owner = target.owner || '';
            var param = target.param || '';
            var value;
            try {
                var d = (dir_value && dir_value.length >= 3) ? dir_value : [0.0, 0.0, 1.0];
                var len = Math.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
                if (len > 0) { d = [d[0]/len, d[1]/len, d[2]/len]; } else { d = [0.0, 0.0, 1.0]; }
                function g6(x) { return parseFloat(x.toPrecision(6)).toString(); }
                value = g6(d[0]) + ',' + g6(d[1]) + ',' + g6(d[2]);
            } catch(e) {
                return window.dash_clientside.no_update;
            }
            return dash_clientside.callback_context.outputs_list.map(function(e) {
                if (e.id && e.id.owner === owner && e.id.param === param) return value;
                return window.dash_clientside.no_update;
            });
        }
        """,
        Output({"type": ALL, "owner": ALL, "param": ALL, "tag": "TripletLike"}, "value"),
        Input("orient-modal-use-btn", "n_clicks"),
        State(f"{_INNER}-value", "data"),
        State(ids.ORIENT_REQUEST, "data"),
        prevent_initial_call=True,
    )
