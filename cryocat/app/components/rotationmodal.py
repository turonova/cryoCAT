"""App-level rotation-builder modal (Phase 11 R2).

One instance per app — mounted once in app.py / tango/layout.py.  Any number
of rotation fields can open it by firing a ``rotation-build-btn`` pattern id.
The modal writes the chosen Euler string back to the requesting control via a
pattern-matching Output that matches only 4-key RotationLike ids.

W4 (ORIENTATION_PICKER_PLACEMENT.md) — the modal body now has two tabs:
  * "Build rotation" — original rotation builder (unchanged).
  * "Pick direction" — orientation picker in rotation mode; the "Use this
    rotation" button reads whichever tab is active and formats its result as
    a phi,theta,psi Euler string before writing back.

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
from cryocat.app.components.orientpicker import (
    get_orientation_picker_panel,
    register_orientation_picker_callbacks,
)

_INNER = "rotation-modal-inner"
_INNER_OP = "rotation-modal-inner-op"
_REGISTERED_APPS: set[int] = set()


def get_rotation_modal() -> html.Div:
    """Single app-level rotation-builder modal.  Mount once in the app layout."""
    return html.Div([
        dcc.Store(id=ids.ROTATION_REQUEST),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Build rotation")),
                dbc.ModalBody(
                    dbc.Tabs(
                        [
                            dbc.Tab(
                                get_rotation_builder_panel(_INNER),
                                label="Build rotation",
                                tab_id="rotation-modal-tab-build",
                            ),
                            dbc.Tab(
                                get_orientation_picker_panel(
                                    _INNER_OP, mode="rotation",
                                    show_structure=True, height="400px"
                                ),
                                label="Pick direction",
                                tab_id="rotation-modal-tab-pick",
                            ),
                        ],
                        id="rotation-modal-tabs",
                        active_tab="rotation-modal-tab-build",
                    )
                ),
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
            size="xl",
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
    register_orientation_picker_callbacks(app, _INNER_OP, mode="rotation", show_structure=True)

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
        State(f"{_INNER_OP}-value", "data"),
        State("rotation-modal-tabs", "active_tab"),
        State(ids.ROTATION_REQUEST, "data"),
        prevent_initial_call=True,
    )
    def _write_back(n_use, euler_str, orient_value, active_tab, request):
        if not n_use:                                                           # 1
            raise PreventUpdate
        target = (request or {}).get("target") or {}                           # 2
        owner, param = target.get("owner", ""), target.get("param", "")        # 3
        if active_tab == "rotation-modal-tab-pick" and orient_value:           # 4
            angles = orient_value  # [phi, theta, psi]
            value = f"{angles[0]:.4f},{angles[1]:.4f},{angles[2]:.4f}"
        else:
            value = euler_str
        return [                                                                # 5
            value if (e["id"].get("owner") == owner and e["id"].get("param") == param)
            else no_update
            for e in ctx.outputs_list
        ]
