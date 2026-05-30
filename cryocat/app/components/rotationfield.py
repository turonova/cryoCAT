"""Rotation field component — text input + "Build…" modal host.

Wrap any form field that accepts a :data:`~cryocat._types.RotationLike` value
with :func:`get_rotation_field` instead of a bare text input.  The component
bundles:

* A plain ``dbc.Input`` for typing / pasting a comma-separated Euler-angle
  string directly (``phi,theta,psi`` in zxz degrees, as consumed by
  :func:`cryocat.utils.geom.as_rotation`).
* A "Build…" button that opens the rotation-builder modal.
* The rotation-builder modal (with "Use this rotation" + "Close" footer
  buttons).

After the user sets a rotation in the modal and clicks "Use this rotation",
the ``phi,theta,psi`` string is written back into the outer text input and the
modal closes.

Public API
----------
get_rotation_field(prefix)
    Return the Dash layout (``InputGroup`` + ``Modal``).
register_rotation_field_callbacks(app, prefix)
    Register open/close and prefill callbacks, plus the builder callbacks.
"""

import dash
from dash import html, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from cryocat.app.components.rotationbuilder import (
    get_rotation_builder_panel,
    register_rotation_builder_callbacks,
)


def get_rotation_field(prefix: str) -> html.Div:
    """A text input paired with a "Build…" button that opens the rotation-builder modal.

    Parameters
    ----------
    prefix : str
        Unique string prefix for all component IDs.  The outer text input will
        be at ``{prefix}-path``; the builder modal uses ``{prefix}-build`` as
        its own prefix.

    Returns
    -------
    dash.html.Div
        Layout containing the InputGroup and the Modal.
    """
    builder_prefix = f"{prefix}-build"
    return html.Div(
        [
            dbc.InputGroup(
                [
                    dbc.Input(
                        id=f"{prefix}-path",
                        type="text",
                        placeholder="phi,theta,psi (zxz, degrees)",
                    ),
                    dbc.Button(
                        "Build…",
                        id=f"{prefix}-build-btn",
                        color="secondary",
                        size="sm",
                    ),
                ]
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Build rotation")),
                    dbc.ModalBody(get_rotation_builder_panel(builder_prefix)),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Use this rotation",
                                id=f"{prefix}-use-btn",
                                color="primary",
                                className="me-2",
                            ),
                            dbc.Button(
                                "Close",
                                id=f"{prefix}-close-btn",
                                color="secondary",
                            ),
                        ]
                    ),
                ],
                id=f"{prefix}-modal",
                size="lg",
                is_open=False,
                centered=True,
            ),
        ]
    )


def register_rotation_field_callbacks(app: dash.Dash, prefix: str) -> None:
    """Register callbacks for the rotation field modal host.

    Registers the builder panel callbacks (via
    :func:`~cryocat.app.components.rotationbuilder.register_rotation_builder_callbacks`)
    plus the open/close and "Use this rotation" callbacks.

    Parameters
    ----------
    app : dash.Dash
        The Dash application instance.
    prefix : str
        Must match the ``prefix`` passed to :func:`get_rotation_field`.
    """
    builder_prefix = f"{prefix}-build"
    register_rotation_builder_callbacks(app, builder_prefix)

    @app.callback(
        Output(f"{prefix}-modal", "is_open"),
        Input(f"{prefix}-build-btn", "n_clicks"),
        Input(f"{prefix}-close-btn", "n_clicks"),
        Input(f"{prefix}-use-btn", "n_clicks"),
        State(f"{prefix}-modal", "is_open"),
        prevent_initial_call=True,
    )
    def _toggle(_open, _close, _use, is_open):
        return not is_open

    @app.callback(
        Output(f"{prefix}-path", "value", allow_duplicate=True),
        Input(f"{prefix}-use-btn", "n_clicks"),
        State(f"{builder_prefix}-rot-euler-store", "data"),
        prevent_initial_call=True,
    )
    def _use(_n, euler_str):
        return euler_str or no_update
