"""Angles path field component — text input + "Build…" modal host.

Wrap any form field that accepts an angles file path with
:func:`get_angles_field` instead of a bare text input.  The component bundles:

* A plain ``dbc.Input`` for typing / pasting a path directly.
* A "Build…" button that opens the angles-builder modal.
* The angles-builder modal (with "Use this file" + "Close" footer buttons).

After the user builds a file in the modal and clicks "Use this file", the path
is written back into the outer text input and the modal closes.

Public API
----------
get_angles_field(prefix)
    Return the Dash layout (``InputGroup`` + ``Modal``).
register_angles_field_callbacks(app, prefix)
    Register open/close and prefill callbacks, plus the builder panel callbacks.
"""

import dash
from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from cryocat.app.components.anglesbuilder import (
    get_angles_builder_panel,
    register_angles_builder_callbacks,
)


def get_angles_field(prefix: str) -> html.Div:
    """A path text-input paired with a "Build…" button that opens the builder modal.

    Parameters
    ----------
    prefix : str
        Unique string prefix for all component IDs.  The outer text input will
        be at ``{prefix}-path``; the modal builder uses ``{prefix}-build`` as
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
                        placeholder="Path to angles file",
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
                    dbc.ModalHeader(dbc.ModalTitle("Build angle list")),
                    dbc.ModalBody(get_angles_builder_panel(builder_prefix)),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Use this file",
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
                size="xl",
                is_open=False,
                centered=True,
                scrollable=True,
            ),
        ]
    )


def register_angles_field_callbacks(app: dash.Dash, prefix: str) -> None:
    """Register callbacks for the angles field modal host.

    Registers the builder panel callbacks (via
    :func:`~cryocat.app.components.anglesbuilder.register_angles_builder_callbacks`)
    plus the open/close and "Use this file" prefill callbacks.

    Parameters
    ----------
    app : dash.Dash
        The Dash application instance.
    prefix : str
        Must match the ``prefix`` passed to :func:`get_angles_field`.
    """
    builder_prefix = f"{prefix}-build"
    register_angles_builder_callbacks(app, builder_prefix)

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
        State(f"{builder_prefix}-created-path", "data"),
        prevent_initial_call=True,
    )
    def _use(_n, created):
        return created or no_update
