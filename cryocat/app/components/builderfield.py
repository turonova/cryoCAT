"""Generic builder-field modal host.

Any builder panel that writes its final value to a ``{panel_prefix}-value``
``dcc.Store`` can be embedded here.  Eliminates the boilerplate toggle +
prefill pattern that every field-with-builder would otherwise duplicate.

Modal ownership rule (Phase 11): a field rendered by ``formgen.build_form``
(i.e. placed inside a dynamic form container) must never own a modal of its
own — it must delegate to the app-level modal for that builder type.  A field
placed *explicitly* by a page (outside a dynamic form) may own a modal via
this module.

Public API
----------
get_builder_field(prefix, *, panel, title, placeholder, target_id, modal_size)
    Layout tree (InputGroup + Modal).
register_builder_field_callbacks(app, prefix, *, value_store_id, target_id)
    Toggle and prefill callbacks.  Raises RuntimeError on duplicate *prefix*.
"""

from __future__ import annotations

import dash
from dash import html, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

_REGISTERED: set[str] = set()


def get_builder_field(
    prefix: str,
    *,
    panel,
    title: str,
    placeholder: str = "",
    target_id,
    modal_size: str = "lg",
) -> html.Div:
    """Text input + 'Build…' button + modal hosting *panel*.

    Parameters
    ----------
    prefix : str
        Unique prefix for all component IDs (``{prefix}-build-btn``,
        ``{prefix}-close-btn``, ``{prefix}-use-btn``, ``{prefix}-modal``).
    panel : dash.development.base_component.Component
        Pre-built builder panel to embed in the modal body.
    title : str
        Modal header title.
    placeholder : str
        Placeholder text for the text input.
    target_id : str | dict
        ID of the text input component (string or pattern-matching dict).
    modal_size : str
        Bootstrap modal size: ``"sm"``, ``"lg"``, or ``"xl"``.
    """
    return html.Div(
        [
            dbc.InputGroup(
                [
                    dbc.Input(
                        id=target_id,
                        type="text",
                        placeholder=placeholder,
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
                    dbc.ModalHeader(dbc.ModalTitle(title)),
                    dbc.ModalBody(panel),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Use",
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
                size=modal_size,
                is_open=False,
                centered=True,
                scrollable=True,
            ),
        ]
    )


def register_builder_field_callbacks(
    app: dash.Dash,
    prefix: str,
    *,
    value_store_id: str | dict,
    target_id: str | dict,
) -> None:
    """Register open/close toggle and 'Use' prefill callbacks.

    Parameters
    ----------
    app : dash.Dash
        The Dash application instance.
    prefix : str
        Must match the *prefix* passed to :func:`get_builder_field`.
    value_store_id : str | dict
        ID of the ``dcc.Store`` the builder writes its result to
        (``{panel_prefix}-value``).
    target_id : str | dict
        ID of the text input to prefill when 'Use' is clicked.
    """
    if prefix in _REGISTERED:
        raise RuntimeError(
            f"Builder field callbacks already registered for prefix {prefix!r}. "
            "Each prefix must be unique."
        )
    _REGISTERED.add(prefix)

    build_btn = f"{prefix}-build-btn"

    @app.callback(
        Output(f"{prefix}-modal", "is_open"),
        Input(f"{prefix}-build-btn", "n_clicks"),
        Input(f"{prefix}-close-btn", "n_clicks"),
        Input(f"{prefix}-use-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _toggle(*_):
        return ctx.triggered_id == build_btn

    @app.callback(
        Output(target_id, "value", allow_duplicate=True),
        Input(f"{prefix}-use-btn", "n_clicks"),
        State(value_store_id, "data"),
        prevent_initial_call=True,
    )
    def _use(_n, value):
        return value or no_update
