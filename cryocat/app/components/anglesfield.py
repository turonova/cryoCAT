"""Angles path field component — text input + "Build…" modal host.

Wrap any form field that accepts an angles file path with
:func:`get_angles_field` instead of a bare text input.  The component bundles:

* A plain ``dbc.Input`` for typing / pasting a path directly.
* A "Build…" button that opens the angles-builder modal.
* The angles-builder modal (with "Use" + "Close" footer buttons).

After the user builds a file in the modal and clicks "Use", the path
is written back into the outer text input and the modal closes.

Public API
----------
get_angles_field(prefix)
    Return the Dash layout (``InputGroup`` + ``Modal``).
register_angles_field_callbacks(app, prefix)
    Register open/close and prefill callbacks, plus the builder panel callbacks.
"""

import dash

from cryocat.app.components.anglesbuilder import (
    get_angles_builder_panel,
    register_angles_builder_callbacks,
)
from cryocat.app.components.builderfield import (
    get_builder_field,
    register_builder_field_callbacks,
)


def get_angles_field(prefix: str):
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
    return get_builder_field(
        prefix,
        panel=get_angles_builder_panel(builder_prefix),
        title="Build angle list",
        placeholder="Path to angles file",
        target_id=f"{prefix}-path",
        modal_size="xl",
    )


def register_angles_field_callbacks(app: dash.Dash, prefix: str) -> None:
    """Register callbacks for the angles field modal host.

    Registers the builder panel callbacks (via
    :func:`~cryocat.app.components.anglesbuilder.register_angles_builder_callbacks`)
    plus the open/close and "Use" prefill callbacks.

    Parameters
    ----------
    app : dash.Dash
        The Dash application instance.
    prefix : str
        Must match the ``prefix`` passed to :func:`get_angles_field`.
    """
    builder_prefix = f"{prefix}-build"
    register_angles_builder_callbacks(app, builder_prefix, with_graphs=True)
    register_builder_field_callbacks(
        app,
        prefix,
        value_store_id=f"{builder_prefix}-value",
        target_id=f"{prefix}-path",
    )
