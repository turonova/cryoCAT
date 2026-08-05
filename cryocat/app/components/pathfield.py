"""Reusable path field: text input + Browse button.  No callbacks to register.

Ids
---
text input  : ``{"type": "path-input",      "owner": prefix}``
browse btn  : ``{"type": "path-browse-btn", "owner": prefix}``
meta store  : ``{"type": "path-browse-meta","owner": prefix}``

The ``owner`` is the *prefix* string, which must be unique within the app.

The single app-level file-browser modal (``filebrowser.py``) listens for
clicks on any ``{"type": "path-browse-btn"}`` and reads the corresponding
meta store to know which mode, kind, and extensions to request.  On Confirm
it writes back to the matching ``{"type": "path-input", "owner": prefix}``
field.  No other registration is needed from the caller's side.
"""
from __future__ import annotations

import json

from dash import html, dcc
import dash_bootstrap_components as dbc


def get_path_field(
    prefix: str,
    *,
    mode: str = "open",
    kind: str = "",
    extensions: tuple[str, ...] = (),
    placeholder: str = "",
    value: str = "",
) -> html.Div:
    """Return an inert path-field component (text input + Browse button).

    Parameters
    ----------
    prefix:
        Unique namespace for this field; becomes the ``owner`` key in both
        component ids.  Must be stable across renders (used for write-back).
    mode:
        ``"open"`` — pick an existing file (default).
        ``"directory"`` — pick an existing directory.
        ``"save"`` — pick a directory and supply a filename.
    kind:
        Last-directory bucket (D4); e.g. ``"motl"``, ``"mask"``, ``"output"``.
        Fields with the same kind re-open in the same location.
    extensions:
        Allowed file extensions, e.g. ``(".em", ".star")``.  Directories are
        never filtered.  Empty tuple means all files are shown.
    placeholder:
        Placeholder shown in the text input when empty.
    value:
        Initial value of the text input.
    """
    meta = {"mode": mode, "kind": kind, "extensions": list(extensions)}

    return html.Div(
        [
            dcc.Store(
                id={"type": "path-browse-meta", "owner": prefix},
                data=meta,
            ),
            dbc.Input(
                id={"type": "path-input", "owner": prefix},
                value=value,
                placeholder=placeholder or "Path…",
                type="text",
                style={"flex": "1 1 0", "minWidth": "0"},
            ),
            dbc.Button(
                "Browse…",
                id={"type": "path-browse-btn", "owner": prefix},
                color="secondary",
                size="sm",
                style={"flexShrink": "0"},
            ),
        ],
        style={
            "display": "flex",
            "gap": "0.35rem",
            "width": "100%",
            "alignItems": "center",
        },
    )
