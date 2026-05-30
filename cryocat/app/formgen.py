"""GUI-side form realizer — turns a function signature into Dash form rows.

This is the *only* module that maps the Dash-free string widget descriptors in
:data:`cryocat.utils.classutils.TYPE_HANDLERS` to actual ``dcc``/``dbc``
components. Forms are built from ``inspect.signature`` (type from the
annotation via :func:`resolve_param_type`, required/default from the signature,
tooltip from the docstring) — not from docstring type text.

Every control's id encodes the resolved tag (``{"type", "param", "tag",
"choices"}``) so :func:`cryocat.app.apputils.generate_kwargs` can round-trip
values through the *same* ``TYPE_HANDLERS`` table — render and parse cannot
drift.
"""

import inspect
import typing

from dash import html, dcc
import dash_bootstrap_components as dbc

from cryocat.utils.classutils import resolve_param_type, process_method_docstring, TYPE_HANDLERS


_HINT_STYLE = {"fontSize": "0.85rem", "color": "var(--color9)", "padding": "2px 0"}
_LABEL_STYLE = {"width": "45%", "display": "flex", "alignItems": "center",
                "boxSizing": "border-box", "paddingRight": "4px"}
_INPUT_STYLE = {"width": "55%"}
_ROW_STYLE = {"display": "flex", "flexDirection": "row", "marginBottom": "0.25rem",
              "width": "100%", "alignItems": "center"}
_COMPACT_INPUT_STYLE = {
    "width": "100%", "height": "22px", "minHeight": "22px",
    "padding": "0 6px", "fontSize": "11px", "lineHeight": "20px",
    "boxSizing": "border-box", "borderRadius": "3px",
}


def _empty(default):
    return default is None or default is inspect.Parameter.empty


def _mk_id(id_type, name, tag, id_extra):
    """Build a pattern-matchable control id.

    Dash dict-id values must be str/number/bool, so no list (e.g. Literal
    choices) is stored here — ``generate_kwargs`` parses Literal values without
    needing them (the dropdown already yields a valid choice)."""
    cid = {"type": id_type, "param": name, "tag": tag}
    if id_extra:
        cid.update(id_extra)
    return cid


# ── Smart dropdown helper ────────────────────────────────────────────────────

def make_dropdown(cid, options, value, clearable=False, **kwargs):
    """Create a dcc.Dropdown; search is enabled automatically when > 10 options."""
    return dcc.Dropdown(
        id=cid, options=options, value=value,
        clearable=clearable,
        searchable=len(options) > 10,
        style={"width": "100%"},
        **kwargs,
    )


# ── Widget factories: descriptor string -> Dash component ───────────────────

def _truly_optional(required, default):
    """True only when the parameter has an explicit None default (can be left blank)."""
    return not required and _empty(default)


def _text_field(cid, default, required, choices=None):
    return dcc.Input(
        type="text", id=cid,
        value="" if _empty(default) else str(default),
        placeholder="Optional" if _truly_optional(required, default) else "",
        style=_COMPACT_INPUT_STYLE,
    )


def _number_field(cid, default, required, choices=None):
    return dcc.Input(
        type="number", id=cid,
        value=None if _empty(default) else default,
        placeholder="Optional" if _truly_optional(required, default) else "",
        style=_COMPACT_INPUT_STYLE,
    )


def _bool_dropdown(cid, default, required, choices=None):
    val = "True" if default is True else "False" if default is False else None
    return make_dropdown(cid, ["True", "False"], val)


def _path_field(cid, default, required, choices=None):
    suffix = " (optional)" if _truly_optional(required, default) else ""
    return dcc.Input(
        type="text", id=cid,
        value="" if _empty(default) else str(default),
        placeholder=f"path to file{suffix}",
        style=_COMPACT_INPUT_STYLE,
    )


def _triplet_field(cid, default, required, choices=None):
    if _empty(default):
        val = ""
    elif isinstance(default, (list, tuple)):
        val = ",".join(str(x) for x in default)
    else:
        val = str(default)
    return dcc.Input(
        type="text", id=cid, value=val,
        placeholder="e.g. 64,64,64 or 64",
        style=_COMPACT_INPUT_STYLE,
    )


def _choice_dropdown(cid, default, required, choices=None):
    choices = list(choices or [])
    val = default if not _empty(default) else (choices[0] if choices else None)
    return make_dropdown(
        cid,
        [{"label": str(c), "value": c} for c in choices],
        val,
    )


def _rotation_field(cid, default, required, choices=None):
    from cryocat.app.components.rotationbuilder import get_rotation_builder_panel
    type_ = cid.get("type", "x") if isinstance(cid, dict) else str(cid)
    param_ = cid.get("param", "x") if isinstance(cid, dict) else "x"
    builder_ = cid.get("builder", "") if isinstance(cid, dict) else ""
    rprefix = (
        f"rotfld-{builder_}-{type_}-{param_}" if builder_
        else f"rotfld-{type_}-{param_}"
    )
    if _empty(default):
        val = ""
    elif isinstance(default, (list, tuple)):
        val = ",".join(str(x) for x in default)
    else:
        val = str(default)
    return html.Div(
        [
            dbc.InputGroup(
                [
                    dbc.Input(
                        id=cid,
                        type="text",
                        value=val,
                        placeholder="phi,theta,psi (zxz, degrees)",
                    ),
                    dbc.Button("Build…", id=f"{rprefix}-build-btn", color="secondary", size="sm"),
                ]
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Build rotation")),
                    dbc.ModalBody(get_rotation_builder_panel(f"{rprefix}-inner")),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Use this rotation",
                                id=f"{rprefix}-use-btn",
                                color="primary",
                                className="me-2",
                            ),
                            dbc.Button("Close", id=f"{rprefix}-close-btn", color="secondary"),
                        ]
                    ),
                ],
                id=f"{rprefix}-modal",
                size="lg",
                is_open=False,
                centered=True,
            ),
        ]
    )


_WIDGET_FACTORIES = {
    "path":     _path_field,
    "triplet":  _triplet_field,
    "csv_text": _text_field,
    "text":     _text_field,
    "number":   _number_field,
    "bool":     _bool_dropdown,
    "dropdown": _choice_dropdown,
    "rotation": _rotation_field,
}


def _form_row(name, widget, description, truly_optional=False, label_id=None):
    if label_id is None:
        label_id = f"formgen-lbl-{name}"
    label_text = name.replace("_", " ").capitalize() + (" (opt.)" if truly_optional else "")
    label = html.Div(
        [
            html.Label(label_text, id=label_id, style={"fontSize": "0.85rem", "margin": 0}),
            dbc.Tooltip(description, target=label_id, placement="right") if description else None,
        ],
        style=_LABEL_STYLE,
    )
    return html.Div([label, html.Div(widget, style=_INPUT_STYLE)], style=_ROW_STYLE)


def build_form(fn, id_type="op-param", id_extra=None, exclude=()):
    """Build Dash form rows for a callable from its signature.

    Parameters
    ----------
    fn : callable or class
        The function/method whose parameters become the form. A class is
        accepted — its ``__init__`` signature is used.
    id_type : str, default="op-param"
        The ``"type"`` field of every control's pattern-matchable id.
    id_extra : dict, optional
        Extra static fields merged into every control's id (e.g.
        ``{"cls_name": "nn-params"}`` to disambiguate multiple forms).
    exclude : iterable of str, optional
        Parameter names to omit (in addition to ``self`` and the decorator's
        ``hide`` set). Used for class-based forms whose ``__init__`` has inputs
        that should not be surfaced (e.g. ``input_data``).

    Returns
    -------
    list
        Dash component rows. Each control id is
        ``{"type": id_type, "param": name, "tag": tag, "choices": [...], **id_extra}``.
    """
    gui = getattr(fn, "_gui", {})
    hide = set(gui.get("hide", ())) | {"self"} | set(exclude)

    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return [html.Div("No parameters.", style=_HINT_STYLE)]

    # `from __future__ import annotations` (PEP 563) makes all annotations lazy
    # strings. `get_type_hints` evaluates them back to live types so
    # resolve_param_type sees e.g. Optional[float], not the string "Optional[float]".
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    # For classes, parameter descriptions normally live in the *class* docstring
    # (numpydoc convention). Try __init__ first, fall back to the class itself.
    if inspect.isclass(fn):
        descriptions = process_method_docstring(fn, "__init__") or process_method_docstring(fn)
    else:
        descriptions = process_method_docstring(fn)

    rows = []
    for name, param in sig.parameters.items():
        if name in hide:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = hints.get(name, param.annotation)
        tag, extra = resolve_param_type(annotation)
        required = param.default is inspect.Parameter.empty
        default = None if required else param.default
        truly_optional = not required and default is None
        choices = extra.get("choices", [])

        handler = TYPE_HANDLERS[tag]
        cid = _mk_id(id_type, name, tag, id_extra)
        widget = _WIDGET_FACTORIES[handler["widget"]](cid, default, required, choices=choices)
        # Build a label ID that is unique across all mounted pages by incorporating
        # id_type and all id_extra values (sorted for stability).
        extra_str = "_".join(str(v) for _, v in sorted((id_extra or {}).items()))
        lbl_id = f"formgen-lbl_{id_type}_{extra_str}_{name}" if extra_str else f"formgen-lbl_{id_type}_{name}"
        rows.append(_form_row(name, widget, descriptions.get(name, ""), truly_optional, label_id=lbl_id))

    if not rows:
        return [html.Div("No parameters required.", style=_HINT_STYLE)]
    return rows
