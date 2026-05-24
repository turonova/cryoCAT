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

from dash import html, dcc
import dash_bootstrap_components as dbc

from cryocat.utils.classutils import resolve_param_type, process_method_docstring, TYPE_HANDLERS


_HINT_STYLE = {"fontSize": "0.85rem", "color": "var(--color9)", "padding": "2px 0"}
_LABEL_STYLE = {"width": "45%", "display": "flex", "alignItems": "center",
                "boxSizing": "border-box", "paddingRight": "4px"}
_INPUT_STYLE = {"width": "55%"}
_ROW_STYLE = {"display": "flex", "flexDirection": "row", "marginBottom": "0.25rem",
              "width": "100%", "alignItems": "center"}


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


# ── Widget factories: descriptor string -> Dash component ───────────────────

def _text_field(cid, default, required, choices=None):
    return dcc.Input(
        type="text", id=cid,
        value="" if _empty(default) else str(default),
        placeholder="" if required else "Optional",
        style={"width": "100%"},
    )


def _number_field(cid, default, required, choices=None):
    return dcc.Input(
        type="number", id=cid,
        value=None if _empty(default) else default,
        placeholder="" if required else "Optional",
        style={"width": "100%"},
    )


def _bool_dropdown(cid, default, required, choices=None):
    val = "True" if default is True else "False" if default is False else None
    return dcc.Dropdown(id=cid, options=["True", "False"], value=val, style={"width": "100%"})


def _path_field(cid, default, required, choices=None):
    return dcc.Input(
        type="text", id=cid,
        value="" if _empty(default) else str(default),
        placeholder="path to file" + ("" if required else " (optional)"),
        style={"width": "100%"},
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
        style={"width": "100%"},
    )


def _choice_dropdown(cid, default, required, choices=None):
    choices = list(choices or [])
    if not _empty(default):
        val = default
    else:
        val = choices[0] if choices else None
    return dcc.Dropdown(
        id=cid,
        options=[{"label": str(c), "value": c} for c in choices],
        value=val,
        style={"width": "100%"},
    )


_WIDGET_FACTORIES = {
    "path":     _path_field,
    "triplet":  _triplet_field,
    "csv_text": _text_field,
    "text":     _text_field,
    "number":   _number_field,
    "bool":     _bool_dropdown,
    "dropdown": _choice_dropdown,
}


def _form_row(name, widget, description, required):
    label_id = f"formgen-lbl-{name}"
    label_text = name.replace("_", " ").capitalize() + ("" if required else " (opt.)")
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

    descriptions = process_method_docstring(fn, "__init__") if inspect.isclass(fn) \
        else process_method_docstring(fn)

    rows = []
    for name, param in sig.parameters.items():
        if name in hide:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        tag, extra = resolve_param_type(param.annotation)
        required = param.default is inspect.Parameter.empty
        default = None if required else param.default
        choices = extra.get("choices", [])

        handler = TYPE_HANDLERS[tag]
        cid = _mk_id(id_type, name, tag, id_extra)
        widget = _WIDGET_FACTORIES[handler["widget"]](cid, default, required, choices=choices)
        rows.append(_form_row(name, widget, descriptions.get(name, ""), required))

    if not rows:
        return [html.Div("No parameters required.", style=_HINT_STYLE)]
    return rows
