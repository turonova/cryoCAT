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
import logging
import typing
from collections.abc import Callable
from typing import Any

_log = logging.getLogger(__name__)

from dash import html, dcc, ALL
import dash_bootstrap_components as dbc

from cryocat.utils.classutils import resolve_param_type, process_method_docstring, TYPE_HANDLERS
from cryocat.app import styles

WidgetFactory = Callable[[dict, Any, bool, list | None, dict | None], Any]


def _empty(default):
    return default is None or default is inspect.Parameter.empty


def _mk_id(id_type, name, tag, id_extra):
    """Build a pattern-matchable control id.

    All ids include "owner" (Phase 11 invariant).  "builder" in id_extra is
    silently remapped to "owner" for backward compatibility.  Other extra keys
    (cls_name, op, …) are carried through unchanged.

    Dash dict-id values must be str/number/bool, so no list (e.g. Literal
    choices) is stored here — ``generate_kwargs`` parses Literal values without
    needing them (the dropdown already yields a valid choice)."""
    extra = dict(id_extra) if id_extra else {}
    owner = extra.pop("owner", None) or extra.pop("builder", "") or ""
    cid = {"type": id_type, "owner": owner, "param": name, "tag": tag}
    cid.update(extra)
    return cid


# ── Smart dropdown helper ────────────────────────────────────────────────────

def make_dropdown(cid, options, value, clearable=False, **kwargs):
    """Create a dcc.Dropdown; search is enabled automatically when > 10 options."""
    style = {"width": "100%"}
    if "style" in kwargs:
        style.update(kwargs.pop("style"))
    return dcc.Dropdown(
        id=cid, options=options, value=value,
        clearable=clearable,
        searchable=len(options) > 10,
        style=style,
        **kwargs,
    )


def section_divider(title: str) -> html.Div:
    """A horizontal rule with the title centred on it."""
    return html.Div(
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "0.5rem",
            "marginTop": styles.SECTION_GAP,
            "marginBottom": styles.SECTION_GAP,
        },
        children=[
            html.Hr(style={"flex": "1", "margin": "0"}),
            html.Span(
                title,
                style={"fontSize": styles.FONT_SM, "color": styles.COLOR_MUTED, "whiteSpace": "nowrap"},
            ),
            html.Hr(style={"flex": "1", "margin": "0"}),
        ],
    )


# ── Widget factories: descriptor string -> Dash component ───────────────────

def _truly_optional(required, default):
    """True only when the parameter has an explicit None default (can be left blank)."""
    return not required and _empty(default)


def _with_var_picker(widget, cid):
    """Wrap *widget* with an @ button that opens the global variable picker.

    Follows the same flex-row pattern as ``_path_field``'s Browse button so
    the two affordances are visually consistent.  The button's ``owner`` encodes
    the target input's id as JSON so the write-back callback can route the
    selected ``@name`` to the correct field.
    """
    import json as _json
    owner = _json.dumps(dict(sorted(cid.items()))) if isinstance(cid, dict) else str(cid)
    return html.Div(
        [
            html.Div(widget, style={"flex": "1 1 0", "minWidth": "0"}),
            dbc.Button(
                "@",
                id={"type": "var-picker-btn", "owner": owner},
                color="secondary",
                size="sm",
                title="Insert a session-variable reference (@name)",
                style={"flexShrink": "0"},
            ),
        ],
        style={"display": "flex", "gap": "0.35rem", "width": "100%", "alignItems": "center"},
    )


def _text_field(cid, default, required, choices=None, extra=None):
    inp = dcc.Input(
        type="text", id=cid,
        value="" if _empty(default) else str(default),
        placeholder="Optional" if _truly_optional(required, default) else "",
        style=styles.FORM_COMPACT_INPUT,
    )
    return _with_var_picker(inp, cid)


def _number_field(cid, default, required, choices=None, extra=None):
    return _with_var_picker(dcc.Input(
        type="text", id=cid,
        value=None if _empty(default) else str(default) if default is not None else None,
        placeholder="Optional" if _truly_optional(required, default) else "",
        style=styles.FORM_COMPACT_INPUT,
    ), cid)


def _bool_dropdown(cid, default, required, choices=None, extra=None):
    val = "True" if default is True else "False" if default is False else None
    return make_dropdown(cid, ["True", "False"], val)


def _path_field(cid, default, required, choices=None, extra=None):
    """Wrap the path input with a Browse button.

    The text input keeps its canonical ``cid`` so :func:`generate_kwargs`
    and the collecting callback's State pattern are unchanged.  The Browse
    button has a derived id that encodes the target cid; the app-level
    file-browser modal reads this to know where to write back (Z4).
    """
    import json as _json

    suffix = " (optional)" if _truly_optional(required, default) else ""
    val = "" if _empty(default) else str(default)

    # Stable JSON encoding of cid used as the "owner" key on the browse
    # button and its meta store so the modal can route the result back.
    owner = _json.dumps(dict(sorted(cid.items()))) if isinstance(cid, dict) else str(cid)

    from dash import dcc as _dcc
    hint_id = {**cid, "id_type": cid["type"], "type": "path-exists-hint"} if isinstance(cid, dict) else None
    return html.Div(
        [
            dcc.Store(
                id={"type": "path-browse-meta", "owner": owner},
                data={"mode": "open", "kind": "", "extensions": []},
            ),
            _dcc.Input(
                type="text", id=cid,
                value=val,
                placeholder=f"path to file{suffix}",
                style={**styles.FORM_COMPACT_INPUT, "flex": "1 1 0", "minWidth": "0"},
            ),
            dbc.Button(
                "Browse…",
                id={"type": "path-browse-btn", "owner": owner},
                color="secondary",
                size="sm",
                style={"flexShrink": "0"},
            ),
            *(
                [html.Span(
                    id=hint_id,
                    children="",
                    style={**styles.HINT, "flexShrink": "0", "whiteSpace": "nowrap"},
                )]
                if hint_id is not None else []
            ),
        ],
        style={"display": "flex", "gap": "0.35rem", "width": "100%", "alignItems": "center"},
    )


def _triplet_field(cid, default, required, choices=None, extra=None):
    if _empty(default):
        val = ""
    elif isinstance(default, (list, tuple)):
        val = ",".join(str(x) for x in default)
    else:
        val = str(default)
    return dcc.Input(
        type="text", id=cid, value=val,
        placeholder="e.g. 64,64,64 or 64",
        style=styles.FORM_COMPACT_INPUT,
    )


def _choice_dropdown(cid, default, required, choices=None, extra=None):
    choices = list(choices or [])
    val = default if not _empty(default) else (choices[0] if choices else None)
    return make_dropdown(
        cid,
        [{"label": str(c), "value": c} for c in choices],
        val,
    )


def _rotation_field(cid, default, required, choices=None, extra=None):
    # Phase 11 R3: inert widget — the app-level rotation modal owns all callbacks.
    owner = cid.get("owner", "") if isinstance(cid, dict) else ""
    param = cid.get("param", "") if isinstance(cid, dict) else ""
    if _empty(default):
        val = ""
    elif isinstance(default, (list, tuple)):
        val = ",".join(str(x) for x in default)
    else:
        val = str(default)
    return dbc.InputGroup(
        [
            dbc.Input(
                id=cid,
                type="text",
                value=val,
                placeholder="phi,theta,psi (zxz, degrees)",
            ),
            dbc.Button(
                "Build…",
                id={"type": "rotation-build-btn", "owner": owner, "param": param},
                color="secondary",
                size="sm",
            ),
        ]
    )


def _tuple_field(cid, default, required, choices=None, extra=None):
    """Composite numeric-tuple widget.

    Renders ``length`` number inputs side-by-side. Each slot is its own
    pattern-matched control sharing the param's ``cid`` with an added
    ``slot`` index and ``elem`` ("float" or "int") so
    :func:`cryocat.app.apputils.generate_kwargs` can re-assemble the slots
    into a Python tuple via :func:`cryocat.utils.classutils._parse_tuple`.

    The ``extra`` dict comes from :func:`cryocat.utils.classutils.resolve_param_type`
    (``{"length": int, "elem": "float" | "int"}``) — we pull length and elem
    from there so the resolver and the form stay in sync.
    """
    length = int((extra or {}).get("length", 2))
    elem = str((extra or {}).get("elem", "float"))
    if _empty(default):
        slot_vals: list = [None] * length
    elif isinstance(default, (list, tuple)):
        slot_vals = list(default) + [None] * max(0, length - len(default))
        slot_vals = slot_vals[:length]
    else:
        slot_vals = [default] * length

    inputs = []
    for slot, slot_value in enumerate(slot_vals):
        slot_cid = dict(cid)
        slot_cid["slot"] = slot
        slot_cid["elem"] = elem
        inputs.append(
            html.Div(
                dcc.Input(
                    type="text", id=slot_cid,
                    value=None if slot_value is None else str(slot_value),
                    placeholder="Optional" if _truly_optional(required, default) else "",
                    style=styles.FORM_COMPACT_INPUT,
                ),
                style={"flex": "1 1 0", "minWidth": "0"},
            )
        )
    return html.Div(
        inputs,
        style={"display": "flex", "flexDirection": "row", "gap": "0.25rem", "width": "100%"},
    )


_PATH_ELEM_TAGS: frozenset[str] = frozenset({
    "MapSource", "DataSource", "TiltStack", "PathOrStr",
})


def _listlike_field(cid, default, required, choices=None, extra=None):
    """ListLike field: path widget for path-type elements, text otherwise."""
    elem_tag = (extra or {}).get("elem_tag", "")
    if elem_tag in _PATH_ELEM_TAGS:
        return _path_field(cid, default, required, choices, extra)
    return _text_field(cid, default, required, choices, extra)


WIDGET_FACTORIES: dict[str, WidgetFactory] = {
    "path":     _path_field,
    "triplet":  _triplet_field,
    "csv_text": _text_field,
    "text":     _text_field,
    "number":   _number_field,
    "bool":     _bool_dropdown,
    "dropdown": _choice_dropdown,
    "rotation": _rotation_field,
    "tuple":    _tuple_field,
    "listlike": _listlike_field,
}


def form_row(name, widget, description, truly_optional=False, label_id=None, label_text=None):
    if label_id is None:
        label_id = f"formgen-lbl-{name}"
    if label_text is None:
        label_text = name.replace("_", " ").capitalize()
    label_text = label_text + (" (opt.)" if truly_optional else "")
    label = html.Div(
        [
            html.Label(label_text, id=label_id, style={"margin": 0}),
            dbc.Tooltip(description, target=label_id, placement="right") if (description and styles.TOOLTIPS_ENABLED) else None,
        ],
        style=styles.FORM_LABEL,
    )
    return html.Div([label, html.Div(widget, style=styles.FORM_INPUT)], style=styles.FORM_ROW)


def build_form(fn_or_entry, id_type="op-param", id_extra=None, exclude=()):
    """Build Dash form rows for a callable from its signature.

    Parameters
    ----------
    fn_or_entry : callable, class, or GuiEntry
        The function/method whose parameters become the form.  Pass a
        :class:`~cryocat.utils.classutils.GuiEntry` to use validated registry
        metadata (preferred).  A plain callable or class is also accepted for
        backward compatibility.
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
    from cryocat.utils.classutils import GuiEntry
    if isinstance(fn_or_entry, GuiEntry):
        fn = fn_or_entry.fn
        hide = fn_or_entry.hide | {"self"} | set(exclude)
    else:
        fn = fn_or_entry
        gui = getattr(fn, "_gui", {})
        hide = set(gui.get("hide", ())) | {"self"} | set(exclude)

    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return [html.Div("No parameters.", style=styles.FORM_HINT)]

    # `from __future__ import annotations` (PEP 563) makes all annotations lazy
    # strings. `get_type_hints` evaluates them back to live types so
    # resolve_param_type sees e.g. Optional[float], not the string "Optional[float]".
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    # For classes, parameter descriptions normally live in the *class* docstring
    # (numpydoc convention). Try __init__ first, fall back to the class itself.
    try:
        if inspect.isclass(fn):
            descriptions = process_method_docstring(fn, "__init__") or process_method_docstring(fn)
        else:
            descriptions = process_method_docstring(fn)
    except Exception:
        descriptions = {}

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
        # Normalise all path-widget aliases (MapSource, PathOrStr, …) to the
        # literal tag "path" so register_path_writeback's pattern matches them.
        cid_tag = "path" if handler["widget"] == "path" else tag
        cid = _mk_id(id_type, name, cid_tag, id_extra)
        # Composite widgets (Tuple) need length/elem from `extra`; pass it
        # through so simpler factories can ignore it without breaking.
        widget_fn = WIDGET_FACTORIES[handler["widget"]]
        widget = widget_fn(cid, default, required, choices=choices, extra=extra)
        # Build a label ID that is unique across all mounted pages by incorporating
        # id_type and all id_extra values (sorted for stability).
        extra_str = "_".join(str(v) for _, v in sorted((id_extra or {}).items()))
        lbl_id = f"formgen-lbl_{id_type}_{extra_str}_{name}" if extra_str else f"formgen-lbl_{id_type}_{name}"
        rows.append(form_row(name, widget, descriptions.get(name, ""), truly_optional, label_id=lbl_id))

    if not rows:
        return [html.Div("No parameters required.", style=styles.FORM_HINT)]
    return rows


def register_var_picker_writeback(app, id_type: str, id_extra: dict | None = None) -> None:
    """Register a callback that writes variable-picker results to form text inputs.

    Call once per unique ``(id_type, id_extra)`` combination used in
    :func:`build_form` calls that may have text-field parameters.  The
    variable picker modal writes to :data:`~cryocat.app.ids.VAR_PICKER_RESULT`;
    this callback routes that result to the matching ``dcc.Input``.

    Parameters
    ----------
    app:
        The Dash app instance.
    id_type:
        The ``"type"`` field used in the :func:`build_form` call.
    id_extra:
        The ``id_extra`` dict used in the :func:`build_form` call.
    """
    import json as _json
    from dash import Input, Output, no_update, ctx
    from cryocat.app import ids as _ids

    id_extra = id_extra or {}
    if "owner" not in id_extra:
        id_extra = {**id_extra, "owner": ALL}
    pattern = {"type": id_type, "param": ALL, "tag": ALL, **id_extra}

    @app.callback(
        Output(pattern, "value", allow_duplicate=True),
        Input(_ids.VAR_PICKER_RESULT, "data"),
        prevent_initial_call=True,
    )
    def _writeback_var_picker(result):
        if not result:
            raise __import__("dash").exceptions.PreventUpdate
        target_owner = result.get("owner", "")
        final_value = result.get("value", "")
        return [
            final_value if _json.dumps(dict(sorted(e["id"].items()))) == target_owner else no_update
            for e in ctx.outputs_list
        ]


def register_path_writeback(app, id_type: str, id_extra: dict | None = None) -> None:
    """Register a callback that writes browser confirm results to formgen path inputs.

    Call once per unique ``(id_type, id_extra)`` combination used in
    :func:`build_form` calls that produce path-tagged parameters.  The
    app-level modal writes to :data:`~cryocat.app.ids.BROWSER_RESULT`; this
    callback routes that result to the matching ``dcc.Input``.

    Parameters
    ----------
    app:
        The Dash app instance.
    id_type:
        The ``"type"`` field used in the :func:`build_form` call.
    id_extra:
        The ``id_extra`` dict used in the :func:`build_form` call (or
        ``None`` / ``{}`` for forms with no extra fields).
    """
    import json as _json
    from dash import Input, Output, no_update, ctx
    from cryocat.app import ids as _ids

    id_extra = id_extra or {}
    pattern = {"type": id_type, "owner": ALL, "param": ALL, "tag": "path", **id_extra}

    @app.callback(
        Output(pattern, "value"),
        Input(_ids.BROWSER_RESULT, "data"),
        prevent_initial_call=True,
    )
    def _writeback_form_paths(result):
        if not result:
            raise __import__("dash").exceptions.PreventUpdate
        target_owner = result.get("owner", "")
        final_value = result.get("value", "")
        updates = []
        for e in ctx.outputs_list:
            if _json.dumps(dict(sorted(e["id"].items()))) == target_owner:
                _log.debug("path writeback: matched %s → %r", e["id"], final_value)
                updates.append(final_value)
            else:
                updates.append(no_update)
        return updates


def register_path_hint_callback(app, id_type: str, id_extra: dict | None = None) -> None:
    """Register a path-existence hint callback for formgen path inputs of a given type.

    Call once per unique ``(id_type, id_extra)`` combination used in
    :func:`build_form` calls that produce path-tagged parameters — the same
    combination passed to :func:`register_path_writeback`.

    The hint span rendered by ``_path_field`` has id
    ``{**cid, "type": "path-exists-hint"}``; this callback updates its text
    on value change, showing "" when the path exists or is empty and
    "not found" when it does not.  This is a hint, not a block — a missing
    path is not an error.
    """
    from pathlib import Path
    from dash import Input, Output, ALL as _ALL

    id_extra = id_extra or {}

    @app.callback(
        Output({"type": "path-exists-hint", "id_type": id_type, "owner": _ALL, "param": _ALL, "tag": "path", **id_extra}, "children"),
        Input({"type": id_type, "owner": _ALL, "param": _ALL, "tag": "path", **id_extra}, "value"),
        prevent_initial_call=True,
    )
    def _update_path_hint(values):
        return ["" if not v or Path(v).exists() else "not found" for v in values]


def register_form_callbacks(app, id_type: str, id_extra: dict | None = None) -> None:
    """Register all per-form-type callbacks for a :func:`build_form` form type.

    Calls :func:`register_path_writeback`, :func:`register_path_hint_callback`,
    and :func:`register_var_picker_writeback` unconditionally.  Inert
    registrations (no matching components) never fire, so the cost of an inert
    one is nothing and the cost of a forgotten one is a dead field.

    Call once per unique ``(id_type, id_extra)`` combination wherever
    :func:`build_form` is used.
    """
    register_path_writeback(app, id_type, id_extra)
    register_path_hint_callback(app, id_type, id_extra)
    register_var_picker_writeback(app, id_type, id_extra)
