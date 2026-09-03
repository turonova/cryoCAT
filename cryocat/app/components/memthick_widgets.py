"""Composite Dash widgets for the memthick tab's three non-scalar inputs.

These don't fit the type→widget pipeline driven by
:data:`cryocat.utils.classutils.TYPE_HANDLERS` (their shapes are too rich:
dict editor, dropdown-or-dict mode picker, nested sub-form). They follow the
shared-component contract instead:

* ``get_<name>(prefix, ...)``    -- layout fragment, all ids scoped by prefix.
* ``register_<name>_callbacks(app, prefix, ...)`` -- wires any reactive
  state. Some widgets are static (no callbacks); their ``register_*`` is a
  no-op kept for contract symmetry.
* ``read_<name>(state)`` -- pure helper that turns the widget's State payload
  into the Python value the code generator embeds verbatim.

The widgets exposed here:

* :func:`get_label_dict_field` -- ``dict[str, int]`` editor for membrane
  labels. Rendered as one ``name | id`` row per entry plus an "+ Add row"
  button.
* :func:`get_per_membrane_mode_field` -- single-mode dropdown with a
  per-membrane override switch; when toggled on, one dropdown per known
  membrane label.
* :func:`get_analyzer_subform` -- nested ``IntensityProfileAnalyzer`` form
  built via :func:`cryocat.app.formgen.build_form`. The Phase-1 tuple widget
  surfaces ``minima_search_nm`` as two number slots automatically.
"""
from __future__ import annotations

from typing import Any

from dash import html, dcc, Input, Output, State, ALL, no_update, ctx
import dash_bootstrap_components as dbc

from cryocat.analysis.memthick import IntensityProfileAnalyzer
from cryocat.app import formgen
from cryocat.app.formgen import make_dropdown
from cryocat.app.apputils import generate_kwargs


# ── label-dict editor ────────────────────────────────────────────────────────


def get_label_dict_field(prefix: str, default: dict[str, int] | None = None) -> html.Div:
    """``dict[str, int]`` editor for ``membrane_labels``.

    Renders one row per (name, id) pair plus an "Add row" button. The page
    threads:

    * ``f"{prefix}-rows"`` -- ``dcc.Store`` holding the canonical row list as
      ``[{"name": str, "id": int}, ...]``.
    * ``f"{prefix}-add-btn"`` -- adds an empty row.
    * ``{"type": f"{prefix}-row", "field": "name"|"id", "row": i}`` -- per-row
      pattern-matched inputs.
    * ``{"type": f"{prefix}-del-btn", "row": i}`` -- per-row delete button.

    Parameters
    ----------
    prefix : str
        Id namespace for every owned control / store.
    default : dict[str, int], optional
        Initial mapping displayed when the page first renders. Defaults to
        ``{"membrane": 1}``.
    """
    default = default or {"membrane": 1}
    rows = [{"name": k, "id": int(v)} for k, v in default.items()]
    return html.Div(
        [
            dcc.Store(id=f"{prefix}-rows", data=rows),
            html.Div(id=f"{prefix}-rows-area"),
            dbc.Button(
                "+ Add row",
                id=f"{prefix}-add-btn",
                color="secondary", size="sm",
                style={"marginTop": "0.3rem"},
            ),
        ],
        id=f"{prefix}-container",
    )


def register_label_dict_callbacks(app, prefix: str) -> None:
    """Wire add / delete / edit -> ``{prefix}-rows`` store + row rendering."""

    @app.callback(
        Output(f"{prefix}-rows-area", "children"),
        Input(f"{prefix}-rows", "data"),
    )
    def _render(rows):
        rows = rows or []
        children = []
        for i, row in enumerate(rows):
            children.append(html.Div(
                [
                    dcc.Input(
                        id={"type": f"{prefix}-row", "field": "name", "row": i},
                        type="text",
                        value=row.get("name", ""),
                        placeholder="membrane name",
                        style={"flex": "2 1 0", "minWidth": "0",
                               "height": "22px"},
                    ),
                    dcc.Input(
                        id={"type": f"{prefix}-row", "field": "id", "row": i},
                        type="number",
                        value=row.get("id"),
                        placeholder="label id",
                        style={"flex": "1 1 0", "minWidth": "0",
                               "height": "22px"},
                    ),
                    dbc.Button(
                        "×",
                        id={"type": f"{prefix}-del-btn", "row": i},
                        color="link", size="sm",
                        style={"padding": "0 6px", "lineHeight": "1"},
                    ),
                ],
                style={**{"display": "flex", "alignItems": "center", "gap": "0.3rem"}, "marginBottom": "0.2rem"},
            ))
        return children

    @app.callback(
        Output(f"{prefix}-rows", "data", allow_duplicate=True),
        Input(f"{prefix}-add-btn", "n_clicks"),
        State(f"{prefix}-rows", "data"),
        prevent_initial_call=True,
    )
    def _add(n_clicks, rows):
        if not n_clicks:
            return no_update
        rows = list(rows or [])
        next_id = (max((int(r["id"]) for r in rows if r.get("id") is not None), default=0) + 1)
        rows.append({"name": "", "id": next_id})
        return rows

    @app.callback(
        Output(f"{prefix}-rows", "data", allow_duplicate=True),
        Input({"type": f"{prefix}-del-btn", "row": ALL}, "n_clicks"),
        State(f"{prefix}-rows", "data"),
        prevent_initial_call=True,
    )
    def _delete(n_clicks_list, rows):
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "row" in triggered):
            return no_update
        if not any(n_clicks_list):
            return no_update
        rows = list(rows or [])
        idx = int(triggered["row"])
        if 0 <= idx < len(rows):
            rows.pop(idx)
        return rows

    @app.callback(
        Output(f"{prefix}-rows", "data", allow_duplicate=True),
        Input({"type": f"{prefix}-row", "field": ALL, "row": ALL}, "value"),
        State({"type": f"{prefix}-row", "field": ALL, "row": ALL}, "id"),
        State(f"{prefix}-rows", "data"),
        prevent_initial_call=True,
    )
    def _edit(values, ids, rows):
        rows = list(rows or [])
        # Mutate a copy keyed by row index.
        for ident, value in zip(ids, values):
            i = int(ident["row"])
            if i >= len(rows):
                continue
            rows[i] = dict(rows[i])
            if ident["field"] == "name":
                rows[i]["name"] = value or ""
            else:
                rows[i]["id"] = value
        return rows


def read_label_dict(rows: list[dict] | None) -> dict[str, int]:
    """Convert the store payload to the python dict the pipeline accepts.

    Rows with blank names or missing ids are silently dropped (the user is
    still typing). Duplicate names take the last-typed id.
    """
    out: dict[str, int] = {}
    for r in rows or []:
        name = (r.get("name") or "").strip()
        try:
            label_id = int(r["id"])
        except (TypeError, ValueError, KeyError):
            continue
        if not name:
            continue
        out[name] = label_id
    return out


# ── per-membrane mode field ──────────────────────────────────────────────────


_MODE_CHOICES = [
    {"label": "planar", "value": "planar"},
    {"label": "closed", "value": "closed"},
]


def get_per_membrane_mode_field(prefix: str, default_mode: str = "planar") -> html.Div:
    """Single-mode dropdown with an optional per-membrane override.

    When the "Per-membrane override" switch is off, the rendered value is a
    single Literal (``"planar"`` or ``"closed"``). When on, the widget shows
    one dropdown per name listed in the linked label-dict (read from the
    page's ``f"{prefix}-labels-store"`` mirror) so the user can pick the mode
    per membrane.

    The page is responsible for keeping the labels store in sync with the
    label-dict widget (a tiny relay callback in :mod:`pmemthick`).

    Stores / ids owned:

    * ``f"{prefix}-toggle"`` -- the override switch (bool).
    * ``f"{prefix}-single-mode"`` -- the single-mode dropdown.
    * ``f"{prefix}-labels-store"`` -- mirror of membrane names; the page
      updates it whenever the label-dict changes.
    * ``{"type": f"{prefix}-per-label-mode", "label": <name>}`` -- per-label
      dropdowns rendered into ``f"{prefix}-per-label-area"`` by the callback.
    """
    return html.Div(
        [
            html.Div(
                [
                    dbc.Switch(
                        id=f"{prefix}-toggle",
                        value=False,
                        label="Per-membrane override",
                        style={"marginRight": "0.5rem"},
                    ),
                    make_dropdown(
                        f"{prefix}-single-mode",
                        _MODE_CHOICES,
                        default_mode,
                        clearable=False,
                        style={"width": "160px"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "0.4rem"},
            ),
            dcc.Store(id=f"{prefix}-labels-store", data=[]),
            html.Div(id=f"{prefix}-per-label-area", style={"marginTop": "0.4rem"}),
        ],
        id=f"{prefix}-container",
    )


def register_per_membrane_mode_callbacks(app, prefix: str) -> None:
    """Wire the per-label area's rendering off the labels-store + toggle."""

    @app.callback(
        Output(f"{prefix}-per-label-area", "children"),
        Input(f"{prefix}-toggle", "value"),
        Input(f"{prefix}-labels-store", "data"),
        State(f"{prefix}-single-mode", "value"),
    )
    def _render(toggle_on, labels, default_mode):
        if not toggle_on or not labels:
            return []
        rows = []
        for name in labels:
            rows.append(html.Div(
                [
                    html.Label(
                        name,
                        style={"flex": "1 1 0"},
                    ),
                    make_dropdown(
                        {"type": f"{prefix}-per-label-mode", "label": name},
                        _MODE_CHOICES,
                        default_mode or "planar",
                        clearable=False,
                        style={"flex": "1 1 0"},
                    ),
                ],
                style={**{"display": "flex", "alignItems": "center", "gap": "0.4rem"}, "marginBottom": "0.2rem"},
            ))
        return rows


def read_per_membrane_mode(
    toggle: bool,
    single_mode: str,
    per_label_ids: list[dict],
    per_label_values: list[str],
) -> str | dict[str, str]:
    """Convert widget state into the pipeline's union value.

    Returns
    -------
    str or dict[str, str]
        A single Literal when the override is off; a ``{label: mode}`` dict
        otherwise. Drops any rows whose value is None (the user hasn't
        picked) and falls back to ``"planar"`` for those.
    """
    if not toggle:
        return single_mode or "planar"
    out: dict[str, str] = {}
    for ident, value in zip(per_label_ids or [], per_label_values or []):
        name = ident.get("label")
        if not name:
            continue
        out[name] = value or (single_mode or "planar")
    if not out:
        return single_mode or "planar"
    return out


# ── analyzer sub-form ────────────────────────────────────────────────────────


def get_analyzer_subform(prefix: str) -> html.Div:
    """Nested form for :class:`cryocat.analysis.memthick.IntensityProfileAnalyzer`.

    Implemented as a thin wrapper around :func:`cryocat.app.formgen.build_form`
    so the analyzer's scalar / Literal / tuple params are surfaced via the
    same pipeline as the rest of the suite. The Phase-1 tuple widget
    automatically handles ``minima_search_nm: tuple[float, float]``.

    The control ids use ``id_type=prefix`` directly so the page reads them
    with a single ``State({"type": prefix, "param": ALL, ...}, ...)`` matcher.
    """
    rows = formgen.build_form(
        IntensityProfileAnalyzer,
        id_type=prefix,
    )
    return html.Div(rows, id=f"{prefix}-container")


def register_analyzer_subform_callbacks(app, prefix: str) -> None:
    """The analyzer sub-form has no reactive state of its own.

    Kept as a no-op for contract symmetry with the other composite widgets;
    the page reads values via the standard ALL-state pattern (see
    :func:`read_analyzer_kwargs`).
    """
    # No-op.


def read_analyzer_kwargs(ids: list[dict], values: list[Any], pool_state) -> dict:
    """Re-use :func:`generate_kwargs` to turn the sub-form's ALL-state into
    the kwargs dict an :class:`IntensityProfileAnalyzer` constructor accepts.

    Filters out the empty / default-meaning values so the code generator
    emits ``IntensityProfileAnalyzer(<only set params>)`` -- a compact call
    that mirrors how a hand-written script would be written.
    """
    kwargs = generate_kwargs(ids, values, pool_state) if (ids and values) else {}
    return {k: v for k, v in kwargs.items() if v not in (None, "", [])}
