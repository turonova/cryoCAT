"""Data pool page — load, view, and publish heterogeneous datasets.

Layout: sticky sidebar with four accordion sections (Load, Data pool, View
options, Publish) + main area with four overlaid panels (table, graph, dict,
empty state).  Panel visibility is switched by CSS display; only one is shown
at a time.

Contract
--------
Exposes ``layout``, ``register_callbacks(app)``, and ``DYNAMIC_IDS``.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import plotly.graph_objects as go

import dash
from dash import html, dcc, Input, Output, State, no_update, ctx, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from cryocat.app import ids, styles
from cryocat.app import datapool
from cryocat.app import discovery
from cryocat.app import formgen
from cryocat.app.apputils import run_operation, generate_kwargs
from cryocat.app.pageshell import page_shell, sidebar_accordion
from cryocat.app.components.pathfield import get_path_field
from cryocat.app.components.tableview import get_table_component, register_table_callbacks
from cryocat.app.components.tableplot import register_table_plot_callbacks
from cryocat.app.components.tablecluster import register_table_cluster_callbacks
from cryocat.app.components.graphsettings import styled_figure
from cryocat.app.components.volumeview import mesh_at
from cryocat.app.datapool import DataPoolState, DataPayloadMissing
from cryocat.app.pool import resolve_df as pool_resolve_df, resolve_n_rows as pool_resolve_n_rows


# ── Dynamic IDs for the suite app router ──────────────────────────────────────

DYNAMIC_IDS: list[tuple[str, str]] = [
    ("dp-view-tabv-grid-container", "dp-view-tabv-grid"),
]

# ── Panel visibility helpers ──────────────────────────────────────────────────

_SHOW: dict = {"display": "block"}
_HIDE: dict = {"display": "none"}


# ── Module-level helpers ──────────────────────────────────────────────────────

def _reader_options() -> list[dict]:
    """Build dropdown options from the reader registry, sorted by label."""
    return [{"label": e.label, "value": e.key} for e in discovery.readers()]


def _render_pool_entry(entry_dict: dict) -> html.Div:
    """Render one pool list item with label, kind badge, and remove button."""
    data_id = entry_dict["data_id"]
    label   = entry_dict.get("label", data_id)
    kind    = entry_dict.get("kind", "?")
    n_rows  = entry_dict.get("n_rows")
    meta    = f"{kind}" + (f" · {n_rows:,}" if n_rows is not None else "")
    return html.Div(
        [
            html.Span(
                label,
                style={
                    "flex": "1 1 0",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "whiteSpace": "nowrap",
                    "fontWeight": 600,
                },
            ),
            html.Span(
                f"[{meta}]",
                style={"fontSize": styles.FONT_SM, "color": styles.COLOR_MUTED, "flexShrink": 0},
            ),
            dbc.Button(
                "✕",
                id={"type": "dp-remove-btn", "data_id": data_id},
                size="sm",
                color=styles.BTN_NEUTRAL,
                n_clicks=0,
                style={"flexShrink": 0, "padding": "0 4px"},
            ),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "0.4rem",
            "padding": "3px 4px",
            "borderRadius": "4px",
        },
    )


def _do_load(
    path: str,
    reader_key: str,
    label_val: str | None,
    extra_values: list,
    extra_ids: list,
    registry: dict,
    next_id: int,
) -> tuple:
    """Execute a reader load; return (registry, next_id, status, selected_id)."""
    entry = discovery.get(reader_key)
    extra_kwargs = generate_kwargs(extra_ids, extra_values) if extra_ids else {}
    kwargs: dict = ({entry.path_arg: path} if entry.path_arg else {})
    kwargs.update(extra_kwargs)
    try:
        result = run_operation(entry.fn, kwargs)
    except Exception as exc:
        return registry, next_id, f"Error ({entry.label}, {pathlib.Path(path).name}): {exc}", None
    if result is None:
        return registry, next_id, f"Reader {entry.label!r} returned None.", None
    effective_label = (label_val or "").strip() or pathlib.Path(path).stem
    state = DataPoolState.from_stores(registry, next_id)
    state, data_id = datapool.insert_entry(
        state, result, label=effective_label, reader=reader_key, source_path=path,
    )
    return *state.to_stores(), f"Loaded {effective_label!r}.", data_id


def _do_remove(
    data_id: str,
    registry: dict,
    next_id: int,
    selected: str | None,
) -> tuple[DataPoolState, str | None]:
    """Remove an entry; return (new_state, new_selected_id)."""
    state = DataPoolState.from_stores(registry, next_id)
    state = datapool.remove_entry(state, data_id)
    new_sel = None if selected == data_id else selected
    return state, new_sel


def _do_select(
    data_id: str | None,
    registry: dict,
    rev: int,
) -> tuple:
    """Compute panel visibility and tablegrid reference for the selected entry.

    Returns
    -------
    (new_rev, tabv_store_data, table_style, graph_style, dict_style, empty_style)
    """
    reg = registry or {}
    if not data_id or data_id not in reg:
        datapool.clear_view_df()
        return rev, None, _HIDE, _HIDE, _HIDE, _SHOW

    kind    = reg[data_id].get("kind", "dataframe")
    state   = DataPoolState.from_stores(reg, 0)
    new_rev = (rev or 0) + 1

    if kind in ("dataframe", "array"):
        datapool.set_view_df(data_id, state)
        ref = {"motl_id": "dp-view", "rev": new_rev}
        return new_rev, ref, _SHOW, _HIDE, _HIDE, _HIDE

    datapool.clear_view_df()
    if kind == "volume":
        return new_rev, None, _HIDE, _SHOW, _HIDE, _HIDE
    if kind == "dict":
        return new_rev, None, _HIDE, _HIDE, _SHOW, _HIDE
    return new_rev, None, _HIDE, _HIDE, _HIDE, _SHOW


def _do_publish(data_id: str | None, name: str | None, registry: dict) -> str:
    """Bind *data_id*'s payload to a console variable.  Returns a status string."""
    from cryocat.app.console.execute import _CONSOLE_LOCALS
    if not data_id:
        return "No entry selected."
    if not name or not name.strip():
        return "Provide a variable name."
    name = name.strip()
    if not name.isidentifier():
        return f"{name!r} is not a valid Python identifier."
    try:
        payload = datapool.get_payload(data_id)
    except DataPayloadMissing as exc:
        return str(exc)
    _CONSOLE_LOCALS[name] = payload
    return f"Registered as @{name}."


def _vol_figure(data_id: str, level: float, gs: dict) -> go.Figure:
    """Build an isosurface figure from a 3D-volume payload."""
    try:
        vol = datapool.get_payload(data_id)
        if not isinstance(vol, np.ndarray) or vol.ndim != 3:
            return styled_figure(go.Figure(), gs or {}, uirevision="dp-vol-empty")
        vmin, vmax = float(vol.min()), float(vol.max())
        lvl = float(np.clip(level, vmin + 1e-6, vmax - 1e-6))
        mesh = mesh_at(vol.astype(np.float32), lvl)
        traces = [go.Mesh3d(**mesh, color="lightblue", opacity=0.7, name="Isosurface")] if mesh else []
        n0, n1, n2 = vol.shape
        m = max(n0, n1, n2)
        scene = {
            "xaxis": {"range": [0, n0]}, "yaxis": {"range": [0, n1]},
            "zaxis": {"range": [0, n2]},
            "aspectmode": "manual",
            "aspectratio": {"x": n0 / m, "y": n1 / m, "z": n2 / m},
        }
        return styled_figure(
            go.Figure(data=traces), gs or {}, uirevision="dp-vol",
            margin={"t": 0, "b": 0, "l": 0, "r": 0}, scene=scene, height=600,
        )
    except Exception:
        return styled_figure(go.Figure(), gs or {}, uirevision="dp-vol-empty")


def _arr_figure(data_id: str, gs: dict) -> go.Figure:
    """Build a line / scatter figure from a 1D or 2D ndarray payload."""
    try:
        arr = datapool.get_payload(data_id)
        if not isinstance(arr, np.ndarray):
            return styled_figure(go.Figure(), gs or {}, uirevision="dp-arr-empty")
        if arr.ndim == 1:
            traces = [go.Scatter(y=arr.tolist(), mode="lines", name="data")]
        else:
            traces = [
                go.Scatter(y=arr[:, c].tolist(), mode="lines", name=f"col_{c}")
                for c in range(min(arr.shape[1], 20))
            ]
        return styled_figure(go.Figure(data=traces), gs or {}, uirevision="dp-arr")
    except Exception:
        return styled_figure(go.Figure(), gs or {}, uirevision="dp-arr-empty")


def _vol_or_arr_figure(
    data_id: str | None,
    level: float,
    registry: dict,
    gs: dict,
) -> go.Figure:
    """Route to the appropriate figure builder for the selected entry."""
    reg = registry or {}
    if not data_id or data_id not in reg:
        return styled_figure(go.Figure(), gs or {}, uirevision="dp-empty")
    kind = reg[data_id].get("kind", "")
    if kind == "volume":
        return _vol_figure(data_id, level, gs)
    if kind == "array":
        return _arr_figure(data_id, gs)
    return styled_figure(go.Figure(), gs or {}, uirevision="dp-empty")


def _dict_text(data_id: str | None) -> str:
    """Return JSON-pretty-printed text of a dict payload; empty string on error."""
    if not data_id:
        return ""
    try:
        payload = datapool.get_payload(data_id)
        return json.dumps(payload, indent=2, default=str)
    except Exception as exc:
        return f"Error: {exc}"


# ── Layout ────────────────────────────────────────────────────────────────────

def _make_stores() -> list:
    return [
        dcc.Store(id="dp-selected-id", data=None),
        dcc.Store(id="dp-view-rev",    data=0),
        dcc.Store(id="dp-view-tabv-global-data-store", data=None),
    ]


def _sidebar() -> list:
    return [
        sidebar_accordion(
            [
                # ── Load ───────────────────────────────────────────────────────
                dbc.AccordionItem(
                    [
                        formgen.form_row(
                            "reader",
                            formgen.make_dropdown(
                                "dp-reader-dd",
                                options=_reader_options(),
                                value=None,
                                clearable=True,
                            ),
                            "Select a reader function to parse the input file.",
                            label_id="dp-reader-lbl",
                            label_text="Reader",
                        ),
                        formgen.form_row(
                            "path",
                            get_path_field(
                                "dp-path",
                                mode="open",
                                extensions=(),
                                placeholder="Select file…",
                            ),
                            "Absolute path to the file to load.",
                            label_id="dp-path-lbl",
                            label_text="File",
                        ),
                        html.Div(id="dp-params-form", children=[]),
                        formgen.form_row(
                            "label",
                            dbc.Input(
                                id="dp-label",
                                type="text",
                                placeholder="auto",
                                debounce=True,
                            ),
                            "Human-readable name for this dataset (blank = file stem).",
                            truly_optional=True,
                            label_id="dp-label-lbl",
                            label_text="Label",
                        ),
                        dbc.Button(
                            "Load",
                            id="dp-load-btn",
                            color=styles.BTN_PRIMARY,
                            size="sm",
                            style={"width": "100%"},
                        ),
                        html.Div(id="dp-load-status", style=styles.HINT),
                    ],
                    title="Load",
                    item_id="dp-acc-load",
                ),
                # ── Data pool ──────────────────────────────────────────────────
                dbc.AccordionItem(
                    [
                        html.Div(id="dp-pool-list", children=[]),
                    ],
                    title="Data pool",
                    item_id="dp-acc-pool",
                ),
                # ── View options ───────────────────────────────────────────────
                dbc.AccordionItem(
                    [
                        formgen.form_row(
                            "iso_level",
                            dcc.Slider(
                                id="dp-iso-level",
                                min=0.0, max=1.0, step=0.01, value=0.5,
                                tooltip={"placement": "bottom"},
                            ),
                            "Isosurface level for 3D volume display (0 = min, 1 = max).",
                            label_id="dp-iso-lbl",
                            label_text="Iso level",
                        ),
                    ],
                    title="View options",
                    item_id="dp-acc-view",
                ),
                # ── Register as variable ───────────────────────────────────────
                dbc.AccordionItem(
                    [
                        formgen.form_row(
                            "variable_name",
                            dbc.Input(
                                id="dp-publish-name",
                                type="text",
                                placeholder="my_data",
                                debounce=True,
                            ),
                            "Python identifier for console access via @name.",
                            label_id="dp-publish-name-lbl",
                            label_text="Variable name",
                        ),
                        dbc.Button(
                            "Register as @name",
                            id="dp-publish-btn",
                            color=styles.BTN_SECONDARY,
                            size="sm",
                            style={"width": "100%"},
                        ),
                        html.Div(id="dp-publish-status", style=styles.HINT),
                    ],
                    title="Register as variable",
                    item_id="dp-acc-publish",
                ),
            ],
            active_item=["dp-acc-load", "dp-acc-pool"],
        ),
    ]


def _main() -> list:
    return [
        html.Div(
            [get_table_component("dp-view-tabv", show_create_from_selected=True)],
            id="dp-panel-table",
            style=_HIDE,
        ),
        html.Div(
            [
                dcc.Graph(
                    id="dp-view-graph",
                    style={"height": "70vh"},
                    config={"displaylogo": False},
                ),
            ],
            id="dp-panel-graph",
            style=_HIDE,
        ),
        html.Div(
            [
                html.Pre(
                    id="dp-view-dict",
                    style={
                        "whiteSpace": "pre-wrap",
                        "fontSize": styles.FONT_SM,
                        "overflowY": "auto",
                        "maxHeight": "70vh",
                        "padding": "0.5rem",
                    },
                ),
            ],
            id="dp-panel-dict",
            style=_HIDE,
        ),
        html.Div(
            "Select an entry from the pool to view it.",
            id="dp-panel-empty",
            style={**styles.HINT, "padding": "1rem"},
        ),
    ]


layout = html.Div(
    [*_make_stores(), page_shell(_sidebar(), _main(), sidebar_width=4)],
    style={"margin": "0", "padding": "0"},
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):  # noqa: C901
    """Register all data pool page callbacks."""

    # ── Build reader params form ───────────────────────────────────────────────
    @app.callback(
        Output("dp-params-form", "children"),
        Input("dp-reader-dd", "value"),
    )
    def _build_params_form(reader_key):
        if not reader_key:
            return []
        entry = discovery.get(reader_key)
        return formgen.build_form(entry, id_type="dp-param", exclude=list(entry.hide))

    # ── Load / Remove (single writer for all three mutable stores) ───────────────
    @app.callback(
        Output(ids.DATA_POOL_REGISTRY, "data"),
        Output(ids.DATA_POOL_NEXT_ID,  "data"),
        Output("dp-load-status",        "children"),
        Output("dp-selected-id",        "data"),
        Input("dp-load-btn",                              "n_clicks"),
        Input({"type": "dp-remove-btn", "data_id": ALL}, "n_clicks"),
        State({"type": "path-input", "owner": "dp-path"}, "value"),
        State("dp-reader-dd",  "value"),
        State("dp-label",      "value"),
        State({"type": "dp-param", "param": ALL, "tag": ALL}, "value"),
        State(ids.DATA_POOL_REGISTRY, "data"),
        State(ids.DATA_POOL_NEXT_ID,  "data"),
        State("dp-selected-id",        "data"),
        prevent_initial_call=True,
    )
    def _mutate(_load_n, _remove_list, path, reader_key, label_val, extra_values, registry, next_id, selected):
        trigger = ctx.triggered_id
        if trigger == "dp-load-btn":
            if not path or not reader_key:
                return no_update, no_update, "Select a file and reader.", no_update
            extra_ids = [s["id"] for s in ctx.states_list[3]]
            return _do_load(path, reader_key, label_val, extra_values, extra_ids, registry, next_id)
        if not any(n for n in (_remove_list or []) if n):
            raise PreventUpdate
        state, new_sel = _do_remove(trigger["data_id"], registry, next_id, selected)
        return *state.to_stores(), no_update, new_sel

    # ── Render pool list ───────────────────────────────────────────────────────
    @app.callback(
        Output("dp-pool-list", "children"),
        Input(ids.DATA_POOL_REGISTRY, "data"),
    )
    def _render_pool_list(registry):
        if not registry:
            return [html.Div("No data loaded.", style=styles.HINT)]
        return [_render_pool_entry(v) for v in registry.values()]

    # ── Select entry ───────────────────────────────────────────────────────────
    @app.callback(
        Output("dp-view-rev",                  "data"),
        Output("dp-view-tabv-global-data-store", "data"),
        Output("dp-panel-table",  "style"),
        Output("dp-panel-graph",  "style"),
        Output("dp-panel-dict",   "style"),
        Output("dp-panel-empty",  "style"),
        Input(ids.DATA_POOL_REGISTRY, "data"),
        Input("dp-selected-id",       "data"),
        State("dp-view-rev",          "data"),
    )
    def _select_entry(registry, data_id, rev):
        return _do_select(data_id, registry, rev)

    # ── Graph viewer ───────────────────────────────────────────────────────────
    @app.callback(
        Output("dp-view-graph", "figure"),
        Input("dp-selected-id",         "data"),
        Input("dp-iso-level",            "value"),
        State(ids.DATA_POOL_REGISTRY,    "data"),
        State(ids.GRAPH_SETTINGS_STORE,  "data"),
    )
    def _render_graph_viewer(data_id, level, registry, gs):
        return _vol_or_arr_figure(data_id, level or 0.5, registry, gs)

    # ── Dict viewer ────────────────────────────────────────────────────────────
    @app.callback(
        Output("dp-view-dict", "children"),
        Input("dp-selected-id", "data"),
    )
    def _render_dict_viewer(data_id):
        return _dict_text(data_id)

    # ── Publish ────────────────────────────────────────────────────────────────
    @app.callback(
        Output("dp-publish-status", "children"),
        Input("dp-publish-btn",     "n_clicks"),
        State("dp-selected-id",     "data"),
        State("dp-publish-name",    "value"),
        State(ids.DATA_POOL_REGISTRY, "data"),
        prevent_initial_call=True,
    )
    def _publish(n_clicks, data_id, name, registry):
        return _do_publish(data_id, name, registry)

    # ── Table sub-component callbacks ──────────────────────────────────────────
    register_table_callbacks(
        app, "dp-view-tabv",
        resolve_df=pool_resolve_df, resolve_n_rows=pool_resolve_n_rows,
    )
    register_table_plot_callbacks(
        app, "dp-view-tabv-table-plot", "dp-view-tabv-global-data-store", resolve_df=pool_resolve_df,
    )
    register_table_cluster_callbacks(
        app, "dp-view-tabv-table-cluster", "dp-view-tabv-global-data-store", pool_aware=True, resolve_df=pool_resolve_df,
    )

    # ── Formgen write-back callbacks for reader param fields ───────────────────
    formgen.register_path_writeback(app, "dp-param", None)
    formgen.register_var_picker_writeback(app, "dp-param", None)
