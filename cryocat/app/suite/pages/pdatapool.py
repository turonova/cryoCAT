"""Data pool page — view and publish heterogeneous datasets.

Layout: sticky sidebar with the table editor (entry picker + operations) at the
top, then secondary accordion sections (Data pool, View options, Register as
variable).  Main area shows the selected entry (table, graph, dict, or empty).

File loading has moved to the Utilities page (W3).  The table editor's source
picker in the sidebar is the primary selection mechanism for both viewing and
transforming entries (W4, W5).

Contract
--------
Exposes ``layout``, ``register_callbacks(app)``, and ``DYNAMIC_IDS``.
"""
from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go

from dash import html, dcc, Input, Output, State, no_update, ctx, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from cryocat.app import ids, styles
from cryocat.app import datapool
from cryocat.app import formgen
from cryocat.app.pageshell import page_shell
from cryocat.app.components.tableview import get_table_component, register_table_callbacks
from cryocat.app.components.tableplot import register_table_plot_callbacks
from cryocat.app.components.tablecluster import register_table_cluster_callbacks
from cryocat.app.components import tableeditor
from cryocat.app.components.graphsettings import styled_figure
from cryocat.app.components.volumeview import mesh_at
from cryocat.app.datapool import DataPoolState, DataPayloadMissing
from cryocat.app.pool import (
    resolve_df as pool_resolve_df,
    resolve_n_rows as pool_resolve_n_rows,
    replace_motl_rows,
    PoolState,
)
from cryocat.app.apputils import run_operation
from cryocat.app.components.customel import customel_graph


# ── Dynamic IDs for the suite app router ──────────────────────────────────────

DYNAMIC_IDS: list[tuple[str, str]] = [
    ("dp-view-tabv-grid-container", "dp-view-tabv-grid"),
]

# ── Panel visibility helpers ──────────────────────────────────────────────────

_SHOW: dict = {"display": "block"}
_HIDE: dict = {"display": "none"}


# ── Module-level helpers ──────────────────────────────────────────────────────

def _render_pool_entry(entry_dict: dict) -> html.Div:
    """Render one pool list item with label, kind badge, and remove button."""
    data_id    = entry_dict["data_id"]
    label      = entry_dict.get("label", data_id)
    kind       = entry_dict.get("kind", "?")
    n_rows     = entry_dict.get("n_rows")
    motl_links = entry_dict.get("motl_links") or {}
    meta = f"{kind}" + (f" · {n_rows:,}" if n_rows is not None else "")
    if motl_links:
        def _fmt(mid):
            return ", ".join(mid) if isinstance(mid, list) else str(mid)
        link_str = ", ".join(f"{r}:{_fmt(mid)}" for r, mid in motl_links.items())
        meta += f" · ↔ {link_str}"
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


def _do_remove(
    data_id: str,
    registry: dict,
    next_id: int,
    selected: str | None,
) -> tuple[DataPoolState, str | None]:
    """Remove an entry; return (new_state, new_selected_id)."""
    from cryocat.app.console.vars import unregister_console_var
    state = DataPoolState.from_stores(registry, next_id)
    state = datapool.remove_entry(state, data_id)
    unregister_console_var(data_id)
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
    from cryocat.app.console.vars import register_console_var
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
    register_console_var(name, payload)
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
        # Primary: entry picker + operations (working-copy mode — W1)
        tableeditor.get_table_editor("dp-edit", multi_source=True, working_copy_mode=True),
        html.Hr(style={"margin": f"{styles.SECTION_GAP} 0"}),
        # Working-copy commit section (W5: below operations, clearly separated)
        html.Div(
            id="dp-wc-section",
            style=_HIDE,  # shown by _on_wc_ui_update when ops are pending
            children=[
                html.Div(
                    id="dp-wc-indicator",
                    style={**styles.HINT, "marginBottom": styles.FORM_ROW_GAP},
                ),
                dbc.Button(
                    "Apply to original",
                    id="dp-wc-apply-btn",
                    color=styles.BTN_PRIMARY,
                    size="sm",
                    disabled=True,
                    style={"width": "100%"},
                    title="Apply the working copy back to the original entry (recorded).",
                ),
                dbc.Button(
                    "Save as new table",
                    id="dp-wc-save-btn",
                    color=styles.BTN_SECONDARY,
                    size="sm",
                    style={"width": "100%", "marginTop": styles.FORM_ROW_GAP},
                    title="Save the working copy as a new data pool entry; source is unchanged.",
                ),
                dbc.Button(
                    "Discard changes",
                    id="dp-wc-discard-btn",
                    color=styles.BTN_NEUTRAL,
                    size="sm",
                    style={"width": "100%", "marginTop": styles.FORM_ROW_GAP},
                    title="Discard the working copy and return to the original.",
                ),
                html.Div(
                    id="dp-wc-commit-status",
                    style={**styles.HINT, "marginTop": styles.FORM_ROW_GAP},
                ),
            ],
        ),
    ]


def _main() -> list:
    return [
        html.Div(
            get_table_component("dp-view-tabv", show_create_from_selected=True),
            id="dp-panel-table",
            style=_HIDE,
        ),
        html.Div(
            [
                customel_graph("dp", "view", dcc.Graph(
                    id={"type": "styled-graph", "owner": "dp", "name": "view"},
                    style={"height": "70vh"},
                    config={"displaylogo": False},
                )),
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
            "Select an entry from the picker to view it.",
            id="dp-panel-empty",
            style={**styles.HINT, "padding": "1rem"},
        ),
    ]


layout = html.Div(
    [*_make_stores(), page_shell(_sidebar(), _main(), sidebar_width=4)],
    style={"margin": "0", "padding": "0"},
)


# ── Working-copy helpers (module-level — thin-callback law) ───────────────────

def _wc_ui_update_op(wc_signal, src_ref):
    from cryocat.app.suite.pages._wcopy import (
        get_copy, get_meta, indicator_text, validate_for_apply, source_id_for_ref,
    )
    if not wc_signal or not src_ref:
        return "", True, "No pending changes.", _HIDE
    signal_source_id = wc_signal.get("source_id") if isinstance(wc_signal, dict) else None
    current_source_id = source_id_for_ref(src_ref)
    if signal_source_id != current_source_id:
        return "", True, "No pending changes.", _HIDE
    meta = get_meta(current_source_id)
    if meta.get("ops_count", 0) == 0:
        return "", True, "No pending changes.", _HIDE
    wc_df = get_copy(current_source_id)
    if wc_df is None:
        return "", True, "Working copy lost — discard and retry.", _SHOW
    ok, reason = validate_for_apply(wc_df, meta.get("source_kind", ""), meta.get("source_reader", ""))
    ind = indicator_text(current_source_id)
    if not ok:
        return ind, True, f"Cannot apply: {reason}", _SHOW
    return ind, False, "Apply the working copy back to the original entry (recorded).", _SHOW


def _apply_to_original_op(src_ref, pool_reg, pool_meta_data, pool_next_id, dp_reg, dp_next_id):
    from cryocat.app.suite.pages._wcopy import (
        get_copy, get_meta, validate_for_apply, source_changed, clear, source_id_for_ref,
    )
    _no = no_update
    _fail = (_no, _no, _no, _no, _no, _no)
    source_id = source_id_for_ref(src_ref)
    wc_df = get_copy(source_id)
    if wc_df is None:
        return *_fail, "No working copy found — nothing applied."
    meta = get_meta(source_id)
    ok, reason = validate_for_apply(wc_df, meta.get("source_kind", ""), meta.get("source_reader", ""))
    if not ok:
        return *_fail, f"Cannot apply: {reason}"
    if "motl_id" in src_ref:
        motl_id = src_ref["motl_id"]
        current_n = (pool_reg or {}).get(motl_id, {}).get("n_rows")
        warn = f" (warning: source was modified since copy was made)" if (
            current_n is not None and source_changed(source_id, current_n)
        ) else ""
        p = PoolState.from_stores(pool_reg, pool_meta_data, pool_next_id)
        p = run_operation(replace_motl_rows, {"state": p, "motl_id": motl_id, "rows": wc_df})
        clear(source_id)
        return *p.to_stores(), _no, _no, None, f"Applied to {motl_id} (revision bumped).{warn}"
    if "data_id" in src_ref:
        data_id = src_ref["data_id"]
        current_n = (dp_reg or {}).get(data_id, {}).get("n_rows")
        warn = f" (warning: source was modified since copy was made)" if (
            current_n is not None and source_changed(source_id, current_n)
        ) else ""
        ds = DataPoolState.from_stores(dp_reg, dp_next_id)
        ds = run_operation(datapool.replace_payload, {"state": ds, "data_id": data_id, "df": wc_df})
        clear(source_id)
        return _no, _no, _no, *ds.to_stores(), None, f"Applied to {data_id}.{warn}"
    return *_fail, "Unknown source type."


def _save_as_new_op(src_ref, label_val, dp_reg, dp_next_id):
    from cryocat.app.suite.pages._wcopy import get_copy, clear, source_id_for_ref
    _no = no_update
    source_id = source_id_for_ref(src_ref)
    wc_df = get_copy(source_id)
    if wc_df is None:
        return _no, _no, _no, _no, "No working copy found — nothing saved."
    label = (label_val or "").strip() or f"Working copy of {source_id}"
    ds = DataPoolState.from_stores(dp_reg, dp_next_id)
    ds, did = run_operation(
        datapool.insert_entry,
        {"state": ds, "payload": wc_df, "label": label, "reader": "table_op", "source_path": ""},
    )
    clear(source_id)
    return *ds.to_stores(), did, None, f"Saved as new table {did}."


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):  # noqa: C901
    """Register all data pool page callbacks."""

    # ── Remove entry ───────────────────────────────────────────────────────────
    @app.callback(
        Output(ids.DATA_POOL_REGISTRY, "data"),
        Output(ids.DATA_POOL_NEXT_ID,  "data"),
        Input({"type": "dp-remove-btn", "data_id": ALL}, "n_clicks"),
        State(ids.DATA_POOL_REGISTRY, "data"),
        State(ids.DATA_POOL_NEXT_ID,  "data"),
        prevent_initial_call=True,
    )
    def _remove_entry(_remove_list, registry, next_id):
        if not any(n for n in (_remove_list or []) if n):
            raise PreventUpdate
        trigger = ctx.triggered_id
        state, _ = _do_remove(trigger["data_id"], registry, next_id, None)
        return state.to_stores()

    # ── Select entry / refresh working-copy view ───────────────────────────────
    @app.callback(
        Output("dp-view-rev",                    "data"),
        Output("dp-view-tabv-global-data-store", "data"),
        Output("dp-panel-table",  "style"),
        Output("dp-panel-graph",  "style"),
        Output("dp-panel-dict",   "style"),
        Output("dp-panel-empty",  "style"),
        Input("dp-edit-src-ref",          "data"),
        Input("dp-edit-wc-changed",       "data"),  # fires on wc op or commit
        State(ids.DATA_POOL_REGISTRY,     "data"),
        State("dp-view-rev",              "data"),
    )
    def _select_entry(src_ref, wc_signal, dp_registry, rev):
        from cryocat.app.suite.pages._wcopy import get_copy, source_id_for_ref
        # When working copy is active for the current source, show it
        if src_ref and wc_signal:
            signal_source_id = wc_signal.get("source_id") if isinstance(wc_signal, dict) else None
            current_source_id = source_id_for_ref(src_ref)
            if signal_source_id and signal_source_id == current_source_id:
                wc_df = get_copy(current_source_id)
                if wc_df is not None:
                    datapool.set_view_df_direct(wc_df)
                    new_rev = (rev or 0) + 1
                    return new_rev, {"motl_id": "dp-view", "rev": new_rev}, _SHOW, _HIDE, _HIDE, _HIDE
        # Normal routing (wc cleared or different source selected)
        if not src_ref:
            datapool.clear_view_df()
            return rev, None, _HIDE, _HIDE, _HIDE, _SHOW
        if "motl_id" in src_ref:
            new_rev = (rev or 0) + 1
            ref = {"motl_id": src_ref["motl_id"], "rev": new_rev}
            return new_rev, ref, _SHOW, _HIDE, _HIDE, _HIDE
        if "data_id" in src_ref:
            return _do_select(src_ref["data_id"], dp_registry, rev)
        datapool.clear_view_df()
        return rev, None, _HIDE, _HIDE, _HIDE, _SHOW

    # ── Sync dp-selected-id (written by tableeditor Apply) → picker ────────────
    @app.callback(
        Output("dp-edit-src-dd", "value", allow_duplicate=True),
        Input("dp-selected-id", "data"),
        prevent_initial_call=True,
    )
    def _sync_selection_to_picker(data_id):
        if not data_id:
            return no_update
        return f"data:{data_id}"

    # ── Clear picker when selected entry is removed ────────────────────────────
    @app.callback(
        Output("dp-edit-src-dd", "value", allow_duplicate=True),
        Input(ids.DATA_POOL_REGISTRY, "data"),
        Input(ids.POOL_REGISTRY,      "data"),
        State("dp-edit-src-dd",       "value"),
        prevent_initial_call=True,
    )
    def _clear_picker_if_stale(dp_reg, pool_reg, current_val):
        if not current_val:
            return no_update
        if current_val.startswith("motl:"):
            mid = current_val[5:]
            if mid not in (pool_reg or {}):
                return None
        elif current_val.startswith("data:"):
            did = current_val[5:]
            if did not in (dp_reg or {}):
                return None
        return no_update

    # ── Graph viewer ───────────────────────────────────────────────────────────
    @app.callback(
        Output({"type": "styled-graph", "owner": "dp", "name": "view"}, "figure"),
        Input("dp-edit-src-ref",         "data"),
        State(ids.DATA_POOL_REGISTRY,    "data"),
        State(ids.GRAPH_SETTINGS_STORE,  "data"),
    )
    def _render_graph_viewer(src_ref, registry, gs):
        data_id = (src_ref or {}).get("data_id")
        return _vol_or_arr_figure(data_id, 0.5, registry, gs)

    # ── Dict viewer ────────────────────────────────────────────────────────────
    @app.callback(
        Output("dp-view-dict", "children"),
        Input("dp-edit-src-ref", "data"),
    )
    def _render_dict_viewer(src_ref):
        data_id = (src_ref or {}).get("data_id")
        return _dict_text(data_id)

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

    # ── Working-copy UI: indicator + Apply button disabled state ──────────────
    @app.callback(
        Output("dp-wc-indicator",  "children"),
        Output("dp-wc-apply-btn",  "disabled"),
        Output("dp-wc-apply-btn",  "title"),
        Output("dp-wc-section",    "style"),
        Input("dp-edit-wc-changed", "data"),
        State("dp-edit-src-ref",    "data"),
    )
    def _on_wc_ui_update(wc_signal, src_ref):
        return _wc_ui_update_op(wc_signal, src_ref)

    # ── Working-copy commit: Apply to original ─────────────────────────────────
    @app.callback(
        # allow_duplicate: tableeditor._on_apply (modal path) also writes pool stores
        Output(ids.POOL_REGISTRY,     "data", allow_duplicate=True),
        Output(ids.POOL_META,         "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID,      "data", allow_duplicate=True),
        Output(ids.DATA_POOL_REGISTRY,"data", allow_duplicate=True),
        Output(ids.DATA_POOL_NEXT_ID, "data", allow_duplicate=True),
        # allow_duplicate: tableeditor._on_apply_wc also writes this store
        Output("dp-edit-wc-changed",   "data", allow_duplicate=True),
        # allow_duplicate: _on_save_as_new and _on_discard also write this
        Output("dp-wc-commit-status",  "children", allow_duplicate=True),
        Input("dp-wc-apply-btn",       "n_clicks"),
        State("dp-edit-src-ref",      "data"),
        State(ids.POOL_REGISTRY,      "data"),
        State(ids.POOL_META,          "data"),
        State(ids.POOL_NEXT_ID,       "data"),
        State(ids.DATA_POOL_REGISTRY, "data"),
        State(ids.DATA_POOL_NEXT_ID,  "data"),
        prevent_initial_call=True,
    )
    def _on_apply_to_original(
        n_clicks, src_ref, pool_reg, pool_meta_data, pool_next_id, dp_reg, dp_next_id,
    ):
        if not n_clicks or not src_ref:
            raise PreventUpdate
        return _apply_to_original_op(src_ref, pool_reg, pool_meta_data, pool_next_id, dp_reg, dp_next_id)

    # ── Working-copy commit: Save as new table ─────────────────────────────────
    @app.callback(
        # allow_duplicate: tableeditor._on_apply (modal) also writes DATA_POOL_REGISTRY
        Output(ids.DATA_POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.DATA_POOL_NEXT_ID,  "data", allow_duplicate=True),
        # allow_duplicate: tableeditor._on_apply (modal) also writes dp-selected-id
        Output("dp-selected-id",       "data", allow_duplicate=True),
        # allow_duplicate: tableeditor._on_apply_wc and _on_discard also write this
        Output("dp-edit-wc-changed",   "data", allow_duplicate=True),
        Output("dp-wc-commit-status",  "children", allow_duplicate=True),
        Input("dp-wc-save-btn",        "n_clicks"),
        State("dp-edit-src-ref",       "data"),
        State("dp-edit-label",         "value"),
        State(ids.DATA_POOL_REGISTRY,  "data"),
        State(ids.DATA_POOL_NEXT_ID,   "data"),
        prevent_initial_call=True,
    )
    def _on_save_as_new(n_clicks, src_ref, label_val, dp_reg, dp_next_id):
        if not n_clicks or not src_ref:
            raise PreventUpdate
        return _save_as_new_op(src_ref, label_val, dp_reg, dp_next_id)

    # ── Working-copy commit: Discard changes ───────────────────────────────────
    @app.callback(
        # allow_duplicate: tableeditor._on_apply_wc and _on_apply_to_original also write this
        Output("dp-edit-wc-changed",  "data", allow_duplicate=True),
        Output("dp-wc-commit-status", "children", allow_duplicate=True),
        Input("dp-wc-discard-btn",    "n_clicks"),
        State("dp-edit-src-ref",      "data"),
        prevent_initial_call=True,
    )
    def _on_discard(n_clicks, src_ref):
        from cryocat.app.suite.pages._wcopy import clear, source_id_for_ref
        if not n_clicks or not src_ref:
            raise PreventUpdate
        source_id = source_id_for_ref(src_ref)
        clear(source_id)
        return None, "Working copy discarded."

    # ── Table editor callbacks (W1–W7) — sidebar mount (working-copy mode) ──────
    tableeditor.register_table_editor_callbacks(app, "dp-edit", multi_source=True, working_copy_mode=True)

