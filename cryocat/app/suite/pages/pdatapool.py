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
    insert_motl as _pool_insert_motl,
    PoolState,
)
from cryocat.app.components.poolslotlist import (
    get_pool_slot_list,
    register_pool_slot_list_callbacks,
    register_slot_focus_callback,
)
from cryocat.core.cryomotl import Motl
from cryocat.app.apputils import run_operation
from cryocat.app.components.customel import customel_graph


# ── Dynamic IDs for the suite app router ──────────────────────────────────────

DYNAMIC_IDS: list[tuple[str, str]] = [
    (f"dp-view-{i}-tabv-grid-container", f"dp-view-{i}-tabv-grid")
    for i in range(5)
]

# ── Panel visibility helpers ──────────────────────────────────────────────────

_SHOW: dict = {"display": "block"}
_HIDE: dict = {"display": "none"}


def dp_resolve_df(ref: dict | None):
    """Resolve a data-pool ref to a DataFrame.

    For refs that carry ``data_id``, fetches directly from the data pool without
    going through the shared ``dp-view`` bridge — each per-slot grid can resolve
    its own entry independently.  Refs without ``data_id`` (WC-preview bridge refs
    and motl refs) delegate to ``pool_resolve_df``.
    """
    if ref is None:
        return None
    if isinstance(ref, dict) and "data_id" in ref:
        try:
            return datapool.get_payload(ref["data_id"])
        except Exception:
            return None
    return pool_resolve_df(ref)


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


def _dp_slot_ref(
    data_id: str | None,
    registry: dict,
    entry_revs: dict,
) -> dict | None:
    """Return the per-slot store ref for *data_id*, or None for non-grid kinds."""
    reg = registry or {}
    if not data_id or data_id not in reg:
        return None
    kind = reg[data_id].get("kind", "dataframe")
    if kind not in ("dataframe", "array"):
        return None
    entry_rev = (entry_revs or {}).get(data_id, 0)
    ref: dict = {"motl_id": "dp-view", "data_id": data_id, "rev": entry_rev}
    n_rows = reg[data_id].get("n_rows")
    if n_rows is not None:
        ref["n_rows"] = n_rows
    return ref


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

_N_SLOTS = 5


def _tab_to_idx(active_tab: str | None) -> int | None:
    if not active_tab or not active_tab.startswith("dp-slot-"):
        return None
    try:
        return int(active_tab.rsplit("-", 1)[-1])
    except ValueError:
        return None


def _slot_tab(i: int) -> dbc.Tab:
    return dbc.Tab(
        html.Div(
            get_table_component(f"dp-view-{i}-tabv", show_create_from_selected=True),
            style={"padding": "0.5rem"},
        ),
        id=f"dp-tab-{i}",
        tab_id=f"dp-slot-{i}",
        label=f"Slot {i + 1}",
        disabled=True,
    )


def _make_stores() -> list:
    per_slot = [
        dcc.Store(id=f"dp-view-{i}-tabv-global-data-store", data=None)
        for i in range(_N_SLOTS)
    ]
    return [
        dcc.Store(id="dp-selected-id",  data=None),
        dcc.Store(id="dp-view-rev",     data=0),
        dcc.Store(id="dp-entry-revs",   data={}),
        dcc.Store(id="dp-slot-map",     data=[None] * _N_SLOTS),
        dcc.Store(id="dp-active-id",    data=None),
        *per_slot,
    ]


_MOTL_COL_GROUPS: list[tuple[str, list[str]]] = [
    ("Coordinates (required)", ["x", "y", "z"]),
    ("Shifts",    ["shift_x", "shift_y", "shift_z"]),
    ("Angles",    ["phi", "psi", "theta"]),
    ("IDs",       ["subtomo_id", "tomo_id", "object_id"]),
    ("Class",     ["class", "score", "subtomo_mean"]),
    ("Geometry",  ["geom1", "geom2", "geom3", "geom4", "geom5"]),
]


def _build_motl_builder_section() -> html.Div:
    """Return the Build Motl accordion content."""
    col_rows = []
    for group_label, cols in _MOTL_COL_GROUPS:
        col_rows.append(formgen.section_divider(group_label))
        for mc in cols:
            is_req = mc in ("x", "y", "z")
            tip = f"Motl column '{mc}'" + (" (required)" if is_req else " — leave blank for default (0.0).")
            col_rows.append(
                formgen.form_row(
                    mc,
                    formgen.make_dropdown(
                        f"dp-mb-col-{mc}",
                        options=[],
                        value=None,
                        clearable=True,
                        placeholder="table column…" if not is_req else "required",
                    ),
                    tip,
                    truly_optional=not is_req,
                    label_id=f"dp-mb-col-{mc}-lbl",
                    label_text=mc,
                )
            )
    return html.Div([
        html.Div(
            "Map table columns to motl columns. x, y, z are required. "
            "subtomo_id is auto-generated (1…N) when not mapped.",
            style={**styles.HINT, "marginBottom": styles.FORM_ROW_GAP},
        ),
        formgen.form_row(
            "motl_label",
            dbc.Input(id="dp-mb-label", type="text", placeholder="auto"),
            "Label for the new motl entry.",
            truly_optional=True,
            label_id="dp-mb-label-lbl",
            label_text="Label",
        ),
        *col_rows,
        html.Div(style={"marginTop": styles.SECTION_GAP}),
        dbc.Button(
            "Build motl",
            id="dp-mb-build-btn",
            color=styles.BTN_PRIMARY,
            size="sm",
        ),
        html.Div(
            id="dp-mb-status",
            style={**styles.HINT, "marginTop": styles.FORM_ROW_GAP},
        ),
    ])


def _sidebar() -> list:
    from cryocat.app.pageshell import sidebar_accordion
    return [
        # Table slot list (GL2) — active-table slot mechanism
        sidebar_accordion([
            dbc.AccordionItem(
                [
                    get_pool_slot_list("dp"),
                    html.Div(
                        id="dp-slot-status",
                        style={**styles.HINT, "marginTop": styles.FORM_ROW_GAP},
                    ),
                ],
                title="Tables",
                item_id="dp-tables",
            ),
            dbc.AccordionItem(
                # Primary: entry picker + operations (working-copy mode — W1)
                tableeditor.get_table_editor("dp-edit", multi_source=True, working_copy_mode=True),
                title="Edit",
                item_id="dp-edit-tab",
            ),
            dbc.AccordionItem(
                _build_motl_builder_section(),
                title="Build Motl",
                item_id="dp-build-motl",
            ),
        ], active_item=["dp-tables"]),
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
        dbc.Tabs(
            [_slot_tab(i) for i in range(_N_SLOTS)],
            id="dp-tabs",
            active_tab="dp-slot-0",
            style={"marginBottom": styles.SECTION_GAP},
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
            "Select a slot above or assign an entry from the Tables panel.",
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


def _apply_to_original_op(src_ref, pool_reg, pool_meta_data, pool_next_id, dp_reg, dp_next_id, entry_revs):
    from cryocat.app.suite.pages._wcopy import (
        get_copy, get_meta, validate_for_apply, source_changed, clear, source_id_for_ref,
    )
    _no = no_update
    _fail = (_no, _no, _no, _no, _no, _no, _no)
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
        return *p.to_stores(), _no, _no, None, f"Applied to {motl_id} (revision bumped).{warn}", _no
    if "data_id" in src_ref:
        data_id = src_ref["data_id"]
        current_n = (dp_reg or {}).get(data_id, {}).get("n_rows")
        warn = f" (warning: source was modified since copy was made)" if (
            current_n is not None and source_changed(source_id, current_n)
        ) else ""
        ds = DataPoolState.from_stores(dp_reg, dp_next_id)
        ds = run_operation(datapool.replace_payload, {"state": ds, "data_id": data_id, "df": wc_df})
        clear(source_id)
        rm = dict(entry_revs or {})
        rm[data_id] = rm.get(data_id, 0) + 1
        return _no, _no, _no, *ds.to_stores(), None, f"Applied to {data_id}.{warn}", rm
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
        Output("dp-view-rev", "data"),
        *[Output(f"dp-view-{_i}-tabv-global-data-store", "data") for _i in range(_N_SLOTS)],
        Input("dp-edit-src-ref",      "data"),
        Input("dp-edit-wc-changed",   "data"),
        State(ids.DATA_POOL_REGISTRY, "data"),
        State("dp-view-rev",          "data"),
        State("dp-slot-map",          "data"),
        State("dp-active-id",         "data"),
        State("dp-entry-revs",        "data"),
    )
    def _select_entry(src_ref, wc_signal, dp_registry, rev, slot_map, active_id, entry_revs):
        from cryocat.app.suite.pages._wcopy import get_copy, source_id_for_ref
        sm = list(slot_map or [None] * _N_SLOTS)
        store_outs: list = [no_update] * _N_SLOTS
        try:
            slot_idx = sm.index(active_id) if active_id else -1
        except ValueError:
            slot_idx = -1

        # WC-preview path: working copy is active for current source
        if src_ref and wc_signal:
            signal_source_id = wc_signal.get("source_id") if isinstance(wc_signal, dict) else None
            current_source_id = source_id_for_ref(src_ref)
            if signal_source_id and signal_source_id == current_source_id:
                wc_df = get_copy(current_source_id)
                if wc_df is not None and slot_idx >= 0:
                    datapool.set_view_df_direct(wc_df)
                    new_rev = (rev or 0) + 1
                    store_outs[slot_idx] = {"motl_id": "dp-view", "rev": new_rev}
                    return new_rev, *store_outs

        # Normal routing (wc cleared, committed, or different source selected)
        if slot_idx < 0:
            return rev, *store_outs

        if not src_ref:
            datapool.clear_view_df()
            store_outs[slot_idx] = None
            return rev, *store_outs

        if "motl_id" in src_ref and "data_id" not in src_ref:
            new_rev = (rev or 0) + 1
            store_outs[slot_idx] = {"motl_id": src_ref["motl_id"], "rev": new_rev}
            return new_rev, *store_outs

        if "data_id" in src_ref:
            store_outs[slot_idx] = _dp_slot_ref(src_ref["data_id"], dp_registry, entry_revs)
            return rev, *store_outs

        datapool.clear_view_df()
        store_outs[slot_idx] = None
        return rev, *store_outs

    # ── Panel visibility (graph, dict, empty) — driven by active slot kind ─────
    @app.callback(
        Output("dp-panel-graph", "style"),
        Output("dp-panel-dict",  "style"),
        Output("dp-panel-empty", "style"),
        Input("dp-active-id",         "data"),
        State(ids.DATA_POOL_REGISTRY, "data"),
    )
    def _update_panels(active_id, dp_reg):
        if not active_id or active_id not in (dp_reg or {}):
            return _HIDE, _HIDE, _SHOW
        kind = (dp_reg or {}).get(active_id, {}).get("kind", "dataframe")
        if kind in ("dataframe", "array"):
            return _HIDE, _HIDE, _HIDE
        if kind == "volume":
            return _SHOW, _HIDE, _HIDE
        if kind == "dict":
            return _HIDE, _SHOW, _HIDE
        return _HIDE, _HIDE, _SHOW

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

    # ── Graph viewer — fires once per slot switch via dp-active-id ────────────
    @app.callback(
        Output({"type": "styled-graph", "owner": "dp", "name": "view"}, "figure"),
        Input("dp-active-id",            "data"),
        State(ids.DATA_POOL_REGISTRY,    "data"),
        State(ids.GRAPH_SETTINGS_STORE,  "data"),
    )
    def _render_graph_viewer(active_id, registry, gs):
        if not active_id:
            return styled_figure(go.Figure(), gs or {}, uirevision="dp-empty")
        kind = (registry or {}).get(active_id, {}).get("kind", "")
        if kind not in ("volume", "array"):
            return no_update
        return _vol_or_arr_figure(active_id, 0.5, registry, gs)

    # ── Dict viewer — fires once per slot switch via dp-active-id ─────────────
    @app.callback(
        Output("dp-view-dict", "children"),
        Input("dp-active-id",            "data"),
        State(ids.DATA_POOL_REGISTRY,    "data"),
    )
    def _render_dict_viewer(active_id, dp_reg):
        if not active_id:
            return ""
        kind = (dp_reg or {}).get(active_id, {}).get("kind", "")
        if kind != "dict":
            return no_update
        return _dict_text(active_id)

    # ── Per-slot table sub-component callbacks ─────────────────────────────────
    for _i in range(_N_SLOTS):
        register_table_callbacks(
            app, f"dp-view-{_i}-tabv",
            resolve_df=dp_resolve_df, resolve_n_rows=pool_resolve_n_rows,
        )
        register_table_plot_callbacks(
            app, f"dp-view-{_i}-tabv-table-plot",
            f"dp-view-{_i}-tabv-global-data-store",
            resolve_df=dp_resolve_df,
        )
        register_table_cluster_callbacks(
            app, f"dp-view-{_i}-tabv-table-cluster",
            f"dp-view-{_i}-tabv-global-data-store",
            pool_aware=True, resolve_df=dp_resolve_df,
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
        Output("dp-entry-revs",        "data", allow_duplicate=True),
        Input("dp-wc-apply-btn",       "n_clicks"),
        State("dp-edit-src-ref",      "data"),
        State(ids.POOL_REGISTRY,      "data"),
        State(ids.POOL_META,          "data"),
        State(ids.POOL_NEXT_ID,       "data"),
        State(ids.DATA_POOL_REGISTRY, "data"),
        State(ids.DATA_POOL_NEXT_ID,  "data"),
        State("dp-entry-revs",        "data"),
        prevent_initial_call=True,
    )
    def _on_apply_to_original(
        n_clicks, src_ref, pool_reg, pool_meta_data, pool_next_id, dp_reg, dp_next_id, entry_revs,
    ):
        if not n_clicks or not src_ref:
            raise PreventUpdate
        return _apply_to_original_op(src_ref, pool_reg, pool_meta_data, pool_next_id, dp_reg, dp_next_id, entry_revs)

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

    # ── GL2: Data pool slot machinery ─────────────────────────────────────────

    def _dp_remove_btn(data_id, entry):
        return [dbc.Button(
            "✕",
            id={"type": "dp-psl-remove-btn", "data_id": data_id},
            size="sm",
            color=styles.BTN_NEUTRAL,
            n_clicks=0,
            style={"flexShrink": 0, "padding": "0 4px"},
        )]

    register_pool_slot_list_callbacks(
        app, "dp",
        pool_registry_id=ids.DATA_POOL_REGISTRY,
        slot_map_id="dp-slot-map",
        n_slots=_N_SLOTS,
        row_extra_fn=_dp_remove_btn,
        active_id_store_id="dp-active-id",
    )

    # ── Slot tab strip: labels, active-id sync, slot_focus cleanup ───────────────

    @app.callback(
        *[Output(f"dp-tab-{i}", "label") for i in range(_N_SLOTS)],
        *[Output(f"dp-tab-{i}", "disabled") for i in range(_N_SLOTS)],
        Input("dp-slot-map", "data"),
        Input(ids.DATA_POOL_REGISTRY, "data"),
    )
    def _update_tab_labels(slot_map, registry):
        sm = list(slot_map or [None] * _N_SLOTS)
        while len(sm) < _N_SLOTS:
            sm.append(None)
        reg = datapool.clean_registry(registry)
        labels, disabled_flags = [], []
        for i, did in enumerate(sm):
            if did and did in reg:
                labels.append(reg[did].get("label", did))
                disabled_flags.append(False)
            else:
                labels.append(f"Slot {i + 1}")
                disabled_flags.append(True)
        return (*labels, *disabled_flags)

    @app.callback(
        Output("dp-active-id", "data", allow_duplicate=True),
        Input("dp-tabs", "active_tab"),
        State("dp-slot-map", "data"),
        State("dp-active-id", "data"),
        prevent_initial_call=True,
    )
    def _sync_tab_to_active_id(active_tab, slot_map, current_active):
        idx = _tab_to_idx(active_tab)
        if idx is None:
            return no_update
        sm = list(slot_map or [None] * _N_SLOTS)
        did = sm[idx] if idx < len(sm) else None
        if not did or did == current_active:
            return no_update
        return did

    @app.callback(
        Output("dp-tabs", "active_tab", allow_duplicate=True),
        Input("dp-active-id", "data"),
        State("dp-slot-map", "data"),
        State("dp-tabs", "active_tab"),
        prevent_initial_call=True,
    )
    def _sync_active_id_to_tab(active_id, slot_map, current_tab):
        if not active_id:
            return no_update
        sm = list(slot_map or [None] * _N_SLOTS)
        for i, did in enumerate(sm):
            if did == active_id:
                new_tab = f"dp-slot-{i}"
                if new_tab == current_tab:
                    return no_update
                return new_tab
        return no_update

    register_slot_focus_callback(
        app, "dp-slot-map", "dp-tabs", "dp-slot-", _N_SLOTS,
        active_id_store_id="dp-active-id",
    )

    # Sync slot-map + entry-revs → per-slot global-data-stores
    @app.callback(
        *[Output(f"dp-view-{_i}-tabv-global-data-store", "data", allow_duplicate=True) for _i in range(_N_SLOTS)],
        Input("dp-slot-map",          "data"),
        Input("dp-entry-revs",        "data"),
        State(ids.DATA_POOL_REGISTRY, "data"),
        *[State(f"dp-view-{_i}-tabv-global-data-store", "data") for _i in range(_N_SLOTS)],
        prevent_initial_call=True,
    )
    def _sync_pool_to_dp_slots(slot_map, entry_revs, dp_registry, *current_refs):
        sm = list(slot_map or [None] * _N_SLOTS)
        reg = dp_registry or {}
        rm = entry_revs or {}
        outs = []
        for i in range(_N_SLOTS):
            data_id = sm[i] if i < len(sm) else None
            current = current_refs[i] if i < len(current_refs) else None
            if not data_id or data_id not in reg:
                outs.append(None if current is not None else no_update)
                continue
            new_ref = _dp_slot_ref(data_id, reg, rm)
            outs.append(new_ref if new_ref != current else no_update)
        return tuple(outs)

    # Sync focused slot → tableeditor picker
    @app.callback(
        Output("dp-edit-src-dd", "value", allow_duplicate=True),
        Input("dp-active-id",     "data"),
        State(ids.DATA_POOL_REGISTRY, "data"),
        State("dp-edit-src-dd",   "value"),
        prevent_initial_call=True,
    )
    def _sync_active_to_picker(active_id, dp_reg, current_val):
        if not active_id or active_id not in (dp_reg or {}):
            return no_update
        expected = f"data:{active_id}"
        if current_val == expected:
            return no_update
        return expected

    # Remove entry from pool via slot list's remove button
    @app.callback(
        Output(ids.DATA_POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.DATA_POOL_NEXT_ID,  "data", allow_duplicate=True),
        Input({"type": "dp-psl-remove-btn", "data_id": ALL}, "n_clicks"),
        State(ids.DATA_POOL_REGISTRY, "data"),
        State(ids.DATA_POOL_NEXT_ID,  "data"),
        prevent_initial_call=True,
    )
    def _psl_remove_entry(n_list, registry, next_id):
        if not any(n for n in (n_list or []) if n):
            raise PreventUpdate
        data_id = ctx.triggered_id["data_id"]
        state, _ = _do_remove(data_id, registry, next_id, None)
        return state.to_stores()

    # ── GL1: Build motl from active table ─────────────────────────────────────

    # Populate motl-column dropdowns from the active table's columns
    @app.callback(
        *[Output(f"dp-mb-col-{mc}", "options") for group in _MOTL_COL_GROUPS for mc in group[1]],
        Input("dp-active-id",         "data"),
        Input(ids.DATA_POOL_REGISTRY, "data"),
        prevent_initial_call=False,
    )
    def _populate_mb_cols(active_id, dp_reg):
        _all_motl_cols = [mc for g in _MOTL_COL_GROUPS for mc in g[1]]
        n = len(_all_motl_cols)
        if not active_id or active_id not in (dp_reg or {}):
            return [[] for _ in range(n)]
        try:
            df = datapool.get_payload(active_id)
            import pandas as _pd
            if not isinstance(df, _pd.DataFrame):
                return [[] for _ in range(n)]
            opts = [{"label": c, "value": c} for c in df.columns]
            return [opts] * n
        except Exception:
            return [[] for _ in range(n)]

    # Pre-fill dropdown values when a table is focused
    @app.callback(
        *[Output(f"dp-mb-col-{mc}", "value") for group in _MOTL_COL_GROUPS for mc in group[1]],
        Input("dp-active-id",         "data"),
        Input(ids.DATA_POOL_REGISTRY, "data"),
        prevent_initial_call=False,
    )
    def _prefill_mb_cols(active_id, dp_reg):
        _all_motl_cols = [mc for g in _MOTL_COL_GROUPS for mc in g[1]]
        if not active_id or active_id not in (dp_reg or {}):
            return [None] * len(_all_motl_cols)
        try:
            df = datapool.get_payload(active_id)
            import pandas as _pd
            if not isinstance(df, _pd.DataFrame):
                return [None] * len(_all_motl_cols)
            return [mc if mc in df.columns else None for mc in _all_motl_cols]
        except Exception:
            return [None] * len(_all_motl_cols)

    # Build motl from table
    @app.callback(
        Output(ids.POOL_REGISTRY,  "data", allow_duplicate=True),
        Output(ids.POOL_META,      "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID,   "data", allow_duplicate=True),
        Output("dp-mb-status",     "children"),
        Input("dp-mb-build-btn",   "n_clicks"),
        State("dp-active-id",      "data"),
        *[State(f"dp-mb-col-{mc}", "value") for group in _MOTL_COL_GROUPS for mc in group[1]],
        State("dp-mb-label",           "value"),
        State(ids.POOL_REGISTRY,       "data"),
        State(ids.POOL_META,           "data"),
        State(ids.POOL_NEXT_ID,        "data"),
        State(ids.DATA_POOL_REGISTRY,  "data"),
        prevent_initial_call=True,
    )
    def _build_motl_from_table(_n, active_id, *args):
        import pandas as _pd
        import numpy as _np
        _all_motl_cols = [mc for g in _MOTL_COL_GROUPS for mc in g[1]]
        n_cols = len(_all_motl_cols)
        col_values = list(args[:n_cols])
        label_val   = args[n_cols]
        pool_reg    = args[n_cols + 1]
        pool_meta   = args[n_cols + 2]
        pool_next   = args[n_cols + 3]
        dp_reg      = args[n_cols + 4]

        _no = no_update
        _fail = (_no, _no, _no)

        if not _n:
            raise PreventUpdate
        if not active_id:
            return *_fail, "Select an active table from the slot list first."
        try:
            src_df = datapool.get_payload(active_id)
        except Exception as exc:
            return *_fail, f"Cannot load table: {exc}"
        if not isinstance(src_df, _pd.DataFrame):
            return *_fail, "Active entry is not a DataFrame."

        mapping = dict(zip(_all_motl_cols, col_values))
        # Validate required columns
        required = ["x", "y", "z"]
        missing_req = [r for r in required if not mapping.get(r)]
        if missing_req:
            return *_fail, f"Required column(s) not mapped: {', '.join(missing_req)}."

        # Build zeroed motl df with correct columns
        n = len(src_df)
        motl_df = _pd.DataFrame(
            _np.zeros((n, len(Motl.motl_columns))),
            columns=Motl.motl_columns,
        )

        errors = []
        for mc, src_col in mapping.items():
            if not src_col:
                continue  # leave default (0.0)
            if src_col not in src_df.columns:
                errors.append(f"'{mc}': column '{src_col}' not found in table.")
                continue
            try:
                motl_df[mc] = src_df[src_col].reset_index(drop=True).astype(float)
            except (ValueError, TypeError):
                errors.append(f"'{mc}': cannot convert '{src_col}' to float.")

        if errors:
            return *_fail, "Errors: " + "  ".join(errors)

        # Auto-generate subtomo_id if not mapped
        if not mapping.get("subtomo_id"):
            motl_df["subtomo_id"] = _np.arange(1, n + 1, dtype=float)

        pool_state = PoolState.from_stores(pool_reg, pool_meta, pool_next)
        entry_label = (dp_reg or {}).get(active_id, {}).get("label") or active_id
        lbl = (label_val or "").strip() or f"motl from {entry_label}"
        pool_state, new_motl_id = _pool_insert_motl(pool_state, motl_df, label=lbl)
        return (*pool_state.to_stores(), f"Created '{lbl}' with {n:,} particles ({new_motl_id}).")

