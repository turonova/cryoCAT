"""Editor sidebar — pool-driven.

Loads motls into the suite-global pool (``pool-*`` stores, declared in
:mod:`cryocat.app.suite.app`) and drives the editor's view slots.

Model
-----
* The **pool** is the source of truth for motl data; it is unbounded.
* The editor renders into a fixed set of ``N_SLOTS`` pre-wired *view slots*
  (the table/viewer/save surfaces, whose callbacks are registered once up
  front with literal ``me-{i}`` prefixes).
* ``me-slot-map`` (a length-``N_SLOTS`` list of ``motl_id`` or ``None``) maps
  each view slot to a pool entry. Loading a motl auto-assigns it to a free
  slot; when the pool exceeds the slots the user re-assigns via the
  "Slot assignment" dropdowns.
"""

import inspect
import os
import pandas as pd

import dash
from dash import html, dcc, Input, Output, State, no_update, ctx, ALL
import dash_bootstrap_components as dbc

from cryocat.core.cryomotl import Motl
from cryocat.app.components.motlio import motl_types, register_motl_load_callbacks
from cryocat.app.components.relionopts import get_relion_options
from cryocat.app.components.savedialog import (
    get_save_dialog,
    register_save_dialog_callbacks,
    build_batch_paths,
    execute_batch_save,
    validate_save,
)
from cryocat.app.components.pathfield import get_path_field
from cryocat.app.components.motlsource import (
    get_multi_motl_picker,
    register_multi_motl_picker_callbacks,
)
from cryocat.app.apputils import (
    generate_kwargs,
    run_operation,
    run_operation_to_pool,
    record_load_to_pool,
)
from cryocat.app import ids, styles
from cryocat.app import discovery as _discovery
from cryocat.app.formgen import build_form, make_dropdown
from cryocat.app.components.poolslotlist import (
    register_slot_change_callback,
    register_slot_focus_callback,
    _first_free_slot,
    _UNASSIGNED as _PSL_UNASSIGNED,
)
from cryocat.app.logger import invoke_operation, dash_logger as _dash_logger
from cryocat.app import session as _session
from cryocat.app.event import message_event
from cryocat.app.pageshell import _SIDEBAR_STYLE, _SIDEBAR_COL_STYLE
from cryocat.app.pool import (
    PoolState,
    remove_motl,
    get_rows,
    PoolPayloadMissing,
    GroupState,
    create_group,
    delete_group,
    reorder_group,
    remove_from_group,
    purge_motl_from_groups,
    set_has_tab,
    natural_sort_key,
    insert_motl,
    replace_motl_rows,
    save_snapshot,
    restore_snapshot,
)

# Number of editor *view slots* (rendered table/viewer surfaces). The motl pool
# itself is unbounded — this only caps how many motls are open as tabs at once.
N_SLOTS = 5

# Fetched at import time from GUI_REGISTRY — adding a new @gui_exposed method
# to Motl is sufficient; no edits needed here.
_MOTL_METHODS = [
    {"label": e.label, "value": e.fn.__name__}
    for e in _discovery.single_motl_ops()
    if e.fn.__module__ == "cryocat.core.cryomotl"
]
_MULTI_MOTL_METHODS = [
    {"label": e.label, "value": e.fn.__name__, "motls": e.motls} for e in _discovery.multi_motl_ops()
]
# Lookup of `method_name -> motls spec` for the run callback.
_MULTI_MOTL_SPECS = {m["value"]: m["motls"] for m in _MULTI_MOTL_METHODS}

# ── Column-merge helpers (used by layout and callbacks) ────────────────────────
_MOTL_COLS = Motl.motl_columns  # 20 column names


def _resolve_merge_motl_ids(method_name, main_id, second_ids, list_ids):
    spec = _MULTI_MOTL_SPECS.get(method_name) if method_name else None
    if spec is not None and spec["arity"] == "list":
        return list(list_ids) if list_ids else []
    seen, ids_out = set(), []
    for m in [main_id] + list(second_ids or []):
        if m and m not in seen:
            seen.add(m)
            ids_out.append(m)
    return ids_out


def _init_merge_draft(n_motls, saved_config):
    saved = saved_config or {}
    return {col: min(int(saved.get(col, 0)), max(n_motls - 1, 0)) for col in _MOTL_COLS}


def _build_col_merge_table(labels, draft):
    header = html.Tr(
        [html.Th("Motl", style={"minWidth": "110px", "fontSize": styles.FONT_TIGHT, "padding": "4px 6px"})]
        + [
            html.Th(
                col,
                style={
                    "writingMode": "vertical-rl",
                    "fontSize": styles.FONT_MED,
                    "textAlign": "center",
                    "minWidth": "44px",
                    "maxWidth": "44px",
                    "padding": "4px 2px",
                },
            )
            for col in _MOTL_COLS
        ],
    )
    body_rows = [
        html.Tr(
            [html.Td(html.Small(label, style={"whiteSpace": "nowrap"}), style={"padding": "3px 6px"})]
            + [
                html.Td(
                    dbc.Button(
                        "●" if draft.get(col, 0) == row_i else "○",
                        id={"type": "me-col-cell", "col": col, "row": row_i},
                        color="primary" if draft.get(col, 0) == row_i else "link",
                        size="sm",
                        style={
                            "padding": "0 3px",
                            "minWidth": "28px",
                            "fontSize": styles.FONT_MED,
                            "lineHeight": "1.2",
                        },
                        n_clicks=0,
                    ),
                    style={"textAlign": "center", "padding": "2px"},
                )
                for col in _MOTL_COLS
            ],
        )
        for row_i, label in enumerate(labels)
    ]
    return html.Div(
        html.Table(
            [html.Thead(header), html.Tbody(body_rows)],
            style={"borderCollapse": "collapse", "width": "100%"},
        ),
        style={"overflowX": "auto"},
    )


def _apply_col_merge(result_df, source_motls, col_config):
    if not col_config or all(int(v) == 0 for v in col_config.values()):
        return result_df
    df = result_df.copy()
    for col, src_idx in col_config.items():
        src_idx = int(src_idx)
        if src_idx == 0 or src_idx >= len(source_motls) or col not in df.columns:
            continue
        src_df = source_motls[src_idx].df
        if col not in src_df.columns:
            continue
        if "subtomo_id" in df.columns and "subtomo_id" in src_df.columns:
            src_series = src_df.set_index("subtomo_id")[col]
            mapped = df["subtomo_id"].map(src_series)
            df[col] = mapped.where(mapped.notna(), df[col])
        else:
            n = min(len(df), len(src_df))
            df.iloc[:n, df.columns.get_loc(col)] = src_df.iloc[:n, src_df.columns.get_loc(col)].values
    return df


def _run_pair_operations(
    fn, main_motl, sec_motls, kwargs, pool_state, op_label, src_labels, col_merge_config, has_col_config
):
    sig_params = list(inspect.signature(fn).parameters.keys())
    last_result, last_mid = None, None
    for i, sec_motl in enumerate(sec_motls):
        sec_label = src_labels[i + 1] if i + 1 < len(src_labels) else str(getattr(sec_motl, "_pool_motl_id", i))
        full_kwargs = {sig_params[0]: main_motl, sig_params[1]: sec_motl, **kwargs}
        pool_state, pair_mid, pair_result = run_operation_to_pool(
            fn,
            full_kwargs,
            pool_state,
            label=f"{op_label} of {src_labels[0]} + {sec_label}",
        )
        if not isinstance(pair_result, Motl):
            raise TypeError(f"'{fn.__name__}' did not return a Motl (got {type(pair_result).__name__}).")
        if has_col_config:
            merged_df = _apply_col_merge(pair_result.df, [main_motl, sec_motl], col_merge_config)
            if merged_df is not pair_result.df:
                pool_state = replace_motl_rows(pool_state, pair_mid, merged_df)
        last_result, last_mid = pair_result, pair_mid
    return pool_state, last_mid, last_result


# ───────────────────────────────────────────────────────────────────────────────



def get_motl_editor_sidebar():
    return dbc.Col(
        html.Div(
            [
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    dbc.RadioItems(
                                        id="me-load-mode",
                                        options=[
                                            {"label": "Single file", "value": "single"},
                                            {"label": "Multiple (glob)", "value": "multi"},
                                        ],
                                        value="single",
                                        inline=True,
                                        style={"display": "flex", "gap": "1.5rem"},
                                    ),
                                    style={"marginBottom": "0.5rem"},
                                ),
                                # Shared: type + Relion options — identical for single and glob
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.Div("Motl type: ", style={"fontStyle": "bold"}),
                                            width=4,
                                            className="d-flex align-items-center",
                                        ),
                                        dbc.Col(
                                            make_dropdown(
                                                "me-load-motl-dropdown",
                                                motl_types,
                                                "emmotl",
                                                style={"padding": "0"},
                                            ),
                                            width=8,
                                        ),
                                    ],
                                    style={"marginTop": "1rem", "marginBottom": "0.4rem"},
                                ),
                                get_relion_options("me-load", for_load=True),
                                # Single-file section
                                html.Div(
                                    id="me-load-single-section",
                                    children=[
                                        html.Div(
                                            get_path_field(
                                                "me-load-motl-path",
                                                mode="open",
                                                kind="motl",
                                                extensions=(".em", ".star", ".csv", ".tbl"),
                                                placeholder="Path to motl file",
                                            ),
                                            style={"marginTop": "0.5rem"},
                                        ),
                                        dbc.Button(
                                            "Load",
                                            id="me-load-motl-load-btn",
                                            color="primary",
                                            style={"width": "100%", "marginTop": "0.4rem"},
                                        ),
                                    ],
                                ),
                                # Multi/glob section
                                html.Div(
                                    id="me-load-multi-section",
                                    style={"display": "none"},
                                    children=[
                                        dbc.Input(
                                            id="me-mload-pattern",
                                            placeholder="e.g. /data/**/*.em or /data/runs/ (glob syntax supported)",
                                            size="sm",
                                            style={"marginBottom": "0.3rem", "marginTop": "0.5rem"},
                                        ),
                                        html.Div(
                                            id="me-mload-count",
                                            style={
                                                "color": styles.COLOR_MUTED,
                                                "marginBottom": "0.3rem",
                                                "fontSize": styles.FONT_SM,
                                            },
                                        ),
                                        dbc.Input(
                                            id="me-mload-group-name",
                                            placeholder="Group name (optional)",
                                            size="sm",
                                            style={"marginBottom": "0.4rem"},
                                        ),
                                        dbc.Button(
                                            "Load All",
                                            id="me-mload-btn",
                                            color="primary",
                                            size="sm",
                                            style={"width": "100%", "marginBottom": "0.3rem"},
                                        ),
                                        html.Div(
                                            id="me-mload-status",
                                            style={"color": styles.COLOR_MUTED, "fontSize": styles.FONT_SM},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="me-load-status",
                                    style={
                                        "marginTop": "0.5rem",
                                        "color": "var(--color9)",
                                        "wordBreak": "break-word",
                                    },
                                ),
                            ],
                            title="Load Motl",
                            item_id="me-sidebar-load",
                        ),
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    id="me-motl-list",
                                    children=html.Div(
                                        "No motls loaded.",
                                        style={"color": "var(--color9)", "padding": "4px"},
                                    ),
                                ),
                                html.Div(id="me-active-unslotted-note", style=styles.HINT),
                            ],
                            title="Loaded Motls (pool)",
                            item_id="me-sidebar-list",
                        ),
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    "Create a named group from pool motls. "
                                    "Groups appear collapsed in the list; "
                                    "order is preserved for multi-motl operations.",
                                    style={"color": "var(--color9)", "marginBottom": "0.5rem"},
                                ),
                                dbc.Input(
                                    id="me-create-group-name",
                                    placeholder="Group name (optional)",
                                    size="sm",
                                    style={"marginBottom": "0.4rem"},
                                ),
                                make_dropdown(
                                    "me-create-group-select",
                                    [],
                                    [],
                                    multi=True,
                                    placeholder="Select motls to group",
                                    style={"marginBottom": "0.4rem"},
                                ),
                                dbc.Button(
                                    "Create Group",
                                    id="me-create-group-btn",
                                    color="primary",
                                    size="sm",
                                    style={"width": "100%"},
                                ),
                                html.Div(
                                    id="me-create-group-status",
                                    style={"color": "var(--color9)", "marginTop": "0.4rem"},
                                ),
                            ],
                            title="Groups",
                            item_id="me-sidebar-groups",
                        ),
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    "Convert all members of the active group to a single format. "
                                    "Click a group label in the pool list to select the target group.",
                                    style={"color": "var(--color9)", "marginBottom": "0.5rem"},
                                ),
                                html.Div(
                                    id="me-batch-convert-target",
                                    style={"color": "var(--color9)", "marginBottom": "0.4rem"},
                                ),
                                get_save_dialog("me-batch-save", mode="batch"),
                            ],
                            title="Batch Convert",
                            item_id="me-sidebar-batch-convert",
                        ),
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    id="me-single-op-target-label",
                                    style={
                                        "color": styles.COLOR_MUTED,
                                        "marginBottom": "0.4rem",
                                        "fontSize": styles.FONT_SM,
                                    },
                                ),
                                make_dropdown(
                                    "me-op-func-select",
                                    _MOTL_METHODS,
                                    None,
                                    placeholder="Select operation",
                                    style={"marginBottom": "0.5rem"},
                                ),
                                html.Div(id="me-op-func-form", style={"marginBottom": "0.5rem"}),
                                html.Div(
                                    id="me-op-group-options",
                                    style={"display": "none"},
                                    children=[
                                        dbc.Checklist(
                                            id="me-op-create-group",
                                            options=[{"label": "Create a group", "value": "create"}],
                                            value=["create"],
                                            style={"marginBottom": "0.3rem"},
                                        ),
                                        dbc.Checklist(
                                            id="me-op-save-to-disk",
                                            options=[{"label": "Save to disk", "value": "save"}],
                                            value=[],
                                            style={"marginBottom": "0.3rem"},
                                        ),
                                        html.Div(
                                            id="me-op-save-options",
                                            style={"display": "none"},
                                            children=[
                                                html.Div(
                                                    get_path_field(
                                                        "me-op-save-dir",
                                                        mode="directory",
                                                        kind="output",
                                                        placeholder="Output directory",
                                                    ),
                                                    style={"marginBottom": "0.3rem"},
                                                ),
                                                make_dropdown(
                                                    "me-op-save-format",
                                                    motl_types,
                                                    "emmotl",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="me-op-group-validation",
                                            style={
                                                "color": "var(--color9)",
                                                "marginTop": "0.2rem",
                                            },
                                        ),
                                    ],
                                ),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Apply",
                                            id="me-op-apply-btn",
                                            color="primary",
                                            size="sm",
                                            className="me-1",
                                            style={"width": "48%"},
                                        ),
                                        dbc.Button(
                                            "Undo",
                                            id="me-op-undo-btn",
                                            color="secondary",
                                            size="sm",
                                            style={"width": "48%"},
                                        ),
                                    ],
                                    style={**{"display": "flex"}, "marginBottom": "0.5rem"},
                                ),
                                html.Div(
                                    id="me-op-status",
                                    style={"color": "var(--color9)", "wordBreak": "break-word"},
                                ),
                            ],
                            title="Single Motl Operations",
                            item_id="me-sidebar-singlemotl",
                        ),
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    id="me-multi-op-target-label",
                                    style={
                                        "color": styles.COLOR_MUTED,
                                        "marginBottom": "0.4rem",
                                        "fontSize": styles.FONT_SM,
                                    },
                                ),
                                make_dropdown(
                                    "me-multi-op-select",
                                    _MULTI_MOTL_METHODS,
                                    None,
                                    placeholder="Select operation",
                                    style={"marginBottom": "0.5rem"},
                                ),
                                get_multi_motl_picker("me-multi"),
                                html.Div(id="me-multi-form", style={"marginBottom": "0.5rem"}),
                                dbc.Button(
                                    "Column assignment",
                                    id="me-col-merge-open-btn",
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                    style={"width": "100%", "marginBottom": "0.3rem"},
                                ),
                                dbc.Button(
                                    "Run",
                                    id="me-multi-run-btn",
                                    color="primary",
                                    size="sm",
                                    style={"width": "100%"},
                                ),
                                html.Div(
                                    id="me-multi-op-status",
                                    style={
                                        "marginTop": "0.5rem",
                                        "color": "var(--color9)",
                                        "wordBreak": "break-word",
                                    },
                                ),
                            ],
                            title="Multiple motl operations",
                            item_id="me-sidebar-multimotl",
                        ),
                    ],
                    always_open=True,
                    active_item=["me-sidebar-load", "me-sidebar-list"],
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Column assignment")),
                        dbc.ModalBody(
                            html.Div(id="me-col-merge-body"),
                            style={"overflowY": "auto"},
                        ),
                        dbc.ModalFooter(
                            [
                                dbc.Button("Cancel", id="me-col-merge-cancel", color="secondary", className="me-2"),
                                dbc.Button("Confirm", id="me-col-merge-confirm", color="primary"),
                            ]
                        ),
                    ],
                    id="me-col-merge-modal",
                    is_open=False,
                    size="xl",
                    scrollable=True,
                ),
            ],
            className="sidebar",
            style=_SIDEBAR_STYLE,
        ),
        id="me-sidebar",
        width=3,
        style=_SIDEBAR_COL_STYLE,
    )




def _relion_params_summary(relion_params):
    """Human-readable one-liner appended to the load status."""
    if not relion_params:
        return ""

    def _scalar(v):
        if v is None:
            return None
        try:
            if hasattr(v, "__len__"):
                v = v[0] if len(v) > 0 else None
            return float(v)
        except (TypeError, ValueError, IndexError):
            return None

    parts = []
    ps = _scalar(relion_params.get("pixel_size"))
    bn = _scalar(relion_params.get("binning"))
    if ps is not None:
        parts.append(f"pixel size: {ps:.4g} Å")
    if bn is not None:
        parts.append(f"binning: {bn:.4g}")
    if relion_params.get("tomo_format"):
        parts.append(f"tomo fmt: {relion_params['tomo_format']}")
    if relion_params.get("subtomo_format"):
        parts.append(f"subtomo fmt: {relion_params['subtomo_format']}")
    return ("  |  " + ",  ".join(parts)) if parts else ""


def register_motl_editor_sidebar_callbacks(app):

    register_motl_load_callbacks(app, "me-load")
    register_save_dialog_callbacks(app, "me-batch-save", mode="batch")

    # ── Load → pool ────────────────────────────────────────────────────────────
    # A freshly loaded motl is appended to the pool with a new motl_id and
    # auto-assigned to the first free view slot (if any).
    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output("me-slot-map", "data", allow_duplicate=True),
        Output("me-tabs", "active_tab", allow_duplicate=True),
        Output("me-load-status", "children", allow_duplicate=True),
        Input("me-load-motl-data-store", "data"),
        State("me-load-motl-extra-data-store", "data"),
        State("me-load-motl-data-type", "data"),
        State("me-load-relion-optics-store", "data"),
        State("me-load-rln-tomos-store", "data"),
        State("me-load-rln-tomos-filename", "data"),
        State({"type": "path-input", "owner": "me-load-motl-path"}, "value"),
        State("me-load-relion-params-store", "data"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State("me-slot-map", "data"),
        prevent_initial_call=True,
    )
    def route_motl(
        motl_data,
        extra,
        dtype,
        optics,
        r5t,
        r5tn,
        motl_path,
        relion_params,
        registry,
        pool_meta,
        next_id,
        slot_map,
    ):
        import os as _os
        from cryocat.app.components.filesystem import resolve_input as _resolve

        if not motl_data:
            raise dash.exceptions.PreventUpdate

        slot_map = list(slot_map or [None] * N_SLOTS)
        while len(slot_map) < N_SLOTS:
            slot_map.append(None)
        # Drop stale slot references from a previous session if the pool restarted.
        _live_ids = set((registry or {}).keys())
        slot_map = [m if (m and m in _live_ids) else None for m in slot_map]

        _nid = next_id or 0
        # Use the basename of the resolved path as the label; fall back to a counter.
        resolved_path, _err = _resolve(motl_path or "")
        label = _os.path.basename(resolved_path) if resolved_path and not _err else f"Motl {_nid + 1}"
        effective_type = dtype or "emmotl"

        # Build rln_kwargs for script rendering (pixel_size/binning/version/formats).
        rln_kwargs: dict = {}
        if effective_type in ("relion", "relion5", "relion5_1") and relion_params:
            if effective_type == "relion":
                ver_str = relion_params.get("version", "")
                try:
                    rln_kwargs["version"] = float(str(ver_str).split()[-1])
                except (ValueError, IndexError):
                    pass
            ps = relion_params.get("pixel_size")
            if ps:
                try:
                    rln_kwargs["pixel_size"] = float(ps)
                except (TypeError, ValueError):
                    pass  # per-particle (multi-optics): not representable as a single kwarg
            bn = relion_params.get("binning")
            if bn:
                rln_kwargs["binning"] = float(bn)
            tf = relion_params.get("tomo_format")
            if tf:
                rln_kwargs["tomo_format"] = tf
            sf = relion_params.get("subtomo_format")
            if sf:
                rln_kwargs["subtomo_format"] = sf

        current_pool = PoolState.from_stores(registry, pool_meta, next_id)
        try:
            pool_state, mid, _ = record_load_to_pool(
                motl_data,
                effective_type,
                resolved_path or label,
                rln_kwargs,
                current_pool,
                label=label,
                extra=extra,
                meta={
                    "data_type": dtype,
                    "relion_optics": optics,
                    "relion5_tomos": r5t,
                    "relion5_tomos_filename": r5tn,
                    "relion_params": relion_params,
                },
            )
        except Exception as exc:
            from cryocat.app import session as _session
            from cryocat.app.event import message_event as _msg_event
            _dash_logger.write(f"Load failed: {exc}", source="error")
            _session.emit(_msg_event(f"Load failed: {exc}", level="error"))
            return (no_update, no_update, no_update, slot_map, no_update, f"Load failed: {exc}")

        free = _first_free_slot(slot_map, N_SLOTS)
        if free is not None:
            slot_map[free] = mid
            active_tab = f"me-tab-{free}"
            status = f"Loaded: {label} ({len(motl_data)} particles) → slot {free + 1}"
        else:
            active_tab = no_update
            status = (
                f"Loaded: {label} ({len(motl_data)} particles) → pool "
                f"(all {N_SLOTS} slots in use; use the slot dropdown in the pool list)"
            )
        status += _relion_params_summary(relion_params)
        return (*pool_state.to_stores(), slot_map, active_tab, status)

    # ── Pool motl list ─────────────────────────────────────────────────────────
    @app.callback(
        Output("me-motl-list", "children"),
        Input(ids.POOL_REGISTRY, "data"),
        Input(ids.POOL_GROUPS, "data"),
        Input("me-slot-map", "data"),
        Input("me-group-expand", "data"),
        Input("me-active-target", "data"),
        prevent_initial_call=True,
    )
    def update_motl_list(registry, groups_data, slot_map_data, expand_data, active_target):
        registry = registry or {}
        gstate = GroupState.from_store(groups_data)
        expand_data = expand_data or {}
        active_target = active_target or {}
        at_type = active_target.get("type")
        at_id = active_target.get("id")
        slot_map = list(slot_map_data or [None] * N_SLOTS)
        mid_to_slot = {m: i for i, m in enumerate(slot_map) if m}
        items = []

        # ── Groups first ──────────────────────────────────────────────────────
        for gid, g in gstate.groups.items():
            members = list(g.get("members", []))
            glabel = g.get("label", gid)
            expanded = expand_data.get(gid, False)
            toggle_icon = "▾" if expanded else "▶"
            is_active_group = at_type == "group" and at_id == gid
            group_row_style = {"display": "flex", "alignItems": "center", "padding": "4px 8px"}
            if is_active_group:
                group_row_style["backgroundColor"] = "var(--bs-primary-bg-subtle)"
            items.append(
                dbc.ListGroupItem(
                    [
                        dbc.Button(
                            toggle_icon,
                            id={"type": "me-group-toggle", "gid": gid},
                            color="link",
                            size="sm",
                            style={"padding": "0 4px", "fontFamily": "monospace", "color": "inherit"},
                        ),
                        html.Span(
                            f"{glabel} ({len(members)})",
                            id={"type": "me-group-label-click", "gid": gid},
                            n_clicks=0,
                            style={
                                "flex": "1",
                                "fontWeight": "500",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                                "whiteSpace": "nowrap",
                                "cursor": "pointer",
                            },
                        ),
                        dbc.Button(
                            "×",
                            id={"type": "me-group-delete", "gid": gid},
                            color="link",
                            size="sm",
                            style={"padding": "0 6px", "color": "var(--color9)"},
                        ),
                    ],
                    style=group_row_style,
                )
            )
            if expanded:
                for idx, mid in enumerate(members):
                    mmeta = registry.get(mid) or {}
                    mlabel = mmeta.get("label", mid)
                    is_first = idx == 0
                    is_last = idx == len(members) - 1
                    items.append(
                        dbc.ListGroupItem(
                            [
                                html.Span(
                                    f"{mlabel} ({mid.replace('-', '_')})",
                                    style={
                                        "flex": "1",
                                        "paddingLeft": "1.5rem",
                                        "overflow": "hidden",
                                        "textOverflow": "ellipsis",
                                        "whiteSpace": "nowrap",
                                        "color": "var(--color9)" if not mmeta else "inherit",
                                    },
                                ),
                                dbc.Button(
                                    "↑",
                                    id={"type": "me-member-up", "gid": gid, "mid": mid},
                                    color="link",
                                    size="sm",
                                    disabled=is_first,
                                    style={"padding": "0 3px", "color": "inherit"},
                                ),
                                dbc.Button(
                                    "↓",
                                    id={"type": "me-member-down", "gid": gid, "mid": mid},
                                    color="link",
                                    size="sm",
                                    disabled=is_last,
                                    style={"padding": "0 3px", "color": "inherit"},
                                ),
                                dbc.Button(
                                    "Open",
                                    id={"type": "me-member-open", "gid": gid, "mid": mid},
                                    color="link",
                                    size="sm",
                                    style={"padding": "0 4px", "color": "inherit"},
                                ),
                                dbc.Button(
                                    "−",
                                    id={"type": "me-member-remove", "gid": gid, "mid": mid},
                                    color="link",
                                    size="sm",
                                    style={"padding": "0 4px", "color": "var(--color9)"},
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center", "padding": "2px 8px"},
                        )
                    )

        # ── Ungrouped motls (R2: grouped motls never appear here) ────────────
        all_grouped = {mid for g in gstate.groups.values() for mid in g.get("members", [])}
        for mid, meta in registry.items():
            if not meta.get("active", True):
                continue
            if mid in all_grouped:
                continue  # R2: grouped motls appear only under their group
            label = meta.get("label", mid)
            is_active_motl = at_type == "motl" and at_id == mid
            motl_row_style = {"display": "flex", "alignItems": "center", "padding": "4px 8px", "cursor": "pointer"}
            if is_active_motl:
                motl_row_style["backgroundColor"] = "var(--bs-primary-bg-subtle)"
            # Build per-motl slot dropdown
            current_slot_str = str(mid_to_slot[mid]) if mid in mid_to_slot else _PSL_UNASSIGNED
            slot_opts = [{"label": "—", "value": _PSL_UNASSIGNED}]
            for slot_i in range(N_SLOTS):
                occupant = slot_map[slot_i] if slot_i < len(slot_map) else None
                if occupant and occupant != mid:
                    slot_opts.append({"label": f"Slot {slot_i + 1} (taken)", "value": str(slot_i)})
                else:
                    slot_opts.append({"label": f"Slot {slot_i + 1}", "value": str(slot_i)})
            items.append(
                dbc.ListGroupItem(
                    [
                        html.Span(
                            f"{label} (id: {mid.replace('-', '_')})",
                            style={
                                "flex": "1",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                                "whiteSpace": "nowrap",
                            },
                        ),
                        html.Div(
                            make_dropdown(
                                {"type": "me-psl-slot", "item_id": mid},
                                slot_opts,
                                current_slot_str,
                                clearable=False,
                            ),
                            style={"width": "8rem", "flexShrink": 0},
                        ),
                        dbc.Button(
                            "×",
                            id={"type": "me-close-motl", "mid": mid},
                            color="link",
                            size="sm",
                            style={"padding": "0 6px", "color": "var(--color9)", "lineHeight": "1"},
                        ),
                    ],
                    id={"type": "me-motl-list-item", "mid": mid},
                    action=True,
                    n_clicks=0,
                    style=motl_row_style,
                )
            )

        if not items:
            return html.Div("No motls loaded.", style={"color": "var(--color9)", "padding": "4px"})
        return dbc.ListGroup(items, flush=True)

    # ── Per-motl slot dropdowns: generic slot-change callback (E5) ────────────
    register_slot_change_callback(app, "me", "me-slot-map", N_SLOTS)
    register_slot_focus_callback(app, "me-slot-map", "me-tabs", "me-tab-", N_SLOTS)

    # ── H3: tab switch → sync active target (motl dict format, not raw string) ─

    @app.callback(
        Output("me-active-target", "data", allow_duplicate=True),
        Input("me-tabs", "active_tab"),
        State("me-slot-map", "data"),
        State("me-active-target", "data"),
        prevent_initial_call=True,
    )
    def _sync_tab_to_active_target(active_tab, slot_map, current_target):
        if not active_tab or not active_tab.startswith("me-tab-"):
            return no_update
        try:
            idx = int(active_tab[len("me-tab-"):])
        except ValueError:
            return no_update
        sm = list(slot_map or [None] * N_SLOTS)
        mid = sm[idx] if idx < len(sm) else None
        if not mid:
            return None
        new_target = {"type": "motl", "id": mid}
        if current_target == new_target:
            return no_update
        return new_target

    # ── H3: unslotted note — visible when the active motl has no slot assigned ─

    @app.callback(
        Output("me-active-unslotted-note", "children"),
        Input("me-active-target", "data"),
        State("me-slot-map", "data"),
    )
    def _update_active_unslotted_note(active_target, slot_map):
        if not active_target or active_target.get("type") != "motl":
            return ""
        mid = active_target.get("id")
        sm = list(slot_map or [None] * N_SLOTS)
        if any(m == mid for m in sm):
            return ""
        return "[not displayed — assign to a slot to see edits live]"

    # ── Clicking a pool motl activates its slot tab and sets the active target ──
    @app.callback(
        Output("me-tabs", "active_tab", allow_duplicate=True),
        Output("me-active-target", "data", allow_duplicate=True),
        Input({"type": "me-motl-list-item", "mid": ALL}, "n_clicks"),
        State("me-slot-map", "data"),
        prevent_initial_call=True,
    )
    def activate_tab_from_list(n_clicks_list, slot_map):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "mid" in triggered):
            raise dash.exceptions.PreventUpdate
        mid = triggered["mid"]
        active_target = {"type": "motl", "id": mid}
        slot_map = slot_map or [None] * N_SLOTS
        for i, m in enumerate(slot_map):
            if m == mid:
                return f"me-tab-{i}", active_target
        return no_update, active_target  # not in any slot but still becomes active target

    # ── Close a pool motl: drop it from the pool, free its slot, purge from groups
    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output(ids.POOL_GROUPS, "data", allow_duplicate=True),
        Output("me-slot-map", "data", allow_duplicate=True),
        Input({"type": "me-close-motl", "mid": ALL}, "n_clicks"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State(ids.POOL_GROUPS, "data"),
        State("me-slot-map", "data"),
        prevent_initial_call=True,
    )
    def close_motl(n_clicks_list, registry, pool_meta, next_id, groups_data, slot_map):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "mid" in triggered):
            raise dash.exceptions.PreventUpdate

        mid = triggered["mid"]
        pool_state = remove_motl(PoolState.from_stores(registry, pool_meta, next_id), mid)
        gstate = purge_motl_from_groups(GroupState.from_store(groups_data), mid)

        old_map = list(slot_map or [None] * N_SLOTS)
        new_map = [None if m == mid else m for m in old_map]
        return (*pool_state.to_stores(), gstate.to_store(), new_map)

    # ── Group: toggle expand/collapse ─────────────────────────────────────────
    @app.callback(
        Output("me-group-expand", "data", allow_duplicate=True),
        Input({"type": "me-group-toggle", "gid": ALL}, "n_clicks"),
        State("me-group-expand", "data"),
        prevent_initial_call=True,
    )
    def toggle_group(n_clicks_list, expand_data):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "gid" in triggered):
            raise dash.exceptions.PreventUpdate
        gid = triggered["gid"]
        expand_data = dict(expand_data or {})
        expand_data[gid] = not expand_data.get(gid, False)
        return expand_data

    # ── Group: delete group (motls remain in pool) ─────────────────────────────
    @app.callback(
        Output(ids.POOL_GROUPS, "data", allow_duplicate=True),
        Output("me-group-expand", "data", allow_duplicate=True),
        Input({"type": "me-group-delete", "gid": ALL}, "n_clicks"),
        State(ids.POOL_GROUPS, "data"),
        State("me-group-expand", "data"),
        prevent_initial_call=True,
    )
    def delete_group_cb(n_clicks_list, groups_data, expand_data):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "gid" in triggered):
            raise dash.exceptions.PreventUpdate
        gid = triggered["gid"]
        gstate = delete_group(GroupState.from_store(groups_data), gid)
        expand_data = {k: v for k, v in (expand_data or {}).items() if k != gid}
        return gstate.to_store(), expand_data

    # ── Group member: move up in order ─────────────────────────────────────────
    @app.callback(
        Output(ids.POOL_GROUPS, "data", allow_duplicate=True),
        Input({"type": "me-member-up", "gid": ALL, "mid": ALL}, "n_clicks"),
        State(ids.POOL_GROUPS, "data"),
        prevent_initial_call=True,
    )
    def move_member_up(n_clicks_list, groups_data):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "gid" in triggered):
            raise dash.exceptions.PreventUpdate
        gid, mid = triggered["gid"], triggered["mid"]
        gstate = GroupState.from_store(groups_data)
        members = list((gstate.groups.get(gid) or {}).get("members", []))
        idx = next((i for i, m in enumerate(members) if m == mid), None)
        if idx is None or idx == 0:
            raise dash.exceptions.PreventUpdate
        members[idx - 1], members[idx] = members[idx], members[idx - 1]
        return reorder_group(gstate, gid, members).to_store()

    # ── Group member: move down in order ───────────────────────────────────────
    @app.callback(
        Output(ids.POOL_GROUPS, "data", allow_duplicate=True),
        Input({"type": "me-member-down", "gid": ALL, "mid": ALL}, "n_clicks"),
        State(ids.POOL_GROUPS, "data"),
        prevent_initial_call=True,
    )
    def move_member_down(n_clicks_list, groups_data):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "gid" in triggered):
            raise dash.exceptions.PreventUpdate
        gid, mid = triggered["gid"], triggered["mid"]
        gstate = GroupState.from_store(groups_data)
        members = list((gstate.groups.get(gid) or {}).get("members", []))
        idx = next((i for i, m in enumerate(members) if m == mid), None)
        if idx is None or idx == len(members) - 1:
            raise dash.exceptions.PreventUpdate
        members[idx], members[idx + 1] = members[idx + 1], members[idx]
        return reorder_group(gstate, gid, members).to_store()

    # ── Group member: remove from group (motl stays in pool) ───────────────────
    @app.callback(
        Output(ids.POOL_GROUPS, "data", allow_duplicate=True),
        Input({"type": "me-member-remove", "gid": ALL, "mid": ALL}, "n_clicks"),
        State(ids.POOL_GROUPS, "data"),
        prevent_initial_call=True,
    )
    def remove_member_cb(n_clicks_list, groups_data):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "gid" in triggered):
            raise dash.exceptions.PreventUpdate
        gid, mid = triggered["gid"], triggered["mid"]
        return remove_from_group(GroupState.from_store(groups_data), gid, mid).to_store()

    # ── Group member: open in editor (sets has_tab=True, assigns to free slot) ──
    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output("me-slot-map", "data", allow_duplicate=True),
        Output("me-tabs", "active_tab", allow_duplicate=True),
        Input({"type": "me-member-open", "gid": ALL, "mid": ALL}, "n_clicks"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State("me-slot-map", "data"),
        prevent_initial_call=True,
    )
    def open_member_in_editor(n_clicks_list, registry, pool_meta, next_id, slot_map):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "mid" in triggered):
            raise dash.exceptions.PreventUpdate
        mid = triggered["mid"]
        pool_state = set_has_tab(PoolState.from_stores(registry, pool_meta, next_id), mid, True)
        slot_map = list(slot_map or [None] * N_SLOTS)
        while len(slot_map) < N_SLOTS:
            slot_map.append(None)
        free = _first_free_slot(slot_map, N_SLOTS)
        if free is not None:
            slot_map[free] = mid
            active = f"me-tab-{free}"
        else:
            active = no_update
        return (*pool_state.to_stores(), slot_map, active)

    # ── Group: create from selected pool motls ─────────────────────────────────
    @app.callback(
        Output(ids.POOL_GROUPS, "data", allow_duplicate=True),
        Output("me-create-group-status", "children"),
        Input("me-create-group-btn", "n_clicks"),
        State("me-create-group-select", "value"),
        State("me-create-group-name", "value"),
        State(ids.POOL_GROUPS, "data"),
        prevent_initial_call=True,
    )
    def create_group_cb(n_clicks, selected_motls, group_name, groups_data):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not selected_motls:
            return no_update, "Select at least one motl to form a group."
        gstate, gid = create_group(
            GroupState.from_store(groups_data),
            selected_motls,
            label=group_name.strip() if group_name and group_name.strip() else None,
        )
        g = gstate.groups[gid]
        return gstate.to_store(), f"Group '{g['label']}' created with {len(g['members'])} motl(s)."

    # ── Group: populate create-group dropdown from pool ────────────────────────
    @app.callback(
        Output("me-create-group-select", "options"),
        Input(ids.POOL_REGISTRY, "data"),
    )
    def populate_create_group_select(registry):
        registry = registry or {}
        return [
            {"label": f"{m.get('label', mid)} ({mid.replace('-', '_')})", "value": mid}
            for mid, m in registry.items()
            if m.get("active", True)
        ]

    # ── Multiple-motl operations: collector-driven (pair / list) ───────────────
    register_multi_motl_picker_callbacks(app, "me-multi")

    @app.callback(
        Output("me-multi-pair-picker", "style"),
        Output("me-multi-list-picker", "style"),
        Output("me-multi-list-label", "children"),
        Input("me-multi-op-select", "value"),
    )
    def _toggle_multi_picker(method_name):
        if not method_name:
            return {"display": "none"}, {"display": "none"}, "Motls"
        spec = _MULTI_MOTL_SPECS.get(method_name) or {}
        arity = spec.get("arity")
        if arity == "pair":
            return {"display": "block"}, {"display": "none"}, "Motls"
        if arity == "list":
            label = (
                "Motls (first = kept on duplicates)" if spec.get("main_first") else "Motls to merge (order preserved)"
            )
            return {"display": "none"}, {"display": "block"}, label
        return {"display": "none"}, {"display": "none"}, "Motls"

    @app.callback(
        Output("me-multi-form", "children"),
        Input("me-multi-op-select", "value"),
    )
    def _build_multi_form(method_name):
        # Scalar params only — the motl params (motl1/motl2 for pair ops,
        # motls.param for list ops) are supplied by the picker, not the form.
        if not method_name:
            return []
        spec = _MULTI_MOTL_SPECS.get(method_name) or {}
        if spec.get("arity") == "pair":
            exclude = ("motl1", "motl2")
        elif spec.get("arity") == "list":
            exclude = (spec.get("param", "motl_list"),)
        else:
            exclude = ()
        fn = getattr(Motl, method_name)
        return build_form(fn, id_type="me-multi-param", exclude=exclude)

    # ── Column-assignment modal callbacks ─────────────────────────────────────
    @app.callback(
        Output("me-col-merge-modal", "is_open"),
        Output("me-col-merge-body", "children"),
        Output("me-col-merge-draft", "data"),
        Output("me-col-merge-motls", "data"),
        Input("me-col-merge-open-btn", "n_clicks"),
        State("me-multi-op-select", "value"),
        State("me-multi-main-select", "value"),
        State("me-multi-second-select", "value"),
        State("me-multi-list-select", "value"),
        State(ids.POOL_REGISTRY, "data"),
        State("me-col-merge-config", "data"),
        prevent_initial_call=True,
    )
    def _open_col_merge(n_clicks, method_name, main_id, second_ids, list_ids, registry, saved_config):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        motl_ids = _resolve_merge_motl_ids(method_name, main_id, second_ids, list_ids)
        if not motl_ids:
            raise dash.exceptions.PreventUpdate
        registry = registry or {}
        labels = [(registry.get(mid) or {}).get("label", mid) for mid in motl_ids]
        draft = _init_merge_draft(len(motl_ids), saved_config)
        return True, _build_col_merge_table(labels, draft), draft, motl_ids

    @app.callback(
        Output("me-col-merge-body", "children", allow_duplicate=True),
        Output("me-col-merge-draft", "data", allow_duplicate=True),
        Input({"type": "me-col-cell", "col": ALL, "row": ALL}, "n_clicks"),
        State("me-col-merge-draft", "data"),
        State("me-col-merge-motls", "data"),
        State(ids.POOL_REGISTRY, "data"),
        prevent_initial_call=True,
    )
    def _on_col_cell_click(n_clicks_list, draft, motl_ids, registry):
        trigger = ctx.triggered_id
        if not trigger or not isinstance(trigger, dict) or not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        new_draft = dict(draft or {})
        new_draft[trigger["col"]] = trigger["row"]
        registry = registry or {}
        labels = [(registry.get(mid) or {}).get("label", mid) for mid in (motl_ids or [])]
        return _build_col_merge_table(labels, new_draft), new_draft

    @app.callback(
        Output("me-col-merge-config", "data"),
        Output("me-col-merge-modal", "is_open", allow_duplicate=True),
        Input("me-col-merge-confirm", "n_clicks"),
        State("me-col-merge-draft", "data"),
        prevent_initial_call=True,
    )
    def _confirm_col_merge(n_clicks, draft):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        return draft, False

    @app.callback(
        Output("me-col-merge-modal", "is_open", allow_duplicate=True),
        Input("me-col-merge-cancel", "n_clicks"),
        prevent_initial_call=True,
    )
    def _cancel_col_merge(n_clicks):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        return False

    # ── Multi-motl run ─────────────────────────────────────────────────────────
    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output("me-slot-map", "data", allow_duplicate=True),
        Output("me-tabs", "active_tab", allow_duplicate=True),
        Output("me-multi-op-status", "children", allow_duplicate=True),
        Input("me-multi-run-btn", "n_clicks"),
        State("me-multi-op-select", "value"),
        State("me-multi-main-select", "value"),
        State("me-multi-second-select", "value"),
        State("me-multi-list-select", "value"),
        State({"type": "me-multi-param", "owner": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "me-multi-param", "owner": ALL, "param": ALL, "tag": ALL}, "id"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State("me-slot-map", "data"),
        State("me-col-merge-config", "data"),
        prevent_initial_call=True,
    )
    def run_multi_op(
        n_clicks,
        method_name,
        main_id,
        second_ids,
        list_ids,
        param_values,
        param_ids,
        registry,
        pool_meta,
        next_id,
        slot_map,
        col_merge_config,
    ):
        pool_noup = (no_update,) * 5

        def _err(msg):
            return (*pool_noup, msg)

        has_col_config = bool(col_merge_config) and any(int(v) != 0 for v in col_merge_config.values())
        if not n_clicks or (not method_name and not has_col_config):
            raise dash.exceptions.PreventUpdate

        spec = _MULTI_MOTL_SPECS.get(method_name) if method_name else None
        if method_name and spec is None:
            return _err(f"Operation '{method_name}' is not registered as multi-motl.")

        # 1) Resolve selected motl_ids -> Motl instances, preserving order.
        try:
            if spec is not None and spec["arity"] == "list":
                if not list_ids or len(list_ids) < 2:
                    return _err("Select at least two motls for this operation.")
                ordered_ids = list(list_ids)
            else:  # pair op or column-merge-only
                if not main_id:
                    return _err("Select the Main motl.")
                valid_seconds = [s for s in (second_ids or []) if s != main_id]
                if spec is not None and not valid_seconds:
                    return _err("Select at least one Second motl (must differ from Main).")
                ordered_ids = _resolve_merge_motl_ids(method_name, main_id, second_ids, list_ids)
                if not ordered_ids:
                    return _err("Select at least one motl.")

            motls = []
            for mid in ordered_ids:
                try:
                    rows_df = get_rows(mid)
                except PoolPayloadMissing as exc:
                    return _err(str(exc))
                motl_obj = Motl(rows_df)
                motl_obj._pool_motl_id = mid
                motls.append(motl_obj)
        except Exception as exc:
            _session.emit(message_event(f"Error preparing motls: {exc}", level="error"))
            return _err(f"Error preparing motls: {exc}")

        # 2) Scalar kwargs from the auto-form.
        current_pool = PoolState.from_stores(registry, pool_meta, next_id)
        kwargs = generate_kwargs(param_ids, param_values, pool_state=current_pool) if param_ids else {}

        # 3) Pre-compute labels.
        gui = getattr(getattr(Motl, method_name).__func__, "_gui", {}) if method_name else {}
        op_label = gui.get("label", method_name) if method_name else "col-merge"
        src_labels = [(registry.get(oid) or {}).get("label", oid) for oid in ordered_ids]
        slot_map = list(slot_map or [None] * N_SLOTS)
        while len(slot_map) < N_SLOTS:
            slot_map.append(None)

        # 4) Run operation, or use main motl as base for column-merge-only.
        if method_name:
            fn = getattr(Motl, method_name)
            if spec["arity"] == "pair":
                try:
                    pool_state, mid, result = _run_pair_operations(
                        fn,
                        motls[0],
                        motls[1:],
                        kwargs,
                        current_pool,
                        op_label,
                        src_labels,
                        col_merge_config,
                        has_col_config,
                    )
                except Exception as exc:
                    return _err(f"Error running '{method_name}': {exc}")
            else:
                try:
                    list_param = spec.get("param", "motl_list")
                    full_kwargs = {list_param: motls, **kwargs}
                    pool_state, mid, result = run_operation_to_pool(
                        fn,
                        full_kwargs,
                        current_pool,
                        label=f"{op_label} of {' + '.join(src_labels)}",
                    )
                except Exception as exc:
                    return _err(f"Error running '{method_name}': {exc}")
                if not isinstance(result, Motl):
                    return _err(f"'{method_name}' did not return a Motl (got {type(result).__name__}).")
                if has_col_config:
                    merged_df = _apply_col_merge(result.df, motls, col_merge_config)
                    if merged_df is not result.df:
                        pool_state = replace_motl_rows(pool_state, mid, merged_df)
        else:
            result = motls[0]
            pool_state, mid = insert_motl(
                current_pool,
                motls[0].df,
                label=f"col-merge of {' + '.join(src_labels)}",
                has_tab=False,
            )
            if has_col_config:
                merged_df = _apply_col_merge(result.df, motls, col_merge_config)
                if merged_df is not result.df:
                    pool_state = replace_motl_rows(pool_state, mid, merged_df)

        # 6) Assign to the first free view slot.
        col_note = " + col-assign" if has_col_config else ""
        free = _first_free_slot(slot_map, N_SLOTS)
        if free is not None:
            slot_map[free] = mid
            active = f"me-tab-{free}"
            status = (
                f"'{op_label}{col_note}' -> new motl in slot {free + 1} "
                f"({len(result.df)} particles, from {len(motls)} motl(s))."
            )
        else:
            active = no_update
            status = (
                f"'{op_label}{col_note}' -> new motl in the pool "
                f"({len(result.df)} particles; no free slot, use the slot dropdown in the pool list)."
            )

        return (*pool_state.to_stores(), slot_map, active, status)

    # ── Single-motl operation form ─────────────────────────────────────────────
    @app.callback(
        Output("me-op-func-form", "children"),
        Input("me-op-func-select", "value"),
        prevent_initial_call=True,
    )
    def generate_op_form(method_name):
        if not method_name:
            return []
        return build_form(getattr(Motl, method_name), id_type="me-op-param")

    # ── Apply a method to the active slot's motl ───────────────────────────────
    # In-place ops (gui output=None) update the active slot; ops that produce a
    # new motl (gui output="motl", e.g. get_random_subset) are added to the pool
    # as a separate entry so the source motl is preserved.
    @app.callback(
        *[Output(f"me-{i}-motl-data-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-undo-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        Output("me-op-status", "children", allow_duplicate=True),
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output("me-slot-map", "data", allow_duplicate=True),
        Output("me-tabs", "active_tab", allow_duplicate=True),
        Output(ids.POOL_GROUPS, "data", allow_duplicate=True),
        Input("me-op-apply-btn", "n_clicks"),
        State("me-op-func-select", "value"),
        State("me-tabs", "active_tab"),
        State({"type": "me-op-param", "owner": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "me-op-param", "owner": ALL, "param": ALL, "tag": ALL}, "id"),
        *[State(f"me-{i}-motl-data-store", "data") for i in range(N_SLOTS)],
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State("me-slot-map", "data"),
        State("me-op-create-group", "value"),
        State("me-op-save-to-disk", "value"),
        State({"type": "path-input", "owner": "me-op-save-dir"}, "value"),
        State("me-op-save-format", "value"),
        State(ids.POOL_GROUPS, "data"),
        State("me-active-target", "data"),
        prevent_initial_call=True,
    )
    def apply_operation(n_clicks, method_name, active_tab, param_values, param_ids, *rest):
        all_slot_data = rest[:N_SLOTS]
        (
            registry,
            pool_meta,
            next_id,
            slot_map,
            create_group_val,
            save_to_disk_val,
            save_dir_val,
            save_fmt_val,
            groups_data,
            active_target,
        ) = rest[N_SLOTS:]

        # 5 pool-related outputs: registry, meta, next_id, slot_map, active_tab
        pool_noup = (no_update,) * 5
        nochange = [no_update] * N_SLOTS

        def _ret(data_out, undo_out, status, pool=pool_noup, groups=no_update):
            return (*data_out, *undo_out, status, *pool, groups)

        if not n_clicks or not method_name or not active_tab:
            raise dash.exceptions.PreventUpdate
        try:
            slot_idx = int(str(active_tab).replace("me-tab-", ""))
        except (ValueError, AttributeError):
            raise dash.exceptions.PreventUpdate
        if slot_idx >= N_SLOTS:
            raise dash.exceptions.PreventUpdate

        current_data = all_slot_data[slot_idx]  # kept for legacy undo fallback only
        slot_map_list_pre = list(slot_map or [])
        active_mid = slot_map_list_pre[slot_idx] if slot_idx < len(slot_map_list_pre) else None
        if not active_mid or not (registry or {}).get(active_mid):
            return _ret(nochange, nochange, "No data in the active slot.")

        current_pool = PoolState.from_stores(registry, pool_meta, next_id)
        kwargs = generate_kwargs(param_ids, param_values, pool_state=current_pool) if param_ids else {}
        gui = getattr(getattr(Motl, method_name), "_gui", {})

        # Operation produces a list of Motls — route through the group output path.
        if gui.get("output") == "motl_group":
            import os as _os

            want_group = bool(create_group_val)
            want_save = bool(save_to_disk_val)
            slot_map = list(slot_map or [None] * N_SLOTS)
            while len(slot_map) < N_SLOTS:
                slot_map.append(None)
            src_mid = slot_map[slot_idx]
            src_label = (registry.get(src_mid) or {}).get("label", f"Slot {slot_idx + 1}")
            try:
                motl = Motl(get_rows(src_mid))
            except PoolPayloadMissing:
                return _ret(nochange, nochange, "Pool entry missing for active slot.")
            motl._pool_motl_id = src_mid
            op_label = gui.get("label", method_name)
            try:
                result_list = invoke_operation(getattr(motl, method_name), kwargs)
            except Exception as exc:
                return _ret(nochange, nochange, f"Error running '{method_name}': {exc}")
            if not isinstance(result_list, list):
                return _ret(
                    nochange, nochange, f"'{method_name}' did not return a list (got {type(result_list).__name__})."
                )
            col_name = kwargs.get("column_name", "")
            new_ids = []
            for i, m in enumerate(result_list):
                df = m.df if hasattr(m, "df") else pd.DataFrame()
                stem = f"{src_label}_{col_name}_{i + 1}" if col_name else f"{src_label}_{op_label}_{i + 1}"
                current_pool, new_mid = insert_motl(current_pool, df, label=stem, has_tab=False)
                new_ids.append(new_mid)
            new_gstate = GroupState.from_store(groups_data)
            if want_group and new_ids:
                glabel = f"{src_label}_by_{col_name}" if col_name else f"{op_label} of {src_label}"
                new_gstate, _ = create_group(new_gstate, new_ids, label=glabel)
            _save_errs = []
            if want_save and new_ids and save_dir_val:
                _ext_map = {
                    "emmotl": ".em",
                    "stopgap": ".star",
                    "dynamo": ".tbl",
                    "relion": ".star",
                    "relion5": ".star",
                    "relion5_1": ".star",
                }
                _fmt = save_fmt_val or "emmotl"
                _ext = _ext_map.get(_fmt, ".em")
                _sdir = save_dir_val.strip()
                _os.makedirs(_sdir, exist_ok=True)
                for new_mid, m in zip(new_ids, result_list):
                    _lbl = (current_pool.registry.get(new_mid) or {}).get("label", new_mid)
                    _path = _os.path.join(_sdir, _lbl + _ext)
                    try:
                        run_operation(m.write_out, {"output_path": _path, "motl_type": _fmt})
                    except Exception as exc:
                        _save_errs.append(f"{_lbl}: {exc}")
            status = f"'{op_label}' → {len(result_list)} motl(s)"
            if want_group:
                status += ", grouped in pool"
            if want_save and new_ids and save_dir_val:
                n_saved = len(new_ids) - len(_save_errs)
                status += f", saved {n_saved}/{len(new_ids)} to disk"
                if _save_errs:
                    status += ". Errors: " + "; ".join(_save_errs[:3])
            return _ret(
                nochange,
                nochange,
                status,
                pool=(*current_pool.to_stores(), slot_map, no_update),
                groups=new_gstate.to_store(),
            )

        # Operation produces a NEW motl — route through the atomic chokepoint.
        if gui.get("output") == "motl":
            slot_map = list(slot_map or [None] * N_SLOTS)
            while len(slot_map) < N_SLOTS:
                slot_map.append(None)
            src_label = (registry.get(slot_map[slot_idx]) or {}).get("label", f"Slot {slot_idx + 1}")
            try:
                motl = Motl(get_rows(slot_map[slot_idx]))
            except PoolPayloadMissing:
                return _ret(nochange, nochange, "Pool entry missing for active slot.")
            motl._pool_motl_id = slot_map[slot_idx]
            try:
                pool_state, mid, result = run_operation_to_pool(
                    getattr(motl, method_name),
                    kwargs,
                    current_pool,
                    label=f"{gui.get('label', method_name)} of {src_label}",
                )
            except Exception as exc:
                return _ret(nochange, nochange, f"Error running '{method_name}': {exc}")
            free = _first_free_slot(slot_map, N_SLOTS)
            if free is not None:
                slot_map[free] = mid
                active = f"me-tab-{free}"
                status = f"'{method_name}' -> new motl in slot {free + 1} ({len(result.df)} particles)."
            else:
                active = no_update
                status = f"'{method_name}' -> new motl in the pool (no free slot; use the slot dropdown in the pool list)."
            return _ret(
                nochange,
                nochange,
                status,
                pool=(*pool_state.to_stores(), slot_map, active),
            )

        # In-place operation — group target: apply to every slot that holds a group member.
        if (active_target or {}).get("type") == "group":
            gid = active_target["id"]
            gstate = GroupState.from_store(groups_data)
            members = list((gstate.groups.get(gid) or {}).get("members", []))
            slot_map_list = list(slot_map or [])
            data_out = [no_update] * N_SLOTS
            undo_out = [no_update] * N_SLOTS
            n_ok, errs = 0, []
            state = current_pool
            for mid in members:
                try:
                    si = slot_map_list.index(mid)
                except ValueError:
                    continue
                try:
                    m = Motl(get_rows(mid))
                except PoolPayloadMissing:
                    errs.append(f"{(registry or {}).get(mid, {}).get('label', mid)}: pool entry missing")
                    continue
                m._pool_motl_id = mid
                pre_op_df = m.df.copy()
                try:
                    res = invoke_operation(getattr(m, method_name), kwargs)
                except Exception as exc:
                    errs.append(f"{(registry or {}).get(mid, {}).get('label', mid)}: {exc}")
                    continue
                if isinstance(res, Motl):
                    result_df = res.df
                elif res is None:
                    result_df = m.df
                else:
                    continue
                save_snapshot(mid, pre_op_df)
                state = replace_motl_rows(state, mid, result_df)
                data_out[si] = no_update  # pool updated; _sync_revisions refreshes the view
                undo_out[si] = mid  # pool-aware undo: restore from snapshot
                n_ok += 1
            status = f"'{method_name}' applied to {n_ok}/{len(members)} motl(s)."
            if errs:
                status += " Errors: " + "; ".join(errs[:3])
            return _ret(data_out, undo_out, status, pool=(*state.to_stores(), slot_map, no_update))

        # In-place operation — single motl: update the active slot.
        slot_map_list = list(slot_map or [])
        mid = active_mid  # already resolved above
        try:
            motl = Motl(get_rows(mid))
            motl._pool_motl_id = mid
        except PoolPayloadMissing:
            return _ret(nochange, nochange, "Pool entry missing for active slot.")
        pre_op_df = motl.df.copy()
        try:
            result = invoke_operation(getattr(motl, method_name), kwargs)
        except Exception as exc:
            return _ret(nochange, nochange, f"Error running '{method_name}': {exc}")

        if isinstance(result, Motl):
            result_df = result.df
        elif result is None:
            result_df = motl.df
        else:
            return _ret(nochange, nochange, f"Ran '{method_name}' — result: {result!r} (table unchanged).")

        save_snapshot(mid, pre_op_df)
        state = replace_motl_rows(current_pool, mid, result_df) if mid else current_pool
        data_out = [no_update] * N_SLOTS  # pool updated; _sync_revisions refreshes the view
        undo_out = [no_update] * N_SLOTS
        undo_out[slot_idx] = mid  # pool-aware undo: restore from snapshot
        status = f"'{method_name}' applied. Particles: {len(pre_op_df)} → {len(result_df)}."
        return _ret(data_out, undo_out, status, pool=(*state.to_stores(), slot_map, no_update))

    # ── Undo the last operation on the active slot ─────────────────────────────
    @app.callback(
        *[Output(f"me-{i}-motl-data-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-undo-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        Output("me-op-status", "children", allow_duplicate=True),
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Input("me-op-undo-btn", "n_clicks"),
        State("me-tabs", "active_tab"),
        *[State(f"me-{i}-undo-store", "data") for i in range(N_SLOTS)],
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def undo_operation(n_clicks, active_tab, *args):
        all_undo_data = args[:N_SLOTS]
        registry, pool_meta, next_id = args[N_SLOTS:]

        if not n_clicks or not active_tab:
            raise dash.exceptions.PreventUpdate

        try:
            slot_idx = int(str(active_tab).replace("me-tab-", ""))
        except (ValueError, AttributeError):
            raise dash.exceptions.PreventUpdate
        if slot_idx >= N_SLOTS:
            raise dash.exceptions.PreventUpdate

        pool_noup = (no_update, no_update, no_update)
        undo_data = all_undo_data[slot_idx]
        if not undo_data:
            empty = [no_update] * N_SLOTS
            return (*empty, *empty, "Nothing to undo for this slot.", *pool_noup)

        empty = [no_update] * N_SLOTS
        data_out = list(empty)
        undo_out = list(empty)
        undo_out[slot_idx] = None

        # Pool-aware undo: undo_data is a motl_id string; restore from server-side snapshot.
        if isinstance(undo_data, str):
            old_df = restore_snapshot(undo_data)
            if old_df is not None and registry is not None:
                state = PoolState.from_stores(registry, pool_meta, next_id)
                state = replace_motl_rows(state, undo_data, old_df)
                return (*data_out, *undo_out, "Undo successful.", *state.to_stores())
            return (*empty, *empty, "Nothing to undo (snapshot expired).", *pool_noup)

        # Legacy undo: undo_data is list[dict] — write rows back to motl-data-store.
        if isinstance(undo_data, list):
            data_out[slot_idx] = undo_data
            return (*data_out, *undo_out, "Undo successful.", *pool_noup)

        return (*empty, *empty, "Nothing to undo.", *pool_noup)

    # ── R1: toggle single / multiple load mode ─────────────────────────────────
    @app.callback(
        Output("me-load-single-section", "style"),
        Output("me-load-multi-section", "style"),
        Input("me-load-mode", "value"),
    )
    def _toggle_load_mode(mode):
        if mode == "multi":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    # ── R1: live match count ────────────────────────────────────────────────────
    @app.callback(
        Output("me-mload-count", "children"),
        Input("me-mload-pattern", "value"),
        prevent_initial_call=True,
    )
    def _count_matches(pattern):
        import glob as _glob
        import os as _os
        from pathlib import Path as _Path

        if not pattern:
            return ""
        pattern = pattern.strip()
        if _os.path.isdir(pattern):
            matches = [
                str(p)
                for p in _Path(pattern).iterdir()
                if p.is_file() and p.suffix.lower() in (".em", ".star", ".csv", ".tbl")
            ]
        else:
            matches = [m for m in _glob.glob(pattern, recursive=True) if _os.path.isfile(m)]
        n = len(matches)
        return f"{n} file(s) matched" if n else "No files matched"

    # ── R1: Load All — expand glob, load each, insert into pool, create group ──
    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output(ids.POOL_GROUPS, "data", allow_duplicate=True),
        Output("me-mload-status", "children"),
        Input("me-mload-btn", "n_clicks"),
        State("me-load-motl-dropdown", "value"),
        State("me-load-rln-value", "data"),
        State("me-load-rln-tomos-store", "data"),
        State("me-mload-pattern", "value"),
        State("me-mload-group-name", "value"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State(ids.POOL_GROUPS, "data"),
        prevent_initial_call=True,
    )
    def _load_multi(
        n_clicks, motl_type, rln_value, rln_tomos, pattern, group_name, registry, pool_meta, next_id, groups_data
    ):
        import glob as _glob
        import os as _os
        from pathlib import Path as _Path
        from cryocat.app.components._motlio_ops import load_motl_from_path as _load, load_kwargs_from_store as _lkfs

        _noup4 = (no_update,) * 4

        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not pattern:
            return (*_noup4, "Enter a glob pattern or folder path.")

        motl_type = motl_type or "emmotl"
        pattern = pattern.strip()

        if _os.path.isdir(pattern):
            ext_map = {"emmotl": ".em", "stopgap": ".csv", "dynamo": ".tbl", "relion": ".star"}
            ext = ext_map.get(motl_type, ".em")
            raw = [str(p) for p in _Path(pattern).iterdir() if p.is_file() and p.suffix.lower() == ext]
        else:
            raw = [m for m in _glob.glob(pattern, recursive=True) if _os.path.isfile(m)]

        if not raw:
            return (*_noup4, "No files matched.")

        matches = sorted(raw, key=lambda p: natural_sort_key(_Path(p).stem))

        pool_state = PoolState.from_stores(registry, pool_meta, next_id)
        gstate = GroupState.from_store(groups_data)
        new_mids = []
        failed = []
        _rln_kws = _lkfs(rln_value)

        for path in matches:
            label = _Path(path).stem
            try:
                table_data, extra_data, _optics, _rln_t, _dtype, _rln_params = _load(
                    path,
                    motl_type,
                    rln_tomos=rln_tomos,
                    **_rln_kws,
                )
            except Exception as exc:
                from cryocat.app import session as _session
                from cryocat.app.event import message_event as _msg_event
                _dash_logger.write(f"Load failed ({label}): {exc}", source="error")
                _session.emit(_msg_event(f"Load failed ({label}): {exc}", level="error"))
                failed.append(f"{label}: {exc}")
                continue
            try:
                pool_state, mid, _ = record_load_to_pool(
                    table_data,
                    motl_type,
                    path,
                    {},
                    pool_state,
                    label=label,
                    extra=extra_data,
                )
                new_mids.append(mid)
            except Exception as exc:
                from cryocat.app import session as _session
                from cryocat.app.event import message_event as _msg_event
                _dash_logger.write(f"Load failed ({label}): {exc}", source="error")
                _session.emit(_msg_event(f"Load failed ({label}): {exc}", level="error"))
                failed.append(f"{label}: {exc}")

        if not new_mids:
            return (*_noup4, f"All loads failed: {'; '.join(failed[:3])}")

        gname = group_name.strip() if group_name and group_name.strip() else None
        gstate, gid = create_group(gstate, new_mids, label=gname)
        status = f"Loaded {len(new_mids)} motl(s) → group '{gstate.groups[gid]['label']}'."
        if failed:
            status += f"  {len(failed)} failed: {failed[0]}{'…' if len(failed) > 1 else ''}."

        return (*pool_state.to_stores(), gstate.to_store(), status)

    # ── R3: clicking a group label sets it as the active target ────────────────
    @app.callback(
        Output("me-active-target", "data", allow_duplicate=True),
        Input({"type": "me-group-label-click", "gid": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _set_active_from_group(n_clicks_list):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "gid" in triggered):
            raise dash.exceptions.PreventUpdate
        return {"type": "group", "id": triggered["gid"]}

    # ── R3: active target label for both operation panels ─────────────────────
    @app.callback(
        Output("me-single-op-target-label", "children"),
        Output("me-multi-op-target-label", "children"),
        Input("me-active-target", "data"),
        Input(ids.POOL_REGISTRY, "data"),
        Input(ids.POOL_GROUPS, "data"),
    )
    def _update_op_target_labels(active_target, registry, groups_data):
        if not active_target:
            return "", ""
        registry = registry or {}
        at_type = active_target.get("type")
        at_id = active_target.get("id")
        if at_type == "motl":
            meta = registry.get(at_id) or {}
            label = meta.get("label", at_id)
            n = meta.get("n_rows", "?")
            base = f"Target: {label} ({n} particles)"
            return base, base
        if at_type == "group":
            gstate = GroupState.from_store(groups_data)
            g = gstate.groups.get(at_id) or {}
            glabel = g.get("label", at_id)
            n = len(g.get("members", []))
            return (
                f"Target: {glabel} ({n} motls — apply to each)",
                f"Target: {glabel} ({n} motls — apply together)",
            )
        return "", ""

    # ── Show/hide group-options panel when a motl_group op is selected ─────────
    @app.callback(
        Output("me-op-group-options", "style"),
        Input("me-op-func-select", "value"),
    )
    def _toggle_group_options(method_name):
        if not method_name:
            return {"display": "none"}
        gui = getattr(getattr(Motl, method_name, None), "_gui", {})
        if gui.get("output") == "motl_group":
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        Output("me-op-save-options", "style"),
        Input("me-op-save-to-disk", "value"),
    )
    def _toggle_save_options(val):
        return {"display": "block"} if val else {"display": "none"}

    @app.callback(
        Output("me-op-apply-btn", "disabled"),
        Output("me-op-group-validation", "children"),
        Input("me-op-create-group", "value"),
        Input("me-op-save-to-disk", "value"),
        Input("me-op-func-select", "value"),
    )
    def _validate_group_options(create_val, save_val, method_name):
        if not method_name:
            return False, ""
        gui = getattr(getattr(Motl, method_name, None), "_gui", {})
        if gui.get("output") != "motl_group":
            return False, ""
        if not create_val and not save_val:
            return True, "Select at least one output option (group or save)."
        return False, ""

    # ── Batch Convert callbacks ────────────────────────────────────────────────
    @app.callback(
        Output("me-batch-convert-target", "children"),
        Input("me-active-target", "data"),
        Input(ids.POOL_GROUPS, "data"),
    )
    def _update_batch_target(active_target, groups_data):
        if not active_target or active_target.get("type") != "group":
            return "No group selected. Click a group label in the pool list to select it."
        gid = active_target["id"]
        gstate = GroupState.from_store(groups_data)
        g = gstate.groups.get(gid) or {}
        glabel = g.get("label", gid)
        n = len(g.get("members", []))
        return f"Group: {glabel} ({n} motl(s))"

    @app.callback(
        Output("me-batch-save-status", "children"),
        Output("me-batch-save-validation", "children"),
        Input("me-batch-save-save-btn", "n_clicks"),
        State("me-active-target", "data"),
        State("me-batch-save-format", "value"),
        State({"type": "path-input", "owner": "me-batch-save-dest-dir"}, "value"),
        State("me-batch-save-filename-policy", "value"),
        State("me-batch-save-filename-suffix", "value"),
        State("me-batch-save-overwrite", "value"),
        State("me-batch-save-rln-value", "data"),
        State({"type": "me-batch-save-writer-param", "owner": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "me-batch-save-writer-param", "owner": ALL, "param": ALL, "tag": ALL}, "id"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_GROUPS, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def _batch_save(
        n_clicks,
        active_target,
        fmt,
        out_dir,
        policy,
        suffix,
        overwrite,
        rln_value,
        writer_vals,
        writer_ids,
        registry,
        groups_data,
        pool_meta,
        pool_next_id,
    ):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not active_target or active_target.get("type") != "group":
            return no_update, "Select a group first (click its label in the pool list)."
        gid = active_target["id"]
        gstate = GroupState.from_store(groups_data)
        g = gstate.groups.get(gid) or {}
        members = list(g.get("members", []))
        registry = registry or {}
        paths = build_batch_paths(members, out_dir or "", fmt or "emmotl", policy or "stem", suffix, registry)
        probs = validate_save(
            out_dir, fmt, rln_value, mode="batch", members=members, paths=paths, overwrite=overwrite or "refuse"
        )
        if probs:
            return no_update, "\n".join(probs)
        pool_state = PoolState.from_stores(registry, pool_meta, pool_next_id)
        writer_kwargs = generate_kwargs(writer_ids, writer_vals, pool_state) if writer_ids else {}
        os.makedirs(out_dir, exist_ok=True)
        status, val = execute_batch_save(members, paths, fmt, rln_value, writer_kwargs, registry)
        return status, val

    # ── Part F: write-back @-variable results to the form text inputs ──────────
    from cryocat.app.formgen import register_form_callbacks

    register_form_callbacks(app, "me-op-param")
    register_form_callbacks(app, "me-multi-param")
