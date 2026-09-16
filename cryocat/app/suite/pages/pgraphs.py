"""Graphs page — five-slot viewer with graph pool and plot editor sidebar (C1–C3).

Layout: sticky sidebar (pool list + plot editor panels) + main area (slot tabs +
shared graph area).  The graph pool stores frozen figures and spec-backed entries.
Each of the five slots is a view onto one pool entry.

Contract
--------
Exposes ``layout``, ``register_callbacks(app)``, and ``DYNAMIC_IDS``.
"""
from __future__ import annotations

import ast
import logging
import dash
from dash import html, dcc, Input, Output, State, no_update, ctx, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from cryocat.app import ids, styles
from cryocat.app.pageshell import page_shell, sidebar_accordion
from cryocat.app.pool import resolve_df as pool_resolve_df
from cryocat.app import datapool
from cryocat.app.components.ploteditor import (
    _ALL_ROLES, _resolve_df, _build_figure, _detect_id_column,
    _apply_layout_only, build_spec, normalize_spec, _add_overlay, OverlaySourceMissing,
    get_plot_editor_sidebar, register_plot_editor_callbacks,
)
from cryocat.app.components.poolslotlist import (
    get_pool_slot_list, register_pool_slot_list_callbacks,
    register_slot_focus_callback, _first_free_slot,
)
from cryocat.app.components.graphsettings import figure_to_dict, apply_settings_to_figure, GRAPH_SETTINGS_DEFAULTS

_log = logging.getLogger(__name__)

N_SLOTS = 5

DYNAMIC_IDS: list[tuple[str, str]] = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tab_to_idx(active_tab: str | None) -> int | None:
    if not active_tab or not active_tab.startswith("gr-slot-"):
        return None
    try:
        return int(active_tab.rsplit("-", 1)[-1])
    except ValueError:
        return None


# ── Layout sections ───────────────────────────────────────────────────────────

def _make_stores() -> list:
    stores = [
        dcc.Store(id="gr-slot-map", data=[None] * N_SLOTS),
        dcc.Store(id="gr-active-id", data=None),
        dcc.Store(id="gr-pe-editing-trace-idx", data=None),
    ]
    for i in range(N_SLOTS):
        stores.append(dcc.Store(id=f"gr-slot-{i}-fig-store", data=None))
    return stores


def _pool_section() -> html.Div:
    return html.Div([
        get_pool_slot_list("gr"),
        html.Div(id="gr-pool-status", style={**styles.HINT, "marginTop": styles.FORM_ROW_GAP}),
    ])


def _sidebar() -> list:
    return [
        sidebar_accordion([
            dbc.AccordionItem(_pool_section(), title="Graph Pool", item_id="pool"),
        ], active_item=["pool"]),
        html.Hr(style={"margin": f"{styles.SECTION_GAP} 0"}),
        *get_plot_editor_sidebar("gr"),
        html.Hr(style={"margin": f"{styles.SECTION_GAP} 0"}),
        sidebar_accordion([
            dbc.AccordionItem(
                html.Div(id="gr-traces-list",
                         children=[html.Div("No active graph.", style=styles.HINT)]),
                title="Traces",
                item_id="traces",
            ),
        ], active_item=["traces"]),
    ]


def _slot_tab(i: int) -> dbc.Tab:
    return dbc.Tab(
        "",
        id=f"gr-tab-{i}",
        tab_id=f"gr-slot-{i}",
        label=f"Slot {i + 1}",
        disabled=True,
    )


def _main() -> list:
    return [
        dbc.Tabs(
            [_slot_tab(i) for i in range(N_SLOTS)],
            id="gr-tabs",
            active_tab="gr-slot-0",
            style={"marginBottom": styles.SECTION_GAP},
        ),
        html.Div(
            id="gr-pe-graph-area",
            children=[
                html.Div(
                    "Select a slot above, assign a graph from the pool, "
                    "or configure the editor and click Plot.",
                    id="gr-pe-graph-placeholder",
                    style={**styles.HINT, "padding": "2rem", "textAlign": "center"},
                ),
                dcc.Graph(
                    id="gr-pe-graph",
                    figure=go.Figure(),
                    config={"displayModeBar": True, "toImageButtonOptions": {"format": "png"}},
                    style={"display": "none", "height": "70vh"},
                    clear_on_unhover=True,
                ),
            ],
        ),
    ]


layout = html.Div(
    [*_make_stores(), page_shell(_sidebar(), _main(), sidebar_width=4)],
    style={"margin": "0", "padding": "0"},
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):  # noqa: C901
    """Register all Graphs page callbacks."""

    register_plot_editor_callbacks(
        app, "gr",
        pool_resolve_df=pool_resolve_df,
        dp_resolve_df=datapool.resolve_payload_df,
        settings_store_id=ids.GRAPH_PALETTE_SIGNAL,
    )

    # ── Pool list (F1: shared PoolSlotList element, no Dup) ───────────────────

    def _graph_row_extra(graph_id, entry):
        return [
            dbc.Button("✕", id={"type": "gr-remove-btn", "graph_id": graph_id},
                       size="sm", color=styles.BTN_NEUTRAL, n_clicks=0,
                       style={"padding": "0 4px", "flexShrink": 0}),
        ]

    register_pool_slot_list_callbacks(
        app, "gr", ids.GRAPH_POOL_REGISTRY, "gr-slot-map", N_SLOTS,
        row_extra_fn=_graph_row_extra,
        active_id_store_id="gr-active-id",
    )

    # ── G5/H3: slot_map change → move focus and active-id off any empty slot ──

    register_slot_focus_callback(
        app, "gr-slot-map", "gr-tabs", "gr-slot-", N_SLOTS,
        active_id_store_id="gr-active-id",
    )

    # ── F3: tab switch → sync active-id (single source of truth) ─────────────

    @app.callback(
        Output("gr-active-id", "data"),
        Input("gr-tabs", "active_tab"),
        State("gr-slot-map", "data"),
        State("gr-active-id", "data"),
        prevent_initial_call=True,
    )
    def _sync_tab_to_active_id(active_tab, slot_map, current_active):
        idx = _tab_to_idx(active_tab)
        if idx is None:
            return no_update
        sm = list(slot_map or [None] * N_SLOTS)
        gid = sm[idx] if idx < len(sm) else None
        if not gid or gid == current_active:
            return no_update
        return gid

    # ── F3: active-id change → switch slot tab if entry is slotted ───────────

    @app.callback(
        Output("gr-tabs", "active_tab", allow_duplicate=True),
        Input("gr-active-id", "data"),
        State("gr-slot-map", "data"),
        State("gr-tabs", "active_tab"),
        prevent_initial_call=True,
    )
    def _sync_active_id_to_tab(active_id, slot_map, current_tab):
        if not active_id:
            return no_update
        sm = list(slot_map or [None] * N_SLOTS)
        for i, gid in enumerate(sm):
            if gid == active_id:
                new_tab = f"gr-slot-{i}"
                if new_tab == current_tab:
                    return no_update
                return new_tab
        return no_update  # not in any slot; don't change tab

    # ── Active-slot kind note / unslotted note (H3) ────────────────────────────

    @app.callback(
        Output("gr-pe-slot-kind-note",   "children"),
        Output("gr-pe-frozen-data-note", "children"),
        Output("gr-pe-add-trace-btn",    "disabled"),
        Input("gr-active-id", "data"),
        Input(ids.GRAPH_POOL_REGISTRY, "data"),
        Input("gr-slot-map", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _update_slot_kind_note(active_id, registry, slot_map):
        if not active_id:
            return "", "", True
        entry = (registry or {}).get(active_id, {})
        sm = list(slot_map or [None] * N_SLOTS)
        in_slot = any(gid == active_id for gid in sm)
        if entry.get("kind") == "frozen":
            return "[frozen]", "Frozen figure — traces are applied on top of the original.", False
        if not in_slot:
            return "[not displayed — assign to a slot to see edits live]", "", False
        return "", "", False

    # ── Remove pool entry ──────────────────────────────────────────────────────

    @app.callback(
        Output(ids.GRAPH_POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.GRAPH_POOL_NEXT_ID,  "data", allow_duplicate=True),
        Output("gr-slot-map", "data", allow_duplicate=True),
        Output("gr-active-id", "data", allow_duplicate=True),
        Output("gr-pool-status", "children", allow_duplicate=True),
        Input({"type": "gr-remove-btn", "graph_id": ALL}, "n_clicks"),
        State(ids.GRAPH_POOL_REGISTRY, "data"),
        State(ids.GRAPH_POOL_NEXT_ID,  "data"),
        State("gr-slot-map", "data"),
        State("gr-active-id", "data"),
        prevent_initial_call=True,
    )
    def _remove_pool_entry(n_list, registry, next_id, slot_map, active_id):
        if not any(n for n in (n_list or []) if n):
            raise dash.exceptions.PreventUpdate
        graph_id = ctx.triggered_id["graph_id"]
        from cryocat.app import graphpool as _graphpool
        state = _graphpool.GraphPoolState.from_stores(registry, next_id)
        state = _graphpool.remove_graph_entry(state, graph_id)
        new_slot_map = [
            (None if sid == graph_id else sid)
            for sid in (slot_map or [None] * N_SLOTS)
        ]
        new_active = None if active_id == graph_id else no_update
        return (*state.to_stores(), new_slot_map, new_active, f"Removed {graph_id}.")

    # ── Tab labels / disabled driven by slot map ───────────────────────────────

    @app.callback(
        *[Output(f"gr-tab-{i}", "label") for i in range(N_SLOTS)],
        *[Output(f"gr-tab-{i}", "disabled") for i in range(N_SLOTS)],
        Input("gr-slot-map", "data"),
        Input(ids.GRAPH_POOL_REGISTRY, "data"),
    )
    def _update_tab_labels(slot_map, registry):
        sm = list(slot_map or [None] * N_SLOTS)
        while len(sm) < N_SLOTS:
            sm.append(None)
        reg = registry or {}
        labels = []
        disabled_flags = []
        for i, gid in enumerate(sm):
            if gid and gid in reg:
                labels.append(reg[gid].get("label", gid))
                disabled_flags.append(False)
            else:
                labels.append(f"Slot {i + 1}")
                disabled_flags.append(True)
        return (*labels, *disabled_flags)

    # ── Tab switch / slot-map change: load slot's figure ──────────────────────
    # G1: spec entries rebuilt on demand from server-side payload.
    # G2: if source is gone, show named message and keep the pool entry.
    # H1: missing overlay sources named in pool-status, never silently dropped.

    @app.callback(
        Output("gr-pe-graph", "figure", allow_duplicate=True),
        Output("gr-pe-graph", "style", allow_duplicate=True),
        Output("gr-pe-graph-placeholder", "style", allow_duplicate=True),
        Output("gr-pe-graph-placeholder", "children", allow_duplicate=True),
        *[Output(f"gr-slot-{i}-fig-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        Output("gr-pool-status", "children", allow_duplicate=True),
        Input("gr-tabs", "active_tab"),
        Input("gr-slot-map", "data"),
        *[State(f"gr-slot-{i}-fig-store", "data") for i in range(N_SLOTS)],
        State(ids.GRAPH_POOL_REGISTRY, "data"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        prevent_initial_call=True,
    )
    def _load_active_slot(active_tab, slot_map, *args):
        slot_figs = list(args[:N_SLOTS])
        registry = args[N_SLOTS]
        settings = args[N_SLOTS + 1]
        slot_idx = _tab_to_idx(active_tab)
        sm = list(slot_map or [None] * N_SLOTS)
        reg = registry or {}
        new_fig_updates = [no_update] * N_SLOTS
        if slot_idx is None:
            return (no_update, no_update, no_update, no_update, *new_fig_updates, no_update)

        fig_data = slot_figs[slot_idx]
        gid = sm[slot_idx] if slot_idx < len(sm) else None
        missing_source_name: str | None = None
        missing_overlays: list[str] = []

        if not fig_data and gid and gid in reg:
            kind = reg[gid].get("kind")
            from cryocat.app import graphpool as _graphpool
            if kind == "frozen":
                try:
                    payload = _graphpool.get_graph_payload(gid)
                    if "frozen_fig" in payload:
                        frozen_fig = payload["frozen_fig"]
                        extra_traces = payload.get("traces", [])
                        if extra_traces:
                            fig = go.Figure(frozen_fig)
                            for trace_cfg in extra_traces:
                                try:
                                    _add_overlay(fig, trace_cfg,
                                                 pool_resolve_df, datapool.resolve_payload_df,
                                                 {}, "", settings)
                                except OverlaySourceMissing as e:
                                    missing_overlays.append(str(e))
                            fig_data = figure_to_dict(fig)
                        else:
                            fig_data = frozen_fig
                    else:
                        fig_data = payload
                    new_fig_updates[slot_idx] = fig_data
                except Exception:
                    pass
            elif kind == "spec":
                try:
                    raw_spec = _graphpool.get_graph_payload(gid)
                    spec = normalize_spec(raw_spec)
                    traces = spec.get("traces", [])
                    primary = traces[0] if traces else {}
                    src_ref = primary.get("source")
                    chart = primary.get("chart")
                    if src_ref and chart:
                        df = _resolve_df(src_ref, pool_resolve_df, datapool.resolve_payload_df)
                        if df is None:
                            missing_source_name = (
                                src_ref.get("motl_id")
                                or src_ref.get("data_id")
                                or str(src_ref)
                            )
                        else:
                            roles = primary.get("roles", {})
                            layout_spec = spec.get("layout", {})
                            chart_opts = primary.get("chart_opts", {})
                            id_col = _detect_id_column(df)
                            fig = _build_figure(
                                chart, df, roles, chart_opts, id_col,
                                settings, layout_spec, None, None,
                                cluster_cols=primary.get("cluster_cols") or None,
                            )
                            if fig:
                                for trace_cfg in traces[1:]:
                                    try:
                                        _add_overlay(fig, trace_cfg,
                                                     pool_resolve_df, datapool.resolve_payload_df,
                                                     roles, chart, settings)
                                    except OverlaySourceMissing as e:
                                        missing_overlays.append(str(e))
                                fig_data = figure_to_dict(fig)
                except Exception:
                    pass

        pool_status: str | dash.no_update = no_update
        if missing_overlays:
            pool_status = (
                f"Overlay source(s) missing for {gid}: "
                + ", ".join(missing_overlays) + "."
            )

        if not fig_data:
            if missing_source_name:
                msg = f"Source '{missing_source_name}' is no longer available — replot."
            else:
                msg = "No figure in this slot. Assign a pool entry or click Plot."
            return (
                no_update,
                {"display": "none"},
                {**styles.HINT, "padding": "2rem", "textAlign": "center"},
                msg,
                *new_fig_updates,
                pool_status,
            )
        return (
            go.Figure(fig_data),
            {"display": "block", "height": "70vh"},
            {"display": "none"},
            no_update,
            *new_fig_updates,
            pool_status,
        )

    # ── Plot: build figure, register in pool, assign to next free slot (F2) ───

    @app.callback(
        Output("gr-pe-graph", "figure", allow_duplicate=True),
        Output("gr-pe-graph", "style", allow_duplicate=True),
        Output("gr-pe-graph-placeholder", "style", allow_duplicate=True),
        Output("gr-pe-status", "children", allow_duplicate=True),
        Output("gr-pool-status", "children", allow_duplicate=True),
        Output("gr-pe-spec-store", "data"),
        Output("gr-slot-map", "data", allow_duplicate=True),
        *[Output(f"gr-slot-{i}-fig-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        Output(ids.GRAPH_POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.GRAPH_POOL_NEXT_ID,  "data", allow_duplicate=True),
        Output("gr-tabs", "active_tab", allow_duplicate=True),
        Output("gr-active-id", "data", allow_duplicate=True),
        Input("gr-pe-plot-btn", "n_clicks"),
        State("gr-pe-src-ref", "data"),
        State("gr-pe-chart", "value"),
        *[State(f"gr-pe-role-{r}", "value") for r in _ALL_ROLES],
        State({"type": "pe-opt", "prefix": "gr", "chart": ALL, "param": ALL}, "value"),
        State({"type": "pe-opt", "prefix": "gr", "chart": ALL, "param": ALL}, "id"),
        State("gr-pe-layout-store", "data"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        State("gr-pe-ext-figure", "data"),
        State("gr-pe-dis-pal-value", "data"),
        State("gr-pe-con-pal-value", "data"),
        State("gr-slot-map", "data"),
        State(ids.GRAPH_POOL_REGISTRY, "data"),
        State(ids.GRAPH_POOL_NEXT_ID,  "data"),
        State(ids.POOL_META,           "data"),
        prevent_initial_call=True,
    )
    def _plot(_n, src_ref, chart, *args):
        n_roles = len(_ALL_ROLES)
        role_values = list(args[:n_roles])
        rest = args[n_roles:]
        opt_values, opt_ids, layout_spec, settings, ext_fig, pal_dis, pal_con, \
            slot_map, registry, next_id, pool_meta = rest

        _log.debug(
            "_plot fired: triggered=%s  src_ref=%r  chart=%r  roles=%r",
            ctx.triggered_id, src_ref, chart,
            {r: v for r, v in zip(_ALL_ROLES, role_values) if v},
        )

        sm = list(slot_map or [None] * N_SLOTS)
        while len(sm) < N_SLOTS:
            sm.append(None)

        def _fail(msg):
            return (no_update, no_update, no_update, msg, no_update, no_update, no_update,
                    *([no_update] * N_SLOTS), no_update, no_update, no_update, no_update)

        if ext_fig:
            fig_dict = _apply_layout_only(ext_fig, layout_spec or {}, settings, pal_dis, pal_con)
            payload = {"frozen_fig": fig_dict, "traces": [], "layout": layout_spec or {}}
            from cryocat.app import graphpool as _graphpool
            state = _graphpool.GraphPoolState.from_stores(registry, next_id)
            lbl = f"External plot {state.next_id}"
            state, graph_id = _graphpool.insert_graph_entry(state, payload, label=lbl, kind="frozen")
            free = _first_free_slot(sm, N_SLOTS)
            slot_figs = [no_update] * N_SLOTS
            if free is not None:
                sm[free] = graph_id
                slot_figs[free] = fig_dict
                new_tab = f"gr-slot-{free}"
                pool_status = f"Registered as {graph_id}, slot {free + 1}."
            else:
                new_tab = no_update
                pool_status = f"Registered as {graph_id} (no free slot — assign via pool list)."
            return (go.Figure(fig_dict), {"display": "block", "height": "70vh"}, {"display": "none"},
                    "Frozen figure applied.", pool_status, no_update, sm,
                    *slot_figs, *state.to_stores(), new_tab, graph_id)

        if not chart:
            _log.debug("_plot early return: chart is None/empty")
            return _fail("Select a chart type.")
        if not src_ref:
            _log.debug("_plot early return: src_ref is None/empty")
            return _fail("Select a source.")

        df = _resolve_df(src_ref, pool_resolve_df, datapool.resolve_payload_df)
        if df is None:
            _log.debug("_plot early return: _resolve_df returned None for src_ref=%r", src_ref)
            return _fail("Source has no data.")

        roles = {r: v for r, v in zip(_ALL_ROLES, role_values) if v}
        chart_opts = {
            oid["param"]: v
            for oid, v in zip(opt_ids, opt_values)
            if oid.get("chart") == chart and v not in (None, "", "None")
        }
        for k, v in list(chart_opts.items()):
            if isinstance(v, str):
                try:
                    chart_opts[k] = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    pass

        motl_id = src_ref.get("motl_id") if isinstance(src_ref, dict) else None
        cluster_cols: list[str] | None = None
        if motl_id and isinstance(pool_meta, dict):
            cluster_cols = (pool_meta.get(motl_id) or {}).get("cluster_cols") or None

        id_col = _detect_id_column(df)
        fig = _build_figure(chart, df, roles, chart_opts, id_col,
                            settings, layout_spec or {}, pal_dis, pal_con,
                            cluster_cols=cluster_cols)
        if fig is None:
            _log.debug("_plot early return: _build_figure returned None (chart=%r roles=%r opts=%r)", chart, roles, chart_opts)
            return _fail("Failed to build figure. Check roles and options.")

        fig_dict = figure_to_dict(fig)
        spec = build_spec(chart, src_ref, roles,
                          layout=layout_spec, chart_opts=chart_opts,
                          cluster_cols=cluster_cols)

        from cryocat.app import graphpool as _graphpool
        state = _graphpool.GraphPoolState.from_stores(registry, next_id)
        lbl = f"Plot {state.next_id}"
        state, graph_id = _graphpool.insert_graph_entry(state, spec, label=lbl, kind="spec")

        free = _first_free_slot(sm, N_SLOTS)
        slot_figs = [no_update] * N_SLOTS
        if free is not None:
            sm[free] = graph_id
            slot_figs[free] = fig_dict
            new_tab = f"gr-slot-{free}"
            pool_status = f"Registered as {graph_id}, slot {free + 1}."
        else:
            new_tab = no_update
            pool_status = f"Registered as {graph_id} (no free slot — assign via pool list)."

        return (fig, {"display": "block", "height": "70vh"}, {"display": "none"},
                "Plot ready.", pool_status, spec, sm,
                *slot_figs, *state.to_stores(), new_tab, graph_id)

    # ── Update layout: re-apply layout spec to current graph ──────────────────

    @app.callback(
        Output("gr-pe-graph", "figure", allow_duplicate=True),
        Output("gr-pe-layout-status", "children"),
        *[Output(f"gr-slot-{i}-fig-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        Input("gr-pe-update-layout-btn", "n_clicks"),
        State("gr-pe-graph", "figure"),
        State("gr-pe-layout-store", "data"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        State("gr-pe-dis-pal-value", "data"),
        State("gr-pe-con-pal-value", "data"),
        State("gr-tabs", "active_tab"),
        prevent_initial_call=True,
    )
    def _update_layout(_n, existing, layout_spec, settings, pal_dis, pal_con, active_tab):
        if not existing:
            return (no_update, "No figure found.", *([no_update] * N_SLOTS))
        updated = _apply_layout_only(existing, layout_spec or {}, settings, pal_dis, pal_con)
        slot_idx = _tab_to_idx(active_tab)
        slot_figs = [no_update] * N_SLOTS
        if slot_idx is not None:
            slot_figs[slot_idx] = updated
        return (go.Figure(updated), "Layout updated.", *slot_figs)

    # ── BC3: Apply to existing — re-render slotted entries with current settings ─

    @app.callback(
        Output("gr-pe-graph", "figure", allow_duplicate=True),
        Output("gr-pe-graph", "style", allow_duplicate=True),
        *[Output(f"gr-slot-{i}-fig-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        Output("gr-pool-status", "children", allow_duplicate=True),
        Output(ids.GRAPH_SETTINGS_STORE, "data", allow_duplicate=True),
        Output("gr-def-status", "children", allow_duplicate=True),
        Input("gr-def-apply-existing-btn", "n_clicks"),
        State("gr-slot-map", "data"),
        State(ids.GRAPH_POOL_REGISTRY, "data"),
        State("gr-tabs", "active_tab"),
        State("gr-pe-layout-store", "data"),
        State("gr-def-font-family", "value"),
        State("gr-def-font-size", "value"),
        State("gr-def-marker-size", "value"),
        State("gr-def-line-width", "value"),
        State("gr-def-line-dash", "value"),
        State("gr-def-dis-pal-value", "data"),
        State("gr-def-con-pal-value", "data"),
        State("gr-def-bg-color", "value"),
        State("gr-def-xaxis-showline", "value"),
        State("gr-def-xaxis-mirror", "value"),
        State("gr-def-xaxis-showgrid", "value"),
        State("gr-def-yaxis-showline", "value"),
        State("gr-def-yaxis-mirror", "value"),
        State("gr-def-yaxis-showgrid", "value"),
        prevent_initial_call=True,
    )
    def _apply_to_existing(_n, slot_map, registry, active_tab, layout_spec,
                           font_family, font_size, marker_size, line_width, line_dash,
                           dis_pal, con_pal, bg_color,
                           x_showline, x_mirror, x_showgrid, y_showline, y_mirror, y_showgrid):
        if not _n:
            raise dash.exceptions.PreventUpdate
        import copy
        from cryocat.app import graphpool as _graphpool
        settings = {
            "font_family": font_family or GRAPH_SETTINGS_DEFAULTS["font_family"],
            "font_size": font_size or GRAPH_SETTINGS_DEFAULTS["font_size"],
            "marker_size": marker_size or GRAPH_SETTINGS_DEFAULTS["marker_size"],
            "line_width": line_width or GRAPH_SETTINGS_DEFAULTS["line_width"],
            "line_dash": line_dash or GRAPH_SETTINGS_DEFAULTS["line_dash"],
            "discrete_palette": dis_pal or GRAPH_SETTINGS_DEFAULTS["discrete_palette"],
            "continuous_palette": con_pal or GRAPH_SETTINGS_DEFAULTS["continuous_palette"],
            "bg_color": bg_color or GRAPH_SETTINGS_DEFAULTS["bg_color"],
            "palette_is_user_set": True,
            "x_showline": bool(x_showline) if x_showline is not None else GRAPH_SETTINGS_DEFAULTS["x_showline"],
            "x_mirror": bool(x_mirror) if x_mirror is not None else GRAPH_SETTINGS_DEFAULTS["x_mirror"],
            "x_showgrid": bool(x_showgrid) if x_showgrid is not None else GRAPH_SETTINGS_DEFAULTS["x_showgrid"],
            "y_showline": bool(y_showline) if y_showline is not None else GRAPH_SETTINGS_DEFAULTS["y_showline"],
            "y_mirror": bool(y_mirror) if y_mirror is not None else GRAPH_SETTINGS_DEFAULTS["y_mirror"],
            "y_showgrid": bool(y_showgrid) if y_showgrid is not None else GRAPH_SETTINGS_DEFAULTS["y_showgrid"],
        }
        sm = list(slot_map or [None] * N_SLOTS)
        while len(sm) < N_SLOTS:
            sm.append(None)
        reg = registry or {}
        slot_figs = [no_update] * N_SLOTS

        for i, gid in enumerate(sm):
            if not gid or gid not in reg:
                continue
            kind = reg[gid].get("kind")
            try:
                payload = _graphpool.get_graph_payload(gid)
            except Exception:
                continue
            if kind == "frozen":
                frozen_src = payload.get("frozen_fig", payload)
                base = _apply_layout_only(
                    copy.deepcopy(frozen_src), layout_spec or {}, settings, None, None
                )
                extra_traces = payload.get("traces", []) if "frozen_fig" in payload else []
                if extra_traces:
                    fig = go.Figure(base)
                    for trace_cfg in extra_traces:
                        try:
                            _add_overlay(fig, trace_cfg, pool_resolve_df, datapool.resolve_payload_df,
                                         {}, "", settings)
                        except OverlaySourceMissing:
                            pass
                    slot_figs[i] = figure_to_dict(fig)
                else:
                    slot_figs[i] = base
            elif kind == "spec":
                spec = normalize_spec(payload)
                traces = spec.get("traces", [])
                primary = traces[0] if traces else {}
                src_ref = primary.get("source")
                chart = primary.get("chart")
                if not src_ref or not chart:
                    continue
                df = _resolve_df(src_ref, pool_resolve_df, datapool.resolve_payload_df)
                if df is None:
                    continue
                roles = primary.get("roles", {})
                entry_layout_spec = spec.get("layout", {})
                chart_opts = primary.get("chart_opts", {})
                id_col = _detect_id_column(df)
                fig = _build_figure(chart, df, roles, chart_opts, id_col,
                                    settings, entry_layout_spec, None, None,
                                    cluster_cols=primary.get("cluster_cols") or None)
                if fig is None:
                    continue
                for trace_cfg in traces[1:]:
                    try:
                        _add_overlay(fig, trace_cfg, pool_resolve_df, datapool.resolve_payload_df,
                                     roles, chart, settings)
                    except OverlaySourceMissing:
                        pass
                slot_figs[i] = figure_to_dict(fig)

        active_idx = _tab_to_idx(active_tab)
        if active_idx is not None and slot_figs[active_idx] is not no_update:
            active_fig = go.Figure(slot_figs[active_idx])
            active_style = {"display": "block", "height": "70vh"}
        else:
            active_fig = no_update
            active_style = no_update

        updated = sum(1 for f in slot_figs if f is not no_update)
        return (active_fig, active_style, *slot_figs,
                f"Settings applied to {updated} slot(s).", settings, "Applied.")

    # ── GU3: helper — build trace row ─────────────────────────────────────────

    def _trace_row(i: int, trace: dict, selected: bool) -> dbc.Row:
        src = trace.get("source") or {}
        src_name = src.get("motl_id") or src.get("data_id") or "—"
        chart_name = trace.get("chart") or "—"
        label_text = f"{src_name} · {chart_name}"
        row_style: dict = {
            "display": "flex", "alignItems": "center",
            "marginBottom": "0.25rem", "padding": "0.2rem 0.35rem",
            "borderRadius": "4px",
        }
        if selected:
            row_style["border"] = "1px solid var(--bs-border-color)"
        return html.Div([
            html.Span(label_text, style={
                "flex": 1, "fontSize": styles.FONT_SM,
                "fontWeight": 600 if selected else "normal",
                "overflow": "hidden", "textOverflow": "ellipsis",
                "whiteSpace": "nowrap",
            }),
            dbc.Button("Select", id={"type": "gr-trace-select-btn", "idx": i},
                       size="sm",
                       color=styles.BTN_PRIMARY if selected else styles.BTN_NEUTRAL,
                       n_clicks=0,
                       style={"marginLeft": "0.25rem", "flexShrink": 0}),
            dbc.Button("✕", id={"type": "gr-trace-remove-btn", "idx": i},
                       size="sm", color=styles.BTN_NEUTRAL, n_clicks=0,
                       style={"padding": "0 4px", "marginLeft": "0.25rem", "flexShrink": 0}),
        ], style=row_style)

    def _render_traces(traces: list, editing_idx: int | None) -> list:
        if not traces:
            return [html.Div("No traces.", style=styles.HINT)]
        return [_trace_row(i, t, i == editing_idx) for i, t in enumerate(traces)]

    def _get_traces(gid: str, kind: str) -> list:
        from cryocat.app import graphpool as _graphpool
        payload = _graphpool.get_graph_payload(gid)
        if kind == "frozen":
            return payload.get("traces", [])
        return normalize_spec(payload).get("traces", [])

    def _rebuild_fig(gid: str, payload: dict, kind: str, settings: dict | None) -> tuple[dict | None, list[str]]:
        if kind == "frozen":
            frozen_fig = payload.get("frozen_fig")
            if not frozen_fig:
                return None, []
            extra = payload.get("traces", [])
            if not extra:
                return frozen_fig, []
            fig = go.Figure(frozen_fig)
            all_warnings: list[str] = []
            for tc in extra:
                try:
                    warns = _add_overlay(fig, tc, pool_resolve_df, datapool.resolve_payload_df,
                                         {}, "", settings)
                    all_warnings.extend(warns)
                except OverlaySourceMissing as e:
                    tc_label = tc.get("label", "Overlay")
                    all_warnings.append(f"Trace '{tc_label}': source '{e}' not found, trace skipped.")
            return figure_to_dict(fig), all_warnings
        spec = normalize_spec(payload)
        traces = spec.get("traces", [])
        if not traces:
            return None, []
        primary = traces[0]
        src_ref = primary.get("source")
        chart = primary.get("chart")
        if not src_ref or not chart:
            return None, []
        df = _resolve_df(src_ref, pool_resolve_df, datapool.resolve_payload_df)
        if df is None:
            return None, []
        primary_label = primary.get("label", "Trace 1")
        all_warnings: list[str] = []
        clean_primary_roles: dict = {}
        for k, v in (primary.get("roles") or {}).items():
            if isinstance(v, list):
                valid = [c for c in v if c in df.columns]
                if valid:
                    clean_primary_roles[k] = valid[0] if (len(valid) == 1 and k in {"x", "y"}) else valid
                elif v:
                    all_warnings.append(
                        f"Trace '{primary_label}': column(s) {v!r} not in source, role '{k}' skipped."
                    )
            elif v and isinstance(v, str) and v in df.columns:
                clean_primary_roles[k] = v
            elif v:
                all_warnings.append(
                    f"Trace '{primary_label}': column '{v}' not in source, role '{k}' skipped."
                )
        layout_sp = spec.get("layout", {})
        copts = primary.get("chart_opts", {})
        cc = primary.get("cluster_cols") or None
        id_col = _detect_id_column(df)
        fig = _build_figure(chart, df, clean_primary_roles, copts, id_col, settings, layout_sp,
                            None, None, cluster_cols=cc)
        if fig is None:
            return None, all_warnings
        for tc in traces[1:]:
            try:
                warns = _add_overlay(fig, tc, pool_resolve_df, datapool.resolve_payload_df,
                                     clean_primary_roles, chart, settings)
                all_warnings.extend(warns)
            except OverlaySourceMissing as e:
                tc_label = tc.get("label", "Overlay")
                all_warnings.append(f"Trace '{tc_label}': source '{e}' not found, trace skipped.")
        return figure_to_dict(fig), all_warnings

    # ── GU3: update traces list when active graph changes ─────────────────────

    @app.callback(
        Output("gr-traces-list", "children"),
        Input("gr-active-id", "data"),
        Input(ids.GRAPH_POOL_REGISTRY, "data"),
        State("gr-pe-editing-trace-idx", "data"),
        prevent_initial_call=True,
    )
    def _update_traces_list(active_id, registry, editing_idx):
        if not active_id:
            return [html.Div("No active graph.", style=styles.HINT)]
        entry = (registry or {}).get(active_id, {})
        kind = entry.get("kind")
        from cryocat.app import graphpool as _graphpool
        try:
            traces = _get_traces(active_id, kind)
        except Exception:
            return [html.Div("Graph not found.", style=styles.HINT)]
        return _render_traces(traces, editing_idx)

    # ── GU1: guard Update trace button ────────────────────────────────────────

    @app.callback(
        Output("gr-pe-update-trace-btn", "disabled"),
        Input("gr-pe-editing-trace-idx", "data"),
        Input("gr-active-id", "data"),
    )
    def _guard_update_btn(editing_idx, active_id):
        return not active_id or editing_idx is None

    # ── GU1: Add trace ────────────────────────────────────────────────────────

    @app.callback(
        Output("gr-pe-graph", "figure", allow_duplicate=True),
        Output("gr-pe-graph", "style", allow_duplicate=True),
        Output("gr-pe-graph-placeholder", "style", allow_duplicate=True),
        Output("gr-pe-status", "children", allow_duplicate=True),
        Output("gr-traces-list", "children", allow_duplicate=True),
        *[Output(f"gr-slot-{i}-fig-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        Input("gr-pe-add-trace-btn", "n_clicks"),
        State("gr-pe-src-ref", "data"),
        State("gr-pe-chart", "value"),
        *[State(f"gr-pe-role-{r}", "value") for r in _ALL_ROLES],
        State({"type": "pe-opt", "prefix": "gr", "chart": ALL, "param": ALL}, "value"),
        State({"type": "pe-opt", "prefix": "gr", "chart": ALL, "param": ALL}, "id"),
        State("gr-pe-layout-store", "data"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        State("gr-active-id", "data"),
        State("gr-slot-map", "data"),
        State(ids.GRAPH_POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State("gr-pe-editing-trace-idx", "data"),
        prevent_initial_call=True,
    )
    def _add_trace(_n, src_ref, chart, *args):
        n_roles = len(_ALL_ROLES)
        role_values = list(args[:n_roles])
        rest = args[n_roles:]
        opt_values, opt_ids, layout_spec, settings, active_id, slot_map, registry, pool_meta, editing_idx = rest

        if not _n or not active_id or not src_ref or not chart:
            raise dash.exceptions.PreventUpdate

        entry = (registry or {}).get(active_id, {})
        kind = entry.get("kind")
        from cryocat.app import graphpool as _graphpool
        try:
            payload = _graphpool.get_graph_payload(active_id)
        except Exception:
            raise dash.exceptions.PreventUpdate

        roles = {r: v for r, v in zip(_ALL_ROLES, role_values) if v}
        chart_opts = {
            oid["param"]: v
            for oid, v in zip(opt_ids, opt_values)
            if oid.get("chart") == chart and v not in (None, "", "None")
        }
        for k, v in list(chart_opts.items()):
            if isinstance(v, str):
                try:
                    chart_opts[k] = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    pass

        motl_id = src_ref.get("motl_id") if isinstance(src_ref, dict) else None
        cluster_cols = None
        if motl_id and isinstance(pool_meta, dict):
            cluster_cols = (pool_meta.get(motl_id) or {}).get("cluster_cols") or None

        if kind == "frozen":
            traces = list(payload.get("traces", []))
            trace_def = {"source": src_ref, "chart": chart, "roles": roles,
                         "chart_opts": chart_opts, "cluster_cols": cluster_cols or [],
                         "label": f"Trace {len(traces) + 1}"}
            traces.append(trace_def)
            new_payload = {**payload, "traces": traces}
        else:
            spec = normalize_spec(payload)
            traces = list(spec.get("traces", []))
            trace_def = {"source": src_ref, "chart": chart, "roles": roles,
                         "chart_opts": chart_opts, "cluster_cols": cluster_cols or [],
                         "label": f"Trace {len(traces) + 1}"}
            traces.append(trace_def)
            new_payload = {**spec, "traces": traces}

        _graphpool.update_graph_payload(active_id, new_payload)

        fig_data, warns = _rebuild_fig(active_id, new_payload, kind, settings)
        sm = list(slot_map or [None] * N_SLOTS)
        slot_figs = [no_update] * N_SLOTS
        if fig_data:
            for i, gid in enumerate(sm):
                if gid == active_id:
                    slot_figs[i] = fig_data

        warn_msg = (" " + "; ".join(warns)) if warns else ""
        traces_list = _render_traces(traces, editing_idx)
        if fig_data:
            return (go.Figure(fig_data), {"display": "block", "height": "70vh"}, {"display": "none"},
                    f"Trace {len(traces)} added.{warn_msg}", traces_list, *slot_figs)
        return (no_update, no_update, no_update,
                f"Trace added (figure rebuild failed).{warn_msg}", traces_list, *slot_figs)

    # ── GU1: Update trace ─────────────────────────────────────────────────────

    @app.callback(
        Output("gr-pe-graph", "figure", allow_duplicate=True),
        Output("gr-pe-graph", "style", allow_duplicate=True),
        Output("gr-pe-graph-placeholder", "style", allow_duplicate=True),
        Output("gr-pe-status", "children", allow_duplicate=True),
        Output("gr-traces-list", "children", allow_duplicate=True),
        *[Output(f"gr-slot-{i}-fig-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        Input("gr-pe-update-trace-btn", "n_clicks"),
        State("gr-pe-src-ref", "data"),
        State("gr-pe-chart", "value"),
        *[State(f"gr-pe-role-{r}", "value") for r in _ALL_ROLES],
        State({"type": "pe-opt", "prefix": "gr", "chart": ALL, "param": ALL}, "value"),
        State({"type": "pe-opt", "prefix": "gr", "chart": ALL, "param": ALL}, "id"),
        State("gr-pe-layout-store", "data"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        State("gr-active-id", "data"),
        State("gr-slot-map", "data"),
        State(ids.GRAPH_POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State("gr-pe-editing-trace-idx", "data"),
        prevent_initial_call=True,
    )
    def _update_trace(_n, src_ref, chart, *args):
        n_roles = len(_ALL_ROLES)
        role_values = list(args[:n_roles])
        rest = args[n_roles:]
        opt_values, opt_ids, layout_spec, settings, active_id, slot_map, registry, pool_meta, editing_idx = rest

        if not _n or not active_id or editing_idx is None or not src_ref or not chart:
            raise dash.exceptions.PreventUpdate

        entry = (registry or {}).get(active_id, {})
        kind = entry.get("kind")
        from cryocat.app import graphpool as _graphpool
        try:
            payload = _graphpool.get_graph_payload(active_id)
        except Exception:
            raise dash.exceptions.PreventUpdate

        roles = {r: v for r, v in zip(_ALL_ROLES, role_values) if v}
        chart_opts = {
            oid["param"]: v
            for oid, v in zip(opt_ids, opt_values)
            if oid.get("chart") == chart and v not in (None, "", "None")
        }
        for k, v in list(chart_opts.items()):
            if isinstance(v, str):
                try:
                    chart_opts[k] = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    pass

        motl_id = src_ref.get("motl_id") if isinstance(src_ref, dict) else None
        cluster_cols = None
        if motl_id and isinstance(pool_meta, dict):
            cluster_cols = (pool_meta.get(motl_id) or {}).get("cluster_cols") or None

        if kind == "frozen":
            traces = list(payload.get("traces", []))
            if editing_idx >= len(traces):
                raise dash.exceptions.PreventUpdate
            orig_label = traces[editing_idx].get("label", f"Trace {editing_idx + 1}")
            traces[editing_idx] = {"source": src_ref, "chart": chart, "roles": roles,
                                   "chart_opts": chart_opts, "cluster_cols": cluster_cols or [],
                                   "label": orig_label}
            new_payload = {**payload, "traces": traces}
        else:
            spec = normalize_spec(payload)
            traces = list(spec.get("traces", []))
            if editing_idx >= len(traces):
                raise dash.exceptions.PreventUpdate
            orig_label = traces[editing_idx].get("label", f"Trace {editing_idx + 1}")
            traces[editing_idx] = {"source": src_ref, "chart": chart, "roles": roles,
                                   "chart_opts": chart_opts, "cluster_cols": cluster_cols or [],
                                   "label": orig_label}
            new_payload = {**spec, "traces": traces}

        _graphpool.update_graph_payload(active_id, new_payload)

        fig_data, warns = _rebuild_fig(active_id, new_payload, kind, settings)
        sm = list(slot_map or [None] * N_SLOTS)
        slot_figs = [no_update] * N_SLOTS
        if fig_data:
            for i, gid in enumerate(sm):
                if gid == active_id:
                    slot_figs[i] = fig_data

        warn_msg = (" " + "; ".join(warns)) if warns else ""
        traces_list = _render_traces(traces, editing_idx)
        if fig_data:
            return (go.Figure(fig_data), {"display": "block", "height": "70vh"}, {"display": "none"},
                    f"Trace updated.{warn_msg}", traces_list, *slot_figs)
        return (no_update, no_update, no_update,
                f"Trace updated (figure rebuild failed).{warn_msg}", traces_list, *slot_figs)

    # ── GU3: Select trace → load into editor ─────────────────────────────────

    @app.callback(
        Output("gr-pe-src-dd", "value", allow_duplicate=True),
        Output("gr-pe-src-ref", "data", allow_duplicate=True),
        Output("gr-pe-chart", "value", allow_duplicate=True),
        *[Output(f"gr-pe-role-{r}", "value", allow_duplicate=True) for r in _ALL_ROLES],
        Output("gr-pe-editing-trace-idx", "data", allow_duplicate=True),
        Output("gr-traces-list", "children", allow_duplicate=True),
        Input({"type": "gr-trace-select-btn", "idx": ALL}, "n_clicks"),
        State("gr-active-id", "data"),
        State(ids.GRAPH_POOL_REGISTRY, "data"),
        State("gr-pe-editing-trace-idx", "data"),
        prevent_initial_call=True,
    )
    def _select_trace(n_list, active_id, registry, current_editing_idx):
        if not any(n for n in (n_list or []) if n):
            raise dash.exceptions.PreventUpdate
        if not active_id:
            raise dash.exceptions.PreventUpdate
        sel_idx = ctx.triggered_id["idx"]
        entry = (registry or {}).get(active_id, {})
        kind = entry.get("kind")
        from cryocat.app import graphpool as _graphpool
        try:
            traces = _get_traces(active_id, kind)
        except Exception:
            raise dash.exceptions.PreventUpdate
        if sel_idx >= len(traces):
            raise dash.exceptions.PreventUpdate
        trace = traces[sel_idx]
        src_ref = trace.get("source") or {}
        chart_val = trace.get("chart")
        roles = dict(trace.get("roles") or {})
        role_values = [roles.get(r) for r in _ALL_ROLES]
        src = src_ref if isinstance(src_ref, dict) else {}
        if "motl_id" in src:
            dd_val = f"motl:{src['motl_id']}"
        elif "data_id" in src:
            dd_val = f"data:{src['data_id']}"
        else:
            dd_val = None
        traces_list = _render_traces(traces, sel_idx)
        return (dd_val, src_ref, chart_val, *role_values, sel_idx, traces_list)

    # ── GU3: Remove trace ─────────────────────────────────────────────────────

    @app.callback(
        Output("gr-pe-graph", "figure", allow_duplicate=True),
        Output("gr-pe-graph", "style", allow_duplicate=True),
        Output("gr-pe-graph-placeholder", "style", allow_duplicate=True),
        Output("gr-pe-status", "children", allow_duplicate=True),
        Output("gr-traces-list", "children", allow_duplicate=True),
        Output("gr-pe-editing-trace-idx", "data", allow_duplicate=True),
        *[Output(f"gr-slot-{i}-fig-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        Input({"type": "gr-trace-remove-btn", "idx": ALL}, "n_clicks"),
        State("gr-active-id", "data"),
        State("gr-slot-map", "data"),
        State(ids.GRAPH_POOL_REGISTRY, "data"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        State("gr-pe-editing-trace-idx", "data"),
        prevent_initial_call=True,
    )
    def _remove_trace(n_list, active_id, slot_map, registry, settings, editing_idx):
        if not any(n for n in (n_list or []) if n):
            raise dash.exceptions.PreventUpdate
        if not active_id:
            raise dash.exceptions.PreventUpdate
        rm_idx = ctx.triggered_id["idx"]
        entry = (registry or {}).get(active_id, {})
        kind = entry.get("kind")
        from cryocat.app import graphpool as _graphpool
        try:
            payload = _graphpool.get_graph_payload(active_id)
        except Exception:
            raise dash.exceptions.PreventUpdate

        if kind == "frozen":
            traces = list(payload.get("traces", []))
            if rm_idx >= len(traces):
                raise dash.exceptions.PreventUpdate
            traces.pop(rm_idx)
            new_payload = {**payload, "traces": traces}
        else:
            spec = normalize_spec(payload)
            traces = list(spec.get("traces", []))
            if rm_idx >= len(traces):
                raise dash.exceptions.PreventUpdate
            traces.pop(rm_idx)
            new_payload = {**spec, "traces": traces}

        _graphpool.update_graph_payload(active_id, new_payload)

        new_editing = None if editing_idx == rm_idx else (
            editing_idx - 1 if editing_idx is not None and editing_idx > rm_idx
            else editing_idx
        )

        fig_data, warns = _rebuild_fig(active_id, new_payload, kind, settings)
        sm = list(slot_map or [None] * N_SLOTS)
        slot_figs = [no_update] * N_SLOTS
        if fig_data:
            for i, gid in enumerate(sm):
                if gid == active_id:
                    slot_figs[i] = fig_data
        elif not traces and kind != "frozen":
            for i, gid in enumerate(sm):
                if gid == active_id:
                    slot_figs[i] = None

        warn_msg = (" " + "; ".join(warns)) if warns else ""
        traces_list = _render_traces(traces, new_editing)
        if fig_data:
            return (go.Figure(fig_data), {"display": "block", "height": "70vh"}, {"display": "none"},
                    f"Trace removed. {len(traces)} remaining.{warn_msg}", traces_list,
                    new_editing, *slot_figs)
        return (no_update, no_update, no_update,
                f"Trace removed. {len(traces)} remaining.{warn_msg}", traces_list,
                new_editing, *slot_figs)

