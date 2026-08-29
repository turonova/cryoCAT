"""Tango twist analysis — suite tab at /tango.

Two sidebar sections + up to 5 descriptor slots in the main tabs:
  Twist vector:   Input motl + TwistDescriptor params + compute button.
  Descriptor:     Descriptor / support / feature form + compute (gated on twist).
                  Up to 5 slots; each slot shows diagnostics and a close button.
  Table → motl:   Merge twist rows into a pool motl or create a clean subset.

DYNAMIC_IDS declares all AG Grid containers so test_app_ids.py accepts them.

body[data-tool="tango"] is set by the suite router (suite/app.py W1) and scoped
in assets/styles.css so tango palette tokens apply to portals too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dash import html, dcc, Input, Output, State, ALL, no_update, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as _PCA

from cryocat.app import ids
from cryocat.app.pool import get_motl, PoolPayloadMissing
from cryocat.app.apputils import generate_kwargs
from cryocat.app.components.registry import Registry
from cryocat.app.components.motlinput import get_motl_input, register_motl_input_callbacks
from cryocat.app.components.tableview import get_table_component, register_table_callbacks
from cryocat.app.components.tableplot import register_table_plot_callbacks
from cryocat.app.components.tablecluster import register_table_cluster_callbacks
from cryocat.app.components.tabletomotl import get_table_to_motl, register_table_to_motl_callbacks
from cryocat.app import formgen, styles
from cryocat.app.pageshell import page_shell
from cryocat.app.components.tablesource import get_table_source, register_table_source_callbacks
from cryocat.app.components.tomoview import get_viewer_component, register_viewer_callbacks
from cryocat.app.components.graphsettings import styled_figure as _styled_figure

from cryocat.analysis.tango import TwistDescriptor, Descriptor, CustomDescriptor
from cryocat.utils.classutils import get_class_names_by_parent, get_classes_from_names

_descriptors: list[str] = get_class_names_by_parent("Descriptor", "cryocat.analysis.tango")
_features: list[str] = get_class_names_by_parent("Feature", "cryocat.analysis.tango")
_supports: list[str] = get_class_names_by_parent("Support", "cryocat.analysis.tango")
_feat_desc_map: dict = Descriptor.build_feature_descriptor_map(_features, _descriptors)
_desc_feat_map: dict = Descriptor.build_descriptor_feature_map(_descriptors, _features)

_twist_objects: Registry[TwistDescriptor] = Registry("tango-twist", max_items=5)

_DESC_SLOTS = 5

from cryocat.app import datapool as _datapool

DYNAMIC_IDS: list[tuple[str, str]] = [
    ("tango-twist-tabv-grid-container", "tango-twist-tabv-grid"),
    *[(f"tango-desc-{i}-grid-container", f"tango-desc-{i}-grid") for i in range(_DESC_SLOTS)],
]

_hint = {"color": "var(--color9)"}


# ── Diagnostics helper ────────────────────────────────────────────────────────

def _build_diagnostics(data, settings=None):
    """Return diagnostics children for a descriptor DataFrame or raw records."""
    if data is None or (not isinstance(data, pd.DataFrame) and not data):
        return []
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    numeric_cols = [c for c in df.columns if c != "qp_id" and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return [html.Div("No numeric features for diagnostics.", style={"color": styles.COLOR_MUTED})]
    df_feat = df[numeric_cols].dropna()
    if len(df_feat) < 2:
        return [html.Div("Too few rows for diagnostics.", style={"color": styles.COLOR_MUTED})]
    children = []
    try:
        X_scaled = StandardScaler().fit_transform(df_feat.values)
        pca = _PCA()
        pca.fit(X_scaled)
        evr = pca.explained_variance_ratio_
        n = len(evr)
        sort_idx = np.argsort(evr)
        evr_sorted = evr[sort_idx]
        cum_sorted = np.cumsum(evr_sorted)
        x_sorted = [numeric_cols[i] for i in sort_idx]
        fig_scree = go.Figure()
        fig_scree.add_bar(x=x_sorted, y=evr_sorted.tolist(), name="per-component")
        fig_scree.add_scatter(x=x_sorted, y=cum_sorted.tolist(), name="cumulative", mode="lines+markers")
        fig_scree.update_layout(
            title=dict(text="Explained variance — z-scored features", font=dict(size=11)),
            xaxis_title="Feature", yaxis_title="Variance explained",
            height=220, margin=dict(l=30, r=10, t=35, b=30),
            legend=dict(orientation="h", y=-0.25), uirevision="scree",
        )
        try:
            _scree_fig = _styled_figure(fig_scree, settings or {}, uirevision="scree")
        except Exception:
            _scree_fig = fig_scree
        children.append(dcc.Graph(figure=_scree_fig, style={"marginBottom": "0.5rem"}))
    except Exception as exc:
        children.append(html.Div(f"Scree plot failed: {exc}", style={"color": styles.COLOR_MUTED}))
    try:
        corr = df_feat.corr().round(2)
        z = corr.values.tolist()
        labels = corr.columns.tolist()
        fig_corr = ff.create_annotated_heatmap(z=z, x=labels, y=labels, colorscale="RdBu", zmin=-1, zmax=1, showscale=True)
        fig_corr.update_layout(
            title=dict(text="Feature correlation heatmap", font=dict(size=11)),
            height=max(220, 60 + 30 * len(labels)),
            margin=dict(l=60, r=10, t=35, b=60), uirevision="corr",
        )
        try:
            _corr_fig = _styled_figure(fig_corr, settings or {}, uirevision="corr")
        except Exception:
            _corr_fig = fig_corr
        children.append(dcc.Graph(figure=_corr_fig, style={"marginBottom": "0.5rem"}))
    except Exception as exc:
        children.append(html.Div(f"Heatmap failed: {exc}", style={"color": styles.COLOR_MUTED}))
    try:
        summary = df_feat.describe().T[["min", "max", "mean", "std"]]
        summary["distinct"] = df_feat.nunique()
        summary["constant"] = summary["std"] == 0
        summary = summary.round(4).reset_index().rename(columns={"index": "feature"})
        children.append(html.Details([
            html.Summary("Per-feature summary", style={"fontSize": styles.FONT_SM, "cursor": "pointer"}),
            html.Table([
                html.Thead(html.Tr([html.Th(c, style={"padding": "2px 6px"}) for c in summary.columns])),
                html.Tbody([
                    html.Tr([
                        html.Td(str(v), style={"padding": "2px 6px", "color": styles.COLOR_MUTED if col == "constant" and v else "inherit"})
                        for col, v in zip(summary.columns, row)
                    ])
                    for row in summary.itertuples(index=False)
                ]),
            ], style={"fontSize": styles.FONT_SM, "borderCollapse": "collapse", "width": "100%"}),
        ], style={"marginTop": "0.5rem"}))
    except Exception as exc:
        children.append(html.Div(f"Summary failed: {exc}", style={"color": styles.COLOR_MUTED}))
    return children


# ── Tile helpers (one per tab) ─────────────────────────────────────────────────

def _twist_load_fn(path, nn_radius=None):
    """Load a TwistDescriptor table from a file; returns dict with df and nn_radius."""
    df = TwistDescriptor.read_in(path)
    return {"df": df, "nn_radius": nn_radius}


def _twist_tile() -> list:
    _compute_children = [
        get_motl_input("tango-mi", label="Input motl"),
        formgen.form_row(
            "nn_radius",
            dcc.Input(
                id="tango-nn-radius",
                type="number",
                placeholder="Required",
                min=0,
                style=styles.FORM_COMPACT_INPUT,
            ),
            "Nearest-neighbour search radius in voxels",
            label_text="NN radius",
        ),
        formgen.form_row(
            "column_name",
            formgen.make_dropdown(
                "tango-column-name",
                ["tomo_id", "object_id", "class", "geom1", "geom2", "geom3", "geom4", "geom5"],
                "tomo_id",
                clearable=False,
            ),
            "Motl column used to group nearest neighbours",
        ),
        formgen.form_row(
            "symm_type",
            formgen.make_dropdown(
                "tango-symm-type",
                ["None", "C", "cube", "tetrahedron", "octahedron", "icosahedron", "dodecahedron"],
                "None",
                clearable=False,
            ),
            "Particle symmetry to apply during twist computation",
        ),
        html.Div(
            formgen.form_row(
                "c_symm_value",
                dcc.Input(
                    id="tango-c-symm-value",
                    type="number",
                    value=2,
                    min=2,
                    step=1,
                    style=styles.FORM_COMPACT_INPUT,
                ),
                "Degree of cyclic symmetry (C2, C3, …)",
                label_text="C-symmetry value",
            ),
            id="tango-c-symm-div",
            style={"display": "none"},
        ),
        html.Div(
            style={"marginTop": "0.5rem"},
            children=[
                dbc.Checklist(
                    id="tango-remove-qp",
                    options=[{"label": "Remove query particle from NN", "value": "on"}],
                    value=[],
                ),
            ],
        ),
        dbc.Checklist(
            id="tango-remove-duplicates",
            options=[{"label": "Remove duplicate pairs", "value": "on"}],
            value=[],
        ),
        dbc.Button(
            "Compute twist",
            id="tango-run-twist-btn",
            color="light",
            style={"width": "100%", "marginTop": "0.75rem"},
        ),
    ]

    _radius_extra = [
        formgen.form_row(
            "nn_radius",
            dcc.Input(
                id={"type": "tango-twist-src-ts-extra", "param": "nn_radius"},
                type="number",
                placeholder="Leave empty to compute from table",
                min=0,
                style=styles.FORM_COMPACT_INPUT,
            ),
            "NN radius in voxels. If empty, computed as the maximum twist-vector magnitude.",
            label_text="Radius (opt.)",
            label_id="tango-twist-file-radius-lbl",
            truly_optional=True,
        )
    ]

    return [
        get_table_source(
            "tango-twist-src",
            compute_children=_compute_children,
            file_extensions=(".csv", ".pkl"),
            extra_file_children=_radius_extra,
            label="Source",
        ),
        html.Div(id="tango-twist-status", style={**_hint, "marginTop": "0.3rem"}),
    ]


def _desc_tile() -> list:
    return [
        formgen.form_row(
            "support_type",
            formgen.make_dropdown(
                "tango-support-dropdown",
                _supports,
                None,
                clearable=True,
                placeholder="Choose support (optional)...",
            ),
            "Optional support function applied before descriptor computation",
            truly_optional=True,
        ),
        html.Div(id="tango-support-form", style={"marginTop": "0.3rem"}),
        formgen.form_row(
            "descriptor",
            formgen.make_dropdown(
                "tango-desc-dropdown",
                _descriptors,
                None,
                placeholder="Choose descriptor...",
            ),
            "Descriptor class to compute from the twist data",
        ),
        html.Div(id="tango-desc-form", style={"marginTop": "0.3rem"}),
        html.Div(
            [
                formgen.form_row(
                    "features",
                    formgen.make_dropdown(
                        "tango-feat-dropdown",
                        _features,
                        None,
                        placeholder="Choose features...",
                        multi=True,
                    ),
                    "Feature functions to combine with the descriptor",
                ),
                html.Div(id="tango-feat-form", style={"marginTop": "0.3rem"}),
            ],
            id="tango-feat-section",
            style={"display": "none"},
        ),
        dbc.Button(
            "Compute descriptors",
            id="tango-run-desc-btn",
            color="light",
            style={"width": "100%", "marginTop": "0.75rem"},
            disabled=True,
        ),
        html.Div(id="tango-desc-status", style={**_hint, "marginTop": "0.3rem"}),
    ]


def _make_stores() -> list:
    return [
        dcc.Store(id="tango-twist-handle"),
        dcc.Store(id="tango-twist-next-id", data=0),
        dcc.Store(id="tango-twist-tabv-global-data-store"),
        dcc.Store(id="tango-twist-tv-data"),
        dcc.Store(id="tango-twist-tv-index", data=0),
        dcc.Store(id="tango-desc-registry-store", data={}),
        dcc.Store(id="tango-desc-next-id", data=0),
        *[dcc.Store(id=f"tango-desc-{i}-global-data-store") for i in range(_DESC_SLOTS)],
    ]


def _desc_slot_content(i: int) -> html.Div:
    """Per-slot tab content: close button, stale notice, table, diagnostics."""
    return html.Div([
        html.Div(
            [
                html.Span(id=f"tango-desc-{i}-stale-badge", style={"display": "none"}),
                dbc.Button(
                    "× Close",
                    id=f"tango-desc-{i}-close-btn",
                    color="link",
                    size="sm",
                    style={"padding": "0 0.25rem", "color": styles.COLOR_MUTED},
                ),
            ],
            style={"display": "flex", "justifyContent": "flex-end", "alignItems": "center", "gap": "0.5rem", "marginBottom": styles.SECTION_GAP},
        ),
        get_table_component(f"tango-desc-{i}"),
        html.Div(id=f"tango-desc-{i}-diagnostics", style={"marginTop": "0.5rem"}),
    ])


# ── Layout ─────────────────────────────────────────────────────────────────────


def _sidebar() -> list:
    return [
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    _twist_tile(),
                    title="Twist vector",
                    item_id="tango-acc-twist",
                ),
                dbc.AccordionItem(
                    _desc_tile(),
                    title="Descriptor",
                    item_id="tango-acc-desc",
                ),
                dbc.AccordionItem(
                    get_table_to_motl("tango-ttm"),
                    title="Table → motl",
                    item_id="tango-acc-ttm",
                ),
            ],
            id="tango-sidebar-acc",
            active_item=["tango-acc-twist"],
            always_open=True,
        ),
    ]


def _main() -> list:
    return [
        dbc.Tabs(
            id="tango-tabs",
            active_tab="tango-tab-twist",
            children=[
                dbc.Tab(
                    html.Div([
                        get_table_component("tango-twist-tabv"),
                        get_viewer_component("tango-twist-tv"),
                    ]),
                    label="Twist vector",
                    tab_id="tango-tab-twist",
                ),
                *[
                    dbc.Tab(
                        _desc_slot_content(i),
                        label=f"Descriptor {i + 1}",
                        tab_id=f"tango-tab-desc-{i}",
                        id=f"tango-tab-desc-{i}",
                        disabled=True,
                    )
                    for i in range(_DESC_SLOTS)
                ],
            ],
        ),
    ]


layout = html.Div(
    [
        *_make_stores(),
        page_shell(_sidebar(), _main()),
    ],
    className="tango-theme",
    style={"margin": "0", "padding": "0"},
)


# ── Callbacks ──────────────────────────────────────────────────────────────────

def register_callbacks(app) -> None:
    register_motl_input_callbacks(app, "tango-mi")
    register_table_source_callbacks(
        app, "tango-twist-src",
        check_fn=TwistDescriptor.check_twist_columns,
        load_fn=_twist_load_fn,
    )

    register_table_callbacks(
        app, "tango-twist-tabv",
        resolve_df=_datapool.resolve_df, resolve_n_rows=_datapool.resolve_n_rows,
        tabs_id="tango-tabs", tab_value="tango-tab-twist",
    )
    register_table_plot_callbacks(
        app, "tango-twist-tabv-table-plot", "tango-twist-tabv-global-data-store",
        table_grid_id="tango-twist-tabv-grid", resolve_df=_datapool.resolve_df,
    )
    register_table_cluster_callbacks(
        app, "tango-twist-tabv-table-cluster", "tango-twist-tabv-global-data-store",
        table_grid_id="tango-twist-tabv-grid", resolve_df=_datapool.resolve_df,
    )

    for _i in range(_DESC_SLOTS):
        register_table_callbacks(
            app, f"tango-desc-{_i}",
            resolve_df=_datapool.resolve_df, resolve_n_rows=_datapool.resolve_n_rows,
            tabs_id="tango-tabs", tab_value=f"tango-tab-desc-{_i}",
        )
        register_table_plot_callbacks(
            app, f"tango-desc-{_i}-table-plot", f"tango-desc-{_i}-global-data-store",
            table_grid_id=f"tango-desc-{_i}-grid", resolve_df=_datapool.resolve_df,
        )
        register_table_cluster_callbacks(
            app, f"tango-desc-{_i}-table-cluster", f"tango-desc-{_i}-global-data-store",
            table_grid_id=f"tango-desc-{_i}-grid", resolve_df=_datapool.resolve_df,
        )

    register_table_to_motl_callbacks(app, "tango-ttm", source_table_id="tango-twist-tabv-grid", id_column="qp_id")

    # ── Show/hide C-symmetry value input ─────────────────────────────────────

    @app.callback(
        Output("tango-c-symm-div", "style"),
        Input("tango-symm-type", "value"),
        State("tango-c-symm-div", "style"),
        prevent_initial_call=True,
    )
    def _toggle_c_symm(symm, current_style):
        show = symm == "C"
        return {**(current_style or {}), "display": "block" if show else "none"}

    # ── Gate descriptor button and accordion sections on twist handle ─────────

    @app.callback(
        Output("tango-run-desc-btn", "disabled"),
        Input("tango-twist-handle", "data"),
    )
    def _gate_on_twist(handle):
        return not bool(handle)

    @app.callback(
        Output("tango-sidebar-acc", "active_item"),
        Input("tango-sidebar-acc", "active_item"),
        State("tango-twist-handle", "data"),
        prevent_initial_call=True,
    )
    def _gate_accordion(active_items, twist_handle):
        if twist_handle:
            return no_update
        gated = {"tango-acc-desc", "tango-acc-ttm"}
        items = active_items if isinstance(active_items, list) else ([active_items] if active_items else [])
        filtered = [item for item in items if item not in gated]
        return filtered if len(filtered) != len(items) else no_update

    # ── Descriptor form generation ────────────────────────────────────────────

    @app.callback(
        Output("tango-support-form", "children"),
        Input("tango-support-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _show_support_form(class_name):
        if not class_name:
            return []
        cls = get_classes_from_names(class_name, "cryocat.analysis.tango")
        return formgen.build_form(
            cls,
            id_type="tango-support-params",
            id_extra={"cls_name": class_name},
            exclude=["twist_desc"],
        )

    @app.callback(
        Output("tango-desc-form", "children"),
        Output("tango-feat-section", "style"),
        Output("tango-feat-dropdown", "options"),
        Output("tango-feat-dropdown", "value"),
        Input("tango-desc-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _show_desc_form(class_name):
        if not class_name:
            return [], {"display": "none"}, [], []
        cls = get_classes_from_names(class_name, "cryocat.analysis.tango")
        if class_name == "CustomDescriptor":
            forms = []
            avail_features = _features
        elif class_name == "TwistDescriptor":
            forms = formgen.build_form(
                cls,
                id_type="tango-desc-params",
                id_extra={"cls_name": class_name},
                exclude=["input_twist", "input_motl", "nn_radius", "column_name",
                         "symm", "remove_qp", "remove_duplicates", "build_unique_desc"],
            )
            avail_features = []
        else:
            forms = formgen.build_form(
                cls,
                id_type="tango-desc-params",
                id_extra={"cls_name": class_name},
                exclude=["twist_df", "build_unique_desc"],
            )
            ending = class_name.replace("Descriptor", "")
            avail_features = [f for f in _features if not f.endswith(ending)]
        return forms, {"display": "block"}, avail_features, []

    @app.callback(
        Output("tango-feat-form", "children"),
        Input("tango-feat-dropdown", "value"),
        State("tango-desc-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _show_feat_form(class_names, desc_class_name):
        if not class_names or not desc_class_name:
            return []
        forms = []
        for cls_name in class_names:
            parent_desc = _feat_desc_map.get(cls_name, "TwistDescriptor")
            if parent_desc == "TwistDescriptor":
                continue
            feat_forms = formgen.build_form(
                get_classes_from_names(parent_desc, "cryocat.analysis.tango"),
                id_type="tango-feat-params",
                id_extra={"cls_name": cls_name},
                exclude=["twist_df", "build_unique_desc"],
            )
            if feat_forms:
                forms.append(formgen.section_divider(cls_name))
                forms.extend(feat_forms)
        return forms

    # ── Compute twist ─────────────────────────────────────────────────────────

    @app.callback(
        Output("tango-twist-handle", "data"),
        Output("tango-twist-tabv-global-data-store", "data"),
        Output("tango-tabs", "active_tab", allow_duplicate=True),
        Output("tango-twist-status", "children"),
        Output("tango-twist-next-id", "data"),
        Input("tango-run-twist-btn", "n_clicks"),
        State("tango-mi-value", "data"),
        State("tango-nn-radius", "value"),
        State("tango-column-name", "value"),
        State("tango-symm-type", "value"),
        State("tango-c-symm-value", "value"),
        State("tango-remove-qp", "value"),
        State("tango-remove-duplicates", "value"),
        State(ids.POOL_REGISTRY, "data"),
        State("tango-twist-next-id", "data"),
        prevent_initial_call=True,
    )
    def _compute_twist(
        n_clicks,
        motl_ids,
        nn_radius,
        column_name,
        symm_type,
        c_symm_value,
        remove_qp,
        remove_duplicates,
        registry,
        twist_next_id,
    ):
        from cryocat.app.instrument import snapshot as _snap, reset as _reset, start_trace as _start_trace
        if not n_clicks:
            raise PreventUpdate
        if not motl_ids:
            return no_update, no_update, no_update, "Select a motl first.", no_update
        _snap("load+select")   # D1/D4: everything since load_motl reset() — also prints trace
        _start_trace()         # D3: begin twist trace
        motl_id = motl_ids[0]
        if nn_radius is None or nn_radius <= 0:
            return no_update, no_update, no_update, "NN radius is required.", no_update
        try:
            motl_obj = get_motl(motl_id)
        except PoolPayloadMissing:
            return no_update, no_update, no_update, "Motl payload missing — reload.", no_update
        symm = None if symm_type == "None" else (c_symm_value if symm_type == "C" else symm_type)
        source_label = (registry or {}).get(motl_id, {}).get("label", "Motl")
        new_twist_id = (twist_next_id or 0) + 1
        twist_id = f"twist-{new_twist_id}"

        from cryocat.app import provenance as _prov
        from cryocat.app import session as _session
        from cryocat.app.logger import invoke_operation as _invoke_op
        var = _prov.bind(twist_id)
        try:
            twist_desc = _invoke_op(
                TwistDescriptor,
                {
                    "input_motl": motl_obj,
                    "nn_radius": nn_radius,
                    "column_name": column_name or "tomo_id",
                    "symm": symm,
                    "remove_qp": bool(remove_qp),
                    "remove_duplicates": bool(remove_duplicates),
                    "build_unique_desc": False,
                },
                assign_to=var,
                pool_id=twist_id,
                label=f"{source_label} twist",
            )
        except Exception as exc:
            return no_update, no_update, no_update, f"Error: {exc}", no_update
        _prov.record(twist_id, _session.last_seq())

        obj_key = _twist_objects.add(twist_desc)
        _df = twist_desc.df if hasattr(twist_desc, "df") and twist_desc.df is not None else pd.DataFrame()
        n = len(_df)
        handle = {
            "obj_key": obj_key,
            "twist_id": twist_id,
            "source_motl_id": motl_id,
            "label": f"{source_label} twist",
            "nn_radius": nn_radius,
        }
        global_ref = _datapool.insert(_df, label=f"{source_label} twist", id_column="qp_id")
        status = f"Twist computed: {n:,} pairs."
        _snap("twist")   # D2: _compute_twist wall time (grid update follows async)
        return handle, global_ref, "tango-tab-twist", status, new_twist_id

    # ── Load twist from file (via tablesource) ────────────────────────────────

    @app.callback(
        Output("tango-twist-handle", "data", allow_duplicate=True),
        Output("tango-twist-tabv-global-data-store", "data", allow_duplicate=True),
        Output("tango-tabs", "active_tab", allow_duplicate=True),
        Output("tango-twist-status", "children", allow_duplicate=True),
        Output("tango-twist-next-id", "data", allow_duplicate=True),
        Input("tango-twist-src-ts-loaded", "data"),
        State("tango-twist-next-id", "data"),
        prevent_initial_call=True,
    )
    def _handle_twist_loaded(loaded, twist_next_id):
        if not loaded:
            raise PreventUpdate

        from io import StringIO
        df = pd.read_json(StringIO(loaded["df"]), orient="split")
        nn_radius = loaded.get("nn_radius")

        new_twist_id = (twist_next_id or 0) + 1
        twist_id = f"twist-{new_twist_id}"

        from cryocat.app import provenance as _prov
        from cryocat.app import session as _session
        from cryocat.app.logger import invoke_operation as _invoke_op
        var = _prov.bind(twist_id)
        try:
            twist_desc = _invoke_op(
                TwistDescriptor,
                {"input_twist": df, "nn_radius": nn_radius},
                assign_to=var,
                pool_id=twist_id,
                label="Loaded twist",
            )
        except Exception as exc:
            return no_update, no_update, no_update, f"Error: {exc}", no_update
        _prov.record(twist_id, _session.last_seq())

        obj_key = _twist_objects.add(twist_desc)
        radius_note = (
            f"{twist_desc.nn_radius:.1f} ({twist_desc.radius_source})"
            if twist_desc.nn_radius is not None
            else "unknown"
        )
        _df = twist_desc.df if twist_desc.df is not None else pd.DataFrame()
        n = len(_df)
        handle = {
            "obj_key": obj_key,
            "twist_id": twist_id,
            "source_motl_id": None,
            "label": "Loaded twist",
            "nn_radius": twist_desc.nn_radius,
        }
        global_ref = _datapool.insert(_df, label="Loaded twist", id_column="qp_id")
        status = f"Twist loaded: {n:,} pairs. Radius: {radius_note}."
        return handle, global_ref, "tango-tab-twist", status, new_twist_id

    # ── Compute descriptor (5-slot) ───────────────────────────────────────────

    @app.callback(
        Output("tango-desc-registry-store", "data", allow_duplicate=True),
        *[Output(f"tango-desc-{i}-global-data-store", "data", allow_duplicate=True) for i in range(_DESC_SLOTS)],
        Output("tango-tabs", "active_tab", allow_duplicate=True),
        Output("tango-desc-status", "children"),
        Output("tango-desc-next-id", "data"),
        Input("tango-run-desc-btn", "n_clicks"),
        State("tango-twist-handle", "data"),
        State("tango-desc-registry-store", "data"),
        State("tango-desc-dropdown", "value"),
        State("tango-support-dropdown", "value"),
        State("tango-feat-dropdown", "value"),
        State({"type": "tango-support-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "tango-support-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "id"),
        State({"type": "tango-desc-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "tango-desc-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "id"),
        State({"type": "tango-feat-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "tango-feat-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "id"),
        State("tango-desc-next-id", "data"),
        prevent_initial_call=True,
    )
    def _compute_desc(
        n_clicks,
        twist_handle,
        desc_registry,
        selected_desc,
        selected_support,
        selected_features,
        supp_values, supp_ids,
        desc_values, desc_ids,
        feat_values, feat_ids,
        desc_next_id,
    ):
        _NU = (no_update,) * (1 + _DESC_SLOTS + 1 + 1 + 1)
        if not n_clicks:
            raise PreventUpdate
        if not twist_handle or not selected_desc:
            return *_NU[:7], "Select a descriptor and compute twist first.", *_NU[8:]
        twist_desc = _twist_objects.get(twist_handle["obj_key"])
        if twist_desc is None:
            return *_NU[:7], "Twist object expired — recompute.", *_NU[8:]
        desc_registry = dict(desc_registry or {})
        slot = next((i for i in range(_DESC_SLOTS) if str(i) not in desc_registry), None)
        if slot is None:
            return *_NU[:7], "All descriptor slots full — close one first.", *_NU[8:]
        twist_df = twist_desc.df
        twist_label = twist_handle.get("label", "Twist")
        desc_label = f"{twist_label} {selected_desc}"

        # Build support
        try:
            if not selected_support:
                support = twist_df
            else:
                support_kwargs = generate_kwargs(supp_ids, supp_values)
                supp_cls = get_classes_from_names(selected_support, "cryocat.analysis.tango")
                support = supp_cls(TwistDescriptor(input_twist=twist_df), **support_kwargs).support.df
        except Exception as exc:
            return *_NU[:7], f"Error: {exc}", *_NU[8:]

        # Determine descriptor class and call kwargs.
        # Descriptor constructors take the support DataFrame as their first
        # positional argument (named differently per class, e.g. "data", "df").
        # Use inspect to find the actual parameter name so invoke_operation can
        # call fn(**kwargs) correctly.
        import inspect as _inspect

        if selected_desc != "CustomDescriptor" and not selected_features:
            desc_cls_call = get_classes_from_names(selected_desc, "cryocat.analysis.tango")
            _init_params = list(_inspect.signature(desc_cls_call.__init__).parameters)
            _data_param = next((p for p in _init_params if p != "self"), "support")
            desc_call_kwargs = {_data_param: support, **generate_kwargs(desc_ids, desc_values)}
        else:
            if selected_desc != "CustomDescriptor":
                all_features = selected_features + _desc_feat_map.get(selected_desc, [])
                all_ids = desc_ids + feat_ids
                all_values = desc_values + feat_values
            else:
                all_features = selected_features or []
                all_ids = feat_ids
                all_values = feat_values
            feat_kwargs = []
            for feat_name in all_features:
                kw = next(
                    (generate_kwargs([d], [all_values[i]])
                     for i, d in enumerate(all_ids)
                     if d.get("cls_name") == feat_name),
                    {},
                )
                feat_kwargs.append(kw)
            desc_cls_call = CustomDescriptor
            _init_params = list(_inspect.signature(CustomDescriptor.__init__).parameters)
            _data_param = next((p for p in _init_params if p != "self"), "support")
            desc_call_kwargs = {
                _data_param: support,
                "feature_list": all_features,
                "feature_kwargs": feat_kwargs,
            }

        # Emit a provenance-correct log line manually: invoke_operation cannot
        # render a plain DataFrame kwarg as twist_N.df, so we build kwargs_src
        # ourselves using the twist provenance variable recorded earlier.
        import time as _time
        from cryocat.app import provenance as _prov
        from cryocat.app import session as _session
        from cryocat.app.event import call_event as _call_event
        from cryocat.app.logger import dash_logger as _dash_logger

        new_desc_next_id = (desc_next_id or 0) + 1
        desc_id = f"desc-{new_desc_next_id}"
        var = _prov.bind(desc_id)

        twist_id = (twist_handle or {}).get("twist_id")
        twist_var = _prov.var_for(twist_id) if twist_id else None
        data_src = (
            f"{twist_var}.df"
            if twist_var
            else f"None  # <DataFrame shape {support.shape}>"
        )

        kwargs_src: dict = {_data_param: data_src}
        for k, v in desc_call_kwargs.items():
            if k != _data_param:
                kwargs_src[k] = repr(v)

        fn_name = f"{desc_cls_call.__module__}.{desc_cls_call.__qualname__}"
        pane_call = (
            f"{desc_cls_call.__name__}"
            f"({', '.join(f'{k}={v}' for k, v in kwargs_src.items())})"
        )
        _dash_logger.write(f"▶ {pane_call}", source="cryocat")

        t0 = _time.monotonic()
        try:
            desc_obj = desc_cls_call(**desc_call_kwargs)
        except Exception as exc:
            duration = _time.monotonic() - t0
            _dash_logger.write(f"✗ {pane_call} — {type(exc).__name__}: {exc}", source="error")
            _session.emit(_call_event(
                fn_name, kwargs_src,
                status="error",
                assign_to=var,
                duration_s=duration,
                error={"type": type(exc).__name__, "msg": str(exc)},
            ))
            return *_NU[:7], f"Error: {exc}", *_NU[8:]

        duration = _time.monotonic() - t0
        _dash_logger.write(f"✓ {pane_call} ({duration:.3f} s)", source="cryocat")
        _session.emit(_call_event(
            fn_name, kwargs_src,
            status="ok",
            assign_to=var,
            duration_s=duration,
            result={"type": type(desc_obj).__name__},
        ))
        _prov.record(desc_id, _session.last_seq())

        desc_df = desc_obj.desc
        twist_sig = twist_handle.get("obj_key")
        desc_registry[str(slot)] = {"label": desc_label, "twist_sig": twist_sig, "stale": False}
        slot_global = [no_update] * _DESC_SLOTS
        slot_global[slot] = _datapool.insert(desc_df, label=desc_label, id_column="qp_id")
        status = f"Descriptor computed: {len(desc_df):,} rows."
        return desc_registry, *slot_global, f"tango-tab-desc-{slot}", status, new_desc_next_id

    # ── Mark descriptors stale when twist changes ─────────────────────────────

    @app.callback(
        Output("tango-desc-registry-store", "data", allow_duplicate=True),
        Input("tango-twist-handle", "data"),
        State("tango-desc-registry-store", "data"),
        prevent_initial_call=True,
    )
    def _mark_stale_on_twist_change(twist_handle, registry):
        registry = dict(registry or {})
        if not twist_handle or not registry:
            return no_update
        sig = twist_handle.get("obj_key")
        updated = False
        for k, v in registry.items():
            if v.get("twist_sig") != sig and not v.get("stale"):
                registry[k] = {**v, "stale": True}
                updated = True
        return registry if updated else no_update

    # ── Sync descriptor tab labels / disabled / stale badge ──────────────────

    @app.callback(
        *[Output(f"tango-tab-desc-{i}", "disabled") for i in range(_DESC_SLOTS)],
        *[Output(f"tango-tab-desc-{i}", "label") for i in range(_DESC_SLOTS)],
        *[Output(f"tango-desc-{i}-stale-badge", "children") for i in range(_DESC_SLOTS)],
        *[Output(f"tango-desc-{i}-stale-badge", "style") for i in range(_DESC_SLOTS)],
        Input("tango-desc-registry-store", "data"),
    )
    def _sync_desc_tabs(registry):
        registry = registry or {}
        disabled, labels, badge_children, badge_styles = [], [], [], []
        for i in range(_DESC_SLOTS):
            entry = registry.get(str(i))
            if entry:
                disabled.append(False)
                labels.append(entry.get("label", f"Descriptor {i + 1}"))
                stale = entry.get("stale", False)
                badge_children.append("⚠ stale" if stale else "")
                badge_styles.append({
                    "display": "inline" if stale else "none",
                    "color": styles.COLOR_MUTED,
                    "fontSize": styles.FONT_SM,
                    "marginRight": "0.5rem",
                })
            else:
                disabled.append(True)
                labels.append(f"Descriptor {i + 1}")
                badge_children.append("")
                badge_styles.append({"display": "none"})
        return *disabled, *labels, *badge_children, *badge_styles

    # ── Close descriptor slots ────────────────────────────────────────────────

    for _i in range(_DESC_SLOTS):
        def _make_close_cb(_slot=_i):
            @app.callback(
                Output("tango-desc-registry-store", "data", allow_duplicate=True),
                Output(f"tango-desc-{_slot}-global-data-store", "data", allow_duplicate=True),
                Input(f"tango-desc-{_slot}-close-btn", "n_clicks"),
                State("tango-desc-registry-store", "data"),
                prevent_initial_call=True,
            )
            def _close_slot(n_clicks, registry, _s=_slot):
                if not n_clicks:
                    raise PreventUpdate
                reg = dict(registry or {})
                reg.pop(str(_s), None)
                return reg, None
        _make_close_cb()

    # ── Per-slot diagnostics ──────────────────────────────────────────────────

    for _i in range(_DESC_SLOTS):
        def _make_diag_cb(_slot=_i):
            @app.callback(
                Output(f"tango-desc-{_slot}-diagnostics", "children"),
                Input(f"tango-desc-{_slot}-global-data-store", "data"),
                State(ids.GRAPH_SETTINGS_STORE, "data"),
                prevent_initial_call=True,
            )
            def _update_diagnostics(ref, settings):
                df = _datapool.resolve_df(ref)
                return _build_diagnostics(df, settings)
        _make_diag_cb()

    # ── Twist particle viewer ─────────────────────────────────────────────────

    @app.callback(
        Output("tango-twist-tv-data", "data"),
        Input("tango-twist-handle", "data"),
        prevent_initial_call=True,
    )
    def _wire_twist_viewer(handle):
        if not handle:
            raise PreventUpdate
        motl_id = handle.get("source_motl_id")
        if not motl_id:
            raise PreventUpdate
        return {"motl_id": motl_id, "rev": 0}

    register_viewer_callbacks(
        app, "tango-twist-tv",
        show_dual_graph=True,
        detailed_table="tango-twist-tabv-global-data-store",
        tabs_id=None,
        radius_store_id="tango-twist-handle",
        resolve_detail_df=_datapool.resolve_df,
    )

