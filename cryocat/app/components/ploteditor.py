"""Plot editor — spec-driven figure builder (PLOT_EDITOR.md).

Two mounts (W3) — one component body, two containers:
    modal    per-table; fixed source; full selection round-trip.
    tab      workbench; entry picker across both pools.

Spec shape (W1):
    {
        "chart":      "scatter",
        "source":     {"motl_id": "…"} | {"data_id": "…"} | None,
        "roles":      {"x": "score", "y": "z", "color": "class"},
        "traces":     [{"source": …, "label": "…", "color": "#…"}],
        "layout":     {"title": …, "xaxis": {"title": …}, …},
        "chart_opts": {"trendline": "ols", "opacity": 0.7, …},
    }

Styling levels (W5):
    defaults  ids.GRAPH_SETTINGS_STORE          (global; Graph Settings panel)
    figure    spec["layout"]                    (this plot; Layout panel)
    trace     spec["traces"][n]["color"], …     (per-overlay; Traces panel)
"""
from __future__ import annotations

import ast
import copy
import inspect
import json
import logging
import typing
from typing import Any

import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State, no_update, ctx, ALL
import dash_bootstrap_components as dbc

from cryocat.app import ids, styles
from cryocat.app.formgen import form_row, make_dropdown, section_divider
from cryocat.app.components import entrypicker
from cryocat.app.components.paletteloader import (
    get_palette_loader,
    register_palette_loader_callbacks,
    _DISCRETE_PRESETS,
    _CONTINUOUS_PRESETS,
)
from cryocat.app.components.graphsettings import (
    apply_settings_to_figure,
    style_figure,
    GRAPH_SETTINGS_DEFAULTS,
    _is_dark,
)

_log = logging.getLogger(__name__)

# ── Spec ─────────────────────────────────────────────────────────────────────

SPEC_DEFAULTS: dict = {
    "chart": None,
    "source": None,
    "roles": {},
    "traces": [],
    "layout": {},
    "chart_opts": {},
}


def build_spec(
    chart: str,
    source: dict | None,
    roles: dict,
    *,
    traces: list | None = None,
    layout: dict | None = None,
    chart_opts: dict | None = None,
    cluster_cols: list[str] | None = None,
) -> dict:
    """Return a fully-populated spec dict (W1)."""
    return {
        "chart": chart,
        "source": source,
        "roles": {k: v for k, v in roles.items() if v},
        "traces": traces or [],
        "layout": layout or {},
        "chart_opts": chart_opts or {},
        "cluster_cols": list(cluster_cols) if cluster_cols else [],
    }


# ── Chart configuration (W2) ─────────────────────────────────────────────────

# Params always excluded from the chart-specific options panel (W9).
# Roles and layout-level settings are handled by dedicated panels.
_ALWAYS_HIDE: frozenset[str] = frozenset({
    "data_frame", "hover_name", "text", "text_auto",
    "animation_frame", "animation_group", "category_orders", "labels",
    "color_discrete_map", "color_discrete_sequence",
    "color_continuous_scale", "color_continuous_midpoint", "range_color",
    "log_x", "log_y", "log_z",
    "range_x", "range_y", "range_z",
    "width", "height", "template", "title", "custom_data",
    "render_mode", "error_x", "error_x_minus", "error_y", "error_y_minus",
    "error_z", "error_z_minus", "base", "color_axis",
    "facet_row_spacing", "facet_col_spacing", "facet_col_wrap",
})

# Canonical set of px parameter names that accept a DataFrame column name.
# Anything in a px function's signature that matches these becomes a role dropdown
# in the Data panel; all other non-excluded params become chart-specific options.
_PX_COLUMN_PARAMS: frozenset[str] = frozenset({
    "x", "y", "z", "color", "size", "symbol",
    "line_dash", "line_group",
    "facet_col", "facet_row",
    "pattern_shape",
    "hover_data",    # P3: multi-column extra hover info
    "dimensions",    # P6: scatter_matrix, parallel_* charts
    # R2: extended role set — each parameter accepts a DataFrame column name
    "r", "theta",                    # polar charts
    "a", "b", "c",                   # ternary charts
    "lat", "lon", "locations",       # geo
    "names", "values",               # pie, funnel_area, treemap, sunburst, icicle
    "path",                          # treemap, sunburst, icicle — list of columns (hierarchy)
})

# Canonical display order for role dropdowns (determines layout + positional args).
_ROLE_ORDER: tuple[str, ...] = (
    # Primary axes / dimensions
    "x", "y", "z",
    "dimensions",                                # scatter_matrix, parallel_*
    # Polar
    "r", "theta",
    # Ternary
    "a", "b", "c",
    # Geographic / map
    "lat", "lon", "locations",
    # Proportional / hierarchical
    "names", "values",                           # pie, funnel_area; treemap, sunburst, icicle
    "path",                                      # treemap, sunburst, icicle — list of columns
    # Common encoding roles
    "color", "size", "symbol", "pattern_shape", "line_group", "line_dash",
    # Facets
    "facet_col", "facet_row",
    # Supplementary
    "hover_data",
)

# Roles whose dropdown is multi-select (value is always a list from the widget).
# Keep this set small and explicit — annotations do not distinguish multi from single.
_MULTI_ROLES: frozenset[str] = frozenset({"x", "y", "hover_data", "dimensions", "path"})


def _derive_roles(px_fn) -> list[str]:
    """Return the column-accepting role names for *px_fn* in canonical display order.

    Introspects the function signature and keeps only params that are both in
    ``_PX_COLUMN_PARAMS`` and present in the function — no hand-written lists.
    """
    params = set(inspect.signature(px_fn).parameters.keys())
    return [r for r in _ROLE_ORDER if r in params and r in _PX_COLUMN_PARAMS]


_CHART_CONFIG: dict[str, dict] = {
    "scatter": {
        "label": "Scatter",
        "fn": px.scatter,
        "required_roles": [],
        "roles": _derive_roles(px.scatter),
    },
    "line": {
        "label": "Line",
        "fn": px.line,
        "required_roles": ["x", "y"],
        "roles": _derive_roles(px.line),
    },
    "bar": {
        "label": "Bar",
        "fn": px.bar,
        "required_roles": ["x"],
        "roles": _derive_roles(px.bar),
    },
    "histogram": {
        "label": "Histogram",
        "fn": px.histogram,
        "required_roles": ["x"],
        "roles": _derive_roles(px.histogram),
    },
    "box": {
        "label": "Box",
        "fn": px.box,
        "required_roles": ["y"],
        "roles": _derive_roles(px.box),
    },
    "violin": {
        "label": "Violin",
        "fn": px.violin,
        "required_roles": ["y"],
        "roles": _derive_roles(px.violin),
    },
    "strip": {
        "label": "Strip",
        "fn": px.strip,
        "required_roles": ["x"],
        "roles": _derive_roles(px.strip),
    },
    "area": {
        "label": "Area",
        "fn": px.area,
        "required_roles": ["x", "y"],
        "roles": _derive_roles(px.area),
    },
    "ecdf": {
        "label": "ECDF",
        "fn": px.ecdf,
        "required_roles": ["x"],
        "roles": _derive_roles(px.ecdf),
    },
    "density_heatmap": {
        "label": "Density heatmap",
        "fn": px.density_heatmap,
        "required_roles": ["x", "y"],
        "roles": _derive_roles(px.density_heatmap),
    },
    "density_contour": {
        "label": "Density contour",
        "fn": px.density_contour,
        "required_roles": ["x", "y"],
        "roles": _derive_roles(px.density_contour),
    },
    "funnel": {
        "label": "Funnel",
        "fn": px.funnel,
        "required_roles": ["x", "y"],
        "roles": _derive_roles(px.funnel),
    },
    "scatter_3d": {
        "label": "Scatter 3D",
        "fn": px.scatter_3d,
        "required_roles": ["x", "y", "z"],
        "roles": _derive_roles(px.scatter_3d),
    },
    "line_3d": {
        "label": "Line 3D",
        "fn": px.line_3d,
        "required_roles": ["x", "y", "z"],
        "roles": _derive_roles(px.line_3d),
    },
    "scatter_matrix": {
        "label": "Scatter matrix",
        "fn": px.scatter_matrix,
        "required_roles": ["dimensions"],
        "roles": _derive_roles(px.scatter_matrix),
    },
    "parallel_coordinates": {
        "label": "Parallel coordinates",
        "fn": px.parallel_coordinates,
        "required_roles": ["dimensions"],
        "roles": _derive_roles(px.parallel_coordinates),
    },
    "parallel_categories": {
        "label": "Parallel categories",
        "fn": px.parallel_categories,
        "required_roles": ["dimensions"],
        "roles": _derive_roles(px.parallel_categories),
    },
    # ── Proportional ──────────────────────────────────────────────────────────
    "pie": {
        "label": "Pie",
        "fn": px.pie,
        "required_roles": ["names"],
        "roles": _derive_roles(px.pie),
    },
    "funnel_area": {
        "label": "Funnel area",
        "fn": px.funnel_area,
        "required_roles": ["names", "values"],
        "roles": _derive_roles(px.funnel_area),
    },
    # ── Hierarchical ──────────────────────────────────────────────────────────
    "treemap": {
        "label": "Treemap",
        "fn": px.treemap,
        "required_roles": ["path"],
        "roles": _derive_roles(px.treemap),
    },
    "sunburst": {
        "label": "Sunburst",
        "fn": px.sunburst,
        "required_roles": ["path"],
        "roles": _derive_roles(px.sunburst),
    },
    "icicle": {
        "label": "Icicle",
        "fn": px.icicle,
        "required_roles": ["path"],
        "roles": _derive_roles(px.icicle),
    },
    # ── Polar ─────────────────────────────────────────────────────────────────
    "scatter_polar": {
        "label": "Scatter polar",
        "fn": px.scatter_polar,
        "required_roles": ["r", "theta"],
        "roles": _derive_roles(px.scatter_polar),
    },
    "line_polar": {
        "label": "Line polar",
        "fn": px.line_polar,
        "required_roles": ["r", "theta"],
        "roles": _derive_roles(px.line_polar),
    },
    "bar_polar": {
        "label": "Bar polar",
        "fn": px.bar_polar,
        "required_roles": ["r", "theta"],
        "roles": _derive_roles(px.bar_polar),
    },
    # ── Ternary ───────────────────────────────────────────────────────────────
    "scatter_ternary": {
        "label": "Scatter ternary",
        "fn": px.scatter_ternary,
        "required_roles": ["a", "b", "c"],
        "roles": _derive_roles(px.scatter_ternary),
    },
    "line_ternary": {
        "label": "Line ternary",
        "fn": px.line_ternary,
        "required_roles": ["a", "b", "c"],
        "roles": _derive_roles(px.line_ternary),
    },
    # ── Geographic ────────────────────────────────────────────────────────────
    "scatter_geo": {
        "label": "Scatter geo",
        "fn": px.scatter_geo,
        "required_roles": ["lat", "lon"],
        "roles": _derive_roles(px.scatter_geo),
    },
    "line_geo": {
        "label": "Line geo",
        "fn": px.line_geo,
        "required_roles": ["lat", "lon"],
        "roles": _derive_roles(px.line_geo),
    },
}

# All role names that appear across any chart; determines which role rows to render.
# Order matches _ROLE_ORDER so positional args in _plot remain stable.
_ALL_ROLES: list[str] = [
    r for r in _ROLE_ORDER if any(r in cfg["roles"] for cfg in _CHART_CONFIG.values())
]

_CHART_OPTIONS: list[dict] = [{"label": v["label"], "value": k} for k, v in _CHART_CONFIG.items()]


# ── Chart-specific options (W9) ───────────────────────────────────────────────

# Per-parameter min/max/step overrides; the generic float path uses step=0.1.
# Add an entry here only when tighter bounds are semantically meaningful.
_PARAM_BOUNDS: dict[str, dict] = {
    "opacity": {"step": 0.05, "min": 0.0, "max": 1.0},
}


def _opt_widget(cid: dict, name: str, default: Any, annotation: Any) -> Any | None:
    """Build a widget for one Express parameter. Returns None for unsupported types."""
    origin = getattr(annotation, "__origin__", None)
    args   = getattr(annotation, "__args__", ())

    # Unwrap Optional[X] / X | None so Literal[…] | None is handled as Literal.
    non_none = tuple(a for a in args if a is not type(None))
    if non_none and origin is not typing.Literal and len(non_none) == 1:
        inner = non_none[0]
        i_origin = getattr(inner, "__origin__", None)
        i_args   = getattr(inner, "__args__", ())
        if i_origin is typing.Literal:
            origin, args = i_origin, i_args

    # Literal choices → dropdown
    if origin is typing.Literal or (
        args and all(isinstance(a, str) for a in args if a is not type(None))
    ):
        choices = [a for a in args if isinstance(a, str)]
        if choices:
            val = str(default) if isinstance(default, str) else None
            return make_dropdown(cid, choices, val, clearable=True)

    bounds   = _PARAM_BOUNDS.get(name, {})
    step_f   = bounds.get("step", 0.1)
    extra_kw = {k: v for k, v in bounds.items() if k != "step"}

    # Dispatch on default value type (bool before int: bool subclasses int).
    if isinstance(default, bool):
        return make_dropdown(cid, ["True", "False"],
                             "True" if default else "False", clearable=True)
    if isinstance(default, int):
        return dcc.Input(id=cid, type="number", value=default, step=1,
                         **extra_kw, style=styles.FORM_COMPACT_INPUT)
    if isinstance(default, float):
        return dcc.Input(id=cid, type="number", value=default,
                         step=step_f, **extra_kw, style=styles.FORM_COMPACT_INPUT)
    if isinstance(default, str):
        return dcc.Input(id=cid, type="text", value=default,
                         style=styles.FORM_COMPACT_INPUT)

    # None default: unwrap annotation and dispatch on inner scalar type.
    if default is None:
        all_inner = tuple(a for a in args if a is not type(None))
        has_bool  = any(a is bool  for a in all_inner)
        has_str   = any(a is str   for a in all_inner)
        has_int   = any(a is int   for a in all_inner)
        has_float = any(a is float for a in all_inner)
        if has_bool:
            return make_dropdown(cid, ["True", "False"], None, clearable=True)
        if has_str:
            return dcc.Input(id=cid, type="text", value="",
                             style=styles.FORM_COMPACT_INPUT)
        if has_int:
            return dcc.Input(id=cid, type="number", value=None, step=1,
                             **extra_kw, style=styles.FORM_COMPACT_INPUT)
        if has_float:
            return dcc.Input(id=cid, type="number", value=None,
                             step=step_f, **extra_kw, style=styles.FORM_COMPACT_INPUT)

    # Text fallback: blank = unset; typed value is parsed by ast.literal_eval at draw time.
    return dcc.Input(id=cid, type="text", value="", style=styles.FORM_COMPACT_INPUT)


def _express_opt_rows(chart_key: str, prefix: str) -> list:
    """Return form_row widgets for Express params not covered by roles (W9)."""
    if chart_key not in _CHART_CONFIG:
        return [html.Div("Select a chart type.", style=styles.HINT)]
    cfg = _CHART_CONFIG[chart_key]
    role_set = frozenset(cfg["roles"])
    exclude = _ALWAYS_HIDE | role_set

    sig = inspect.signature(cfg["fn"])
    try:
        hints = typing.get_type_hints(cfg["fn"]) if hasattr(cfg["fn"], "__annotations__") else {}
    except Exception:
        hints = {}
    rows = []
    for name, param in sig.parameters.items():
        if name in exclude or name.startswith("_"):
            continue
        default = param.default
        if default is inspect.Parameter.empty:
            continue
        ann = hints.get(name, param.annotation)
        cid = {"type": "pe-opt", "prefix": prefix, "chart": chart_key, "param": name}
        widget = _opt_widget(cid, name, default, ann)
        tip = (
            f"Plotly Express {cfg['fn'].__name__}(…, {name}=…). "
            "Hidden: data_frame, role params, and layout-level settings."
        )
        rows.append(form_row(name, widget, tip, truly_optional=True,
                             label_id=f"{prefix}-pe-{chart_key}-lbl-{name}"))
    if not rows:
        return [html.Div("No additional options for this chart type.", style=styles.HINT)]
    return rows


# ── Export helpers (W10) ─────────────────────────────────────────────────────

_UNIT_FACTORS: dict[str, float] = {
    "mm":   1 / 25.4 * 96,
    "cm":   10 / 25.4 * 96,
    "inch": 96.0,
    "px":   1.0,
}

_JOURNAL_PRESETS: list[dict] = [
    {"label": "Single column (85 mm)", "value": "single"},
    {"label": "1.5 column (114 mm)", "value": "half"},
    {"label": "Double column (178 mm)", "value": "double"},
]
_JOURNAL_WIDTH_MM = {"single": 85, "half": 114, "double": 178}

_EXPORT_FORMATS = ["png", "svg", "jpeg", "webp"]


def _to_px(value: float, unit: str) -> float:
    return value * _UNIT_FACTORS.get(unit, 1.0)


def _export_dims(w: float | None, h: float | None, unit: str, dpi: int) -> tuple[int, int, float]:
    """Return (width_px, height_px, scale) for export (W10)."""
    w_px = int(_to_px(w or 800, unit))
    h_px = int(_to_px(h or 600, unit))
    scale = dpi / 96 if unit != "px" else 1.0
    return w_px, h_px, scale


# ── Layout helpers ────────────────────────────────────────────────────────────

def _role_row(role: str, prefix: str) -> html.Div:
    multi = role in _MULTI_ROLES
    tip = f"Column{'s' if multi else ''} to map to the '{role}' role."
    return html.Div(
        id=f"{prefix}-pe-role-row-{role}",
        children=form_row(
            role,
            make_dropdown(f"{prefix}-pe-role-{role}", [], [] if multi else None,
                          clearable=True, multi=multi,
                          placeholder=f"Select column{'s' if multi else ''} for {role}…"),
            tip,
            truly_optional=True,
        ),
        style={"display": "none"},
    )


def _data_panel(prefix: str) -> html.Div:
    return html.Div([
        entrypicker.get_entry_picker(f"{prefix}-pe-src"),
        html.Div(style={"marginBottom": styles.FORM_ROW_GAP}),
        form_row(
            "chart_type",
            make_dropdown(
                f"{prefix}-pe-chart",
                _CHART_OPTIONS, None,
                clearable=True,
                placeholder="Select chart type…",
            ),
            "Chart type from Plotly Express (W2).",
        ),
        section_divider("Roles"),
        *[_role_row(r, prefix) for r in _ALL_ROLES],
        section_divider("Chart options"),
        html.Div(id=f"{prefix}-pe-opts-container", children=[
            html.Div(
                "Select a chart type to see its options.",
                id=f"{prefix}-pe-opts-placeholder",
                style=styles.HINT,
            ),
            *[
                html.Div(
                    _express_opt_rows(chart_key, prefix),
                    id=f"{prefix}-pe-opts-{chart_key}",
                    style={"display": "none"},
                )
                for chart_key in _CHART_CONFIG
            ],
        ]),
        section_divider("Overlays"),
        html.Div(
            id=f"{prefix}-pe-overlays-list",
            children=[html.Div("No overlay traces.", style=styles.HINT)],
        ),
        html.Div(style={"marginTop": "0.5rem"}),
        html.Div([
            dbc.Button("Plot", id=f"{prefix}-pe-plot-btn",
                       color=styles.BTN_PRIMARY, size="sm", className="me-1",
                       n_clicks=0, disabled=True),
            dcc.Loading(
                html.Span(id=f"{prefix}-pe-status", style=styles.HINT),
                type="circle",
                style={"display": "inline-flex", "alignItems": "center",
                       "marginLeft": "0.5rem"},
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "0.25rem"}),
        html.Div(id=f"{prefix}-pe-selection-note", style={"display": "none"}),
        dbc.Button("Add overlay trace", id=f"{prefix}-pe-add-overlay-btn",
                   color=styles.BTN_SECONDARY, size="sm"),
        dcc.Store(id=f"{prefix}-pe-overlays-store", data=[]),
    ])


def _layout_panel(prefix: str) -> html.Div:
    num_inp = {"type": "number", "style": styles.FORM_COMPACT_INPUT}
    return html.Div([
        html.Div(id=f"{prefix}-pe-slot-kind-note", style={**styles.HINT_SM, "marginBottom": styles.FORM_ROW_GAP}),
        form_row(
            "title",
            dcc.Input(id=f"{prefix}-pe-title", type="text", value="",
                      placeholder="Figure title…", style=styles.FORM_COMPACT_INPUT),
            "Figure title (figure level; overrides none).",
        ),
        section_divider("X axis"),
        form_row("x_title",
                 dcc.Input(id=f"{prefix}-pe-xaxis-title", type="text", value="",
                           placeholder="X axis label…", style=styles.FORM_COMPACT_INPUT),
                 "X axis title."),
        form_row("x_min",
                 dcc.Input(id=f"{prefix}-pe-xaxis-min", **num_inp),
                 "X axis minimum (leave blank for auto).", truly_optional=True),
        form_row("x_max",
                 dcc.Input(id=f"{prefix}-pe-xaxis-max", **num_inp),
                 "X axis maximum (leave blank for auto).", truly_optional=True),
        form_row("x_log",
                 dbc.Switch(id=f"{prefix}-pe-xaxis-log", value=False, label="Log scale"),
                 "Enable log scale on X axis."),
        section_divider("Y axis"),
        form_row("y_title",
                 dcc.Input(id=f"{prefix}-pe-yaxis-title", type="text", value="",
                           placeholder="Y axis label…", style=styles.FORM_COMPACT_INPUT),
                 "Y axis title."),
        form_row("y_min",
                 dcc.Input(id=f"{prefix}-pe-yaxis-min", **num_inp),
                 "Y axis minimum (leave blank for auto).", truly_optional=True),
        form_row("y_max",
                 dcc.Input(id=f"{prefix}-pe-yaxis-max", **num_inp),
                 "Y axis maximum (leave blank for auto).", truly_optional=True),
        form_row("y_log",
                 dbc.Switch(id=f"{prefix}-pe-yaxis-log", value=False, label="Log scale"),
                 "Enable log scale on Y axis."),
        section_divider("Legend"),
        form_row("legend_visible",
                 dbc.Switch(id=f"{prefix}-pe-legend-visible", value=True, label="Visible"),
                 "Show/hide the legend."),
        form_row("legend_orient",
                 make_dropdown(f"{prefix}-pe-legend-orient",
                               [{"label": "Vertical", "value": "v"},
                                {"label": "Horizontal", "value": "h"}],
                               "v", clearable=False),
                 "Legend orientation."),
        section_divider("Palette override"),
        html.Div([
            html.Label("Discrete", style=styles.FORM_LABEL),
            html.Div(
                get_palette_loader(f"{prefix}-pe-dis-pal", mode="discrete", allow_auto=True),
                style=styles.FORM_INPUT,
            ),
        ], style=styles.FORM_ROW),
        html.Div([
            html.Label("Continuous", style=styles.FORM_LABEL),
            html.Div(
                get_palette_loader(f"{prefix}-pe-con-pal", mode="continuous", allow_auto=True),
                style=styles.FORM_INPUT,
            ),
        ], style=styles.FORM_ROW),
        dcc.Store(id=f"{prefix}-pe-layout-store", data={}),
        html.Div([
            dbc.Button("Update layout", id=f"{prefix}-pe-update-layout-btn",
                       color=styles.BTN_SECONDARY, size="sm", className="me-1", n_clicks=0),
            html.Span(id=f"{prefix}-pe-layout-status",
                      style={**styles.HINT, "marginLeft": "0.5rem"}),
        ], style={"display": "flex", "alignItems": "center", "marginTop": styles.SECTION_GAP}),
    ])


def _export_panel(prefix: str) -> html.Div:
    return html.Div([
        form_row("format",
                 make_dropdown(f"{prefix}-pe-exp-fmt",
                               _EXPORT_FORMATS, "png", clearable=False),
                 "Export file format. SVG is vector and preferred for publication."),
        section_divider("Journal presets (W10)"),
        form_row("preset",
                 make_dropdown(f"{prefix}-pe-exp-preset",
                               _JOURNAL_PRESETS, None, clearable=True,
                               placeholder="Quick journal width…"),
                 "Preset column widths from common journal specifications."),
        section_divider("Size"),
        form_row("unit",
                 make_dropdown(f"{prefix}-pe-exp-unit",
                               [{"label": u, "value": u} for u in ("mm", "cm", "inch", "px")],
                               "mm", clearable=False),
                 "Physical unit for width/height. SVG uses px at 96 dpi internally."),
        form_row("width",
                 dcc.Input(id=f"{prefix}-pe-exp-width", type="number", value=85,
                           min=1, style=styles.FORM_COMPACT_INPUT),
                 "Export width in the chosen unit."),
        form_row("height",
                 dcc.Input(id=f"{prefix}-pe-exp-height", type="number", value=65,
                           min=1, style=styles.FORM_COMPACT_INPUT),
                 "Export height in the chosen unit."),
        form_row("dpi",
                 dcc.Input(id=f"{prefix}-pe-exp-dpi", type="number", value=300,
                           min=72, max=1200, step=1, style=styles.FORM_COMPACT_INPUT),
                 "Target DPI (raster only; ignored for SVG)."),
        html.Div(id=f"{prefix}-pe-exp-dims-hint", style=styles.HINT),
        section_divider("Options"),
        form_row("transparent",
                 dbc.Switch(id=f"{prefix}-pe-exp-transparent", value=False,
                            label="Transparent background"),
                 "Transparent background. Not available for JPEG."),
        form_row("export_font_size",
                 dcc.Input(id=f"{prefix}-pe-exp-font-size", type="number", value=14,
                           min=6, max=48, style=styles.FORM_COMPACT_INPUT),
                 "Font size for export (print typically needs larger font than screen)."),
        dcc.Download(id=f"{prefix}-pe-download"),
        html.Div(style={"marginTop": "0.5rem"}),
        dbc.Button("Export figure", id=f"{prefix}-pe-export-btn",
                   color=styles.BTN_PRIMARY, size="sm"),
        html.Div(id=f"{prefix}-pe-export-status", style=styles.HINT),
    ])


def _defaults_panel(prefix: str) -> html.Div:
    """Graph Settings absorbed as the Defaults panel (W11).

    Uses {prefix}-def-* IDs to avoid conflicts with the existing global
    graph-settings-modal.  An Apply callback writes to GRAPH_SETTINGS_STORE.
    """
    return html.Div([
        html.P(
            "These settings apply to every figure in the app (defaults level, W5). "
            "Figure-level overrides are in the Layout tab.",
            style=styles.HINT,
        ),
        form_row("font_family",
                 make_dropdown(f"{prefix}-def-font-family",
                               ["Arial", "Helvetica", "Courier New", "Times New Roman", "Verdana"],
                               "Arial", clearable=False),
                 "Default font family."),
        form_row("font_size",
                 dcc.Input(id=f"{prefix}-def-font-size", type="number",
                           value=12, min=6, max=30, step=1,
                           style=styles.FORM_COMPACT_INPUT),
                 "Default font size (pt)."),
        form_row("marker_size",
                 dcc.Input(id=f"{prefix}-def-marker-size", type="number",
                           value=6, min=1, max=30, step=1,
                           style=styles.FORM_COMPACT_INPUT),
                 "Default marker size."),
        form_row("line_width",
                 dcc.Input(id=f"{prefix}-def-line-width", type="number",
                           value=2, min=0.5, max=10, step=0.5,
                           style=styles.FORM_COMPACT_INPUT),
                 "Default line width."),
        form_row("line_dash",
                 make_dropdown(f"{prefix}-def-line-dash",
                               [{"label": "Solid", "value": "solid"},
                                {"label": "Dashed", "value": "dash"},
                                {"label": "Dotted", "value": "dot"},
                                {"label": "Dash-dot", "value": "dashdot"}],
                               "solid", clearable=False),
                 "Default line dash style.",
                 label_id=f"{prefix}-def-lbl-line-dash"),
        section_divider("Palettes"),
        html.Div([
            html.Label("Discrete", style=styles.FORM_LABEL),
            html.Div(
                get_palette_loader(f"{prefix}-def-dis-pal", mode="discrete",
                                   default=GRAPH_SETTINGS_DEFAULTS["discrete_palette"]),
                style=styles.FORM_INPUT,
            ),
        ], style=styles.FORM_ROW),
        html.Div([
            html.Label("Continuous", style=styles.FORM_LABEL),
            html.Div(
                get_palette_loader(f"{prefix}-def-con-pal", mode="continuous",
                                   default=GRAPH_SETTINGS_DEFAULTS["continuous_palette"]),
                style=styles.FORM_INPUT,
            ),
        ], style=styles.FORM_ROW),
        form_row("bg_color",
                 make_dropdown(f"{prefix}-def-bg-color",
                               [{"label": "White", "value": "white"},
                                {"label": "Light grey", "value": "#f5f5f5"},
                                {"label": "Dark", "value": "#1e1e1e"}],
                               "white", clearable=False),
                 "Default background colour."),
        html.Div(style={"marginTop": "0.5rem"}),
        html.Div([
            dbc.Button("Apply from now on", id=f"{prefix}-def-apply-btn",
                       color=styles.BTN_PRIMARY, size="sm", n_clicks=0),
            dbc.Button("Apply also for existing", id=f"{prefix}-def-apply-existing-btn",
                       color=styles.BTN_PRIMARY, size="sm", n_clicks=0,
                       style={"marginLeft": "0.5rem"}),
            html.Span(id=f"{prefix}-def-status",
                      style={"marginLeft": "1rem", **styles.HINT}),
        ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "0.25rem"}),
    ])


# ── Public layout API ─────────────────────────────────────────────────────────

def get_plot_editor_sidebar(prefix: str) -> list:
    """Sidebar-mode plot editor: panels only, no graph area (C1).

    Returns a list of elements to embed directly in a page sidebar.
    Register callbacks with ``register_plot_editor_callbacks`` and provide
    your own ``_plot`` / ``_update_layout`` callbacks in the page.
    """
    from cryocat.app.pageshell import sidebar_accordion
    return [
        dcc.Store(id=f"{prefix}-pe-spec-store", data=copy.deepcopy(SPEC_DEFAULTS)),
        dcc.Store(id=f"{prefix}-pe-selected-ids", data=[]),
        dcc.Store(id=f"{prefix}-pe-ext-figure", data=None),
        sidebar_accordion([
            dbc.AccordionItem(_data_panel(prefix), title="Data", item_id="data"),
            dbc.AccordionItem(_layout_panel(prefix), title="Layout", item_id="layout"),
            dbc.AccordionItem(_export_panel(prefix), title="Export", item_id="export"),
            dbc.AccordionItem(_defaults_panel(prefix), title="Defaults", item_id="defaults"),
        ], active_item=["data"]),
    ]



# ── Resolve helpers ───────────────────────────────────────────────────────────

def _resolve_df(src_ref: dict | None, pool_resolve_df, dp_resolve_df) -> Any:
    """Resolve a source ref to a DataFrame, or None if unavailable."""
    if not src_ref:
        return None
    try:
        if "motl_id" in src_ref and pool_resolve_df:
            return pool_resolve_df(src_ref)
        if "data_id" in src_ref and dp_resolve_df:
            return dp_resolve_df(src_ref)
    except Exception:
        pass
    return None


def _detect_id_column(df) -> str | None:
    """Return the first plausible row-identity column, or None."""
    import pandas as pd
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    for candidate in ("tomo_id", "object_id", "id", "index"):
        if candidate in df.columns:
            return candidate
    return None


# ── Figure builders ───────────────────────────────────────────────────────────

_NOISE_COLOR = "rgba(128,128,128,0.45)"


def _eff_settings(
    settings: dict | None,
    fig_palette_dis: str | None,
    fig_palette_con: str | None,
) -> dict:
    """Merge global graph settings with per-figure palette overrides (BB4)."""
    eff = dict(settings or GRAPH_SETTINGS_DEFAULTS)
    # Always apply palette in figure styling regardless of palette_is_user_set flag.
    # That flag governs URL-based auto-switching only, not figure rendering.
    eff["palette_is_user_set"] = True
    if fig_palette_dis:
        eff["discrete_palette"] = fig_palette_dis
    if fig_palette_con:
        eff["continuous_palette"] = fig_palette_con
    return eff


def _build_figure(
    chart: str,
    df,
    roles: dict,
    chart_opts: dict,
    id_col: str | None,
    settings: dict,
    layout_spec: dict,
    fig_palette_dis: str | None = None,
    fig_palette_con: str | None = None,
    cluster_cols: list[str] | None = None,
) -> go.Figure | None:
    """Build a styled Plotly figure from a spec (W5 three-level styling)."""
    import pandas as _pd
    cfg = _CHART_CONFIG.get(chart)
    if cfg is None or df is None:
        return None

    # Build clean_roles: filter to this chart's supported roles and validate columns.
    # Handle multi-select list values (P3): unwrap single-item lists for x/y only.
    chart_role_set = set(cfg["roles"])
    clean_roles: dict = {}
    for k, v in roles.items():
        if k not in chart_role_set:
            continue  # role not supported by this chart type; skip stale values
        if isinstance(v, list):
            valid = [c for c in v if c in df.columns]
            if not valid:
                continue
            # Keep as list for multi-always roles; unwrap single item for x/y to avoid wide mode
            clean_roles[k] = valid[0] if (len(valid) == 1 and k in {"x", "y"}) else valid
        elif v and v in df.columns:
            clean_roles[k] = v

    opts = {k: v for k, v in chart_opts.items() if v is not None and v != ""}

    # Inject customdata for selection round-trip (W7); not all chart types accept it (P2 fix)
    sig_params = inspect.signature(cfg["fn"]).parameters
    if id_col and id_col in df.columns and "custom_data" in sig_params:
        opts["custom_data"] = [id_col]

    # Cluster-column rendering: when the color role is a known cluster column, force
    # categorical rendering with -1 shown as "noise" in neutral grey.
    color_col = clean_roles.get("color") if isinstance(clean_roles.get("color"), str) else None
    _is_cluster_color = bool(cluster_cols and color_col and color_col in cluster_cols)
    if _is_cluster_color:
        df = df.copy()
        def _to_cat(v):
            try:
                return "noise" if int(v) == -1 else str(int(v))
            except (TypeError, ValueError):
                return "noise"
        df[color_col] = df[color_col].apply(_to_cat)
        opts["color_discrete_map"] = {"noise": _NOISE_COLOR}

    _log.debug(
        "px.%s(df, %s)",
        cfg["fn"].__name__,
        ", ".join(f"{k}={v!r}" for k, v in {**clean_roles, **opts}.items()),
    )
    try:
        fig = cfg["fn"](df, **clean_roles, **opts)
    except Exception:
        _log.exception("px.%s raised", cfg["fn"].__name__)
        return None

    # Levels 1+2: global defaults merged with per-figure palette overrides (W5).
    eff_settings = _eff_settings(settings, fig_palette_dis, fig_palette_con)

    fig_dict = style_figure(fig, eff_settings)

    # Re-pin noise colour AFTER style_figure, not before.  style_figure calls
    # apply_settings_to_figure which, when palette_is_user_set=True, clears
    # existing scalar marker colours and overwrites them from the palette —
    # including the "noise" trace colour set via color_discrete_map before the
    # px call.  color_discrete_map alone is not sufficient; the re-pin here is
    # the only guarantee that noise stays grey.  Moving this block above
    # style_figure would silently revert noise to the palette colour.
    if _is_cluster_color:
        for trace in fig_dict.get("data", []):
            if trace.get("name") == "noise":
                trace.setdefault("marker", {})["color"] = _NOISE_COLOR

    # Level 2: figure layout from spec (W5 level 2)
    eff_layout: dict = {}
    if layout_spec.get("title"):
        eff_layout["title"] = {"text": layout_spec["title"]}
    xaxis: dict = {}
    if layout_spec.get("xaxis_title"):
        xaxis["title"] = layout_spec["xaxis_title"]
    if layout_spec.get("xaxis_min") is not None or layout_spec.get("xaxis_max") is not None:
        xaxis["range"] = [layout_spec.get("xaxis_min"), layout_spec.get("xaxis_max")]
    if layout_spec.get("xaxis_log"):
        xaxis["type"] = "log"
    if xaxis:
        eff_layout.setdefault("xaxis", {}).update(xaxis)
    yaxis: dict = {}
    if layout_spec.get("yaxis_title"):
        yaxis["title"] = layout_spec["yaxis_title"]
    if layout_spec.get("yaxis_min") is not None or layout_spec.get("yaxis_max") is not None:
        yaxis["range"] = [layout_spec.get("yaxis_min"), layout_spec.get("yaxis_max")]
    if layout_spec.get("yaxis_log"):
        yaxis["type"] = "log"
    if yaxis:
        eff_layout.setdefault("yaxis", {}).update(yaxis)
    legend_vis = layout_spec.get("legend_visible", True)
    legend_orient = layout_spec.get("legend_orient", "v")
    eff_layout["showlegend"] = legend_vis
    if legend_vis:
        eff_layout.setdefault("legend", {})["orientation"] = legend_orient
    if eff_layout:
        fig_dict.setdefault("layout", {}).update(eff_layout)

    return go.Figure(fig_dict)


def _apply_layout_only(existing_fig: dict, layout_spec: dict, settings: dict,
                        fig_palette_dis: str | None, fig_palette_con: str | None) -> dict:
    """Re-apply layout settings to an existing figure without re-fetching data (W6)."""
    fig_dict = copy.deepcopy(existing_fig)
    eff_settings = _eff_settings(settings, fig_palette_dis, fig_palette_con)
    fig_dict = apply_settings_to_figure(fig_dict, eff_settings)

    eff_layout: dict = {}
    if layout_spec.get("title"):
        eff_layout["title"] = {"text": layout_spec["title"]}
    xaxis: dict = {}
    if layout_spec.get("xaxis_title"):
        xaxis["title"] = layout_spec["xaxis_title"]
    if layout_spec.get("xaxis_min") is not None or layout_spec.get("xaxis_max") is not None:
        xaxis["range"] = [layout_spec.get("xaxis_min"), layout_spec.get("xaxis_max")]
    if layout_spec.get("xaxis_log"):
        xaxis["type"] = "log"
    if xaxis:
        eff_layout.setdefault("xaxis", {}).update(xaxis)
    yaxis: dict = {}
    if layout_spec.get("yaxis_title"):
        yaxis["title"] = layout_spec["yaxis_title"]
    if layout_spec.get("yaxis_min") is not None or layout_spec.get("yaxis_max") is not None:
        yaxis["range"] = [layout_spec.get("yaxis_min"), layout_spec.get("yaxis_max")]
    if layout_spec.get("yaxis_log"):
        yaxis["type"] = "log"
    if yaxis:
        eff_layout.setdefault("yaxis", {}).update(yaxis)
    legend_vis = layout_spec.get("legend_visible", True)
    eff_layout["showlegend"] = legend_vis
    if legend_vis:
        eff_layout.setdefault("legend", {})["orientation"] = layout_spec.get("legend_orient", "v")
    if eff_layout:
        fig_dict.setdefault("layout", {}).update(eff_layout)
    return fig_dict


# ── Callback registration ─────────────────────────────────────────────────────

def register_plot_editor_callbacks(
    app,
    prefix: str,
    *,
    pool_resolve_df=None,
    dp_resolve_df=None,
    connected_grid_id: str | None = None,
    fixed_source_store_id: str | None = None,
    settings_store_id: str | None = None,
) -> None:
    """Register all plot editor callbacks for *prefix*.

    Parameters
    ----------
    pool_resolve_df
        ``pool.resolve_df`` for motl pool sources (required for modal mount).
    dp_resolve_df
        ``datapool.resolve_df`` for data pool sources.
    connected_grid_id
        When given, selection in the plot syncs back to this AG-Grid (W7).
    fixed_source_store_id
        When given (modal mount), the source picker is hidden and this store
        provides the source ref directly.  When None (workbench mount), the
        entry picker is active.
    settings_store_id
        ID of the global graph-settings ``dcc.Store``.  When provided, the
        Auto swatch in the per-plot palette dropdowns reflects the current
        effective default palette in real time.
    """
    # Register palette loader callbacks for the Layout panel overlays (Auto-capable).
    register_palette_loader_callbacks(app, f"{prefix}-pe-dis-pal", mode="discrete",
                                      settings_store_id=settings_store_id)
    register_palette_loader_callbacks(app, f"{prefix}-pe-con-pal", mode="continuous",
                                      settings_store_id=settings_store_id)
    # Defaults panel palette loaders — no Auto (they define what Auto resolves to).
    register_palette_loader_callbacks(app, f"{prefix}-def-dis-pal", mode="discrete")
    register_palette_loader_callbacks(app, f"{prefix}-def-con-pal", mode="continuous")
    # Entry picker (workbench mount only).
    if not fixed_source_store_id:
        entrypicker.register_entry_picker_callbacks(app, f"{prefix}-pe-src")

    # ── Column population ──────────────────────────────────────────────────────

    # Source ref: workbench reads from entry picker store; modal reads from fixed store.
    src_store = fixed_source_store_id or f"{prefix}-pe-src-ref"

    @app.callback(
        *[Output(f"{prefix}-pe-role-{r}", "options") for r in _ALL_ROLES],
        Input(src_store, "data"),
        prevent_initial_call=True,
    )
    def _populate_columns(src_ref):
        df = _resolve_df(src_ref, pool_resolve_df, dp_resolve_df)
        if df is None:
            empty = [no_update] * len(_ALL_ROLES)
            return empty
        import pandas as pd
        if not isinstance(df, pd.DataFrame) or df.empty:
            return [[] for _ in _ALL_ROLES]
        cols = [{"label": c, "value": c} for c in df.columns]
        return [cols for _ in _ALL_ROLES]

    # ── Role show/hide ─────────────────────────────────────────────────────────

    _show = {"display": "block"}
    _hide = {"display": "none"}

    @app.callback(
        *[Output(f"{prefix}-pe-role-row-{r}", "style") for r in _ALL_ROLES],
        Input(f"{prefix}-pe-chart", "value"),
        prevent_initial_call=True,
    )
    def _toggle_roles(chart):
        active = set(_CHART_CONFIG.get(chart, {}).get("roles", []))
        return [_show if r in active else _hide for r in _ALL_ROLES]

    # ── Plot button guard (Q2: enforce required_roles) ─────────────────────────

    @app.callback(
        Output(f"{prefix}-pe-plot-btn", "disabled"),
        Output(f"{prefix}-pe-plot-btn", "title"),
        Input(src_store, "data"),
        Input(f"{prefix}-pe-chart", "value"),
        *[Input(f"{prefix}-pe-role-{r}", "value") for r in _ALL_ROLES],
        State(f"{prefix}-pe-ext-figure", "data"),
    )
    def _guard_plot_btn(src_ref, chart, *args):
        *role_values, ext_fig = args
        if ext_fig:
            return False, ""
        if not (src_ref and chart):
            return True, ""
        cfg = _CHART_CONFIG.get(chart, {})
        role_map = dict(zip(_ALL_ROLES, role_values))
        missing = [r for r in cfg.get("required_roles", []) if not role_map.get(r)]
        if missing:
            return True, f"Missing required role(s): {', '.join(missing)}"
        return False, ""

    # ── Wide-mode color disable (Q3) ───────────────────────────────────────────

    @app.callback(
        Output(f"{prefix}-pe-role-color", "disabled"),
        Output(f"{prefix}-pe-role-color", "placeholder"),
        Input(f"{prefix}-pe-role-x", "value"),
        Input(f"{prefix}-pe-role-y", "value"),
        prevent_initial_call=True,
    )
    def _wide_mode_color(x_val, y_val):
        wide = (isinstance(x_val, list) and len(x_val) > 1) or \
               (isinstance(y_val, list) and len(y_val) > 1)
        if wide:
            return True, "Wide mode — coloured by column name"
        return False, "Select column for color…"

    # ── Chart-specific options toggle (Q1: pre-built, never rebuilt) ───────────

    @app.callback(
        Output(f"{prefix}-pe-opts-placeholder", "style"),
        *[Output(f"{prefix}-pe-opts-{k}", "style") for k in _CHART_CONFIG],
        Input(f"{prefix}-pe-chart", "value"),
    )
    def _toggle_chart_opts(chart):
        ph_style = _hide if chart else styles.HINT
        return (ph_style, *[(_show if k == chart else _hide) for k in _CHART_CONFIG])

    # ── Selection support note (Q4) ───────────────────────────────────────────────

    @app.callback(
        Output(f"{prefix}-pe-selection-note", "children"),
        Output(f"{prefix}-pe-selection-note", "style"),
        Input(f"{prefix}-pe-chart", "value"),
        prevent_initial_call=True,
    )
    def _selection_support_note(chart):
        if not chart:
            return "", {"display": "none"}
        cfg = _CHART_CONFIG.get(chart)
        if cfg and "custom_data" not in inspect.signature(cfg["fn"]).parameters:
            return (
                "Point selection is not available for this chart type — aggregated charts do not carry row identifiers.",
                {**styles.HINT_SM, "marginTop": "0.25rem"},
            )
        return "", {"display": "none"}

    # ── Layout store update ────────────────────────────────────────────────────

    @app.callback(
        Output(f"{prefix}-pe-layout-store", "data"),
        Input(f"{prefix}-pe-title", "value"),
        Input(f"{prefix}-pe-xaxis-title", "value"),
        Input(f"{prefix}-pe-xaxis-min", "value"),
        Input(f"{prefix}-pe-xaxis-max", "value"),
        Input(f"{prefix}-pe-xaxis-log", "value"),
        Input(f"{prefix}-pe-yaxis-title", "value"),
        Input(f"{prefix}-pe-yaxis-min", "value"),
        Input(f"{prefix}-pe-yaxis-max", "value"),
        Input(f"{prefix}-pe-yaxis-log", "value"),
        Input(f"{prefix}-pe-legend-visible", "value"),
        Input(f"{prefix}-pe-legend-orient", "value"),
        prevent_initial_call=True,
    )
    def _collect_layout(title, xt, xmin, xmax, xlog, yt, ymin, ymax, ylog,
                        leg_vis, leg_or):
        return {
            "title": title or "",
            "xaxis_title": xt or "",
            "xaxis_min": xmin,
            "xaxis_max": xmax,
            "xaxis_log": bool(xlog),
            "yaxis_title": yt or "",
            "yaxis_min": ymin,
            "yaxis_max": ymax,
            "yaxis_log": bool(ylog),
            "legend_visible": bool(leg_vis) if leg_vis is not None else True,
            "legend_orient": leg_or or "v",
        }

    # ── Export (W10) ───────────────────────────────────────────────────────────

    @app.callback(
        Output(f"{prefix}-pe-exp-dims-hint", "children"),
        Input(f"{prefix}-pe-exp-width", "value"),
        Input(f"{prefix}-pe-exp-height", "value"),
        Input(f"{prefix}-pe-exp-unit", "value"),
        Input(f"{prefix}-pe-exp-dpi", "value"),
        Input(f"{prefix}-pe-exp-fmt", "value"),
        prevent_initial_call=True,
    )
    def _update_dims_hint(w, h, unit, dpi, fmt):
        if not w or not h or not unit:
            return ""
        w_px, h_px, scale = _export_dims(w, h, unit, dpi or 96)
        dpi_note = f" · scale {scale:.2f}×" if fmt != "svg" else " · vector (no scale)"
        return f"→ {w_px} × {h_px} px{dpi_note}"

    @app.callback(
        Output(f"{prefix}-pe-exp-width", "value", allow_duplicate=True),
        Output(f"{prefix}-pe-exp-height", "value", allow_duplicate=True),
        Output(f"{prefix}-pe-exp-unit", "value", allow_duplicate=True),
        Input(f"{prefix}-pe-exp-preset", "value"),
        prevent_initial_call=True,
    )
    def _apply_preset(preset):
        if not preset:
            return no_update, no_update, no_update
        w_mm = _JOURNAL_WIDTH_MM.get(preset, 85)
        return w_mm, round(w_mm * 0.75), "mm"

    @app.callback(
        Output(f"{prefix}-pe-exp-dpi", "disabled"),
        Output(f"{prefix}-pe-exp-transparent", "disabled"),
        Input(f"{prefix}-pe-exp-fmt", "value"),
        prevent_initial_call=True,
    )
    def _toggle_export_controls(fmt):
        dpi_dis = fmt == "svg"
        trans_dis = fmt == "jpeg"
        return dpi_dis, trans_dis

    @app.callback(
        Output(f"{prefix}-pe-download", "data"),
        Output(f"{prefix}-pe-export-status", "children"),
        Input(f"{prefix}-pe-export-btn", "n_clicks"),
        State(f"{prefix}-pe-graph", "figure"),
        State(f"{prefix}-pe-exp-fmt", "value"),
        State(f"{prefix}-pe-exp-width", "value"),
        State(f"{prefix}-pe-exp-height", "value"),
        State(f"{prefix}-pe-exp-unit", "value"),
        State(f"{prefix}-pe-exp-dpi", "value"),
        State(f"{prefix}-pe-exp-transparent", "value"),
        State(f"{prefix}-pe-exp-font-size", "value"),
        prevent_initial_call=True,
    )
    def _export(_n, existing, fmt, w, h, unit, dpi, transparent, font_size):
        if not existing:
            return no_update, "No figure to export."

        w_px, h_px, scale = _export_dims(w or 85, h or 65, unit or "mm", dpi or 300)
        fig_copy = copy.deepcopy(existing)
        layout = fig_copy.setdefault("layout", {})
        layout["width"] = w_px
        layout["height"] = h_px
        if font_size:
            layout.setdefault("font", {})["size"] = font_size
        if transparent and fmt != "jpeg":
            layout["paper_bgcolor"] = "rgba(0,0,0,0)"
            layout["plot_bgcolor"] = "rgba(0,0,0,0)"

        try:
            fig = go.Figure(fig_copy)
            if fmt == "svg":
                content = fig.to_image(format="svg").decode("utf-8")
                return {"content": content, "filename": f"figure.{fmt}", "type": "text/plain"}, "Exported."
            else:
                import base64
                img_bytes = fig.to_image(format=fmt, scale=scale)
                encoded = base64.b64encode(img_bytes).decode()
                return {"base64": True, "content": encoded,
                        "filename": f"figure.{fmt}", "type": f"image/{fmt}"}, "Exported."
        except Exception as exc:
            return no_update, f"Export failed: {exc}"

    # ── Defaults panel — write to GRAPH_SETTINGS_STORE (W11) ──────────────────

    @app.callback(
        Output(ids.GRAPH_SETTINGS_STORE, "data", allow_duplicate=True),
        Output(f"{prefix}-def-status", "children"),
        Input(f"{prefix}-def-apply-btn", "n_clicks"),
        State(f"{prefix}-def-font-family", "value"),
        State(f"{prefix}-def-font-size", "value"),
        State(f"{prefix}-def-marker-size", "value"),
        State(f"{prefix}-def-line-width", "value"),
        State(f"{prefix}-def-line-dash", "value"),
        State(f"{prefix}-def-dis-pal-value", "data"),
        State(f"{prefix}-def-con-pal-value", "data"),
        State(f"{prefix}-def-bg-color", "value"),
        prevent_initial_call=True,
    )
    def _save_defaults(_, font_family, font_size, marker_size, line_width, line_dash,
                       dis_pal, con_pal, bg_color):
        return {
            "font_family": font_family or GRAPH_SETTINGS_DEFAULTS["font_family"],
            "font_size": font_size or GRAPH_SETTINGS_DEFAULTS["font_size"],
            "marker_size": marker_size or GRAPH_SETTINGS_DEFAULTS["marker_size"],
            "line_width": line_width or GRAPH_SETTINGS_DEFAULTS["line_width"],
            "line_dash": line_dash or GRAPH_SETTINGS_DEFAULTS["line_dash"],
            "discrete_palette": dis_pal or GRAPH_SETTINGS_DEFAULTS["discrete_palette"],
            "continuous_palette": con_pal or GRAPH_SETTINGS_DEFAULTS["continuous_palette"],
            "bg_color": bg_color or GRAPH_SETTINGS_DEFAULTS["bg_color"],
            "palette_is_user_set": True,
        }, "Applied."

    # ── Selection round-trip (W7) ──────────────────────────────────────────────

    @app.callback(
        Output(f"{prefix}-pe-selected-ids", "data"),
        Input(f"{prefix}-pe-graph", "selectedData"),
        State(src_store, "data"),
        prevent_initial_call=True,
    )
    def _on_select(selected_data, src_ref):
        if not selected_data or not selected_data.get("points"):
            return []
        ids_list = []
        for pt in selected_data["points"]:
            cd = pt.get("customdata")
            if cd:
                ids_list.append(cd[0] if isinstance(cd, (list, tuple)) else cd)
        return ids_list

    # ── Overlay trace add (W4) ─────────────────────────────────────────────────

    @app.callback(
        Output(f"{prefix}-pe-overlays-store", "data"),
        Output(f"{prefix}-pe-overlays-list", "children"),
        Input(f"{prefix}-pe-add-overlay-btn", "n_clicks"),
        State(f"{prefix}-pe-overlays-store", "data"),
        prevent_initial_call=True,
    )
    def _add_overlay_trace(_n, overlays):
        overlays = list(overlays or [])
        idx = len(overlays)
        overlays.append({"source": None, "label": f"Trace {idx + 2}", "color": None})
        items = [
            html.Div(
                f"Trace {i + 2}: {t.get('label', '?')} (source not yet set)",
                style=styles.HINT,
            )
            for i, t in enumerate(overlays)
        ]
        return overlays, items or [html.Div("No overlay traces.", style=styles.HINT)]


# ── Overlay helper (called from _plot) ───────────────────────────────────────

class OverlaySourceMissing(LookupError):
    """Raised when an overlay's source ref is set but the source is not in the pool."""
    def __init__(self, src_ref: dict):
        self.src_ref = src_ref
        name = src_ref.get("motl_id") or src_ref.get("data_id") or str(src_ref)
        super().__init__(name)


def _add_overlay(
    fig: go.Figure,
    trace_cfg: dict,
    pool_resolve_df,
    dp_resolve_df,
    primary_roles: dict,
    chart: str,
    settings: dict,
) -> None:
    """Add one overlay trace to *fig* in-place (W4).

    Raises OverlaySourceMissing when the source ref is set but the source
    cannot be resolved.  Any other exception indicates a bug and is not caught.
    """
    src_ref = trace_cfg.get("source")
    label = trace_cfg.get("label", "Overlay")
    color = trace_cfg.get("color")

    df = _resolve_df(src_ref, pool_resolve_df, dp_resolve_df)
    if df is None:
        if src_ref is None:
            return  # no source configured yet — skip silently
        raise OverlaySourceMissing(src_ref)
    cfg = _CHART_CONFIG.get(chart)
    if not cfg:
        return
    clean_roles = {k: v for k, v in primary_roles.items() if v and v in df.columns}
    overlay_fig = cfg["fn"](df, **clean_roles)  # bugs surface, not swallowed
    for trace in overlay_fig.data:
        trace.name = label
        if color:
            try:
                trace.marker.color = color  # type: ignore[attr-defined]
            except Exception:
                pass
        fig.add_trace(trace)
