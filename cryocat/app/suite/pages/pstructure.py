"""Surfaces page — pool-aware mesh + parametric-surface workflows.

The page exposes two accordion items in the sidebar:

* **Loading** — a single dropdown lists every supported way to bring a
  surface into the page: ``Mesh`` from an MRC segmentation, ``Mesh`` with
  pre-computed curvatures from a VTP file, ``OrientedPointCloud`` from
  MRC / from a motl file, and ``ParametricSurface`` from a pool motl or
  from a saved CSV parameter file.  Picking an entry renders that
  method's signature via :func:`cryocat.app.formgen.build_form` (or a
  small custom picker for the pool-motl flow), and a single Run button
  dispatches.  Mesh / OPC loads register a new surface in the page pool;
  parametric loads set the active parametric fit.

* **Operations** — a single dropdown lists every supported operation,
  grouped by ``[Mesh] / [Parametric] / [Intersection]``.  Picking an
  entry renders the matching form and (for parametric ops) the required
  motl picker(s).  The Run button dispatches against the *selected*
  surface (for mesh ops), against the *active fit* (for parametric ops),
  or against both (for particle-mesh intersection).

Live ``PleomorphicSurface`` objects live in
:mod:`cryocat.app.components.surface_registry`; the page's
``dcc.Store(id="surfaces-pool")`` only carries lightweight handles so the
state stays JSON-serialisable.  Active parametric fits live in
:mod:`cryocat.app.components.parametric_registry`.

Contract: exposes :data:`layout` and :func:`register_callbacks(app)`.

**Viewer vs. Graphs**: to see surfaces together with ray-intersection hit points,
use the Structure tab 3D viewer (renders every visible pool entry plus a
``Scatter3d`` hit-point layer coloured by ``distance_nm``).  To plot hit points
alone as a scatter, use the Graphs tab (``px.scatter_3d`` on the exported
motl); the Graphs editor is ``px``-only and cannot render ``go.Mesh3d`` geometry.
"""
from __future__ import annotations

import inspect
import math
import traceback as _tb
from typing import Any, Callable

import numpy as np
import pandas as pd

import dash
from dash import html, dcc, Input, Output, State, ALL, no_update, ctx
import dash_bootstrap_components as dbc

from cryocat.analysis.structure import ParametricSurface, PleomorphicSurface, rays_from_motl as _rays_from_motl
from cryocat.core.cryomotl import Motl
from cryocat.core.surface import Mesh, OrientedPointCloud, DiscreteSurface
from cryocat.app import ids, formgen
from cryocat.app.formgen import make_dropdown
from cryocat.app.apputils import generate_kwargs, run_operation, flatten_result_dict
from cryocat.app.logger import invoke_operation as _invoke_op, dash_logger
from cryocat.app.components import parametric_registry as pr
from cryocat.app.components import surface_registry as sr
from cryocat.app.components.motlsink import (
    get_send_to_editor_button,
    register_send_to_editor_callbacks,
)
from cryocat.app.components.motlsource import (
    get_motl_source,
    register_motl_source_callbacks,
)
from cryocat.app.components.surfaceview import (
    get_surface_view,
    register_surface_view_callbacks,
)
from cryocat.analysis import visplot
from cryocat.app.suite.pages._pstructure_intersect import (
    hits_summary_dataframe,
    subset_motl_rows,
)
from cryocat.app.pageshell import page_shell, sidebar_accordion
import cryocat.app.pool as _pool
import cryocat.app.datapool as _datapool
import cryocat.app.provenance as _prov
from cryocat.app import session as _session
from cryocat.app.components.customel import customel_graph
from cryocat.app.components.alphashape import (
    alpha_tetra_cache as _struct_alpha_cache,
    compute_tetra as _compute_alpha_tetra,
    slider_to_alpha as _slider_to_alpha,
    render_alpha_figure as _render_alpha_figure,
)


# ── Dynamically-rendered component IDs (§11.3) ───────────────────────────────
# These IDs are rendered into placeholder divs by callbacks, so they are absent
# from the static layout.  The test harness reads this list to accept them.
# Format: (container_id, component_id).  A container_id not in the layout is
# itself a defect — the test harness checks both sides.

DYNAMIC_IDS: list[tuple[str, str]] = [
    ("surfaces-send-area",         "surfaces-send-tomo-id"),
    ("surfaces-send-area",         "surfaces-build-motl-btn"),
    ("surfaces-send-area",         "surfaces-send-send-label"),
    ("surfaces-send-area",         "surfaces-send-send-to-editor"),
    ("surfaces-send-area",         "surfaces-send-send-status"),
    ("surfaces-isect-results-area", "surfaces-isect-filter-btn"),
    ("surfaces-isect-results-area", "surfaces-isect-send-send-label"),
    ("surfaces-isect-results-area", "surfaces-isect-send-send-to-editor"),
    ("surfaces-isect-results-area", "surfaces-isect-send-send-status"),
    # param-send is rendered by _render_param_results when a parametric motl op runs
    ("surfaces-param-results-area", "surfaces-param-send-send-label"),
    ("surfaces-param-results-area", "surfaces-param-send-send-to-editor"),
    ("surfaces-param-results-area", "surfaces-param-send-send-status"),
]


# ── Module-level styles ──────────────────────────────────────────────────────


_HINT = {"color": "var(--color9)"}
_LBL = {"marginBottom": "2px"}
_SECTION_HEADER = {"fontWeight": 600,
                   "margin": "0.5rem 0 0.3rem"}
# Horizontal label/input row (label on left, control fills the right).
_FIELD_ROW = {
    "display": "flex", "alignItems": "center", "gap": "0.5rem",
}
_FIELD_LABEL = {**_LBL, "flex": "0 0 45%", "margin": 0, "alignSelf": "center"}
_FIELD_INPUT = {"flex": "1 1 0", "minWidth": "0"}


def _hrow(label: str, control) -> html.Div:
    """Title + control on one row."""
    return html.Div(
        [html.Label(label, style=_FIELD_LABEL),
         html.Div(control, style=_FIELD_INPUT)],
        style={**_FIELD_ROW, "marginBottom": "0.4rem"},
    )


# ── Surface resolvers (used by op dispatch) ──────────────────────────────────


# ── Loading registry ─────────────────────────────────────────────────────────
#
# Each entry binds a load id to:
#
#   "label"       -- dropdown label.
#   "kind"        -- "formgen" (form built from method signature) or
#                    "motl_pool" (small custom form: pool picker + column).
#   "method"      -- for "formgen": the callable formgen reads + run_operation
#                    invokes.
#   "method_name" -- for "motl_pool": attribute name on ParametricSurface.
#   "result"      -- "surface" (Mesh/OPC -> page pool) or
#                    "parametric" (ParametricSurface -> active fit).
#   "exclude"     -- extra param names to hide from the form (formgen kind).

LOAD_OPS: dict[str, dict[str, Any]] = {
    "mesh_mrc": {
        "label": "Mesh from MRC (segmentation)",
        "kind": "formgen",
        "method": Mesh.from_mrc,
        "reg_key": "Mesh.from_mrc",
        "result": "surface",
        "exclude": [],
    },
    "mesh_read": {
        "label": "Mesh from file (.ply / .obj / .stl / .off)",
        "kind": "formgen",
        "method": Mesh.read,
        "reg_key": "Mesh.read",
        "result": "surface",
        "exclude": [],
    },
    "mesh_vtp": {
        "label": "Mesh with curvatures from VTP",
        "kind": "formgen",
        "method": Mesh.read_curvatures,
        "reg_key": "Mesh.read_curvatures",
        "result": "surface",
        "exclude": [],
    },
    "opc_mrc": {
        "label": "Point cloud from MRC",
        "kind": "formgen",
        "method": OrientedPointCloud.from_mrc,
        "reg_key": "OrientedPointCloud.from_mrc",
        "result": "surface",
        "exclude": [],
    },
    "opc_motl_path": {
        "label": "Point cloud from motl file (path)",
        "kind": "formgen",
        "method": OrientedPointCloud.from_motl,
        "reg_key": "OrientedPointCloud.from_motl",
        "result": "surface",
        "exclude": [],
    },
    "opc_motl_pool": {
        "label": "Point cloud from pool motl",
        "kind": "motl_pool",
        "method": OrientedPointCloud.from_motl,
        "reg_key": "OrientedPointCloud.from_motl",
        "motl_kwarg": "input_path",
        "result": "surface",
        # The motl object is supplied from the pool picker, so the method's
        # ``input_path`` parameter is hidden from the formgen form.
        "exclude": ["input_path"],
    },
    "param_motl": {
        "label": "Parametric (ellipsoid) from pool motl",
        "kind": "motl_pool",
        "method": ParametricSurface.from_motl,
        "motl_kwarg": "input_motl",
        "result": "parametric",
        "exclude": ["input_motl"],
    },
    "param_csv": {
        "label": "Parametric (ellipsoid) from CSV parameter file",
        "kind": "formgen",
        "method": ParametricSurface.from_csv,
        "result": "parametric",
        "exclude": [],
    },
}


# ── Operations registry ──────────────────────────────────────────────────────
#
# Surface ops (category="mesh") are built from discovery.entries(SURFACE_OP).
# The op id is the registry key (e.g. "Mesh.cleanup_mesh").
# method_for is derived from entry.owner in _build_surface_ops().
# Custom-UI ops (alpha_shape, intersection) and parametric ops are hand-written.
#
# Per-entry fields:
#   "label"           -- from entry.label (derived).
#   "category"        -- "mesh" for all SURFACE_OP entries (derived).
#   "kind"            -- derived by _derive_kind() from entry.returns + annotation.
#   "reg_key"         -- the registry key (= op id for surface ops).
#   "method_for"      -- callable (PleomorphicSurface) -> bound method (overrides).
#   "needs_selection" -- always True for surface ops (derived default).
#   "method_name"     -- for "parametric": attribute name on ParametricSurface.
#   "needs_active_fit"-- for "parametric": True iff the op consumes the active fit.
#   "result_kind"     -- for "parametric": "motl" or "dataframe".
#   "extra_pickers"   -- for "parametric": list of extra pool motl picker names.
#   "needs_selection" -- for "mesh": True iff the op consumes the selected
#                        surface.
#   "method_name"     -- for "parametric": attribute name on
#                        ParametricSurface (instance or @staticmethod).
#   "needs_active_fit"-- for "parametric": True iff the op consumes the
#                        active fit.
#   "result_kind"     -- for "parametric": "motl" or "dataframe".
#   "extra_pickers"   -- for "parametric": list of extra pool motl picker names.

def _derive_kind(entry) -> str:
    """Derive page dispatch kind from a GuiEntry's returns field and annotation."""
    ret = entry.returns
    if ret == "surface_pair":
        return "split"
    if ret == "surface":
        return "create"
    if ret == "field":
        return "field-source"
    if ret == "scalar":
        return "scalar"
    # export: has an output_path parameter
    try:
        import typing
        sig = inspect.signature(entry.fn)
        if "output_path" in sig.parameters:
            return "export"
        hints = typing.get_type_hints(entry.fn)
        ann = hints.get("return", inspect.Parameter.empty)
        if ann is not inspect.Parameter.empty and ann in (bool, int, float, dict):
            return "scalar"
    except Exception:
        pass
    return "unary-inplace"


def _mesh_only(psurf: "PleomorphicSurface | None") -> "Mesh | None":
    """Return psurf.surface when it is a Mesh, otherwise None."""
    if psurf is None:
        return None
    return psurf.surface if isinstance(psurf.surface, Mesh) else None


def _build_surface_ops() -> dict[str, dict[str, Any]]:
    """Build OPERATIONS entries for all SURFACE_OP registry entries via discovery.

    method_for is derived from the entry's owning class: PleomorphicSurface
    methods are looked up on the wrapper (psurf); all others on psurf.surface.
    getattr(..., None) returns None when the surface type does not have the
    method (e.g. a Mesh-only method called on an OPC), which the dispatch
    already handles as "No compatible surface selected."
    """
    from cryocat.app import discovery
    from cryocat.utils.classutils import GuiCategory
    ops: dict[str, dict[str, Any]] = {}
    for entry in discovery.entries(category=GuiCategory.SURFACE_OP):
        key = entry.key
        method_name = entry.fn.__name__
        on_wrapper = entry.owner.rsplit(".", 1)[-1] == "PleomorphicSurface"
        if on_wrapper:
            method_for = (lambda p, mn=method_name: getattr(p, mn, None) if p is not None else None)
        else:
            method_for = (lambda p, mn=method_name: getattr(p.surface, mn, None) if p is not None else None)
        ops[key] = {
            "label": entry.label,
            "category": "mesh",
            "kind": _derive_kind(entry),
            "reg_key": key,
            "method_for": method_for,
            "needs_selection": True,
        }
    return ops


OPERATIONS: dict[str, dict[str, Any]] = {
    **_build_surface_ops(),
    # ── Alpha shape (custom widget UI — not a SURFACE_OP registry entry) ─
    "alpha_shape": {
        "label": "[OPC→Mesh] Alpha shape",
        "category": "alpha_shape",
        "kind": "create",
        "needs_selection": True,
    },
    # ── Intersection (custom UI) ────────────────────────────────────────
    "intersection": {
        "label": "[Intersection] Particle–mesh ray cast",
        "category": "intersection",
    },
    # ── Parametric ops ──────────────────────────────────────────────────
    "param_distance": {
        "label": "[Parametric] Distance to surface",
        "category": "parametric",
        "method_name": "compute_point_surface_distance",
        "needs_active_fit": True,
        "result_kind": "motl",
        "extra_pickers": [],
    },
    "param_assign_distance": {
        "label": "[Parametric] Assign affiliation (distance)",
        "category": "parametric",
        "method_name": "assign_affiliation_distance_based",
        "needs_active_fit": True,
        "result_kind": "motl",
        "extra_pickers": [],
    },
    "param_assign_intersection": {
        "label": "[Parametric] Assign affiliation (intersection)",
        "category": "parametric",
        "method_name": "assign_affiliation_intersection_based",
        "needs_active_fit": True,
        "result_kind": "motl",
        "extra_pickers": [],
    },
    "param_intersection": {
        "label": "[Parametric] Intersection distances",
        "category": "parametric",
        "method_name": "compute_intersection",
        "needs_active_fit": True,
        "result_kind": "dataframe",
        "extra_pickers": [],
    },
    "param_normal_angle": {
        "label": "[Parametric] Normal angle",
        "category": "parametric",
        "method_name": "compute_normals_angle",
        "needs_active_fit": True,
        "result_kind": "motl",
        "extra_pickers": [],
    },
    "param_clean_normals": {
        "label": "[Parametric] Clean by normals",
        "category": "parametric",
        "method_name": "clean_by_normals",
        "needs_active_fit": True,
        "result_kind": "motl",
        "extra_pickers": [],
    },
    "param_clean_radius": {
        "label": "[Parametric] Clean by radius",
        "category": "parametric",
        "method_name": "clean_by_radius",
        "needs_active_fit": True,
        "result_kind": "motl",
        "extra_pickers": [],
    },
    "param_assign_mask": {
        "label": "[Parametric] Assign affiliation (mask)",
        "category": "parametric",
        "method_name": "assign_affiliation_mask_based",
        "needs_active_fit": False,
        "result_kind": "motl",
        "extra_pickers": ["object_motl"],
    },
    "param_oversample_spherical": {
        "label": "[Parametric] Spherical oversampling",
        "category": "parametric",
        "method_name": "create_spherical_oversampling",
        "needs_active_fit": False,
        "result_kind": "motl",
        "extra_pickers": [],
    },
    "param_write_out": {
        "label": "[Parametric] Save fit to file",
        "category": "parametric",
        "method_name": "write_out",
        "needs_active_fit": True,
        "result_kind": "export",
        "extra_pickers": [],
        "needs_input_motl": False,
        "exclude": [],
    },
}


# Id-type roots for the form pattern-matchers.
_LOAD_ID_TYPE = "surfaces-load-param"
_OP_ID_TYPE = "surfaces-op-param"



# Parametric extra-picker ids are pre-allocated for the ops that need them
# so register_motl_source_callbacks can wire once at startup.
_PARAM_INPUT_PICKER = "surfaces-op-param-input"
_PARAM_OBJECT_PICKER = "surfaces-op-param-object"


# ── Layout helpers ───────────────────────────────────────────────────────────


def _load_panel() -> html.Div:
    return html.Div(
        [
            html.Label("Loader", style=_LBL),
            make_dropdown("surfaces-load-select", [
                {"label": v["label"], "value": k}
                for k, v in LOAD_OPS.items()
            ], None, clearable=False, placeholder="Pick a loader",
                style={"marginBottom": "0.4rem"}),
            # The form for the selected loader (formgen rows OR the small
            # column-name input when "motl_pool" is selected).
            html.Div(id="surfaces-load-form", style={"marginBottom": "0.4rem"}),
            # Motl-pool picker, only visible for parametric-from-motl.
            html.Div(
                id="surfaces-load-motl-wrapper",
                children=get_motl_source("surfaces-load-motl", multi=False),
                style={"display": "none", "marginBottom": "0.4rem"},
            ),
            dcc.RadioItems(
                id="surfaces-load-mode",
                options=[
                    {"label": " Show only this", "value": "replace"},
                    {"label": " Add", "value": "add"},
                ],
                value="replace",
                inline=True,
                style={"marginBottom": "0.4rem", "fontSize": "0.85rem"},
            ),
            dbc.Button(
                "Run loader",
                id="surfaces-load-run-btn",
                color="primary",
                size="sm",
                style={"width": "100%"},
            ),
            html.Div(
                id="surfaces-load-status",
                style={**_HINT, "marginTop": "0.4rem",
                       "wordBreak": "break-word"},
            ),
        ]
    )


def _intersection_form() -> html.Div:
    """Custom form for the [Intersection] op (not a formgen build)."""
    return html.Div(
        [
            _hrow("Mode",
                  make_dropdown(
                      "surfaces-isect-mode",
                      [
                          {"label": "Ray cast (uses orientation)", "value": "ray"},
                          {"label": "Closest point (ignores orientation)", "value": "distance"},
                      ],
                      "ray",
                      clearable=False,
                  )),
            get_motl_source("surfaces-isect-motl", multi=False),
            _hrow("Pixel size (motl→mesh)",
                  dbc.Input(id="surfaces-isect-pixel-size", type="number",
                            value=1.0, step=0.001, size="sm")),
            _hrow("Reverse ray direction",
                  dbc.Checkbox(id="surfaces-isect-reverse", value=True)),
            _hrow("One hit per ray",
                  dbc.Checkbox(id="surfaces-isect-one-hit", value=True)),
            _hrow("Surface orientation",
                  make_dropdown(
                      "surfaces-isect-orient",
                      [
                          {"label": "Normal", "value": "normal"},
                          {"label": "Principal curvature 1", "value": "principal_1"},
                          {"label": "Principal curvature 2", "value": "principal_2"},
                      ],
                      "normal",
                      clearable=False,
                  )),
            _hrow("Max source-target distance",
                  dbc.Input(id="surfaces-isect-max-dist", type="number",
                            value=20.0, step=0.1, size="sm")),
            _hrow("Inner radius (nm)",
                  dbc.Input(id="surfaces-isect-inner-r", type="number",
                            value=9.0, step=0.1, size="sm")),
            _hrow("Outer radius (nm)",
                  dbc.Input(id="surfaces-isect-outer-r", type="number",
                            value=18.0, step=0.1, size="sm")),
            html.Div(
                [
                    html.Label(
                        "Include curvatures",
                        htmlFor="surfaces-isect-curvatures",
                        id="surfaces-isect-curvatures-lbl",
                        style={**_FIELD_LABEL, "cursor": "help"},
                        title=(
                            "Computes per-vertex curvatures on the mesh if not "
                            "pre-computed. Slow on large meshes."
                        ),
                    ),
                    html.Div(
                        dbc.Checkbox(id="surfaces-isect-curvatures", value=False),
                        style=_FIELD_INPUT,
                    ),
                ],
                style={**_FIELD_ROW, "marginBottom": "0.4rem"},
            ),
        ]
    )


# ── Declarative widget-override factories ────────────────────────────────────
#
# Each factory receives a ``param_spec`` dict  (keys: "op_id", "name") and an
# optional ``context`` dict (unused at layout-build time; reserved for future
# use).  It returns a pre-mountable Dash component whose ``id`` follows the
# predictable scheme ``"surfaces-op-override-{op_id}-{name}"`` so that the
# live-preview callback builder can reference it without seeing the component.


def _alpha_slider(param_spec: dict) -> dcc.Slider:
    """Log-scaled [0, 1] slider for `alpha` — mapped via `Mesh.suggest_alpha_range`."""
    return dcc.Slider(
        id=f"surfaces-op-override-{param_spec['op_id']}-{param_spec['name']}",
        min=0.0, max=1.0, step=0.001, value=0.5,
        tooltip={"placement": "bottom", "always_visible": False},
        marks=None,
    )


def _show_pts_checkbox(param_spec: dict) -> dbc.Checkbox:
    """Toggle: overlay source points on the preview mesh."""
    return dbc.Checkbox(
        id=f"surfaces-op-override-{param_spec['op_id']}-{param_spec['name']}",
        value=True,
        label="Show source points",
        inputStyle={"marginRight": "4px"},
    )


def _render_alpha_preview(
    psurf: PleomorphicSurface,
    kwargs: dict,
    *,
    selected_id: str,
    gs: dict,
) -> tuple:
    """Live-preview render for alpha_shape.  Called by the builder's callback."""
    alpha_raw = float(kwargs.get("alpha", 0.5))
    show_pts = bool(kwargs.get("show_pts", True))
    coords = psurf.surface.vertices
    source_key = selected_id
    # Compute and cache tetra on first call; subsequent calls are fast.
    if source_key not in _struct_alpha_cache:
        tetra_info = _compute_alpha_tetra(source_key, coords)
        if tetra_info is None:
            from cryocat.app.components.graphsettings import error_figure as _ef
            return _ef("Fewer than 4 points or coplanar — cannot compute alpha shape."), "", no_update
    else:
        lo, hi = Mesh.suggest_alpha_range(coords)
        tetra_info = {
            "source_key": source_key,
            "log_min": math.log10(lo),
            "log_max": math.log10(hi),
        }
    lo = 10 ** tetra_info["log_min"]
    hi = 10 ** tetra_info["log_max"]
    marks = {0: f"α≈{lo:.3g}", 1: f"α≈{hi:.3g}"}
    alpha = _slider_to_alpha(alpha_raw, tetra_info)
    fig, stats = _render_alpha_figure(alpha, source_key, coords, show_pts, gs)
    return fig, stats, marks


# ── OP_UI — declarative widget overrides and live preview ────────────────────
#
# Format:
#   op_id -> {
#       "widgets":      {param_name: factory_fn, ...},
#       "live_preview": {"on": [param_names], "render": render_fn},
#   }
#
# An operation with no entry renders exactly as normal (formgen + Run).
# An operation with an entry has the listed parameters replaced by the
# factory's component; formgen still handles the rest.  The presence of
# "live_preview" causes the builder to register one preview callback.

OP_UI: dict[str, dict] = {
    "alpha_shape": {
        "widgets": {
            "alpha":    _alpha_slider,
            "show_pts": _show_pts_checkbox,
        },
        "live_preview": {
            "on":     ["alpha", "show_pts"],
            "render": _render_alpha_preview,
            "widget_outputs": [
                ("surfaces-op-override-alpha_shape-alpha", "marks"),
            ],
        },
    },
}


def _op_ui_widgets() -> list[html.Div]:
    """Build pre-mounted (hidden) widget containers for every OP_UI entry."""
    containers: list[html.Div] = []
    for op_id, spec in OP_UI.items():
        controls: list = [
            html.Div(
                id=f"surfaces-op-override-{op_id}-status",
                style={**_HINT, "marginBottom": "0.3rem"},
            ),
        ]
        for param_name, factory in spec.get("widgets", {}).items():
            ps = {"op_id": op_id, "name": param_name}
            controls.append(
                html.Div(factory(ps), style={"marginBottom": "0.4rem"})
            )
        containers.append(html.Div(
            controls,
            id=f"surfaces-op-override-area-{op_id}",
            style={"display": "none", "marginBottom": "0.4rem"},
        ))
    return containers


def _register_one_preview(app, op_id: str, lp: dict) -> None:
    """Register the live-preview callback for one OP_UI entry with `live_preview`."""
    on_params: list[str] = lp["on"]
    render_fn = lp["render"]
    widget_outputs: list[tuple[str, str]] = lp.get("widget_outputs", [])
    override_status_id = f"surfaces-op-override-{op_id}-status"
    input_list = [
        Input(f"surfaces-op-override-{op_id}-{p}", "value")
        for p in on_params
    ]

    @app.callback(
        Output({"type": "styled-graph", "owner": "structure",
                "name": "op-preview"}, "figure", allow_duplicate=True),
        Output("surfaces-op-preview-stats", "children", allow_duplicate=True),
        Output(override_status_id, "children"),
        *[Output(cid, prop, allow_duplicate=True) for cid, prop in widget_outputs],
        *input_list,
        State("surfaces-op-select", "value"),
        State("surfaces-selected", "data"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        prevent_initial_call=True,
    )
    def _live_preview(*args):
        n_on = len(on_params)
        param_vals = args[:n_on]
        current_op = args[n_on]
        selected_id = args[n_on + 1]
        gs = args[n_on + 2]
        if current_op != op_id or not selected_id:
            raise dash.exceptions.PreventUpdate
        psurf = sr.registry.get(selected_id)
        if psurf is None:
            raise dash.exceptions.PreventUpdate
        kwargs = dict(zip(on_params, param_vals))
        n_widget_noupdate = (no_update,) * len(widget_outputs)
        try:
            result = render_fn(psurf, kwargs, selected_id=selected_id, gs=gs or {})
        except Exception as exc:
            from cryocat.app.components.graphsettings import error_figure as _ef
            return _ef(f"Preview error: {exc}"), str(exc), "", *n_widget_noupdate
        if widget_outputs:
            fig, stats, *widget_vals = result
        else:
            fig, stats = result
            widget_vals = []
        n_pts = 0
        if getattr(psurf, "is_point_cloud", False) and psurf.surface.vertices is not None:
            n_pts = len(psurf.surface.vertices)
        return fig, stats, (f"{n_pts:,} points" if n_pts else ""), *widget_vals

    _live_preview.__name__ = f"_live_preview_{op_id}"


def _op_panel() -> html.Div:
    return html.Div(
        [
            html.Label("Operation", style=_LBL),
            make_dropdown("surfaces-op-select", [
                {"label": v["label"], "value": k}
                for k, v in OPERATIONS.items()
            ], None, clearable=False, placeholder="Pick an operation",
                style={"marginBottom": "0.4rem"}),
            html.Div(
                id="surfaces-op-form-wrapper",
                children=html.Div(
                    "Pick an operation to render its form.",
                    style=_HINT,
                ),
                style={"marginBottom": "0.4rem"},
            ),
            # Parametric pickers are pre-mounted (hidden by default) so
            # register_motl_source_callbacks wires once at startup.
            html.Div(
                [
                    html.Label("Input motl (pool)", style=_LBL),
                    get_motl_source(_PARAM_INPUT_PICKER, multi=False),
                ],
                id="surfaces-op-input-picker-wrapper",
                style={"display": "none", "marginBottom": "0.4rem"},
            ),
            html.Div(
                [
                    html.Label("Object motl (pool)", style=_LBL),
                    get_motl_source(_PARAM_OBJECT_PICKER, multi=False),
                ],
                id="surfaces-op-object-picker-wrapper",
                style={"display": "none", "marginBottom": "0.4rem"},
            ),
            # Intersection custom UI -- pre-mounted (hidden by default) so
            # its motlsource registration sticks.
            html.Div(
                _intersection_form(),
                id="surfaces-op-isect-wrapper",
                style={"display": "none", "marginBottom": "0.4rem"},
            ),
            # Widget-override containers — one per OP_UI entry, pre-mounted hidden.
            *_op_ui_widgets(),
            dcc.Loading(
                type="circle",
                overlay_style={"visibility": "visible"},
                children=[
                    dbc.Button(
                        "Run operation",
                        id="surfaces-op-run-btn",
                        color="primary",
                        size="sm",
                        style={"width": "100%"},
                    ),
                    html.Div(
                        id="surfaces-op-status",
                        style={**_HINT, "marginTop": "0.4rem",
                               "wordBreak": "break-word"},
                    ),
                ],
            ),
        ]
    )


def _surfaces_panel() -> html.Div:
    return html.Div(
        [
            html.Div("Surfaces", style=_SECTION_HEADER),
            html.Div(
                id="surfaces-active-fit",
                style={**_HINT, "fontStyle": "italic", "marginBottom": "0.3rem"},
            ),
            html.Div(
                id="surfaces-pool-list",
                children=html.Div(
                    "No surfaces yet. Use Loading to begin.",
                    style={**_HINT, "padding": "0.25rem"},
                ),
                style={"maxHeight": "30vh", "overflowY": "auto"},
            ),
            html.Hr(style={"margin": "0.5rem 0"}),
            html.Div(
                id="surfaces-send-area",
                children=html.Div(
                    "Select a point-cloud surface to send to the Motl editor.",
                    style=_HINT,
                ),
            ),
        ]
    )


def _sidebar() -> list:
    return [
        sidebar_accordion(
            [
                dbc.AccordionItem(
                    _load_panel(),
                    title="Loading",
                    item_id="surfaces-acc-load",
                ),
                dbc.AccordionItem(
                    _op_panel(),
                    title="Operations",
                    item_id="surfaces-acc-op",
                ),
            ],
            active_item=["surfaces-acc-load", "surfaces-acc-op"],
        ),
        _surfaces_panel(),
    ]


def _main() -> list:
    _pad = {"padding": "0.5rem 0"}
    return [
        dbc.Tabs(
            id="surfaces-main-tabs",
            active_tab="surfaces-tab-view",
            children=[
                # ── View tab: 3-D viewer and its controls only ──────────────
                dbc.Tab(
                    label="View",
                    tab_id="surfaces-tab-view",
                    children=html.Div(
                        get_surface_view("surfaces-view"),
                        style=_pad,
                    ),
                ),
                # ── Results tab: every op's output ──────────────────────────
                dbc.Tab(
                    label="Results",
                    tab_id="surfaces-tab-results",
                    children=html.Div(
                        [
                            # Op-label header — updated by _track_op_label callback.
                            html.Div(id="surfaces-results-op-label"),
                            # Scalar-result panel (surface area, etc.)
                            html.Div(
                                id="surfaces-scalar-results-area",
                                children=html.Div(
                                    "Scalar operation results (e.g. surface area) "
                                    "appear here.",
                                    style=_HINT,
                                ),
                                style={"marginBottom": "0.5rem"},
                            ),
                            # Intersection results (populated after Cast)
                            html.Div(
                                id="surfaces-isect-results-area",
                                children=html.Div(
                                    "Run a particle–mesh intersection from the "
                                    "sidebar to see hits, region summary, and "
                                    "distance histogram.",
                                    style=_HINT,
                                ),
                            ),
                            # Live-preview area for ops with a preview figure
                            html.Div(
                                id="surfaces-op-preview-area",
                                style={"display": "none", "marginBottom": "0.5rem"},
                                children=[
                                    customel_graph(
                                        "structure", "op-preview",
                                        dcc.Graph(
                                            id={"type": "styled-graph",
                                                "owner": "structure",
                                                "name": "op-preview"},
                                            style={"height": "400px"},
                                        ),
                                    ),
                                    html.Div(id="surfaces-op-preview-stats",
                                             style=_HINT),
                                ],
                            ),
                            # Parametric results (table or motl send-to-editor)
                            html.Div(
                                id="surfaces-param-results-area",
                                children=html.Div(
                                    "Parametric ops that return motls appear "
                                    "here with a Send-to-editor control. "
                                    "Ops that return a table render the table.",
                                    style=_HINT,
                                ),
                            ),
                        ],
                        style=_pad,
                    ),
                ),
            ],
        ),
    ]


layout = html.Div(
    [
        # Page-local pool of handles. Live surfaces live in surface_registry.
        dcc.Store(id="surfaces-pool", data={}),
        dcc.Store(id="surfaces-selected", data=None),
        # Send-to-editor: result motl rows go here, motlsink picks them up.
        dcc.Store(id="surfaces-send-result"),
        # Intersection stores.
        dcc.Store(id="surfaces-isect-result"),
        dcc.Store(id="surfaces-isect-motl-rows"),
        dcc.Store(id="surfaces-isect-filtered-motl"),
        # data_id of the pool entry created from the last ray_intersections run.
        # Written by _adopt_isect_to_pool; read by the surface viewer to draw
        # hit points from the data pool without touching surfaces-isect-result.
        dcc.Store(id="surfaces-isect-pool-id"),
        # Column prefix for hit-point coordinates: "hit_points" (ray cast) or
        # "closest_points" (distance mode). Written by _sync_coord_prefix.
        dcc.Store(id="surfaces-isect-coord-prefix", data="hit_points"),
        # Scalar-op result snapshot (label + value), rendered into the main
        # area's scalar-results panel.  Stays a list so successive scalar
        # ops accumulate instead of overwriting.
        dcc.Store(id="surfaces-scalar-result", data=[]),
        # Tracks the label of the most-recently-run operation (for Results tab header).
        dcc.Store(id="surfaces-last-op-label"),
        # Parametric active-fit handle + per-op result stores.
        dcc.Store(id="parametric-active"),
        dcc.Store(id="surfaces-param-result-motl"),
        dcc.Store(id="surfaces-param-intersection-df"),
        page_shell(_sidebar(), _main()),
    ],
    style={"margin": "0", "padding": "0"},
)


# ── Runtime helpers ──────────────────────────────────────────────────────────


def _filter_kwargs_to_signature(method: Callable, kwargs: dict) -> dict:
    """Drop kwargs the bound method doesn't accept (handles superset forms)."""
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return kwargs
    accepted = set(sig.parameters)
    return {k: v for k, v in kwargs.items() if k in accepted}


def _adopt_result(result: Any, parent_id: str | None, label_root: str) -> list[tuple[str, dict]]:
    """Wrap backend output(s) into ``PleomorphicSurface`` and register them."""
    out: list[tuple[str, dict]] = []

    def _add(surface, label: str):
        # Capture the variable the loader bound the raw surface to BEFORE wrapping,
        # so we can emit a synthetic wrapping call event for the script.
        raw_var: str | None = (
            None if isinstance(surface, PleomorphicSurface)
            else _prov.var_for_obj(surface)
        )
        # Fallback: when the caller did not pre-bind a name (e.g. unary-inplace returning
        # a new surface, or any future path not yet covered), generate a synthetic name and
        # emit a placeholder so the script stays complete, even if not automatically runnable.
        if raw_var is None and not isinstance(surface, PleomorphicSurface):
            raw_var = "_" + _prov.bind(sr.registry.peek_next_key())
            _prov.register_intermediate(surface, raw_var)
            from cryocat.app.event import call_event as _ce
            _session.emit(_ce(
                "unknown",
                kwargs_src={},
                status="ok",
                imports=[],
                command_src=f"{raw_var} = ...  # surface produced by operation; cannot reconstruct call",
            ))
        psurf = surface if isinstance(surface, PleomorphicSurface) else PleomorphicSurface(surface)
        sid = sr.registry.add(psurf)
        surf_var = _prov.bind(sid)
        if raw_var is not None:
            # Emit a synthetic call event so the script includes the wrapping step:
            #   surf_0 = structure.PleomorphicSurface(surf_0)
            # command_src is used verbatim by the script projection.
            from cryocat.app.event import call_event as _ce
            _session.emit(_ce(
                "cryocat.analysis.structure.PleomorphicSurface.__init__",
                kwargs_src={},
                status="ok",
                imports=[["structure", "from cryocat.analysis import structure"]],
                command_src=f"{surf_var} = structure.PleomorphicSurface({raw_var})",
            ))
        psurf._pool_surface_id = sid
        if (hasattr(psurf, "surface")
                and psurf.surface is not psurf
                and not isinstance(psurf.surface, PleomorphicSurface)):
            psurf.surface._pool_surface_inner_id = sid
        _prov.record(sid, _session.last_seq())
        handle = sr.make_handle(psurf, label=label, parent_id=parent_id, visible=True)
        out.append((sid, handle))

    if isinstance(result, dict):
        for suffix, surface in result.items():
            _add(surface, f"{label_root}:{suffix}")
    elif isinstance(result, (tuple, list)):
        for i, surface in enumerate(result):
            _add(surface, f"{label_root}#{i}")
    else:
        _add(result, label_root)
    return out


def _motl_from_pool_rows(motl_id: str | None) -> Motl | None:
    """Reconstruct a :class:`Motl` from the server-side pool, or None."""
    if not motl_id:
        return None
    try:
        motl = Motl(_pool.get_rows(motl_id))
        motl._pool_motl_id = motl_id
        return motl
    except _pool.PoolPayloadMissing:
        return None


def _result_to_store(
    data: dict,
    particle_ids_seen: list[int],
    raw: dict | None = None,
    raw_label: str = "ray_intersections",
) -> dict:
    """Snapshot a ray-intersection result into a JSON-friendly store value.

    When *raw* is the direct output of ``ray_intersections`` (a dict of
    equal-length arrays), all rows (hits AND misses) are flattened via
    :func:`~cryocat.app.apputils.flatten_result_dict` and stored under
    ``raw_records`` as transit data for :func:`_adopt_isect_to_pool`.
    Row *i* of ``raw_records`` corresponds to input ray *i*; the ``hit``
    boolean distinguishes hits from misses so the full ray count is
    preserved and results can be joined back to the input motl by position.
    """
    out: dict = {"hit_source_ids": [int(i) for i in particle_ids_seen]}
    hits = data.get("hits")
    if isinstance(hits, pd.DataFrame):
        out["hits_records"] = hits.to_dict("records")
    elif isinstance(hits, dict):
        out["hits_records"] = pd.DataFrame(hits).to_dict("records")
    else:
        out["hits_records"] = []

    rs = hits_summary_dataframe(data)
    out["region_summary_records"] = rs.to_dict("records")

    regions = data.get("regions", {}) or {}
    out["regions"] = {
        str(k): [int(i) for i in np.asarray(v).tolist()]
        for k, v in regions.items()
    }

    if raw is not None:
        flat_df = flatten_result_dict(raw)
        if flat_df is not None:
            out["raw_records"] = flat_df.to_dict("records")
            out["n_rays_total"] = len(flat_df)
            out["raw_label"] = raw_label
            # When raw contains a hit column, build hits_records from the
            # hit-filtered raw rows. This brings orientation columns
            # (angles_deg, dot_products, surface_orientations_x/y/z, etc.)
            # into the results panel — intersection_data's hits DataFrame
            # does not carry these fields.
            if "hit" in flat_df.columns:
                hit_df = flat_df[flat_df["hit"].astype(bool)]
                out["hits_records"] = hit_df.to_dict("records")
                out["hit_source_ids"] = hit_df.index.tolist()

    return out


def _fmt_cell(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _records_table(records: list[dict]) -> html.Table:
    """Tiny in-page table from a list of dicts."""
    if not records:
        return html.Table()
    cols = list(records[0].keys())
    header = html.Thead(html.Tr([
        html.Th(c, style={"padding": "2px 6px"})
        for c in cols
    ]))
    body = html.Tbody([
        html.Tr([
            html.Td(_fmt_cell(r.get(c)),
                    style={"padding": "2px 6px"})
            for c in cols
        ])
        for r in records
    ])
    return html.Table([header, body], style={"borderCollapse": "collapse"})


# ── Callbacks ────────────────────────────────────────────────────────────────


def register_callbacks(app):
    # Clear status label immediately when the Run button is clicked, before the
    # server-side callback returns.
    app.clientside_callback(
        "function(n) { return ''; }",
        Output("surfaces-op-status", "children", allow_duplicate=True),
        Input("surfaces-op-run-btn", "n_clicks"),
        prevent_initial_call=True,
    )

    # Live viewer + motlsink wiring.
    register_surface_view_callbacks(
        app, "surfaces-view",
        pool_store_id="surfaces-pool",
        selected_store_id="surfaces-selected",
        isect_pool_id_store_id="surfaces-isect-pool-id",
        isect_coord_prefix_store_id="surfaces-isect-coord-prefix",
    )
    register_send_to_editor_callbacks(app, "surfaces-param-send",
                                      "surfaces-param-result-motl")
    register_send_to_editor_callbacks(app, "surfaces-isect-send",
                                      "surfaces-isect-filtered-motl")

    from dash import ALL as _ALL
    formgen.register_form_callbacks(app, _LOAD_ID_TYPE, {"op": _ALL})
    formgen.register_form_callbacks(app, _OP_ID_TYPE,   {"op": _ALL})

    # Pool pickers used by the Operations + Loading panels.
    register_motl_source_callbacks(app, "surfaces-load-motl", multi=False)
    register_motl_source_callbacks(app, _PARAM_INPUT_PICKER, multi=False)
    register_motl_source_callbacks(app, _PARAM_OBJECT_PICKER, multi=False)
    register_motl_source_callbacks(app, "surfaces-isect-motl", multi=False)

    # ── Loading form rendering ───────────────────────────────────────────────
    @app.callback(
        Output("surfaces-load-form", "children"),
        Output("surfaces-load-motl-wrapper", "style"),
        Input("surfaces-load-select", "value"),
    )
    def _render_load_form(load_id):
        if not load_id or load_id not in LOAD_OPS:
            return html.Div("Pick a loader.", style=_HINT), {"display": "none"}
        op = LOAD_OPS[load_id]
        # Both loader kinds render the method's formgen form; the only
        # difference is whether the pool picker is unhidden (motl_pool) or
        # not (formgen-only).
        rows = formgen.build_form(
            op["method"],
            id_type=_LOAD_ID_TYPE,
            id_extra={"op": load_id},
            exclude=op.get("exclude", []),
        )
        if op["kind"] == "motl_pool":
            return html.Div(rows), {"display": "block", "marginBottom": "0.4rem"}
        return html.Div(rows), {"display": "none"}

    # ── Loading Run ──────────────────────────────────────────────────────────
    @app.callback(
        Output("surfaces-pool", "data", allow_duplicate=True),
        Output("surfaces-selected", "data", allow_duplicate=True),
        Output("parametric-active", "data", allow_duplicate=True),
        Output("surfaces-load-status", "children"),
        Input("surfaces-load-run-btn", "n_clicks"),
        State("surfaces-load-select", "value"),
        State("surfaces-load-mode", "value"),
        State({"type": _LOAD_ID_TYPE, "owner": ALL, "op": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": _LOAD_ID_TYPE, "owner": ALL, "op": ALL, "param": ALL, "tag": ALL}, "id"),
        State("surfaces-load-motl-motl-select", "value"),
        State("surfaces-pool", "data"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def _run_loader(n_clicks, load_id, load_mode, values, ids, motl_id, pool, registry, pool_meta, pool_next_id):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not load_id:
            return no_update, no_update, no_update, "Pick a loader first."

        op = LOAD_OPS[load_id]
        pool = dict(pool or {})
        if load_mode == "replace":
            for h in pool.values():
                h["visible"] = False
        method = op["method"]

        # Collect form kwargs (formgen form is rendered for every loader).
        pool_state = _pool.PoolState.from_stores(registry, pool_meta, pool_next_id)
        kwargs = generate_kwargs(ids, values, pool_state) if (ids and values) else {}
        kwargs = {k: v for k, v in kwargs.items() if v not in (None, "", [])}

        # Pool-motl-driven loader: inject the Motl under the right kwarg.
        if op["kind"] == "motl_pool":
            motl = _motl_from_pool_rows(motl_id)
            if motl is None:
                return no_update, no_update, no_update, "Pick a non-empty motl from the pool."
            motl._pool_motl_id = motl_id
            kwargs[op["motl_kwarg"]] = motl
            source_tag = f"motl:{motl_id}"
        else:
            source_tag = f"path:{kwargs.get('path', kwargs.get('input_path', '?'))}"

        kwargs = _filter_kwargs_to_signature(method, kwargs)
        _load_var = "_" + _prov.bind(sr.registry.peek_next_key())
        try:
            result = _invoke_op(method, kwargs, assign_to=_load_var)
        except Exception as exc:
            return no_update, no_update, no_update, f"Load failed: {exc}"

        if op["result"] == "parametric":
            pr.registry.add(result)
            handle = pr.make_handle(result, source=source_tag)
            return no_update, no_update, handle, (
                f"Loaded {handle['n_quadrics']} parametric surface(s) "
                f"({source_tag})."
            )

        # Mesh / OPC -> page pool.
        new_entries = _adopt_result(result, parent_id=None, label_root=op["label"])
        for sid, h in new_entries:
            pool[sid] = h
        first_sid = new_entries[0][0] if new_entries else no_update
        sid_str = ", ".join(s for s, _ in new_entries)
        return pool, first_sid, no_update, (
            f"Loaded {len(new_entries)} surface(s): {sid_str}."
        )

    # ── Operations form rendering ────────────────────────────────────────────
    def _op_entry(op_id: str):
        from cryocat.utils.classutils import GUI_REGISTRY
        return GUI_REGISTRY.get(op_id)

    @app.callback(
        Output("surfaces-op-form-wrapper", "children"),
        Output("surfaces-op-input-picker-wrapper", "style"),
        Output("surfaces-op-object-picker-wrapper", "style"),
        Output("surfaces-op-isect-wrapper", "style"),
        Output("surfaces-op-override-area-alpha_shape", "style"),
        Output("surfaces-op-preview-area", "style"),
        Input("surfaces-op-select", "value"),
        Input("surfaces-selected", "data"),
        State("surfaces-pool", "data"),
    )
    def _render_op_form(op_id, selected_id, pool):
        _hidden = {"display": "none"}
        _show_over = {"display": "block", "marginBottom": "0.4rem"}
        _show_prev = {"display": "block", "marginBottom": "0.5rem"}

        # All override areas hidden by default; toggled per-category below.
        _all_hidden = (_hidden,) * len(OP_UI)  # one per OP_UI entry

        if not op_id or op_id not in OPERATIONS:
            return (
                html.Div("Pick an operation to render its form.", style=_HINT),
                _hidden, _hidden, _hidden, *_all_hidden, _hidden,
            )
        op = OPERATIONS[op_id]

        if op["category"] == "intersection":
            return (
                html.Div(),
                _hidden, _hidden,
                {"display": "block", "marginBottom": "0.4rem"},
                *_all_hidden, _hidden,
            )

        if op["category"] == "alpha_shape":
            opc_ready = (
                bool(selected_id)
                and (pool or {}).get(selected_id, {}).get("representation") == "point_cloud"
            )
            hint = (
                "Adjust the slider to preview; click Run to commit the mesh."
                if opc_ready else
                "Select a point-cloud (OPC) surface to enable the alpha-shape preview."
            )
            return (
                html.Div(hint, style=_HINT),
                _hidden, _hidden, _hidden,
                _show_over,
                _show_prev if opc_ready else _hidden,
            )

        if op["category"] == "mesh":
            entry = _op_entry(op_id)
            rows = formgen.build_form(entry, id_type=_OP_ID_TYPE, id_extra={"op": op_id})
            return (
                html.Div(rows),
                _hidden, _hidden, _hidden, *_all_hidden, _hidden,
            )

        # parametric
        method = getattr(ParametricSurface, op["method_name"])
        if "exclude" in op:
            exclude = list(op["exclude"]) + list(op.get("extra_pickers", []))
        else:
            exclude = ["input_motl", "output_path"] + list(op.get("extra_pickers", []))
        rows = formgen.build_form(
            method,
            id_type=_OP_ID_TYPE,
            id_extra={"op": op_id},
            exclude=exclude,
        )
        input_style = (
            {"display": "block", "marginBottom": "0.4rem"}
            if op.get("needs_input_motl", True)
            else _hidden
        )
        object_style = (
            {"display": "block", "marginBottom": "0.4rem"}
            if "object_motl" in op.get("extra_pickers", [])
            else _hidden
        )
        return (
            html.Div(rows), input_style, object_style,
            _hidden, *_all_hidden, _hidden,
        )

    # ── Operations Run dispatch ──────────────────────────────────────────────
    @app.callback(
        Output("surfaces-pool", "data", allow_duplicate=True),
        Output("surfaces-param-result-motl", "data", allow_duplicate=True),
        Output("surfaces-param-intersection-df", "data", allow_duplicate=True),
        Output("surfaces-isect-result", "data", allow_duplicate=True),
        Output("surfaces-isect-motl-rows", "data", allow_duplicate=True),
        Output("surfaces-scalar-result", "data", allow_duplicate=True),
        Output("surfaces-main-tabs", "active_tab", allow_duplicate=True),
        Output("surfaces-op-status", "children"),
        Input("surfaces-op-run-btn", "n_clicks"),
        State("surfaces-op-select", "value"),
        State({"type": _OP_ID_TYPE, "owner": ALL, "op": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": _OP_ID_TYPE, "owner": ALL, "op": ALL, "param": ALL, "tag": ALL}, "id"),
        State("surfaces-pool", "data"),
        State("surfaces-selected", "data"),
        # Parametric pickers.
        State(f"{_PARAM_INPUT_PICKER}-motl-select", "value"),
        State(f"{_PARAM_OBJECT_PICKER}-motl-select", "value"),
        # Intersection inputs.
        State("surfaces-isect-motl-motl-select", "value"),
        State("surfaces-isect-pixel-size", "value"),
        State("surfaces-isect-reverse", "value"),
        State("surfaces-isect-one-hit", "value"),
        State("surfaces-isect-orient", "value"),
        State("surfaces-isect-max-dist", "value"),
        State("surfaces-isect-inner-r", "value"),
        State("surfaces-isect-outer-r", "value"),
        State("surfaces-isect-curvatures", "value"),
        State("surfaces-isect-mode", "value"),
        # Alpha-shape override widget.
        State("surfaces-op-override-alpha_shape-alpha", "value"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def _run_operation(
        n_clicks, op_id, values, ids, pool, selected_id,
        in_motl_id, obj_motl_id,
        isect_motl_id, isect_px, isect_rev, isect_oh, isect_orient,
        isect_maxd, isect_inner, isect_outer,
        isect_curv, isect_mode,
        alpha_override_val,
        registry, pool_meta, pool_next_id,
    ):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not op_id:
            return (no_update,) * 7 + ("Pick an operation first.",)
        op = OPERATIONS[op_id]
        pool = dict(pool or {})
        pool_state = _pool.PoolState.from_stores(registry, pool_meta, pool_next_id)

        # ── Mesh op ─────────────────────────────────────────────────────
        if op["category"] == "mesh":
            kwargs = generate_kwargs(ids, values, pool_state) if (ids and values) else {}
            kwargs = {k: v for k, v in kwargs.items() if v not in (None, "", [])}

            psurf: PleomorphicSurface | None = None
            if op.get("needs_selection"):
                if not selected_id or selected_id not in pool:
                    return (no_update,) * 7 + (
                        "Select a surface from the list first.",)
                psurf = sr.registry.get(selected_id)
                if psurf is None:
                    return (no_update,) * 7 + (
                        f"Surface {selected_id} is no longer in the registry.",)

            method = op["method_for"](psurf)
            if method is None:
                return (no_update,) * 7 + (
                    "This operation is not available for the selected surface.",)
            kwargs = _filter_kwargs_to_signature(method, kwargs)
            _scalar_var: str | None = None
            if op["kind"] == "scalar":
                _scalar_var = _prov.bind(_prov.next_result_id())
            _op_raw_var: str | None = None
            if op["kind"] in ("create", "split"):
                _op_raw_var = "_" + _prov.bind(sr.registry.peek_next_key())
            try:
                result = _invoke_op(method, kwargs, assign_to=_scalar_var or _op_raw_var)
            except Exception as exc:
                return (no_update,) * 7 + (f"Error: {exc}",)

            if op["kind"] == "export":
                tgt = kwargs.get("output_path") or "(unspecified path)"
                return (no_update,) * 7 + (f"Saved to {tgt}.",)
            if op["kind"] == "scalar":
                rows: list[dict] = []
                if isinstance(result, dict):
                    for k, v in result.items():
                        rows.append({"label": f"{op['label']} / {k}", "value": _fmt_cell(v)})
                else:
                    rows.append({"label": op["label"], "value": _fmt_cell(result)})
                return (no_update, None, None, None, None, rows,
                        no_update,
                        f"{op['label']} -> see Results.")
            if op["kind"] == "field-source":
                handle = pool.get(selected_id)
                if handle is not None and psurf is not None:
                    handle["has_curvatures"] = sr._mesh_has_curvatures(psurf.surface)
                    pool[selected_id] = handle
                return (pool, None, None, None, None, None, no_update,
                        f"{op['label']} applied; curvature fields populated.")
            if op["kind"] in ("unary-inplace", "unary") and (result is None or result is psurf.surface or result is psurf):
                handle = pool.get(selected_id)
                if handle is not None and psurf is not None:
                    handle["n_elements"] = (
                        len(psurf.surface.vertices)
                        if psurf.surface.vertices is not None
                        else handle["n_elements"]
                    )
                    pool[selected_id] = handle
                return (pool, None, None, None, None, None, no_update,
                        f"{op['label']} applied in place.")

            if result is None:
                return (no_update,) * 7 + (
                    f"{op['label']}: no output (result was empty or all elements masked).",)
            new_entries = _adopt_result(
                result, parent_id=selected_id, label_root=op["label"],
            )
            for sid, h in new_entries:
                pool[sid] = h
            return (pool, None, None, None, None, None, no_update,
                    f"Created {len(new_entries)} new surface(s): "
                    + ", ".join(s for s, _ in new_entries) + ".")

        # ── Parametric op ───────────────────────────────────────────────
        if op["category"] == "parametric":
            if op.get("needs_input_motl", True):
                in_motl = _motl_from_pool_rows(in_motl_id)
                if in_motl is None:
                    return (no_update,) * 7 + (
                        "Pick a non-empty input motl from the pool.",)
                kwargs: dict = {"input_motl": in_motl}
            else:
                kwargs: dict = {}
            if "object_motl" in op.get("extra_pickers", []):
                obj = _motl_from_pool_rows(obj_motl_id)
                if obj is None:
                    return (no_update,) * 7 + (
                        "Pick a non-empty motl for 'object_motl'.",)
                kwargs["object_motl"] = obj
            scalar_kwargs = generate_kwargs(ids, values, pool_state) if (ids and values) else {}
            scalar_kwargs = {k: v for k, v in scalar_kwargs.items()
                             if v not in (None, "", [])}
            kwargs.update(scalar_kwargs)
            if op["needs_active_fit"]:
                _pkeys = pr.registry.keys()
                psurf = pr.registry.get(_pkeys[0]) if _pkeys else None
                if psurf is None:
                    return (no_update,) * 7 + (
                        "No active fit -- load a parametric surface first.",)
                method = getattr(psurf, op["method_name"])
            else:
                method = getattr(ParametricSurface, op["method_name"])
            try:
                result = run_operation(method, kwargs)
            except Exception as exc:
                return (no_update,) * 7 + (f"{op['label']} failed: {exc}",)
            if op["result_kind"] == "export":
                tgt = kwargs.get("output_path", "?")
                return (no_update,) * 7 + (f"Saved to {tgt}.",)
            if op["result_kind"] == "dataframe":
                if not isinstance(result, pd.DataFrame):
                    return (no_update,) * 7 + (
                        f"{op['label']} did not return a DataFrame.",)
                records = result.to_dict("records")
                return (no_update, None, records, None, None, None,
                        no_update,
                        f"{op['label']} -> {len(records)} rows; "
                        "see results table.")
            if not isinstance(result, Motl):
                return (no_update,) * 7 + (
                    f"{op['label']} did not return a Motl "
                    f"({type(result).__name__}).",)
            rows = result.df.to_dict("records")
            return (no_update, rows, None, None, None, None,
                    no_update,
                    f"{op['label']} -> {len(rows)} particles, "
                    "ready to send to editor.")

        # ── Intersection (custom flow) ───────────────────────────────────
        if op["category"] == "intersection":
            if not selected_id:
                return (no_update,) * 7 + ("Select a mesh surface first.",)
            psurf = sr.registry.get(selected_id)
            if psurf is None or not psurf.is_mesh:
                return (no_update,) * 7 + ("Selected surface must be a mesh.",)
            if not isect_motl_id:
                return (no_update,) * 7 + ("Pick a motl from the pool.",)
            try:
                isect_df = _pool.get_rows(isect_motl_id)
            except _pool.PoolPayloadMissing:
                isect_df = None
            if isect_df is None or isect_df.empty:
                return (no_update,) * 7 + (
                    f"Motl '{isect_motl_id}' has no data.",)
            isect_motl = Motl(isect_df)
            isect_motl._pool_motl_id = isect_motl_id
            motl_rows = isect_df.to_dict("records")
            _px = float(isect_px or 1.0)

            # ── Distance mode (closest-point, ignores orientation) ────────
            if isect_mode == "distance":
                target = np.asarray(
                    isect_motl.get_coordinates() * _px, dtype=np.float32
                )
                dash_logger.write(
                    f"DEBUG distance_to_points: target.shape={target.shape!r}",
                    source="cryocat",
                )
                try:
                    raw = _invoke_op(
                        psurf.distance_to_points,
                        {"target": target, "return_closest_points": True},
                        assign_to="raw",
                    )
                except Exception as exc:
                    _tb_str = _tb.format_exc()
                    print(_tb_str, flush=True)
                    dash_logger.write(f"TRACEBACK (distance_to_points):\n{_tb_str}", source="error")
                    return (no_update,) * 7 + (f"distance_to_points failed: {exc!r}",)
                n_pts = target.shape[0]
                op_label = f"distance_to_points:{selected_id}"
                # Add hit=True for all points — every target has a closest point.
                # This enables scatter drawing in the surface view and shows results
                # in the results panel.
                raw = dict(raw)
                raw["hit"] = np.ones(n_pts, dtype=bool)
                store_value = _result_to_store({}, [], raw=raw, raw_label=op_label)
                store_value["hit_coord_prefix"] = "closest_points"
                return (no_update, None, None, store_value, motl_rows,
                        None, no_update,
                        f"Computed distances for {n_pts} particles.")

            # ── Ray-cast mode (uses particle orientations) ────────────────
            _rays_kwargs = {
                "motl": isect_motl,
                "pixel_size": _px,
                "reverse_direction": bool(isect_rev),
            }
            dash_logger.write(
                f"DEBUG rays_from_motl kwargs: pixel_size={_rays_kwargs['pixel_size']!r}, "
                f"reverse_direction={_rays_kwargs['reverse_direction']!r}, "
                f"motl rows={len(isect_motl.df)}",
                source="cryocat",
            )
            try:
                rays = _invoke_op(
                    _rays_from_motl,
                    _rays_kwargs,
                    assign_to="rays",
                )
            except Exception as exc:
                _tb_str = _tb.format_exc()
                print(_tb_str, flush=True)
                dash_logger.write(f"TRACEBACK (rays_from_motl):\n{_tb_str}", source="error")
                return (no_update,) * 7 + (f"Ray construction failed: {exc!r}",)
            dash_logger.write(
                f"DEBUG ray_intersections kwargs: rays.shape={rays.shape!r}, "
                f"one_hit_per_target={bool(isect_oh)!r}",
                source="cryocat",
            )
            try:
                raw = _invoke_op(
                    psurf.ray_intersections,
                    {
                        "rays": rays,
                        "one_hit_per_target": bool(isect_oh),
                        "return_orientations": True,
                        "target_orientation": isect_orient or "normal",
                    },
                    assign_to="raw",
                )
            except Exception as exc:
                _tb_str = _tb.format_exc()
                print(_tb_str, flush=True)
                dash_logger.write(f"TRACEBACK (ray_intersections):\n{_tb_str}", source="error")
                return (no_update,) * 7 + (
                    f"ray_intersections failed: {exc!r}",)
            # Add hit boolean so the surface viewer can draw the scatter layer.
            raw = dict(raw)
            raw["hit"] = np.isfinite(np.asarray(raw.get("t_hit", []))).astype(bool)
            radii = sorted({r for r in (float(isect_inner or 0.0),
                                        float(isect_outer or 0.0)) if r > 0})
            _has_curv = psurf.surface._mean_curvature is not None
            dash_logger.write(
                f"DEBUG: reached intersection_data; "
                f"raw keys={list(raw.keys())!r}, "
                f"t_hit finite={int(np.isfinite(np.asarray(raw.get('t_hit', []))).sum())}/"
                f"{len(np.asarray(raw.get('t_hit', [])))}, "
                f"radii={radii!r}, "
                f"include_curvatures={bool(isect_curv)}, mesh._mean_curvature cached={_has_curv}",
                source="cryocat",
            )
            _isect_kwargs = {
                "result": raw, "query_type": "ray",
                "max_distance_source_target": (
                    float(isect_maxd) if isect_maxd is not None else None
                ),
                "surface_radii": radii or None,
                "include_curvatures": bool(isect_curv),
            }
            try:
                data = run_operation(
                    psurf.intersection_data,
                    _isect_kwargs,
                )
            except Exception as exc:
                _tb_str = _tb.format_exc()
                print(_tb_str, flush=True)
                dash_logger.write(f"TRACEBACK (intersection_data):\n{_tb_str}", source="error")
                return (no_update,) * 7 + (
                    f"intersection_data failed: {exc!r}",)
            hits_df = data.get("hits")
            if isinstance(hits_df, pd.DataFrame) and "source_id" in hits_df.columns:
                seen = sorted({int(x) for x in hits_df["source_id"].tolist()})
            else:
                seen = []
            op_label = f"ray_intersections:{selected_id}"
            store_value = _result_to_store(data, seen, raw=raw, raw_label=op_label)
            store_value["hit_coord_prefix"] = "hit_points"
            n_rays = rays.shape[0]
            n_hits = len(store_value["hits_records"])
            n_miss = n_rays - n_hits
            pct = 100 * n_hits / n_rays if n_rays > 0 else 0.0

            return (pool, None, None, store_value, motl_rows,
                    None, no_update,
                    f"Cast {n_rays} rays; {n_hits} hits ({pct:.0f}%), "
                    f"{n_miss} misses.")

        # ── Alpha shape ──────────────────────────────────────────────────
        if op["category"] == "alpha_shape":
            if not selected_id or selected_id not in pool:
                return (no_update,) * 7 + ("Select a surface first.",)
            psurf = sr.registry.get(selected_id)
            if psurf is None or not psurf.is_point_cloud:
                return (no_update,) * 7 + (
                    "Alpha shape requires an OrientedPointCloud surface.",)
            coords = psurf.surface.vertices
            source_key = selected_id
            if source_key not in _struct_alpha_cache:
                tetra_info = _compute_alpha_tetra(source_key, coords)
                if tetra_info is None:
                    return (no_update,) * 7 + (
                        "Cannot compute alpha shape: fewer than 4 points or coplanar.",)
            lo, hi = Mesh.suggest_alpha_range(coords)
            alpha = _slider_to_alpha(
                float(alpha_override_val or 0.5),
                {"log_min": math.log10(lo), "log_max": math.log10(hi)},
            )
            tetra_pair = _struct_alpha_cache.get(source_key, (None, None))
            _raw_var = "_" + _prov.bind(sr.registry.peek_next_key())
            try:
                mesh = _invoke_op(
                    Mesh.from_alpha_shape,
                    {"points": coords, "alpha": alpha,
                     "tetra_mesh": tetra_pair[0], "pt_map": tetra_pair[1]},
                    assign_to=_raw_var,
                )
            except Exception as exc:
                return (no_update,) * 7 + (f"Alpha shape failed: {exc}",)
            new_entries = _adopt_result(
                mesh, parent_id=selected_id, label_root=f"alpha={alpha:.4g}",
            )
            pool = dict(pool or {})
            for sid, h in new_entries:
                pool[sid] = h
            return (pool,) + (no_update,) * 6 + (
                f"Committed alpha={alpha:.4g}: "
                f"{len(new_entries)} surface(s) added to pool.",)

        return (no_update,) * 7 + (f"Unknown op category: {op['category']!r}.",)

    # ── Fix 8: track and display the label of the most-recently-run operation ──
    @app.callback(
        Output("surfaces-last-op-label", "data"),
        Input("surfaces-op-status", "children"),
        State("surfaces-op-select", "value"),
        prevent_initial_call=True,
    )
    def _track_op_label(status, op_id):
        if not status or not op_id or op_id not in OPERATIONS:
            return no_update
        _skip = ("Error:", "Pick ", "Select ", "No ", "Cannot ", "Alpha shape requires")
        if any(status.startswith(s) for s in _skip) or "failed" in status.lower():
            return no_update
        return OPERATIONS[op_id].get("label", op_id)

    @app.callback(
        Output("surfaces-results-op-label", "children"),
        Input("surfaces-last-op-label", "data"),
    )
    def _render_results_op_label(label):
        if not label:
            return ""
        return html.H5(label, style={"marginBottom": "0.5rem"})

    # ── FK2: adopt raw ray_intersections result into the data pool ───────────
    @app.callback(
        Output(ids.DATA_POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.DATA_POOL_NEXT_ID, "data", allow_duplicate=True),
        Output("surfaces-isect-pool-id", "data", allow_duplicate=True),
        Input("surfaces-isect-result", "data"),
        State(ids.DATA_POOL_REGISTRY, "data"),
        State(ids.DATA_POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def _adopt_isect_to_pool(snap, dp_reg, dp_next):
        if not snap or "raw_records" not in snap:
            raise dash.exceptions.PreventUpdate
        try:
            df = pd.DataFrame(snap["raw_records"])
            label = snap.get("raw_label", "ray_intersections")
            ds = _datapool.DataPoolState.from_stores(dp_reg, dp_next)
            new_ds, data_id = _datapool.insert_entry(
                ds, df,
                label=label,
                reader="dataframe",
                source_path="",
                entry_kind="isect",
            )
            return (*new_ds.to_stores(), data_id)
        except Exception as exc:
            dash_logger.write(
                f"_adopt_isect_to_pool failed: {exc!r}\n"
                f"{_tb.format_exc()}",
                source="error",
            )
            raise dash.exceptions.PreventUpdate

    # ── FK2b: sync coordinate prefix for the surface viewer ──────────────────
    @app.callback(
        Output("surfaces-isect-coord-prefix", "data"),
        Input("surfaces-isect-result", "data"),
        prevent_initial_call=True,
    )
    def _sync_coord_prefix(snap):
        if not snap:
            raise dash.exceptions.PreventUpdate
        return snap.get("hit_coord_prefix", "hit_points")

    # ── Scalar-results panel (main area) ─────────────────────────────────────
    @app.callback(
        Output("surfaces-scalar-results-area", "children"),
        Input("surfaces-scalar-result", "data"),
    )
    def _render_scalar_results(records):
        if not records:
            return html.Div(
                "Scalar operation results (e.g. surface area) appear here.",
                style=_HINT,
            )
        return html.Div([
            html.H6("Scalar results"),
            html.Div(
                _records_table(records),
                style={"maxHeight": "200px", "overflowY": "auto"},
            ),
        ])

    # ── Active-fit info readout ──────────────────────────────────────────────
    @app.callback(
        Output("surfaces-active-fit", "children"),
        Input("parametric-active", "data"),
    )
    def _render_active_fit(handle):
        if not handle:
            return ""
        return (
            f"Active parametric fit: {handle.get('surface_type', '?')} on "
            f"'{handle.get('column_name', '?')}'; "
            f"{int(handle.get('n_quadrics', 0))} surface(s); "
            f"source={handle.get('source', '?')}."
        )

    # ── Render intersection results (main area) ──────────────────────────────
    @app.callback(
        Output("surfaces-isect-results-area", "children"),
        Input("surfaces-isect-result", "data"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
    )
    def _render_isect_results(snap, gs):
        if not snap:
            return html.Div(
                "Run a particle–mesh intersection from the sidebar to see "
                "hits, region summary, and distance histogram.",
                style=_HINT,
            )
        children: list = [html.H6("Intersection results")]
        rs = snap.get("region_summary_records") or []
        if rs:
            children.append(html.Div("Region summary",
                                     style={"fontWeight": "bold"}))
            children.append(html.Div(
                _records_table(rs),
                style={"maxHeight": "220px", "overflowY": "auto",
                       "marginBottom": "0.6rem"},
            ))
        hits = snap.get("hits_records") or []
        if hits:
            children.append(html.Div(
                f"Hits ({len(hits)} rows; showing first 200)",
                style={"fontWeight": "bold"},
            ))
            children.append(html.Div(
                _records_table(hits[:200]),
                style={"maxHeight": "260px", "overflowY": "auto",
                       "marginBottom": "0.6rem"},
            ))
            try:
                hits_df = pd.DataFrame(hits)
                _hist_col = (
                    "distance_nm" if "distance_nm" in hits_df.columns
                    else "t_hit" if "t_hit" in hits_df.columns
                    else None
                )
                if _hist_col is not None:
                    from cryocat.app.components.graphsettings import styled_figure as _sf
                    fig = _sf(visplot.plot_histogram(hits_df[[_hist_col]], bins=30), gs or {})
                    children.append(customel_graph("structure", "intersection-hist", dcc.Graph(id={"type": "styled-graph", "owner": "structure", "name": "intersection-hist"}, figure=fig, style={"height": "320px"})))
            except Exception as exc:
                children.append(html.Div(
                    f"Histogram skipped: {exc}", style=_HINT,
                ))
        regions = snap.get("regions") or {}
        if regions:
            children.append(html.Div("Extract region",
                                     style={"fontWeight": "bold"}))
            buttons = []
            for region_name, idx in regions.items():
                if not idx:
                    continue
                buttons.append(dbc.Button(
                    f"{region_name} ({len(idx)})",
                    id={"type": "surfaces-isect-extract-btn",
                        "region": region_name},
                    color="secondary", size="sm",
                    style={"marginRight": "0.4rem", "marginBottom": "0.3rem"},
                ))
            children.append(html.Div(buttons))
        n_filt = len(snap.get("hit_source_ids") or [])
        children.append(html.Hr(style={"margin": "0.5rem 0"}))
        children.append(html.Div("Filter motl by intersection",
                                 style={"fontWeight": "bold"}))
        children.append(html.Div(
            f"{n_filt} unique particle(s) intersected the surface.",
            style={**_HINT, "marginBottom": "0.4rem"},
        ))
        children.append(dbc.Button(
            "Filter particles -> result store",
            id="surfaces-isect-filter-btn",
            color="secondary", size="sm",
            style={"marginRight": "0.4rem", "marginBottom": "0.4rem"},
        ))
        children.append(get_send_to_editor_button("surfaces-isect-send"))
        return html.Div(children)

    # ── Extract region (per-button via pattern-matched id) ───────────────────
    @app.callback(
        Output("surfaces-pool", "data", allow_duplicate=True),
        Output("surfaces-op-status", "children", allow_duplicate=True),
        Input({"type": "surfaces-isect-extract-btn", "region": ALL}, "n_clicks"),
        State("surfaces-isect-result", "data"),
        State("surfaces-selected", "data"),
        State("surfaces-pool", "data"),
        prevent_initial_call=True,
    )
    def _extract_region(n_clicks_list, snap, selected_surface, pool):
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict)
                and triggered.get("type") == "surfaces-isect-extract-btn"):
            raise dash.exceptions.PreventUpdate
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        region_name = triggered["region"]
        regions = (snap or {}).get("regions") or {}
        idx = regions.get(region_name)
        if not idx:
            return no_update, f"Region '{region_name}' has no indices."
        psurf = sr.registry.get(selected_surface)
        if psurf is None or not psurf.is_mesh:
            return no_update, "Source surface no longer in the registry."
        _op_var = _prov.bind(sr.registry.peek_next_key())
        try:
            new_psurf = _invoke_op(
                psurf.extract_region,
                {"indices": np.asarray(idx, dtype=int),
                 "element": "triangles"},
                assign_to=_op_var,
            )
        except Exception as exc:
            return no_update, f"extract_region failed: {exc}"
        pool = dict(pool or {})
        new_entries = _adopt_result(
            new_psurf, parent_id=selected_surface,
            label_root=f"region {region_name}",
        )
        for sid, h in new_entries:
            pool[sid] = h
        sid_str = ", ".join(s for s, _ in new_entries)
        return pool, f"Extracted '{region_name}' as {sid_str}."

    # ── Filter motl rows to the intersecting subset ──────────────────────────
    @app.callback(
        Output("surfaces-isect-filtered-motl", "data"),
        Output("surfaces-op-status", "children", allow_duplicate=True),
        Input("surfaces-isect-filter-btn", "n_clicks"),
        State("surfaces-isect-result", "data"),
        State("surfaces-isect-motl-rows", "data"),
        prevent_initial_call=True,
    )
    def _filter_motl(n_clicks, snap, motl_rows):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not snap:
            return no_update, "No intersection result -- run Cast first."
        if not motl_rows:
            return no_update, "Source motl rows are missing."
        idx = snap.get("hit_source_ids") or []
        subset = subset_motl_rows(motl_rows, idx)
        if not subset:
            return no_update, "No particles intersected the surface."
        return subset, (
            f"Filtered {len(subset)} particles -> ready to send to editor."
        )

    # ── Render parametric results (motl send-to-editor or intersection table) ──
    @app.callback(
        Output("surfaces-param-results-area", "children"),
        Input("surfaces-param-intersection-df", "data"),
        Input("surfaces-param-result-motl", "data"),
    )
    def _render_param_results(records, motl_rows):
        if motl_rows is not None:
            return html.Div([
                html.Div(
                    f"{len(motl_rows)} particles ready.",
                    style={**_HINT, "marginBottom": "0.4rem"},
                ),
                get_send_to_editor_button("surfaces-param-send"),
            ])
        if records:
            return html.Div([
                html.H6("Intersection distances"),
                html.Div(
                    f"{len(records)} rows; showing first 200.",
                    style={**_HINT, "marginBottom": "0.4rem"},
                ),
                html.Div(
                    _records_table(records[:200]),
                    style={"maxHeight": "400px", "overflowY": "auto"},
                ),
            ])
        return html.Div(
            "Parametric results (motl or intersection-distance table) appear here.",
            style=_HINT,
        )

    # ── Surfaces list rendering ──────────────────────────────────────────────
    @app.callback(
        Output("surfaces-pool-list", "children"),
        Input("surfaces-pool", "data"),
        Input("surfaces-selected", "data"),
    )
    def _render_list(pool, selected_id):
        pool = pool or {}
        if not pool:
            return html.Div(
                "No surfaces yet. Use Loading to begin.",
                style={**_HINT, "padding": "0.25rem"},
            )
        rows = []
        for sid, h in pool.items():
            is_sel = (sid == selected_id)
            badge_color = "primary" if h["representation"] == "mesh" else "info"
            visible = h.get("visible", True)
            row_bg = "var(--color10)" if is_sel else ""
            rows.append(
                dbc.ListGroupItem(
                    [
                        dbc.Badge(h["representation"], color=badge_color,
                                  className="me-2"),
                        html.Span(
                            f"{h['label']} ({h['n_elements']})",
                            style={"flex": "1", "overflow": "hidden",
                                   "textOverflow": "ellipsis"},
                        ),
                        dbc.Button(
                            "👁" if visible else "○",
                            id={"type": "surfaces-row-visible", "sid": sid},
                            color="link", size="sm",
                            style={"padding": "0 6px", "lineHeight": "1",
                                   "fontSize": "0.95rem"},
                        ),
                        dbc.Button(
                            "×",
                            id={"type": "surfaces-row-delete", "sid": sid},
                            color="link", size="sm",
                            style={"padding": "0 6px", "lineHeight": "1"},
                        ),
                    ],
                    id={"type": "surfaces-row-select", "sid": sid},
                    action=True, n_clicks=0,
                    style={"display": "flex", "alignItems": "center",
                           "padding": "4px 6px", "cursor": "pointer",
                           "backgroundColor": row_bg},
                )
            )
        return dbc.ListGroup(rows, flush=True)

    # ── Select a surface ─────────────────────────────────────────────────────
    @app.callback(
        Output("surfaces-selected", "data"),
        Input({"type": "surfaces-row-select", "sid": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _select_row(n_clicks_list):
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "sid" in triggered):
            raise dash.exceptions.PreventUpdate
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        return triggered["sid"]

    # ── Toggle visibility ────────────────────────────────────────────────────
    @app.callback(
        Output("surfaces-pool", "data", allow_duplicate=True),
        Input({"type": "surfaces-row-visible", "sid": ALL}, "n_clicks"),
        State("surfaces-pool", "data"),
        prevent_initial_call=True,
    )
    def _toggle_visible(n_clicks_list, pool):
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "sid" in triggered):
            raise dash.exceptions.PreventUpdate
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        pool = dict(pool or {})
        sid = triggered["sid"]
        if sid not in pool:
            raise dash.exceptions.PreventUpdate
        pool[sid]["visible"] = not pool[sid].get("visible", True)
        return pool

    # ── Delete a surface ─────────────────────────────────────────────────────
    @app.callback(
        Output("surfaces-pool", "data", allow_duplicate=True),
        Output("surfaces-selected", "data", allow_duplicate=True),
        Input({"type": "surfaces-row-delete", "sid": ALL}, "n_clicks"),
        State("surfaces-pool", "data"),
        State("surfaces-selected", "data"),
        prevent_initial_call=True,
    )
    def _delete_row(n_clicks_list, pool, selected_id):
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "sid" in triggered):
            raise dash.exceptions.PreventUpdate
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        sid = triggered["sid"]
        pool = {k: v for k, v in (pool or {}).items() if k != sid}
        sr.registry.remove(sid)
        new_selected = None if selected_id == sid else selected_id
        return pool, new_selected

    # ── Send-to-editor area (point clouds only) ──────────────────────────────
    @app.callback(
        Output("surfaces-send-area", "children"),
        Input("surfaces-selected", "data"),
        Input("surfaces-pool", "data"),
    )
    def _render_send_area(selected_id, pool):
        pool = pool or {}
        if not selected_id or selected_id not in pool:
            return html.Div(
                "Select a point-cloud surface to send to the Motl editor.",
                style=_HINT,
            )
        handle = pool[selected_id]
        if handle["representation"] != "point_cloud":
            return html.Div(
                "Send-to-editor is available for point clouds only.",
                style=_HINT,
            )
        return html.Div(
            [
                html.Label("tomo_id (optional)", style=_HINT),
                dbc.Input(
                    id="surfaces-send-tomo-id",
                    type="number",
                    placeholder="e.g. 1",
                    size="sm",
                    style={"marginBottom": "0.4rem"},
                ),
                dbc.Button(
                    "Build motl from selection",
                    id="surfaces-build-motl-btn",
                    color="secondary",
                    size="sm",
                    style={"width": "100%", "marginBottom": "0.4rem"},
                ),
                get_send_to_editor_button("surfaces-send"),
            ]
        )

    @app.callback(
        Output("surfaces-send-result", "data"),
        Output("surfaces-send-send-status", "children", allow_duplicate=True),
        Input("surfaces-build-motl-btn", "n_clicks"),
        State("surfaces-selected", "data"),
        State("surfaces-send-tomo-id", "value"),
        prevent_initial_call=True,
    )
    def _build_motl(n_clicks, selected_id, tomo_id):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not selected_id:
            return no_update, "No surface selected."
        psurf = sr.registry.get(selected_id)
        if psurf is None or not psurf.is_point_cloud:
            return no_update, "Selected surface is not a point cloud."
        try:
            motl = run_operation(
                psurf.surface.to_motl,
                {"tomo_id": int(tomo_id) if tomo_id is not None else None},
            )
        except Exception as exc:
            return no_update, f"Build motl failed: {exc}"
        rows = motl.df.to_dict("records")
        return rows, f"Built motl with {len(rows)} particles."

    # The Send-to-editor button needs its callback registered AFTER its
    # parent div exists; the button is conditionally rendered above so we
    # register here at startup -- Dash tolerates the late-mount.
    register_send_to_editor_callbacks(app, "surfaces-send", "surfaces-send-result")

    # ── BY: register one live-preview callback per OP_UI live_preview entry ───
    for _op_id, _spec in OP_UI.items():
        _lp = _spec.get("live_preview")
        if _lp:
            _register_one_preview(app, _op_id, _lp)
