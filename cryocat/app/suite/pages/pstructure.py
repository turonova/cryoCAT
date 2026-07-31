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
"""
from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np
import pandas as pd

import dash
from dash import html, dcc, Input, Output, State, ALL, no_update, ctx
import dash_bootstrap_components as dbc

from cryocat.analysis.structure import ParametricSurface, PleomorphicSurface
from cryocat.core.cryomotl import Motl
from cryocat.core.surface import Mesh, OrientedPointCloud, DiscreteSurface
from cryocat.app import ids, formgen
from cryocat.app.apputils import generate_kwargs, run_operation
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
    motl_rows_to_rays,
    subset_motl_rows,
)
from cryocat.app.pageshell import page_shell, sidebar_accordion


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
]


# ── Module-level styles ──────────────────────────────────────────────────────


_HINT = {"fontSize": "0.85rem", "color": "var(--color9)"}
_LBL = {"fontSize": "0.85rem", "marginBottom": "2px"}
_SECTION_HEADER = {"fontSize": "0.9rem", "fontWeight": 600,
                   "margin": "0.5rem 0 0.3rem"}
# Horizontal label/input row (label on left, control fills the right).
_FIELD_ROW = {
    "display": "flex", "alignItems": "center", "gap": "0.5rem",
    "marginBottom": "0.4rem",
}
_FIELD_LABEL = {**_LBL, "flex": "0 0 45%", "margin": 0, "alignSelf": "center"}
_FIELD_INPUT = {"flex": "1 1 0", "minWidth": "0"}


def _hrow(label: str, control) -> html.Div:
    """Title + control on one row."""
    return html.Div(
        [html.Label(label, style=_FIELD_LABEL),
         html.Div(control, style=_FIELD_INPUT)],
        style=_FIELD_ROW,
    )


# ── Surface resolvers (used by op dispatch) ──────────────────────────────────


def _mesh_only(psurf: PleomorphicSurface | None):
    """Return ``psurf.surface`` iff it's a Mesh, else None."""
    if psurf is None or not psurf.is_mesh:
        return None
    return psurf.surface


def _refine_method(psurf):
    return None if psurf is None else psurf.surface.refine_normals


def _oversample_method(psurf):
    return None if psurf is None else psurf.surface.oversample


def _flip_method(psurf):
    return None if psurf is None else psurf.surface.flip_normals


def _save_method(psurf):
    return None if psurf is None else psurf.surface.save


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
        "result": "surface",
        "exclude": [],
    },
    "mesh_vtp": {
        "label": "Mesh with curvatures from VTP",
        "kind": "formgen",
        "method": Mesh.read_curvatures,
        "result": "surface",
        "exclude": [],
    },
    "opc_mrc": {
        "label": "Point cloud from MRC",
        "kind": "formgen",
        "method": OrientedPointCloud.from_mrc,
        "result": "surface",
        "exclude": [],
    },
    "opc_motl_path": {
        "label": "Point cloud from motl file (path)",
        "kind": "formgen",
        "method": OrientedPointCloud.from_motl,
        "result": "surface",
        "exclude": [],
    },
    "opc_motl_pool": {
        "label": "Point cloud from pool motl",
        "kind": "motl_pool",
        "method": OrientedPointCloud.from_motl,
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
# Each entry binds an op id to:
#
#   "label"           -- dropdown label.
#   "category"        -- "mesh" / "parametric" / "intersection".
#   "kind"            -- "create" / "unary" / "unary-inplace" / "split" /
#                        "scalar" / "field-source" / "export" / "intersection".
#   "form_method"     -- for "mesh" / "parametric": the callable formgen
#                        reads.  None for "intersection" (custom UI).
#   "exclude"         -- iterable of param names to drop from the form.
#   "method_for"      -- for "mesh": callable (PleomorphicSurface) -> bound
#                        method.
#   "needs_selection" -- for "mesh": True iff the op consumes the selected
#                        surface.
#   "method_name"     -- for "parametric": attribute name on
#                        ParametricSurface (instance or @staticmethod).
#   "needs_active_fit"-- for "parametric": True iff the op consumes the
#                        active fit.
#   "result_kind"     -- for "parametric": "motl" or "dataframe".
#   "extra_pickers"   -- for "parametric": list of extra pool motl picker names.

OPERATIONS: dict[str, dict[str, Any]] = {
    # ── Mesh ops ────────────────────────────────────────────────────────
    "cleanup_mesh": {
        "label": "[Mesh] Cleanup",
        "category": "mesh",
        "kind": "create",
        "form_method": Mesh.cleanup_mesh,
        "exclude": [],
        "method_for": lambda p: (
            _mesh_only(p).cleanup_mesh if _mesh_only(p) is not None else None
        ),
        "needs_selection": True,
    },
    "smooth": {
        "label": "[Mesh] Smooth",
        "category": "mesh",
        "kind": "unary-inplace",
        "form_method": Mesh.smooth,
        "exclude": [],
        "method_for": lambda p: (
            _mesh_only(p).smooth if _mesh_only(p) is not None else None
        ),
        "needs_selection": True,
    },
    "compute_curvatures": {
        "label": "[Mesh] Compute curvatures",
        "category": "mesh",
        "kind": "field-source",
        "form_method": Mesh.compute_curvatures,
        "exclude": ["force_recompute", "min_triangle_area",
                    "lstsq_rcond", "n_jobs"],
        "method_for": lambda p: (
            _mesh_only(p).compute_curvatures if _mesh_only(p) is not None else None
        ),
        "needs_selection": True,
    },
    "surface_area": {
        "label": "[Mesh] Surface area",
        "category": "mesh",
        "kind": "scalar",
        "form_method": Mesh.get_surface_area,
        "exclude": [],
        "method_for": lambda p: (
            _mesh_only(p).get_surface_area if _mesh_only(p) is not None else None
        ),
        "needs_selection": True,
    },
    "refine_normals": {
        "label": "[Mesh/OPC] Refine normals",
        "category": "mesh",
        "kind": "unary",
        "form_method": DiscreteSurface.refine_normals,
        "exclude": ["mask", "logger", "inplace", "batch_size"],
        "method_for": _refine_method,
        "needs_selection": True,
    },
    "separate_closed": {
        "label": "[Mesh] Separate closed surface (inner/outer)",
        "category": "mesh",
        "kind": "split",
        "form_method": DiscreteSurface.separate_closed_surface,
        "exclude": ["reference_point"],
        "method_for": lambda p: (
            p.surface.separate_closed_surface if p is not None else None
        ),
        "needs_selection": True,
    },
    "oversample": {
        "label": "[Mesh/OPC] Oversample",
        "category": "mesh",
        "kind": "unary",
        "form_method": Mesh.oversample,
        "exclude": [],
        "method_for": _oversample_method,
        "needs_selection": True,
    },
    "flip_normals": {
        "label": "[Mesh/OPC] Flip normals",
        "category": "mesh",
        "kind": "unary",
        "form_method": Mesh.flip_normals,
        "exclude": ["inplace"],
        "method_for": _flip_method,
        "needs_selection": True,
    },
    "save": {
        "label": "[Mesh/OPC] Save to disk",
        "category": "mesh",
        "kind": "export",
        "form_method": Mesh.save,
        "exclude": [],
        "method_for": _save_method,
        "needs_selection": True,
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
    # ── Intersection (custom UI) ────────────────────────────────────────
    "intersection": {
        "label": "[Intersection] Particle–mesh ray cast",
        "category": "intersection",
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
            dcc.Dropdown(
                id="surfaces-load-select",
                options=[
                    {"label": v["label"], "value": k}
                    for k, v in LOAD_OPS.items()
                ],
                value=None,
                placeholder="Pick a loader",
                clearable=False,
                style={"fontSize": "0.85rem", "marginBottom": "0.4rem"},
            ),
            # The form for the selected loader (formgen rows OR the small
            # column-name input when "motl_pool" is selected).
            html.Div(id="surfaces-load-form", style={"marginBottom": "0.4rem"}),
            # Motl-pool picker, only visible for parametric-from-motl.
            html.Div(
                id="surfaces-load-motl-wrapper",
                children=get_motl_source("surfaces-load-motl", multi=False),
                style={"display": "none", "marginBottom": "0.4rem"},
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
            html.Div(
                "Cast rays from a pool motl onto the selected mesh.",
                style={**_HINT, "marginBottom": "0.4rem"},
            ),
            get_motl_source("surfaces-isect-motl", multi=False),
            _hrow("Pixel size (motl→mesh)",
                  dbc.Input(id="surfaces-isect-pixel-size", type="number",
                            value=1.0, step=0.001, size="sm")),
            _hrow("Reverse ray direction",
                  dbc.Checkbox(id="surfaces-isect-reverse", value=True)),
            _hrow("One hit per ray",
                  dbc.Checkbox(id="surfaces-isect-one-hit", value=True)),
            _hrow("Max source-target distance",
                  dbc.Input(id="surfaces-isect-max-dist", type="number",
                            value=20.0, step=0.1, size="sm")),
            _hrow("Inner radius (nm)",
                  dbc.Input(id="surfaces-isect-inner-r", type="number",
                            value=9.0, step=0.1, size="sm")),
            _hrow("Outer radius (nm)",
                  dbc.Input(id="surfaces-isect-outer-r", type="number",
                            value=18.0, step=0.1, size="sm")),
        ]
    )


def _op_panel() -> html.Div:
    return html.Div(
        [
            html.Label("Operation", style=_LBL),
            dcc.Dropdown(
                id="surfaces-op-select",
                options=[
                    {"label": v["label"], "value": k}
                    for k, v in OPERATIONS.items()
                ],
                value=None,
                placeholder="Pick an operation",
                clearable=False,
                style={"fontSize": "0.85rem", "marginBottom": "0.4rem"},
            ),
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
    return [
        html.H4("Surfaces", style={"marginBottom": "0.5rem"}),
        get_surface_view("surfaces-view"),
        html.Hr(style={"margin": "0.5rem 0"}),
        # Scalar-result panel (surface area, etc.) populated by the
        # Operations Run dispatch when the op's kind is "scalar".
        html.Div(
            id="surfaces-scalar-results-area",
            children=html.Div(
                "Scalar operation results (e.g. surface area) appear here.",
                style=_HINT,
            ),
            style={"marginBottom": "0.5rem"},
        ),
        # Intersection results live here; populated after Cast.
        html.Div(
            id="surfaces-isect-results-area",
            children=html.Div(
                "Run a particle–mesh intersection from the sidebar "
                "to see hits, region summary, and distance histogram.",
                style=_HINT,
            ),
        ),
        # Parametric DataFrame results.
        html.Div(
            id="surfaces-param-results-area",
            children=html.Div(
                "Parametric ops that return motls go to the editor "
                "via the side panel. Ops that return a table "
                "(intersection distances) render here.",
                style=_HINT,
            ),
        ),
        # motlsink for parametric motl outputs.
        get_send_to_editor_button("surfaces-param-send"),
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
        # Scalar-op result snapshot (label + value), rendered into the main
        # area's scalar-results panel.  Stays a list so successive scalar
        # ops accumulate instead of overwriting.
        dcc.Store(id="surfaces-scalar-result", data=[]),
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
        psurf = surface if isinstance(surface, PleomorphicSurface) else PleomorphicSurface(surface)
        sid = sr.registry.add(psurf)
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


def _motl_from_pool_rows(pool_motls: dict | None, motl_id: str | None) -> Motl | None:
    """Reconstruct a :class:`Motl` from the suite-pool row list, or None."""
    pool_motls = pool_motls or {}
    if not motl_id:
        return None
    rows = pool_motls.get(motl_id) or []
    if not rows:
        return None
    return Motl(pd.DataFrame(rows))


def _result_to_store(data: dict, particle_ids_seen: list[int]) -> dict:
    """Snapshot a ray-intersection result into a JSON-friendly store value."""
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
        html.Th(c, style={"padding": "2px 6px", "fontSize": "0.8rem"})
        for c in cols
    ]))
    body = html.Tbody([
        html.Tr([
            html.Td(_fmt_cell(r.get(c)),
                    style={"padding": "2px 6px", "fontSize": "0.8rem"})
            for c in cols
        ])
        for r in records
    ])
    return html.Table([header, body], style={"borderCollapse": "collapse"})


# ── Callbacks ────────────────────────────────────────────────────────────────


def register_callbacks(app):
    # Live viewer + motlsink wiring.
    register_surface_view_callbacks(
        app, "surfaces-view",
        pool_store_id="surfaces-pool",
        selected_store_id="surfaces-selected",
    )
    register_send_to_editor_callbacks(app, "surfaces-param-send",
                                      "surfaces-param-result-motl")
    register_send_to_editor_callbacks(app, "surfaces-isect-send",
                                      "surfaces-isect-filtered-motl")

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
        Output("parametric-active", "data", allow_duplicate=True),
        Output("surfaces-load-status", "children"),
        Input("surfaces-load-run-btn", "n_clicks"),
        State("surfaces-load-select", "value"),
        State({"type": _LOAD_ID_TYPE, "owner": ALL, "op": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": _LOAD_ID_TYPE, "owner": ALL, "op": ALL, "param": ALL, "tag": ALL}, "id"),
        State("surfaces-load-motl-motl-select", "value"),
        State(ids.POOL_MOTLS, "data"),
        State("surfaces-pool", "data"),
        prevent_initial_call=True,
    )
    def _run_loader(n_clicks, load_id, values, ids, motl_id, pool_motls, pool):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not load_id:
            return no_update, no_update, "Pick a loader first."

        op = LOAD_OPS[load_id]
        pool = dict(pool or {})
        method = op["method"]

        # Collect form kwargs (formgen form is rendered for every loader).
        kwargs = generate_kwargs(ids, values) if (ids and values) else {}
        kwargs = {k: v for k, v in kwargs.items() if v not in (None, "", [])}

        # Pool-motl-driven loader: inject the Motl under the right kwarg.
        if op["kind"] == "motl_pool":
            motl = _motl_from_pool_rows(pool_motls, motl_id)
            if motl is None:
                return no_update, no_update, "Pick a non-empty motl from the pool."
            kwargs[op["motl_kwarg"]] = motl
            source_tag = f"motl:{motl_id}"
        else:
            source_tag = f"path:{kwargs.get('path', kwargs.get('input_path', '?'))}"

        kwargs = _filter_kwargs_to_signature(method, kwargs)
        try:
            result = run_operation(method, kwargs)
        except Exception as exc:
            return no_update, no_update, f"Load failed: {exc}"

        if op["result"] == "parametric":
            pr.registry.add(result)
            handle = pr.make_handle(result, source=source_tag)
            return no_update, handle, (
                f"Loaded {handle['n_quadrics']} parametric surface(s) "
                f"({source_tag})."
            )

        # Mesh / OPC -> page pool.
        new_entries = _adopt_result(result, parent_id=None, label_root=op["label"])
        for sid, h in new_entries:
            pool[sid] = h
        sid_str = ", ".join(s for s, _ in new_entries)
        return pool, no_update, (
            f"Loaded {len(new_entries)} surface(s): {sid_str}."
        )

    # ── Operations form rendering ────────────────────────────────────────────
    @app.callback(
        Output("surfaces-op-form-wrapper", "children"),
        Output("surfaces-op-input-picker-wrapper", "style"),
        Output("surfaces-op-object-picker-wrapper", "style"),
        Output("surfaces-op-isect-wrapper", "style"),
        Input("surfaces-op-select", "value"),
    )
    def _render_op_form(op_id):
        if not op_id or op_id not in OPERATIONS:
            return (
                html.Div("Pick an operation to render its form.", style=_HINT),
                {"display": "none"}, {"display": "none"}, {"display": "none"},
            )
        op = OPERATIONS[op_id]
        if op["category"] == "intersection":
            return (
                html.Div(),  # intersection's custom UI lives in its own wrapper
                {"display": "none"}, {"display": "none"},
                {"display": "block", "marginBottom": "0.4rem"},
            )
        if op["category"] == "mesh":
            rows = formgen.build_form(
                op["form_method"],
                id_type=_OP_ID_TYPE,
                id_extra={"op": op_id},
                exclude=op.get("exclude", []),
            )
            return (
                html.Div(rows),
                {"display": "none"}, {"display": "none"}, {"display": "none"},
            )
        # parametric
        method = getattr(ParametricSurface, op["method_name"])
        exclude = ["input_motl", "output_path"] + list(op.get("extra_pickers", []))
        rows = formgen.build_form(
            method,
            id_type=_OP_ID_TYPE,
            id_extra={"op": op_id},
            exclude=exclude,
        )
        input_style = {"display": "block", "marginBottom": "0.4rem"}
        object_style = (
            {"display": "block", "marginBottom": "0.4rem"}
            if "object_motl" in op.get("extra_pickers", [])
            else {"display": "none"}
        )
        return (
            html.Div(rows), input_style, object_style, {"display": "none"},
        )

    # ── Operations Run dispatch ──────────────────────────────────────────────
    @app.callback(
        Output("surfaces-pool", "data", allow_duplicate=True),
        Output("surfaces-param-result-motl", "data", allow_duplicate=True),
        Output("surfaces-param-intersection-df", "data", allow_duplicate=True),
        Output("surfaces-isect-result", "data", allow_duplicate=True),
        Output("surfaces-isect-motl-rows", "data", allow_duplicate=True),
        Output("surfaces-scalar-result", "data", allow_duplicate=True),
        Output("surfaces-op-status", "children"),
        Input("surfaces-op-run-btn", "n_clicks"),
        State("surfaces-op-select", "value"),
        State({"type": _OP_ID_TYPE, "owner": ALL, "op": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": _OP_ID_TYPE, "owner": ALL, "op": ALL, "param": ALL, "tag": ALL}, "id"),
        State("surfaces-pool", "data"),
        State("surfaces-selected", "data"),
        State(ids.POOL_MOTLS, "data"),
        State("surfaces-scalar-result", "data"),
        # Parametric pickers.
        State(f"{_PARAM_INPUT_PICKER}-motl-select", "value"),
        State(f"{_PARAM_OBJECT_PICKER}-motl-select", "value"),
        # Intersection inputs.
        State("surfaces-isect-motl-motl-select", "value"),
        State("surfaces-isect-pixel-size", "value"),
        State("surfaces-isect-reverse", "value"),
        State("surfaces-isect-one-hit", "value"),
        State("surfaces-isect-max-dist", "value"),
        State("surfaces-isect-inner-r", "value"),
        State("surfaces-isect-outer-r", "value"),
        prevent_initial_call=True,
    )
    def _run_operation(
        n_clicks, op_id, values, ids, pool, selected_id, pool_motls,
        scalar_state,
        in_motl_id, obj_motl_id,
        isect_motl_id, isect_px, isect_rev, isect_oh,
        isect_maxd, isect_inner, isect_outer,
    ):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not op_id:
            return (no_update,) * 6 + ("Pick an operation first.",)
        op = OPERATIONS[op_id]
        pool = dict(pool or {})

        # ── Mesh op ─────────────────────────────────────────────────────
        if op["category"] == "mesh":
            kwargs = generate_kwargs(ids, values) if (ids and values) else {}
            kwargs = {k: v for k, v in kwargs.items() if v not in (None, "", [])}

            psurf: PleomorphicSurface | None = None
            if op.get("needs_selection"):
                if not selected_id or selected_id not in pool:
                    return (no_update,) * 6 + (
                        "Select a surface from the list first.",)
                psurf = sr.registry.get(selected_id)
                if psurf is None:
                    return (no_update,) * 6 + (
                        f"Surface {selected_id} is no longer in the registry.",)

            method = op["method_for"](psurf)
            if method is None:
                return (no_update,) * 6 + (
                    "This operation is not available for the selected surface.",)
            kwargs = _filter_kwargs_to_signature(method, kwargs)
            try:
                result = run_operation(method, kwargs)
            except Exception as exc:
                return (no_update,) * 6 + (f"Error: {exc}",)

            if op["kind"] == "export":
                tgt = kwargs.get("output_path") or "(unspecified path)"
                return (no_update,) * 6 + (f"Saved to {tgt}.",)
            if op["kind"] == "scalar":
                # Append the (label, value) row to the scalar-result store so
                # the main-area panel can render the accumulated results.
                new_scalar = list(scalar_state or []) + [
                    {"label": op["label"], "value": _fmt_cell(result)}
                ]
                return (no_update,) * 5 + (
                    new_scalar,
                    f"{op['label']} -> see main area.",
                )
            if op["kind"] == "field-source":
                handle = pool.get(selected_id)
                if handle is not None and psurf is not None:
                    handle["has_curvatures"] = sr._mesh_has_curvatures(psurf.surface)
                    pool[selected_id] = handle
                return (pool,) + (no_update,) * 5 + (
                    f"{op['label']} applied; curvature fields populated.",)
            if op["kind"] in ("unary-inplace", "unary") and (result is None or result is psurf.surface):
                handle = pool.get(selected_id)
                if handle is not None and psurf is not None:
                    handle["n_elements"] = (
                        len(psurf.surface.vertices)
                        if psurf.surface.vertices is not None
                        else handle["n_elements"]
                    )
                    pool[selected_id] = handle
                return (pool,) + (no_update,) * 5 + (
                    f"{op['label']} applied in place.",)

            new_entries = _adopt_result(
                result, parent_id=selected_id, label_root=op["label"],
            )
            for sid, h in new_entries:
                pool[sid] = h
            return (pool,) + (no_update,) * 5 + (
                f"Created {len(new_entries)} new surface(s): "
                + ", ".join(s for s, _ in new_entries) + ".",)

        # ── Parametric op ───────────────────────────────────────────────
        if op["category"] == "parametric":
            in_motl = _motl_from_pool_rows(pool_motls, in_motl_id)
            if in_motl is None:
                return (no_update,) * 6 + (
                    "Pick a non-empty input motl from the pool.",)
            kwargs: dict = {"input_motl": in_motl}
            if "object_motl" in op.get("extra_pickers", []):
                obj = _motl_from_pool_rows(pool_motls, obj_motl_id)
                if obj is None:
                    return (no_update,) * 6 + (
                        "Pick a non-empty motl for 'object_motl'.",)
                kwargs["object_motl"] = obj
            scalar_kwargs = generate_kwargs(ids, values) if (ids and values) else {}
            scalar_kwargs = {k: v for k, v in scalar_kwargs.items()
                             if v not in (None, "", [])}
            kwargs.update(scalar_kwargs)
            if op["needs_active_fit"]:
                _pkeys = pr.registry.keys()
                psurf = pr.registry.get(_pkeys[0]) if _pkeys else None
                if psurf is None:
                    return (no_update,) * 6 + (
                        "No active fit -- load a parametric surface first.",)
                method = getattr(psurf, op["method_name"])
            else:
                method = getattr(ParametricSurface, op["method_name"])
            try:
                result = run_operation(method, kwargs)
            except Exception as exc:
                return (no_update,) * 6 + (f"{op['label']} failed: {exc}",)
            if op["result_kind"] == "dataframe":
                if not isinstance(result, pd.DataFrame):
                    return (no_update,) * 6 + (
                        f"{op['label']} did not return a DataFrame.",)
                records = result.to_dict("records")
                return (no_update, no_update, records, no_update, no_update,
                        no_update,
                        f"{op['label']} -> {len(records)} rows; "
                        "see results table.")
            if not isinstance(result, Motl):
                return (no_update,) * 6 + (
                    f"{op['label']} did not return a Motl "
                    f"({type(result).__name__}).",)
            rows = result.df.to_dict("records")
            return (no_update, rows, no_update, no_update, no_update,
                    no_update,
                    f"{op['label']} -> {len(rows)} particles, "
                    "ready to send to editor.")

        # ── Intersection (custom flow) ───────────────────────────────────
        if op["category"] == "intersection":
            if not selected_id:
                return (no_update,) * 6 + ("Select a mesh surface first.",)
            psurf = sr.registry.get(selected_id)
            if psurf is None or not psurf.is_mesh:
                return (no_update,) * 6 + ("Selected surface must be a mesh.",)
            if not isect_motl_id:
                return (no_update,) * 6 + ("Pick a motl from the pool.",)
            rows = (pool_motls or {}).get(isect_motl_id) or []
            if not rows:
                return (no_update,) * 6 + (
                    f"Motl '{isect_motl_id}' has no data.",)
            try:
                rays = motl_rows_to_rays(
                    rows, pixel_size=float(isect_px or 1.0),
                    reverse_direction=bool(isect_rev),
                )
            except Exception as exc:
                return (no_update,) * 6 + (f"Ray construction failed: {exc}",)
            try:
                raw = run_operation(
                    psurf.ray_intersections,
                    {"rays": rays, "one_hit_per_target": bool(isect_oh)},
                )
            except Exception as exc:
                return (no_update,) * 6 + (
                    f"ray_intersections failed: {exc}",)
            radii = sorted({r for r in (float(isect_inner or 0.0),
                                        float(isect_outer or 0.0)) if r > 0})
            try:
                data = run_operation(
                    psurf.intersection_data,
                    {
                        "result": raw, "query_type": "ray",
                        "max_distance_source_target": (
                            float(isect_maxd) if isect_maxd is not None else None
                        ),
                        "surface_radii": radii or None,
                        "include_curvatures": True,
                    },
                )
            except Exception as exc:
                return (no_update,) * 6 + (
                    f"intersection_data failed: {exc}",)
            hits_df = data.get("hits")
            if isinstance(hits_df, pd.DataFrame) and "source_id" in hits_df.columns:
                seen = sorted({int(x) for x in hits_df["source_id"].tolist()})
            else:
                seen = []
            store_value = _result_to_store(data, seen)
            n_hits = len(store_value["hits_records"])
            return (no_update, no_update, no_update, store_value, rows,
                    no_update,
                    f"Cast {len(rays)} rays; {n_hits} hits "
                    f"across {len(radii)} radii.")

        return (no_update,) * 6 + (f"Unknown op category: {op['category']!r}.",)

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
    )
    def _render_isect_results(snap):
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
                if "distance_nm" in hits_df.columns:
                    fig = visplot.plot_histogram(hits_df[["distance_nm"]], bins=30)
                    children.append(dcc.Graph(figure=fig,
                                              style={"height": "320px"}))
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
        try:
            new_psurf = run_operation(
                psurf.extract_region,
                {"indices": np.asarray(idx, dtype=int),
                 "element": "triangles"},
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

    # ── Render intersection-distances results table (parametric) ─────────────
    @app.callback(
        Output("surfaces-param-results-area", "children"),
        Input("surfaces-param-intersection-df", "data"),
    )
    def _render_param_results(records):
        if not records:
            return html.Div(
                "Parametric ops that return motls go to the editor via the "
                "side panel. Ops that return a table (intersection distances) "
                "render here.",
                style=_HINT,
            )
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
                        dbc.Checkbox(
                            id={"type": "surfaces-row-visible", "sid": sid},
                            value=h.get("visible", True),
                            style={"marginRight": "0.4rem"},
                        ),
                        dbc.Button(
                            "×",
                            id={"type": "surfaces-row-delete", "sid": sid},
                            color="link", size="sm",
                            style={"padding": "0 6px", "lineHeight": "1"},
                        ),
                    ],
                    id={"type": "surfaces-row-select", "sid": sid},
                    action=True, n_clicks=0, active=is_sel,
                    style={"display": "flex", "alignItems": "center",
                           "padding": "4px 6px", "cursor": "pointer",
                           "fontSize": "0.85rem"},
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
        Input({"type": "surfaces-row-visible", "sid": ALL}, "value"),
        State({"type": "surfaces-row-visible", "sid": ALL}, "id"),
        State("surfaces-pool", "data"),
        prevent_initial_call=True,
    )
    def _toggle_visible(values, ids, pool):
        pool = dict(pool or {})
        changed = False
        for value, ident in zip(values, ids):
            sid = ident["sid"]
            if sid in pool and bool(pool[sid].get("visible", True)) != bool(value):
                pool[sid]["visible"] = bool(value)
                changed = True
        if not changed:
            raise dash.exceptions.PreventUpdate
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
