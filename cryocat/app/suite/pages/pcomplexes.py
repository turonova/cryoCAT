"""Complexes page — multi-subunit complex workflows (NPC first).

Pool-aware tab for the analyses in :class:`cryocat.analysis.structure.NPC`.
Built around a :data:`COMPLEX_TYPES` registry so future complex types
(centrioles, ribosomes, …) can be added by dropping a new entry plus its
ops dict — no rewiring of the page shell required.

Architecture
------------
* Sidebar: complex-type selector + an accordion of the selected complex's
  ops.  Each op accordion item exposes the picker(s) it consumes (suite
  pool motls), the scalar form built by :func:`cryocat.app.formgen.build_form`
  (motl inputs + mask args excluded — see :func:`_op_form` below), a Run
  button, and a per-op status line.
* Main area: a Send-to-editor sink (motl outputs are pushed back to the
  suite pool by clicking the button) plus a results table for the
  array-returning op (``compute_diameter``).

For :meth:`NPC.cluster_subunits_to_rings`, the mask args render as a small
bespoke widget — "Create from shift" exposes the three coord / size
inputs; "Provide masks" exposes the two path inputs.  Same in-memory
mask code path either way (see :func:`_resolve_cluster_kwargs`).

Dispatch
--------
All commit actions (every Run button) route through
:func:`cryocat.app.apputils.run_operation` so they log to the session
script.  No previews.

Contract: exposes :data:`layout` and :func:`register_callbacks(app)`.
"""
from __future__ import annotations

from typing import Any

import dash
from dash import html, dcc, Input, Output, State, ALL, no_update, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd

from cryocat.core.cryomotl import EmMotl, Motl
from cryocat.analysis.structure import NPC
from cryocat.app import formgen
from cryocat.app.apputils import generate_kwargs, run_operation
from cryocat.app.components.logpanel import get_log_panel, register_log_panel_callbacks
from cryocat.app.components.motlsource import (
    get_motl_source, register_motl_source_callbacks,
)
from cryocat.app.components.motlsink import (
    get_send_to_editor_button, register_send_to_editor_callbacks,
)


_HINT = {"fontSize": "0.8rem", "color": "var(--color9)", "margin": "0.3rem 0"}
_SECTION_HEADER = {"fontSize": "0.95rem", "fontWeight": 600, "margin": "0.4rem 0 0.2rem"}


# ── NPC ops registry (option C — bind forms to the typed staticmethods) ──────
#
# Each entry maps an op id to:
#   label           : sidebar accordion title.
#   method_name     : NPC @staticmethod looked up via getattr.
#   pickers         : list of (kwarg_name, multi) pairs — each renders one
#                     motl-source picker; the kwarg name becomes the key in
#                     the kwargs dict passed to the method.
#   excluded_params : extra kwarg names to drop from build_form (because the
#                     page supplies them via pickers or via the bespoke
#                     mask-source widget; not because they're hidden from
#                     the user).
#   result_kind     : "motl"    → push to editor pool via motlsink.
#                     "motls"   → push every Motl in the returned list.
#                     "diameter"→ render in the diameter results table.
#   needs_mask_widget : bool — when True, the mask-source toggle is rendered
#                     between the pickers and the form.
NPC_OPS: dict[str, dict] = {
    "cluster_subunits": {
        "label": "Cluster subunits → rings",
        "method_name": "cluster_subunits_to_rings",
        "pickers": [("input_motl", False)],
        "excluded_params": (
            "input_motl", "entry_mask", "exit_mask",
            "entry_mask_coord", "exit_mask_coord", "mask_size",
        ),
        "result_kind": "motl",
        "needs_mask_widget": True,
    },
    "unify_orientations": {
        "label": "Unify NN orientations",
        "method_name": "unify_nn_orientations",
        "pickers": [("input_motl", False)],
        "excluded_params": ("input_motl",),
        "result_kind": "motl",
        "needs_mask_widget": False,
    },
    "merge_subunits": {
        "label": "Merge subunits",
        "method_name": "merge_subunits",
        "pickers": [("input_motl", False)],
        "excluded_params": ("input_motl",),
        "result_kind": "motl",
        "needs_mask_widget": False,
    },
    "merge_rings": {
        "label": "Merge rings (across motls)",
        "method_name": "merge_rings",
        "pickers": [("input_motls", True)],
        "excluded_params": ("input_motls",),
        "result_kind": "motls",
        "needs_mask_widget": False,
    },
    "centers": {
        "label": "Centers → motl",
        "method_name": "get_centers_as_motl",
        "pickers": [("tomo_motl", False)],
        "excluded_params": ("tomo_motl",),
        "result_kind": "motl",
        "needs_mask_widget": False,
    },
    "diameter": {
        "label": "NPC diameter",
        "method_name": "compute_diameter",
        "pickers": [("input_motl", False)],
        "excluded_params": ("input_motl",),
        "result_kind": "diameter",
        "needs_mask_widget": False,
    },
}


# ── COMPLEX_TYPES registry ───────────────────────────────────────────────────


COMPLEX_TYPES: dict[str, dict] = {
    "npc": {
        "label": "NPC (Nuclear Pore Complex)",
        "namespace": NPC,
        "ops": NPC_OPS,
        "id_prefix": "complexes-npc",
    },
    # Future complex types: add one entry here + its OPS dict above.
}

DEFAULT_COMPLEX_ID = "npc"


# ── Pure helpers (unit-testable) ─────────────────────────────────────────────


def motl_from_pool_rows(pool_motls: dict | None, motl_id: str | None) -> Motl | None:
    """Reconstruct an :class:`EmMotl` from the suite-pool row list.

    Returns ``None`` for missing / empty entries so the caller can show a
    targeted error message.
    """
    if not motl_id:
        return None
    rows = (pool_motls or {}).get(motl_id) or []
    if not rows:
        return None
    return EmMotl(pd.DataFrame(rows))


def motl_to_pool_rows(motl: Motl) -> list[dict]:
    """Convert a :class:`Motl` to the suite pool's row-dict format."""
    if motl is None:
        return []
    return motl.df.to_dict("records")


def resolve_mask_kwargs(
    mode: str,
    mask_size: Any,
    entry_coord: Any,
    exit_coord: Any,
    entry_path: str | None,
    exit_path: str | None,
) -> dict:
    """Turn the mask-source widget state into kwargs for cluster_subunits_to_rings.

    ``mode`` is ``"shift"`` for the create-from-coord branch and ``"paths"``
    for user-supplied masks.  Empty paths / missing coords are dropped so
    the validation in :meth:`NPC.cluster_subunits_to_rings` surfaces a
    clear error message.
    """
    if mode == "paths":
        out: dict = {}
        if entry_path and str(entry_path).strip():
            out["entry_mask"] = str(entry_path).strip()
        if exit_path and str(exit_path).strip():
            out["exit_mask"] = str(exit_path).strip()
        return out

    # Default "shift" branch: feed coord + size kwargs straight through.
    out: dict = {}
    if mask_size not in (None, ""):
        out["mask_size"] = mask_size
    if entry_coord:
        out["entry_mask_coord"] = entry_coord
    if exit_coord:
        out["exit_mask_coord"] = exit_coord
    return out


# ── Layout helpers ───────────────────────────────────────────────────────────


def _triplet_input(input_id: str, placeholder: str = "x,y,z") -> dcc.Input:
    return dcc.Input(
        id=input_id, type="text", placeholder=placeholder,
        style={"width": "100%", "fontSize": "0.85rem"},
    )


def _parse_triplet(text: str | None) -> tuple[int, int, int] | None:
    if not text:
        return None
    parts = [p.strip() for p in str(text).replace(";", ",").split(",")]
    parts = [p for p in parts if p]
    if len(parts) != 3:
        return None
    try:
        return tuple(int(round(float(p))) for p in parts)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _mask_source_widget(prefix: str) -> html.Div:
    """Bespoke widget: toggle between coord-based and path-based mask sources.

    The page reads:

    * ``f"{prefix}-mode"`` -- the selected branch ("shift" / "paths").
    * ``f"{prefix}-size"`` -- mask box size (single int or x,y,z triplet).
    * ``f"{prefix}-entry-coord"`` / ``f"{prefix}-exit-coord"`` -- centre
      coords (x,y,z triplets) for the shift branch.
    * ``f"{prefix}-entry-path"`` / ``f"{prefix}-exit-path"`` -- mask paths
      for the paths branch.
    """
    return html.Div(
        [
            html.Small("Mask source", style=_HINT),
            dcc.RadioItems(
                id=f"{prefix}-mode",
                options=[
                    {"label": " Create from shift (coord + size)", "value": "shift"},
                    {"label": " Provide masks (paths)", "value": "paths"},
                ],
                value="shift",
                style={"fontSize": "0.85rem"},
                inputStyle={"marginRight": "0.25rem"},
                labelStyle={"display": "block", "marginBottom": "0.2rem"},
            ),
            html.Div(
                [
                    html.Small("Mask size (single int or x,y,z)", style=_HINT),
                    _triplet_input(f"{prefix}-size", "72 or 72,72,72"),
                    html.Small("Entry mask centre (x,y,z)", style=_HINT),
                    _triplet_input(f"{prefix}-entry-coord", "e.g. 34,61,36"),
                    html.Small("Exit mask centre (x,y,z)", style=_HINT),
                    _triplet_input(f"{prefix}-exit-coord", "e.g. 34,17,36"),
                ],
                id=f"{prefix}-shift-section",
                style={"display": "block"},
            ),
            html.Div(
                [
                    html.Small("Entry mask path (.em / .mrc)", style=_HINT),
                    dcc.Input(id=f"{prefix}-entry-path", type="text",
                              placeholder="path/to/entry_mask.em",
                              style={"width": "100%", "fontSize": "0.85rem"}),
                    html.Small("Exit mask path (.em / .mrc)", style=_HINT),
                    dcc.Input(id=f"{prefix}-exit-path", type="text",
                              placeholder="path/to/exit_mask.em",
                              style={"width": "100%", "fontSize": "0.85rem"}),
                ],
                id=f"{prefix}-paths-section",
                style={"display": "none"},
            ),
        ]
    )


def _picker_label(name: str, multi: bool) -> str:
    suffix = " (multi-select)" if multi else ""
    return f"{name}{suffix}"


def _op_form(complex_id: str, op_id: str) -> list:
    """Build the per-op scalar form, excluding the picker/mask kwargs."""
    complex_def = COMPLEX_TYPES[complex_id]
    op = complex_def["ops"][op_id]
    method = getattr(complex_def["namespace"], op["method_name"])
    return formgen.build_form(
        method,
        id_type=f"{complex_def['id_prefix']}-{op_id}-param",
        id_extra={"op": op_id, "complex": complex_id},
        exclude=list(op["excluded_params"]),
    )


def _op_accordion_item(complex_id: str, op_id: str) -> dbc.AccordionItem:
    complex_def = COMPLEX_TYPES[complex_id]
    op = complex_def["ops"][op_id]
    pfx = complex_def["id_prefix"]
    picker_blocks: list = []
    for picker_name, multi in op["pickers"]:
        picker_blocks.append(html.Div(
            [
                html.Label(_picker_label(picker_name, multi),
                           style={"fontSize": "0.85rem", "fontWeight": "bold"}),
                get_motl_source(f"{pfx}-{op_id}-{picker_name}", multi=multi),
            ],
            style={"marginBottom": "0.4rem"},
        ))
    mask_widget = (
        [_mask_source_widget(f"{pfx}-{op_id}-mask")]
        if op["needs_mask_widget"]
        else []
    )
    return dbc.AccordionItem(
        [
            *picker_blocks,
            *mask_widget,
            html.Hr(style={"margin": "0.4rem 0"}),
            html.Div(_op_form(complex_id, op_id)),
            html.Hr(style={"margin": "0.4rem 0"}),
            dbc.Button(
                "Run", color="primary", size="sm",
                id={"type": "complexes-op-run", "complex": complex_id, "op": op_id},
                style={"width": "100%"},
            ),
            html.Div(
                id={"type": "complexes-op-status", "complex": complex_id, "op": op_id},
                style={**_HINT, "marginTop": "0.4rem", "wordBreak": "break-word"},
            ),
        ],
        title=op["label"],
        item_id=f"{pfx}-{op_id}",
    )


def _ops_accordion(complex_id: str) -> dbc.Accordion:
    return dbc.Accordion(
        [_op_accordion_item(complex_id, op_id)
         for op_id in COMPLEX_TYPES[complex_id]["ops"]],
        always_open=True,
        active_item=list(COMPLEX_TYPES[complex_id]["ops"])[:1],
        id=f"complexes-{complex_id}-accordion",
    )


def _sidebar() -> dbc.Col:
    return dbc.Col(
        html.Div(
            [
                html.Div("Complex type", style=_SECTION_HEADER),
                dcc.Dropdown(
                    id="complexes-type-select",
                    options=[{"label": d["label"], "value": cid}
                             for cid, d in COMPLEX_TYPES.items()],
                    value=DEFAULT_COMPLEX_ID,
                    clearable=False,
                    style={"fontSize": "0.85rem", "marginBottom": "0.5rem"},
                ),
                html.Div(
                    [
                        html.Div(
                            _ops_accordion(cid),
                            id=f"complexes-{cid}-ops",
                            style={"display": "block" if cid == DEFAULT_COMPLEX_ID
                                   else "none"},
                        )
                        for cid in COMPLEX_TYPES
                    ],
                ),
            ],
            className="sidebar",
            style={
                "padding": "0.5rem",
                "overflowY": "auto",
                "height": "100vh",
                "display": "flex",
                "flexDirection": "column",
            },
        ),
        width=4,
        style={"margin": 0, "padding": 0, "height": "100vh",
               "position": "sticky", "top": "0px"},
    )


def _main() -> dbc.Col:
    return dbc.Col(
        html.Div(
            [
                html.Div("Result motl(s)", style=_SECTION_HEADER),
                get_send_to_editor_button("complexes-export"),
                html.Div(
                    id="complexes-export-extra",
                    style={**_HINT, "marginTop": "0.3rem"},
                ),
                dcc.Store(id="complexes-result-motl"),
                dcc.Store(id="complexes-result-motls"),
                html.Hr(style={"margin": "0.6rem 0"}),
                html.Div("Diameter results", style=_SECTION_HEADER),
                html.Div(
                    id="complexes-diameter-table",
                    style={"padding": "0.25rem"},
                ),
                dcc.Store(id="complexes-diameter-store"),
            ],
            style={"padding": "0.5rem"},
        ),
        width=8,
        style={"margin": 0, "padding": 0},
    )


layout = html.Div(
    [
        dbc.Row([_sidebar(), _main()], className="g-0",
                style={"margin": 0, "padding": 0}),
        *get_log_panel("complexes-log"),
    ],
    style={"margin": 0, "padding": 0},
)


# ── Callbacks ────────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:
    register_log_panel_callbacks(app, "complexes-log")
    register_send_to_editor_callbacks(app, "complexes-export", "complexes-result-motl")

    # Per-op picker callbacks (motlsource wiring).
    for complex_id, complex_def in COMPLEX_TYPES.items():
        for op_id, op in complex_def["ops"].items():
            for picker_name, multi in op["pickers"]:
                register_motl_source_callbacks(
                    app,
                    f"{complex_def['id_prefix']}-{op_id}-{picker_name}",
                    multi=multi,
                )

    # Show only the ops block for the active complex type.
    # Dash treats a single Output specially: the callback must return a single
    # value, not a 1-tuple. With multiple Outputs it expects a tuple. Branch on
    # the registry size so adding a second complex type just works.
    @app.callback(
        *[Output(f"complexes-{cid}-ops", "style") for cid in COMPLEX_TYPES],
        Input("complexes-type-select", "value"),
    )
    def _switch_complex(active):
        styles = [
            {"display": "block" if cid == active else "none"}
            for cid in COMPLEX_TYPES
        ]
        return styles[0] if len(styles) == 1 else tuple(styles)

    # Show / hide the mask-source widget branches (NPC-only for now).
    @app.callback(
        Output("complexes-npc-cluster_subunits-mask-shift-section", "style"),
        Output("complexes-npc-cluster_subunits-mask-paths-section", "style"),
        Input("complexes-npc-cluster_subunits-mask-mode", "value"),
    )
    def _toggle_mask_sections(mode):
        if mode == "paths":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    # ── Render the diameter results table ────────────────────────────────────
    @app.callback(
        Output("complexes-diameter-table", "children"),
        Input("complexes-diameter-store", "data"),
    )
    def _render_diameter(records):
        if not records:
            return html.Small(
                "Run \"NPC diameter\" to populate this table.",
                style=_HINT,
            )
        df = pd.DataFrame(records)
        if df.empty:
            return html.Small(
                "No opposite-pair matches in the input motl.",
                style=_HINT,
            )
        header = [html.Tr([
            html.Th("tomo_id"), html.Th("object_id"),
            html.Th("mean_diameter"), html.Th("n_pairs"),
        ])]
        rows = [
            html.Tr([
                html.Td(f"{r['tomo_id']:.0f}"),
                html.Td(f"{r['object_id']:.0f}"),
                html.Td(f"{r['mean_diameter']:.3f}"),
                html.Td(f"{r['n_pairs']}"),
            ])
            for r in records
        ]
        return dbc.Table(
            header + rows,
            bordered=True, striped=True, hover=True, size="sm",
            style={"fontSize": "0.85rem"},
        )

    # ── Render the extra-motls list (merge_rings produces > 1 motl) ──────────
    @app.callback(
        Output("complexes-export-extra", "children"),
        Input("complexes-result-motls", "data"),
        State("pool-registry", "data"),
        State("pool-motls", "data"),
        State("pool-next-id", "data"),
    )
    def _show_motls_summary(motls_rows, _registry, _pool_motls, _next_id):
        if not motls_rows:
            return ""
        n = len(motls_rows)
        return html.Small(
            f"{n} motl(s) ready (\"merge_rings\"). Use the buttons below to push each.",
            style=_HINT,
        )

    # ── Run callbacks: one per complex × op ──────────────────────────────────
    for complex_id, complex_def in COMPLEX_TYPES.items():
        for op_id in complex_def["ops"]:
            _register_run(app, complex_id, op_id)

    # ── merge_rings: push each motl in the list to the pool ──────────────────
    @app.callback(
        Output("pool-registry", "data", allow_duplicate=True),
        Output("pool-motls", "data", allow_duplicate=True),
        Output("pool-next-id", "data", allow_duplicate=True),
        Output("complexes-export-extra", "children", allow_duplicate=True),
        Input("complexes-result-motls", "data"),
        State("pool-registry", "data"),
        State("pool-motls", "data"),
        State("pool-next-id", "data"),
        prevent_initial_call=True,
    )
    def _push_motls_list(motls_rows, registry, pool_motls, next_id):
        if not motls_rows:
            raise PreventUpdate
        registry = dict(registry or {})
        pool_motls = dict(pool_motls or {})
        next_id = int(next_id or 0)
        pushed: list[str] = []
        for i, rows in enumerate(motls_rows):
            if not rows:
                continue
            mid = f"motl-{next_id}"
            registry[mid] = {
                "label": f"merge_rings-{i + 1}",
                "type": "emmotl",
                "n_rows": len(rows),
                "active": True,
            }
            pool_motls[mid] = rows
            pushed.append(mid)
            next_id += 1
        msg = html.Small(
            f"Pushed {len(pushed)} motl(s) to the pool: {', '.join(pushed)}.",
            style=_HINT,
        )
        return registry, pool_motls, next_id, msg


def _register_run(app: dash.Dash, complex_id: str, op_id: str) -> None:
    """Register the per-op Run button callback.

    Defined out-of-line so the closure captures ``complex_id`` and
    ``op_id`` cleanly across the COMPLEX_TYPES × OPS product.
    """
    complex_def = COMPLEX_TYPES[complex_id]
    op = complex_def["ops"][op_id]
    pfx = complex_def["id_prefix"]
    id_type = f"{pfx}-{op_id}-param"

    picker_state_specs: list = []
    for picker_name, _ in op["pickers"]:
        picker_state_specs.append(State(
            f"{pfx}-{op_id}-{picker_name}-motl-select", "value",
        ))

    mask_states: list = []
    if op["needs_mask_widget"]:
        mp = f"{pfx}-{op_id}-mask"
        mask_states = [
            State(f"{mp}-mode", "value"),
            State(f"{mp}-size", "value"),
            State(f"{mp}-entry-coord", "value"),
            State(f"{mp}-exit-coord", "value"),
            State(f"{mp}-entry-path", "value"),
            State(f"{mp}-exit-path", "value"),
        ]

    @app.callback(
        Output("complexes-result-motl", "data", allow_duplicate=True),
        Output("complexes-result-motls", "data", allow_duplicate=True),
        Output("complexes-diameter-store", "data", allow_duplicate=True),
        Output({"type": "complexes-op-status",
                "complex": complex_id, "op": op_id}, "children"),
        Input({"type": "complexes-op-run",
               "complex": complex_id, "op": op_id}, "n_clicks"),
        State("pool-motls", "data"),
        State({"type": id_type, "op": op_id, "complex": complex_id,
               "param": ALL, "tag": ALL}, "value"),
        State({"type": id_type, "op": op_id, "complex": complex_id,
               "param": ALL, "tag": ALL}, "id"),
        *picker_state_specs,
        *mask_states,
        prevent_initial_call=True,
    )
    def _run(n_clicks, pool_motls, values, ids, *extras):
        if not n_clicks:
            raise PreventUpdate

        # Split the *extras tuple back into pickers + mask state in order.
        n_pickers = len(op["pickers"])
        picker_values = list(extras[:n_pickers])
        mask_values = list(extras[n_pickers:]) if op["needs_mask_widget"] else []

        kwargs: dict[str, Any] = {}
        for (picker_name, multi), value in zip(op["pickers"], picker_values):
            if multi:
                ids_list = value or []
                motls: list[Motl] = []
                for mid in ids_list:
                    m = motl_from_pool_rows(pool_motls, mid)
                    if m is None:
                        return (no_update, no_update, no_update,
                                f"Pick non-empty motl(s) for '{picker_name}'.")
                    motls.append(m)
                if len(motls) < 2 and op["method_name"] == "merge_rings":
                    return (no_update, no_update, no_update,
                            "merge_rings needs at least two motls in the picker.")
                kwargs[picker_name] = motls
            else:
                motl = motl_from_pool_rows(pool_motls, value)
                if motl is None:
                    return (no_update, no_update, no_update,
                            f"Pick a non-empty motl for '{picker_name}'.")
                kwargs[picker_name] = motl

        scalar_kwargs = generate_kwargs(ids, values) if (ids and values) else {}
        scalar_kwargs = {k: v for k, v in scalar_kwargs.items()
                         if v not in (None, "", [])}
        kwargs.update(scalar_kwargs)

        if op["needs_mask_widget"]:
            mode, size, e_coord, x_coord, e_path, x_path = mask_values
            kwargs.update(resolve_mask_kwargs(
                mode or "shift",
                _parse_triplet(size) if size and "," in str(size) else (
                    int(round(float(size))) if size not in (None, "") else None
                ),
                _parse_triplet(e_coord), _parse_triplet(x_coord),
                e_path, x_path,
            ))

        # Diameter returns (summary_df, motl_out) — the summary is rendered
        # in the results table and the motl (with the diameter copied into
        # `store_column`) is queued for the Send-to-editor button so the
        # user can push it to the suite pool.
        if op["result_kind"] == "diameter":
            try:
                summary_df, motl_out = run_operation(
                    NPC.compute_diameter, kwargs,
                )
            except Exception as exc:
                return no_update, no_update, no_update, f"{op['label']} failed: {exc}"
            if summary_df.empty:
                return no_update, no_update, no_update, (
                    f"{op['label']} → no opposite pairs found."
                )
            rows = motl_to_pool_rows(motl_out)
            return (
                rows, no_update,
                summary_df.to_dict("records"),
                f"{op['label']} → {len(summary_df)} NPC(s); diameter written to "
                f"'{kwargs.get('store_column', 'geom4')}'. Results table updated; "
                "send-to-editor ready.",
            )

        method = getattr(complex_def["namespace"], op["method_name"])
        try:
            result = run_operation(method, kwargs)
        except Exception as exc:
            return no_update, no_update, no_update, f"{op['label']} failed: {exc}"

        if op["result_kind"] == "motl":
            if not isinstance(result, Motl):
                return (no_update, no_update, no_update,
                        f"{op['label']} did not return a Motl "
                        f"({type(result).__name__}).")
            rows = motl_to_pool_rows(result)
            return (rows, no_update, no_update,
                    f"{op['label']} → {len(rows)} particles, ready to send to editor.")

        if op["result_kind"] == "motls":
            if not isinstance(result, list):
                return (no_update, no_update, no_update,
                        f"{op['label']} did not return a list of Motls.")
            motls_rows = [motl_to_pool_rows(m) for m in result]
            return (no_update, motls_rows, no_update,
                    f"{op['label']} → {len(motls_rows)} motls queued for push.")

        return (no_update, no_update, no_update,
                f"{op['label']}: unknown result_kind={op['result_kind']!r}.")

    _run.__name__ = f"_run_{complex_id}_{op_id}"
