"""Save modals and callbacks for tableview: CSV, motl Save-As, overwrite."""
from __future__ import annotations

import os

import pandas as pd
from dash import html, dcc, Input, Output, State, exceptions, no_update, ctx
import dash_bootstrap_components as dbc

from cryocat.app.apputils import save_motl
from cryocat.app.components.customel import InlineLabeledDropdown, InlineInputForm
from cryocat.app.components.relionopts import (
    get_relion_options,
    register_relion_options_callbacks,
)
from cryocat.app.components._motlio_ops import save_kwargs_from_store


_MOTL_TYPES = [
    {"label": "EM", "value": "emmotl"},
    {"label": "STOPGAP", "value": "stopgap"},
    {"label": "Relion", "value": "relion"},
    {"label": "Dynamo", "value": "dynamo"},
]


# ── Modal builders ─────────────────────────────────────────────────────────────


def get_csv_save_modal(prefix: str) -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Save as CSV")),
            dbc.ModalBody(
                InlineInputForm(
                    id_=f"{prefix}-csv-path",
                    label="Filename:",
                    type="text",
                    placeholder="Full path including .csv extension",
                )
            ),
            dbc.ModalFooter(
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "width": "100%",
                    },
                    children=[
                        html.H5("", id=f"{prefix}-csv-status-label", style={"margin": 0}),
                        dbc.Button("Save", id=f"{prefix}-csv-save-btn", className="ms-auto", n_clicks=0),
                    ],
                )
            ),
        ],
        id=f"{prefix}-csv-modal",
        is_open=False,
    )


def get_motl_save_modal(prefix: str) -> dbc.Modal:
    """Motl Save-As modal using the shared relionopts component."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Save As")),
            dbc.ModalBody(
                html.Div([
                    InlineLabeledDropdown(
                        id_=f"{prefix}-save-type-dropdown",
                        options=_MOTL_TYPES,
                        label="Output type:",
                        multi=False,
                        placeholder="Output type",
                    ),
                    get_relion_options(prefix, for_load=False),
                    InlineInputForm(
                        id_=f"{prefix}-save-path",
                        label="Filename:",
                        type="text",
                        placeholder="Filename (including its path)",
                    ),
                ])
            ),
            dbc.ModalFooter(
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "width": "100%",
                    },
                    children=[
                        html.H5("", id=f"{prefix}-save-status-label", style={"margin": 0}),
                        dbc.Button("Save", id=f"{prefix}-save-file", className="ms-auto", n_clicks=0),
                    ],
                )
            ),
        ],
        id=f"{prefix}-save-modal",
        is_open=False,
    )


def get_overwrite_confirm_modal(prefix: str) -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Confirm overwrite")),
            dbc.ModalBody(html.P(id=f"{prefix}-overwrite-body-text", children="")),
            dbc.ModalFooter([
                dbc.Button("Yes", id=f"{prefix}-overwrite-yes-btn", color="danger", className="me-2"),
                dbc.Button("No", id=f"{prefix}-overwrite-no-btn", color="secondary"),
            ]),
        ],
        id=f"{prefix}-overwrite-modal",
        is_open=False,
    )


# ── CSV callbacks (always present) ────────────────────────────────────────────


def register_tablesave_csv_callbacks(
    app,
    prefix: str,
    *,
    extra_csv_states: list | None = None,
    custom_csv_save_fn=None,
) -> None:
    @app.callback(
        Output(f"{prefix}-csv-modal", "is_open", allow_duplicate=True),
        Input(f"{prefix}-save-csv-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_csv_modal(_):
        return True

    @app.callback(
        Output(f"{prefix}-csv-modal", "is_open", allow_duplicate=True),
        Output(f"{prefix}-csv-status-label", "children"),
        Input(f"{prefix}-csv-save-btn", "n_clicks"),
        State(f"{prefix}-csv-path", "value"),
        State(f"{prefix}-grid", "rowData"),
        *(extra_csv_states or []),
        prevent_initial_call=True,
    )
    def do_csv_save(_, path, grid_data, *extra):
        if not path or not grid_data:
            return no_update, "Specify a filename."
        try:
            if custom_csv_save_fn is not None:
                result = custom_csv_save_fn(path, grid_data, *extra)
                if result is not None:
                    return result
            pd.DataFrame(grid_data).to_csv(path, index=False)
            return False, f"Saved to {path}"
        except Exception as e:
            return no_update, str(e)


# ── Pure helpers ──────────────────────────────────────────────────────────────


def _create_from_selected_data(
    selected_rows: list[dict],
    motl_type: str,
    registry: dict,
    slot_idx: int,
    max_motls: int,
) -> tuple:
    """Compute create-from-selected outputs. Pure.

    Returns (data_out, type_out, new_registry, active_tab) where data_out is
    None when no free slot exists.
    """
    registry = registry or {}
    target = next(
        (i for i in range(max_motls) if str(i) not in registry or not registry[str(i)].get("active")),
        None,
    )
    if target is None:
        return None, None, None, None
    nones = [no_update] * max_motls
    data_out = list(nones)
    type_out = list(nones)
    data_out[target] = selected_rows
    type_out[target] = motl_type
    source_label = registry.get(str(slot_idx), {}).get("label", f"Slot {slot_idx + 1}")
    short = source_label[:15] + "…" if len(source_label) > 15 else source_label
    new_registry = dict(registry)
    new_registry[str(target)] = {"label": f"Sel from {short} ({len(selected_rows)})", "active": True}
    return data_out, type_out, new_registry, f"me-tab-{target}"


# ── Motl-mode save callbacks ──────────────────────────────────────────────────


def register_table_save_callbacks(
    app,
    prefix: str,
    *,
    connected_motl_prefix: str,
    slot_idx: int | None = None,
    max_motls: int | None = None,
    save_dialog_prefix: str | None = None,
) -> None:
    """Register motl Save-As, overwrite, and (optionally) create-from-selected.

    When *save_dialog_prefix* is given, wires the Save As button to open a
    savedialog offcanvas instead of the legacy modal.
    """
    if save_dialog_prefix is not None:
        from cryocat.app.components.savedialog import register_save_dialog_callbacks

        register_save_dialog_callbacks(app, save_dialog_prefix, mode="single")

        @app.callback(
            Output(f"{save_dialog_prefix}-offcanvas", "is_open"),
            Input(f"{prefix}-save-btn", "n_clicks"),
            prevent_initial_call=True,
        )
        def _open_save_offcanvas(_):
            return True

    else:
        register_relion_options_callbacks(
            app,
            prefix,
            for_load=False,
            type_input_id=f"{prefix}-save-type-dropdown",
            connected_motl_prefix=connected_motl_prefix,
        )

        @app.callback(
            Output(f"{prefix}-save-modal", "is_open", allow_duplicate=True),
            Input(f"{prefix}-save-btn", "n_clicks"),
            prevent_initial_call=True,
        )
        def open_save_as_modal(_):
            return True

        _update_registry = slot_idx is not None

        @app.callback(
            Output(f"{prefix}-save-modal", "is_open", allow_duplicate=True),
            Output(f"{prefix}-save-status-label", "children", allow_duplicate=True),
            Output(f"{prefix}-last-save-params-store", "data"),
            *([Output("motls-registry", "data", allow_duplicate=True)] if _update_registry else []),
            Input(f"{prefix}-save-file", "n_clicks"),
            State(f"{prefix}-save-path", "value"),
            State(f"{prefix}-save-type-dropdown", "value"),
            State(f"{prefix}-rln-value", "data"),
            State(f"{connected_motl_prefix}-rln-tomos-store", "data"),
            State(f"{connected_motl_prefix}-motl-extra-data-store", "data"),
            State(f"{connected_motl_prefix}-relion-optics-store", "data"),
            State(f"{prefix}-global-data-store", "data"),
            *([State("motls-registry", "data")] if _update_registry else []),
            prevent_initial_call=True,
        )
        def save_as(*args):
            if _update_registry:
                n_clicks, path, motl_type, rln_state, rln_tomos_orig, extra_df, rln_optics, data, registry = args
            else:
                n_clicks, path, motl_type, rln_state, rln_tomos_orig, extra_df, rln_optics, data = args
                registry = None

            if not path or not motl_type or not data:
                base = (no_update, "Specify output type and filename.", no_update)
                return (*base, no_update) if _update_registry else base

            kwargs = save_kwargs_from_store(rln_state, rln_tomos_orig)
            status = save_motl(
                file_path=path, data_to_save=data, motl_type=motl_type,
                extra_df=extra_df, rln_optics=rln_optics, **kwargs,
            )
            params = {"path": path, "motl_type": motl_type, "rln_state": rln_state, "rln_tomos_orig": rln_tomos_orig}

            if _update_registry:
                new_registry = dict(registry or {})
                entry = new_registry.get(str(slot_idx), {})
                new_registry[str(slot_idx)] = {"label": os.path.basename(path), "active": entry.get("active", True)}
                return False, status, params, new_registry

            return False, status, params

        @app.callback(
            Output(f"{prefix}-overwrite-modal", "is_open"),
            Output(f"{prefix}-overwrite-body-text", "children"),
            Output(f"{prefix}-overwrite-yes-btn", "disabled"),
            Input(f"{prefix}-save-overwrite-btn", "n_clicks"),
            State(f"{prefix}-last-save-params-store", "data"),
            prevent_initial_call=True,
        )
        def open_overwrite_confirm(_, params):
            if not params or not params.get("path"):
                return True, "No output path set. Please use 'Save As' first to specify the file.", True
            return True, f"Overwrite '{params['path']}'?", False

        @app.callback(
            Output(f"{prefix}-overwrite-modal", "is_open", allow_duplicate=True),
            Input(f"{prefix}-overwrite-yes-btn", "n_clicks"),
            Input(f"{prefix}-overwrite-no-btn", "n_clicks"),
            State(f"{prefix}-last-save-params-store", "data"),
            State(f"{connected_motl_prefix}-motl-extra-data-store", "data"),
            State(f"{connected_motl_prefix}-relion-optics-store", "data"),
            State(f"{prefix}-global-data-store", "data"),
            prevent_initial_call=True,
        )
        def do_overwrite_save(_yes, _no, params, extra_df, rln_optics, data):
            if ctx.triggered_id == f"{prefix}-overwrite-no-btn":
                return False
            if not params or not data:
                return False
            kwargs = save_kwargs_from_store(params.get("rln_state"), params.get("rln_tomos_orig"))
            save_motl(
                file_path=params["path"], data_to_save=data, motl_type=params["motl_type"],
                extra_df=extra_df, rln_optics=rln_optics, **kwargs,
            )
            return False

    if slot_idx is not None and max_motls is not None:
        _si, _mm = slot_idx, max_motls

        @app.callback(
            *[Output(f"me-{i}-motl-data-store", "data", allow_duplicate=True) for i in range(_mm)],
            *[Output(f"me-{i}-motl-data-type", "data", allow_duplicate=True) for i in range(_mm)],
            Output("motls-registry", "data", allow_duplicate=True),
            Output("me-tabs", "active_tab", allow_duplicate=True),
            Input(f"{prefix}-create-from-selected-btn", "n_clicks"),
            State(f"{prefix}-grid", "selectedRows"),
            State(f"{connected_motl_prefix}-motl-data-type", "data"),
            State("motls-registry", "data"),
            prevent_initial_call=True,
        )
        def create_from_selected(n_clicks, selected_rows, motl_type, registry):
            if not n_clicks or not selected_rows:
                raise exceptions.PreventUpdate
            data_out, type_out, new_registry, active_tab = _create_from_selected_data(
                selected_rows, motl_type, registry, _si, _mm
            )
            if data_out is None:
                return (*([no_update] * _mm), *([no_update] * _mm), no_update, no_update)
            return (*data_out, *type_out, new_registry, active_tab)
