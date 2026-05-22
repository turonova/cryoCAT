import base64
import tempfile
import os
import pandas as pd

import dash
from dash import html, dcc, Input, Output, State, no_update, ctx, ALL
import dash_bootstrap_components as dbc
from cryocat.core.cryomotl import Motl
from cryocat.utils.ioutils import dimensions_load
from cryocat.app.components.motlio import get_motl_load_component, register_motl_load_callbacks
from cryocat.app.apputils import generate_motl_op_form, generate_kwargs, get_motl_operation_methods
from cryocat.app.logger import dash_logger

MAX_MOTLS = 8

# Fetched once at import time from the live Motl class – no hardcoding.
_MOTL_METHODS = get_motl_operation_methods()


def get_motl_editor_sidebar():
    return dbc.Col(
        html.Div(
            [
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            [
                                get_motl_load_component("me-load"),
                                html.Div(
                                    id="me-load-status",
                                    style={
                                        "marginTop": "0.5rem",
                                        "fontSize": "0.9rem",
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
                                        style={"color": "var(--color9)", "fontSize": "0.9rem", "padding": "4px"},
                                    ),
                                ),
                            ],
                            title="Loaded Motls",
                            item_id="me-sidebar-list",
                        ),
                        dbc.AccordionItem(
                            [
                                dcc.Dropdown(
                                    id="me-op-func-select",
                                    options=_MOTL_METHODS,
                                    placeholder="Select operation",
                                    style={"marginBottom": "0.5rem"},
                                ),
                                html.Div(id="me-op-func-form", style={"marginBottom": "0.5rem"}),
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
                                    style={"display": "flex", "marginBottom": "0.5rem"},
                                ),
                                html.Div(
                                    id="me-op-status",
                                    style={"fontSize": "0.85rem", "color": "var(--color9)", "wordBreak": "break-word"},
                                ),
                            ],
                            title="Single Motl Operations",
                            item_id="me-sidebar-singlemotl",
                        ),
                        dbc.AccordionItem(
                            [
                                dcc.Dropdown(
                                    id="me-op-motl-select",
                                    multi=True,
                                    placeholder="Select motls",
                                    style={"marginBottom": "0.5rem"},
                                ),
                                dcc.Dropdown(
                                    id="me-op-column-select",
                                    placeholder="Match column (common/unique)",
                                    style={"marginBottom": "0.75rem"},
                                ),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Merge",
                                            id="me-merge-btn",
                                            color="primary",
                                            size="sm",
                                            className="mb-1",
                                            style={"width": "100%"},
                                        ),
                                        dbc.Button(
                                            "Common rows",
                                            id="me-intersect-btn",
                                            color="secondary",
                                            size="sm",
                                            className="mb-1",
                                            style={"width": "100%"},
                                        ),
                                        dbc.Button(
                                            "Unique rows",
                                            id="me-unique-btn",
                                            color="secondary",
                                            size="sm",
                                            style={"width": "100%"},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="me-multiop-status",
                                    style={"marginTop": "0.5rem", "fontSize": "0.9rem", "color": "var(--color9)"},
                                ),
                            ],
                            title="Multi-Motl Operations",
                            item_id="me-sidebar-ops",
                        ),
                    ],
                    always_open=True,
                    active_item=["me-sidebar-load", "me-sidebar-list"],
                ),
                html.Div(
                    dbc.Button(
                        "Show log",
                        id="me-open-log-btn",
                        className="custom-radius-button",
                        style={"width": "100%"},
                    ),
                    style={"padding": "0.5rem", "marginTop": "auto"},
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
        id="me-sidebar",
        width=3,
        style={
            "margin": "0",
            "padding": "0",
            "height": "100vh",
            "position": "sticky",
            "top": "0px",
        },
    )


def register_motl_editor_sidebar_callbacks(app, max_motls=MAX_MOTLS):

    register_motl_load_callbacks(app, "me-load")

    # Route newly loaded motl from the shared load form to the next free slot.
    @app.callback(
        *[Output(f"me-{i}-motl-data-store", "data", allow_duplicate=True) for i in range(max_motls)],
        *[Output(f"me-{i}-motl-extra-data-store", "data", allow_duplicate=True) for i in range(max_motls)],
        *[Output(f"me-{i}-motl-data-type", "data", allow_duplicate=True) for i in range(max_motls)],
        *[Output(f"me-{i}-relion-optics-store", "data", allow_duplicate=True) for i in range(max_motls)],
        *[Output(f"me-{i}-relion5-tomos-store", "data", allow_duplicate=True) for i in range(max_motls)],
        *[Output(f"me-{i}-relion-params-store", "data", allow_duplicate=True) for i in range(max_motls)],
        Output("motls-registry", "data", allow_duplicate=True),
        Output("me-load-status", "children", allow_duplicate=True),
        Input("me-load-motl-data-store", "data"),
        State("me-load-motl-extra-data-store", "data"),
        State("me-load-motl-data-type", "data"),
        State("me-load-relion-optics-store", "data"),
        State("me-load-relion5-tomos-store", "data"),
        State("me-load-motl-upload", "filename"),
        State("me-load-relion-params-store", "data"),
        State("motls-registry", "data"),
        prevent_initial_call=True,
    )
    def route_motl(motl_data, extra_data, data_type, optics, r5tomos, filename, relion_params, registry):
        if not motl_data:
            raise dash.exceptions.PreventUpdate

        registry = registry or {}
        target = None
        for i in range(max_motls):
            if str(i) not in registry or not registry[str(i)].get("active"):
                target = i
                break

        nones = [no_update] * max_motls

        if target is None:
            return (*nones, *nones, *nones, *nones, *nones, *nones, no_update, f"All {max_motls} slots are full.")

        def _fill(value):
            updates = [no_update] * max_motls
            updates[target] = value
            return updates

        new_registry = dict(registry)
        label = filename or f"Motl {target + 1}"
        new_registry[str(target)] = {"label": label, "active": True}

        n_particles = len(motl_data)
        status = f"Loaded: {label} ({n_particles} particles) → slot {target + 1}"

        if relion_params:
            parts = []

            def _scalar_local(v):
                if v is None:
                    return None
                try:
                    if hasattr(v, "__len__"):
                        v = v[0] if len(v) > 0 else None
                    return float(v)
                except (TypeError, ValueError, IndexError):
                    return None

            ps = _scalar_local(relion_params.get("pixel_size"))
            bn = _scalar_local(relion_params.get("binning"))
            if ps is not None:
                parts.append(f"pixel size: {ps:.4g} Å")
            if bn is not None:
                parts.append(f"binning: {bn:.4g}")
            if relion_params.get("tomo_format"):
                parts.append(f"tomo fmt: {relion_params['tomo_format']}")
            if relion_params.get("subtomo_format"):
                parts.append(f"subtomo fmt: {relion_params['subtomo_format']}")
            if parts:
                status += "  |  " + ",  ".join(parts)

        return (
            *_fill(motl_data),
            *_fill(extra_data),
            *_fill(data_type),
            *_fill(optics),
            *_fill(r5tomos),
            *_fill(relion_params),
            new_registry,
            status,
        )

    # Populate the column dropdown from the first loaded motl's columns.
    @app.callback(
        Output("me-op-column-select", "options"),
        Output("me-op-column-select", "value"),
        Input("motls-registry", "data"),
        *[State(f"me-{i}-motl-data-store", "data") for i in range(max_motls)],
        prevent_initial_call=True,
    )
    def update_column_options(registry, *all_data):
        for data in all_data:
            if data:
                cols = list(pd.DataFrame(data).columns)
                options = [{"label": c, "value": c} for c in cols]
                default = "subtomo_id" if "subtomo_id" in cols else cols[0]
                return options, default
        return [], None

    # Refresh the motl list and multi-op selector whenever the registry changes.
    @app.callback(
        Output("me-motl-list", "children"),
        Output("me-op-motl-select", "options"),
        Input("motls-registry", "data"),
        prevent_initial_call=True,
    )
    def update_motl_list(registry):
        if not registry:
            return (
                html.Div("No motls loaded.", style={"color": "var(--color9)", "fontSize": "0.9rem", "padding": "4px"}),
                [],
            )

        items = []
        options = []
        for key in sorted(registry.keys(), key=int):
            meta = registry[key]
            if not meta.get("active"):
                continue
            idx = int(key)
            label = meta.get("label", f"Motl {idx + 1}")
            items.append(
                dbc.ListGroupItem(
                    [
                        html.Span(
                            label,
                            style={
                                "flex": "1",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                                "whiteSpace": "nowrap",
                                "fontSize": "0.9rem",
                            },
                        ),
                        dbc.Button(
                            "×",
                            id={"type": "me-close-motl", "index": idx},
                            color="link",
                            size="sm",
                            style={"padding": "0 6px", "color": "var(--color9)", "lineHeight": "1"},
                        ),
                    ],
                    id={"type": "me-motl-list-item", "index": idx},
                    action=True,
                    n_clicks=0,
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "padding": "4px 8px",
                        "cursor": "pointer",
                    },
                )
            )
            options.append({"label": label, "value": idx})

        if not items:
            return (
                html.Div("No motls loaded.", style={"color": "var(--color9)", "fontSize": "0.9rem", "padding": "4px"}),
                [],
            )

        return dbc.ListGroup(items, flush=True), options

    # Clicking a list item activates the corresponding main tab.
    @app.callback(
        Output("me-tabs", "active_tab"),
        Input({"type": "me-motl-list-item", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def activate_tab_from_list(n_clicks_list):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if triggered and isinstance(triggered, dict) and "index" in triggered:
            return f"me-tab-{triggered['index']}"
        raise dash.exceptions.PreventUpdate

    # Remove a motl slot by clearing its data and marking inactive in registry.
    @app.callback(
        *[Output(f"me-{i}-motl-data-store", "data", allow_duplicate=True) for i in range(max_motls)],
        Output("motls-registry", "data", allow_duplicate=True),
        Input({"type": "me-close-motl", "index": ALL}, "n_clicks"),
        State("motls-registry", "data"),
        prevent_initial_call=True,
    )
    def close_motl(n_clicks_list, registry):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (triggered and isinstance(triggered, dict) and "index" in triggered):
            raise dash.exceptions.PreventUpdate

        idx = triggered["index"]
        slot_data = [no_update] * max_motls
        slot_data[idx] = None

        new_registry = dict(registry or {})
        if str(idx) in new_registry:
            new_registry[str(idx)] = {"label": "", "active": False}

        return (*slot_data, new_registry)

    # Merge selected motls into the results store.
    @app.callback(
        Output("me-results-store", "data", allow_duplicate=True),
        Output("me-results-label-store", "data", allow_duplicate=True),
        Output("me-multiop-status", "children", allow_duplicate=True),
        Input("me-merge-btn", "n_clicks"),
        State("me-op-motl-select", "value"),
        *[State(f"me-{i}-motl-data-store", "data") for i in range(max_motls)],
        prevent_initial_call=True,
    )
    def merge_motls(n_clicks, selected, *all_data):
        if not n_clicks or not selected:
            raise dash.exceptions.PreventUpdate
        dfs = [pd.DataFrame(all_data[s]) for s in selected if all_data[s]]
        if not dfs:
            return no_update, no_update, "No data in selected slots."
        merged = pd.concat(dfs, ignore_index=True)
        label = f"Merged ({len(merged)} particles)"
        slot_names = [f"motl_slot{s+1}" for s in selected if all_data[s]]
        dash_logger.write(f"merged = pd.concat([{', '.join(slot_names)}], ignore_index=True)", source="cryocat")
        return merged.to_dict("records"), label, f"Merged {len(dfs)} motls: {len(merged)} particles total."

    # Find rows whose value in the selected column appears in ALL selected motls.
    @app.callback(
        Output("me-results-store", "data", allow_duplicate=True),
        Output("me-results-label-store", "data", allow_duplicate=True),
        Output("me-multiop-status", "children", allow_duplicate=True),
        Input("me-intersect-btn", "n_clicks"),
        State("me-op-motl-select", "value"),
        State("me-op-column-select", "value"),
        *[State(f"me-{i}-motl-data-store", "data") for i in range(max_motls)],
        prevent_initial_call=True,
    )
    def intersect_motls(n_clicks, selected, match_col, *all_data):
        if not n_clicks or not selected or len(selected) < 2:
            raise dash.exceptions.PreventUpdate
        if not match_col:
            return no_update, no_update, "Select a column to match on."
        dfs = [pd.DataFrame(all_data[s]) for s in selected if all_data[s]]
        if len(dfs) < 2:
            return no_update, no_update, "Need at least 2 motls with data."
        if match_col not in dfs[0].columns:
            return no_update, no_update, f"Column '{match_col}' not found in motl."
        common_vals = set(dfs[0][match_col])
        for df in dfs[1:]:
            common_vals &= set(df[match_col])
        result = dfs[0][dfs[0][match_col].isin(common_vals)].reset_index(drop=True)
        label = f"Common rows ({len(result)} particles)"
        slot_names = [f"motl_slot{s+1}" for s in selected if all_data[s]]
        dash_logger.write(
            f"# Common rows on '{match_col}' across [{', '.join(slot_names)}]\n"
            f"common_vals = set.intersection(*[set(m['{match_col}']) for m in [{', '.join(slot_names)}]])\n"
            f"result = motl_slot{selected[0]+1}[motl_slot{selected[0]+1}['{match_col}'].isin(common_vals)]",
            source="cryocat",
        )
        return (
            result.to_dict("records"),
            label,
            f"Found {len(result)} rows with common '{match_col}' across {len(dfs)} motls.",
        )

    # Find rows whose value in the selected column appears in exactly one motl.
    @app.callback(
        Output("me-results-store", "data", allow_duplicate=True),
        Output("me-results-label-store", "data", allow_duplicate=True),
        Output("me-multiop-status", "children", allow_duplicate=True),
        Input("me-unique-btn", "n_clicks"),
        State("me-op-motl-select", "value"),
        State("me-op-column-select", "value"),
        *[State(f"me-{i}-motl-data-store", "data") for i in range(max_motls)],
        prevent_initial_call=True,
    )
    def unique_motls(n_clicks, selected, match_col, *all_data):
        if not n_clicks or not selected or len(selected) < 2:
            raise dash.exceptions.PreventUpdate
        if not match_col:
            return no_update, no_update, "Select a column to match on."
        dfs = [pd.DataFrame(all_data[s]) for s in selected if all_data[s]]
        if len(dfs) < 2:
            return no_update, no_update, "Need at least 2 motls with data."
        if match_col not in dfs[0].columns:
            return no_update, no_update, f"Column '{match_col}' not found in motl."
        all_val_sets = [set(df[match_col]) for df in dfs]
        common = all_val_sets[0].intersection(*all_val_sets[1:])
        unique_vals = set().union(*all_val_sets) - common
        result = pd.concat([df[df[match_col].isin(unique_vals)] for df in dfs], ignore_index=True)
        label = f"Unique rows ({len(result)} particles)"
        slot_names = [f"motl_slot{s+1}" for s in selected if all_data[s]]
        dash_logger.write(
            f"# Unique rows on '{match_col}' across [{', '.join(slot_names)}]\n"
            f"all_sets = [set(m['{match_col}']) for m in [{', '.join(slot_names)}]]\n"
            f"unique_vals = set.union(*all_sets) - set.intersection(*all_sets)\n"
            f"result = pd.concat([m[m['{match_col}'].isin(unique_vals)] for m in [{', '.join(slot_names)}]])",
            source="cryocat",
        )
        return (
            result.to_dict("records"),
            label,
            f"Found {len(result)} rows with unique '{match_col}' across {len(dfs)} motls.",
        )

    # Generate form when a method is selected from the dropdown.
    @app.callback(
        Output("me-op-func-form", "children"),
        Input("me-op-func-select", "value"),
        prevent_initial_call=True,
    )
    def generate_op_form(method_name):
        if not method_name:
            return []
        return generate_motl_op_form(method_name)

    # Apply the selected method to the active slot's motl.
    @app.callback(
        *[Output(f"me-{i}-motl-data-store", "data", allow_duplicate=True) for i in range(max_motls)],
        *[Output(f"me-{i}-undo-store", "data", allow_duplicate=True) for i in range(max_motls)],
        Output("me-op-status", "children", allow_duplicate=True),
        Input("me-op-apply-btn", "n_clicks"),
        State("me-op-func-select", "value"),
        State("me-tabs", "active_tab"),
        State({"type": "me-op-param", "param": ALL, "p_type": ALL}, "value"),
        State({"type": "me-op-param", "param": ALL, "p_type": ALL}, "id"),
        *[State(f"me-{i}-motl-data-store", "data") for i in range(max_motls)],
        prevent_initial_call=True,
    )
    def apply_operation(n_clicks, method_name, active_tab, param_values, param_ids, *all_slot_data):
        if not n_clicks or not method_name or not active_tab:
            raise dash.exceptions.PreventUpdate

        try:
            slot_idx = int(active_tab.replace("me-tab-", ""))
        except (ValueError, AttributeError):
            raise dash.exceptions.PreventUpdate

        if slot_idx >= max_motls:
            raise dash.exceptions.PreventUpdate

        current_data = all_slot_data[slot_idx]
        if not current_data:
            empty = [no_update] * max_motls
            return (*empty, *empty, "No data in the active slot.")

        kwargs = generate_kwargs(param_ids, param_values) if param_ids else {}

        try:
            motl = Motl(pd.DataFrame(current_data))
            method = getattr(motl, method_name)
            result = method(**kwargs)
        except Exception as exc:
            empty = [no_update] * max_motls
            return (*empty, *empty, f"Error: {exc}")

        if isinstance(result, Motl):
            new_data = result.df.to_dict("records")
        elif result is None:
            new_data = motl.df.to_dict("records")
        else:
            empty = [no_update] * max_motls
            return (*empty, *empty, f"Ran '{method_name}' — result: {result!r} (table unchanged).")

        data_out = [no_update] * max_motls
        data_out[slot_idx] = new_data
        undo_out = [no_update] * max_motls
        undo_out[slot_idx] = current_data

        n_before = len(current_data)
        n_after = len(new_data)
        status = f"'{method_name}' applied. Particles: {n_before} → {n_after}."
        return (*data_out, *undo_out, status)

    # Undo the last operation on the active slot.
    @app.callback(
        *[Output(f"me-{i}-motl-data-store", "data", allow_duplicate=True) for i in range(max_motls)],
        *[Output(f"me-{i}-undo-store", "data", allow_duplicate=True) for i in range(max_motls)],
        Output("me-op-status", "children", allow_duplicate=True),
        Input("me-op-undo-btn", "n_clicks"),
        State("me-tabs", "active_tab"),
        *[State(f"me-{i}-undo-store", "data") for i in range(max_motls)],
        prevent_initial_call=True,
    )
    def undo_operation(n_clicks, active_tab, *all_undo_data):
        if not n_clicks or not active_tab:
            raise dash.exceptions.PreventUpdate

        try:
            slot_idx = int(active_tab.replace("me-tab-", ""))
        except (ValueError, AttributeError):
            raise dash.exceptions.PreventUpdate

        if slot_idx >= max_motls:
            raise dash.exceptions.PreventUpdate

        undo_data = all_undo_data[slot_idx]
        if not undo_data:
            empty = [no_update] * max_motls
            return (*empty, *empty, "Nothing to undo for this slot.")

        data_out = [no_update] * max_motls
        data_out[slot_idx] = undo_data
        undo_out = [no_update] * max_motls
        undo_out[slot_idx] = None  # clear after use — one level of undo

        return (*data_out, *undo_out, "Undo successful.")
