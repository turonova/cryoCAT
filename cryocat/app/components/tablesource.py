"""Table-source selector component.

Two-mode selector: "Compute from motl" (existing compute form) or "Load from
file" (path field + optional extra controls + Load button).  The component
stores the loaded data in a ``dcc.Store`` keyed ``{prefix}-ts-loaded`` so page
callbacks can react to it without knowing how the data was produced.

Component IDs
-------------
``{prefix}-ts-radio``          — source RadioItems
``{prefix}-ts-compute-panel``  — Div wrapping compute_children
``{prefix}-ts-file-panel``     — Div wrapping the file-load controls
``{prefix}-ts-load-btn``       — Load button
``{prefix}-ts-status``         — status Div inside the file panel
``{prefix}-ts-loaded``         — dcc.Store; data = {"df": "<json orient=split>", ...metadata}

Extra-field convention
----------------------
Children in *extra_file_children* whose id is a dict with
``{"type": f"{prefix}-ts-extra", "param": "<name>"}`` will have their value
collected by the Load callback and forwarded as keyword arguments to load_fn.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd
import pathlib

from dash import html, dcc, ALL, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from cryocat.app import styles
from cryocat.app.apputils import run_operation
from cryocat.app.components.pathfield import get_path_field


def _build_load_store(result, path: str, check_fn) -> tuple[str, dict | None]:
    """Validate a load_fn result dict; return (status_message, store_or_None).

    Returns None as the second element when the result is invalid or fails
    the column check so the callback can return ``no_update`` in that case.
    """
    if result is None:
        return "Load failed.", None
    df = result.get("df")
    if df is None or not isinstance(df, pd.DataFrame):
        return "load_fn did not return a DataFrame.", None
    missing = check_fn(df)
    if missing:
        return f"Missing columns: {', '.join(missing)}", None
    status = f"Loaded {len(df):,} rows from {pathlib.Path(path).name}"
    status_extra = result.get("status_extra")
    if status_extra:
        status += f". {status_extra}"
    store = {k: v for k, v in result.items() if k not in ("df", "status_extra")}
    store["df"] = df.to_json(orient="split")
    return status, store


def get_table_source(
    prefix: str,
    *,
    compute_children: list,
    file_extensions: tuple[str, ...] = (".csv",),
    extra_file_children: list | None = None,
    label: str = "Source",
) -> html.Div:
    """Return the table-source selector layout (no callbacks; call register_table_source_callbacks).

    Parameters
    ----------
    prefix:
        Unique ID namespace (e.g. ``"nn-src"`` or ``"tango-twist-src"``).
    compute_children:
        Controls shown when "Compute from motl" is selected.
    file_extensions:
        File types accepted by the Browse dialog (e.g. ``(".csv", ".pkl")``).
    extra_file_children:
        Additional controls shown in the file panel below the path field.
        Any component whose ``id`` is ``{"type": f"{prefix}-ts-extra", "param": "<name>"}``
        will have its ``value`` forwarded to load_fn as a keyword argument.
    label:
        Label shown on the source-selector row.
    """
    extra = extra_file_children or []

    radio = dbc.RadioItems(
        id=f"{prefix}-ts-radio",
        options=[
            {"label": "Compute from motl", "value": "compute"},
            {"label": "Load from file", "value": "file"},
        ],
        value="compute",
        inline=True,
        inputStyle=styles.RADIO_INLINE_INPUT,
        labelStyle=styles.RADIO_INLINE_LABEL,
        style={"display": "flex", "gap": "1.5rem", "fontSize": styles.FONT_SM},
    )

    from cryocat.app.formgen import form_row

    radio_row = form_row(
        label, radio,
        "Choose how to supply the data table.",
        label_id=f"{prefix}-ts-source-lbl",
    )

    file_panel = html.Div(
        [
            get_path_field(
                f"{prefix}-ts-path",
                mode="open",
                kind="",
                extensions=file_extensions,
                placeholder="Select file…",
            ),
            *extra,
            html.Div(
                dbc.Button(
                    "Load",
                    id=f"{prefix}-ts-load-btn",
                    color=styles.BTN_PRIMARY,
                    size="sm",
                    style={"marginTop": "0.3rem"},
                ),
            ),
            html.Div(id=f"{prefix}-ts-status", style={**styles.HINT, "marginTop": "0.2rem"}),
        ],
        id=f"{prefix}-ts-file-panel",
        style={"display": "none"},
    )

    return html.Div(
        [
            dcc.Store(id=f"{prefix}-ts-loaded"),
            radio_row,
            html.Div(
                compute_children,
                id=f"{prefix}-ts-compute-panel",
            ),
            file_panel,
        ]
    )


def register_table_source_callbacks(
    app,
    prefix: str,
    *,
    check_fn: Callable[[pd.DataFrame], list[str]],
    load_fn: Callable[..., dict],
) -> None:
    """Register the two callbacks needed by the source-selector component.

    Parameters
    ----------
    app:
        The Dash app instance.
    prefix:
        Same prefix passed to :func:`get_table_source`.
    check_fn:
        ``check_fn(df) -> list[str]`` — return column names missing from *df*.
        An empty list means the table is valid.
    load_fn:
        ``load_fn(path, **extra_kwargs) -> dict`` where the returned dict must
        contain ``"df": pd.DataFrame``.  An optional ``"status_extra": str`` key
        is appended to the success message.  All other keys are forwarded
        verbatim into the ``{prefix}-ts-loaded`` store.
        The call is routed through :func:`run_operation` for session provenance.
    """
    from dash import callback, callback_context as ctx
    from dash.dependencies import Input, Output, State

    @app.callback(
        Output(f"{prefix}-ts-compute-panel", "style"),
        Output(f"{prefix}-ts-file-panel", "style"),
        Input(f"{prefix}-ts-radio", "value"),
        prevent_initial_call=False,
    )
    def _toggle_panels(source):
        if source == "file":
            return {"display": "none"}, {}
        return {}, {"display": "none"}

    @app.callback(
        Output(f"{prefix}-ts-status", "children"),
        Output(f"{prefix}-ts-loaded", "data"),
        Input(f"{prefix}-ts-load-btn", "n_clicks"),
        State({"type": "path-input", "owner": f"{prefix}-ts-path"}, "value"),
        State({"type": f"{prefix}-ts-extra", "param": ALL}, "value"),
        prevent_initial_call=True,
    )
    def _load(_n_clicks, path, extra_values):
        if not path:
            return "No file selected.", no_update
        extra_ids = [s["id"]["param"] for s in ctx.states_list[1]]
        extra_kwargs = {pid: val for pid, val in zip(extra_ids, extra_values) if val is not None}
        try:
            result = run_operation(load_fn, {"path": path, **extra_kwargs})
        except Exception as exc:
            return f"Error: {exc}", no_update
        status, store = _build_load_store(result, path, check_fn)
        return status, (store if store is not None else no_update)
