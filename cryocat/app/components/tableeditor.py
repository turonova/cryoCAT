"""Table editor component — transformation panel for motl and data-pool entries (W1–W7).

Layout
------
``get_table_editor(prefix, *, multi_source)`` returns the form Div that goes
inside an Offcanvas or modal.  It includes a source entry picker
(:mod:`entrypicker`) so the user can select any motl or DataFrame-kind data
pool entry as the operation's input.

``register_table_editor_callbacks(app, prefix, *, multi_source)`` wires all
callbacks.

Source types and result routing (W4)
-------------------------------------
Source = motl pool entry (``{"motl_id": …}``):
  • result satisfies motl schema  → inserted into motl pool; motl editor updates
  • result drops required column  → inserted into data pool; status states why

Source = data pool DataFrame entry (``{"data_id": …}``):
  • result satisfies motl schema AND "Add to motl pool" checked → both pools
  • otherwise                                                   → data pool only

Commit path (W6)
----------------
Every Apply click routes through :func:`~apputils.run_operation`.  DataFrame
arguments render as a placeholder in the logged script line — a known
limitation of ``_render_value`` in logger.py for server-side payloads.
"""
from __future__ import annotations

from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from cryocat.app import ids, styles, formgen
from cryocat.app import datapool as dp_module
from cryocat.app.apputils import run_operation
from cryocat.app.components import _tableops as ops
from cryocat.app.components import entrypicker
from cryocat.app.pool import replace_motl_rows


# ── Operation lists ────────────────────────────────────────────────────────────

_OPS_SINGLE: list[dict] = [
    {"label": "Derive column",   "value": "derive"},
    {"label": "Rename columns",  "value": "rename"},
    {"label": "Drop columns",    "value": "drop"},
    {"label": "Reorder columns", "value": "reorder"},
    {"label": "Cast column",     "value": "cast"},
]

_OPS_MULTI: list[dict] = _OPS_SINGLE + [
    {"label": "Merge tables",  "value": "merge"},
    {"label": "Concat tables", "value": "concat"},
]

_ALL_SECTIONS: list[str] = ["derive", "rename", "drop", "reorder", "cast", "merge", "concat"]


# ── Layout ─────────────────────────────────────────────────────────────────────

def get_table_editor(
    prefix: str,
    *,
    multi_source: bool = False,
    working_copy_mode: bool = False,
) -> html.Div:
    """Return the editor panel content div.

    All section divs (including merge / concat) are always present in the DOM
    so callbacks referencing their IDs remain valid regardless of *multi_source*.
    When ``multi_source=False`` merge/concat options are absent from the
    operation dropdown so those sections are never shown.

    When ``working_copy_mode=True`` the Apply button writes to a server-side
    working copy instead of the pools; a ``{prefix}-wc-changed`` store is added
    to the layout to signal downstream callbacks.
    """
    op_list = _OPS_MULTI if multi_source else _OPS_SINGLE
    col_opts: list[dict] = []   # placeholder; populated by _on_source_change

    # ── Source picker ────────────────────────────────────────────────────
    source_picker = entrypicker.get_entry_picker(f"{prefix}-src")

    # ── Op selector + result label ───────────────────────────────────────
    header = html.Div([
        formgen.form_row(
            "operation",
            formgen.make_dropdown(
                f"{prefix}-op-dd", op_list, None, clearable=True,
            ),
            "Select the transformation to apply to the selected entry.",
            label_id=f"{prefix}-op-lbl",
            label_text="Operation",
        ),
        formgen.form_row(
            "result_label",
            dbc.Input(id=f"{prefix}-label", type="text", placeholder="auto"),
            "Human-readable name for the result table (blank = auto-generated).",
            truly_optional=True,
            label_id=f"{prefix}-label-lbl",
            label_text="Result label",
        ),
    ])

    # ── Derive ───────────────────────────────────────────────────────────
    derive_section = html.Div(
        id=f"{prefix}-derive-section",
        style={"display": "none"},
        children=[
            formgen.form_row(
                "new_column_name",
                dbc.Input(id=f"{prefix}-derive-name", type="text", placeholder="new_col"),
                "Name for the new (or replaced) column.",
                label_id=f"{prefix}-derive-name-lbl",
                label_text="Column name",
            ),
            formgen.form_row(
                "expression",
                dbc.Textarea(
                    id=f"{prefix}-derive-expr",
                    placeholder="e.g.  geom2 * 2   or   score > 0.5",
                    style={"height": "64px", "resize": "none", "fontSize": styles.FONT_SM},
                ),
                "pandas.eval expression. Column names are available directly. "
                "Arithmetic, comparison, and boolean operators are supported (W7).",
                label_id=f"{prefix}-derive-expr-lbl",
                label_text="Expression",
            ),
        ],
    )

    # ── Rename ───────────────────────────────────────────────────────────
    rename_section = html.Div(
        id=f"{prefix}-rename-section",
        style={"display": "none"},
        children=[
            formgen.form_row(
                "rename_pairs",
                dbc.Textarea(
                    id=f"{prefix}-rename-map",
                    placeholder="old_col=new_col\nother_col=renamed",
                    style={"height": "100px", "resize": "none", "fontSize": styles.FONT_SM},
                ),
                "One old_name=new_name pair per line. Lines without '=' are ignored.",
                label_id=f"{prefix}-rename-map-lbl",
                label_text="Rename pairs",
            ),
        ],
    )

    # ── Drop ─────────────────────────────────────────────────────────────
    drop_section = html.Div(
        id=f"{prefix}-drop-section",
        style={"display": "none"},
        children=[
            formgen.form_row(
                "drop_columns",
                formgen.make_dropdown(
                    f"{prefix}-drop-dd", col_opts, None, clearable=True, multi=True,
                ),
                "Columns to remove from the result. Columns not present are silently ignored.",
                label_id=f"{prefix}-drop-lbl",
                label_text="Drop columns",
            ),
        ],
    )

    # ── Reorder ──────────────────────────────────────────────────────────
    reorder_section = html.Div(
        id=f"{prefix}-reorder-section",
        style={"display": "none"},
        children=[
            formgen.form_row(
                "column_order",
                formgen.make_dropdown(
                    f"{prefix}-reorder-dd", col_opts, None, clearable=True, multi=True,
                ),
                "Select columns in the desired leading order. Unselected columns are appended after.",
                label_id=f"{prefix}-reorder-lbl",
                label_text="Column order",
            ),
        ],
    )

    # ── Cast ─────────────────────────────────────────────────────────────
    cast_section = html.Div(
        id=f"{prefix}-cast-section",
        style={"display": "none"},
        children=[
            formgen.form_row(
                "cast_column",
                formgen.make_dropdown(f"{prefix}-cast-col-dd", col_opts, None),
                "Column whose data type will be changed.",
                label_id=f"{prefix}-cast-col-lbl",
                label_text="Column",
            ),
            formgen.form_row(
                "cast_dtype",
                formgen.make_dropdown(
                    f"{prefix}-cast-dtype-dd",
                    [{"label": d, "value": d} for d in ops.CAST_DTYPES],
                    None,
                ),
                "Target dtype. Use str for text, bool to convert 0/1 integer flags.",
                label_id=f"{prefix}-cast-dtype-lbl",
                label_text="Dtype",
            ),
        ],
    )

    # ── Merge ────────────────────────────────────────────────────────────
    merge_section = html.Div(
        id=f"{prefix}-merge-section",
        style={"display": "none"},
        children=[
            formgen.form_row(
                "right_table",
                formgen.make_dropdown(f"{prefix}-merge-right-dd", [], None),
                "Second (right) table to merge with the selected entry.",
                label_id=f"{prefix}-merge-right-lbl",
                label_text="Right table",
            ),
            formgen.form_row(
                "key_columns",
                formgen.make_dropdown(
                    f"{prefix}-merge-keys-dd", [], None, clearable=True, multi=True,
                ),
                "Columns present in both tables to join on.",
                label_id=f"{prefix}-merge-keys-lbl",
                label_text="Key columns",
            ),
            formgen.form_row(
                "join_type",
                formgen.make_dropdown(
                    f"{prefix}-merge-how-dd",
                    [{"label": h.capitalize(), "value": h} for h in ops.MERGE_HOW],
                    "inner",
                ),
                "inner keeps only matched rows; left/right keep all rows from one side; outer keeps all.",
                label_id=f"{prefix}-merge-how-lbl",
                label_text="Join type",
            ),
            html.Div(
                id=f"{prefix}-merge-report",
                style={**styles.HINT, "marginTop": styles.FORM_ROW_GAP},
            ),
        ],
    )

    # ── Concat ───────────────────────────────────────────────────────────
    concat_section = html.Div(
        id=f"{prefix}-concat-section",
        style={"display": "none"},
        children=[
            formgen.form_row(
                "additional_tables",
                formgen.make_dropdown(
                    f"{prefix}-concat-dd", [], None, clearable=True, multi=True,
                ),
                "Tables to stack below the selected entry. Mismatched columns become NaN.",
                label_id=f"{prefix}-concat-lbl",
                label_text="Extra tables",
            ),
            formgen.form_row(
                "source_label_column",
                dbc.Input(
                    id=f"{prefix}-concat-label-col",
                    type="text",
                    placeholder="source",
                    value="source",
                ),
                "Name of the new column that records which table each row came from.",
                label_id=f"{prefix}-concat-label-col-lbl",
                label_text="Label column",
            ),
        ],
    )

    # ── Footer ───────────────────────────────────────────────────────────
    if working_copy_mode:
        # In working-copy mode the Apply button previews the op on the copy;
        # commit actions live in the separate section added by pdatapool.
        footer = html.Div([
            html.Div(
                dbc.Button(
                    "Apply to working copy",
                    id=f"{prefix}-apply-btn",
                    color=styles.BTN_SECONDARY,
                    size="sm",
                    title="Apply this operation to the working copy — not committed until you click 'Apply to original'.",
                ),
                style={"marginTop": styles.SECTION_GAP},
            ),
            html.Div(id=f"{prefix}-status", style={**styles.HINT, "marginTop": styles.FORM_ROW_GAP}),
            dcc.Store(id=f"{prefix}-wc-changed", data=None),
        ])
    else:
        footer = html.Div([
            dbc.Checklist(
                id=f"{prefix}-create-new",
                options=[{
                    "label": "Create as new table",
                    "value": "yes",
                }],
                value=[],
                style={"marginBottom": styles.FORM_ROW_GAP},
            ),
            dbc.Checklist(
                id=f"{prefix}-add-as-motl",
                options=[{
                    "label": "Add result to motl pool (requires all 20 motl columns)",
                    "value": "yes",
                }],
                value=[],
            ),
            html.Div(
                dbc.Button(
                    "Apply",
                    id=f"{prefix}-apply-btn",
                    color=styles.BTN_PRIMARY,
                    size="sm",
                ),
                style={"marginTop": styles.SECTION_GAP},
            ),
            html.Div(id=f"{prefix}-status", style={**styles.HINT, "marginTop": styles.FORM_ROW_GAP}),
        ])

    return html.Div([
        source_picker,
        html.Hr(style={"margin": f"{styles.SECTION_GAP} 0"}),
        header,
        html.Hr(style={"margin": f"{styles.SECTION_GAP} 0"}),
        derive_section,
        rename_section,
        drop_section,
        reorder_section,
        cast_section,
        merge_section,
        concat_section,
        html.Hr(style={"margin": f"{styles.SECTION_GAP} 0"}),
        footer,
    ])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fetch_df(ref: dict) -> "pd.DataFrame":
    """Fetch the source DataFrame from the motl pool or data pool.

    Raises ``ValueError`` when the entry cannot be fetched or is not a DataFrame.
    """
    import pandas as pd
    if not ref:
        raise ValueError("No source entry selected.")
    if "motl_id" in ref:
        from cryocat.app.pool import get_rows
        df = get_rows(ref["motl_id"])
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Motl pool entry {ref['motl_id']!r} is not a DataFrame.")
        return df
    if "data_id" in ref:
        df = dp_module.get_payload(ref["data_id"])
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Selected data pool entry is not a DataFrame.")
        return df
    raise ValueError(f"Unknown ref format: {ref!r}")


def _src_label(
    ref: dict | None,
    pool_registry: dict | None,
    dp_registry: dict | None,
) -> str:
    return entrypicker.ref_label(ref, pool_registry, dp_registry)


def _source_kind_reader(
    src_ref: dict,
    dp_registry: dict | None,
) -> tuple[str, str]:
    """Return (source_kind, source_reader) for W4 validation."""
    if "motl_id" in src_ref:
        return "motl", ""
    data_id = src_ref.get("data_id", "")
    reader = (dp_registry or {}).get(data_id, {}).get("reader", "")
    return "data", reader


def _execute_operation(
    op: str,
    src_df: "pd.DataFrame",
    src_label: str,
    *,
    record: bool,
    pool_registry: "dict | None",
    dp_registry: "dict | None",
    derive_name: "str | None" = None,
    derive_expr: "str | None" = None,
    rename_map_text: "str | None" = None,
    drop_cols: "list | None" = None,
    reorder_cols: "list | None" = None,
    cast_col: "str | None" = None,
    cast_dtype: "str | None" = None,
    merge_right_val: "str | None" = None,
    merge_keys: "list | None" = None,
    merge_how: "str | None" = None,
    concat_extra_vals: "list | None" = None,
    concat_label_col: "str | None" = None,
) -> "tuple[pd.DataFrame | None, str | None, str]":
    """Apply a table operation; return (result_df, extra_msg, error).

    When *record* is True each op is logged via run_operation (modal / commit
    path).  When False the underlying function is called directly — used by the
    working-copy path where individual ops are previews, not recorded (W2).
    """
    import pandas as pd

    def _run(fn, kwargs):
        return run_operation(fn, kwargs) if record else fn(**kwargs)

    try:
        if op == "derive":
            if not derive_name or not derive_expr:
                return None, None, "Provide both a column name and an expression."
            result_df = _run(
                ops.derive_column,
                {"name": derive_name, "expression": derive_expr, "df": src_df},
            )
        elif op == "rename":
            mapping = ops.parse_rename_pairs(rename_map_text or "")
            if not mapping:
                return None, None, "Enter at least one old_col=new_col pair."
            result_df = _run(ops.rename_columns, {"mapping": mapping, "df": src_df})
        elif op == "drop":
            if not drop_cols:
                return None, None, "Select at least one column to drop."
            result_df = _run(
                ops.drop_columns, {"columns": list(drop_cols), "df": src_df}
            )
        elif op == "reorder":
            if not reorder_cols:
                return None, None, "Select at least one column to lead the order."
            result_df = _run(
                ops.reorder_columns, {"order": list(reorder_cols), "df": src_df}
            )
        elif op == "cast":
            if not cast_col or not cast_dtype:
                return None, None, "Select a column and a target dtype."
            result_df = _run(
                ops.cast_column, {"column": cast_col, "dtype": cast_dtype, "df": src_df}
            )
        elif op == "merge":
            if not merge_right_val or not merge_keys or not merge_how:
                return None, None, "Select a right table, at least one key column, and join type."
            right_ref = entrypicker.decode_value(merge_right_val)
            try:
                right_df = _fetch_df(right_ref)
            except Exception as exc:
                return None, None, f"Cannot load right table: {exc}"
            left_n, right_n = len(src_df), len(right_df)
            result_df = _run(
                ops.merge_tables,
                {"on": list(merge_keys), "how": merge_how,
                 "left": src_df, "right": right_df},
            )
            extra_msg = ops.merge_post_report(left_n, right_n, len(result_df), merge_how)
            return result_df, extra_msg, ""
        elif op == "concat":
            extra_vals = list(concat_extra_vals or [])
            if not extra_vals:
                return None, None, "Select at least one extra table."
            extra_frames, extra_labels = [], []
            for ev in extra_vals:
                er = entrypicker.decode_value(ev)
                try:
                    ef = _fetch_df(er)
                except Exception as exc:
                    return None, None, f"Cannot load extra table: {exc}"
                extra_frames.append(ef)
                extra_labels.append(entrypicker.ref_label(er, pool_registry, dp_registry))
            lbl_col = (concat_label_col or "").strip() or "source"
            all_frames = [src_df] + extra_frames
            all_labels = [src_label] + extra_labels
            result_df = _run(
                ops.concat_tables,
                {"labels": all_labels, "label_column": lbl_col, "frames": all_frames},
            )
            extra_msg = ops.concat_nan_report(result_df, len(all_frames))
            return result_df, extra_msg, ""
        else:
            return None, None, f"Unknown operation: {op!r}."
    except Exception as exc:
        return None, None, f"Operation failed: {exc}"

    if result_df is None:
        return None, None, "Operation returned no result."
    return result_df, None, ""


def _apply_wc_op(
    op, label_val,
    derive_name, derive_expr,
    rename_map_text, drop_cols, reorder_cols,
    cast_col, cast_dtype,
    merge_right_val, merge_keys, merge_how,
    concat_extra_vals, concat_label_col,
    src_ref, pool_reg, dp_reg,
):
    """Working-copy apply logic — extracted for thin-callback compliance."""
    from cryocat.app.suite.pages import _wcopy as _wc
    _no = no_update
    source_id = _wc.source_id_for_ref(src_ref)
    if source_id is None:
        return _no, "Cannot determine source id."
    if not _wc.has_copy(source_id):
        try:
            source_df = _fetch_df(src_ref)
        except Exception as exc:
            return _no, f"Cannot load source: {exc}"
        source_kind, source_reader = _source_kind_reader(src_ref, dp_reg)
        _wc.init_copy(source_id, source_df, source_n_rows=len(source_df),
                      source_kind=source_kind, source_reader=source_reader)
    base_df = _wc.get_copy(source_id)
    src_label = _src_label(src_ref, pool_reg, dp_reg)
    result_df, extra_msg, err = _execute_operation(
        op, base_df, src_label, record=False,
        pool_registry=pool_reg, dp_registry=dp_reg,
        derive_name=derive_name, derive_expr=derive_expr,
        rename_map_text=rename_map_text, drop_cols=drop_cols,
        reorder_cols=reorder_cols, cast_col=cast_col, cast_dtype=cast_dtype,
        merge_right_val=merge_right_val, merge_keys=merge_keys,
        merge_how=merge_how, concat_extra_vals=concat_extra_vals,
        concat_label_col=concat_label_col,
    )
    if err:
        return _no, err
    if result_df is None:
        return _no, "Operation returned no result."
    ops_count = _wc.apply_op(source_id, result_df)
    dp_module.set_view_df_direct(result_df)
    status_parts = [
        f"Result: {len(result_df):,} rows × {len(result_df.columns)} cols.",
        f"[{ops_count} op{'s' if ops_count != 1 else ''} pending]",
    ]
    if extra_msg:
        status_parts.append(extra_msg)
    return {"source_id": source_id, "ops_count": ops_count}, "  ".join(status_parts)


# ── Callbacks ──────────────────────────────────────────────────────────────────

def register_table_editor_callbacks(
    app,
    prefix: str,
    *,
    multi_source: bool = False,
    working_copy_mode: bool = False,
) -> None:
    """Wire all table editor callbacks.

    When ``working_copy_mode=False`` (default — modal path): Apply writes
    directly to the motl / data pools and is recorded via run_operation.

    When ``working_copy_mode=True`` (Tables tab): Apply writes to a
    server-side working copy and is NOT recorded — each op is a preview.
    Commit actions (Apply to original / Save as new / Discard) are registered
    separately by the page that mounts the editor in this mode.

    Non-wc mode outputs:
    - ``ids.POOL_REGISTRY / POOL_META / POOL_NEXT_ID`` when the result is a
      motl (source was motl + schema satisfied, or data source + "Add as motl").
    - ``ids.DATA_POOL_REGISTRY / DATA_POOL_NEXT_ID`` for data results.
    - ``dp-selected-id`` to advance the data pool viewer to the new entry.
    """
    # Register entry picker callbacks (populates options from both pools)
    entrypicker.register_entry_picker_callbacks(app, f"{prefix}-src")

    # ── 1. Show / hide operation sections ────────────────────────────────
    section_style_outputs = [
        Output(f"{prefix}-{s}-section", "style", allow_duplicate=True)
        for s in _ALL_SECTIONS
    ]

    @app.callback(
        *section_style_outputs,
        Input(f"{prefix}-op-dd", "value"),
        prevent_initial_call=True,
    )
    def _on_op_select(op):
        return [
            {"display": "block"} if op == s else {"display": "none"}
            for s in _ALL_SECTIONS
        ]

    # ── 2. Update column dropdowns when source ref changes ────────────────
    @app.callback(
        Output(f"{prefix}-drop-dd",      "options", allow_duplicate=True),
        Output(f"{prefix}-reorder-dd",   "options", allow_duplicate=True),
        Output(f"{prefix}-cast-col-dd",  "options", allow_duplicate=True),
        Output(f"{prefix}-merge-keys-dd","options", allow_duplicate=True),
        Input(f"{prefix}-src-ref",         "data"),
        State(ids.POOL_REGISTRY,           "data"),
        State(ids.DATA_POOL_REGISTRY,      "data"),
        prevent_initial_call=True,
    )
    def _on_source_change(ref, pool_reg, dp_reg):
        cols: list[str] = []
        if ref:
            try:
                df = _fetch_df(ref)
                cols = list(df.columns)
            except Exception:
                pass
        opts = [{"label": c, "value": c} for c in cols]
        return opts, opts, opts, opts

    # ── 3. Populate merge / concat secondary pickers (multi_source) ───────
    if multi_source:
        @app.callback(
            Output(f"{prefix}-merge-right-dd", "options"),
            Output(f"{prefix}-concat-dd",      "options"),
            Input(ids.POOL_REGISTRY,      "data"),
            Input(ids.DATA_POOL_REGISTRY, "data"),
            State(f"{prefix}-src-ref",    "data"),
            prevent_initial_call=False,
        )
        def _on_registry_change(pool_reg, dp_reg, src_ref):
            # Build combined options from both pools, excluding the source
            src_key = None
            if src_ref:
                if "motl_id" in src_ref:
                    src_key = f"motl:{src_ref['motl_id']}"
                elif "data_id" in src_ref:
                    src_key = f"data:{src_ref['data_id']}"

            opts: list[dict] = []
            for mid, meta in (pool_reg or {}).items():
                v = f"motl:{mid}"
                if v != src_key:
                    opts.append({"label": meta.get("label", mid), "value": v})
            for did, meta in (dp_reg or {}).items():
                if meta.get("kind") not in ("dataframe", None):
                    continue
                v = f"data:{did}"
                if v != src_key:
                    opts.append({"label": meta.get("label", did), "value": v})
            return opts, opts

        @app.callback(
            Output(f"{prefix}-merge-keys-dd", "options"),
            Output(f"{prefix}-merge-report",  "children"),
            Input(f"{prefix}-merge-right-dd", "value"),
            State(f"{prefix}-src-ref",         "data"),
            State(ids.POOL_REGISTRY,           "data"),
            State(ids.DATA_POOL_REGISTRY,      "data"),
            prevent_initial_call=True,
        )
        def _on_right_change(right_val, src_ref, pool_reg, dp_reg):
            import pandas as pd
            if not right_val or not src_ref:
                return [], ""
            right_ref = entrypicker.decode_value(right_val)
            try:
                src_df   = _fetch_df(src_ref)
                right_df = _fetch_df(right_ref)
            except Exception as exc:
                return [], str(exc)
            common = sorted(set(src_df.columns) & set(right_df.columns))
            opts = [{"label": c, "value": c} for c in common]
            if not common:
                return [], "No shared columns between the two tables."
            report = ops.merge_pre_report(src_df, right_df, common)
            msg = (
                f"Left: {report['left_n']:,} rows · "
                f"right: {report['right_n']:,} rows · "
                f"{report['matching_keys']:,} matching key value(s)."
            )
            return opts, msg

    # ── 4a. Apply — working-copy path (Tables tab) ───────────────────────
    if working_copy_mode:
        @app.callback(
            # allow_duplicate: pdatapool also writes this store for commit actions
            Output(f"{prefix}-wc-changed", "data", allow_duplicate=True),
            Output(f"{prefix}-status", "children"),
            Input(f"{prefix}-apply-btn", "n_clicks"),
            State(f"{prefix}-op-dd",           "value"),
            State(f"{prefix}-label",           "value"),
            State(f"{prefix}-derive-name",     "value"),
            State(f"{prefix}-derive-expr",     "value"),
            State(f"{prefix}-rename-map",      "value"),
            State(f"{prefix}-drop-dd",         "value"),
            State(f"{prefix}-reorder-dd",      "value"),
            State(f"{prefix}-cast-col-dd",     "value"),
            State(f"{prefix}-cast-dtype-dd",   "value"),
            State(f"{prefix}-merge-right-dd",  "value"),
            State(f"{prefix}-merge-keys-dd",   "value"),
            State(f"{prefix}-merge-how-dd",    "value"),
            State(f"{prefix}-concat-dd",       "value"),
            State(f"{prefix}-concat-label-col","value"),
            State(f"{prefix}-src-ref",         "data"),
            State(ids.POOL_REGISTRY,           "data"),
            State(ids.DATA_POOL_REGISTRY,      "data"),
            prevent_initial_call=True,
        )
        def _on_apply_wc(
            _n, op, label_val,
            derive_name, derive_expr,
            rename_map_text, drop_cols, reorder_cols,
            cast_col, cast_dtype,
            merge_right_val, merge_keys, merge_how,
            concat_extra_vals, concat_label_col,
            src_ref, pool_reg, dp_reg,
        ):
            _no = no_update
            if not op:
                return _no, "Select an operation first."
            if not src_ref:
                return _no, "Select a source entry from the picker."
            return _apply_wc_op(
                op, label_val, derive_name, derive_expr,
                rename_map_text, drop_cols, reorder_cols,
                cast_col, cast_dtype, merge_right_val, merge_keys, merge_how,
                concat_extra_vals, concat_label_col, src_ref, pool_reg, dp_reg,
            )

        return  # ── working-copy mode: no pool-writing Apply registered ──

    # ── 4b. Apply — direct / modal path ──────────────────────────────────
    @app.callback(
        # allow_duplicate: many callbacks write POOL_REGISTRY (pmotl, motlsidebar, …)
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        # allow_duplicate: same writers as POOL_REGISTRY
        Output(ids.POOL_META, "data", allow_duplicate=True),
        # allow_duplicate: same writers as POOL_REGISTRY
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        # allow_duplicate: _mutate in pdatapool.py also writes DATA_POOL_REGISTRY
        Output(ids.DATA_POOL_REGISTRY, "data", allow_duplicate=True),
        # allow_duplicate: same as DATA_POOL_REGISTRY
        Output(ids.DATA_POOL_NEXT_ID, "data", allow_duplicate=True),
        # allow_duplicate: _mutate in pdatapool.py also writes dp-selected-id
        Output("dp-selected-id", "data", allow_duplicate=True),
        Output(f"{prefix}-status", "children"),
        Input(f"{prefix}-apply-btn", "n_clicks"),
        # ── op + common fields ──
        State(f"{prefix}-op-dd",    "value"),
        State(f"{prefix}-label",    "value"),
        # ── per-op fields ──
        State(f"{prefix}-derive-name",      "value"),
        State(f"{prefix}-derive-expr",      "value"),
        State(f"{prefix}-rename-map",       "value"),
        State(f"{prefix}-drop-dd",          "value"),
        State(f"{prefix}-reorder-dd",       "value"),
        State(f"{prefix}-cast-col-dd",      "value"),
        State(f"{prefix}-cast-dtype-dd",    "value"),
        State(f"{prefix}-merge-right-dd",   "value"),
        State(f"{prefix}-merge-keys-dd",    "value"),
        State(f"{prefix}-merge-how-dd",     "value"),
        State(f"{prefix}-concat-dd",        "value"),
        State(f"{prefix}-concat-label-col", "value"),
        # ── pool states ──
        State(f"{prefix}-create-new",     "value"),
        State(f"{prefix}-add-as-motl",    "value"),
        State(f"{prefix}-src-ref",        "data"),
        State(ids.POOL_REGISTRY,          "data"),
        State(ids.POOL_META,              "data"),
        State(ids.POOL_NEXT_ID,           "data"),
        State(ids.DATA_POOL_REGISTRY,     "data"),
        State(ids.DATA_POOL_NEXT_ID,      "data"),
        prevent_initial_call=True,
    )
    def _on_apply(
        _n,
        op, label_val,
        derive_name, derive_expr,
        rename_map_text,
        drop_cols,
        reorder_cols,
        cast_col, cast_dtype,
        merge_right_val, merge_keys, merge_how,
        concat_extra_vals, concat_label_col,
        create_new_val,
        add_as_motl_val,
        src_ref,
        pool_registry, pool_meta, pool_next_id,
        dp_registry, dp_next_id,
    ):
        from cryocat.app.pool import PoolState, insert_motl
        from cryocat.app.datapool import DataPoolState

        _no = no_update
        _fail = (_no, _no, _no, _no, _no, _no)

        if not op:
            return *_fail, "Select an operation first."
        if not src_ref:
            return *_fail, "Select a source entry from the picker."

        # ── Fetch source DataFrame ────────────────────────────────────────
        try:
            src_df = _fetch_df(src_ref)
        except Exception as exc:
            return *_fail, f"Cannot load source: {exc}"

        src_is_motl = "motl_id" in src_ref
        src_label   = _src_label(src_ref, pool_registry, dp_registry)

        # ── Apply operation (W6: through run_operation) ───────────────────
        result_df, extra_msg, err = _execute_operation(
            op, src_df, src_label,
            record=True,
            pool_registry=pool_registry, dp_registry=dp_registry,
            derive_name=derive_name, derive_expr=derive_expr,
            rename_map_text=rename_map_text, drop_cols=drop_cols,
            reorder_cols=reorder_cols, cast_col=cast_col, cast_dtype=cast_dtype,
            merge_right_val=merge_right_val, merge_keys=merge_keys,
            merge_how=merge_how, concat_extra_vals=concat_extra_vals,
            concat_label_col=concat_label_col,
        )
        if err:
            return *_fail, err
        if result_df is None:
            return *_fail, "Operation returned no result."

        # ── Determine result label ────────────────────────────────────────
        effective_label = (label_val or "").strip() or ops.suggested_label(src_label, op)

        # ── Build status ──────────────────────────────────────────────────
        status_parts = [f"Result: {len(result_df):,} rows × {len(result_df.columns)} cols."]
        if extra_msg:
            status_parts.append(extra_msg)

        # ── Schema check (W4) ─────────────────────────────────────────────
        schema_ok, missing_cols = ops.satisfies_motl_schema(result_df)
        replace_checked  = "yes" not in (create_new_val or [])
        add_motl_checked = "yes" in (add_as_motl_val or [])

        # ── Route result (W4) ─────────────────────────────────────────────
        def _insert_motl_pool(df, label):
            """Insert df into motl pool; return (new_pr, new_pm, new_pn, motl_id)."""
            p = PoolState.from_stores(pool_registry, pool_meta, pool_next_id)
            p, mid = insert_motl(
                p, df, label=label, motl_type="emmotl",
                extra=None, meta={}, source_path="", has_tab=True,
            )
            return (*p.to_stores(), mid)

        def _replace_in_motl_pool(motl_id, df):
            """Replace rows for an existing motl; return (new_pr, new_pm, new_pn)."""
            p = PoolState.from_stores(pool_registry, pool_meta, pool_next_id)
            p = replace_motl_rows(p, motl_id, df)
            return p.to_stores()

        def _insert_data_pool(df, label):
            """Insert df into data pool; return (new_dr, new_dn, data_id)."""
            ds = DataPoolState.from_stores(dp_registry, dp_next_id)
            ds, did = dp_module.insert_entry(
                ds, df, label=label, reader="table_op", source_path="",
            )
            return (*ds.to_stores(), did)

        # Case 0: replace source entry in motl pool (same id, revision bumped)
        if src_is_motl and replace_checked:
            if not schema_ok:
                return *_fail, (
                    "Replace requires the result to keep all motl columns. "
                    f"Missing: {', '.join(missing_cols[:5])}{'…' if len(missing_cols) > 5 else ''}."
                )
            new_pr, new_pm, new_pn = _replace_in_motl_pool(src_ref["motl_id"], result_df)
            status_parts.append(f"Replaced {src_ref['motl_id']} in place (revision bumped).")
            return new_pr, new_pm, new_pn, _no, _no, _no, "  ".join(status_parts)

        # Case 1: source is motl, create new entry, result satisfies schema → motl pool
        if src_is_motl and schema_ok:
            new_pr, new_pm, new_pn, mid = _insert_motl_pool(result_df, effective_label)
            status_parts.append(f"Saved as new motl {mid} in motl pool.")
            return new_pr, new_pm, new_pn, _no, _no, _no, "  ".join(status_parts)

        # Case 2: source is motl, result doesn't satisfy schema → data pool
        if src_is_motl and not schema_ok:
            new_dr, new_dn, did = _insert_data_pool(result_df, effective_label)
            status_parts.append(
                f"Missing motl columns ({', '.join(missing_cols[:5])}{'…' if len(missing_cols) > 5 else ''}); "
                f"saved as data entry {did}."
            )
            return _no, _no, _no, new_dr, new_dn, did, "  ".join(status_parts)

        # Case 3: source is data pool, "Add as motl" checked, schema fails
        if not src_is_motl and add_motl_checked and not schema_ok:
            new_dr, new_dn, did = _insert_data_pool(result_df, effective_label)
            status_parts.append(
                f"Cannot add to motl pool — missing: {', '.join(missing_cols[:5])}"
                f"{'…' if len(missing_cols) > 5 else ''}."
            )
            return _no, _no, _no, new_dr, new_dn, did, "  ".join(status_parts)

        # Case 4: source is data pool, "Add as motl" checked, schema ok → both
        if not src_is_motl and add_motl_checked and schema_ok:
            new_pr, new_pm, new_pn, mid = _insert_motl_pool(result_df, effective_label)
            new_dr, new_dn, did = _insert_data_pool(result_df, effective_label)
            status_parts.append(f"Added to motl pool as {mid}.")
            return new_pr, new_pm, new_pn, new_dr, new_dn, did, "  ".join(status_parts)

        # Case 5: source is data pool, no motl → data pool only
        new_dr, new_dn, did = _insert_data_pool(result_df, effective_label)
        if schema_ok:
            status_parts.append(
                "Result satisfies motl schema — enable 'Add to motl pool' to include it."
            )
        return _no, _no, _no, new_dr, new_dn, did, "  ".join(status_parts)
