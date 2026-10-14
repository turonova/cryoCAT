"""Shared "table → motl" sidebar component.

``get_table_to_motl(prefix)`` builds the layout.
``register_table_to_motl_callbacks(app, prefix, *, source_table_id, id_column)``
wires the callbacks.

The component writes table columns into a target motl from the pool, matching
by a chosen key column or by row position.  It can also create a filtered copy
of the target motl containing only the matched particles.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from dash import html, ctx, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from cryocat.core.cryomotl import Motl
from cryocat.app import ids as _ids
from cryocat.app import styles
from cryocat.app import formgen
from cryocat.app.formgen import make_dropdown
from cryocat.app.pool import PoolState, insert_motl as _insert_motl, get_rows as _get_rows, PoolPayloadMissing


_NU3 = (no_update, no_update, no_update)

_N_PAIRS = 4

# Placeholder value for the "row position" option in the match-source dropdown.
_ROW_POSITION = "__row_position__"

# Columns that are logically integer identifiers.  Float widening is skipped for
# these so an inadvertent float assignment surfaces as a clear pandas TypeError
# rather than silently truncating.
_FIXED_INT_COLS = frozenset({"subtomo_id", "tomo_id", "object_id", "class"})


def _widen_if_needed(motl_df: pd.DataFrame, dst_col: str) -> pd.DataFrame:
    """Widen *dst_col* in *motl_df* from int64 to float64 when it is integer-typed.

    Only applied to free-dtype columns (not in ``_FIXED_INT_COLS``).
    """
    if (
        dst_col not in _FIXED_INT_COLS
        and dst_col in motl_df.columns
        and pd.api.types.is_integer_dtype(motl_df[dst_col])
    ):
        motl_df[dst_col] = motl_df[dst_col].astype(float)
    return motl_df


def _parse_fill_value(raw: str | None) -> tuple[float | None, str | None]:
    """Parse and validate the user-supplied fill value.

    Returns ``(value, None)`` on success or ``(None, error_message)`` on failure.
    Accepts any floating-point number or the string ``'NaN'`` (case-insensitive).
    An empty or missing value is refused.
    """
    if raw is None or str(raw).strip() == "":
        return None, "Enter a fill value (a number or NaN) before writing."
    s = str(raw).strip()
    if s.lower() == "nan":
        return float("nan"), None
    try:
        return float(s), None
    except ValueError:
        return None, f"'{s}' is not a valid fill value — enter a number or NaN."


def _apply_pairs(
    motl_df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    match_src_col: str,
    match_motl_col: str,
    src_df: pd.DataFrame,
    *,
    fill_value: float = 0.0,
    keep_only_matched: bool = False,
) -> tuple[pd.DataFrame | None, str]:
    """Write *pairs* into *motl_df* using the chosen match strategy.

    Write mode (``keep_only_matched=False``):  all particles are kept;
    unmatched ones receive *fill_value*.

    Create mode (``keep_only_matched=True``):  only matched particles are kept;
    no fill is applied.  Source rows without a particle are dropped.

    Returns ``(updated_df, status_str)`` on success or ``(None, error_str)`` on
    failure.  *motl_df* must already be a copy.
    """
    if match_src_col == _ROW_POSITION:
        if len(src_df) != len(motl_df):
            return None, (
                f"Row count mismatch: table has {len(src_df)}, motl has {len(motl_df)}."
            )
        for val_col, dst_col in pairs:
            motl_df = _widen_if_needed(motl_df, dst_col)
            motl_df[dst_col] = src_df[val_col].values
        return motl_df, f"{len(motl_df)} rows written by position"

    # Key-based matching.
    if match_src_col not in src_df.columns:
        return None, f"Source has no column '{match_src_col}'."

    src_keys = src_df[match_src_col].dropna()
    motl_keys = motl_df[match_motl_col]
    has_row_mask = motl_keys.isin(src_keys)
    n_with_row = int(has_row_mask.sum())
    n_without_row = len(motl_df) - n_with_row
    n_rows_no_particle = int((~src_keys.isin(motl_keys)).sum())

    if keep_only_matched:
        if n_with_row == 0:
            return None, "No particles matched the source table."
        motl_df = motl_df[has_row_mask].copy()

    for val_col, dst_col in pairs:
        motl_df = _widen_if_needed(motl_df, dst_col)
        id_to_val = (
            src_df.drop_duplicates(subset=[match_src_col])
            .set_index(match_src_col)[val_col]
            .dropna()
            .to_dict()
        )
        if keep_only_matched:
            motl_df[dst_col] = motl_df[match_motl_col].map(id_to_val)
        else:
            motl_df[dst_col] = motl_df[match_motl_col].map(id_to_val).fillna(fill_value)

    if keep_only_matched:
        parts = [f"{n_with_row} particles kept"]
        if n_rows_no_particle > 0:
            parts.append(f"{n_rows_no_particle} row(s) without a particle dropped")
    else:
        fv_display = "NaN" if pd.isna(fill_value) else str(fill_value)
        parts = [f"{n_with_row} matched"]
        if n_without_row > 0:
            parts.append(f"{n_without_row} particle(s) without a row (filled {fv_display})")
        if n_rows_no_particle > 0:
            parts.append(f"{n_rows_no_particle} row(s) without a particle dropped")
    return motl_df, ", ".join(parts)


def _do_write_cols(
    target_id, pairs, match_src_col, match_motl_col, fill_raw,
    active_rows, registry, pool_meta, next_id,
):
    """Write multiple source→destination column pairs into the target motl atomically."""
    from cryocat.app.pool import replace_motl_rows
    # Row-position matching maps every row 1-to-1; no unmatched particles exist,
    # so a fill value is meaningless and must not be required.
    if match_src_col == _ROW_POSITION:
        fill_value = 0.0  # unused — _apply_pairs row-position path never reads it
    else:
        fill_value, fill_err = _parse_fill_value(fill_raw)
        if fill_err:
            return fill_err, *_NU3
    if not pairs:
        return "No column pairs specified.", *_NU3
    try:
        motl_df = pd.DataFrame(_get_rows(target_id)).copy()
    except PoolPayloadMissing:
        return "Target motl not found in pool.", *_NU3
    src_df = pd.DataFrame(active_rows)

    errors = []
    for val_col, dst_col in pairs:
        if val_col not in src_df.columns:
            errors.append(f"Source has no column '{val_col}'.")
        if dst_col not in Motl.motl_columns:
            errors.append(f"'{dst_col}' is not a motl column.")
    if errors:
        return "Validation failed — no changes written. " + " ".join(errors), *_NU3

    motl_df, status_extra = _apply_pairs(
        motl_df, pairs, match_src_col, match_motl_col, src_df,
        fill_value=fill_value,
        keep_only_matched=False,
    )
    if motl_df is None:
        return status_extra, *_NU3

    pair_desc = ", ".join(f"'{s}'→'{d}'" for s, d in pairs)
    pool_state = PoolState.from_stores(registry, pool_meta, next_id)
    pool_state = replace_motl_rows(pool_state, target_id, motl_df)
    return f"Wrote {len(pairs)} pair(s) ({pair_desc}) — {status_extra}.", *pool_state.to_stores()


def _do_create_motl(
    target_id, pairs, match_src_col, match_motl_col, label,
    active_rows, registry, pool_meta, next_id,
):
    """Copy the target motl keeping only matched particles; optionally write table columns."""
    try:
        motl_df = pd.DataFrame(_get_rows(target_id)).copy()
    except PoolPayloadMissing:
        return "Target motl not found in pool.", *_NU3
    src_df = pd.DataFrame(active_rows)

    if pairs:
        errors = []
        for val_col, dst_col in pairs:
            if val_col not in src_df.columns:
                errors.append(f"Source has no column '{val_col}'.")
            if dst_col not in Motl.motl_columns:
                errors.append(f"'{dst_col}' is not a motl column.")
        if errors:
            return "Validation failed — motl not created. " + " ".join(errors), *_NU3

    motl_df, status_extra = _apply_pairs(
        motl_df, pairs, match_src_col, match_motl_col, src_df,
        keep_only_matched=True,
    )
    if motl_df is None:
        return status_extra, *_NU3

    pool_state = PoolState.from_stores(registry, pool_meta, next_id)
    pool_state, new_id = _insert_motl(pool_state, motl_df, label=label)
    display_label = pool_state.registry[new_id]["label"]
    col_note = f", {len(pairs)} column(s) written" if pairs else ""
    return (
        f"Created '{display_label}' with {len(motl_df)} particles ({status_extra}{col_note}).",
        *pool_state.to_stores(),
    )


def get_table_to_motl(prefix: str, *, show_rows_mode: bool = True) -> html.Div:
    """Return sidebar content for table→motl operations."""
    _motl_opts = [{"label": c, "value": c} for c in Motl.motl_columns]
    _match_src_opts = [{"label": "row position", "value": _ROW_POSITION}]
    return html.Div(
        [
            formgen.form_row(
                f"{prefix}_target_motl",
                make_dropdown(
                    f"{prefix}-ttm-target-motl",
                    [],
                    None,
                    clearable=True,
                    placeholder="Choose motl from pool…",
                ),
                "Target motl from the editor pool",
                label_id=f"{prefix}-ttm-target-motl-lbl",
                label_text="Target motl",
            ),
            html.Div(
                [
                    html.Label(
                        "Column pairs (source → dest):",
                        style={"fontSize": styles.FONT_SM, "fontWeight": 600, "marginBottom": "0.2rem"},
                    ),
                    html.P(
                        "Select source and destination for each pair. Empty rows are skipped. "
                        "All or none are written.",
                        style={"fontSize": styles.FONT_SM, "color": styles.COLOR_MUTED, "marginBottom": "0.4rem"},
                    ),
                    *[
                        html.Div(
                            [
                                make_dropdown(
                                    f"{prefix}-ttm-val-col-{i}",
                                    [],
                                    None,
                                    clearable=True,
                                    placeholder=f"Source {i + 1}…",
                                    style={"flex": "1"},
                                ),
                                html.Span("→", style={"padding": "0 0.4rem", "lineHeight": "2"}),
                                make_dropdown(
                                    f"{prefix}-ttm-dst-col-{i}",
                                    _motl_opts,
                                    None,
                                    clearable=True,
                                    placeholder=f"Dest {i + 1}…",
                                    style={"flex": "1"},
                                ),
                            ],
                            style={"display": "flex", "gap": "0.25rem", "marginBottom": "0.25rem"},
                        )
                        for i in range(_N_PAIRS)
                    ],
                ],
                style={"marginBottom": styles.SECTION_GAP},
            ),
            html.Div(
                [
                    html.Hr(style={"margin": "0.4rem 0"}),
                    formgen.form_row(
                        f"{prefix}_match_src_col",
                        make_dropdown(
                            f"{prefix}-ttm-match-src-col",
                            _match_src_opts,
                            _ROW_POSITION,
                            clearable=False,
                        ),
                        "Source column whose values are looked up in the motl key column, "
                        "or 'row position' to match by row order",
                        label_id=f"{prefix}-ttm-match-src-col-lbl",
                        label_text="Match source on",
                    ),
                    html.Div(
                        formgen.form_row(
                            f"{prefix}_match_motl_col",
                            make_dropdown(
                                f"{prefix}-ttm-match-motl-col",
                                _motl_opts,
                                "subtomo_id",
                                clearable=False,
                            ),
                            "Motl column to match source key values against",
                            label_id=f"{prefix}-ttm-match-motl-col-lbl",
                            label_text="Match motl on",
                        ),
                        id=f"{prefix}-ttm-match-motl-row",
                        style={"display": "none"},
                    ),
                    html.Div(
                        formgen.form_row(
                            f"{prefix}_fill_value",
                            html.Div(
                                [
                                    dbc.Input(
                                        id=f"{prefix}-ttm-fill-value",
                                        placeholder="e.g. 0, −2, NaN",
                                        size="sm",
                                        type="text",
                                        style={"marginBottom": "0.15rem"},
                                    ),
                                    html.Small(
                                        "⚠ NaN will be saved as 0 by the EM writer.",
                                        id=f"{prefix}-ttm-fill-nan-warn",
                                        style={"color": styles.COLOR_POSITIVE, "display": "none"},
                                    ),
                                ]
                            ),
                            "Value written to motl particles with no matching source row "
                            "(key matching, Write to motl only). Not used by Create motl.",
                            label_id=f"{prefix}-ttm-fill-value-lbl",
                            label_text="Fill unmatched",
                        ),
                        id=f"{prefix}-ttm-fill-section",
                        style={"display": "none"},
                    ),
                ],
                style={"marginBottom": styles.SECTION_GAP},
            ),
            html.Div(
                [
                    html.Hr(style={"margin": "0.4rem 0"}),
                    formgen.form_row(
                        f"{prefix}_rows_mode",
                        dbc.RadioItems(
                            id=f"{prefix}-ttm-rows-mode",
                            options=[
                                {"label": "All rows", "value": "all"},
                                {"label": "Selected rows only", "value": "selected"},
                            ],
                            value="all",
                            inline=True,
                            className="sidebar-checklist",
                            labelStyle={"marginRight": "0.7rem"},
                        ),
                        "Which rows to include when writing or creating a new motl",
                        label_id=f"{prefix}-ttm-rows-mode-lbl",
                        label_text="Rows to include",
                    ),
                ],
                style={} if show_rows_mode else {"display": "none"},
            ),
            formgen.form_row(
                f"{prefix}_motl_label",
                dbc.Input(id=f"{prefix}-ttm-label", placeholder="Optional label", size="sm"),
                "Label for the new motl entry in the editor",
                label_id=f"{prefix}-ttm-label-lbl",
                label_text="New motl label",
                truly_optional=True,
            ),
            html.Div(
                [
                    dbc.Button(
                        "Write columns to motl",
                        id=f"{prefix}-ttm-write-btn",
                        color=styles.BTN_PRIMARY,
                        size="sm",
                        disabled=True,
                        style={"width": "100%", "marginBottom": "0.3rem"},
                    ),
                    dbc.Button(
                        "Create new motl",
                        id=f"{prefix}-ttm-create-btn",
                        color=styles.BTN_SECONDARY,
                        size="sm",
                        disabled=True,
                        style={"width": "100%"},
                    ),
                ]
            ),
            html.Div(
                id=f"{prefix}-ttm-status",
                style={"fontSize": styles.FONT_SM, "color": styles.COLOR_MUTED, "marginTop": "0.4rem"},
            ),
        ],
        style={"padding": "0.5rem 0"},
    )


def register_table_to_motl_callbacks(
    app,
    prefix: str,
    *,
    source_table_id: str = "",
    id_column: str = "qp_id",
    source_store_id: str | None = None,
    source_sel_store_id: str | None = None,
    resolve_df: Callable | None = None,
) -> None:
    """Register all callbacks for a table→motl component instance.

    Parameters
    ----------
    app:
        The Dash application.
    prefix:
        Must match the prefix passed to :func:`get_table_to_motl`.
    source_table_id:
        Id of the ``AgGrid`` whose ``selectedRows`` supply selected rows.
        ``rowData`` is watched for columns only when *source_store_id* is not
        provided.  Unused when both *source_store_id* and
        *source_sel_store_id* are set.
    id_column:
        Retained for backward compatibility with call sites; no longer used
        for matching (the user selects match columns at runtime).
    source_store_id:
        Optional ``dcc.Store`` holding the table's data reference (used with
        infinite/server-side row models where ``rowData`` is never populated).
    source_sel_store_id:
        Optional ``dcc.Store`` whose ``data`` holds the selected rows list.
        When provided, replaces ``State(source_table_id, "selectedRows")``.
    resolve_df:
        Callable ``(store_data) -> pd.DataFrame | None`` required when
        *source_store_id* is provided.
    """
    _has_store = source_store_id is not None and resolve_df is not None
    _has_sel_store = source_sel_store_id is not None

    # Pool registry → target-motl options.
    app.clientside_callback(
        """function(registry) {
            var reg = registry || {};
            return Object.keys(reg).map(function(k) {
                return {label: (reg[k].label || k), value: k};
            });
        }""",
        Output(f"{prefix}-ttm-target-motl", "options"),
        Input(_ids.POOL_REGISTRY, "data"),
    )

    # Enable/disable action buttons when a target is (un)selected.
    app.clientside_callback(
        """function(val) {
            var off = !val;
            return [off, off];
        }""",
        Output(f"{prefix}-ttm-write-btn", "disabled"),
        Output(f"{prefix}-ttm-create-btn", "disabled"),
        Input(f"{prefix}-ttm-target-motl", "value"),
    )

    # Show/hide motl key row and fill section when match-source switches between
    # row-position and key-column mode.
    app.clientside_callback(
        """function(val) {
            var isKey = val && val !== '__row_position__';
            var show = isKey ? {} : {"display": "none"};
            return [show, show];
        }""",
        Output(f"{prefix}-ttm-match-motl-row", "style"),
        Output(f"{prefix}-ttm-fill-section", "style"),
        Input(f"{prefix}-ttm-match-src-col", "value"),
        prevent_initial_call=True,
    )

    # Show/hide the NaN warning inside the fill section.
    app.clientside_callback(
        """function(val) {
            var s = (val || '').trim().toLowerCase();
            return s === 'nan' ? {} : {"display": "none"};
        }""",
        Output(f"{prefix}-ttm-fill-nan-warn", "style"),
        Input(f"{prefix}-ttm-fill-value", "value"),
        prevent_initial_call=True,
    )

    # Auto-select default for match-src-col when source options update:
    # pick the current motl key column name if it appears in source, else row position.
    app.clientside_callback(
        """function(options, motlCol) {
            var vals = (options || []).map(function(o) { return o.value; });
            if (motlCol && vals.indexOf(motlCol) >= 0) return motlCol;
            return '__row_position__';
        }""",
        Output(f"{prefix}-ttm-match-src-col", "value"),
        Input(f"{prefix}-ttm-match-src-col", "options"),
        State(f"{prefix}-ttm-match-motl-col", "value"),
        prevent_initial_call=True,
    )

    # Populate source column dropdowns (val-col-* and match-src-col) when source data changes.
    if _has_store:
        @app.callback(
            Output(f"{prefix}-ttm-match-src-col", "options"),
            *[Output(f"{prefix}-ttm-val-col-{i}", "options") for i in range(_N_PAIRS)],
            Input(source_store_id, "data"),
            prevent_initial_call=True,
        )
        def _populate_val_cols(ref):
            df = resolve_df(ref)
            cols = [{"label": c, "value": c} for c in df.columns] if df is not None and not df.empty else []
            match_opts = [{"label": "row position", "value": _ROW_POSITION}] + cols
            return match_opts, *(cols for _ in range(_N_PAIRS))
    elif source_table_id:
        @app.callback(
            Output(f"{prefix}-ttm-match-src-col", "options"),
            *[Output(f"{prefix}-ttm-val-col-{i}", "options") for i in range(_N_PAIRS)],
            Input(source_table_id, "rowData"),
            prevent_initial_call=True,
        )
        def _populate_val_cols(row_data):
            cols = (
                [{"label": c, "value": c} for c in pd.DataFrame(row_data or []).columns]
                if row_data else []
            )
            match_opts = [{"label": "row position", "value": _ROW_POSITION}] + cols
            return match_opts, *(cols for _ in range(_N_PAIRS))

    # Build conditional State lists — no empty-string IDs are ever added.
    _act_extra = (
        [State(source_store_id, "data")] if _has_store
        else ([State(source_table_id, "rowData")] if source_table_id else [])
    )
    _sel_states = (
        [State(source_sel_store_id, "data")] if _has_sel_store
        else ([State(source_table_id, "selectedRows")] if source_table_id else [])
    )
    _has_raw = bool(_act_extra)
    _has_sel = bool(_sel_states)

    @app.callback(
        Output(f"{prefix}-ttm-status", "children"),
        Output(_ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(_ids.POOL_META, "data", allow_duplicate=True),
        Output(_ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Input(f"{prefix}-ttm-write-btn", "n_clicks"),
        Input(f"{prefix}-ttm-create-btn", "n_clicks"),
        State(f"{prefix}-ttm-target-motl", "value"),
        *[State(f"{prefix}-ttm-val-col-{i}", "value") for i in range(_N_PAIRS)],
        *[State(f"{prefix}-ttm-dst-col-{i}", "value") for i in range(_N_PAIRS)],
        State(f"{prefix}-ttm-rows-mode", "value"),
        State(f"{prefix}-ttm-label", "value"),
        State(f"{prefix}-ttm-match-src-col", "value"),
        State(f"{prefix}-ttm-match-motl-col", "value"),
        State(f"{prefix}-ttm-fill-value", "value"),
        *_act_extra,
        *_sel_states,
        State(_ids.POOL_REGISTRY, "data"),
        State(_ids.POOL_META, "data"),
        State(_ids.POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def _act(_write_click, _create_click, target_id, *rest):
        n = _N_PAIRS
        val_cols = list(rest[:n])
        dst_cols = list(rest[n:2 * n])
        _i = 2 * n
        rows_mode = rest[_i]; _i += 1
        label = rest[_i]; _i += 1
        match_src_col = rest[_i] or _ROW_POSITION; _i += 1
        match_motl_col = rest[_i] or "subtomo_id"; _i += 1
        fill_raw = rest[_i]; _i += 1
        if _has_raw:
            raw_data = rest[_i]; _i += 1
        else:
            raw_data = []
        if _has_sel:
            selected_rows = rest[_i]; _i += 1
        else:
            selected_rows = []
        registry = rest[_i]; _i += 1
        pool_meta = rest[_i]; _i += 1
        next_id = rest[_i]

        if _has_store:
            df = resolve_df(raw_data)
            all_rows = df.to_dict("records") if df is not None else []
        else:
            all_rows = list(raw_data or [])

        active = (list(selected_rows or [])) if (rows_mode or "all") == "selected" else all_rows
        if not active:
            return ("No rows selected." if rows_mode == "selected" else "No rows in table."), *_NU3

        if ctx.triggered_id == f"{prefix}-ttm-write-btn":
            pairs = [(vc, dc) for vc, dc in zip(val_cols, dst_cols) if vc and dc]
            if not pairs:
                return "Choose at least one source and destination column pair.", *_NU3
            return _do_write_cols(
                target_id, pairs, match_src_col, match_motl_col, fill_raw,
                active, registry, pool_meta, next_id,
            )
        if ctx.triggered_id == f"{prefix}-ttm-create-btn":
            pairs = [(vc, dc) for vc, dc in zip(val_cols, dst_cols) if vc and dc]
            return _do_create_motl(
                target_id, pairs, match_src_col, match_motl_col,
                label, active, registry, pool_meta, next_id,
            )
        raise PreventUpdate
