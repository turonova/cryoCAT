"""Unified entry picker — single-select spanning the motl pool and data pool (W1).

One shared component used by both the table-editor workbench (pdatapool.py)
and the motl-slot editor modal (pmotl.py via tableview.py).

Non-DataFrame data pool entries (volumes, arrays, dicts) are **absent** from
the options — they cannot be transformed by the table editor, and showing
them as selectable-then-failing is ruled out by W3.

Value encoding
--------------
The dropdown value is a type-prefixed string:
  ``"motl:motl_1"``  — motl pool entry
  ``"data:data_1"``  — data pool DataFrame entry

The ``{prefix}-ref`` store holds the decoded ref dict ready for pool reads:
  ``{"motl_id": "motl_1", "label": "run1"}``
  ``{"data_id": "data_1", "label": "defocus"}``
  or ``None`` when nothing is selected.

NN / twist / tango tables (table pool, ``table_N`` prefix) do not appear here
because the table pool has no browser-side registry.  Unifying those requires
writing to DATA_POOL_REGISTRY at result time — a separate task.
"""
from __future__ import annotations

from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from cryocat.app import ids, styles, formgen


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_options(pool_registry: dict | None, dp_registry: dict | None) -> list[dict]:
    """Build grouped option list from both registries (W1, W3).

    Groups:
      1. Motls — all entries from pool_registry
      2. Tables — DataFrame-kind entries from dp_registry only

    Non-DataFrame data pool entries are excluded (W3 — absent, not disabled).
    """
    opts: list[dict] = []

    pool = pool_registry or {}
    if pool:
        opts.append({"label": "── Motls ──", "value": "_sep_motl", "disabled": True})
        for mid, meta in pool.items():
            label = meta.get("label", mid)
            n_rows = meta.get("n_rows", "?")
            opts.append({
                "label": f"{label}  [{n_rows:,} rows]" if isinstance(n_rows, int) else label,
                "value": f"motl:{mid}",
            })

    dp = dp_registry or {}
    df_entries = {k: v for k, v in dp.items() if v.get("kind") in ("dataframe", None)}
    if df_entries:
        opts.append({"label": "── Tables ──", "value": "_sep_tables", "disabled": True})
        for did, meta in df_entries.items():
            label = meta.get("label", did)
            n_rows = meta.get("n_rows", "?")
            opts.append({
                "label": f"{label}  [{n_rows:,} rows]" if isinstance(n_rows, int) else label,
                "value": f"data:{did}",
            })

    return opts


def decode_value(value: str | None) -> dict | None:
    """Convert a picker value string to a ref dict, or ``None`` if nothing selected."""
    if not value or value.startswith("_sep_"):
        return None
    if value.startswith("motl:"):
        mid = value[len("motl:"):]
        return {"motl_id": mid}
    if value.startswith("data:"):
        did = value[len("data:"):]
        return {"data_id": did}
    return None


def ref_label(ref: dict | None, pool_registry: dict | None, dp_registry: dict | None) -> str:
    """Return a human-readable label for the given ref dict."""
    if not ref:
        return ""
    if "motl_id" in ref:
        mid = ref["motl_id"]
        return (pool_registry or {}).get(mid, {}).get("label", mid)
    if "data_id" in ref:
        did = ref["data_id"]
        return (dp_registry or {}).get(did, {}).get("label", did)
    return ""


# ── Layout ─────────────────────────────────────────────────────────────────────

def get_entry_picker(prefix: str) -> html.Div:
    """Return the picker layout (dropdown + ref store).

    Stores:
        ``{prefix}-dd``   — dcc.Dropdown value (type-prefixed string or None)
        ``{prefix}-ref``  — dcc.Store; decoded ref dict or None
    """
    return html.Div([
        formgen.form_row(
            "source_entry",
            formgen.make_dropdown(
                f"{prefix}-dd",
                options=[],
                value=None,
                clearable=True,
                placeholder="Select a motl or table…",
            ),
            "Source motl or table to operate on. Motls appear first; DataFrames from the "
            "data pool appear below. Volumes, arrays, and dicts are excluded (W3).",
            label_id=f"{prefix}-dd-lbl",
            label_text="Source entry",
        ),
        dcc.Store(id=f"{prefix}-ref", data=None),
    ])


# ── Callbacks ──────────────────────────────────────────────────────────────────

def register_entry_picker_callbacks(app, prefix: str) -> None:
    """Populate picker from both pools and decode value to ref dict."""

    @app.callback(
        Output(f"{prefix}-dd", "options"),
        Input(ids.POOL_REGISTRY,      "data"),
        Input(ids.DATA_POOL_REGISTRY, "data"),
    )
    def _populate_opts(pool_reg, dp_reg):
        return _build_options(pool_reg, dp_reg)

    @app.callback(
        Output(f"{prefix}-ref", "data"),
        Input(f"{prefix}-dd", "value"),
        prevent_initial_call=True,
    )
    def _decode(value):
        return decode_value(value)
