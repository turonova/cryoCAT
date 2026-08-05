"""Pool picker — multi-select from pool with visual confirmation list.

Drop :func:`get_pool_picker` into the page layout and call
:func:`register_pool_picker_callbacks` in ``register_callbacks``.

Contract
--------
Owned stores
  ``{prefix}-value``       list[str] of resolved motl_ids (output)
  ``{prefix}-pp-excluded`` list[str] of motl_ids removed from within a group
  ``{prefix}-pp-expand``   dict[gid → bool] of expanded groups
  ``{prefix}-pp-order``    list[str] of selected values in display order

Public result
  ``{prefix}-value``  ``data`` is ``list[str]`` of motl_ids.

Dropdown options
  Groups listed first, then orphan motls (not members of any group).

Pattern-matching ids (all scoped via ``"owner": prefix``):
  ``{"type": "pp-rm-item",   "owner": prefix, "val": v}``     remove top-level item
  ``{"type": "pp-rm-member", "owner": prefix, "mid": mid}``   exclude group member
  ``{"type": "pp-toggle",    "owner": prefix, "gid": gid}``   toggle group expand
  ``{"type": "pp-up",        "owner": prefix, "val": v}``     move item up
  ``{"type": "pp-down",      "owner": prefix, "val": v}``     move item down
"""
from __future__ import annotations

from dash import html, dcc, Input, Output, State, ctx, ALL
import dash_bootstrap_components as dbc

from cryocat.app import ids
from cryocat.app.formgen import make_dropdown


# ── Module-level helpers (loops live here, not in callbacks) ──────────────────

def _picker_opts(registry: dict, groups: dict) -> list[dict]:
    """Groups first, then orphan motls (not members of any group)."""
    grps = groups or {}
    reg = registry or {}
    grouped: set[str] = set()
    for g in grps.values():
        grouped.update(g.get("members", []))
    opts: list[dict] = []
    for gid, g in grps.items():
        if g.get("members"):
            opts.append({
                "label": f"[group]  {g.get('label', gid)}  ({len(g['members'])})",
                "value": f"__group__{gid}",
            })
    for mid, meta in reg.items():
        if mid not in grouped:
            opts.append({"label": meta.get("label", mid), "value": mid})
    return opts


def _clean_value(current: list | None, opts: list[dict]) -> list:
    valid = {o["value"] for o in opts}
    return [v for v in (current or []) if v in valid]


def _sync_order_list(dropdown_val: list | None, current_order: list | None) -> list:
    """Keep existing order, append new items, drop removed items."""
    dv_set = set(dropdown_val or [])
    kept = [v for v in (current_order or []) if v in dv_set]
    appended: set[str] = set(kept)
    for v in (dropdown_val or []):
        if v not in appended:
            kept.append(v)
            appended.add(v)
    return kept


def _reorder_list(order: list, direction: str, val: str) -> list:
    lst = list(order)
    if val not in lst:
        return lst
    idx = lst.index(val)
    if direction == "up" and idx > 0:
        lst[idx - 1], lst[idx] = lst[idx], lst[idx - 1]
    elif direction == "down" and idx < len(lst) - 1:
        lst[idx + 1], lst[idx] = lst[idx], lst[idx + 1]
    return lst


def _order_btns(val: str, prefix: str) -> list:
    common = {"size": "sm", "color": "link",
              "style": {"padding": "0 2px", "color": "inherit", "lineHeight": "1"}}
    return [
        dbc.Button("↑", id={"type": "pp-up",   "owner": prefix, "val": val}, **common),
        dbc.Button("↓", id={"type": "pp-down", "owner": prefix, "val": val}, **common),
    ]


def _motl_row(mid: str, registry: dict, prefix: str) -> html.Div:
    label = (registry or {}).get(mid, {}).get("label", mid)
    return html.Div(
        [
            *_order_btns(mid, prefix),
            html.Span(label, style={"flex": "1", "fontSize": "0.875rem"}),
            dbc.Button(
                "×",
                id={"type": "pp-rm-item", "owner": prefix, "val": mid},
                size="sm", color="link",
                style={"padding": "0 4px", "color": "inherit", "lineHeight": "1"},
            ),
        ],
        style={"display": "flex", "alignItems": "center",
               "marginBottom": "0.15rem", "paddingLeft": "0.25rem"},
    )


def _member_row(mid: str, registry: dict, prefix: str) -> html.Div:
    label = (registry or {}).get(mid, {}).get("label", mid)
    return html.Div(
        [
            html.Span(
                f"→ {label}",
                style={"flex": "1", "fontSize": "0.85rem", "color": "var(--color9)"},
            ),
            dbc.Button(
                "×",
                id={"type": "pp-rm-member", "owner": prefix, "mid": mid},
                size="sm", color="link",
                style={"padding": "0 4px", "color": "inherit", "lineHeight": "1"},
            ),
        ],
        style={"display": "flex", "alignItems": "center",
               "marginBottom": "0.1rem", "paddingLeft": "1.75rem"},
    )


def _group_row(gid: str, group: dict, active_count: int, expanded: bool, prefix: str) -> html.Div:
    label = group.get("label", gid)
    return html.Div(
        [
            dbc.Button(
                "▾" if expanded else "▶",
                id={"type": "pp-toggle", "owner": prefix, "gid": gid},
                size="sm", color="link",
                style={"padding": "0 4px", "color": "inherit", "lineHeight": "1"},
            ),
            *_order_btns(f"__group__{gid}", prefix),
            html.Span(
                f"{label}  ({active_count} motls)",
                style={"flex": "1", "fontSize": "0.875rem", "fontWeight": 500},
            ),
            dbc.Button(
                "×",
                id={"type": "pp-rm-item", "owner": prefix, "val": f"__group__{gid}"},
                size="sm", color="link",
                style={"padding": "0 4px", "color": "inherit", "lineHeight": "1"},
            ),
        ],
        style={"display": "flex", "alignItems": "center",
               "marginBottom": "0.15rem", "paddingLeft": "0.25rem"},
    )


def _build_list(
    order: list,
    excluded: list,
    registry: dict,
    groups: dict,
    expand: dict,
    prefix: str,
) -> list:
    excl = set(excluded or [])
    reg = registry or {}
    grps = groups or {}
    exp = expand or {}
    rows: list = []
    for v in (order or []):
        if isinstance(v, str) and v.startswith("__group__"):
            gid = v[len("__group__"):]
            g = grps.get(gid, {})
            members = [m for m in g.get("members", []) if m not in excl]
            rows.append(_group_row(gid, g, len(members), exp.get(gid, False), prefix))
            if exp.get(gid, False):
                for mid in members:
                    rows.append(_member_row(mid, reg, prefix))
        else:
            rows.append(_motl_row(v, reg, prefix))
    if not rows:
        return [html.Div(
            "No motls selected.",
            style={"color": "var(--color9)", "fontSize": "0.85rem", "padding": "0.3rem 0"},
        )]
    return rows


def _resolve_value(order: list, excluded: list, groups: dict) -> list:
    excl = set(excluded or [])
    seen: set = set()
    result: list = []
    for v in (order or []):
        if isinstance(v, str) and v.startswith("__group__"):
            gid = v[len("__group__"):]
            for m in (groups or {}).get(gid, {}).get("members", []):
                if m not in excl and m not in seen:
                    seen.add(m)
                    result.append(m)
        elif v not in excl and v not in seen:
            seen.add(v)
            result.append(v)
    return result


# ── Layout ────────────────────────────────────────────────────────────────────

def get_pool_picker(prefix: str, *, label: str = "Motl source") -> html.Div:
    """Pool-aware multi-select with a visual confirmation list.

    Parameters
    ----------
    prefix : str
        Unique id prefix for this picker instance.
    label : str
        Section label shown above the dropdown.

    Public result
    -------------
    ``{prefix}-value``  — ``dcc.Store``; ``data`` is ``list[str]`` of motl_ids.
    Removing an item from the confirmation list does **not** remove it from the pool.
    """
    return html.Div(
        [
            dcc.Store(id=f"{prefix}-value", data=[]),
            dcc.Store(id=f"{prefix}-pp-excluded", data=[]),
            dcc.Store(id=f"{prefix}-pp-expand", data={}),
            dcc.Store(id=f"{prefix}-pp-order", data=[]),
            html.Label(label, style={"marginBottom": "2px", "color": "var(--color11)"}),
            make_dropdown(
                f"{prefix}-pp-dropdown",
                [],
                [],
                multi=True,
                placeholder="Select motls or groups from the pool",
            ),
            html.Div(
                id=f"{prefix}-pp-list",
                style={
                    "marginTop": "0.5rem",
                    "border": "1px solid var(--color3)",
                    "borderRadius": "4px",
                    "padding": "0.35rem 0.4rem",
                    "minHeight": "2rem",
                    "maxHeight": "12rem",
                    "overflowY": "auto",
                },
            ),
        ],
        id=f"{prefix}-pool-picker",
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_pool_picker_callbacks(app, prefix: str) -> None:
    """Wire the pool-picker component to the suite pool."""
    import dash

    @app.callback(
        Output(f"{prefix}-pp-dropdown", "options"),
        Output(f"{prefix}-pp-dropdown", "value"),
        Input(ids.POOL_REGISTRY, "data"),
        Input(ids.POOL_GROUPS, "data"),
        State(f"{prefix}-pp-dropdown", "value"),
    )
    def _populate(registry, groups_data, current):
        from cryocat.app.pool import GroupState
        groups = GroupState.from_store(groups_data).groups
        opts = _picker_opts(registry, groups)
        return opts, _clean_value(current, opts)

    @app.callback(
        Output(f"{prefix}-pp-order", "data"),
        Input(f"{prefix}-pp-dropdown", "value"),
        State(f"{prefix}-pp-order", "data"),
    )
    def _sync_order(dropdown_val, current_order):
        return _sync_order_list(dropdown_val, current_order)

    @app.callback(
        Output(f"{prefix}-pp-list", "children"),
        Input(f"{prefix}-pp-order", "data"),
        Input(f"{prefix}-pp-excluded", "data"),
        Input(f"{prefix}-pp-expand", "data"),
        Input(ids.POOL_REGISTRY, "data"),
        Input(ids.POOL_GROUPS, "data"),
    )
    def _render(order, excluded, expand, registry, groups_data):
        from cryocat.app.pool import GroupState
        groups = GroupState.from_store(groups_data).groups
        return _build_list(order, excluded, registry, groups, expand, prefix)

    @app.callback(
        Output(f"{prefix}-pp-dropdown", "value", allow_duplicate=True),
        Input({"type": "pp-rm-item", "owner": prefix, "val": ALL}, "n_clicks"),
        State(f"{prefix}-pp-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _rm_item(n_clicks_list, curr):
        if not any(n for n in (n_clicks_list or []) if n):
            raise dash.exceptions.PreventUpdate
        val = ctx.triggered_id["val"]
        return [v for v in (curr or []) if v != val]

    @app.callback(
        Output(f"{prefix}-pp-excluded", "data"),
        Input({"type": "pp-rm-member", "owner": prefix, "mid": ALL}, "n_clicks"),
        State(f"{prefix}-pp-excluded", "data"),
        prevent_initial_call=True,
    )
    def _rm_member(n_clicks_list, excluded):
        if not any(n for n in (n_clicks_list or []) if n):
            raise dash.exceptions.PreventUpdate
        mid = ctx.triggered_id["mid"]
        return list({*(excluded or []), mid})

    @app.callback(
        Output(f"{prefix}-pp-expand", "data"),
        Input({"type": "pp-toggle", "owner": prefix, "gid": ALL}, "n_clicks"),
        State(f"{prefix}-pp-expand", "data"),
        prevent_initial_call=True,
    )
    def _toggle(n_clicks_list, expand):
        if not any(n for n in (n_clicks_list or []) if n):
            raise dash.exceptions.PreventUpdate
        gid = ctx.triggered_id["gid"]
        exp = dict(expand or {})
        exp[gid] = not exp.get(gid, False)
        return exp

    @app.callback(
        Output(f"{prefix}-pp-order", "data", allow_duplicate=True),
        Input({"type": "pp-up", "owner": prefix, "val": ALL}, "n_clicks"),
        State(f"{prefix}-pp-order", "data"),
        prevent_initial_call=True,
    )
    def _move_up(n_clicks_list, order):
        if not any(n for n in (n_clicks_list or []) if n):
            raise dash.exceptions.PreventUpdate
        return _reorder_list(order or [], "up", ctx.triggered_id["val"])

    @app.callback(
        Output(f"{prefix}-pp-order", "data", allow_duplicate=True),
        Input({"type": "pp-down", "owner": prefix, "val": ALL}, "n_clicks"),
        State(f"{prefix}-pp-order", "data"),
        prevent_initial_call=True,
    )
    def _move_down(n_clicks_list, order):
        if not any(n for n in (n_clicks_list or []) if n):
            raise dash.exceptions.PreventUpdate
        return _reorder_list(order or [], "down", ctx.triggered_id["val"])

    @app.callback(
        Output(f"{prefix}-value", "data"),
        Input(f"{prefix}-pp-order", "data"),
        Input(f"{prefix}-pp-excluded", "data"),
        Input(ids.POOL_GROUPS, "data"),
    )
    def _update_value(order, excluded, groups_data):
        from cryocat.app.pool import GroupState
        groups = GroupState.from_store(groups_data).groups
        return _resolve_value(order, excluded, groups)
