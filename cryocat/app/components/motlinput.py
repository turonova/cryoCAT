"""Universal motl picker — pool / group.

Usage
-----
Drop :func:`get_motl_input` into the layout and call
:func:`register_motl_input_callbacks` in ``register_callbacks``.

The component publishes the user's selection via ``dcc.Store(id=f"{prefix}-value")``
as a ``list[str]`` of motl_ids.  Consuming callbacks read this store as State or Input.

Source modes
------------
``pool``   — multi-select over all active pool entries (flat list + group section headers).
``group``  — pick one group; members appear below a header row that shows the group
             name, member count, and ``[select all]`` / ``[select none]`` buttons.
             Deselecting members does NOT change the pool group.
"""

from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from cryocat.app import ids
from cryocat.app.formgen import make_dropdown


def get_motl_input(prefix: str, *, label: str = "Motl source") -> html.Div:
    """Layout for a pool-aware motl picker.

    Parameters
    ----------
    prefix : str
        Unique id prefix for this picker instance.
    label : str
        Section label shown above the source-mode toggle.

    Key ids exposed to consumers
    ----------------------------
    ``{prefix}-value``  —  ``dcc.Store``; ``data`` is ``list[str]`` of motl_ids.
    """
    return html.Div(
        [
            dcc.Store(id=f"{prefix}-value", data=[]),

            html.Label(label, style={"marginBottom": "2px", "color": "var(--color11)"}),

            dbc.RadioItems(
                id=f"{prefix}-source-mode",
                options=[
                    {"label": "Individual motls", "value": "pool"},
                    {"label": "Grouped motls", "value": "group"},
                ],
                value="pool",
                inline=True,
                style={"display": "flex", "gap": "1.5rem", "marginBottom": "0.5rem"},
            ),

            # ── From pool ────────────────────────────────────────────────────
            html.Div(
                id=f"{prefix}-pool-section",
                children=[
                    make_dropdown(
                        f"{prefix}-pool-select",
                        [],
                        [],
                        multi=True,
                        placeholder="Select motl(s) from the pool",
                    ),
                    html.Div(
                        id=f"{prefix}-pool-status",
                        style={
                            "color": "var(--color9)",
                            "fontSize": "0.85rem",
                            "marginTop": "0.3rem",
                        },
                    ),
                ],
            ),

            # ── From group ───────────────────────────────────────────────────
            html.Div(
                id=f"{prefix}-group-section",
                style={"display": "none"},
                children=[
                    make_dropdown(
                        f"{prefix}-group-select",
                        [],
                        None,
                        placeholder="Select a group",
                        style={"marginBottom": "0.5rem"},
                    ),
                    # Group header row — shown when a group is selected
                    html.Div(
                        [
                            html.Span(
                                id=f"{prefix}-group-header-text",
                                style={"flex": "1", "fontWeight": 500},
                            ),
                            dbc.Button(
                                "select all",
                                id=f"{prefix}-sel-all",
                                size="sm",
                                color="link",
                                style={"padding": "0 4px", "color": "inherit"},
                            ),
                            dbc.Button(
                                "select none",
                                id=f"{prefix}-sel-none",
                                size="sm",
                                color="link",
                                style={"padding": "0 4px", "color": "inherit"},
                            ),
                        ],
                        id=f"{prefix}-group-header-row",
                        style={"display": "none", "alignItems": "center", "marginBottom": "0.3rem"},
                    ),
                    dcc.Checklist(
                        id=f"{prefix}-group-checklist",
                        options=[],
                        value=[],
                        labelStyle={
                            "display": "block",
                            "paddingLeft": "1.5rem",
                            "marginBottom": "0.2rem",
                        },
                    ),
                    html.Div(
                        "Deselecting members does not change the pool group — "
                        "it narrows the selection for this panel only.",
                        style={
                            "color": "var(--color9)",
                            "fontSize": "0.82rem",
                            "marginTop": "0.5rem",
                        },
                    ),
                ],
            ),
        ],
        id=f"{prefix}-motlinput",
    )


def _group_member_opts(members: list, registry: dict) -> list[dict]:
    """Build checklist options for group members."""
    reg = registry or {}
    return [
        {"label": f"→ {reg.get(mid, {}).get('label', mid)}", "value": mid}
        for mid in members
    ]


def _pool_picker_options(registry: dict, current: list | None) -> tuple[list, list, str]:
    """Multi-select options: every pool motl as an individual entry."""
    reg = registry or {}
    current = current or []
    opts = [{"label": meta.get("label", mid), "value": mid} for mid, meta in reg.items()]
    valid_vals = {o["value"] for o in opts}
    kept = [v for v in current if v in valid_vals]
    status = "" if opts else "Pool is empty."
    return opts, kept, status


def register_motl_input_callbacks(app, prefix: str) -> None:
    """Wire the motl-input component to the suite pool."""

    @app.callback(
        Output(f"{prefix}-pool-section", "style"),
        Output(f"{prefix}-group-section", "style"),
        Input(f"{prefix}-source-mode", "value"),
    )
    def _toggle_sections(mode):
        if mode == "pool":
            return {"display": "block"}, {"display": "none"}
        return {"display": "none"}, {"display": "block"}

    @app.callback(
        Output(f"{prefix}-pool-select", "options"),
        Output(f"{prefix}-pool-select", "value"),
        Output(f"{prefix}-pool-status", "children"),
        Output(f"{prefix}-group-select", "options"),
        Input(ids.POOL_REGISTRY, "data"),
        Input(ids.POOL_GROUPS, "data"),
        State(f"{prefix}-pool-select", "value"),
    )
    def _populate(registry, groups_data, current_pool):
        from cryocat.app.pool import GroupState
        groups = GroupState.from_store(groups_data).groups
        pool_opts, pool_val, status = _pool_picker_options(registry, current_pool)
        group_opts = [
            {
                "label": f"{g.get('label', gid)} ({len(g.get('members', []))} motls)",
                "value": gid,
            }
            for gid, g in groups.items()
            if g.get("members")
        ]
        return pool_opts, pool_val, status, group_opts

    @app.callback(
        Output(f"{prefix}-group-header-row", "style"),
        Output(f"{prefix}-group-header-text", "children"),
        Output(f"{prefix}-group-checklist", "options"),
        Output(f"{prefix}-group-checklist", "value"),
        Input(f"{prefix}-group-select", "value"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_GROUPS, "data"),
    )
    def _render_group(gid, registry, groups_data):
        if not gid:
            return {"display": "none"}, "", [], []
        from cryocat.app.pool import GroupState
        groups = GroupState.from_store(groups_data).groups
        g = groups.get(gid, {})
        members = g.get("members", [])
        opts = _group_member_opts(members, registry)
        return (
            {"display": "flex", "alignItems": "center", "marginBottom": "0.3rem"},
            f"{g.get('label', gid)} · {len(members)} motls",
            opts,
            [o["value"] for o in opts],
        )

    @app.callback(
        Output(f"{prefix}-group-checklist", "value", allow_duplicate=True),
        Input(f"{prefix}-sel-all", "n_clicks"),
        State(f"{prefix}-group-checklist", "options"),
        prevent_initial_call=True,
    )
    def _select_all(_n, opts):
        return [o["value"] for o in (opts or [])]

    @app.callback(
        Output(f"{prefix}-group-checklist", "value", allow_duplicate=True),
        Input(f"{prefix}-sel-none", "n_clicks"),
        prevent_initial_call=True,
    )
    def _select_none(_n):
        return []

    @app.callback(
        Output(f"{prefix}-value", "data"),
        Input(f"{prefix}-source-mode", "value"),
        Input(f"{prefix}-pool-select", "value"),
        Input(f"{prefix}-group-checklist", "value"),
    )
    def _update_value(mode, pool_val, group_val):
        if mode == "pool":
            if pool_val is None:
                return []
            return pool_val if isinstance(pool_val, list) else [pool_val]
        return group_val or []
