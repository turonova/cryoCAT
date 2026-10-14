"""Standard page layout helpers for the cryoCAT GUI.

§8 of GUI_CONVENTIONS.md — the sticky-sidebar + scrolling-main skeleton
is duplicated across 10 modules.  Import these helpers; do not re-type the
dbc.Col / dbc.Row boilerplate.

Public API
----------
page_shell(sidebar_children, main_children, *, sidebar_width, sidebar_id)
    Return the standard dbc.Row containing a sticky sidebar and a main column.
sidebar_accordion(items, *, active_item)
    Return a dbc.Accordion configured for sidebar use (always_open=True).
"""

from dash import html
import dash_bootstrap_components as dbc

_SIDEBAR_STYLE: dict = {
    "padding": "0.5rem",
    "overflowY": "auto",
    "height": "100vh",
    "display": "flex",
    "flexDirection": "column",
}

_SIDEBAR_COL_STYLE: dict = {
    "margin": "0",
    "padding": "0",
    "height": "100vh",
    "position": "sticky",
    "top": "0px",
}

_MAIN_COL_STYLE: dict = {"margin": "0", "padding": "0"}
_MAIN_INNER_STYLE: dict = {"padding": "0.5rem"}
_ROW_STYLE: dict = {"margin": "0", "padding": "0"}


def page_shell(
    sidebar_children,
    main_children,
    *,
    sidebar_width: int = 3,
    sidebar_id: str | None = None,
) -> dbc.Row:
    """Standard two-column page row: sticky sidebar + scrolling main area.

    Parameters
    ----------
    sidebar_children
        Components placed inside the sidebar Div (className="sidebar").
    main_children
        Components placed inside the main content Div (padding: 0.5rem).
    sidebar_width
        dbc grid width for the sidebar column (default 3; use 4 for wide sidebars).
    sidebar_id
        Optional id for the sidebar Col (e.g. ``"me-sidebar"``).
    """
    main_width = 12 - sidebar_width
    sidebar_inner = html.Div(
        sidebar_children,
        className="sidebar",
        style=_SIDEBAR_STYLE,
    )
    sidebar_col_kwargs: dict = {
        "width": sidebar_width,
        "style": _SIDEBAR_COL_STYLE,
    }
    if sidebar_id is not None:
        sidebar_col_kwargs["id"] = sidebar_id
    sidebar_col = dbc.Col(sidebar_inner, **sidebar_col_kwargs)

    main_col = dbc.Col(
        html.Div(main_children, style=_MAIN_INNER_STYLE),
        width=main_width,
        style=_MAIN_COL_STYLE,
    )
    return dbc.Row([sidebar_col, main_col], className="g-0", style=_ROW_STYLE)


def sidebar_accordion(items, *, active_item=None) -> dbc.Accordion:
    """Sidebar accordion with always_open=True.

    Parameters
    ----------
    items
        List of dbc.AccordionItem components.
    active_item
        Initially open item(s) — string or list of strings.
    """
    kwargs: dict = {"always_open": True}
    if active_item is not None:
        kwargs["active_item"] = active_item
    return dbc.Accordion(items, **kwargs)
