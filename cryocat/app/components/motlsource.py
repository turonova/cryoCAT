"""Pool-aware motl picker — a reusable Suite component.

Two pickers live here:

* :func:`get_motl_source` — the standard single-or-multi pool dropdown (one
  call site per consumer tool).
* :func:`get_multi_motl_picker` — composed picker for multi-motl operations
  driven by the ``motls`` spec on ``@gui_exposed``. It always renders **both**
  a pair picker (Main + Second single dropdowns) and an ordered list picker
  (multi-select that preserves selection order). The consumer toggles
  visibility based on the selected op's ``motls["arity"]`` — see
  ``cryocat.app.suite.motlsidebar``.

A tool drops ``get_motl_source(prefix, ...)`` into its layout and calls
``register_motl_source_callbacks(app, prefix, ...)`` in its ``register_callbacks``.
The picker reads the suite-global motl pool (``pool-registry`` / ``pool-motls``,
declared in :mod:`cryocat.app.suite.app`) and exposes the user's choice via the
dropdown ``value`` at ``f"{prefix}-motl-select"``. Consuming callbacks read
``pool-motls`` by that id.

Parameters of note:
  * ``multi``      -- dropdown allows several pool entries (value is a list).
  * ``show_table`` -- render a read-only tableview bound to the (single)
                      selection so the user can inspect entries before picking.
"""

from dash import html, dcc, Input, Output, no_update
import dash_bootstrap_components as dbc

from cryocat.app.components.tableview import get_table_component, register_table_callbacks
from cryocat.app.components.tableplot import register_table_plot_callbacks


def get_motl_source(prefix, show_table=False, multi=False):
    """Layout for a pool-aware motl picker.

    Parameters
    ----------
    prefix : str
        Unique id prefix for this picker instance.
    show_table : bool, default=False
        If True, render a read-only tableview (prefix ``f"{prefix}-src-tabv"``)
        bound to the single selection for inspection.
    multi : bool, default=False
        If True, the dropdown allows selecting several pool entries.

    Notes
    -----
    Key ids:
      * ``f"{prefix}-motl-select"`` -- dropdown; ``value`` is the selected
        ``motl_id`` (or list of ids when ``multi=True``).
      * ``f"{prefix}-src-tabv-global-data-store"`` -- table backing store
        (only when ``show_table=True``).
    """
    children = [
        html.Label(
            "Motl source",
            style={"fontSize": "0.85rem", "marginBottom": "2px", "color": "var(--color11)"},
        ),
        dcc.Dropdown(
            id=f"{prefix}-motl-select",
            multi=multi,
            placeholder="Select motl(s) from the pool" if multi else "Select a motl from the pool",
            style={"marginBottom": "0.5rem"},
        ),
        html.Div(
            id=f"{prefix}-motl-source-status",
            style={"fontSize": "0.8rem", "color": "var(--color9)", "marginBottom": "0.5rem"},
        ),
    ]

    if show_table:
        children += [
            dcc.Store(id=f"{prefix}-src-tabv-global-data-store"),
            get_table_component(f"{prefix}-src-tabv"),
        ]

    return html.Div(children, id=f"{prefix}-motl-source")


def register_motl_source_callbacks(app, prefix, multi=False, show_table=False):
    """Wire a motl picker to the suite pool.

    - Populates ``f"{prefix}-motl-select"`` options from ``pool-registry``
      (label -> motl_id), defaulting the value to the active entry (all active
      entries when ``multi=True``).
    - When ``show_table``: on selection, pushes the chosen ``pool-motls[id]``
      into ``f"{prefix}-src-tabv-global-data-store"`` and registers the
      inspect-table callbacks.
    """

    @app.callback(
        Output(f"{prefix}-motl-select", "options"),
        Output(f"{prefix}-motl-select", "value"),
        Output(f"{prefix}-motl-source-status", "children"),
        Input("pool-registry", "data"),
    )
    def _populate(registry):
        registry = registry or {}
        options = []
        active_ids = []
        for mid, meta in registry.items():
            if not meta.get("active", True):
                continue
            options.append({"label": meta.get("label", mid), "value": mid})
            active_ids.append(mid)

        if not options:
            return [], ([] if multi else None), "Pool is empty — load a motl in the editor."

        status = f"{len(options)} motl(s) in the pool."
        if multi:
            return options, active_ids, status
        return options, active_ids[0], status

    if show_table:

        @app.callback(
            Output(f"{prefix}-src-tabv-global-data-store", "data"),
            Input(f"{prefix}-motl-select", "value"),
            Input("pool-motls", "data"),
            prevent_initial_call=True,
        )
        def _to_table(selected, pool_motls):
            pool_motls = pool_motls or {}
            if not selected:
                return no_update
            mid = selected[0] if isinstance(selected, list) else selected
            if not mid:
                return no_update
            return pool_motls.get(mid)

        register_table_callbacks(app, f"{prefix}-src-tabv", csv_only=True)
        register_table_plot_callbacks(
            app,
            f"{prefix}-src-tabv-table-plot",
            f"{prefix}-src-tabv-global-data-store",
            table_grid_id=f"{prefix}-src-tabv-grid",
        )


# ── Multi-motl picker (pair + ordered list) ─────────────────────────────────────

def get_multi_motl_picker(prefix):
    """Layout for the multi-motl picker.

    Renders *both* a pair picker and a list picker. The consumer toggles each
    container's ``display`` via callback based on the selected op's
    ``motls["arity"]``. Labels are also consumer-driven (e.g. swap "Motls
    (order preserved)" for "Motls (first = kept on duplicates)").

    Key ids:
      * ``f"{prefix}-pair-picker"``    — wrapper for the pair controls.
      * ``f"{prefix}-main-select"``    — pair: Main motl (motl1).
      * ``f"{prefix}-second-select"``  — pair: Second motl (motl2).
      * ``f"{prefix}-list-picker"``    — wrapper for the list control.
      * ``f"{prefix}-list-label"``     — list-picker label (consumer rewrites
        the text based on ``main_first``).
      * ``f"{prefix}-list-select"``    — ordered multi-select; ``value`` is a
        list of ``motl_id`` in the order the user picked them.
    """
    label_style = {"fontSize": "0.85rem", "marginBottom": "2px", "color": "var(--color11)"}
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Main motl", style=label_style),
                    dcc.Dropdown(
                        id=f"{prefix}-main-select",
                        placeholder="Main motl (motl1)",
                        style={"marginBottom": "0.4rem"},
                    ),
                    html.Label("Second motl", style=label_style),
                    dcc.Dropdown(
                        id=f"{prefix}-second-select",
                        placeholder="Second motl (motl2)",
                        style={"marginBottom": "0.5rem"},
                    ),
                ],
                id=f"{prefix}-pair-picker",
                style={"display": "none"},
            ),
            html.Div(
                [
                    html.Label(
                        "Motls (order preserved)",
                        id=f"{prefix}-list-label",
                        style=label_style,
                    ),
                    dcc.Dropdown(
                        id=f"{prefix}-list-select",
                        multi=True,
                        placeholder="Pick pool motls — selection order is preserved",
                        style={"marginBottom": "0.5rem"},
                    ),
                ],
                id=f"{prefix}-list-picker",
                style={"display": "none"},
            ),
        ],
        id=f"{prefix}-multi-motl-picker",
    )


def register_multi_motl_picker_callbacks(app, prefix):
    """Populate the multi-motl picker dropdowns from the pool registry.

    All three dropdowns (pair Main, pair Second, list multi-select) share the
    same option set — every active pool entry. Default values are left as
    ``None`` so visibility-toggling and the consumer's op-change callback can
    set them.
    """

    @app.callback(
        Output(f"{prefix}-main-select", "options"),
        Output(f"{prefix}-second-select", "options"),
        Output(f"{prefix}-list-select", "options"),
        Input("pool-registry", "data"),
    )
    def _populate(registry):
        registry = registry or {}
        options = [
            {"label": meta.get("label", mid), "value": mid}
            for mid, meta in registry.items()
            if meta.get("active", True)
        ]
        return options, options, options
