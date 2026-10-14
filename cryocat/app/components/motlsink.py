"""'Send to editor' output — hands a tool-produced motl into the suite pool.

A tool drops ``get_send_to_editor_button(prefix)`` into its layout and calls
``register_send_to_editor_callbacks(app, prefix, result_store_id)`` in its
``register_callbacks``. On click, the motl currently in ``result_store_id`` is
appended to the suite-global pool (see :mod:`cryocat.app.ids` and
:mod:`cryocat.app.pool`) as a new active entry. The editor surfaces it as a
new tab via the same pool-driven tab creation as a fresh load.

Used by tools that emit motls (STA, NN, future structure). Pana does *not* use
this — it produces CSVs, not motls.
"""

from dash import html, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from cryocat.app import ids
from cryocat.app.pool import PoolState, insert_motl


def get_send_to_editor_button(prefix):
    """A label input + 'Send to editor' button for handing a motl to the pool.

    Ids:
      * ``f"{prefix}-send-label"``     -- optional text label for the new entry
      * ``f"{prefix}-send-to-editor"`` -- the action button
      * ``f"{prefix}-send-status"``    -- status text
    """
    return html.Div(
        [
            dbc.Input(
                id=f"{prefix}-send-label",
                type="text",
                placeholder="Label for the new motl (optional)",
                size="sm",
                style={"marginBottom": "0.4rem"},
            ),
            dbc.Button(
                "Send to editor",
                id=f"{prefix}-send-to-editor",
                color="primary",
                size="sm",
                style={"width": "100%"},
            ),
            html.Div(
                id=f"{prefix}-send-status",
                style={"color": "var(--color9)", "marginTop": "0.4rem"},
            ),
        ],
        id=f"{prefix}-motl-sink",
    )


def register_send_to_editor_callbacks(app, prefix, result_store_id):
    """Wire the 'Send to editor' button to the suite pool.

    On click: read the tool's result motl from ``result_store_id``, allocate a
    new ``motl_id`` via :func:`~cryocat.app.pool.insert_motl`, and append it to
    all four pool data stores. The editor picks it up reactively as a new tab.

    Parameters
    ----------
    app : dash.Dash
        The Dash app.
    prefix : str
        The id prefix used in :func:`get_send_to_editor_button`.
    result_store_id : str
        Id of the tool's store holding the result motl rows (a list of dicts).
    """

    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output(f"{prefix}-send-status", "children"),
        Input(f"{prefix}-send-to-editor", "n_clicks"),
        State(result_store_id, "data"),
        State(f"{prefix}-send-label", "value"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def _send(n_clicks, result_data, label, registry, pool_meta, next_id):
        if not n_clicks:
            return no_update, no_update, no_update, no_update
        if not result_data:
            return no_update, no_update, no_update, "No result motl to send."

        state = PoolState.from_stores(registry, pool_meta, next_id)
        # TODO(P9): route through run_operation_to_pool once load is tracked.
        state, motl_id = insert_motl(state, result_data, label=label)
        display_label = state.registry[motl_id]["label"]
        return (*state.to_stores(), f"Sent '{display_label}' to the editor.")
