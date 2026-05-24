"""'Send to editor' output — hands a tool-produced motl into the suite pool.

A tool drops ``get_send_to_editor_button(prefix)`` into its layout and calls
``register_send_to_editor_callbacks(app, prefix, result_store_id)`` in its
``register_callbacks``. On click, the motl currently in ``result_store_id`` is
appended to the suite-global pool (``pool-registry`` / ``pool-motls``, declared
in :mod:`cryocat.app.suite.app`) as a new active entry. The editor surfaces it
as a new tab via the same pool-driven tab creation as a fresh load.

Used by tools that emit motls (STA, NN, future structure). Pana does *not* use
this — it produces CSVs, not motls.
"""

from dash import html, Input, Output, State, no_update
import dash_bootstrap_components as dbc


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
                style={"fontSize": "0.8rem", "color": "var(--color9)", "marginTop": "0.4rem"},
            ),
        ],
        id=f"{prefix}-motl-sink",
    )


def register_send_to_editor_callbacks(app, prefix, result_store_id):
    """Wire the 'Send to editor' button to the suite pool.

    On click: read the tool's result motl from ``result_store_id``, allocate a
    new ``motl_id`` (using the ``pool-next-id`` counter), and append it to
    ``pool-registry`` / ``pool-motls`` as an active entry. The editor picks it
    up reactively as a new tab.

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
        Output("pool-registry", "data", allow_duplicate=True),
        Output("pool-motls", "data", allow_duplicate=True),
        Output("pool-next-id", "data", allow_duplicate=True),
        Output(f"{prefix}-send-status", "children"),
        Input(f"{prefix}-send-to-editor", "n_clicks"),
        State(result_store_id, "data"),
        State(f"{prefix}-send-label", "value"),
        State("pool-registry", "data"),
        State("pool-motls", "data"),
        State("pool-next-id", "data"),
        prevent_initial_call=True,
    )
    def _send(n_clicks, result_data, label, registry, pool_motls, next_id):
        if not n_clicks:
            return no_update, no_update, no_update, no_update
        if not result_data:
            return no_update, no_update, no_update, "No result motl to send."

        registry = dict(registry or {})
        pool_motls = dict(pool_motls or {})
        next_id = next_id or 0

        motl_id = f"motl-{next_id}"
        try:
            n_rows = len(result_data)
        except TypeError:
            n_rows = 0

        display_label = label or f"Motl {next_id + 1}"
        registry[motl_id] = {
            "label": display_label,
            "type": "emmotl",
            "n_rows": n_rows,
            "active": True,
        }
        pool_motls[motl_id] = result_data

        return registry, pool_motls, next_id + 1, f"Sent '{display_label}' to the editor."
