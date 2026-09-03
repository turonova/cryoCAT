import dash
import dash_bootstrap_components as dbc
from dash import html, Input, Output, State, ALL, ctx, no_update

from cryocat.app import styles
from cryocat.app.formgen import make_dropdown


def extract_style(default_style: dict, input_kwargs: dict) -> tuple[dict, dict]:
    """Merge default style with overrides from input_kwargs['style'].
    Returns (style, cleaned_kwargs)."""
    style = default_style.copy()
    cleaned = input_kwargs.copy()
    if "style" in cleaned and isinstance(cleaned["style"], dict):
        style.update(cleaned["style"])
        del cleaned["style"]
    return style, cleaned


def LabeledDropdown(id_, label, **dropdown_kwargs):
    return html.Div(
        [
            dbc.Label(label, html_for=id_, className="label-dark mb-1"),
            make_dropdown(
                id_,
                dropdown_kwargs.pop("options", []),
                dropdown_kwargs.pop("value", None),
                **dropdown_kwargs,
            ),
        ],
        className="mb-2",
    )


def InlineLabeledDropdown(id_, label, default_visibility="flex", tooltip_text="", **dropdown_kwargs):
    if not tooltip_text:
        tooltip_text = label

    class_style = dropdown_kwargs.pop("className", default_visibility)
    options = dropdown_kwargs.pop("options", [])
    value = dropdown_kwargs.pop("value", None)
    clearable = dropdown_kwargs.pop("clearable", False)
    # Caller-supplied style merges into FORM_INPUT
    extra_style, dropdown_kwargs = extract_style({}, dropdown_kwargs)
    input_style = {**styles.FORM_INPUT, **extra_style}

    return html.Div(
        [
            dbc.Label(
                label,
                id=f"{id_}-lbl",
                html_for=id_,
                className="label-dark mb-0 me-2",
                style={"whiteSpace": "nowrap"},
            ),
            html.Div(
                make_dropdown(id_, options, value, clearable=clearable, **dropdown_kwargs),
                style=input_style,
            ),
            dbc.Tooltip(tooltip_text, target=f"{id_}-lbl"),
        ],
        style=styles.FORM_ROW,
        className=class_style,
        id=f"{id_}-topdiv",
    )


def InlineInputForm(id_, label, default_visibility="flex", **input_kwargs):
    class_style = input_kwargs.pop("className", default_visibility)
    extra_style, input_kwargs = extract_style({}, input_kwargs)
    input_style = {**styles.FORM_INPUT, **extra_style}

    return html.Div(
        [
            dbc.Label(
                label,
                html_for=id_,
                className="label-dark mb-0 me-2",
                style={"whiteSpace": "nowrap"},
            ),
            dbc.Input(id=id_, style=input_style, **input_kwargs),
        ],
        style=styles.FORM_ROW,
        className=class_style,
        id=f"{id_}-topdiv",
    )


# ── Graph wrapper (BC4) ────────────────────────────────────────────────────────

def customel_graph(owner: str, name, graph_component) -> html.Div:
    """Wrap a styled-graph in a container with a Send-to-editor button.

    Parameters
    ----------
    owner, name :
        Must match the ``owner``/``name`` keys of the wrapped graph's id dict.
    graph_component :
        The ``dcc.Graph`` to wrap.
    """
    return html.Div([
        html.Div(
            dbc.Button(
                "Send to editor",
                id={"type": "customel-send-btn", "owner": owner, "name": name},
                size="sm",
                color="light",
                n_clicks=0,
                style={"marginBottom": "0.3rem"},
            ),
        ),
        graph_component,
    ])


def register_customel_callbacks(app: dash.Dash) -> None:
    """Register the pattern-matching Send-to-editor callback."""
    from cryocat.app import ids as _ids

    @app.callback(
        Output(_ids.GRAPH_POOL_REGISTRY, "data", allow_duplicate=True),
        Output(_ids.GRAPH_POOL_NEXT_ID, "data", allow_duplicate=True),
        Output("gr-pool-status", "children", allow_duplicate=True),
        Input({"type": "customel-send-btn", "owner": ALL, "name": ALL}, "n_clicks"),
        State({"type": "styled-graph", "owner": ALL, "name": ALL}, "figure"),
        State(_ids.GRAPH_POOL_REGISTRY, "data"),
        State(_ids.GRAPH_POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def _send_to_editor(all_clicks, all_figures, registry, next_id):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or triggered.get("type") != "customel-send-btn":
            raise dash.exceptions.PreventUpdate
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            raise dash.exceptions.PreventUpdate
        owner, name = triggered["owner"], triggered["name"]
        figure = None
        for entry in ctx.states_list[0]:
            if entry["id"]["owner"] == owner and entry["id"]["name"] == name:
                figure = entry["value"]
                break
        if not figure:
            return no_update, no_update, "No figure to send."
        from cryocat.app import graphpool as _graphpool
        state = _graphpool.GraphPoolState.from_stores(registry, next_id)
        lbl = f"Graph {state.next_id}"
        state, graph_id = _graphpool.insert_graph_entry(state, figure, label=lbl, kind="frozen")
        return *state.to_stores(), f"Sent {graph_id} to editor."
