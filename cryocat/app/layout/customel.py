import dash_bootstrap_components as dbc
from dash import html, dcc


def LabeledDropdown(id_, label, **dropdown_kwargs):
    return html.Div(
        [
            dbc.Label(label, html_for=id_, className="label-dark mb-1"),  # consistent label spacing
            dcc.Dropdown(
                id=id_,
                style={
                    "width": "100%",
                    "padding": "0",  # reduce padding
                },
                **dropdown_kwargs,
            ),
        ],
        className="mb-2",  # consistent block spacing
    )


def InlineLabeledDropdown(id_, label, default_visibility="flex", tooltip_text="", **dropdown_kwargs):

    if tooltip_text == "":
        tooltip_text = label

    class_style = dropdown_kwargs.get("className", default_visibility)

    return html.Div(
        [
            dbc.Label(
                label,
                html_for=id_,
                className=f"label-dark mb-0 me-2",  # right margin so it doesn’t stick to dropdown
                style={"whiteSpace": "nowrap"},  # keep label on one line
            ),
            dcc.Dropdown(
                id=id_,
                style={
                    "flex": "1",  # take up remaining horizontal space
                    "padding": "0",  # reduce padding
                    # "marginBottom": "0.5rem",
                },
                **dropdown_kwargs,
            ),
            # dbc.Tooltip(
            #     tooltip_text,
            #    target=id_,
            # ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": "0.5rem"},  # horizontal alignment
        className=class_style,
        id=f"{id_}-topdiv",
    )


def InlineInputForm(id_, label, **input_kwargs):
    return html.Div(
        [
            dbc.Label(
                label,
                html_for=id_,
                className="label-dark mb-0 me-2",  # right margin so it doesn’t stick to dropdown
                style={"whiteSpace": "nowrap"},  # keep label on one line
            ),
            dbc.Input(
                id=id_,
                style={
                    "flex": "1",  # take up remaining horizontal space
                    # "marginBottom": "0.5rem",
                },
                **input_kwargs,
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": "0.5rem"},  # horizontal alignment
    )
