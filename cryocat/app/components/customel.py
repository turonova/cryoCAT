import dash_bootstrap_components as dbc
from dash import html

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
