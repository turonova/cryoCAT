"""Palette loader component — preset dropdown + custom text input + swatch preview.

A self-contained palette selector that works for both discrete (categorical)
and continuous (colorscale) palettes.  Predefined options cover the most
commonly used Plotly palettes and the custom Monet/MonetWhite palettes from
:mod:`cryocat.analysis.visplot`.  The user may also type any valid Plotly
palette or colorscale name into the custom input; the component validates it
live and shows an error message if the name is unknown.

The resolved palette name is written to ``{prefix}-value`` (a ``dcc.Store``).
Callers should read from that store, not from the inner dropdown.

Public API
----------
get_palette_loader(prefix, mode, default)
    Layout: preset dropdown, custom text input, status line, colour swatch.
register_palette_loader_callbacks(app, prefix, mode)
    Register all callbacks for the component.
"""

from __future__ import annotations

import dash
from dash import html, dcc, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate


_HINT_STYLE = {"fontSize": "0.75rem", "color": "var(--color9)", "marginTop": "2px"}

_DISCRETE_PRESETS = [
    "Monet", "MonetWhite",
    "Plotly", "D3", "G10", "Vivid", "Bold", "Pastel", "Safe",
    "Alphabet", "Dark2", "Set1", "Set2", "Set3",
]

_CONTINUOUS_PRESETS = [
    "Viridis", "Plasma", "Inferno", "Magma", "Cividis",
    "Jet", "Hot", "Blues", "RdBu", "Spectral",
    "Turbo", "Rainbow", "Portland", "Picnic",
]


# ── Swatch helpers ─────────────────────────────────────────────────────────────

def _discrete_swatch(colors: list) -> html.Div:
    boxes = [
        html.Div(style={
            "background": c,
            "width": "18px",
            "height": "16px",
            "display": "inline-block",
            "marginRight": "2px",
            "borderRadius": "2px",
            "border": "1px solid rgba(0,0,0,0.15)",
            "flexShrink": "0",
        })
        for c in colors
    ]
    return html.Div(boxes, style={"display": "flex", "flexWrap": "wrap", "gap": "0"})


def _continuous_swatch(palette_val: str) -> html.Div:
    import plotly.express as px
    n = 24
    sampled = px.colors.sample_colorscale(palette_val, [i / (n - 1) for i in range(n)])
    gradient = f"linear-gradient(to right, {', '.join(sampled)})"
    return html.Div(style={
        "background": gradient,
        "width": "100%",
        "height": "16px",
        "borderRadius": "3px",
        "border": "1px solid rgba(0,0,0,0.15)",
    })


def _make_swatch(palette_val: str, mode: str):
    """Return swatch children for *palette_val*, or [] on failure."""
    if not palette_val:
        return []
    try:
        if mode == "discrete":
            from cryocat.analysis.visplot import resolve_palette
            colors = resolve_palette(palette_val)
            return _discrete_swatch(colors)
        else:
            return _continuous_swatch(palette_val)
    except Exception:
        return []


def _validate(palette_val: str, mode: str) -> None:
    """Raise ValueError/KeyError if *palette_val* is not a known palette."""
    if not palette_val:
        raise ValueError("Empty palette name.")
    if mode == "discrete":
        from cryocat.analysis.visplot import resolve_palette
        resolve_palette(palette_val)
    else:
        # For continuous, try sample_colorscale which accepts any valid name
        import plotly.express as px
        px.colors.sample_colorscale(palette_val, [0.0, 0.5, 1.0])


# ── Layout ─────────────────────────────────────────────────────────────────────

def get_palette_loader(prefix: str, mode: str = "discrete", default: str | None = None) -> html.Div:
    """Palette selector: preset dropdown + free-text custom input + swatch.

    Parameters
    ----------
    prefix : str
        Unique ID prefix.  The resolved palette name is stored in
        ``{prefix}-value`` (a ``dcc.Store``).
    mode : {'discrete', 'continuous'}, default='discrete'
        Controls which preset list is offered and how the swatch is rendered.
    default : str, optional
        Initial palette name.  If it matches a preset it is pre-selected in
        the dropdown; otherwise it appears in the custom text field.
    """
    presets = _DISCRETE_PRESETS if mode == "discrete" else _CONTINUOUS_PRESETS
    preset_val = default if (default and default in presets) else None
    custom_val = default if (default and default not in presets) else ""

    return html.Div(
        [
            dcc.Dropdown(
                id=f"{prefix}-preset",
                options=[{"label": p, "value": p} for p in presets],
                value=preset_val,
                clearable=True,
                searchable=True,
                placeholder="Choose a preset…",
            ),
            dbc.Input(
                id=f"{prefix}-custom",
                type="text",
                value=custom_val,
                placeholder="…or type any Plotly palette name",
                debounce=True,
                size="sm",
                style={"marginTop": "0.3rem"},
            ),
            html.Div(id=f"{prefix}-status", style=_HINT_STYLE),
            html.Div(
                id=f"{prefix}-swatch",
                children=_make_swatch(default, mode) if default else [],
                style={"marginTop": "0.4rem", "minHeight": "18px"},
            ),
            dcc.Store(id=f"{prefix}-value", data=default),
        ]
    )


# ── Callbacks ──────────────────────────────────────────────────────────────────

def register_palette_loader_callbacks(app: dash.Dash, prefix: str, mode: str = "discrete") -> None:
    """Register callbacks for the palette loader identified by *prefix*."""

    @app.callback(
        Output(f"{prefix}-value", "data"),
        Output(f"{prefix}-swatch", "children"),
        Output(f"{prefix}-status", "children"),
        Input(f"{prefix}-preset", "value"),
        Input(f"{prefix}-custom", "value"),
        prevent_initial_call=True,
    )
    def _update(preset, custom_str):
        triggered = ctx.triggered_id
        if triggered == f"{prefix}-preset":
            if preset is None:
                return no_update, [], ""
            return preset, _make_swatch(preset, mode), ""
        else:
            val = (custom_str or "").strip()
            if not val:
                return no_update, no_update, ""
            try:
                _validate(val, mode)
            except Exception as exc:
                return no_update, no_update, f"✗ {exc}"
            return val, _make_swatch(val, mode), "✓ Valid"

    @app.callback(
        Output(f"{prefix}-preset", "value", allow_duplicate=True),
        Output(f"{prefix}-custom", "value", allow_duplicate=True),
        Input(f"{prefix}-preset", "value"),
        Input(f"{prefix}-custom", "value"),
        prevent_initial_call=True,
    )
    def _cross_clear(preset, custom_str):
        triggered = ctx.triggered_id
        if triggered == f"{prefix}-preset" and preset is not None:
            return no_update, ""
        if triggered == f"{prefix}-custom" and custom_str and custom_str.strip():
            return None, no_update
        return no_update, no_update
