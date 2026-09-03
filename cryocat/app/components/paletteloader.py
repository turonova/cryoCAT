"""Palette loader component — preset dropdown + custom text input + swatch preview.

A self-contained palette selector that works for both discrete (categorical)
and continuous (colorscale) palettes.  Predefined options cover the most
commonly used Plotly palettes and the custom Monet/MonetWhite palettes from
:mod:`cryocat.analysis.visplot`.  The user may also type any valid Plotly
palette or colorscale name into the custom input; the component validates it
live and shows an error message if the name is unknown.

The resolved palette name is written to ``{prefix}-value`` (a ``dcc.Store``).
Callers should read from that store, not from the inner dropdown.  When the
user has not picked an explicit preset the store holds ``""`` (Auto).

Public API
----------
get_palette_loader(prefix, mode, default)
    Layout: preset dropdown, custom text input, status line, colour swatch.
register_palette_loader_callbacks(app, prefix, mode)
    Register all callbacks for the component.
"""

from __future__ import annotations

import dash
from dash import html, dcc, Input, Output, no_update

from cryocat.app import styles
from cryocat.app.formgen import make_dropdown

# Built-in Plotly palettes to include in the chooser (stable subset; the full
# registry has hundreds of names that would make the dropdown unusable).
_BUILTIN_DISCRETE_PRESETS = [
    "Plotly", "D3", "G10", "Vivid", "Bold", "Pastel", "Safe",
    "Alphabet", "Dark2", "Set1", "Set2", "Set3",
]
_BUILTIN_CONTINUOUS_PRESETS = [
    "Viridis", "Plasma", "Inferno", "Magma", "Cividis",
    "Jet", "Hot", "Blues", "RdBu", "Spectral",
    "Turbo", "Rainbow", "Portland", "Picnic",
]


def _discrete_presets() -> list[str]:
    """Return discrete preset names: all registered custom palettes, then selected built-ins."""
    from cryocat.analysis.visplot import CUSTOM_PALETTE_NAMES
    seen: set[str] = set()
    result: list[str] = []
    for name in CUSTOM_PALETTE_NAMES:
        if name not in seen:
            seen.add(name)
            result.append(name)
    for name in _BUILTIN_DISCRETE_PRESETS:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _continuous_presets() -> list[str]:
    """Return continuous preset names: all registered custom scales, then selected built-ins."""
    from cryocat.analysis.visplot import CUSTOM_SCALE_NAMES
    seen: set[str] = set()
    result: list[str] = []
    for name in CUSTOM_SCALE_NAMES:
        if name not in seen:
            seen.add(name)
            result.append(name)
    for name in _BUILTIN_CONTINUOUS_PRESETS:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


# Evaluated once at import time after visplot registrations have run.
_DISCRETE_PRESETS = _discrete_presets()
_CONTINUOUS_PRESETS = _continuous_presets()

# Palette name that Auto resolves to before any user choice (shown in the swatch).
_AUTO_DEFAULT_PAL = "StarryNight"


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
    from cryocat.analysis.visplot import resolve_colorscale
    import plotly.express as px
    scale = [[p, c] for p, c in resolve_colorscale(palette_val)]
    n = 24
    sampled = px.colors.sample_colorscale(scale, [i / (n - 1) for i in range(n)])
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
        from cryocat.analysis.visplot import resolve_colorscale
        resolve_colorscale(palette_val)


# ── Layout ─────────────────────────────────────────────────────────────────────

def get_palette_loader(
    prefix: str,
    mode: str = "discrete",
    default: str | None = None,
    allow_auto: bool = False,
) -> html.Div:
    """Palette selector: preset dropdown + swatch preview.

    Parameters
    ----------
    prefix : str
        Unique ID prefix.  The resolved palette name is stored in
        ``{prefix}-value`` (a ``dcc.Store``).  When *allow_auto* is True the
        store may hold ``""`` (Auto); otherwise it always holds a palette name.
    mode : {'discrete', 'continuous'}, default='discrete'
        Controls which preset list is offered and how the swatch is rendered.
    default : str, optional
        Initial palette name when *allow_auto* is False.  Falls back to the
        first preset when *default* is not in the preset list.
    allow_auto : bool, default=False
        When True, prepend an "Auto" option (``value=""``) and start the
        dropdown there.  Use for per-plot overrides where Auto means "follow
        the global default".  When False (default), no Auto option is shown
        and the dropdown starts at *default* (or the first preset).
    """
    presets = _DISCRETE_PRESETS if mode == "discrete" else _CONTINUOUS_PRESETS
    if allow_auto:
        options = [{"label": "Auto", "value": ""}] + [{"label": p, "value": p} for p in presets]
        initial = ""
        # Pre-render the swatch for the startup default so the control is never blank.
        initial_swatch = _make_swatch(_AUTO_DEFAULT_PAL, mode)
    else:
        options = [{"label": p, "value": p} for p in presets]
        initial = default if (default and default in presets) else (presets[0] if presets else "")
        initial_swatch = _make_swatch(initial, mode) if initial else []

    return html.Div(
        [
            make_dropdown(
                f"{prefix}-preset",
                options,
                initial,
                clearable=False,
            ),
            html.Div(id=f"{prefix}-status", style=styles.HINT_SM),
            html.Div(
                id=f"{prefix}-swatch",
                children=initial_swatch,
                style={"marginTop": "0.4rem", "minHeight": "18px"},
            ),
            dcc.Store(id=f"{prefix}-value", data=initial),
        ]
    )


# ── Callbacks ──────────────────────────────────────────────────────────────────

def register_palette_loader_callbacks(
    app: dash.Dash,
    prefix: str,
    mode: str = "discrete",
    settings_store_id: str | None = None,
) -> None:
    """Register callbacks for the palette loader identified by *prefix*.

    Parameters
    ----------
    settings_store_id : str, optional
        ID of the ``dcc.Store`` that holds the global graph settings dict.
        When provided the callback adds it as an ``Input`` so that the Auto
        swatch updates whenever the effective default palette changes.
        Pass this for every Auto-capable loader (``allow_auto=True``).
    """
    _auto_key = "discrete_palette" if mode == "discrete" else "continuous_palette"

    if settings_store_id:
        @app.callback(
            Output(f"{prefix}-value", "data"),
            Output(f"{prefix}-swatch", "children"),
            Output(f"{prefix}-status", "children"),
            Input(f"{prefix}-preset", "value"),
            Input(settings_store_id, "data"),
            prevent_initial_call=True,
        )
        def _update(preset, settings):
            if not preset:  # Auto — show the current effective default
                auto_pal = (settings or {}).get(_auto_key) or _AUTO_DEFAULT_PAL
                return "", _make_swatch(auto_pal, mode), ""
            return preset, _make_swatch(preset, mode), ""
    else:
        @app.callback(
            Output(f"{prefix}-value", "data"),
            Output(f"{prefix}-swatch", "children"),
            Output(f"{prefix}-status", "children"),
            Input(f"{prefix}-preset", "value"),
            prevent_initial_call=True,
        )
        def _update(preset):
            if not preset:
                return "", [], ""
            return preset, _make_swatch(preset, mode), ""
