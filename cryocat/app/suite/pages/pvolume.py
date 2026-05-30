"""Map & Mask Preview page — multi-type masks, tight mask, and layer compositing.

§1  Mask type dropdown (sphere / shell / cylinder / ellipsoid / tight).
§2  Tight mask: reuses the already-loaded binned map + iso slider as threshold.
§3  Layer system: independent named layers, boolean op (union/intersection/
    subtraction/difference), Layers vs Result view mode, full-res compose+create.

Layer param controls use Dash pattern-matching IDs so the form can be built
dynamically per mask type without pre-registering one callback per parameter.

Contract: exposes ``layout`` and ``register_callbacks(app)``.
"""

import uuid
import numpy as np
import dash
from dash import html, dcc, Input, Output, State, no_update, ALL, MATCH
import dash_bootstrap_components as dbc

from cryocat.core import cryomap, cryomask
from cryocat.app.components.volumeview import (
    get_volume_view, register_volume_view_callbacks, mesh_at,
)


# ── Constants ──────────────────────────────────────────────────────────────────

MASK_REGISTRY = {
    "sphere":          {"label": "Sphere",          "fn": cryomask.spherical_mask},
    "spherical_shell": {"label": "Spherical shell", "fn": cryomask.spherical_shell_mask},
    "cylinder":        {"label": "Cylinder",         "fn": cryomask.cylindrical_mask},
    "ellipsoid":       {"label": "Ellipsoid",        "fn": cryomask.ellipsoid_mask},
    "ellipsoid_shell": {"label": "Ellipsoid shell",  "fn": cryomask.ellipsoid_shell_mask},
    "tight":           {"label": "Tight (from map)", "fn": cryomask.map_tight_mask},
}

# Which param groups to show per mask type
SHOW_PARAMS = {
    "sphere":          {"radius", "center", "gaussian", "gaussian_outwards"},
    "spherical_shell": {"radius", "shell_thickness", "center", "gaussian"},
    "cylinder":        {"radius", "height", "center", "angles", "gaussian", "gaussian_outwards"},
    "ellipsoid":       {"radii", "center", "angles", "gaussian", "gaussian_outwards"},
    "ellipsoid_shell": {"radii", "shell_thickness", "center", "angles", "gaussian"},
    "tight":           {"dilation_size", "gaussian", "gaussian_outwards", "n_regions", "angles"},
}

# Length params scaled by bin_factor at create time
_LEN_PARAMS = {"radius", "shell_thickness", "height", "dilation_size", "gaussian"}

LAYER_COLORS = [
    "#865B96", "#2E86C1", "#28B463", "#E67E22",
    "#E74C3C", "#1ABC9C", "#F39C12", "#8E44AD",
]

BOOL_OPS = {
    "union":        cryomask.union,
    "intersection": cryomask.intersection,
    "subtraction":  cryomask.subtraction,
    "difference":   cryomask.difference,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _default_params(mask_type, shape):
    c = [s // 2 for s in shape]
    hmin = max(1, min(shape) // 4)
    hmx  = max(1, max(shape) // 4)
    return {
        "sphere":          {"radius": hmin, "center": c, "gaussian": 0.0, "gaussian_outwards": True},
        "spherical_shell": {"radius": hmin, "shell_thickness": max(1, hmin // 4), "center": c, "gaussian": 0.0},
        "cylinder":        {"radius": hmin, "height": shape[0] // 2, "center": c,
                            "angles": [0, 0, 0], "gaussian": 0.0, "gaussian_outwards": True},
        "ellipsoid":       {"radii": [hmin, hmin, hmin], "center": c,
                            "angles": [0, 0, 0], "gaussian": 0.0, "gaussian_outwards": True},
        "ellipsoid_shell": {"radii": [hmin, hmin, hmin], "shell_thickness": max(1, hmin // 4),
                            "center": c, "angles": [0, 0, 0], "gaussian": 0.0},
        "tight":           {"dilation_size": 0, "gaussian": 0.0, "gaussian_outwards": True,
                            "n_regions": 1, "angles": [0, 0, 0]},
    }[mask_type]


def _call_mask_fn(mask_type, shape, params, vol_array=None, iso_level=None):
    """Call the cryomask function for preview (binned, in memory).

    map_tight_mask has no mask_size param and takes input_map instead.
    All length params that end up as array-slice indices must be int.
    """
    if mask_type == "tight":
        # No mask_size param; input_map drives the shape.
        if vol_array is None:
            return None
        kwargs = {
            "input_map":    vol_array,
            "threshold":    float(iso_level or 0.0),
            "dilation_size":int(params.get("dilation_size") or 0),
            "n_regions":    int(params.get("n_regions") or 1),
            "gaussian":     float(params.get("gaussian") or 0.0),
            "gaussian_outwards": bool(params.get("gaussian_outwards", True)),
        }
        angles = params.get("angles", [0, 0, 0])
        if angles and any(a for a in angles if a):
            kwargs["angles"] = [float(a or 0) for a in angles]
        return MASK_REGISTRY["tight"]["fn"](**kwargs)

    # All analytic masks take mask_size.
    kwargs = {"mask_size": tuple(int(s) for s in shape)}

    if "radius" in params and params["radius"] is not None:
        kwargs["radius"] = int(round(float(params["radius"])))
    if "radii" in params and params["radii"] is not None:
        r = params["radii"]
        kwargs["radii"] = [int(round(float(r[i]))) if r[i] is not None else 1 for i in range(3)]
    if "shell_thickness" in params and params["shell_thickness"] is not None:
        kwargs["shell_thickness"] = int(round(float(params["shell_thickness"])))
    if "height" in params and params["height"] is not None:
        # cylindrical_mask does `height //= 2` then uses as slice index → must be int
        kwargs["height"] = int(round(float(params["height"])))

    if "center" in params and params["center"] is not None:
        c = params["center"]
        kwargs["center"] = [int(round(float(c[i]))) if c[i] is not None else shape[i] // 2
                            for i in range(3)]

    kwargs["gaussian"] = float(params.get("gaussian") or 0.0)

    if mask_type not in ("spherical_shell", "ellipsoid_shell"):
        kwargs["gaussian_outwards"] = bool(params.get("gaussian_outwards", True))

    if mask_type in ("cylinder", "ellipsoid", "ellipsoid_shell"):
        angles = params.get("angles", [0, 0, 0])
        if angles and any(a for a in angles if a):
            kwargs["angles"] = [float(a or 0) for a in angles]

    return MASK_REGISTRY[mask_type]["fn"](**kwargs)


def _has_smooth_transition(m):
    """True if the mask has gradual intermediate values (i.e. gaussian > 0 was applied)."""
    return bool(np.any((m > 0.15) & (m < 0.85)))


def _scale_params(mask_type, params, bin_factor):
    """Return params scaled from binned to full-resolution."""
    bf = float(bin_factor)
    s = dict(params)
    for k in _LEN_PARAMS:
        if k in s and s[k] is not None:
            s[k] = s[k] * bf
    for k in ("center", "radii"):
        if k in s and s[k] is not None:
            s[k] = [v * bf if v is not None else 0.0 for v in s[k]]
    return s


# ── Layout helpers ─────────────────────────────────────────────────────────────

def _slider_row(label, slider_id, input_id, min_=0, max_=100, step=1, value=0):
    return html.Div(
        [
            html.Label(label, style={"fontSize": "0.85rem", "width": "40%",
                                     "flexShrink": 0, "paddingRight": "0.4rem"}),
            html.Div(
                dcc.Slider(id=slider_id, min=min_, max=max_, step=step, value=value,
                           marks=None, tooltip={"placement": "left", "always_visible": False}),
                style={"flex": 1, "minWidth": 0},
            ),
            dbc.Input(id=input_id, type="number", value=value, min=min_, max=max_, step=step,
                      style={"width": "64px", "flexShrink": 0, "fontSize": "0.8rem",
                             "padding": "2px 4px", "marginLeft": "0.4rem"}),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": "0.15rem"},
    )


def _lp_slider_row(label, param, value=0, min_=0, max_=100, step=1):
    """Layer-param slider row with pattern-matching IDs."""
    sid = {"type": "vol-lp-slider", "param": param}
    iid = {"type": "vol-lp-input",  "param": param}
    return html.Div(
        [
            html.Label(label, style={"fontSize": "0.85rem", "width": "40%",
                                     "flexShrink": 0, "paddingRight": "0.4rem"}),
            html.Div(
                dcc.Slider(id=sid, min=min_, max=max_, step=step, value=value,
                           marks=None, tooltip={"placement": "left", "always_visible": False}),
                style={"flex": 1, "minWidth": 0},
            ),
            dbc.Input(id=iid, type="number", value=value, min=min_, max=max_, step=step,
                      style={"width": "64px", "flexShrink": 0, "fontSize": "0.8rem",
                             "padding": "2px 4px", "marginLeft": "0.4rem"}),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": "0.15rem"},
    )


def _lp_number_row(label, param, value=0, min_=None, max_=None, step=1):
    """Layer-param plain number input row (for angles, n_regions)."""
    iid = {"type": "vol-lp-number", "param": param}
    kw = {"type": "number", "id": iid, "value": value, "step": step,
          "style": {"flex": 1, "fontSize": "0.8rem", "padding": "2px 4px"}}
    if min_ is not None:
        kw["min"] = min_
    if max_ is not None:
        kw["max"] = max_
    return html.Div(
        [
            html.Label(label, style={"fontSize": "0.85rem", "width": "40%",
                                     "flexShrink": 0, "paddingRight": "0.4rem"}),
            dbc.Input(**kw),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": "0.15rem"},
    )


def _build_layer_params_form(mask_type, params, shape):
    """Build the layer-param form for the given mask type + current values."""
    show = SHOW_PARAMS.get(mask_type, set())
    hmin   = max(1, min(shape) // 2)
    maxdim = max(shape)
    halfmx = max(1, max(shape) // 2)
    rows = []

    if mask_type == "tight":
        rows.append(html.Div(
            "Uses loaded map as template; threshold = current isosurface level.",
            style={"fontSize": "0.8rem", "color": "var(--color9)", "marginBottom": "0.4rem",
                   "fontStyle": "italic"},
        ))

    if "radius" in show:
        rows.append(_lp_slider_row("Radius", "radius",
                                   value=params.get("radius", hmin // 2),
                                   min_=1, max_=hmin, step=1))

    if "shell_thickness" in show:
        rows.append(_lp_slider_row("Shell thickness", "shell_thickness",
                                   value=params.get("shell_thickness", max(1, hmin // 4)),
                                   min_=1, max_=hmin, step=1))

    if "height" in show:
        rows.append(_lp_slider_row("Height", "height",
                                   value=params.get("height", shape[0] // 2),
                                   min_=1, max_=maxdim, step=1))

    if "radii" in show:
        rv = params.get("radii", [hmin // 2, hmin // 2, hmin // 2])
        rows.append(html.Label("Radii (x, y, z)", style={"fontSize": "0.85rem"}))
        for i, ax in enumerate(("x", "y", "z")):
            rows.append(_lp_slider_row(f"  {ax}", f"radii_{i}",
                                       value=rv[i] if rv and rv[i] is not None else hmin // 2,
                                       min_=1, max_=halfmx, step=1))

    if "center" in show:
        cv = params.get("center", [s // 2 for s in shape])
        for i, ax in enumerate(("x", "y", "z")):
            rows.append(_lp_slider_row(f"Center {ax}", f"center_{i}",
                                       value=cv[i] if cv and cv[i] is not None else shape[i] // 2,
                                       min_=0, max_=shape[i] - 1, step=1))

    if "angles" in show:
        av = params.get("angles", [0, 0, 0])
        for i, lbl in enumerate(("Euler angle φ (Z)", "Euler angle θ (X)", "Euler angle ψ (Z)")):
            rows.append(_lp_number_row(lbl, f"angles_{i}",
                                       value=av[i] if av else 0, step=1))

    if "gaussian" in show:
        rows.append(_lp_slider_row("Gaussian", "gaussian",
                                   value=params.get("gaussian", 0.0),
                                   min_=0.0, max_=10.0, step=0.5))

    if "gaussian_outwards" in show:
        gv = params.get("gaussian_outwards", True)
        rows.append(html.Div(
            [
                html.Label("Gaussian outwards", style={"fontSize": "0.85rem", "width": "40%",
                                                        "flexShrink": 0}),
                dcc.Dropdown(
                    id={"type": "vol-lp-dropdown", "param": "gaussian_outwards"},
                    options=["True", "False"],
                    value="True" if gv else "False",
                    style={"flex": 1, "fontSize": "0.8rem"},
                    clearable=False,
                    searchable=False,
                ),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "0.15rem"},
        ))

    if "dilation_size" in show:
        rows.append(_lp_slider_row("Dilation size", "dilation_size",
                                   value=params.get("dilation_size", 0),
                                   min_=0, max_=20, step=1))

    if "n_regions" in show:
        rows.append(_lp_number_row("N regions", "n_regions",
                                   value=params.get("n_regions", 1), min_=1, step=1))

    return rows


def _layer_row(layer, idx, n_total, is_selected):
    """Build one row in the layer list."""
    lid = layer["id"]
    color = layer.get("color", LAYER_COLORS[0])
    name = layer.get("name", f"Layer {idx + 1}")
    visible = layer.get("visible", True)
    btn = lambda action, label, **kw: dbc.Button(
        label,
        id={"type": "vol-layer-btn", "action": action, "lid": lid},
        size="sm", color="link", n_clicks=0,
        style={"padding": "0 2px", "fontSize": "0.75rem", "lineHeight": "1"},
        **kw,
    )
    return html.Div(
        [
            html.Span(f"{idx + 1}.", style={"width": "16px", "fontSize": "0.75rem", "flexShrink": 0}),
            html.Div(style={"width": "10px", "height": "10px", "borderRadius": "2px",
                            "backgroundColor": color, "flexShrink": 0, "marginRight": "3px"}),
            html.Span(name, style={"flex": 1, "fontSize": "0.8rem",
                                   "overflow": "hidden", "textOverflow": "ellipsis",
                                   "whiteSpace": "nowrap",
                                   "color": "var(--color11)" if is_selected else "var(--color12)"}),
            btn("select", "✎"),
            btn("toggle", "👁" if visible else "○"),
            btn("up",     "▲", disabled=(idx == 0)),
            btn("down",   "▼", disabled=(idx == n_total - 1)),
            btn("remove", "✕"),
        ],
        style={
            "display": "flex", "alignItems": "center", "gap": "1px",
            "padding": "2px 4px", "marginBottom": "2px", "borderRadius": "4px",
            "backgroundColor": "var(--color10)" if is_selected else "var(--color6)",
            "cursor": "pointer",
        },
    )


def _sidebar():
    type_options = [{"label": v["label"], "value": k} for k, v in MASK_REGISTRY.items()]
    return dbc.Col(
        html.Div(
            [
                dbc.Accordion(
                    [
                        # ── Map ───────────────────────────────────────────────
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    [
                                        html.Label("Map file", style={"fontSize": "0.85rem",
                                                                       "width": "40%", "flexShrink": 0,
                                                                       "paddingRight": "0.4rem"}),
                                        dbc.Input(id="vol-path-input", type="text",
                                                  placeholder="Path to .em / .mrc file",
                                                  style={"flex": 1}),
                                    ],
                                    style={"display": "flex", "alignItems": "center",
                                           "marginBottom": "0.3rem"},
                                ),
                                html.Div(
                                    [
                                        html.Label("Bin factor", style={"fontSize": "0.85rem",
                                                                         "width": "40%", "flexShrink": 0,
                                                                         "paddingRight": "0.4rem"}),
                                        dbc.Input(id="vol-bin-input", type="number",
                                                  value=1, min=1, max=16, step=1,
                                                  style={"flex": 1}),
                                    ],
                                    style={"display": "flex", "alignItems": "center",
                                           "marginBottom": "0.3rem"},
                                ),
                                dbc.Button("Load map", id="vol-load-btn", color="primary",
                                           size="sm", style={"width": "100%"}),
                                html.Div(id="vol-load-status",
                                         style={"fontSize": "0.8rem", "color": "var(--color9)",
                                                "marginTop": "0.4rem", "wordBreak": "break-word"}),
                            ],
                            title="Map", item_id="vol-acc-map",
                        ),
                        # ── Display ───────────────────────────────────────────
                        dbc.AccordionItem(
                            [
                                _slider_row("Isosurface level",   "vol-iso-slider",          "vol-iso-input",
                                            min_=0.0, max_=1.0, step=0.01, value=0.5),
                                _slider_row("Mask overlay opacity","vol-mask-opacity-slider", "vol-mask-opacity-input",
                                            min_=0.0, max_=1.0, step=0.05, value=0.3),
                            ],
                            title="Display", item_id="vol-acc-display",
                        ),
                        # ── Mask layers ───────────────────────────────────────
                        dbc.AccordionItem(
                            [
                                # Add-layer row
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="vol-add-type-dropdown",
                                            options=type_options,
                                            value="sphere",
                                            clearable=False,
                                            searchable=False,
                                            style={"flex": 1, "fontSize": "0.8rem"},
                                        ),
                                        dbc.Button("Add", id="vol-add-layer-btn",
                                                   color="primary", size="sm",
                                                   style={"marginLeft": "0.4rem"}),
                                    ],
                                    style={"display": "flex", "alignItems": "center",
                                           "marginBottom": "0.5rem"},
                                ),
                                # Layer list
                                html.Div(id="vol-layer-list",
                                         children=[html.Div("No layers yet.",
                                                            style={"fontSize": "0.8rem",
                                                                   "color": "var(--color9)"})],
                                         style={"marginBottom": "0.5rem"}),
                                html.Hr(style={"margin": "0.4rem 0"}),
                                # Combine controls
                                html.Div(
                                    [
                                        html.Label("Combine", style={"fontSize": "0.85rem",
                                                                      "width": "40%", "flexShrink": 0}),
                                        dcc.Dropdown(
                                            id="vol-bool-op",
                                            options=[
                                                {"label": "Union",        "value": "union"},
                                                {"label": "Intersection", "value": "intersection"},
                                                {"label": "Subtraction",  "value": "subtraction"},
                                                {"label": "Difference",   "value": "difference"},
                                            ],
                                            value="union", clearable=False,
                                            searchable=False,
                                            style={"flex": 1, "fontSize": "0.8rem"},
                                        ),
                                    ],
                                    style={"display": "flex", "alignItems": "center",
                                           "marginBottom": "0.3rem"},
                                ),
                                html.Div(
                                    [
                                        html.Label("View", style={"fontSize": "0.85rem",
                                                                   "width": "40%", "flexShrink": 0}),
                                        dcc.RadioItems(
                                            id="vol-view-mode",
                                            options=[{"label": " Layers", "value": "layers"},
                                                     {"label": " Result", "value": "result"}],
                                            value="layers", inline=True,
                                            style={"fontSize": "0.8rem", "display": "flex",
                                                   "alignItems": "center"},
                                        ),
                                    ],
                                    style={"display": "flex", "alignItems": "center"},
                                ),
                            ],
                            title="Mask layers", item_id="vol-acc-layers",
                        ),
                        # ── Selected layer params ─────────────────────────────
                        dbc.AccordionItem(
                            [
                                html.Div(id="vol-layer-params-form",
                                         children=[html.Div("Select a layer to edit.",
                                                            style={"fontSize": "0.8rem",
                                                                   "color": "var(--color9)"})]),
                            ],
                            title="Layer parameters", item_id="vol-acc-lparams",
                        ),
                        # ── Create mask ───────────────────────────────────────
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    [
                                        html.Label("Output path", style={"fontSize": "0.85rem",
                                                                          "width": "40%", "flexShrink": 0,
                                                                          "paddingRight": "0.4rem"}),
                                        dbc.Input(id="vol-output-path-input", type="text",
                                                  placeholder="output_mask.em",
                                                  style={"flex": 1}),
                                    ],
                                    style={"display": "flex", "alignItems": "center",
                                           "marginBottom": "0.3rem"},
                                ),
                                html.Div(
                                    [
                                        html.Label("Pixel size (Å)", style={"fontSize": "0.85rem",
                                                                              "width": "40%", "flexShrink": 0,
                                                                              "paddingRight": "0.4rem"}),
                                        dbc.Input(id="vol-pixel-size-input", type="number",
                                                  value=1.0, min=0.001, step=0.001,
                                                  style={"flex": 1}),
                                    ],
                                    style={"display": "flex", "alignItems": "center",
                                           "marginBottom": "0.3rem"},
                                ),
                                dbc.Button("Create mask", id="vol-create-btn",
                                           size="sm",
                                           style={"width": "100%", "backgroundColor": "var(--color12)", "borderColor": "var(--color12)"}),
                                html.Div(id="vol-create-status",
                                         style={"fontSize": "0.8rem", "color": "var(--color9)",
                                                "marginTop": "0.4rem", "wordBreak": "break-word"}),
                            ],
                            title="Create mask", item_id="vol-acc-create",
                        ),
                    ],
                    always_open=True,
                    active_item=["vol-acc-map", "vol-acc-display", "vol-acc-layers",
                                 "vol-acc-lparams", "vol-acc-create"],
                ),
            ],
            className="sidebar",
            style={"padding": "0.5rem", "overflowY": "auto", "height": "100vh",
                   "display": "flex", "flexDirection": "column"},
        ),
        width=3,
        style={"margin": "0", "padding": "0", "height": "100vh",
               "position": "sticky", "top": "0px"},
    )


def _main():
    return dbc.Col(
        html.Div(
            [get_volume_view("vol")],
            style={"padding": "0.5rem"},
        ),
        width=9,
        style={"margin": "0", "padding": "0"},
    )


layout = html.Div(
    [
        dcc.Store(id="vol-meta-store"),
        dcc.Store(id="vol-mask-layers", data=[]),
        dcc.Store(id="vol-selected-layer", data=None),
        dbc.Row([_sidebar(), _main()], className="g-0",
                style={"margin": "0", "padding": "0"}),
    ],
    style={"margin": "0", "padding": "0"},
)


# ── Callbacks ──────────────────────────────────────────────────────────────────

def register_callbacks(app):
    register_volume_view_callbacks(app, "vol", register_mask=False)

    # ── Load map ──────────────────────────────────────────────────────────────
    @app.callback(
        Output("vol-map-store",       "data"),
        Output("vol-meta-store",      "data"),
        Output("vol-iso-slider",      "min"),
        Output("vol-iso-slider",      "max"),
        Output("vol-iso-slider",      "value"),
        Output("vol-iso-slider",      "step"),
        Output("vol-load-status",     "children"),
        Input("vol-load-btn",  "n_clicks"),
        State("vol-path-input", "value"),
        State("vol-bin-input",  "value"),
        prevent_initial_call=True,
    )
    def _load_map(n_clicks, path, bin_factor):
        if not n_clicks or not path:
            raise dash.exceptions.PreventUpdate
        bin_factor = max(1, int(bin_factor or 4))
        try:
            vol = cryomap.read(path)
        except Exception as exc:
            fail = [no_update] * 6 + [f"Error loading map: {exc}"]
            return fail

        shape_full   = list(vol.shape)
        vol_binned   = vol[::bin_factor, ::bin_factor, ::bin_factor].astype(np.float32)
        shape_binned = list(vol_binned.shape)

        dmin      = float(vol_binned.min())
        dmax      = float(vol_binned.max())
        iso_def   = float(np.percentile(vol_binned, 70))
        iso_step  = float(max((dmax - dmin) / 200, 1e-6))

        map_data = {"data": vol_binned.tolist(), "shape": shape_binned}
        meta     = {"bin_factor": bin_factor, "shape_full": shape_full,
                    "shape_binned": shape_binned, "path": path}

        status = (
            f"Loaded: {path}  |  Full: {shape_full}  |  "
            f"Binned {bin_factor}×: {shape_binned}  |  "
            f"Range: [{dmin:.3f}, {dmax:.3f}]"
        )
        return map_data, meta, dmin, dmax, iso_def, iso_step, status

    # ── Iso slider → input sync ───────────────────────────────────────────────
    @app.callback(
        Output("vol-iso-input",          "value"),
        Output("vol-mask-opacity-input", "value"),
        Input("vol-iso-slider",          "value"),
        Input("vol-mask-opacity-slider", "value"),
        prevent_initial_call=True,
    )
    def _display_sliders_to_inputs(iso, opacity):
        return iso, opacity

    _display_pairs = [
        ("vol-iso-input",          "vol-iso-slider"),
        ("vol-mask-opacity-input", "vol-mask-opacity-slider"),
    ]
    for _iid, _sid in _display_pairs:
        @app.callback(
            Output(_sid, "value", allow_duplicate=True),
            Input(_iid, "n_blur"),
            Input(_iid, "n_submit"),
            State(_iid, "value"),
            prevent_initial_call=True,
        )
        def _display_input_to_slider(n_blur, n_submit, val):
            if val is None:
                raise dash.exceptions.PreventUpdate
            return val

    # ── Add layer ─────────────────────────────────────────────────────────────
    @app.callback(
        Output("vol-mask-layers",    "data",  allow_duplicate=True),
        Output("vol-selected-layer", "data",  allow_duplicate=True),
        Input("vol-add-layer-btn",   "n_clicks"),
        State("vol-add-type-dropdown", "value"),
        State("vol-mask-layers",       "data"),
        State("vol-meta-store",        "data"),
        prevent_initial_call=True,
    )
    def _add_layer(n_clicks, mask_type, layers, meta):
        if not n_clicks or not mask_type:
            raise dash.exceptions.PreventUpdate
        layers = layers or []
        shape = (meta or {}).get("shape_binned", [64, 64, 64])
        lid = str(uuid.uuid4())
        color = LAYER_COLORS[len(layers) % len(LAYER_COLORS)]
        new_layer = {
            "id":      lid,
            "name":    f"{MASK_REGISTRY[mask_type]['label']} {len(layers) + 1}",
            "type":    mask_type,
            "params":  _default_params(mask_type, shape),
            "visible": True,
            "color":   color,
        }
        return layers + [new_layer], lid

    # ── Layer list actions (select / toggle / up / down / remove) ─────────────
    @app.callback(
        Output("vol-mask-layers",    "data",  allow_duplicate=True),
        Output("vol-selected-layer", "data",  allow_duplicate=True),
        Input({"type": "vol-layer-btn", "action": ALL, "lid": ALL}, "n_clicks"),
        State("vol-mask-layers",    "data"),
        State("vol-selected-layer", "data"),
        prevent_initial_call=True,
    )
    def _layer_action(n_clicks_list, layers, selected):
        ctx = dash.callback_context
        if not ctx.triggered_id or not any(n for n in (n_clicks_list or []) if n):
            raise dash.exceptions.PreventUpdate
        tid = ctx.triggered_id
        action = tid["action"]
        lid    = tid["lid"]
        layers = list(layers or [])

        if action == "select":
            return no_update, lid

        if action == "toggle":
            for l in layers:
                if l["id"] == lid:
                    l["visible"] = not l.get("visible", True)
            return layers, selected

        if action == "remove":
            layers = [l for l in layers if l["id"] != lid]
            new_sel = None if selected == lid else selected
            return layers, new_sel

        if action == "up":
            idx = next((i for i, l in enumerate(layers) if l["id"] == lid), None)
            if idx is not None and idx > 0:
                layers[idx - 1], layers[idx] = layers[idx], layers[idx - 1]
            return layers, selected

        if action == "down":
            idx = next((i for i, l in enumerate(layers) if l["id"] == lid), None)
            if idx is not None and idx < len(layers) - 1:
                layers[idx + 1], layers[idx] = layers[idx], layers[idx + 1]
            return layers, selected

        raise dash.exceptions.PreventUpdate

    # ── Build layer list display ──────────────────────────────────────────────
    @app.callback(
        Output("vol-layer-list", "children"),
        Input("vol-mask-layers",    "data"),
        Input("vol-selected-layer", "data"),
    )
    def _build_layer_list(layers, selected):
        if not layers:
            return [html.Div("No layers yet.",
                             style={"fontSize": "0.8rem", "color": "var(--color9)"})]
        return [_layer_row(l, i, len(layers), l["id"] == selected)
                for i, l in enumerate(layers)]

    # ── Build layer params form ───────────────────────────────────────────────
    @app.callback(
        Output("vol-layer-params-form", "children"),
        Input("vol-selected-layer",  "data"),
        State("vol-mask-layers",     "data"),
        State("vol-meta-store",      "data"),
        prevent_initial_call=True,
    )
    def _build_layer_params_form_cb(selected_id, layers, meta):
        if not selected_id or not layers:
            return [html.Div("Select a layer to edit its parameters.",
                             style={"fontSize": "0.8rem", "color": "var(--color9)"})]
        layer = next((l for l in layers if l["id"] == selected_id), None)
        if not layer:
            return []
        shape = (meta or {}).get("shape_binned", [64, 64, 64])
        return _build_layer_params_form(layer["type"], layer.get("params", {}), shape)

    # ── Layer param slider → input sync (pattern-matching) ───────────────────
    @app.callback(
        Output({"type": "vol-lp-input", "param": MATCH}, "value"),
        Input({"type": "vol-lp-slider", "param": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def _lp_slider_to_input(val):
        return val

    @app.callback(
        Output({"type": "vol-lp-slider", "param": MATCH}, "value", allow_duplicate=True),
        Input({"type": "vol-lp-input",   "param": MATCH}, "n_blur"),
        Input({"type": "vol-lp-input",   "param": MATCH}, "n_submit"),
        State({"type": "vol-lp-input",   "param": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def _lp_input_to_slider(n_blur, n_submit, val):
        if val is None:
            raise dash.exceptions.PreventUpdate
        return val

    # ── Save layer params to store ────────────────────────────────────────────
    @app.callback(
        Output("vol-mask-layers", "data", allow_duplicate=True),
        Input({"type": "vol-lp-slider",   "param": ALL}, "value"),
        Input({"type": "vol-lp-dropdown", "param": ALL}, "value"),
        Input({"type": "vol-lp-number",   "param": ALL}, "value"),
        State({"type": "vol-lp-slider",   "param": ALL}, "id"),
        State({"type": "vol-lp-dropdown", "param": ALL}, "id"),
        State({"type": "vol-lp-number",   "param": ALL}, "id"),
        State("vol-selected-layer", "data"),
        State("vol-mask-layers",    "data"),
        prevent_initial_call=True,
    )
    def _save_layer_params(s_vals, d_vals, n_vals,
                           s_ids,  d_ids,  n_ids,
                           selected_id, layers):
        if not selected_id or not layers:
            raise dash.exceptions.PreventUpdate

        # Build flat params dict, collapsing triplet params
        params = {}
        for val, cid in zip(s_vals, s_ids):
            p = cid["param"]
            if "_" in p and p.rsplit("_", 1)[-1].isdigit():
                name, idx = p.rsplit("_", 1)
                params.setdefault(name, [None, None, None])
                params[name][int(idx)] = val
            else:
                params[p] = val
        for val, cid in zip(d_vals, d_ids):
            p = cid["param"]
            params[p] = (val == "True") if val in ("True", "False") else val
        for val, cid in zip(n_vals, n_ids):
            p = cid["param"]
            if "_" in p and p.rsplit("_", 1)[-1].isdigit():
                name, idx = p.rsplit("_", 1)
                params.setdefault(name, [0, 0, 0])
                params[name][int(idx)] = val or 0
            else:
                params[p] = val

        new_layers = []
        for layer in layers:
            if layer["id"] == selected_id:
                merged = {**layer.get("params", {}), **params}
                layer  = {**layer, "params": merged}
            new_layers.append(layer)
        return new_layers

    # ── Compute layer meshes (multi-layer preview) ────────────────────────────
    @app.callback(
        Output("vol-mask-mesh", "data"),
        Input("vol-mask-layers",    "data"),
        Input("vol-bool-op",        "value"),
        Input("vol-view-mode",      "value"),
        Input("vol-iso-slider",     "value"),
        State("vol-map-store",      "data"),
        prevent_initial_call=True,
    )
    def _extract_layers(layers, bool_op, view_mode, iso_level, map_data):
        if not layers or not map_data:
            return None
        shape    = map_data.get("shape")
        if not shape:
            return None
        vol_arr  = np.asarray(map_data["data"], dtype=np.float32)
        visible  = [l for l in layers if l.get("visible", True)]
        if not visible:
            return None

        layer_masks = []
        for layer in visible:
            try:
                m = _call_mask_fn(layer["type"], shape, layer.get("params", {}),
                                  vol_arr, iso_level)
                if m is not None:
                    layer_masks.append((layer, m))
            except Exception:
                continue

        if not layer_masks:
            return None

        if view_mode == "result" and len(layer_masks) > 0:
            arrays = [m for _, m in layer_masks]
            if len(arrays) == 1:
                combined = arrays[0]
            else:
                op_fn    = BOOL_OPS.get(bool_op or "union", cryomask.union)
                combined = op_fn(arrays)
            smooth = _has_smooth_transition(combined)
            return [{"core": mesh_at(combined, 0.99),
                     "outer": mesh_at(combined, 0.10) if smooth else None,
                     "color": "#865B96", "name": "Result"}]

        return [
            {"core":  mesh_at(m, 0.99),
             "outer": mesh_at(m, 0.10) if _has_smooth_transition(m) else None,
             "color": layer.get("color", LAYER_COLORS[0]),
             "name":  layer.get("name", "Layer")}
            for layer, m in layer_masks
        ]

    # ── Create mask (full resolution) ─────────────────────────────────────────
    @app.callback(
        Output("vol-create-status", "children"),
        Input("vol-create-btn",     "n_clicks"),
        State("vol-output-path-input", "value"),
        State("vol-pixel-size-input",  "value"),
        State("vol-mask-layers",       "data"),
        State("vol-meta-store",        "data"),
        State("vol-bool-op",           "value"),
        State("vol-iso-slider",        "value"),
        prevent_initial_call=True,
    )
    def _create_mask(n_clicks, output_path, pixel_size, layers, meta, bool_op, iso_level):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not output_path:
            return "Provide an output path."
        if not layers or not meta:
            return "Load a map and add at least one layer."

        shape_full = meta.get("shape_full")
        bin_factor = float(meta.get("bin_factor", 1))
        map_path   = meta.get("path")
        visible    = [l for l in layers if l.get("visible", True)]
        if not visible:
            return "No visible layers to create."

        mask_arrays = []
        for layer in visible:
            mask_type = layer["type"]
            scaled    = _scale_params(mask_type, layer.get("params", {}), bin_factor)
            fn        = MASK_REGISTRY[mask_type]["fn"]
            kwargs    = {"mask_size": tuple(int(s) for s in shape_full)}

            try:
                if mask_type == "tight":
                    if not map_path:
                        return "Map path unavailable for tight mask — reload the map."
                    # tight mask: no mask_size param; drives shape from input_map
                    kwargs = {
                        "input_map":     map_path,
                        "threshold":     float(iso_level or 0.0),  # intensity, not scaled
                        "dilation_size": int(scaled.get("dilation_size") or 0),
                        "n_regions":     int(layer["params"].get("n_regions") or 1),
                        "gaussian":      float(scaled.get("gaussian") or 0.0),
                        "gaussian_outwards": bool(layer["params"].get("gaussian_outwards", True)),
                    }
                    angles = layer["params"].get("angles", [0, 0, 0])
                    if angles and any(a for a in angles if a):
                        kwargs["angles"] = [float(a or 0) for a in angles]
                else:
                    if "radius" in scaled and scaled["radius"] is not None:
                        kwargs["radius"] = int(round(float(scaled["radius"])))
                    if "radii" in scaled and scaled["radii"] is not None:
                        kwargs["radii"] = [int(round(float(r or 1))) for r in scaled["radii"]]
                    if "shell_thickness" in scaled and scaled["shell_thickness"] is not None:
                        kwargs["shell_thickness"] = int(round(float(scaled["shell_thickness"])))
                    if "height" in scaled and scaled["height"] is not None:
                        kwargs["height"] = int(round(float(scaled["height"])))

                    if "center" in scaled and scaled["center"] is not None:
                        kwargs["center"] = [int(round(float(c or 0))) for c in scaled["center"]]

                    kwargs["gaussian"] = float(scaled.get("gaussian") or 0.0)

                    if mask_type not in ("spherical_shell", "ellipsoid_shell"):
                        kwargs["gaussian_outwards"] = bool(
                            layer["params"].get("gaussian_outwards", True))

                    if mask_type in ("cylinder", "ellipsoid", "ellipsoid_shell"):
                        angles = layer["params"].get("angles", [0, 0, 0])
                        if angles and any(a for a in angles if a):
                            kwargs["angles"] = [float(a or 0) for a in angles]

                mask_arrays.append(fn(**kwargs))
            except Exception as exc:
                return f"Error creating layer '{layer.get('name')}': {exc}"

        try:
            if len(mask_arrays) == 1:
                final = mask_arrays[0]
            else:
                op_fn = BOOL_OPS.get(bool_op or "union", cryomask.union)
                final = op_fn(mask_arrays)
            cryomap.write(final.astype(np.float32), output_path, pixel_size=float(pixel_size or 1.0))
        except Exception as exc:
            return f"Error saving: {exc}"

        names = ", ".join(l.get("name", "") for l in visible)
        return f"Mask created: {output_path}  ({len(visible)} layer(s): {names})"
