"""Shared Relion options panel — version, pixel size, binning, formats, tomos.

One component used by the motl-load component (for_load=True) and all
motl-save components (for_load=False).  Publishes its assembled kwargs to
``{prefix}-rln-value`` so callers read one store instead of eight controls.

IDs follow the ``{prefix}-rln-*`` scheme exclusively.
"""
from __future__ import annotations

import os

from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from cryocat.app import styles
from cryocat.utils.ioutils import dimensions_load

from cryocat.app.components.customel import InlineLabeledDropdown, InlineInputForm
from cryocat.app.components.pathfield import get_path_field


RELION_VERSIONS: list[dict] = [
    {"label": "Version 3.0", "value": 3.0},
    {"label": "Version 3.1", "value": 3.1},
    {"label": "Version 4.x", "value": 4.0},
    {"label": "Version 5.0", "value": 5.0},
    {"label": "Version 5.1", "value": 5.1},
]


def get_relion_options(prefix: str, *, for_load: bool) -> html.Div:
    """Version dropdown + pixel size, binning, tomo/subtomo format, and Relion-5 upload.

    ``for_load`` selects the read-side fields (no 'use original entries') vs
    the write-side ones.
    """
    save_only = [] if for_load else [
        dbc.Tooltip(
            "Use the original input particle list entries where possible.",
            target=f"{prefix}-rln-use-original",
        ) if styles.TOOLTIPS_ENABLED else None,
        dbc.Checkbox(
            id=f"{prefix}-rln-use-original",
            label="Use original entries",
            value=False,
            inputStyle={"marginRight": "5px"},
            className="sidebar-checklist",
            labelStyle={"color": "var(--color9)"},
            disabled=True,
            style={"width": "30%"},
        ),
        html.Div(
            id=f"{prefix}-rln-v5-opts",
            className="hidden",
            style={"flexDirection": "column", "width": "100%", "gap": "0.3rem", "marginTop": "0.3rem"},
            children=[
                InlineInputForm(
                    id_=f"{prefix}-rln-subtomo-size",
                    type="number",
                    placeholder="Optional",
                    label="Subtomo size:",
                ),
                dbc.Checkbox(
                    id=f"{prefix}-rln-convert",
                    label="Convert to Relion STAR format",
                    value=False,
                    inputStyle={"marginRight": "5px"},
                    className="sidebar-checklist",
                    labelStyle={"color": "var(--color9)"},
                ),
            ],
        ),
    ]

    return html.Div([
        InlineLabeledDropdown(
            id_=f"{prefix}-rln-version",
            label="Version:",
            default_visibility="hidden",
            options=RELION_VERSIONS,
        ),
        html.Div(
            id=f"{prefix}-rln-options-div",
            className="hidden",
            style={"flexDirection": "column", "width": "100%"},
            children=[
                html.Div(
                    [
                        InlineInputForm(
                            id_=f"{prefix}-rln-pixelsize",
                            type="number",
                            placeholder="Pixel size",
                            step=0.001,
                            label="Pixel size (Å):",
                        ),
                        InlineInputForm(
                            id_=f"{prefix}-rln-binning",
                            type="number",
                            placeholder="Binning",
                            step=0.001,
                            label="Binning:",
                        ),
                    ],
                    style={"display": "flex", "gap": "8px", "width": "100%"},
                ),
                InlineInputForm(
                    id_=f"{prefix}-rln-tomoformat",
                    type="text",
                    placeholder="Optional",
                    label="Tomo format:",
                ),
                InlineInputForm(
                    id_=f"{prefix}-rln-subtomoformat",
                    type="text",
                    placeholder="Optional",
                    label="Subtomo format:",
                ),
                *save_only,
            ],
        ),
        html.Div(
            id=f"{prefix}-rln-tomos-row",
            className="hidden",
            style={
                "flexDirection": "column",
                "marginBottom": "0.5rem",
                "width": "100%",
                "gap": "0.3rem",
            },
            children=[
                html.Div(
                    "No tomogram file loaded",
                    id=f"{prefix}-rln-tomos-status",
                    style={"color": "var(--color9)"},
                ),
                get_path_field(
                    f"{prefix}-rln-tomos-path",
                    mode="open",
                    kind="tomos",
                    extensions=(".star",),
                    placeholder="Path to tomogram STAR file",
                ),
                dbc.Button(
                    "Load tomogram file",
                    id=f"{prefix}-rln-tomos-load-btn",
                    color="secondary",
                    size="sm",
                    style={"width": "100%"},
                ),
            ],
        ),
        dcc.Store(id=f"{prefix}-rln-tomos-store"),
        dcc.Store(id=f"{prefix}-rln-tomos-filename"),
        dcc.Store(id=f"{prefix}-rln-value"),
    ])


def register_relion_options_callbacks(
    app,
    prefix: str,
    *,
    for_load: bool,
    type_input_id: str,
    connected_motl_prefix: str | None = None,
) -> None:
    """Wire version-driven visibility and the tomogram upload handler.

    Args:
        type_input_id: ID of the external type-selector dropdown whose value
            ``"relion"`` makes the version row visible.
        connected_motl_prefix: Save-side only — prefix of the source motl whose
            loaded tomos may be reused (enables 'use original entries').
    """

    @app.callback(
        Output(f"{prefix}-rln-version", "value", allow_duplicate=True),
        Output(f"{prefix}-rln-version-topdiv", "className", allow_duplicate=True),
        Output(f"{prefix}-rln-options-div", "className", allow_duplicate=True),
        Input(type_input_id, "value"),
        prevent_initial_call=True,
    )
    def _on_type(motl_type):
        if motl_type == "relion":
            return 3.0, "flex", "flex"
        return no_update, "hidden", "hidden"

    @app.callback(
        Output(f"{prefix}-rln-tomos-store", "data", allow_duplicate=True),
        Output(f"{prefix}-rln-tomos-filename", "data", allow_duplicate=True),
        Output(f"{prefix}-rln-tomos-status", "children", allow_duplicate=True),
        Input(f"{prefix}-rln-tomos-load-btn", "n_clicks"),
        State({"type": "path-input", "owner": f"{prefix}-rln-tomos-path"}, "value"),
        prevent_initial_call=True,
    )
    def _on_tomos_load(n_clicks, path):
        import dash
        if not n_clicks or not path:
            raise dash.exceptions.PreventUpdate
        filename = os.path.basename(path)
        if for_load:
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            return content, filename, f"Tomograms loaded: {filename}"
        rln_tomos = dimensions_load(path)
        return rln_tomos.to_dict("records"), filename, f"Currently loaded: {filename}"

    if for_load:
        @app.callback(
            Output(f"{prefix}-rln-tomos-row", "className", allow_duplicate=True),
            Output(f"{prefix}-rln-pixelsize-topdiv", "className"),
            Output(f"{prefix}-rln-binning", "placeholder"),
            Input(f"{prefix}-rln-version", "value"),
            prevent_initial_call=True,
        )
        def _on_version(version):
            pxsize_cls = "hidden" if version == 3.0 else "flex"
            binning_ph = "Required" if version in (4.0, 5.0) else "Binning"
            return "flex" if version == 5.0 else "hidden", pxsize_cls, binning_ph

    elif connected_motl_prefix is not None:
        @app.callback(
            Output(f"{prefix}-rln-tomos-row", "className", allow_duplicate=True),
            Output(f"{prefix}-rln-tomos-status", "children", allow_duplicate=True),
            Output(f"{prefix}-rln-tomos-load-btn", "children"),
            Output(f"{prefix}-rln-use-original", "disabled"),
            Output(f"{prefix}-rln-use-original", "value"),
            Output(f"{prefix}-rln-pixelsize-topdiv", "className"),
            Output(f"{prefix}-rln-binning", "placeholder"),
            Input(f"{prefix}-rln-version", "value"),
            Input(f"{prefix}-rln-tomos-store", "data"),
            State(f"{connected_motl_prefix}-motl-data-type", "data"),
            State(f"{prefix}-rln-tomos-filename", "data"),
            State(f"{connected_motl_prefix}-rln-tomos-store", "data"),
            State(f"{connected_motl_prefix}-rln-tomos-filename", "data"),
            prevent_initial_call=True,
        )
        def _on_version_save(
            version, tomos, input_type, tomos_name, tomos_orig, tomos_name_orig
        ):
            return _save_version_outputs(
                version, tomos, input_type, tomos_name, tomos_orig, tomos_name_orig
            )

    else:
        @app.callback(
            Output(f"{prefix}-rln-tomos-row", "className", allow_duplicate=True),
            Output(f"{prefix}-rln-tomos-status", "children", allow_duplicate=True),
            Output(f"{prefix}-rln-tomos-load-btn", "children"),
            Output(f"{prefix}-rln-use-original", "disabled"),
            Output(f"{prefix}-rln-use-original", "value"),
            Output(f"{prefix}-rln-pixelsize-topdiv", "className"),
            Output(f"{prefix}-rln-binning", "placeholder"),
            Input(f"{prefix}-rln-version", "value"),
            Input(f"{prefix}-rln-tomos-store", "data"),
            prevent_initial_call=True,
        )
        def _on_version_save_simple(version, tomos):
            return _save_version_outputs(version, tomos, None, None, None, None)

    if not for_load:
        @app.callback(
            Output(f"{prefix}-rln-v5-opts", "className"),
            Input(f"{prefix}-rln-version", "value"),
            prevent_initial_call=True,
        )
        def _on_v5_opts(version):
            return "flex" if version in (5.0, 5.1) else "hidden"

    value_inputs = [
        Input(f"{prefix}-rln-version", "value"),
        Input(f"{prefix}-rln-pixelsize", "value"),
        Input(f"{prefix}-rln-binning", "value"),
        Input(f"{prefix}-rln-tomoformat", "value"),
        Input(f"{prefix}-rln-subtomoformat", "value"),
        Input(f"{prefix}-rln-tomos-store", "data"),
    ]
    if not for_load:
        value_inputs += [
            Input(f"{prefix}-rln-use-original", "value"),
            Input(f"{prefix}-rln-subtomo-size", "value"),
            Input(f"{prefix}-rln-convert", "value"),
        ]

    @app.callback(
        Output(f"{prefix}-rln-value", "data"),
        *value_inputs,
        prevent_initial_call=True,
    )
    def _publish(*args):
        keys = ["version", "pixel_size", "binning", "tomo_format", "subtomo_format", "tomos"]
        if not for_load:
            keys += ["use_original", "subtomo_size", "convert"]
        return dict(zip(keys, args))


def read_relion_kwargs(state: dict) -> dict:
    """Widget-state dict → kwargs accepted by RelionMotl / RelionMotlv5.  Pure."""
    if not state:
        return {}
    version = state.get("version")
    result: dict = {}
    if version is not None:
        result["version"] = float(version)
    for key, kw in (
        ("pixel_size", "pixel_size"),
        ("binning", "binning"),
        ("tomo_format", "tomo_format"),
        ("subtomo_format", "subtomo_format"),
        ("subtomo_size", "subtomo_size"),
    ):
        val = state.get(key)
        if val:
            result[kw] = val
    if state.get("convert"):
        result["convert"] = True
    if state.get("use_original"):
        result["use_original_entries"] = True
    return result


def _save_version_outputs(version, tomos, input_type, tomos_name, tomos_orig, tomos_name_orig):
    btn_title = "Load tomogram file"
    if input_type == "relion" and version in [3.0, 3.1, 4.0]:
        disable_orig = (False, False)
    elif input_type in ["relion5", "relion5_1"] and version in [5.0, 5.1]:
        disable_orig = (False, False)
    else:
        disable_orig = (True, False)

    pxsize_cls = "hidden" if version == 3.0 else "flex"
    binning_ph = "Required" if version in (4.0, 5.0) else "Binning"

    if version == 5.0:
        status = "Currently no tomogram file loaded"
        if tomos:
            status = f"Currently loaded: {tomos_name}"
            btn_title = "Load a different tomogram file"
        elif input_type == "relion5" and tomos_orig:
            status = f"Currently loaded: {tomos_name_orig}"
            btn_title = "Load a different tomogram file"
        return "flex", status, btn_title, *disable_orig, pxsize_cls, binning_ph
    return "hidden", "", btn_title, *disable_orig, pxsize_cls, binning_ph
