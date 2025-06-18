from cryocat.app.logger import dash_logger

import base64
import tempfile
import os
from dash import html, dcc
from dash import Input, Output, State, callback, no_update
import pandas as pd
import dash_bootstrap_components as dbc
from cryocat.cryomotl import Motl
from cryocat.classutils import get_class_names_by_parent
from cryocat.app.globalvars import tomo_ids
from cryocat.app.apputils import get_print_out

# motl_types = [{"label": name, "value": name} for name in get_class_names_by_parent("Motl", "cryocat.cryomotl")]


def get_motl_load_component(prefix: str, display_option="block"):
    return html.Div(
        id=f"{prefix}-motl-container",
        style={"marginTop": "1rem", "display": display_option},
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            "Motl type: ",
                            # className="text-left text-muted mb-2",
                            style={"fontStyle": "bold", "fontSize": "1.3rem", "fontColor": "var(--color11)"},
                        ),
                        width=4,
                        className="d-flex align-items-center",
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id=f"{prefix}-motl-dropdown",
                            options=["EM motl", "STOPGAP", "Relion", "Dynamo"],
                            multi=False,
                            value="EM motl",
                            style={
                                "width": "100%",
                                # "height": "20px",  # reduce height
                                # "fontSize": "1.3rem",  # smaller font
                                "padding": "0",  # reduce padding
                            },
                        ),
                        width=8,
                    ),
                ],
            ),
            dbc.Row(
                dcc.Dropdown(
                    id=f"{prefix}-motl-relion-version-dropdown",
                    options=["Version 3.0", "Version 3.1", "Version 4.x"],
                ),
                style={"display": "none"},
                id=f"{prefix}-motl-relion-version",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Input(
                            id=f"{prefix}-motl-relion-pixelsize",
                            type="number",
                            placeholder="Pixel size",
                            min=1.0,
                            step=1,
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Input(
                            id=f"{prefix}-motl-relion-binning",
                            type="number",
                            placeholder="Binning",
                            min=1.0,
                            step=1,
                        ),
                        width=6,
                    ),
                ],
                style={"display": "none"},
                id=f"{prefix}-motl-relion-options",
            ),
            dbc.Row(
                dcc.Upload(
                    id=f"{prefix}-motl-upload",
                    children=dbc.Button(
                        f"Upload {prefix} motl file", color="light", className="upload-button", size="sm"
                    ),
                    multiple=False,
                ),
                style={"marginTop": "1rem"},
            ),
        ],
    )


def register_motl_load_callbacks(prefix: str):

    @callback(
        Output(f"{prefix}-motl-data-store", "data", allow_duplicate=True),
        Input(f"{prefix}-motl-upload", "contents"),
        State(f"{prefix}-motl-upload", "filename"),
        State(f"{prefix}-motl-dropdown", "value"),
        State(f"{prefix}-motl-relion-version-dropdown", "value"),
        State(f"{prefix}-motl-relion-pixelsize", "value"),
        State(f"{prefix}-motl-relion-binning", "value"),
        prevent_initial_call=True,
    )
    def load_motl(upload_content, filename, motl_type, rln_version, rln_pixelsize, rln_binning):

        _, content_string = upload_content.split(",")
        decoded = base64.b64decode(content_string)

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[-1]) as tmp_file:
            tmp_file.write(decoded)
            tmp_file_path = tmp_file.name

        motl_type = motl_type.lower().replace(" ", "")
        if motl_type != "Relion":
            motl = Motl.load(tmp_file_path, motl_type)
        else:
            if rln_version == "Version 3.0":
                rln_kwargs = {"version": 3.0}
            elif rln_version == "Version 3.1":
                rln_kwargs = {"version": 3.0}
            else:
                rln_kwargs = {"version": 3.0, "pixel_size": rln_pixelsize, "binning": rln_binning}

            motl = Motl.load(tmp_file_path, motl_type, rln_kwargs)

        global tomo_ids
        tomo_ids = motl.get_unique_values("tomo_id")

        file_status = f"Loaded {filename};   " + get_print_out(motl)
        table_data = motl.df.to_dict("records")

        return table_data

    @callback(
        Output(f"{prefix}-motl-relion-version", "style", allow_duplicate=True),
        Output(f"{prefix}-motl-relion-options", "style", allow_duplicate=True),
        Input(f"{prefix}-motl-dropdown", "value"),
        prevent_initial_call=True,
    )
    def display_options_motl_type(motl_type):

        if motl_type == "Relion":
            return {"display": "flex", "marginTop": "1rem"}, {"display": "none"}
        else:
            return {"display": "none"}, {"display": "none"}

    @callback(
        Output(f"{prefix}-motl-relion-options", "style", allow_duplicate=True),
        Input(f"{prefix}-motl-relion-version-dropdown", "value"),
        prevent_initial_call=True,
    )
    def display_options_relion_version(relion_version):

        if relion_version == "Version 3.0" or relion_version == "Version 3.1":
            return {"display": "none"}
        else:
            return {"display": "flex", "marginTop": "1rem"}
