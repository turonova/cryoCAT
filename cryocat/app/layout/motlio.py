from cryocat.app.logger import dash_logger

import base64
import tempfile
import os
import numpy as np

from dash import html, dcc
from dash import Input, Output, State, callback, no_update
import pandas as pd
import dash_bootstrap_components as dbc
from cryocat.cryomotl import Motl
from cryocat.classutils import get_class_names_by_parent
from cryocat.app.globalvars import tomo_ids
from cryocat.app.apputils import get_print_out, save_output

# motl_types = [{"label": name, "value": name} for name in get_class_names_by_parent("Motl", "cryocat.cryomotl")]


def get_motl_save_component(prefix: str):
    return html.Div(
        children=[
            html.Div(
                [
                    dbc.Button(
                        "Save output",
                        id=f"{prefix}-save-output-btn",
                        color="light",
                        style={"width": "100%"},
                    ),
                ],
                style={"width": "100%"},
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Save output")),
                    dbc.ModalBody(
                        [
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id=f"{prefix}-datasave-dropdown",
                                        multi=False,
                                        placeholder="Select data to save",
                                        style={
                                            "width": "100%",
                                            # "height": "20px",  # reduce height
                                            # "fontSize": "1.3rem",  # smaller font
                                            "padding": "0",  # reduce padding
                                            "marginBottom": "0.5rem",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id=f"{prefix}-assignment-dropdown",
                                        multi=False,
                                        options=Motl.motl_columns,
                                        placeholder="Select column to store the output",
                                        style={
                                            "width": "100%",
                                            # "height": "20px",  # reduce height
                                            # "fontSize": "1.3rem",  # smaller font
                                            "padding": "0",  # reduce padding
                                            "marginBottom": "0.5rem",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id=f"{prefix}-data-save-dropdown",
                                        options=["All", "Specific classes"],
                                        multi=False,
                                        placeholder="What to save",
                                        style={
                                            "width": "100%",
                                            # "height": "20px",  # reduce height
                                            # "fontSize": "1.3rem",  # smaller font
                                            "padding": "0",  # reduce padding
                                            "marginBottom": "0.5rem",
                                        },
                                    ),
                                    dcc.Checklist(
                                        options=[],
                                        inline=True,
                                        id=f"{prefix}-classes-checklist",
                                        labelStyle={"color": "var(--color12)", "marginRight": "1.0rem"},
                                        inputStyle={"marginRight": "5px"},
                                        className="sidebar-checklist",
                                        style={
                                            "width": "100%",
                                            # "height": "20px",  # reduce height
                                            # "fontSize": "1.3rem",  # smaller font
                                            "padding": "0",  # reduce padding
                                            "marginBottom": "0.5rem",
                                        },
                                    ),
                                    dbc.Input(
                                        id=f"{prefix}-save-path-input",
                                        type="text",
                                        placeholder="Filename (including its path)",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dbc.ModalFooter(
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "width": "100%",
                            },
                            children=[
                                html.H5("", id=f"{prefix}-status-label", style={"margin": 0}),
                                dbc.Button("Save", id=f"{prefix}-save-output-file", className="ms-auto", n_clicks=0),
                            ],
                        )
                    ),
                ],
                id=f"{prefix}-save-output-modal",
                is_open=False,
            ),
        ]
    )


def register_motl_save_callbacks(prefix: str, stored_outputs, connected_store_id):

    store_states = [State(store_id, "data") for store_id in stored_outputs.values()]

    @callback(
        Output(f"{prefix}-datasave-dropdown", "options", allow_duplicate=True),
        Output(f"{prefix}-save-output-modal", "is_open", allow_duplicate=True),
        Input(f"{prefix}-save-output-btn", "n_clicks"),
        store_states,
        prevent_initial_call=True,
    )
    def generate_data_options(n_clicks, *store_states):

        existing_data_options = []

        for key, st_data in zip(stored_outputs.keys(), store_states):
            if st_data is not None:
                existing_data_options.append(key)

        return existing_data_options, True

    @callback(
        Output(f"{prefix}-classes-checklist", "value"),
        Input(f"{prefix}-data-save-dropdown", "value"),
        prevent_initial_call=True,
    )
    def reset_checklist_value(class_value):
        return []  # Clear checklist selection

    @callback(
        Output(f"{prefix}-classes-checklist", "options"),
        Input(f"{prefix}-data-save-dropdown", "value"),
        State(f"{prefix}-datasave-dropdown", "value"),
        store_states,
        prevent_initial_call=True,
    )
    def generate_classes(class_value, data_type_value, *store_states):

        if class_value == "All":
            options = ["Drop unassigned entries"]
        elif class_value == "Specific classes":
            data_index = list(stored_outputs.keys()).index(data_type_value)
            df = pd.DataFrame(store_states[data_index])
            options = [str(v) for v in sorted(df["class"].unique())]
        else:
            options = []

        return options

    @callback(
        Output(f"{prefix}-status-label", "children", allow_duplicate=True),
        Input(f"{prefix}-save-output-file", "n_clicks"),
        State(f"{prefix}-datasave-dropdown", "value"),
        State(f"{prefix}-save-path-input", "value"),
        State(f"{prefix}-assignment-dropdown", "value"),
        State(f"{prefix}-classes-checklist", "value"),
        State(f"{prefix}-classes-checklist", "options"),
        State(connected_store_id, "data"),
        store_states,
        prevent_initial_call=True,
    )
    def save_data(
        n_clicks, data_type_value, file_path, column_id, class_filter, checklist_options, data_to_save, *store_states
    ):

        data_index = list(stored_outputs.keys()).index(data_type_value)
        results_df = pd.DataFrame(store_states[data_index])

        # Create a mapping from subtomo_id to class
        class_map = results_df.set_index("subtomo_id")["class"]

        motl_df = pd.DataFrame(data_to_save)

        # Map classes to column_id using subtomo_id
        motl_df = motl_df.copy()
        motl_df[column_id] = np.nan
        motl_df[column_id] = motl_df["subtomo_id"].map(class_map)

        if len(checklist_options) == 0:
            return no_update
        elif len(checklist_options) == 1:
            if checklist_options[0] == "Drop unassigned entries":
                all_classes = True
                if not class_filter:
                    drop_values = False
                else:
                    drop_values = True
            else:
                all_classes = False
        else:
            all_classes = False

        if all_classes:
            if drop_values:  # True: drop class==0
                motl_df = motl_df[motl_df[column_id] != 0]
            else:  # False: keep all rows, even class==0
                pass  # no filtering
        else:
            class_int = [int(x) for x in class_filter]
            motl_df = motl_df[motl_df[column_id].isin(class_int)]

        # Drop rows with no match (i.e., column_id is NaN)
        motl_df = motl_df.dropna(subset=[column_id])

        status = save_output(file_path=file_path, data_to_save=motl_df, csv_only=False)

        return status


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
                    children=dbc.Col(
                        dbc.Button(
                            f"Upload {prefix} motl file",
                            color="light",
                        ),
                        width=12,
                        className="d-grid gap-1 col-6 mx-auto mt-3 mb-3",
                    ),
                    multiple=False,
                ),
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
