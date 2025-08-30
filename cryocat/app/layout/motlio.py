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
from cryocat.app.apputils import get_print_out, save_output, save_motl
from cryocat.app.layout.customel import InlineLabeledDropdown, InlineInputForm


# motl_types = [{"label": name, "value": name} for name in get_class_names_by_parent("Motl", "cryocat.cryomotl")]

motl_types = [
    {"label": "EM", "value": "emmotl"},
    {"label": "STOPGAP", "value": "stopgap"},
    {"label": "Relion", "value": "relion"},
    {"label": "Dynamo", "value": "dynamo"},
]

relion_versions = [
    {"label": "Version 3.0", "value": 3.0},
    {"label": "Version 3.1", "value": 3.1},
    {"label": "Version 4.x", "value": 4.0},
    {"label": "Version 5.x", "value": 5.0},
]


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
                                    InlineLabeledDropdown(
                                        id_=f"{prefix}-datasave-dropdown",
                                        label="Data to save:",
                                        multi=False,
                                        placeholder="Select data to save",
                                    ),
                                    InlineLabeledDropdown(
                                        id_=f"{prefix}-assignment-dropdown",
                                        label="Store in the column:",
                                        multi=False,
                                        options=Motl.motl_columns,
                                        placeholder="Select column to store the output",
                                    ),
                                    InlineLabeledDropdown(
                                        id_=f"{prefix}-data-save-dropdown",
                                        options=["All", "Specific classes"],
                                        label="Select class:",
                                        multi=False,
                                        placeholder="Select what to save",
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
                                    InlineLabeledDropdown(
                                        id_=f"{prefix}-data-save-type-dropdown",
                                        options=motl_types,
                                        label="Output type:",
                                        multi=False,
                                        placeholder="Output type",
                                    ),
                                    InlineLabeledDropdown(
                                        id_=f"{prefix}-save-motl-relion-version-dropdown",
                                        label="Version:",
                                        default_visibility="hidden",
                                        options=relion_versions,
                                    ),
                                    html.Div(
                                        id=f"{prefix}-save-motl-relion-options",
                                        className="hidden",
                                        children=[
                                            InlineInputForm(
                                                id_=f"{prefix}-save-motl-relion-pixelsize",
                                                type="number",
                                                placeholder="Pixel size",
                                                min=1.0,
                                                step=1,
                                                label="Pixel size (A):",
                                            ),
                                            InlineInputForm(
                                                id_=f"{prefix}-save-motl-relion-binning",
                                                type="number",
                                                placeholder="Binning",
                                                min=1.0,
                                                step=1,
                                                label="Binning:",
                                            ),
                                        ],
                                    ),
                                    InlineInputForm(
                                        id_=f"{prefix}-save-path-input",
                                        label="Filename:",
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
        Output(f"{prefix}-save-motl-relion-version-dropdown", "value", allow_duplicate=True),
        Output(f"{prefix}-save-motl-relion-version-dropdown-topdiv", "className", allow_duplicate=True),
        Output(f"{prefix}-save-motl-relion-options", "className", allow_duplicate=True),
        Input(f"{prefix}-data-save-type-dropdown", "value"),
        prevent_initial_call=True,
    )
    def display_options_motl_type(motl_type):

        if motl_type == "relion":
            return "Version 3.0", "flex", "hidden"
        else:
            return "Version 3.0", "hidden", "hidden"

    @callback(
        Output(f"{prefix}-save-motl-relion-options", "className", allow_duplicate=True),
        Input(f"{prefix}-save-motl-relion-version-dropdown", "value"),
        prevent_initial_call=True,
    )
    def display_options_relion_version(relion_version):

        if relion_version == "Version 3.0" or relion_version == "Version 3.1":
            return "hidden"
        elif relion_version == "Version 4.x":
            return "flex"
        else:
            return "flex"

    @callback(
        Output(f"{prefix}-status-label", "children", allow_duplicate=True),
        Input(f"{prefix}-save-output-file", "n_clicks"),
        State(f"{prefix}-datasave-dropdown", "value"),
        State(f"{prefix}-save-path-input", "value"),
        State(f"{prefix}-assignment-dropdown", "value"),
        State(f"{prefix}-classes-checklist", "value"),
        State(f"{prefix}-classes-checklist", "options"),
        State(connected_store_id, "data"),
        State(f"{prefix}-data-save-type-dropdown", "value"),
        State(f"{prefix}-motl-extra-data-store", "data"),
        State(f"{prefix}-relion-optics-store", "data"),
        State(f"{prefix}-relion5-tomos-store", "data"),
        State(f"{prefix}-save-motl-relion-binning", "value"),
        State(f"{prefix}-save-motl-relion-pixelsize", "value"),
        State(f"{prefix}-save-motl-relion-version-dropdown", "value"),
        store_states,
        prevent_initial_call=True,
    )
    def save_data(
        n_clicks,
        data_type_value,
        file_path,
        column_id,
        class_filter,
        checklist_options,
        data_to_save,
        motl_type,
        extra_df_data,
        relion_optics,
        relion_tomos,
        relion_binning,
        relion_pixel_size,
        relion_version,
        *store_states,
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

        status = save_motl(
            file_path=file_path,
            data_to_save=motl_df,
            motl_type=motl_type,
            extra_df=extra_df_data,
            rln_optics=relion_optics,
            rln_tomos=relion_tomos,
            rln_binning=relion_binning,
            rln_pixel_size=relion_pixel_size,
            rln_version=relion_version,
        )

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
                            options=motl_types,
                            multi=False,
                            value="emmotl",
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
                    options=["Version 3.0", "Version 3.1", "Version 4.x", "Version 5.x"],
                ),
                style={"marginTop": "1rem"},
                className="hidden",
                id=f"{prefix}-motl-relion-version",
            ),
            dbc.Row(
                [
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
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Upload(
                                    id=f"{prefix}-tomos-upload",
                                    children=dbc.Col(
                                        dbc.Button(
                                            f"Upload {prefix} tomogram file (if any)",
                                            color="light",
                                        ),
                                        width=12,
                                        className="d-grid gap-1 col-6 mx-auto mt-3 mb-3",
                                    ),
                                    multiple=False,
                                ),
                                width=12,
                            ),
                            dbc.Col(html.Div(id=f"{prefix}-tomo-load-status"), width=12),
                        ],
                        style={"display": "none"},
                        id=f"{prefix}-motl-relion-tomos-display",
                    ),
                ],
                style={"marginTop": "1rem"},
                className="hidden",
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
        Output(f"{prefix}-relion5-tomos-store", "data"),
        Output(f"{prefix}-tomo-load-status", "children"),
        Input(f"{prefix}-tomos-upload", "contents"),
        State(f"{prefix}-tomos-upload", "filename"),
        prevent_initial_call=True,
    )
    def load_motl_file(contents, file_name):
        if contents is None:
            return no_update

        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        return decoded.decode("utf-8"), f"Tomograms loaded: {file_name}"

    @callback(
        Output(f"{prefix}-motl-data-store", "data", allow_duplicate=True),
        Output(f"{prefix}-motl-extra-data-store", "data", allow_duplicate=True),
        Output(f"{prefix}-relion-optics-store", "data", allow_duplicate=True),
        Output(f"{prefix}-relion5-tomos-store", "data", allow_duplicate=True),
        Input(f"{prefix}-motl-upload", "contents"),
        State(f"{prefix}-motl-upload", "filename"),
        State(f"{prefix}-motl-dropdown", "value"),
        State(f"{prefix}-motl-relion-version-dropdown", "value"),
        State(f"{prefix}-motl-relion-pixelsize", "value"),
        State(f"{prefix}-motl-relion-binning", "value"),
        State(f"{prefix}-relion5-tomos-store", "data"),
        prevent_initial_call=True,
    )
    def load_motl(upload_content, filename, motl_type, rln_version, rln_pixelsize, rln_binning, rln_tomos):

        _, content_string = upload_content.split(",")
        decoded = base64.b64decode(content_string)

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[-1]) as tmp_file:
            tmp_file.write(decoded)
            tmp_file_path = tmp_file.name
            tmp_file.close()

        extra_data = None
        relion_optics = None
        relion_tomos = None

        if motl_type != "relion":

            motl = Motl.load(tmp_file_path, motl_type)

            if motl_type == "stopgap":
                extra_data = motl.stopgap_df.to_dict("records")
            elif motl_type == "dynamo":
                extra_data = motl.dynamo_df.to_dict("records")
        else:
            if rln_version == "Version 3.0":
                rln_kwargs = {"version": 3.0}
            elif rln_version == "Version 3.1":
                rln_kwargs = {"version": 3.1}
            elif rln_version == "Version 4.x":
                rln_kwargs = {"version": 4.0, "pixel_size": rln_pixelsize, "binning": rln_binning}
            else:

                rln_kwargs = {"pixel_size": rln_pixelsize, "binning": rln_binning}
                motl_type = "relion5"

                if rln_tomos:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(filename)[-1], mode="w", encoding="utf-8", newline="\n"
                    ) as tomo_tmp_file:
                        tomo_tmp_file.write(rln_tomos)
                        tomo_tmp_file_path = tomo_tmp_file.name
                        tomo_tmp_file.close()  # necessary on windows
                        rln_kwargs["input_tomograms"] = tomo_tmp_file_path

            motl = Motl.load(tmp_file_path, motl_type, **rln_kwargs)
            extra_data = motl.relion_df.to_dict("records")
            relion_optics = motl.optics_data.to_dict("records")

            if motl_type == "relion5":
                relion_tomos = motl.tomo_df.to_dict("records")
                if rln_tomos:
                    os.remove(tomo_tmp_file_path)

        os.remove(tmp_file_path)

        global tomo_ids
        tomo_ids = motl.get_unique_values("tomo_id")

        file_status = f"Loaded {filename};   " + get_print_out(motl)
        table_data = motl.df.to_dict("records")

        return table_data, extra_data, relion_optics, relion_tomos

    @callback(
        Output(f"{prefix}-motl-relion-version-dropdown", "value", allow_duplicate=True),
        Output(f"{prefix}-motl-relion-version", "className", allow_duplicate=True),
        Output(f"{prefix}-motl-relion-options", "className", allow_duplicate=True),
        Input(f"{prefix}-motl-dropdown", "value"),
        prevent_initial_call=True,
    )
    def display_options_motl_type(motl_type):

        if motl_type == "relion":
            return "Version 3.0", "flex", "hidden"
        else:
            return "Version 3.0", "hidden", "hidden"

    @callback(
        Output(f"{prefix}-motl-relion-options", "className", allow_duplicate=True),
        Output(f"{prefix}-motl-relion-tomos-display", "className", allow_duplicate=True),
        Input(f"{prefix}-motl-relion-version-dropdown", "value"),
        prevent_initial_call=True,
    )
    def display_options_relion_version(relion_version):

        if relion_version == "Version 3.0" or relion_version == "Version 3.1":
            return "hidden", "hidden"
        elif relion_version == "Version 4.x":
            return "flex", "hidden"
        else:
            return "flex", "flex"
