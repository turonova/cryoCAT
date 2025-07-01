from cryocat.app.logger import dash_logger, print_dash

import numpy as np
import pandas as pd
import pickle
import io
import sys
import inspect
import importlib
import pandas.api.types as ptypes
from dash.dash_table.Format import Format, Scheme
from dash import html, dcc
import dash_bootstrap_components as dbc

import plotly.graph_objects as go

from cryocat.classutils import process_method_docstring
from cryocat.cryomotl import Motl

# mport dash_html_components as html


# def series_to_dash_list(series):
#    return html.Ul([html.Li(f"{idx}: {val}") for idx, val in series.items()])


def save_output(file_path, data_to_save, csv_only=True):

    if not isinstance(data_to_save, pd.DataFrame):
        df = pd.DataFrame(data_to_save)
    else:
        df = data_to_save

    status = f"File saved to {file_path}"

    if file_path.endswith(".csv"):
        df.to_csv(file_path, index=False)
    elif file_path.endswith(".pkl"):
        with open(file_path, "wb") as f:
            pickle.dump(data_to_save, f)
    elif csv_only:
        print_dash("The table can be saved only to a csv file.")
        status = "The table can be saved only to a csv file."
    elif file_path.endswith(".em"):
        m = Motl(df)
        m.write_out(file_path)
    else:
        print_dash("Currently only csv, pkl, and em formats are supported")
        status = "Currently only csv, pkl, and em formats are supported"

    return status


def get_class_by_name(class_name: str, module_path="cryocat.tango"):
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ValueError(f"Class '{class_name}' not found in {module_path}")
    return cls


def make_axis_trace(fig, origin=None, length=1, colors=None):

    if not origin:
        origin = np.array([0, 0, 0])

    # Define endpoints of x, y, z axes
    x_axis = origin + np.array([length, 0, 0])
    y_axis = origin + np.array([0, length, 0])
    z_axis = origin + np.array([0, 0, length])

    def make_axis_trace(start, end, color):
        return go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines",
            line=dict(color=color, width=6),
            showlegend=False,
        )

    if colors is None:
        colors = ["#AEC684", "#59AFAF", "#865B96"]

    fig.add_trace(make_axis_trace(origin, x_axis, colors[0]))
    fig.add_trace(make_axis_trace(origin, y_axis, colors[1]))
    fig.add_trace(make_axis_trace(origin, z_axis, colors[2]))


def format_columns(df):
    columns = [
        {
            "name": col,
            "id": col,
            "editable": False,
            **(
                {
                    "type": "numeric",
                    "format": Format(precision=3, scheme=Scheme.fixed),
                }
                if ptypes.is_float_dtype(df[col])
                else {}
            ),
        }
        for col in df.columns
    ]

    return columns


def get_print_out(to_print):

    return ""
    # buffer = io.StringIO()
    # sys.stdout = buffer
    # print(to_print)
    # sys.stdout = sys.__stdout__
    # single_line_output = buffer.getvalue().replace("\n", ";   ").strip()

    # return single_line_output


def generate_form_from_untyped_init(cls_name, exclude_params=None):

    cls = get_class_by_name(cls_name)
    sig = inspect.signature(cls.__init__)
    inputs = []

    if exclude_params is not None:
        exclude_params.append("self")
    else:
        exclude_params = ["self"]

    for name, param in sig.parameters.items():
        if name in exclude_params:
            continue

        default = param.default if param.default != inspect.Parameter.empty else ""

        print(name, param)
        inputs.append({"name": name, "param": param, "default": default})

    return inputs


def generate_form_from_docstring(cls_name, id_type, id_name, exclude_params=None, module_path="cryocat.tango"):

    def correct_form(value):

        if isinstance(value.get("types"), type):
            value["types"] = [value["types"].__name__]

        form_id = {"type": id_type, "cls_name": id_name, "param": value["name"], "p_type": value["types"][0]}
        if len(value["options"]) > 0:
            value_form = (
                dcc.Dropdown(
                    id=form_id,
                    options=value["options"],
                    multi=False,
                    value=value["options"][0],
                    style={
                        "width": "100%",
                    },
                ),
            )
        elif value["types"][0] == "bool":
            value_form = (
                dcc.Dropdown(
                    id=form_id,
                    options=["True", "False"],
                    multi=False,
                    value=str(value["default"]),
                    style={"width": "100%"},
                ),
            )
        elif value["types"][0] in ["float", "int"]:
            value_form = (
                dcc.Input(
                    type="number",
                    value=value["default"],
                    id=form_id,
                    style={"width": "100%"},
                ),
            )
        elif value["types"][0] == "numpy.ndarray":
            value_form = (
                dcc.Input(
                    type="text",
                    value="0,0,1",
                    id=form_id,
                    style={"width": "100%"},
                ),
            )
        else:
            value_form = (
                dcc.Input(
                    type="text",
                    value=value["default"] if value["default"] is not None else None,
                    placeholder="Optional" if not value["required"] and value["default"] is None else "",
                    id=form_id,
                    style={"width": "100%"},
                ),
            )

        return value_form

    cls = get_class_by_name(cls_name, module_path=module_path)
    params_dict = process_method_docstring(cls, "__init__", pretty_print=True)

    if exclude_params is None:
        exclude_params = []

    forms = []

    label_style = {
        "marginRight": "0.5rem",
        "width": "40%",
        "display": "flex",
        "alignItems": "center",
        "boxSizing": "border-box",
    }
    param_style = {
        "marginRight": "0.0rem",
        "marginBottom": "0.2rem",
        "width": "60%",
    }

    for key, value in params_dict.items():
        if value["name"] in exclude_params:
            continue
        else:
            label_id = f"{cls_name}-{value['name']}"
            form = html.Div(
                [
                    html.Div(
                        [
                            html.Label(key, id=label_id),
                            dbc.Tooltip(value["desc"], target=label_id, placement="top"),
                        ],
                        style=label_style,
                    ),
                    html.Div(
                        correct_form(value),
                        style=param_style,
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "row",
                    "boxSizing": "border-box",
                    "width": "100%",
                    "marginBottom": "0.2rem",
                },
            )
            forms.append(form)

    return forms


def generate_kwargs(ids_dict, values):

    def parse_array_string(s):
        # Remove brackets if present and strip spaces
        s_clean = s.strip().lstrip("[").rstrip("]")
        # Split by comma and convert to float or int
        elements = [float(x) if "." in x else int(x) for x in s_clean.split(",") if x.strip()]
        return np.array(elements)

    new_kwargs = {}

    for id, value in zip(ids_dict, values):

        if id["p_type"] == "numpy.ndarray":
            new_kwargs[id["param"]] = parse_array_string(value)

        elif value == 'True':
            new_kwargs[id["param"]] = True

        elif value == 'False':
            new_kwargs[id["param"]] = False

        else:
            new_kwargs[id["param"]] = value

    return new_kwargs


def get_relevant_features(desc_name, all_features):

    avail_features = [s for s in all_features if not s.endswith(desc_name)]

    return avail_features
