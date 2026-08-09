from cryocat.app.logger import dash_logger

from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from cryocat.core.cryomotl import Motl
from cryocat.app.apputils import save_motl
from cryocat.app.components.customel import InlineLabeledDropdown, InlineInputForm
from cryocat.app.components.relionopts import (
    get_relion_options,
    register_relion_options_callbacks,
)
from cryocat.app.components._motlio_ops import (
    load_motl_from_path,
    load_kwargs_from_store,
    save_kwargs_from_store,
    filter_by_class,
)
from cryocat.app.components.pathfield import get_path_field
from cryocat.app.formgen import make_dropdown
import pandas as pd


motl_types = [
    {"label": "EM", "value": "emmotl"},
    {"label": "STOPGAP", "value": "stopgap"},
    {"label": "Relion", "value": "relion"},
    {"label": "Dynamo", "value": "dynamo"},
]


# ── Shared save-modal pieces ──────────────────────────────────────────────────


def get_class_selector(prefix: str) -> html.Div:
    """Data-source, column, class filter, and checklist — full-save only."""
    return html.Div([
        InlineLabeledDropdown(
            id_=f"{prefix}-datasave-dropdown",
            label="Data to save:",
            multi=False,
            placeholder="Select data to save",
        ),
        InlineLabeledDropdown(
            id_=f"{prefix}-assignment-dropdown",
            label="Store in the column:",
            tooltip_text=(
                "Choose a column to store the classification info. "
                "Use 'class' if you choose Dynamo, Relion, or Stopgap as an output."
            ),
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
            style={"width": "100%", "padding": "0", "marginBottom": "0.5rem"},
        ),
    ])


def get_save_modal(
    prefix: str,
    *,
    title: str = "Save motl",
    prepend_body: list | None = None,
) -> dbc.Modal:
    """Type dropdown, Relion options, filename input, and footer confirm."""
    body = list(prepend_body or []) + [
        InlineLabeledDropdown(
            id_=f"{prefix}-data-save-type-dropdown",
            options=motl_types,
            label="Output type:",
            multi=False,
            placeholder="Output type",
        ),
        get_relion_options(prefix, for_load=False),
        InlineInputForm(
            id_=f"{prefix}-save-path-input",
            label="Filename:",
            type="text",
            placeholder="Filename (including its path)",
        ),
    ]
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(title)),
            dbc.ModalBody(html.Div(body)),
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
                        dbc.Button(
                            "Save",
                            id=f"{prefix}-save-output-file",
                            className="ms-auto",
                            n_clicks=0,
                        ),
                    ],
                )
            ),
        ],
        id=f"{prefix}-save-output-modal",
        is_open=False,
    )


# ── Public save components ────────────────────────────────────────────────────


def get_motl_save_component(prefix: str) -> html.Div:
    """Full save: class/column selector + type + Relion opts + filename."""
    return html.Div([
        html.Div(
            dbc.Button(
                "Save output",
                id=f"{prefix}-save-output-btn",
                color="light",
                style={"width": "100%"},
            ),
            style={"width": "100%"},
        ),
        get_save_modal(
            prefix,
            title="Save output",
            prepend_body=[get_class_selector(prefix)],
        ),
    ])


# ── Load component ────────────────────────────────────────────────────────────


def get_motl_load_component(prefix: str, display_option="block") -> html.Div:
    return html.Div(
        id=f"{prefix}-motl-container",
        style={"marginTop": "1rem", "display": display_option},
        children=[
            dbc.Row([
                dbc.Col(
                    html.Div("Motl type: ", style={"fontStyle": "bold"}),
                    width=4,
                    className="d-flex align-items-center",
                ),
                dbc.Col(
                    make_dropdown(
                        f"{prefix}-motl-dropdown",
                        motl_types,
                        "emmotl",
                        style={"padding": "0"},
                    ),
                    width=8,
                ),
            ]),
            get_relion_options(prefix, for_load=True),
            html.Div(
                get_path_field(
                    f"{prefix}-motl-path",
                    mode="open",
                    kind="motl",
                    extensions=(".em", ".star", ".csv", ".tbl"),
                    placeholder="Path to motl file",
                ),
                style={"marginTop": "0.5rem"},
            ),
            dbc.Button(
                "Load",
                id=f"{prefix}-motl-load-btn",
                color="primary",
                style={"width": "100%", "marginTop": "0.4rem"},
            ),
        ],
    )


# ── Save callbacks — full save (with class filtering) ─────────────────────────


def register_motl_save_callbacks(
    app, prefix: str, stored_outputs, connected_store_id, connected_input_motl_prefix
):
    store_states = [State(sid, "data") for sid in stored_outputs.values()]

    register_relion_options_callbacks(
        app,
        prefix,
        for_load=False,
        type_input_id=f"{prefix}-data-save-type-dropdown",
        connected_motl_prefix=connected_input_motl_prefix,
    )

    @app.callback(
        Output(f"{prefix}-datasave-dropdown", "options", allow_duplicate=True),
        Output(f"{prefix}-save-output-modal", "is_open", allow_duplicate=True),
        Input(f"{prefix}-save-output-btn", "n_clicks"),
        store_states,
        prevent_initial_call=True,
    )
    def generate_data_options(n_clicks, *s_states):
        existing = [k for k, v in zip(stored_outputs.keys(), s_states) if v is not None]
        return existing, True

    @app.callback(
        Output(f"{prefix}-classes-checklist", "value"),
        Input(f"{prefix}-data-save-dropdown", "value"),
        prevent_initial_call=True,
    )
    def reset_checklist_value(_):
        return []

    @app.callback(
        Output(f"{prefix}-classes-checklist", "options"),
        Input(f"{prefix}-data-save-dropdown", "value"),
        State(f"{prefix}-datasave-dropdown", "value"),
        store_states,
        prevent_initial_call=True,
    )
    def generate_classes(class_value, data_type_value, *s_states):
        if class_value == "All":
            return ["Drop unassigned entries"]
        if class_value == "Specific classes":
            idx = list(stored_outputs.keys()).index(data_type_value)
            df = pd.DataFrame(s_states[idx])
            return [str(v) for v in sorted(df["class"].unique())]
        return []

    @app.callback(
        Output(f"{prefix}-status-label", "children", allow_duplicate=True),
        Input(f"{prefix}-save-output-file", "n_clicks"),
        State(f"{prefix}-datasave-dropdown", "value"),
        State(f"{prefix}-save-path-input", "value"),
        State(f"{prefix}-assignment-dropdown", "value"),
        State(f"{prefix}-classes-checklist", "value"),
        State(f"{prefix}-classes-checklist", "options"),
        State(connected_store_id, "data"),
        State(f"{prefix}-data-save-type-dropdown", "value"),
        State(f"{prefix}-rln-value", "data"),
        State(f"{connected_input_motl_prefix}-rln-tomos-store", "data"),
        State(f"{connected_input_motl_prefix}-motl-extra-data-store", "data"),
        State(f"{connected_input_motl_prefix}-relion-optics-store", "data"),
        store_states,
        prevent_initial_call=True,
    )
    def save_data(
        n_clicks, data_type_value, path, column_id, class_filter, checklist_options,
        data_to_save, motl_type, rln_state, rln_tomos_orig, extra_df, rln_optics, *s_states,
    ):
        if isinstance(extra_df, str):
            from cryocat.app.pool import get_extra as _get_extra
            extra_df = _get_extra(extra_df)
        idx = list(stored_outputs.keys()).index(data_type_value)
        results_df = pd.DataFrame(s_states[idx])
        motl_df = filter_by_class(
            pd.DataFrame(data_to_save), column_id, class_filter, checklist_options, results_df
        )
        if motl_df is None:
            return no_update
        kwargs = save_kwargs_from_store(rln_state, rln_tomos_orig)
        return save_motl(file_path=path, data_to_save=motl_df, motl_type=motl_type,
                         extra_df=extra_df, rln_optics=rln_optics, **kwargs)


# ── Load callbacks ────────────────────────────────────────────────────────────


def register_motl_load_callbacks(app, prefix: str):
    import dash
    from dash import ALL as _ALL

    register_relion_options_callbacks(
        app,
        prefix,
        for_load=True,
        type_input_id=f"{prefix}-motl-dropdown",
    )

    @app.callback(
        Output(f"{prefix}-motl-data-store", "data", allow_duplicate=True),
        Output(f"{prefix}-motl-extra-data-store", "data", allow_duplicate=True),
        Output(f"{prefix}-relion-optics-store", "data", allow_duplicate=True),
        Output(f"{prefix}-rln-tomos-store", "data", allow_duplicate=True),
        Output(f"{prefix}-motl-data-type", "data"),
        Output(f"{prefix}-relion-params-store", "data"),
        Input(f"{prefix}-motl-load-btn", "n_clicks"),
        State({"type": "path-input", "owner": f"{prefix}-motl-path"}, "value"),
        State(f"{prefix}-motl-dropdown", "value"),
        State(f"{prefix}-rln-value", "data"),
        State(f"{prefix}-rln-tomos-store", "data"),
        prevent_initial_call=True,
    )
    def load_motl(n_clicks, path, motl_type, rln_value, rln_tomos):
        if not n_clicks or not path:
            raise dash.exceptions.PreventUpdate
        kwargs = load_kwargs_from_store(rln_value)
        return load_motl_from_path(path, motl_type, rln_tomos=rln_tomos, **kwargs)
