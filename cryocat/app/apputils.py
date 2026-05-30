# Dispatch convention for cryocat operations:
#
#   Preview callbacks (live-recompute, fires on every form-value change, updates a plot):
#       Call the cryocat function directly.  NOT through run_operation.
#       These callbacks run many times per second; logging each call floods the
#       session script with duplicate lines.  Only the final "commit" call matters.
#
#   Commit callbacks (Create, Save, Apply, Add to pool — one explicit user action):
#       Route through run_operation (or run_operation_to_pool for Motl outputs).
#       This is the single chokepoint: it writes one runnable line to the session
#       script, logs ▶/✓/✗ to the log pane, and records result Motl sources.
#
# Never log plot calls or diagnostic queries.

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

from cryocat.utils.classutils import TYPE_HANDLERS
from cryocat.core.cryomotl import Motl, EmMotl, StopgapMotl, RelionMotl, RelionMotlv5, DynamoMotl

# mport dash_html_components as html


# def series_to_dash_list(series):
#    return html.Ul([html.Li(f"{idx}: {val}") for idx, val in series.items()])


def save_motl(
    file_path,
    data_to_save,
    motl_type,
    extra_df=None,
    rln_optics=None,
    rln_tomos=None,
    rln_binning=1,
    rln_pixel_size=1.0,
    rln_tomo_format="",
    rln_subtomo_format="",
    rln_version=3.1,
    rln_use_original=False,
):

    if not isinstance(data_to_save, pd.DataFrame):
        df = pd.DataFrame(data_to_save)
    else:
        df = data_to_save

    status = f"File saved to {file_path} as {motl_type}."

    if motl_type == "emmotl":
        m = EmMotl(df)
        m.write_out(file_path)
    elif motl_type == "stopgap":
        m = StopgapMotl(df)
        if extra_df:
            m.stopgap_df = pd.DataFrame(extra_df)
        m.write_out(file_path)
    elif motl_type == "dynamo":
        m = DynamoMotl(df)
        if extra_df:
            m.dynamo_df = pd.DataFrame(extra_df)
        m.write_out(file_path)
    elif motl_type == "relion":
        if rln_optics:
            optics_data = pd.DataFrame(rln_optics)
            write_optics = True
        else:
            optics_data = None
            write_optics = False

        if rln_version == 5.0:
            if rln_tomos:
                input_tomograms = pd.DataFrame(rln_tomos)
            else:
                return "Tomogram data needs to be provided for Relion 5.0 file type."
            m = RelionMotlv5(
                df,
                input_tomograms=input_tomograms,
                pixel_size=rln_pixel_size,
                binning=rln_binning,
                optics_data=optics_data,
            )
        elif rln_version == 5.1:
            m = RelionMotl(
                df, version=5.1, pixel_size=rln_pixel_size, binning=rln_binning, optics_data=optics_data
            )
        else:
            m = RelionMotl(
                df, version=rln_version, pixel_size=rln_pixel_size, binning=rln_binning, optics_data=optics_data
            )
        m.relion_df = pd.DataFrame(extra_df)

        m.write_out(
            file_path,
            write_optics=write_optics,
            use_original_entries=rln_use_original,
            optics_data=optics_data,
            tomo_format=rln_tomo_format or "",
            subtomo_format=rln_subtomo_format or "",
        )

    return status


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


def get_class_by_name(class_name: str, module_path="cryocat.analysis.tango"):
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


def generate_kwargs(ids, values):
    """Round-trip GUI form values to a kwargs dict via the central type table.

    Each control id carries the resolved handler ``tag`` (set by
    :func:`cryocat.app.formgen.build_form`); the matching ``TYPE_HANDLERS``
    entry's ``parse`` turns the widget value into the python value. Render and
    parse therefore read the *same* table and cannot drift.
    """
    out = {}
    for id_, value in zip(ids, values):
        tag = id_["tag"]
        parse = TYPE_HANDLERS[tag]["parse"]
        out[id_["param"]] = parse(value, id_.get("choices")) if tag == "Literal" else parse(value)
    return out


def _scalar(v):
    if v is None:
        return None
    try:
        if hasattr(v, "__len__"):
            v = v[0] if len(v) > 0 else None
        return float(v) if v is not None else None
    except (TypeError, ValueError, IndexError):
        return None


def _format_relion_params(params):
    if not params:
        return ""
    parts = [f"Relion {params.get('version', '')}"]
    ps = _scalar(params.get("pixel_size"))
    bn = _scalar(params.get("binning"))
    if ps is not None:
        parts.append(f"pixel size: {ps:.4g} Å")
    if bn is not None:
        parts.append(f"binning: {bn:.4g}")
    if params.get("tomo_format"):
        parts.append(f"tomo format: {params['tomo_format']}")
    if params.get("subtomo_format"):
        parts.append(f"subtomo format: {params['subtomo_format']}")
    return "  |  ".join(parts)


def get_relevant_features(desc_name, all_features):

    avail_features = [s for s in all_features if not s.endswith(desc_name)]

    return avail_features


def iter_standalone_builders() -> list[dict]:
    """Return descriptor dicts for every ``@gui_exposed`` builder with ``standalone=True``.

    Each entry is a dict with keys ``id`` (function name), ``label``, ``fn``
    (the callable), and ``preview`` (plot style string or None).

    The registry is populated at decoration time by :data:`cryocat.utils.classutils
    ._GUI_BUILDER_REGISTRY`. Importing the builder modules here ensures their
    decorators fire before the registry is read.
    """
    import cryocat.utils.geom  # noqa: ensure @gui_exposed decorators run
    from cryocat.utils.classutils import _GUI_BUILDER_REGISTRY
    return list(_GUI_BUILDER_REGISTRY)


def _iter_gui_methods():
    """Yield ``(name, gui_dict)`` for every ``@gui_exposed`` method on ``Motl``.

    Walks instance methods *and* class/static methods. Class/staticmethods are
    bound to ``Motl`` when accessed via the class, so ``inspect.ismethod``
    catches them and ``__func__`` exposes the underlying function where
    ``_gui`` is stored (see :func:`cryocat.utils.classutils.gui_exposed`).
    """
    from cryocat.core.cryomotl import Motl

    members = inspect.getmembers(
        Motl,
        predicate=lambda o: inspect.isfunction(o) or inspect.ismethod(o),
    )
    for name, fn in members:
        target = getattr(fn, "__func__", fn)
        gui = getattr(target, "_gui", None)
        if gui is not None:
            yield name, gui


def get_single_motl_methods():
    """Dropdown options for the editor's *single-motl* operation menu.

    A method is single-motl iff its ``_gui["motls"]`` is None (the default for
    ``@gui_exposed``). Multi-motl ops (those carrying a ``motls=...`` spec) are
    excluded — they live in their own section, see
    :func:`get_multi_motl_methods`.
    """
    return [
        {"label": g["label"], "value": n}
        for n, g in _iter_gui_methods()
        if not g.get("motls")
    ]


def get_multi_motl_methods():
    """Dropdown options for the editor's *multiple-motl* operation menu.

    Returns the methods marked ``motls={...}``; each option carries the spec
    so the sidebar can drive the picker layout (arity, ordering, main-first).
    """
    return [
        {"label": g["label"], "value": n, "motls": g["motls"]}
        for n, g in _iter_gui_methods()
        if g.get("motls")
    ]


# Backwards-compatible alias: the editor's single-motl dropdown used to call
# this. Multi-motl ops were not in the registry then, so the old behaviour
# matches the new single-only filter.
def get_motl_operation_methods():
    return get_single_motl_methods()


# ── Dispatch helpers ──────────────────────────────────────────────────────────

def run_operation(fn, kwargs: dict):
    """Invoke a cryocat function through the logging chokepoint.

    Use in **commit callbacks** only (Create, Save, Apply, Add to pool).
    Preview callbacks that re-run on every form change must call the function
    directly so the session script is not flooded with redundant lines.

    Writes one runnable line to the session script, logs ▶/✓/✗ to the pane,
    and records any Motl result in the source side-table.
    """
    from cryocat.app.logger import invoke_operation
    return invoke_operation(fn, kwargs)

