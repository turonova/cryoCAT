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
    """Save a motl through the logging chokepoint so saves appear in the event stream.

    Routes each format through ``run_operation(m.save_to, ...)`` so the call
    is recorded as a library call (``cryocat.core.cryomotl``, not
    ``cryocat.app``) in the session script.  The Relion branch stamps
    ``_ctor_kwargs`` on the instance so the script also renders the
    constructor parameters (version, pixel_size, binning).
    """
    if not isinstance(data_to_save, pd.DataFrame):
        df = pd.DataFrame(data_to_save)
    else:
        df = data_to_save

    if motl_type == "emmotl":
        m = EmMotl(df)
        run_operation(m.save_to, {"output_path": file_path})
    elif motl_type == "stopgap":
        m = StopgapMotl(df)
        if extra_df:
            m.stopgap_df = pd.DataFrame(extra_df)
        run_operation(m.save_to, {"output_path": file_path})
    elif motl_type == "dynamo":
        m = DynamoMotl(df)
        if extra_df:
            m.dynamo_df = pd.DataFrame(extra_df)
        run_operation(m.save_to, {"output_path": file_path})
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
            m._ctor_kwargs = {"pixel_size": rln_pixel_size, "binning": rln_binning}
        elif rln_version == 5.1:
            m = RelionMotl(
                df, version=5.1, pixel_size=rln_pixel_size, binning=rln_binning, optics_data=optics_data
            )
            m._ctor_kwargs = {"version": 5.1, "pixel_size": rln_pixel_size, "binning": rln_binning}
        else:
            m = RelionMotl(
                df, version=rln_version, pixel_size=rln_pixel_size, binning=rln_binning, optics_data=optics_data
            )
            m._ctor_kwargs = {"version": rln_version, "pixel_size": rln_pixel_size, "binning": rln_binning}
        m.relion_df = pd.DataFrame(extra_df) if extra_df else pd.DataFrame()
        run_operation(m.save_to, {
            "output_path": file_path,
            "write_optics": write_optics,
            "use_original_entries": rln_use_original,
            "optics_data": optics_data,
            "tomo_format": rln_tomo_format or "",
            "subtomo_format": rln_subtomo_format or "",
        })

    return f"File saved to {file_path} as {motl_type}."


def save_output(file_path, data_to_save, csv_only=True):
    if not isinstance(data_to_save, pd.DataFrame):
        df = pd.DataFrame(data_to_save)
    else:
        df = data_to_save

    if file_path.endswith(".csv"):
        run_operation(df.to_csv, {"path_or_buf": file_path, "index": False})
    elif file_path.endswith(".pkl"):
        with open(file_path, "wb") as f:
            pickle.dump(data_to_save, f)
        return f"File saved to {file_path}"
    elif csv_only:
        print_dash("The table can be saved only to a csv file.")
        return "The table can be saved only to a csv file."
    elif file_path.endswith(".em"):
        m = Motl(df)
        run_operation(m.save_to, {"output_path": file_path})
    else:
        print_dash("Currently only csv, pkl, and em formats are supported")
        return "Currently only csv, pkl, and em formats are supported"

    return f"File saved to {file_path}"


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


def generate_kwargs(ids, values):
    """Round-trip GUI form values to a kwargs dict via the central type table.

    Each control id carries the resolved handler ``tag`` (set by
    :func:`cryocat.app.formgen.build_form`); the matching ``TYPE_HANDLERS``
    entry's ``parse`` turns the widget value into the python value. Render and
    parse therefore read the *same* table and cannot drift.

    Composite widgets (``tag == "Tuple"``) emit one control per slot, all
    sharing the same ``param`` and carrying a ``slot`` index in the id. They
    are collected here into a single list (sorted by slot) and handed to the
    Tuple parser, which converts it to the requested Python tuple type.
    """
    # Pass 1: bucket composite-widget slots by param; collect everything else
    # into a flat list to parse normally.
    tuple_buckets: dict = {}
    flat: list = []
    for id_, value in zip(ids, values):
        if id_.get("tag") == "Tuple":
            bucket = tuple_buckets.setdefault(id_["param"], {
                "elem": id_.get("elem", "float"),
                "slots": {},
            })
            bucket["slots"][int(id_["slot"])] = value
        else:
            flat.append((id_, value))

    out = {}
    for id_, value in flat:
        tag = id_["tag"]
        parse = TYPE_HANDLERS[tag]["parse"]
        out[id_["param"]] = parse(value, id_.get("choices")) if tag == "Literal" else parse(value)

    parse_tuple = TYPE_HANDLERS["Tuple"]["parse"]
    for param, bucket in tuple_buckets.items():
        ordered = [bucket["slots"][k] for k in sorted(bucket["slots"])]
        out[param] = parse_tuple(ordered, elem=bucket["elem"])

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


def run_operation_to_pool(
    fn,
    kwargs: dict,
    state,
    *,
    label: str | None = None,
    replaces: str | None = None,
):
    """Invoke *fn* through the chokepoint and land the result in the pool.

    This is the atomic load/apply chokepoint for operations that produce a
    Motl which must live in the pool.  Either:

    - success: pool gains (or updates) the entry **and** the event stream
      gains a ``status="ok"`` call event, **or**
    - failure: pool is untouched **and** the stream gains a
      ``status="error"`` call event; the exception is re-raised.

    Parameters
    ----------
    fn:
        The callable to invoke (bound method or free function).
    kwargs:
        Keyword arguments forwarded to *fn*.
    state:
        Current :class:`~cryocat.app.pool.PoolState`; never mutated.
    label:
        Optional human-readable label for the pool entry.
    replaces:
        ``motl_id`` of an existing pool entry to overwrite in place.
        When ``None`` a new entry is appended.

    Returns
    -------
    tuple[PoolState, str, Any]
        ``(new_state, motl_id, result)``
    """
    from cryocat.app import provenance as _prov
    from cryocat.app import session as _session
    from cryocat.app.logger import invoke_operation
    from cryocat.app.pool import insert_motl, replace_motl_rows

    # Determine target variable name before the call so the event carries it.
    if replaces is not None:
        motl_id = replaces
        var = _prov.bind(replaces)
    else:
        motl_id = f"motl-{state.next_id + 1}"
        var = _prov.bind(motl_id)

    # Run through the chokepoint.  Re-raises on exception; pool is untouched.
    result = invoke_operation(fn, kwargs, assign_to=var, pool_id=motl_id, label=label)

    # Derive row list and motl type from the returned object.
    if hasattr(result, "df"):
        rows = result.df.to_dict("records")
        motl_type = type(result).__name__
    else:
        rows = []
        motl_type = "emmotl"

    # Update pool (immutable: original state unchanged if this throws).
    if replaces is not None:
        new_state = replace_motl_rows(state, replaces, rows, label=label, motl_type=motl_type)
    else:
        new_state, motl_id = insert_motl(state, rows, label=label, motl_type=motl_type)

    # Stamp pool id on result so future invoke_operation can resolve the receiver.
    try:
        result._pool_motl_id = motl_id
    except (AttributeError, TypeError):
        pass

    # Record provenance: motl_id → seq of the call event just emitted.
    _prov.record(motl_id, _session.last_seq())

    return new_state, motl_id, result


def record_load_to_pool(
    motl_data: list[dict],
    motl_type: str,
    display_name: str,
    rln_kwargs: dict | None,
    pool_state,
    *,
    label: str | None = None,
    extra: list[dict] | None = None,
    meta: dict | None = None,
) -> tuple:
    """Insert a file-loaded motl into the pool via the logging chokepoint.

    The motl content is taken from already-deserialized ``motl_data`` (the
    upload callback already read the file — no I/O here).  A thin callable
    wrapped with :func:`functools.update_wrapper` against :meth:`Motl.load`
    makes :func:`~cryocat.app.logger.invoke_operation` render the canonical
    ``cryomotl.Motl.load(display_name, motl_type, ...)`` script line.

    Returns
    -------
    tuple[PoolState, str, Motl]
        ``(new_pool_state, motl_id, motl)``
    """
    import functools
    from cryocat.core.cryomotl import Motl
    from cryocat.app import provenance as _prov
    from cryocat.app import session as _session
    from cryocat.app.logger import invoke_operation
    from cryocat.app.pool import insert_motl

    rln_kwargs = rln_kwargs or {}
    motl = Motl(pd.DataFrame(motl_data))

    def _preloaded(input_motl, motl_type="emmotl", **kwargs):
        return motl

    functools.update_wrapper(_preloaded, Motl.load)

    motl_id = f"motl-{pool_state.next_id + 1}"
    var = _prov.bind(motl_id)

    call_kwargs: dict = {"input_motl": display_name, "motl_type": motl_type, **rln_kwargs}
    invoke_operation(_preloaded, call_kwargs, assign_to=var, pool_id=motl_id, label=label or display_name, source=display_name)

    pool_state, motl_id = insert_motl(
        pool_state,
        motl_data,
        label=label or display_name,
        motl_type=motl_type,
        extra=extra,
        meta=meta,
    )
    try:
        motl._pool_motl_id = motl_id
    except (AttributeError, TypeError):
        pass

    _prov.record(motl_id, _session.last_seq())
    return pool_state, motl_id, motl

