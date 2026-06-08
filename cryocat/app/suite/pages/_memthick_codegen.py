"""Pure code generator for the Memthick tab — no Dash, no IO.

The Memthick tab is a *configure → generate code → run elsewhere* workflow:
the user picks parameters in the form, then hands a ``.py`` / ``.ipynb`` /
SLURM script off to a cluster login node. This module is the seam where form
state becomes runnable code.

Keeping it Dash-free makes it trivial to unit-test (string snapshots) and
means the generated code does not depend on importing any of ``cryocat.app``.

The notebook structure mirrors ``docs/source/tutorials/membrane_thickness/
memthick_run.ipynb``: title → imports → path variables → membrane labels →
analyzer → run.

Public API
----------
* :func:`render_pipeline_py(kwargs)`        -> ``str``
* :func:`render_pipeline_ipynb(kwargs)`     -> ``dict`` (nb v4)
* :func:`render_pipeline_ipynb_json(kwargs)`-> ``str`` (JSON-encoded notebook)
* :func:`render_slurm_wrapper(py_filename, cluster_params, module_loads)`
  -> ``str``

Backwards-compatible aliases ``render_py`` / ``render_ipynb`` /
``render_ipynb_json`` / ``wrap_slurm`` are also exported.
"""
from __future__ import annotations

import io
import json
from typing import Any


_PIPELINE_FN = "run_full_pipeline"
_ANALYZER_CLS = "IntensityProfileAnalyzer"

# Path params lifted to named variables at the top of the script so the
# user can edit cluster paths in one obvious place.
_PATH_PARAMS: dict[str, str] = {
    "segmentation_map": "segmentation_path",
    "output_path": "output_path",
    "tomogram_map": "tomogram_path",
}


# ── kwargs splitting ────────────────────────────────────────────────────────


def _split_kwargs(kwargs: dict[str, Any]) -> tuple[dict, dict, dict, dict]:
    """Partition the assembled kwargs into rendering buckets.

    Returns
    -------
    paths : dict[var_name, value]
        Path params extracted from ``kwargs`` (in the order defined by
        :data:`_PATH_PARAMS`).
    labels : dict or None
        ``membrane_labels`` if present; rendered as a standalone dict
        literal so the user can edit it without scrolling.
    analyzer_kwargs : dict or None
        :class:`IntensityProfileAnalyzer` constructor kwargs (kept under
        ``kwargs["analyzer"]`` by the page).
    call_kwargs : dict
        Everything left -- the kwargs passed to ``run_full_pipeline``.
    """
    paths: dict[str, Any] = {}
    for kwarg_name, var_name in _PATH_PARAMS.items():
        if kwarg_name in kwargs:
            paths[var_name] = kwargs[kwarg_name]
    labels = kwargs.get("membrane_labels")
    analyzer_kwargs = kwargs.get("analyzer") if isinstance(kwargs.get("analyzer"), dict) else None
    skip = set(_PATH_PARAMS) | {"membrane_labels", "analyzer"}
    call_kwargs = {k: v for k, v in kwargs.items() if k not in skip}
    return paths, labels, analyzer_kwargs, call_kwargs


# ── value formatters ────────────────────────────────────────────────────────


class _Verbatim:
    """Marker so :func:`_format_value` emits a bare identifier (variable
    name) instead of quoting it."""

    __slots__ = ("expr",)

    def __init__(self, expr: str) -> None:
        self.expr = expr

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return self.expr


def _format_value(value: Any) -> str:
    """Return a Python source representation of ``value``.

    ``repr`` handles every leaf type we expect from the form (str, int,
    float, bool, None, tuple of numbers). :class:`_Verbatim` is used to
    inject already-defined identifiers (e.g. the path variables and the
    constructed analyzer) into the pipeline call.
    """
    if isinstance(value, _Verbatim):
        return value.expr
    if isinstance(value, dict):
        return _format_dict(value)
    return repr(value)


def _format_dict(d: dict, indent: str = "    ") -> str:
    """Render a dict literal across multiple indented lines."""
    if not d:
        return "{}"
    items = "\n".join(
        f"{indent}    {_format_value(k)}: {_format_value(v)},"
        for k, v in d.items()
    )
    return "{\n" + items + f"\n{indent}}}"


def _format_kwargs(kwargs: dict[str, Any], indent: str = "    ") -> str:
    """Multi-line ``key=value`` block for inlining into a function call."""
    if not kwargs:
        return ""
    return "\n".join(
        f"{indent}{name}={_format_value(value)}," for name, value in kwargs.items()
    )


# ── .py renderer ────────────────────────────────────────────────────────────


def render_pipeline_py(kwargs: dict[str, Any]) -> str:
    """Render a runnable ``.py`` script for the configured pipeline.

    Sections mirror the notebook structure:

    1. Module docstring + ``from cryocat.analysis import memthick``.
    2. Path variables hoisted to the top (``segmentation_path``,
       ``output_path``, ``tomogram_path`` when set).
    3. ``membrane_labels`` dict literal.
    4. ``analyzer = memthick.IntensityProfileAnalyzer(...)`` (omitted when
       the form left the analyzer at its defaults).
    5. ``results = memthick.run_full_pipeline(...)``.

    Parameters
    ----------
    kwargs : dict
        Combined form state. The page passes the pipeline kwargs as
        top-level keys plus ``kwargs["analyzer"]`` (a dict of
        :class:`IntensityProfileAnalyzer` kwargs) when the user filled the
        analyzer sub-form.
    """
    paths, labels, analyzer_kwargs, call_kwargs = _split_kwargs(kwargs)

    lines: list[str] = [
        '"""Auto-generated by the cryoCAT membrane-thickness tab. Edit freely."""',
        "from cryocat.analysis import memthick",
        "",
    ]

    # ── path variables ─────────────────────────────────────────────────────
    if paths:
        for var_name, value in paths.items():
            lines.append(f"{var_name} = {_format_value(value)}")
        lines.append("")

    # ── membrane labels ────────────────────────────────────────────────────
    if labels:
        lines.append(f"membrane_labels = {_format_dict(labels, indent='')}")
        lines.append("")

    # ── analyzer ───────────────────────────────────────────────────────────
    if analyzer_kwargs:
        lines.append(f"analyzer = memthick.{_ANALYZER_CLS}(")
        lines.append(_format_kwargs(analyzer_kwargs, indent="    "))
        lines.append(")")
        lines.append("")

    # ── pipeline call ──────────────────────────────────────────────────────
    referenced_call_kwargs: dict[str, Any] = {}
    for kwarg_name, var_name in _PATH_PARAMS.items():
        if var_name in paths:
            referenced_call_kwargs[kwarg_name] = _Verbatim(var_name)
    if labels:
        referenced_call_kwargs["membrane_labels"] = _Verbatim("membrane_labels")
    referenced_call_kwargs.update(call_kwargs)
    if analyzer_kwargs:
        referenced_call_kwargs["analyzer"] = _Verbatim("analyzer")

    lines.append(f"results = memthick.{_PIPELINE_FN}(")
    lines.append(_format_kwargs(referenced_call_kwargs, indent="    "))
    lines.append(")")
    lines.append("")
    lines.append('print("done:", results)')
    lines.append("")
    return "\n".join(lines)


# ── .ipynb renderer ─────────────────────────────────────────────────────────


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def render_pipeline_ipynb(kwargs: dict[str, Any]) -> dict:
    """Render an ``nbformat`` v4 notebook dict.

    Cells follow the source notebook
    (``memthick_run.ipynb``): title → imports → path variables →
    ``membrane_labels`` → analyzer → run.

    Returned as a plain dict so the page (or a test) can ``json.dumps`` it
    without an ``nbformat`` dependency.
    """
    paths, labels, analyzer_kwargs, call_kwargs = _split_kwargs(kwargs)

    cells: list[dict] = []
    cells.append(_markdown_cell(
        "# Running the membrane thickness estimation pipeline\n"
        "\n"
        "Auto-generated by the cryoCAT membrane-thickness tab. Edit freely.\n"
    ))
    cells.append(_code_cell("from cryocat.analysis import memthick\n"))

    if paths:
        path_lines = "\n".join(
            f"{var_name} = {_format_value(value)}"
            for var_name, value in paths.items()
        ) + "\n"
        cells.append(_code_cell(path_lines))

    if labels:
        cells.append(_code_cell(f"membrane_labels = {_format_dict(labels, indent='')}\n"))

    if analyzer_kwargs:
        cells.append(_code_cell(
            f"analyzer = memthick.{_ANALYZER_CLS}(\n"
            + _format_kwargs(analyzer_kwargs, indent="    ") + "\n"
            + ")\n"
        ))

    referenced_call_kwargs: dict[str, Any] = {}
    for kwarg_name, var_name in _PATH_PARAMS.items():
        if var_name in paths:
            referenced_call_kwargs[kwarg_name] = _Verbatim(var_name)
    if labels:
        referenced_call_kwargs["membrane_labels"] = _Verbatim("membrane_labels")
    referenced_call_kwargs.update(call_kwargs)
    if analyzer_kwargs:
        referenced_call_kwargs["analyzer"] = _Verbatim("analyzer")

    cells.append(_code_cell(
        f"results = memthick.{_PIPELINE_FN}(\n"
        + _format_kwargs(referenced_call_kwargs, indent="    ") + "\n"
        + ")\n"
        + "results\n"
    ))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render_pipeline_ipynb_json(kwargs: dict[str, Any]) -> str:
    """:func:`render_pipeline_ipynb` then ``json.dumps`` with stable indent."""
    return json.dumps(render_pipeline_ipynb(kwargs), indent=1)


# ── SLURM wrapper ───────────────────────────────────────────────────────────


def render_slurm_wrapper(
    py_filename: str,
    cluster_params: dict | None = None,
    module_loads: list[str] | None = None,
    interpreter: str = "#!/bin/bash",
) -> str:
    """Wrap a generated ``.py`` script in a SLURM submission script.

    Delegates SBATCH directive emission to
    :func:`cryocat.utils.scriptutils.process_cluster_params` so the format
    matches the rest of the codebase. The shebang, ``module load`` lines,
    and final ``python <py_filename>`` command are added around it.

    Parameters
    ----------
    py_filename : str
        Path the cluster will see for the generated python script.
    cluster_params : dict, optional
        SBATCH parameters. Keys starting with ``--`` produce
        ``#SBATCH --key=value``; keys starting with ``-`` produce
        ``#SBATCH -key value``.
    module_loads : list of str, optional
        ``module load`` directives.
    interpreter : str
        Shebang line.
    """
    from cryocat.utils.scriptutils import process_cluster_params

    buf = io.StringIO()
    buf.write(interpreter + "\n")
    if cluster_params:
        process_cluster_params(buf, cluster_params)
    for mod in module_loads or []:
        buf.write(f"module load {mod}\n")
    buf.write(f"python {py_filename}\n")
    return buf.getvalue()


# ── Backwards-compatible aliases ────────────────────────────────────────────


def render_py(kwargs: dict[str, Any], analyzer_kwargs: dict[str, Any] | None = None) -> str:
    """Compatibility shim: accepts the older ``(kwargs, analyzer_kwargs)`` split."""
    merged = dict(kwargs)
    if analyzer_kwargs:
        merged["analyzer"] = dict(analyzer_kwargs)
    return render_pipeline_py(merged)


def render_ipynb(kwargs: dict[str, Any], analyzer_kwargs: dict[str, Any] | None = None) -> dict:
    """Compatibility shim mirroring :func:`render_py`."""
    merged = dict(kwargs)
    if analyzer_kwargs:
        merged["analyzer"] = dict(analyzer_kwargs)
    return render_pipeline_ipynb(merged)


def render_ipynb_json(kwargs: dict[str, Any], analyzer_kwargs: dict[str, Any] | None = None) -> str:
    """Compatibility shim mirroring :func:`render_py`."""
    merged = dict(kwargs)
    if analyzer_kwargs:
        merged["analyzer"] = dict(analyzer_kwargs)
    return render_pipeline_ipynb_json(merged)


def wrap_slurm(
    py_path: str,
    cluster_params: dict | None = None,
    module_loads: list[str] | None = None,
    interpreter: str = "#!/bin/bash",
) -> str:
    """Compatibility shim for :func:`render_slurm_wrapper`."""
    return render_slurm_wrapper(py_path, cluster_params, module_loads, interpreter)
