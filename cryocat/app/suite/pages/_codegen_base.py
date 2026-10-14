"""Shared code-generation utilities for the memthick and pana tabs.

Both tabs use a *configure → generate code → run elsewhere* workflow. This
module consolidates the formatting primitives and the SLURM wrapper so they
are defined once.

Public API
----------
* :class:`Verbatim` — marker that suppresses quoting in :func:`format_value`.
* :func:`format_value`  — repr a value, with ``Verbatim``/dict special cases.
* :func:`format_dict`   — multi-line dict literal.
* :func:`format_kwargs` — multi-line ``key=value`` block.
* :func:`render_slurm_wrapper` — wrap a ``.py`` in a SLURM submission script.
* :func:`parse_sbatch_text`    — parse ``--key=value`` / ``-k v`` SBATCH lines.
* :func:`code_cell`     — nbformat v4 code cell dict.
* :func:`markdown_cell` — nbformat v4 markdown cell dict.
* :func:`notebook`      — nbformat v4 notebook envelope.
"""
from __future__ import annotations

import io
from typing import Any


class Verbatim:
    """Marker so :func:`format_value` emits a bare identifier instead of quoting.

    Used to splice already-defined Python identifiers (path variables,
    constructed objects) into generated function calls without adding quotes.
    """

    __slots__ = ("expr",)

    def __init__(self, expr: str) -> None:
        self.expr = expr

    def __repr__(self) -> str:  # pragma: no cover
        return self.expr


def format_value(value: Any) -> str:
    """Return a Python source representation of *value*.

    * :class:`Verbatim` → bare identifier (no quotes).
    * ``dict`` → :func:`format_dict` (multi-line literal).
    * Anything else → ``repr``.
    """
    if isinstance(value, Verbatim):
        return value.expr
    if isinstance(value, dict):
        return format_dict(value)
    return repr(value)


def format_dict(d: dict, indent: str = "    ") -> str:
    """Render a dict literal across multiple indented lines."""
    if not d:
        return "{}"
    items = "\n".join(
        f"{indent}    {format_value(k)}: {format_value(v)},"
        for k, v in d.items()
    )
    return "{\n" + items + f"\n{indent}}}"


def format_kwargs(kwargs: dict[str, Any], indent: str = "    ") -> str:
    """Multi-line ``key=value`` block for inlining into a function call."""
    if not kwargs:
        return ""
    return "\n".join(
        f"{indent}{name}={format_value(value)}," for name, value in kwargs.items()
    )


def render_slurm_wrapper(
    py_filename: str,
    cluster_params: dict | None = None,
    module_loads: list[str] | None = None,
    interpreter: str = "#!/bin/bash",
) -> str:
    """Wrap a generated ``.py`` script in a SLURM submission script.

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


def code_cell(source: str) -> dict:
    """Return an nbformat v4 code cell dict."""
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    """Return an nbformat v4 markdown cell dict."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    """Wrap *cells* in an nbformat v4 notebook envelope.

    Returns a plain dict; call ``json.dumps`` to serialise.
    """
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


def parse_sbatch_text(text: str) -> dict:
    """Parse newline-separated SBATCH directive lines into a kwargs dict.

    Handles both ``--key=value`` and ``-k value`` forms. Blank lines and
    lines starting with ``#`` are silently skipped.

    Parameters
    ----------
    text : str
        Raw text from a textarea containing one directive per line.

    Returns
    -------
    dict
        Mapping of key → value strings suitable for passing to
        :func:`render_slurm_wrapper` as ``cluster_params``.
    """
    result: dict[str, Any] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
        else:
            parts = line.split(None, 1)
            result[parts[0]] = parts[1] if len(parts) > 1 else ""
    return result
