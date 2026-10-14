"""Deployment safety assertions and startup regression guards.

Covers:
- server.py binds 127.0.0.1, not 0.0.0.0.
- No gunicorn reference in server.py.
- run_simple() host and use_reloader arguments.
- session.start_session() default path is inside HOME.
- Suite app import time is under 10 seconds.
- build_form is not called during import (currently xfail).
- Non-pool resolver returns real rows from the grid.
- Grid callback registration accepts a non-pool resolver.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

_SERVER_PY = (
    pathlib.Path(__file__).parent.parent.parent / "cryocat" / "app" / "server.py"
)


def _server_src() -> str:
    return _SERVER_PY.read_text(encoding="utf-8")


def _server_tree() -> ast.Module:
    return ast.parse(_server_src())


def _find_run_simple_args() -> dict[str, str]:
    tree = _server_tree()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_simple"
        ):
            args = node.args
            kws = {kw.arg: kw.value for kw in node.keywords}
            return {
                "host": ast.unparse(args[0]) if args else None,
                "port": ast.unparse(args[1]) if len(args) > 1 else None,
                "use_reloader": ast.unparse(kws["use_reloader"]) if "use_reloader" in kws else None,
                "use_debugger": ast.unparse(kws["use_debugger"]) if "use_debugger" in kws else None,
            }
    return {}


# ===========================================================================
# Bind address
# ===========================================================================

def test_server_binds_localhost_not_all_interfaces():
    """run_simple() in server.py must pass '127.0.0.1', never '0.0.0.0'."""
    args = _find_run_simple_args()
    host = args.get("host")
    assert host == "'127.0.0.1'", (
        f"run_simple host is {host!r} — must be '127.0.0.1'. "
        "Binding 0.0.0.0 exposes the GUI console to the network."
    )


def test_server_no_gunicorn_reference():
    """server.py must not document or reference gunicorn.

    The GUI is single-process; gunicorn spins up multiple workers that
    share no callback state, causing race conditions on the pool and session.
    """
    assert "gunicorn" not in _server_src()


def test_run_simple_host_is_127():
    args = _find_run_simple_args()
    assert args.get("host") == "'127.0.0.1'"


def test_run_simple_reloader_disabled():
    """run_simple() must have use_reloader=False (prevents double-import of singletons)."""
    tree = _server_tree()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_simple"
        ):
            for kw in node.keywords:
                if kw.arg == "use_reloader":
                    val = ast.unparse(kw.value)
                    assert val == "False"
                    return
    pytest.skip("use_reloader kwarg not found in run_simple call")


def test_start_session_defaults_inside_home(tmp_path, monkeypatch):
    """session.start_session() with no args creates a directory under HOME."""
    import os
    import cryocat.app.session as _sess

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    log_path = _sess.start_session()
    assert log_path is not None
    assert str(tmp_path) in str(log_path) or log_path.is_relative_to(tmp_path)
    assert log_path.parent.exists()


# ===========================================================================
# Startup regression guards
# ===========================================================================

def test_app_imports_quickly():
    """Importing the suite app must take under 10 seconds."""
    import time

    code = "import cryocat.app.suite.app"
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        timeout=60,
    )
    elapsed = time.perf_counter() - t0

    assert result.returncode == 0, (
        f"Suite app import failed:\n{result.stderr.decode()}"
    )
    assert elapsed < 10.0, (
        f"Suite app import took {elapsed:.1f}s — limit is 10s. "
        "Check for heavy work at module level."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Pages currently call build_form() in module-level layout variables. "
        "Fix: move build_form() into callback-populated containers so forms are "
        "deferred to callback time. Remove xfail when that is done."
    ),
)
def test_forms_are_not_built_at_import():
    """build_form is not called during suite app import."""
    code = """
import cryocat.app.formgen as formgen

_original = formgen.build_form
_count = 0

def _patched(*args, **kwargs):
    global _count
    _count += 1
    return _original(*args, **kwargs)

formgen.build_form = _patched

import cryocat.app.suite.app  # noqa: F401

import os
os.write(1, f"__count__={_count}\\n".encode())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Subprocess failed:\n{result.stderr.decode()}"
    )
    stdout = result.stdout.decode()
    marker_line = next((l for l in stdout.splitlines() if l.startswith("__count__=")), None)
    assert marker_line is not None, f"Marker not found in stdout:\n{stdout}"
    count = int(marker_line.split("=", 1)[1])
    assert count == 0, (
        f"build_form was called {count} times during suite app import. "
        "Form building must be deferred to callback time."
    )


def test_grid_resolves_non_pool_tables():
    """A grid whose rows come from a non-pool source returns real rows."""
    from cryocat.app.components.tablegrid import rows_response

    df = pd.DataFrame({
        "x":     [1.0, 2.0, 3.0],
        "score": [0.9, 0.7, 0.5],
        "label": ["a", "b", "c"],
    })

    def my_resolve_df(ref):
        return df

    def my_resolve_n_rows(ref):
        return len(df)

    ref = {"source": "custom-registry", "key": "my-twist-key"}

    resolved_df = my_resolve_df(ref)
    n_rows = my_resolve_n_rows(ref)

    request = {
        "startRow": 0,
        "endRow": 100,
        "filterModel": {},
        "sortModel": [],
    }
    result = rows_response(request, resolved_df, n_rows)

    assert result["rowCount"] == 3
    assert len(result["rowData"]) == 3
    assert result["rowData"][0]["x"] == pytest.approx(1.0)
    assert result["rowData"][2]["label"] == "c"


def test_rows_callback_registered_with_non_pool_resolver():
    """register_tablegrid_callbacks accepts a non-pool resolver without error."""
    from dash import Dash, html
    from cryocat.app.components.tablegrid import register_tablegrid_callbacks

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div(id="nr-grid-container")

    df = pd.DataFrame({"val": [10, 20, 30]})

    register_tablegrid_callbacks(
        app, "nr",
        resolve_df=lambda ref: df,
        resolve_n_rows=lambda ref: len(df),
    )

    rows_output = "nr-grid.getRowsResponse"
    cb_entry = next(
        (c for c in app._callback_list if c.get("output") == rows_output),
        None,
    )
    assert cb_entry is not None
    assert not cb_entry.get("prevent_initial_call")
