"""Shared utilities and fixtures for GUI app tests (T0+).

Utility functions (collect_ids, collect_id_paths) are plain functions, not
fixtures.  They can be imported directly:
    from tests.app.conftest import collect_ids, collect_id_paths
"""

import pytest

# ---------------------------------------------------------------------------
# Component-tree helpers
# ---------------------------------------------------------------------------


def _norm_id(id_val):
    """Normalise a Dash id to a hashable form.

    String ids are returned as-is.  Dict ids (pattern-matching) are converted
    to a sorted tuple of (key, value) pairs so they are hashable and
    comparable regardless of insertion order.
    """
    if isinstance(id_val, dict):
        return tuple(sorted(id_val.items(), key=lambda kv: kv[0]))
    return id_val


def _walk(node, ids: set) -> None:
    """Recursive DFS over a Dash component tree, collecting every id."""
    if node is None or isinstance(node, (str, int, float, bool)):
        return
    if isinstance(node, (list, tuple)):
        for child in node:
            _walk(child, ids)
        return
    # Assume it is a Dash Component.
    node_id = getattr(node, "id", None)
    if node_id is not None:
        ids.add(_norm_id(node_id))
    _walk(getattr(node, "children", None), ids)


def _walk_paths(node, path: list[str], result: dict) -> None:
    """As _walk, but records the path at which each id was found.

    result maps normalised id → list[str] of human-readable paths.
    Multiple entries for the same id mean duplicate ids in the tree.
    """
    if node is None or isinstance(node, (str, int, float, bool)):
        return
    if isinstance(node, (list, tuple)):
        for i, child in enumerate(node):
            _walk_paths(child, path + [f"[{i}]"], result)
        return
    cls = type(node).__name__
    node_id = getattr(node, "id", None)
    if node_id is not None:
        key = _norm_id(node_id)
        label = f"{cls}(id={node_id!r})"
        result.setdefault(key, []).append(" > ".join(path + [label]))
    _walk_paths(getattr(node, "children", None), path + [cls], result)


def make_motl_rows(n: int = 5) -> list[dict]:
    """Return *n* rows in the 20-column Motl format (all zeroes, subtomo_id 1..n)."""
    import numpy as np
    from cryocat.core.cryomotl import Motl
    import pandas as pd

    data = {col: np.zeros(n, dtype=float) for col in Motl.motl_columns}
    data["subtomo_id"] = np.arange(1, n + 1, dtype=float)
    data["tomo_id"] = np.ones(n, dtype=float)
    return pd.DataFrame(data).to_dict("records")


def collect_ids(component) -> set:
    """Walk a Dash component tree; return every id (normalised).

    String ids are returned as-is.  Dict ids are returned as sorted
    (key, value) tuples so they are hashable and set-comparable.
    """
    result: set = set()
    _walk(component, result)
    return result


def collect_id_paths(component) -> dict:
    """Walk a Dash component tree; return {normalised_id: [path_strings]}.

    An id that appears at more than one location has len(paths) > 1 —
    the basis of the duplicate-id check.
    """
    result: dict = {}
    _walk_paths(component, [], result)
    return result


# ---------------------------------------------------------------------------
# App fixtures  (session-scoped: both apps import exactly once per test run)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# State-cleaning fixture  (autouse — must run around every test)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_server_state():
    """Clear every server-side singleton between tests.

    Prevents order-dependent results: a test that leaks provenance / logger
    state into its successor would cause failures that appear only when the
    suite runs in that specific order.
    """
    import cryocat.app.provenance as _prov
    import cryocat.app.session as _sess
    from cryocat.app.logger import dash_logger

    yield  # run the test

    # Teardown: reset all mutable singletons.
    _prov.clear()
    dash_logger.clear()
    try:
        from cryocat.app.pool import clear_payloads

        clear_payloads()
    except ImportError:
        pass
    try:
        from cryocat.app.console.execute import _CONSOLE_LOCALS, _add_pending

        _CONSOLE_LOCALS.clear()
        _add_pending.clear()
    except ImportError:
        pass
    # Do NOT call _sess.close_session() — session files are persistent artifacts.
    # Verify the provenance table is empty so any leaking test is caught here.
    assert _prov._producers == {}, "provenance table not empty after test — a test leaked pool state"


# ---------------------------------------------------------------------------
# App fixtures  (session-scoped: both apps import exactly once per test run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def suite_app():
    """The assembled Dash app from cryocat.app.suite.app."""
    import cryocat.app.suite.app as m

    return m.app


# ---------------------------------------------------------------------------
# Browser helper: open one suite page and collect what went wrong
# ---------------------------------------------------------------------------

# Firefox's webdriver gives no access to the browser console, so the page records
# console.error calls and uncaught errors itself before Dash loads.
_CONSOLE_CAPTURE = (
    "<script>window.__consoleErrors=[];"
    "(function(){var e=console.error;console.error=function(){"
    "window.__consoleErrors.push(Array.from(arguments).map(String).join(' '));"
    "e.apply(console,arguments);};"
    "window.addEventListener('error',function(ev){window.__consoleErrors.push(String(ev.message));});"
    "})();</script>"
)

# Components whose children are not in the DOM until opened or selected.
_NOT_RENDERED = {"Modal", "Offcanvas", "Tabs", "Tooltip", "Popover", "Store", "Download", "Interval", "Location"}


def main_component_ids(layout) -> list[str]:
    """String ids of a page's components that are rendered when the page shows."""
    found: list[str] = []

    def _walk(node):
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                _walk(child)
            return
        if type(node).__name__ in _NOT_RENDERED:
            return
        if isinstance(getattr(node, "id", None), str):
            found.append(node.id)
        _walk(getattr(node, "children", None))

    _walk(layout)
    return found


def open_suite_page(dash_duo, app, tool_id: str, monkeypatch) -> list[str]:
    """Serve the suite app, open the page of *tool_id*, wait until it shows.

    Returns the console errors the page recorded and the Dash dev-tools error count text.
    """
    from cryocat.app import ids
    from cryocat.app.suite.tools import TOOLS

    path = next(t["path"] for t in TOOLS if t["id"] == tool_id)
    monkeypatch.setattr(app, "index_string", app.index_string.replace("<head>", "<head>" + _CONSOLE_CAPTURE, 1))
    dash_duo.start_server(app, debug=True, use_reloader=False, dev_tools_hot_reload=False)
    dash_duo.wait_for_page(url=f"{dash_duo.server_url}{path}")
    dash_duo.wait_for_style_to_equal(f"#{ids.page_wrap_id(tool_id)}", "display", "block")
    dash_duo.wait_for_style_to_equal("#app-spinner", "display", "none")
    errors = list(dash_duo.driver.execute_script("return window.__consoleErrors || []"))
    errors += [f"Dash error count: {e.text}" for e in dash_duo.find_elements(".test-devtools-error-count")]
    return errors
