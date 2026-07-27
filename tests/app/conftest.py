"""Shared utilities and fixtures for GUI app tests (T0+).

Utility functions (collect_ids, collect_id_paths) are plain functions, not
fixtures.  They can be imported directly:
    from tests.app.conftest import collect_ids, collect_id_paths
"""

import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Stub optional binary deps before any app module is imported.
# ---------------------------------------------------------------------------

for _mod_name in ("emfile",):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)


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

@pytest.fixture(scope="session")
def suite_app():
    """The assembled Dash app from cryocat.app.suite.app."""
    import cryocat.app.suite.app as m
    return m.app


@pytest.fixture(scope="session")
def tango_app():
    """The assembled Dash app from cryocat.app.tango.app."""
    import cryocat.app.tango.app as m
    return m.app
