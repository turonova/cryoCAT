"""T0 — Minimal test harness: id collection and app coupling.

Three checks are run against both apps.  The tango parametrisation is marked
xfail because tango's ~40 top-level global stores are not yet migrated to
prefix-scoped state (doc 7, §2.4).

The suite parametrisation must PASS.  If it does not, the failures are real
defects — do not adjust the permitted-id set to paper over them.

Dependency on Dash version
--------------------------
Dash 4.1.0 does not expose ``allow_duplicate`` in its programmatic API:
* ``app._callback_list`` entries always have ``allow_duplicate=None``.
* ``app.callback_map`` deduplicates entries (last-write wins), and its Output
  objects show ``allow_duplicate=False`` regardless of the original setting.

Check 3 is therefore informational: it counts outputs written by multiple
callbacks and prints the tally, but cannot verify ``allow_duplicate=True``
from the API.  Tighten the check when Dash exposes this, or move it to a
source-code grep (doc 3).
"""

from __future__ import annotations

import json
from collections import Counter

import dash
import pytest

from tests.app.conftest import collect_id_paths, collect_ids
from cryocat.app import ids as _ids

# Fail loudly if Dash changes its internal representation of dict-pattern ids.
# In 4.1.0 they are JSON-encoded strings in _callback_list; a future version
# might return Python dicts.  If the assertion trips, re-audit _is_wildcard and
# _norm_id before removing it.  (§11.3, GUI_CONVENTIONS.md)
_DASH_TESTED_VERSION = "4.1.0"
assert dash.__version__ == _DASH_TESTED_VERSION, (
    f"Dash version changed from {_DASH_TESTED_VERSION!r} to {dash.__version__!r}. "
    "Re-audit _is_wildcard/_norm_id in this file before updating _DASH_TESTED_VERSION."
)

try:
    from dash import ALL, ALLSMALLER, MATCH

    _WILDCARDS = {ALL, MATCH, ALLSMALLER}
except ImportError:
    _WILDCARDS: set = set()

# Dash 4.1.0 serialises ALL/MATCH/ALLSMALLER as JSON arrays in _callback_list.
_WILDCARD_JSON: tuple = (["ALL"], ["MATCH"], ["ALLSMALLER"])


# ---------------------------------------------------------------------------
# §2.4 permitted global ids — the only unprefixed ids callbacks may reference.
# (Populated by T1/ids.py; hardcoded here until that task lands.)
# ---------------------------------------------------------------------------

_PERMITTED_EXACT: frozenset[str] = frozenset(
    {
        _ids.GRAPH_SETTINGS_STORE,
        _ids.POOL_REGISTRY,
        _ids.POOL_MOTLS,
        _ids.POOL_EXTRA,
        _ids.POOL_META,
        _ids.POOL_NEXT_ID,
        _ids.SUITE_URL,
        _ids.SUITE_TOOL_SELECTOR,
        _ids.SUITE_PAGE_CONTENT,
    }
)

_PERMITTED_PREFIXES: tuple[str, ...] = (_ids.PAGE_WRAP_PREFIX, _ids.SUITE_LOG_PREFIX)


def _is_permitted_global(component_id: object) -> bool:
    if not isinstance(component_id, str):
        return False
    return component_id in _PERMITTED_EXACT or any(
        component_id.startswith(p) for p in _PERMITTED_PREFIXES
    )


def _is_wildcard(component_id: object) -> bool:
    """True if the id contains Dash pattern-matching wildcards (ALL, MATCH…).

    Dash 4.1.0 stores dict ids in ``_callback_list`` as JSON-encoded strings,
    with ALL/MATCH/ALLSMALLER serialised as the JSON arrays ``["ALL"]`` etc.
    This function handles both the Python-dict and JSON-string forms.
    """
    if isinstance(component_id, str) and component_id.startswith("{"):
        try:
            component_id = json.loads(component_id)
        except Exception:
            return False
    if not isinstance(component_id, dict):
        return False
    for v in component_id.values():
        if isinstance(v, list) and v in _WILDCARD_JSON:
            return True
        try:
            if v in _WILDCARDS:
                return True
        except TypeError:
            pass
    return False


def _norm_id(component_id: object):
    """Normalise a Dash id to a hashable, comparable form.

    Handles plain strings, Python dicts, and JSON-encoded dict strings
    (the form Dash 4.1.0 uses in ``_callback_list``).  List values inside
    parsed dicts (JSON-serialised wildcards) are converted to tuples so the
    result is hashable — those ids are already filtered by ``_is_wildcard``,
    but the conversion keeps ``_norm_id`` safe against stray non-wildcard lists.
    """
    if isinstance(component_id, str) and component_id.startswith("{"):
        try:
            component_id = json.loads(component_id)
        except Exception:
            return component_id
    if isinstance(component_id, dict):
        def _v(val):
            return tuple(val) if isinstance(val, list) else val
        return tuple(sorted(((k, _v(v)) for k, v in component_id.items()), key=lambda kv: kv[0]))
    return component_id


def _all_dep_ids(app):
    """Yield every component_id referenced in any callback dependency.

    * Inputs and State: from ``app._callback_list`` (all registered callbacks).
    * Outputs: from ``app.callback_map`` (unique output → last-registered entry).
    """
    for item in app._callback_list:
        for dep in list(item.get("inputs", [])) + list(item.get("state", [])):
            yield dep["id"]
    for entry in app.callback_map.values():
        out = entry["output"]
        for o in (out if isinstance(out, list) else [out]):
            yield o.component_id


# ---------------------------------------------------------------------------
# Parametrisation — suite must pass; tango is xfail until doc 7 lands.
# ---------------------------------------------------------------------------

APPS = [
    pytest.param("suite", id="suite"),
    pytest.param(
        "tango",
        id="tango",
        marks=pytest.mark.xfail(
            reason="tango global store migration pending — doc 7", strict=False
        ),
    ),
]


def _app(name: str, request) -> object:
    return request.getfixturevalue(f"{name}_app")


# ---------------------------------------------------------------------------
# Check 1 — No duplicate ids in the mounted layout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_name", APPS)
def test_no_duplicate_ids(app_name, request):
    app = _app(app_name, request)
    paths = collect_id_paths(app.layout)
    duplicates = {k: v for k, v in paths.items() if len(v) > 1}
    assert not duplicates, (
        "Duplicate component ids found in layout:\n"
        + "\n".join(
            f"  {k!r}:\n" + "\n".join(f"    {p}" for p in v)
            for k, v in sorted(str(k) for k in duplicates)
        )
    )


# ---------------------------------------------------------------------------
# Check 2 — Every callback id resolves to a layout id or a permitted global
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_name", APPS)
def test_callback_ids_resolve(app_name, request):
    app = _app(app_name, request)
    layout_ids = collect_ids(app.layout)

    missing: dict[str, list] = {}
    for dep_id in _all_dep_ids(app):
        if _is_wildcard(dep_id):
            continue
        normed = _norm_id(dep_id)
        if normed not in layout_ids and not _is_permitted_global(dep_id):
            key = repr(dep_id)
            missing.setdefault(key, [])

    assert not missing, (
        f"{len(missing)} callback id(s) not found in the layout "
        f"or the §2.4 permitted set:\n"
        + "\n".join(f"  {k}" for k in sorted(missing))
    )


# ---------------------------------------------------------------------------
# Check 3 — Duplicate output report (informational; see module docstring)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_name", APPS)
def test_duplicate_output_report(app_name, request, capsys):
    """Report outputs with multiple writers.

    This check always passes — it exists to surface the count so a reviewer
    can spot regressions.  Verify allow_duplicate=True in source code.
    """
    app = _app(app_name, request)
    counts = Counter(item["output"] for item in app._callback_list)
    multi = {out: n for out, n in counts.items() if n > 1}

    if multi:
        with capsys.disabled():
            print(
                f"\n  [{app_name}] {len(multi)} output spec(s) written by multiple "
                f"callbacks (total extra callbacks: {sum(multi.values()) - len(multi)})."
                "\n  Each must have allow_duplicate=True in source (§3). "
                "Run pytest -s to see this message."
            )

    # The count must be non-negative; any real assertion lives in source review.
    assert all(n >= 1 for n in multi.values())
