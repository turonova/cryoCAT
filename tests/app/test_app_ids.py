"""T0 — Minimal test harness: id collection and app coupling.

Two checks are run against both apps.  The tango parametrisation is marked
xfail because tango's ~40 top-level global stores are not yet migrated to
prefix-scoped state (doc 7, §2.4).

The suite parametrisation must PASS.  If it does not, the failures are real
defects — do not adjust the permitted-id set to paper over them.
"""

from __future__ import annotations

import json

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
        _ids.POOL_META,
        _ids.POOL_NEXT_ID,
        _ids.POOL_GROUPS,
        _ids.SUITE_URL,
        _ids.SUITE_TOOL_SELECTOR,
        _ids.SUITE_PAGE_CONTENT,
        # Phase 10 — single app-level file browser (D1, Z2)
        _ids.BROWSER_REQUEST,
        _ids.BROWSER_CWD,
        _ids.BROWSER_LAST_DIR,
        _ids.BROWSER_RESULT,
        # Phase 11 — single app-level rotation-builder modal (D1)
        _ids.ROTATION_REQUEST,
        # ORIENTATION_PICKER_PLACEMENT.md — single app-level orient-picker modal (D1)
        _ids.ORIENT_REQUEST,
    }
)

_PERMITTED_PREFIXES: tuple[str, ...] = (
    _ids.PAGE_WRAP_PREFIX,
    _ids.SUITE_LOG_PREFIX,
    "browser-",    # Phase 10: single app-level file-browser modal (D1)
    "rotation-modal-",  # Phase 11: single app-level rotation-builder modal (D1)
    "orient-modal-",    # ORIENTATION_PICKER_PLACEMENT.md: single app-level orient-picker modal
)


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
            for k, v in sorted(duplicates.items(), key=lambda kv: str(kv[0]))
        )
    )


# ---------------------------------------------------------------------------
# §11.3 — DYNAMIC_IDS registry
#
# Modules that render controls into placeholder divs declare:
#   DYNAMIC_IDS: list[tuple[str, str]] = [(container_id, component_id), ...]
#
# The coupling check accepts an unresolved component_id only if it is declared
# here.  The container_id MUST be present in the static layout — a declaration
# without a mounted container is itself a defect and is caught below.
# ---------------------------------------------------------------------------

_DYNAMIC_ID_MODULES: tuple = ()


def _load_dynamic_modules(app_name: str = "suite"):
    """Lazy import to avoid circular deps at module level."""
    if app_name == "suite":
        from cryocat.app.suite.pages import passembly, pmotl, pnn, ptango, pdatapool, pgraphs
        return (passembly, pmotl, pnn, ptango, pdatapool, pgraphs)
    return ()


def _collect_dynamic_ids(layout_ids: set, app_name: str = "suite") -> frozenset:
    """Return the set of accepted dynamic component ids (normalised).

    Also asserts that every declared container_id is in the layout, so a stale
    declaration cannot silently paper over a broken id.
    """
    modules = _load_dynamic_modules(app_name)
    accepted: set = set()
    for mod in modules:
        for container_id, component_id in getattr(mod, "DYNAMIC_IDS", []):
            container_norm = _norm_id(container_id)
            assert container_norm in layout_ids, (
                f"DYNAMIC_IDS in {mod.__name__!r} declares container "
                f"{container_id!r} which is not present in the layout. "
                "Either mount the container or remove the declaration."
            )
            accepted.add(_norm_id(component_id))
    return frozenset(accepted)


# ---------------------------------------------------------------------------
# Check 2 — Every callback id resolves to a layout id or a permitted global
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_name", APPS)
def test_callback_ids_resolve(app_name, request):
    app = _app(app_name, request)
    layout_ids = collect_ids(app.layout)
    dynamic_ids = _collect_dynamic_ids(layout_ids, app_name)

    missing: dict[str, list] = {}
    for dep_id in _all_dep_ids(app):
        if _is_wildcard(dep_id):
            continue
        normed = _norm_id(dep_id)
        if (
            normed not in layout_ids
            and normed not in dynamic_ids
            and not _is_permitted_global(dep_id)
        ):
            key = repr(dep_id)
            missing.setdefault(key, [])

    assert not missing, (
        f"{len(missing)} callback id(s) not found in the layout, "
        f"the §2.4 permitted set, or DYNAMIC_IDS:\n"
        + "\n".join(f"  {k}" for k in sorted(missing))
    )



# ---------------------------------------------------------------------------
# Every page and component module imports and builds its layout
# ---------------------------------------------------------------------------

import importlib
import inspect
import pkgutil

from dash import html


def _page_modules() -> list[str]:
    from cryocat.app.suite.tools import TOOLS
    return [t["module"] for t in TOOLS]


def _app_modules() -> list[str]:
    """Every module under cryocat/app except the page modules (tested separately)."""
    import cryocat.app
    pages = set(_page_modules())
    names = []
    for info in pkgutil.walk_packages(cryocat.app.__path__, "cryocat.app."):
        if info.name not in pages and ".e2e" not in info.name:
            names.append(info.name)
    return sorted(names)


def _page_layout(module):
    layout = module.layout
    return layout() if callable(layout) else layout


# Builders whose arguments beyond the prefix have no default.
_BUILDER_ARGS: dict[str, dict] = {
    "get_builder_field": {"panel": html.Div(), "title": "Builder", "target_id": "gt-target"},
    "get_table_source": {"compute_children": []},
    "get_relion_options": {"for_load": True},
}


def _layout_builders(module) -> list:
    """Public ``get_*`` functions defined in a component module or the motl sidebar."""
    if not module.__name__.startswith("cryocat.app.components.") and module.__name__ != "cryocat.app.suite.motlsidebar":
        return []
    return [
        fn for name, fn in vars(module).items()
        if name.startswith("get_") and inspect.isfunction(fn) and fn.__module__ == module.__name__
    ]


def _call_builder(fn):
    params = inspect.signature(fn).parameters
    kwargs = dict(_BUILDER_ARGS.get(fn.__name__, {}))
    args = ["gt"] if "prefix" in params else []
    return fn(*args, **kwargs)


@pytest.mark.parametrize("module_name", _page_modules())
def test_page_builds_its_layout(module_name):
    """The page imports, exposes a layout with components, and a register_callbacks function."""
    module = importlib.import_module(module_name)
    assert collect_ids(_page_layout(module)), f"{module_name}: layout has no component ids"
    assert callable(getattr(module, "register_callbacks", None))


@pytest.mark.parametrize("module_name", _app_modules())
def test_app_module_imports_and_builds(module_name):
    """The module imports; each of its get_* layout builders returns a component tree."""
    module = importlib.import_module(module_name)
    for fn in _layout_builders(module):
        built = _call_builder(fn)
        assert built is not None, f"{module_name}.{fn.__name__} returned None"


# ---------------------------------------------------------------------------
# Every callback id of a page exists in that page's layout
# ---------------------------------------------------------------------------

def _shell_ids(suite_app) -> set:
    """Ids of the app-level layout (pool stores, modals, log panel), without any page."""
    page_ids: set = set()
    for name in _page_modules():
        page_ids |= collect_ids(_page_layout(importlib.import_module(name)))
    return collect_ids(suite_app.layout) - page_ids


def _parse_dep(raw):
    if isinstance(raw, str) and raw.startswith("{"):
        return json.loads(raw)
    return raw


def _page_deps(app) -> list[tuple[str, object]]:
    deps = []
    for item in app._callback_list:
        for role in ("inputs", "state"):
            for dep in item.get(role, []):
                deps.append((role, _parse_dep(dep["id"])))
    for entry in app.callback_map.values():
        out = entry["output"]
        for o in (out if isinstance(out, list) else [out]):
            deps.append(("output", _parse_dep(o.component_id)))
    return deps


def _is_wild(v) -> bool:
    return (isinstance(v, list) and v in _WILDCARD_JSON) or (not isinstance(v, (list, dict)) and v in _WILDCARDS)


def _static_types(ids: set) -> set:
    return {dict(i).get("type") for i in ids if isinstance(i, tuple)}


@pytest.mark.parametrize("module_name", _page_modules())
def test_page_callback_ids_exist_in_its_layout(module_name, suite_app, monkeypatch):
    """Each Input, State and Output of the page's callbacks is in its layout, the app shell or DYNAMIC_IDS.

    A pattern dependency is created by a pattern the page defines when its ``type``
    is used by a component of the page or shell, or by no static component at all
    (a form or graph its callbacks build). It fails when only another page's
    components use that type.
    """
    from cryocat.app.components import builderfield

    module = importlib.import_module(module_name)
    layout = _page_layout(module)
    shell = _shell_ids(suite_app)
    known = collect_ids(layout) | shell | {c for _, c in getattr(module, "DYNAMIC_IDS", [])}
    own_types = _static_types(known)
    other_types: set = set()
    for name in _page_modules():
        if name != module_name:
            other_types |= _static_types(collect_ids(_page_layout(importlib.import_module(name))))

    # The suite app already registered this page once; builderfield refuses a second prefix.
    monkeypatch.setattr(builderfield, "_REGISTERED", set())
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div(layout)
    module.register_callbacks(app)

    missing = set()
    for role, dep in _page_deps(app):
        if isinstance(dep, dict):
            dep_type = dep.get("type")
            if not _is_wild(dep_type) and dep_type not in own_types and dep_type in other_types:
                missing.add(f"{role}: {json.dumps(dep, default=str, sort_keys=True)}")
        elif dep not in known:
            missing.add(f"{role}: {dep}")
    assert not missing, f"{module_name}: {len(missing)} callback id(s) not in its layout:\n" + "\n".join(
        f"  {m}" for m in sorted(missing)
    )
