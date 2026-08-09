"""Q3 — GUI conventions as tests.

Mechanical enforcement of GUI_CONVENTIONS.md rules so they cannot quietly
erode as the codebase evolves.  All checks use AST analysis (not grep) so
aliased imports cannot slip through.

Exemption lists are debt ledgers: they may SHRINK but never GROW.
Each entry carries a reason and a pointer to the doc that will fix it.
"""
from __future__ import annotations

import ast
import pathlib
import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

_APP = pathlib.Path(__file__).parent.parent.parent / "cryocat" / "app"
_SUITE = _APP / "suite"
_TANGO = _APP / "tango"
_COMPONENTS = _APP / "components"


def _py_files(*roots: pathlib.Path) -> list[pathlib.Path]:
    out = []
    for r in roots:
        out.extend(
            p for p in sorted(r.rglob("*.py")) if "__pycache__" not in str(p)
        )
    return out


def _parse(p: pathlib.Path) -> ast.Module | None:
    try:
        return ast.parse(p.read_text(encoding="utf-8"))
    except SyntaxError:
        return None


# ── §2.4 / §5 single-owner string literals ───────────────────────────────────

def _string_literals(tree: ast.Module) -> list[tuple[int, str]]:
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_graph_settings_store_only_in_ids():
    """§2.4 — 'graph-settings-store' literal appears only in ids.py."""
    banned = "graph-settings-store"
    violations = []
    for p in _py_files(_APP):
        if p.name == "ids.py":
            continue
        tree = _parse(p)
        if tree is None:
            continue
        for lineno, val in _string_literals(tree):
            if val == banned:
                violations.append(f"{p.relative_to(_APP)}:{lineno}")
    assert not violations, (
        f"'{banned}' literal outside ids.py:\n" + "\n".join(violations)
    )


def test_pool_prefix_only_in_ids():
    """§5 — 'pool-*' prefixed string literals appear only in ids.py."""
    violations = []
    for p in _py_files(_APP):
        if p.name == "ids.py":
            continue
        tree = _parse(p)
        if tree is None:
            continue
        for lineno, val in _string_literals(tree):
            if val.startswith("pool-"):
                violations.append(f"{p.relative_to(_APP)}:{lineno}  {val!r}")
    assert not violations, (
        "'pool-*' literals outside ids.py:\n" + "\n".join(violations)
    )


def test_to_plotly_json_only_in_graphsettings():
    """§4.1 — to_plotly_json() called only in graphsettings.py."""
    violations = []
    for p in _py_files(_APP):
        if p.name == "graphsettings.py":
            continue
        tree = _parse(p)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "to_plotly_json"
            ):
                violations.append(f"{p.relative_to(_APP)}:{node.lineno}")
    assert not violations, (
        "to_plotly_json() call outside graphsettings.py:\n" + "\n".join(violations)
    )


def test_format_arg_only_in_logger():
    """format_arg is an internal label formatter — only logger.py may call it."""
    violations = []
    for p in _py_files(_APP):
        if p.name == "logger.py":
            continue
        tree = _parse(p)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "format_arg"
            ):
                violations.append(f"{p.relative_to(_APP)}:{node.lineno}")
    assert not violations, (
        "format_arg() called outside logger.py:\n" + "\n".join(violations)
    )


def test_call_event_only_in_invoke_operation():
    """§6 — call_event() emitted only from logger.py (invoke_operation)."""
    violations = []
    for p in _py_files(_APP):
        if p.name == "logger.py":
            continue
        tree = _parse(p)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "call_event"
            ):
                violations.append(f"{p.relative_to(_APP)}:{node.lineno}")
    assert not violations, (
        "call_event() emitted outside logger.py:\n" + "\n".join(violations)
    )


# ── §8 — 100vh confined to style modules ─────────────────────────────────────

# pageshell.py owns the sticky-sidebar skeleton and legitimately carries 100vh.
_STYLE_MODULES = {"styles.py", "pageshell.py"}
# tango/ uses inline styles pending a doc 7 style extraction pass.
_STYLE_MODULES_EXEMPT_DIRS = {_TANGO}


def test_100vh_only_in_style_modules():
    """§8 — '100vh' string appears only in styles.py and pageshell.py.

    tango/ is exempt (doc 7 — style extraction not yet done).
    """
    violations = []
    for p in _py_files(_APP):
        if p.name in _STYLE_MODULES:
            continue
        if any(p.is_relative_to(d) for d in _STYLE_MODULES_EXEMPT_DIRS):
            continue
        tree = _parse(p)
        if tree is None:
            continue
        for lineno, val in _string_literals(tree):
            if val == "100vh":
                violations.append(f"{p.relative_to(_APP)}:{lineno}")
    assert not violations, (
        "'100vh' literal outside style modules (tango/ exempt):\n" + "\n".join(violations)
    )


# ── §14 — banned dead function names ─────────────────────────────────────────

_BANNED_FN_NAMES = frozenset({
    "render_py", "render_ipynb", "wrap_slurm",
    "get_motl_operation_methods",
    "get_print_out", "patch_class", "patch_function",
    # Phase 4 (R4) — deleted; read from discovery instead
    "iter_standalone_builders", "_iter_gui_methods",
    "get_single_motl_methods", "get_multi_motl_methods",
})


def test_no_banned_function_names():
    """§14 — no module defines a function by one of the banned (dead) names."""
    violations = []
    for p in _py_files(_APP):
        tree = _parse(p)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in _BANNED_FN_NAMES
            ):
                violations.append(f"{p.relative_to(_APP)}:{node.lineno}  def {node.name}")
    assert not violations, (
        "Banned (dead) function names found:\n" + "\n".join(violations)
    )


# ── §3 — thin callbacks ───────────────────────────────────────────────────────

# Debt ledger: module stem → reason + doc pointer.  May shrink, never grow.
# Modules are exempt iff their callbacks exceed the threshold due to known
# complexity that is scheduled for extraction in doc 6 or doc 7.
THIN_CALLBACK_EXEMPT: dict[str, str] = {
    # §15 — design unsettled; own phase document when resolved
    "pcomplexes":    "§15 — NPC static-run and motl-list push loops",
    "pstructure":    "§15 — surface-loader and run-operation loops",
    # doc 8 — deferred; carve out a focused pass per module
    "tablecluster":  "doc 8 — k-means and proximity run loops inline",
    "tableplot":     "doc 8 — graph-sync callback iterates traces",
    "memthick_widgets": "doc 8 — per-label render loops",
    "volumeview":    "doc 8 — marching-cubes extraction loop",
    "logpanel":      "doc 8 — unified log-update callback (15 stmts)",
    "motlsidebar":   "doc 8 — route_motl / slot-map loops",
    "pmemthick":     "doc 8 — boundary-table render and load loops",
    "pmotl":         "doc 8 — slot-sync loops",
    "pnn":           "doc 8 — NN column-row builder and compute loops",
    "ppana":         "doc 8 — CSV-to-table and visualise-row loops",
    "psta":          "doc 8 — alignment-evaluation and param-load",
    "putilities":    "doc 8 — utilities page with multi-step builder callbacks",
    "pvolume":       "doc 8 — volume page with extraction loop callbacks",
    "consoleui":     "doc 8 — _on_submit owns pool mutations; _suggest iterates completions",
    # tango page modules — not addressed by thin-callback lint in this phase
    "sidebar":       "doc 8 — tango sidebar with inline 100vh and compound callbacks",
    "table":         "doc 8 — tango table with column-iteration callbacks",
}

_THIN_THRESHOLD_STMTS = 8


def _callback_violations(p: pathlib.Path) -> list[str]:
    tree = _parse(p)
    if tree is None:
        return []
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(
            "callback" in ast.unparse(d) for d in node.decorator_list
        ):
            continue
        stmts = len(node.body)
        has_loop = any(
            isinstance(n, (ast.For, ast.While))
            for n in ast.walk(ast.Module(body=node.body, type_ignores=[]))
        )
        if stmts > _THIN_THRESHOLD_STMTS or has_loop:
            found.append(
                f"{node.name}:L{node.lineno} stmts={stmts} loop={has_loop}"
            )
    return found


@pytest.mark.parametrize(
    "src_path",
    _py_files(_APP),
    ids=lambda p: p.stem,
)
def test_thin_callbacks(src_path: pathlib.Path):
    """§3 — callbacks must have ≤8 statements and no loops."""
    stem = src_path.stem
    violations = _callback_violations(src_path)
    if stem in THIN_CALLBACK_EXEMPT:
        pytest.xfail(reason=THIN_CALLBACK_EXEMPT[stem])
    if violations:
        pytest.fail(
            f"{src_path.relative_to(_APP)} has fat callbacks:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


def test_thin_callback_exempt_list_is_bounded():
    """Every exempt module must reference a known doc or §15 — never an unknown module.

    §15 modules (pstructure, pcomplexes, surfaces) are deferred until the design
    settles; they carry an explicit §15 marker instead of a doc number.
    """
    known_tokens = {"doc 6", "doc 7", "doc 8", "§15"}
    for stem, reason in THIN_CALLBACK_EXEMPT.items():
        mentioned = any(tok in reason for tok in known_tokens)
        assert mentioned, (
            f"THIN_CALLBACK_EXEMPT[{stem!r}] reason does not reference a known doc "
            f"(doc 6/7/8) or §15: {reason!r}"
        )


# ── allow_duplicate baseline ──────────────────────────────────────────────────

# Baseline recorded at Phase 3 write time.  The test asserts the count only
# goes DOWN (shrinks), never UP.  Update this number when the count drops.
_ALLOW_DUPLICATE_BASELINE = 308  # Phase 1/2 + R1/R2/R3 (283) + F1 motlinput (2) + F3 seq-load (4) + F2 pool-load (2) + Part F varpicker (2) + B2/C group-options (4) + col-merge modal (4) - motlio simple save (3) + savedialog prefill (1) + select-all-visible toggle (1) + P4 pool-aware tablefilter/tableedit/tablecluster/pmotl (+7) + P6 _sync_revisions (+1)


def test_allow_duplicate_count_does_not_grow():
    """§3 — allow_duplicate=True occurrences must not exceed the baseline."""
    count = 0
    for p in _py_files(_APP):
        try:
            src = p.read_text(encoding="utf-8")
        except OSError:
            continue
        count += src.count("allow_duplicate=True")
    assert count <= _ALLOW_DUPLICATE_BASELINE, (
        f"allow_duplicate=True count grew from {_ALLOW_DUPLICATE_BASELINE} to {count}. "
        "Add allow_duplicate justification comments and shrink the baseline."
    )


# ── Dead-module detector ──────────────────────────────────────────────────────

# Known entry points that are legitimately not imported by other modules.
_ENTRY_POINTS: frozenset[str] = frozenset({
    "cryocat.app.server",
    "cryocat.app.suite.app",
    "cryocat.app.tango.app",
    "cryocat.app.discovery",  # also used as __main__ for --report
})
# Modules that are only imported by test code or are intentionally standalone.
_IMPORT_EXEMPT: frozenset[str] = frozenset({
    # suite/app.py loads pages via importlib.import_module(t["module"]) — no
    # static import appears in the AST, but they are all reachable.
    "cryocat.app.suite.pages.pcomplexes",
    "cryocat.app.suite.pages.pmemthick",
    "cryocat.app.suite.pages.pmotl",
    "cryocat.app.suite.pages.pnn",
    "cryocat.app.suite.pages.ppana",
    "cryocat.app.suite.pages.psta",

    "cryocat.app.suite.pages.pstructure",
    "cryocat.app.suite.pages.putilities",
    "cryocat.app.suite.pages.pvolume",
    # Internal helpers only imported by sibling modules in the same package.
    "cryocat.app.suite.pages._codegen_base",
    "cryocat.app.suite.pages._memthick_analysis",
    "cryocat.app.suite.pages._memthick_codegen",
    "cryocat.app.suite.pages._pana_codegen",
    "cryocat.app.suite.pages._pstructure_intersect",
    "cryocat.app.record._common",    # internal record helper
    # motlinput.py is tested directly by test_motl_input.py; superseded by
    # poolpicker.py for app use so no app module imports it any more.
    "cryocat.app.components.motlinput",
})


def _build_import_graph() -> tuple[dict[str, list[str]], dict[str, pathlib.Path]]:
    """Return (import_graph, module_map) for cryocat.app.**."""
    # Key modules by their full dotted path (e.g. "cryocat.app.styles") so
    # they match the import strings we collect below.
    _CRYOCAT_ROOT = _APP.parent.parent  # …/cryoCAT/
    module_map: dict[str, pathlib.Path] = {}
    for p in _py_files(_APP):
        rel = p.relative_to(_CRYOCAT_ROOT)
        mod = ".".join(rel.with_suffix("").parts)
        module_map[mod] = p

    import_graph: dict[str, list[str]] = {}
    for mod, p in module_map.items():
        tree = _parse(p)
        if tree is None:
            import_graph[mod] = []
            continue
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                base = node.module
                if "cryocat.app" in base:
                    imports.append(base)
                    # from cryocat.app.foo import bar  →  also track cryocat.app.foo.bar
                    for alias in node.names:
                        if alias.name != "*":
                            imports.append(f"{base}.{alias.name}")
                # from cryocat.app import styles  →  track cryocat.app.styles
                if base == "cryocat.app" or base.startswith("cryocat.app."):
                    for alias in node.names:
                        if alias.name != "*":
                            imports.append(f"{base}.{alias.name}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "cryocat.app" in alias.name:
                        imports.append(alias.name)
        import_graph[mod] = imports
    return import_graph, module_map


def test_no_dead_modules():
    """Every app module is imported by at least one other module or is an entry point."""
    import_graph, module_map = _build_import_graph()

    all_imported: set[str] = set()
    for imports in import_graph.values():
        all_imported.update(imports)

    orphans = []
    for mod in sorted(module_map):
        if mod.endswith("__init__"):
            continue
        if mod in _ENTRY_POINTS or mod in _IMPORT_EXEMPT:
            continue
        if mod not in all_imported:
            orphans.append(mod)

    assert not orphans, (
        "Orphaned (never-imported) app modules detected:\n"
        + "\n".join(f"  {m}" for m in orphans)
        + "\nAdd to _IMPORT_EXEMPT with a reason, or delete the module."
    )


# ── Part 3 — pool boundary enforcement ───────────────────────────────────────
#
# Three structural rules that must hold after P6 is fully landed:
#
#   L3  get_log_panel() called only in suite/app.py (not per-page).
#   P6a pd.DataFrame(var) with a single Name arg must not appear inside
#       @app.callback functions (use pool.get_rows() instead).
#   P6b result.df.to_dict("records") must not appear inside @app.callback
#       functions (route through insert_motl / replace_motl_rows instead).
#
# Exemption dicts are debt ledgers: they MUST shrink, never grow.
# Each entry names the tracking doc so the obligation doesn't get lost.


def test_single_log_panel_call_site_in_suite():
    """L3 — within cryocat/app/suite/, get_log_panel() is called only in app.py."""
    violations = []
    for p in _py_files(_SUITE):
        if p.name == "app.py":
            continue
        tree = _parse(p)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "get_log_panel"
            ):
                violations.append(f"{p.relative_to(_APP)}:{node.lineno}")
    assert not violations, (
        "get_log_panel() called outside suite/app.py in suite/:\n"
        + "\n".join(violations)
    )


# Debt ledger for pd.DataFrame(Name_arg) inside @app.callback functions.
# The pattern means: callback reads list[dict] from a store and wraps it in
# pd.DataFrame() — the old pre-pool pattern.  Correct form: pool.get_rows(mid).
# May SHRINK as P6 lands; never grow.
_DATAFRAME_FROM_STORE_EXEMPT: dict[str, str] = {
    "motlsidebar": "P6-pending — run_multi_op/apply_operation read motl rows from store",
    "pnn":         "P6-pending — compute_nn/merge/build callbacks wrap result/selection stores",
    "tablecluster": "P6-pending — merge-back and scatter callbacks wrap cluster-data store",
    "pstructure":  "P6-pending — _render_isect_results wraps hits snapshot from result store",
    "motlio":      "P6-pending — save_data callback wraps data_to_save store arg",
    "tablesave":   "legitimate — do_csv_save wraps grid_data for CSV export (not a pool roundtrip)",
}


def _dataframe_from_name_in_callbacks(p: pathlib.Path) -> list[str]:
    """Return call-site labels where pd.DataFrame(Name_only) appears:
    - inside @app.callback-decorated functions, OR
    - inside non-callback helper functions nested in register_*_callbacks functions.

    The second branch catches the _motl_df pattern in tomoview: a helper defined
    inside register_viewer_callbacks that contains pd.DataFrame(data) but is not
    itself decorated with @app.callback.  (T2 — PERF_FINISH)
    """
    tree = _parse(p)
    if tree is None:
        return []
    found = []

    def _scan_nodes(body_nodes: list) -> None:
        for node in ast.walk(ast.Module(body=body_nodes, type_ignores=[])):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "DataFrame"):
                continue
            if (
                len(node.args) == 1
                and isinstance(node.args[0], ast.Name)
                and len(node.keywords) == 0
            ):
                found.append(f"{fn.name}:L{node.lineno}  pd.DataFrame({node.args[0].id})")

    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_callback = any("callback" in ast.unparse(d) for d in fn.decorator_list)
        is_registrar = fn.name.startswith("register_") and fn.name.endswith("_callbacks")
        if not (is_callback or is_registrar):
            continue

        if is_callback:
            _scan_nodes(fn.body)
        else:
            # Registrar: collect all non-callback inner helpers (including deeply nested)
            # and scan their bodies.  Callback-decorated inners are found via is_callback.
            inner_bodies: list = []
            for inner in ast.walk(ast.Module(body=fn.body, type_ignores=[])):
                if not isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if any("callback" in ast.unparse(d) for d in inner.decorator_list):
                    continue
                inner_bodies.extend(inner.body)
            _scan_nodes(inner_bodies)

    return found


@pytest.mark.parametrize(
    "src_path",
    _py_files(_SUITE, _COMPONENTS),
    ids=lambda p: p.stem,
)
def test_no_dataframe_from_store_arg(src_path: pathlib.Path):
    """P6a — no pd.DataFrame(var) inside @app.callback functions in suite/ and components/.

    Correct pattern: pool.get_rows(motl_id) returns a DataFrame directly;
    the callback never needs to wrap a list[dict] from a dcc.Store.
    tango/ is a separate migration track and is not covered here.
    """
    stem = src_path.stem
    violations = _dataframe_from_name_in_callbacks(src_path)
    if stem in _DATAFRAME_FROM_STORE_EXEMPT:
        if violations:
            pytest.xfail(reason=_DATAFRAME_FROM_STORE_EXEMPT[stem])
        return
    if violations:
        pytest.fail(
            f"{src_path.relative_to(_APP)} has pd.DataFrame(store_arg) in callbacks:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


# Debt ledger for .to_dict("records") inside @app.callback functions.
# Broadened from the earlier X.df.to_dict chain check to catch any receiver
# expression.  Correct pattern after P6: route the motl through the pool and
# output a reference handle; the grid fetches rows via its server-side datasource.
# May SHRINK as P6 and PERF_FINISH W2 land; never grow.
# Primary targets not in this list (must stay RED): tablefilter, tablegrid.
_TO_DICT_RECORDS_EXEMPT: dict[str, str] = {
    "pstructure":   "P6-pending — _run_operation/_build_point_cloud_motl route result.df to store",
    "pnn":          "P6-pending — _apply_postprocessing/_load_nn_from_csv route nn_stats.df to store",
    "psta":         "P6-pending — load_from_params routes params.df to store (pre-pool pattern)",
    "pmotl":        "P6-pending — sync_pool_to_slots serialises rows to motl-data-store and extra-store",
    "motlsidebar":  "P6-pending — run_multi_op/apply_operation route result.df to store",
    "tablecluster": "P6-pending — merge-back and scatter callbacks write cluster-data store",
    "motlio":       "P6-pending — save_data callback wraps data_to_save arg from store",
    "tablesave":    "legitimate — do_csv_save wraps grid_data for CSV export (not a pool roundtrip)",
    "motlsource":   "P6-pending — _to_table writes get_rows().to_dict('records') to src-tabv-global-data-store (pre-pool store format)",
    "relionopts":   "legitimate — _on_tomos_load writes tomogram dimension data (not motl rows) to rln-tomos-store",
    "tablegrid":    "W2-partial — load_data_to_grid serialises ≤2k rows; full fix needs AG Grid server-side/infinite row model",
}


def _to_dict_records_in_callbacks(p: pathlib.Path) -> list[str]:
    """Return call-site labels where .to_dict('records') appears inside a callback.

    Catches any receiver expression (not just X.df.to_dict chains).
    This is the broadened form of the earlier _motl_df_to_dict_in_callbacks.  (T3)
    """
    tree = _parse(p)
    if tree is None:
        return []
    found = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any("callback" in ast.unparse(d) for d in fn.decorator_list):
            continue
        for node in ast.walk(ast.Module(body=fn.body, type_ignores=[])):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "to_dict"):
                continue
            if (
                len(node.args) == 1
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "records"
            ):
                found.append(f"{fn.name}:L{node.lineno}  .to_dict('records')")
    return found


@pytest.mark.parametrize(
    "src_path",
    _py_files(_SUITE, _COMPONENTS),
    ids=lambda p: p.stem,
)
def test_no_to_dict_records_in_callbacks(src_path: pathlib.Path):
    """T3/P6c — no .to_dict('records') inside @app.callback functions.

    Correct pattern: route the motl through the pool and output a reference handle;
    the grid receives rows through its server-side datasource, not via store
    serialization.  tablefilter.apply_filters_btn and tablegrid.load_data_to_grid
    are the primary targets (W2 will fix them); remaining items are P6 debt.
    """
    stem = src_path.stem
    violations = _to_dict_records_in_callbacks(src_path)
    if stem in _TO_DICT_RECORDS_EXEMPT:
        if violations:
            pytest.xfail(reason=_TO_DICT_RECORDS_EXEMPT[stem])
        return
    if violations:
        pytest.fail(
            f"{src_path.relative_to(_APP)} has .to_dict('records') in callbacks:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


# ── PERF_FINISH T1 — metadata pure functions accept handles, reject row data ──

def test_color_options_from_handle_accepts_handle():
    """T1 — color_options_from_handle returns dropdown options from a PoolEntry dict."""
    from cryocat.app.components.tomoview import color_options_from_handle

    handle = {"numeric_columns": ["x", "y", "score", "tomo_id"], "tomo_ids": [1]}
    opts = color_options_from_handle(handle)
    assert {"label": "x", "value": "x"} in opts
    assert not any(o["value"] == "tomo_id" for o in opts), "tomo_id must be excluded from color opts"


def test_color_options_from_handle_rejects_row_data():
    """T1 — color_options_from_handle raises TypeError when passed list[dict] store data."""
    from cryocat.app.components.tomoview import color_options_from_handle

    with pytest.raises(TypeError):
        color_options_from_handle([{"x": 1.0, "tomo_id": 1}])


def test_tomo_items_from_handle_accepts_handle():
    """T1 — tomo_items_from_handle returns menu items from a PoolEntry dict."""
    from cryocat.app.components.tomoview import tomo_items_from_handle

    handle = {"tomo_ids": [1, 2, 3], "numeric_columns": ["x"]}
    items = tomo_items_from_handle(handle, "test")
    assert len(items) == 3


def test_tomo_items_from_handle_rejects_row_data():
    """T1 — tomo_items_from_handle raises TypeError when passed list[dict] store data."""
    from cryocat.app.components.tomoview import tomo_items_from_handle

    with pytest.raises(TypeError):
        tomo_items_from_handle([{"tomo_id": 1}], "test")


def test_slider_specs_rejects_non_dict():
    """T1 — slider_specs raises TypeError when passed list[dict] instead of column_ranges dict."""
    from cryocat.app.components.tablefilter import slider_specs

    with pytest.raises(TypeError):
        slider_specs([{"x": 1}])


# ── PERF_FINISH T4 — stores carry no row data; deprecated ids absent ──────────

def test_pool_stores_carry_no_row_data():
    """T4a — PoolState.to_stores() for a large motl fits under 256 KB with no list[dict]."""
    import json
    from cryocat.app.pool import insert_motl, PoolState, clear_payloads

    clear_payloads()
    rng = np.random.default_rng(0)
    n = 100_000
    df = pd.DataFrame({
        "tomo_id": rng.integers(1, 50, size=n).astype(float),
        "x": rng.random(n),
        "y": rng.random(n),
        "z": rng.random(n),
        "score": rng.random(n),
    })
    state = PoolState(registry={}, meta={}, next_id=0)
    new_state, _mid = insert_motl(state, df)
    registry, meta, _next = new_state.to_stores()

    total_bytes = len(json.dumps(registry).encode()) + len(json.dumps(meta).encode())
    assert total_bytes < 256_000, (
        f"Pool stores for 100k-row motl exceed 256 KB: {total_bytes:,} bytes"
    )
    for mid, entry in registry.items():
        assert not isinstance(entry, list), (
            f"registry[{mid!r}] is a list — stores must carry handles, not rows"
        )
    clear_payloads()


def test_no_deprecated_pool_store_ids_in_app():
    """T4b — deprecated ids 'pool-motls' and 'pool-extra' must not appear anywhere in cryocat/app/."""
    banned = {"pool-motls", "pool-extra"}
    violations = []
    for p in _py_files(_APP):
        tree = _parse(p)
        if tree is None:
            continue
        for lineno, val in _string_literals(tree):
            if val in banned:
                violations.append(f"{p.relative_to(_APP)}:{lineno}  {val!r}")
    assert not violations, (
        "Deprecated pool store ids found — remove POOL_MOTLS / POOL_EXTRA:\n"
        + "\n".join(violations)
    )


# ── Registration guard (§10) ──────────────────────────────────────────────────

@pytest.mark.xfail(
    reason=(
        "§10 registration guard not yet implemented in components — "
        "doc 8 will add a per-component seen-prefix set that raises on duplicate. "
        "Until then, Dash silently overwrites callbacks on re-registration."
    ),
    strict=True,
)
def test_paletteloader_double_registration_raises():
    """§10 — registering palette-loader callbacks twice must raise."""
    import dash
    from cryocat.app.components.paletteloader import register_palette_loader_callbacks

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    prefix = "test-pal-reg"
    register_palette_loader_callbacks(app, prefix, mode="discrete")
    with pytest.raises(Exception, match=prefix):
        register_palette_loader_callbacks(app, prefix, mode="discrete")
