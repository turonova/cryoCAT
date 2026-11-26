"""Shared test oracles for GUI app tests.

Import these instead of re-writing the same checks per module.
"""
from __future__ import annotations

import ast
import builtins


# ── Figure invariants ─────────────────────────────────────────────────────────

def assert_styled(fig: dict, settings: dict, *, uirevision: str) -> None:
    """Assert figure layout honours the graph settings contract.

    Checks:
    - ``uirevision`` is set to *uirevision*
    - ``layout.font.family`` matches settings
    - ``layout.font.size`` matches settings
    """
    layout = fig.get("layout", {})
    assert layout.get("uirevision") == uirevision, (
        f"uirevision mismatch: {layout.get('uirevision')!r} != {uirevision!r}"
    )
    font = layout.get("font", {})
    if "font_family" in settings:
        assert font.get("family") == settings["font_family"], (
            f"font.family {font.get('family')!r} != {settings['font_family']!r}"
        )
    if "font_size" in settings:
        assert font.get("size") == settings["font_size"], (
            f"font.size {font.get('size')!r} != {settings['font_size']!r}"
        )


def assert_no_explicit_style_clobbered(before: dict, after: dict) -> None:
    """Fill-only invariant (§4.2): restyling must never overwrite explicit marker sizes.

    For every trace that had an explicit ``marker.size`` before restyling, the
    value must be identical after.
    """
    b_data = before.get("data", [])
    a_data = after.get("data", [])
    for i, (bt, at) in enumerate(zip(b_data, a_data)):
        bm = (bt.get("marker") or {}).get("size")
        am = (at.get("marker") or {}).get("size")
        if bm is not None:
            assert bm == am, (
                f"trace[{i}] marker.size was {bm!r} before restyle, {am!r} after"
            )


# ── Script correctness ────────────────────────────────────────────────────────

_BUILTIN_NAMES = frozenset(dir(builtins))


def assert_names_bound_before_use(source: str) -> None:
    """AST-walk: every Name load is bound by a prior assignment, import, or builtin.

    This is a lightweight, single-pass, non-scoping check — good enough to catch
    the most common render/provenance bugs (undefined variable names in generated
    scripts) without a full data-flow analysis.
    """
    tree = ast.parse(source)
    bound: set[str] = set(_BUILTIN_NAMES)

    for node in ast.walk(tree):
        # Collect names that are bound (assigned / imported).
        if isinstance(node, (ast.Import,)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound.add(target.id)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                bound.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)

    # Second pass: check every Load.
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            assert node.id in bound, (
                f"Name {node.id!r} loaded at line {node.lineno} but never bound"
            )


def assert_runs(source: str, namespace: dict | None = None) -> dict:
    """exec *source* in a fresh namespace; return the namespace.

    Raises ``AssertionError`` wrapping any exception raised during exec so the
    test failure message is clear.
    """
    ns: dict = dict(namespace or {})
    try:
        exec(compile(source, "<test_script>", "exec"), ns)  # noqa: S102
    except Exception as exc:
        raise AssertionError(f"Script raised {type(exc).__name__}: {exc}\n\n{source}") from exc
    return ns


# ── AG Grid content ───────────────────────────────────────────────────────────

def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    return isinstance(value, str) and not value.strip()


def assert_grid_shows(col_defs: list[dict], rows: list[dict], source) -> None:
    """Assert an AG Grid given *col_defs* and *rows* displays the *source* data.

    *rows* are the records the grid receives (first block, source order).
    *source* is the DataFrame (or list of records) the callback read.

    Checks:
    - columnDefs and rows are not empty
    - every ``field`` in columnDefs is a key in every row
    - every source column that has a value in the served rows is a columnDefs
      field, and at least one row carries a non-empty value for it
    """
    import pandas as pd

    src = source if isinstance(source, pd.DataFrame) else pd.DataFrame(source)
    assert col_defs, "columnDefs is empty — the grid shows no columns"
    assert rows, "rowData is empty — the grid shows no rows"
    fields = [c["field"] for c in col_defs if "field" in c]
    for i, row in enumerate(rows):
        missing = [f for f in fields if f not in row]
        assert not missing, (
            f"row {i} lacks keys for columnDefs fields {missing} — those cells render blank "
            f"(row keys: {sorted(row)})"
        )
    src = src.iloc[: len(rows)]
    for col in src.columns:
        if all(_is_blank(v) for v in src[col].tolist()):
            continue
        assert col in fields, f"source column {col!r} has values but no columnDefs field shows it"
        assert any(not _is_blank(row.get(col)) for row in rows), (
            f"column {col!r} has values in the source but every grid row is blank for it"
        )


def assert_grid_cells_have_text(dash_duo, grid_id: str, *, timeout: float = 15) -> None:
    """Browser oracle: the grid *grid_id* renders cells, and every rendered
    column has at least one cell with text."""
    from dash.testing.wait import until

    selector = f"#{grid_id} .ag-cell"
    until(
        lambda: any(c.text.strip() for c in dash_duo.find_elements(selector)),
        timeout,
        msg=f"no .ag-cell with text appeared in #{grid_id}",
    )
    by_col: dict[str, list[str]] = {}
    for cell in dash_duo.find_elements(selector):
        by_col.setdefault(cell.get_attribute("col-id"), []).append(cell.text.strip())
    blank = sorted(col for col, texts in by_col.items() if not any(texts))
    assert not blank, f"#{grid_id}: rendered columns with no text in any cell: {blank}"
