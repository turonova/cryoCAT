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
