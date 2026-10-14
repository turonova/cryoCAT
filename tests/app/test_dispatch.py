"""Guard test: no module under app/suite or app/tango calls mutating entry points
directly (write_out, to_csv, pickle.dump, save_motl, save_output).

All such calls must go through run_operation / run_operation_to_pool.

Files still pending routing are listed in _PENDING.
Remove an entry once the routing is in place.
"""
from __future__ import annotations

import ast
import pathlib

import pytest


_APP_ROOT = pathlib.Path(__file__).parent.parent.parent / "cryocat" / "app"
_SUITE_ROOT = _APP_ROOT / "suite"
_TANGO_ROOT = _APP_ROOT / "tango"

# Files pending routing — remove each entry once the corresponding task is done.
_PENDING: frozenset[str] = frozenset()

# Attribute-call names that must not appear as direct calls.
_BANNED_ATTRS = frozenset({"write_out", "to_csv"})
# pickle.dump — only flag when receiver is the name "pickle".
_BANNED_PICKLE_ATTRS = frozenset({"dump"})
# Free-function names that must not be called directly.
_BANNED_NAMES = frozenset({"save_motl", "save_output"})


def _collect_files(*roots: pathlib.Path) -> list[pathlib.Path]:
    files = []
    for root in roots:
        files.extend(root.rglob("*.py"))
    return sorted(files)


def _violations(src_path: pathlib.Path) -> list[str]:
    """Return a list of human-readable violation strings found in *src_path*."""
    try:
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"SyntaxError: {exc}"]

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func

        # attr call: obj.write_out(...) or obj.to_csv(...)
        if isinstance(fn, ast.Attribute) and fn.attr in _BANNED_ATTRS:
            found.append(f"line {node.lineno}: .{fn.attr}()")

        # pickle.dump(...)
        if (
            isinstance(fn, ast.Attribute)
            and fn.attr in _BANNED_PICKLE_ATTRS
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "pickle"
        ):
            found.append(f"line {node.lineno}: pickle.dump()")

        # free-function call: save_motl(...) or save_output(...)
        if isinstance(fn, ast.Name) and fn.id in _BANNED_NAMES:
            found.append(f"line {node.lineno}: {fn.id}()")

    return found


def _all_checked_files() -> list[pathlib.Path]:
    return [
        p for p in _collect_files(_SUITE_ROOT, _TANGO_ROOT)
        if "__pycache__" not in p.parts and p.name not in _PENDING
    ]


@pytest.mark.parametrize("src_path", _all_checked_files(), ids=lambda p: p.name)
def test_no_direct_mutating_calls(src_path: pathlib.Path):
    """Each checked file must contain zero direct mutating entry-point calls."""
    viols = _violations(src_path)
    assert not viols, (
        f"{src_path.relative_to(_APP_ROOT.parent.parent)} contains direct "
        f"mutating entry-point call(s) — route through run_operation or "
        f"run_operation_to_pool:\n"
        + "\n".join(f"  {v}" for v in viols)
    )


def test_pending_names_exist():
    """Sanity: every name in _PENDING must correspond to a real file."""
    if not _PENDING:
        return
    missing = [
        name for name in _PENDING
        if not any(
            p.name == name
            for p in _collect_files(_SUITE_ROOT, _TANGO_ROOT)
        )
    ]
    assert not missing, f"_PENDING names non-existent files: {missing}"


def test_pending_files_still_have_violations():
    """Verify that pending files DO have violations so removing them from
    _PENDING is meaningful (catches accidental early-removal)."""
    for name in _PENDING:
        candidates = [
            p for p in _collect_files(_SUITE_ROOT, _TANGO_ROOT)
            if p.name == name
        ]
        for p in candidates:
            viols = _violations(p)
            if viols:
                break
        else:
            pytest.fail(
                f"{name} is in _PENDING but has no detected violations — "
                "remove it from _PENDING."
            )
