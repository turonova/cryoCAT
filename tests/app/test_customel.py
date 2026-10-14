"""Tests for customel component builders and dcc.Graph wrapping enforcement.

Covers:
- extract_style: merges style override, no-override, immutability.
- InlineLabeledDropdown: className, style kwarg, label className.
- InlineInputForm: className.
- AST-based enforcement that every dcc.Graph() is the graph_component arg
  of a customel_graph() call across cryocat/app/.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from cryocat.app.components.customel import (
    InlineLabeledDropdown,
    InlineInputForm,
    extract_style,
)


# ===========================================================================
# extract_style
# ===========================================================================

def test_extract_style_merges_override():
    default = {"flex": "1", "padding": "0"}
    kwargs = {"style": {"padding": "4px"}, "multi": True}
    style, cleaned = extract_style(default, kwargs)
    assert style["padding"] == "4px"
    assert style["flex"] == "1"
    assert "style" not in cleaned
    assert cleaned["multi"] is True


def test_extract_style_no_override():
    default = {"flex": "1"}
    kwargs = {"multi": True}
    style, cleaned = extract_style(default, kwargs)
    assert style == {"flex": "1"}
    assert cleaned == {"multi": True}


def test_extract_style_does_not_mutate_inputs():
    default = {"a": "1"}
    kwargs = {"style": {"a": "2"}}
    extract_style(default, kwargs)
    assert default == {"a": "1"}
    assert kwargs == {"style": {"a": "2"}}


# ===========================================================================
# InlineLabeledDropdown
# ===========================================================================

def test_inline_dropdown_className_not_in_dropdown_kwargs():
    """className must be popped, not left in **dropdown_kwargs passed to Dropdown."""
    div = InlineLabeledDropdown("test-dd", "Label", className="my-class")
    assert div.className == "my-class"
    dropdown = div.children[1]
    assert not getattr(dropdown, "className", None)


def test_inline_dropdown_default_className():
    div = InlineLabeledDropdown("test-dd", "Label")
    assert div.className == "flex"


def test_inline_dropdown_style_kwarg_does_not_raise():
    """style= must not raise TypeError 'multiple values for keyword argument'."""
    div = InlineLabeledDropdown("test-dd2", "Label", style={"color": "red"})
    dropdown = div.children[1]
    assert dropdown.style.get("color") == "red"


def test_inline_dropdown_style_merges_with_defaults():
    div = InlineLabeledDropdown("test-dd3", "Label", style={"flex": "2"})
    dropdown = div.children[1]
    assert dropdown.style["flex"] == "2"


def test_inline_dropdown_label_className_is_plain_string():
    """The label className is a plain string constant (no runtime f-string overhead)."""
    div = InlineLabeledDropdown("test-dd4", "Label")
    label = div.children[0]
    assert label.className == "label-dark mb-0 me-2"


# ===========================================================================
# InlineInputForm
# ===========================================================================

def test_inline_input_className_not_in_input_kwargs():
    """className must be popped from input_kwargs so it doesn't reach dbc.Input."""
    div = InlineInputForm("test-inp", "Label", className="inp-class")
    assert div.className == "inp-class"
    input_el = div.children[1]
    assert not getattr(input_el, "className", None)


def test_inline_input_default_className():
    div = InlineInputForm("test-inp2", "Label")
    assert div.className == "flex"


# ===========================================================================
# dcc.Graph wrapping enforcement
# ===========================================================================

_CRYOCAT_APP = pathlib.Path(__file__).parent.parent.parent / "cryocat" / "app"

_SKIP_FILES: dict[str, str] = {
    "suite/pages/pgraphs.py": (
        "plot editor's own graph — is the target for Send-to-editor; cannot wrap itself"
    ),
    "components/tomoview.py": (
        "tomo-viewer graphs are interactive particle-display input controls"
    ),
    "components/orientpicker.py": (
        "orientation-picker sphere is an interactive rotation input control"
    ),
    "suite/pages/ppana.py": (
        "wedge-mask XZ preview is a parameter-tuning control inside the wedge-builder modal"
    ),
}


def _is_dcc_graph_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Graph"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "dcc"
    )


def _is_customel_graph_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "customel_graph"
    )


def _wrapped_locations(tree: ast.AST) -> set[tuple[int, int]]:
    wrapped: set[tuple[int, int]] = set()
    for node in ast.walk(tree):
        if not _is_customel_graph_call(node):
            continue
        for arg in node.args:
            if _is_dcc_graph_call(arg):
                wrapped.add((arg.lineno, arg.col_offset))
    return wrapped


def _graph_has_styled_id(node: ast.Call) -> bool:
    for kw in node.keywords:
        if kw.arg != "id":
            continue
        if not isinstance(kw.value, ast.Dict):
            return False
        for k, v in zip(kw.value.keys, kw.value.values):
            if (isinstance(k, ast.Constant) and k.value == "type"
                    and isinstance(v, ast.Constant) and v.value == "styled-graph"):
                return True
    return False


def _violations_in(py_file: pathlib.Path) -> list[tuple[pathlib.Path, int, str]]:
    source = py_file.read_text(encoding="utf-8")
    if "dcc.Graph" not in source:
        return []
    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError:
        return []
    wrapped = _wrapped_locations(tree)
    violations: list[tuple[pathlib.Path, int, str]] = []
    for node in ast.walk(tree):
        if not _is_dcc_graph_call(node):
            continue
        loc = (node.lineno, node.col_offset)
        if loc not in wrapped:
            violations.append((py_file, node.lineno, "not wrapped in customel_graph"))
        elif not _graph_has_styled_id(node):
            violations.append((py_file, node.lineno,
                                'wrapped but missing {"type": "styled-graph", ...} id'))
    return violations


def test_all_dcc_graphs_wrapped_in_customel_graph() -> None:
    """Every dcc.Graph() in cryocat/app/ must be the graph_component arg of customel_graph().

    Files are skipped only when every graph in them is an interactive input
    control or a parameter-exploration utility. See _SKIP_FILES for the explicit
    list with per-file reasons.
    """
    violations: list[tuple[pathlib.Path, int, str]] = []

    for py_file in sorted(_CRYOCAT_APP.rglob("*.py")):
        rel = py_file.relative_to(_CRYOCAT_APP).as_posix()
        if rel in _SKIP_FILES:
            continue
        violations.extend(_violations_in(py_file))

    if violations:
        lines = [f"  {v[0]}:{v[1]}  ({v[2]})" for v in violations]
        pytest.fail(
            f"{len(violations)} dcc.Graph() construction(s) failed customel_graph coverage:\n"
            + "\n".join(lines)
        )
