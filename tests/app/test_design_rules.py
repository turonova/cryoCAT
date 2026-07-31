"""D2 — Design-compliance enforcement tests.

AST-based enforcement of GUI_DESIGN_RULES.md §1–§9, scoped to
cryocat/app/suite/** and cryocat/app/components/*.

These tests are expected to fail initially (D2 by design).  Each rule's
violations are collected and printed with file+line so they serve as the D3–D10
worklist.  Exemptions start empty; every added entry requires a written reason
and may only shrink.

Run::

    pytest tests/app/test_design_rules.py -v

to get the full violation listing before starting D3.
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys
import types
from collections.abc import Generator

import pytest

# ── Scope ─────────────────────────────────────────────────────────────────────

_REPO = pathlib.Path(__file__).parent.parent.parent
_APP = _REPO / "cryocat" / "app"
_SUITE = _APP / "suite"
_COMPONENTS = _APP / "components"
_STYLES_PY = _APP / "styles.py"
_FORMGEN_PY = _APP / "formgen.py"
_PATHFIELD_PY = _COMPONENTS / "pathfield.py"


def _suite_files() -> list[pathlib.Path]:
    """All Python files in scope: suite/** + components/*.py."""
    files: list[pathlib.Path] = list(_SUITE.rglob("*.py")) + list(_COMPONENTS.glob("*.py"))
    return sorted(files)


def _files_excl(*excl: pathlib.Path) -> list[pathlib.Path]:
    excl_set = set(excl)
    return [f for f in _suite_files() if f not in excl_set]


def _rel(path: pathlib.Path) -> str:
    return str(path.relative_to(_REPO))


# ── AST helpers ───────────────────────────────────────────────────────────────

def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_str(node: ast.expr, value: str | None = None) -> bool:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return False
    return value is None or node.value == value


def _dict_pairs(node: ast.Dict) -> Generator[tuple[ast.expr, ast.expr], None, None]:
    for k, v in zip(node.keys, node.values):
        if k is not None:
            yield k, v


# ─────────────────────────────────────────────────────────────────────────────
# §1 — No fontSize literal outside styles.py
# ─────────────────────────────────────────────────────────────────────────────

_S1_EXEMPT: dict[tuple[str, int], str] = {
    # (rel_path, line): reason
}


def _s1_collect() -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    for path in _files_excl(_STYLES_PY):
        tree = _parse(path)
        rel = _rel(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for k, v in _dict_pairs(node):
                if _is_str(k, "fontSize") and isinstance(v, ast.Constant):
                    key = (rel, k.lineno)
                    if key not in _S1_EXEMPT:
                        out.append((rel, k.lineno, f'fontSize="{v.value}"'))
    return out


def test_s1_no_fontsize_literal() -> None:
    """§1: no fontSize literal in any file outside styles.py."""
    violations = _s1_collect()
    if violations:
        lines = [f"  {f}:{ln}  {d}" for f, ln, d in violations]
        pytest.fail(f"§1 fontSize literal ({len(violations)}):\n" + "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# §3 — No marginBottom / marginTop literal in a form-row-shaped style dict
# ─────────────────────────────────────────────────────────────────────────────
# A "form-row-shaped" dict is one that contains both "display"="flex" and
# "marginBottom" or "marginTop" — i.e. a hand-rolled row.  Spacing tokens must
# come from styles.py, not be typed per call site.

_S3_EXEMPT: dict[tuple[str, int], str] = {
    # (rel_path, line): reason
}

_S3_MARGIN_KEYS = {"marginBottom", "marginTop"}


def _s3_collect() -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    for path in _files_excl(_STYLES_PY):
        tree = _parse(path)
        rel = _rel(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys_vals = list(_dict_pairs(node))
            key_strs = {k.value for k, _ in keys_vals if _is_str(k)}
            # Only flag if this looks like a flex row (has "display"="flex")
            has_flex = False
            for k, v in keys_vals:
                if _is_str(k, "display") and _is_str(v, "flex"):
                    has_flex = True
                    break
            if not has_flex:
                continue
            for k, v in keys_vals:
                if _is_str(k) and k.value in _S3_MARGIN_KEYS and isinstance(v, ast.Constant):
                    key = (rel, k.lineno)
                    if key not in _S3_EXEMPT:
                        out.append((rel, k.lineno, f'{k.value}="{v.value}" in flex row dict'))
    return out


def test_s3_no_margin_literal_in_flex_row() -> None:
    """§3: no marginBottom/marginTop literal in a flex-row style dict outside styles.py."""
    violations = _s3_collect()
    if violations:
        lines = [f"  {f}:{ln}  {d}" for f, ln, d in violations]
        pytest.fail(f"§3 margin literal in flex row ({len(violations)}):\n" + "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# §4 — No direct dcc.Dropdown() outside formgen.py
# ─────────────────────────────────────────────────────────────────────────────

_S4_EXEMPT: dict[tuple[str, int], str] = {
    # (rel_path, line): reason
}


def _is_dcc_dropdown(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Dropdown"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "dcc"
    )


def _s4_collect() -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    for path in _files_excl(_FORMGEN_PY):
        tree = _parse(path)
        rel = _rel(path)
        for node in ast.walk(tree):
            if _is_dcc_dropdown(node):
                key = (rel, node.lineno)
                if key not in _S4_EXEMPT:
                    out.append((rel, node.lineno, "dcc.Dropdown("))
    return out


def test_s4_no_direct_dropdown() -> None:
    """§4: no direct dcc.Dropdown() call outside formgen.py."""
    violations = _s4_collect()
    if violations:
        lines = [f"  {f}:{ln}  {d}" for f, ln, d in violations]
        pytest.fail(f"§4 direct dcc.Dropdown() ({len(violations)}):\n" + "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# §5 — No marginTop / paddingTop on a checkbox or switch
# ─────────────────────────────────────────────────────────────────────────────

_S5_EXEMPT: dict[tuple[str, int], str] = {
    # (rel_path, line): reason
}

_CHECKBOX_ATTRS = {"Checkbox", "Checklist", "Switch"}
_S5_NUDGE_KEYS = {"marginTop", "paddingTop"}


def _s5_collect() -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    for path in _suite_files():
        tree = _parse(path)
        rel = _rel(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr in _CHECKBOX_ATTRS):
                continue
            for kw in node.keywords:
                if kw.arg != "style" or not isinstance(kw.value, ast.Dict):
                    continue
                for k, _ in _dict_pairs(kw.value):
                    if _is_str(k) and k.value in _S5_NUDGE_KEYS:
                        key = (rel, k.lineno)
                        if key not in _S5_EXEMPT:
                            out.append((rel, k.lineno, f"{k.value} on {func.attr}"))
    return out


def test_s5_no_nudge_on_checkbox() -> None:
    """§5: no marginTop/paddingTop hand-nudge on a checkbox or switch."""
    violations = _s5_collect()
    if violations:
        lines = [f"  {f}:{ln}  {d}" for f, ln, d in violations]
        pytest.fail(f"§5 vertical nudge on checkbox/switch ({len(violations)}):\n" + "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# §6 — No color="success" and no green hex on buttons
# ─────────────────────────────────────────────────────────────────────────────

_S6_EXEMPT: dict[tuple[str, int], str] = {
    # (rel_path, line): reason
}

_GREEN_HEX = re.compile(r"^#?([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$")
_GREEN_STYLE_KEYS = {"backgroundColor", "background", "borderColor"}


def _is_green(s: str) -> bool:
    m = _GREEN_HEX.match(s.strip())
    if not m:
        return False
    r, g, b = int(m.group(1), 16), int(m.group(2), 16), int(m.group(3), 16)
    return g > r and g > b and g > 80


def _s6_collect() -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    for path in _suite_files():
        tree = _parse(path)
        rel = _rel(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "Button"):
                continue
            for kw in node.keywords:
                if kw.arg == "color" and _is_str(kw.value, "success"):
                    key = (rel, kw.value.lineno)
                    if key not in _S6_EXEMPT:
                        out.append((rel, kw.value.lineno, 'color="success"'))
                elif kw.arg == "style" and isinstance(kw.value, ast.Dict):
                    for k, v in _dict_pairs(kw.value):
                        if (
                            _is_str(k) and k.value in _GREEN_STYLE_KEYS
                            and isinstance(v, ast.Constant)
                            and isinstance(v.value, str)
                            and _is_green(v.value)
                        ):
                            key = (rel, k.lineno)
                            if key not in _S6_EXEMPT:
                                out.append((rel, k.lineno, f'{k.value}="{v.value}" (green)'))
    return out


def test_s6_no_green_buttons() -> None:
    """§6: no color='success' and no green hex on Button components."""
    violations = _s6_collect()
    if violations:
        lines = [f"  {f}:{ln}  {d}" for f, ln, d in violations]
        pytest.fail(f"§6 green button ({len(violations)}):\n" + "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# §7 — Every visplot palette appears in paletteloader presets
# ─────────────────────────────────────────────────────────────────────────────

def test_s7_palette_coverage() -> None:
    """§7: every palette/scale registered in visplot is listed in paletteloader."""
    sys.modules.setdefault("emfile", types.ModuleType("emfile"))

    from cryocat.analysis.visplot import CUSTOM_PALETTES, CUSTOM_SCALES
    from cryocat.app.components.paletteloader import _DISCRETE_PRESETS, _CONTINUOUS_PRESETS

    discrete_lower = {p.lower() for p in _DISCRETE_PRESETS}
    continuous_lower = {p.lower() for p in _CONTINUOUS_PRESETS}

    missing_disc = [n for n in CUSTOM_PALETTES if n.lower() not in discrete_lower]
    missing_cont = [n for n in CUSTOM_SCALES if n.lower() not in continuous_lower]

    errors: list[str] = []
    if missing_disc:
        errors.append(f"  Missing from _DISCRETE_PRESETS:  {sorted(missing_disc)}")
    if missing_cont:
        errors.append(f"  Missing from _CONTINUOUS_PRESETS: {sorted(missing_cont)}")

    if errors:
        pytest.fail("§7 visplot palette not in chooser:\n" + "\n".join(errors))


# ─────────────────────────────────────────────────────────────────────────────
# §8 — Every form_row() call provides a non-empty description
# ─────────────────────────────────────────────────────────────────────────────

_S8_EXEMPT: dict[tuple[str, int], str] = {
    # (rel_path, line): reason
}


def _s8_collect() -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    for path in _suite_files():
        tree = _parse(path)
        rel = _rel(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_fr = (isinstance(func, ast.Name) and func.id == "form_row") or (
                isinstance(func, ast.Attribute) and func.attr == "form_row"
            )
            if not is_fr:
                continue
            # form_row(name, widget, description, ...) — description is arg[2]
            desc: ast.expr | None = None
            if len(node.args) >= 3:
                desc = node.args[2]
            else:
                for kw in node.keywords:
                    if kw.arg == "description":
                        desc = kw.value
                        break
            if desc is None:
                key = (rel, node.lineno)
                if key not in _S8_EXEMPT:
                    out.append((rel, node.lineno, "form_row() missing description="))
            elif isinstance(desc, ast.Constant) and not desc.value:
                key = (rel, node.lineno)
                if key not in _S8_EXEMPT:
                    out.append((rel, node.lineno, "form_row() description is empty string"))
    return out


def test_s8_form_row_description() -> None:
    """§8: every form_row() call has a non-empty description."""
    violations = _s8_collect()
    if violations:
        lines = [f"  {f}:{ln}  {d}" for f, ln, d in violations]
        pytest.fail(f"§8 form_row missing description ({len(violations)}):\n" + "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# §9 — No bare path dbc.Input in suite pages; all via get_path_field / get_file_loader
# ─────────────────────────────────────────────────────────────────────────────
# A "bare path input" is a dbc.Input(type="text") in a suite page whose id or
# placeholder suggests it is for a file path.  Correct usage wraps path inputs
# via get_path_field() from components/pathfield.py.

_S9_EXEMPT: dict[tuple[str, int], str] = {
    # (rel_path, line): reason
}

_PATH_HINT = re.compile(r"path|file|folder|dir", re.IGNORECASE)


def _s9_collect() -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    # Only scan suite pages, not the canonical pathfield.py implementation or fileloader
    excl = {_PATHFIELD_PY, _COMPONENTS / "fileloader.py"}
    for path in _files_excl(*excl):
        if not str(path).startswith(str(_SUITE)):
            continue  # only suite pages, not all components
        tree = _parse(path)
        rel = _rel(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "Input"):
                continue
            # Gather type= and id= kwarg values
            type_val: str | None = None
            id_val: str | None = None
            placeholder_val: str | None = None
            for kw in node.keywords:
                if kw.arg == "type" and isinstance(kw.value, ast.Constant):
                    type_val = kw.value.value
                elif kw.arg == "id" and isinstance(kw.value, ast.Constant):
                    id_val = str(kw.value.value)
                elif kw.arg == "placeholder" and isinstance(kw.value, ast.Constant):
                    placeholder_val = str(kw.value.value)
            if type_val not in (None, "text", "email", "password"):
                continue  # not a text input
            # Flag if the id or placeholder signals a path
            hint = " ".join(filter(None, [id_val, placeholder_val]))
            if _PATH_HINT.search(hint):
                key = (rel, node.lineno)
                if key not in _S9_EXEMPT:
                    out.append((rel, node.lineno, f"bare path Input (id={id_val!r})"))
    return out


def test_s9_no_bare_path_input() -> None:
    """§9: no bare path dbc.Input in suite pages; all via get_path_field or get_file_loader."""
    violations = _s9_collect()
    if violations:
        lines = [f"  {f}:{ln}  {d}" for f, ln, d in violations]
        pytest.fail(f"§9 bare path Input in suite ({len(violations)}):\n" + "\n".join(lines))
