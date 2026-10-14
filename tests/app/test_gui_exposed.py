"""Tests for the @gui_exposed pipeline.

Converts "flagging a function with @gui_exposed should be enough" from an
intention into a guarantee.  Tests at the build_form level so composite
widgets (Tuple slot/elem) are exercised the way a real form exercises them.

discover() is the ONLY source of parametrisation — when the registry widens,
coverage widens automatically with no test edit.

Expect failures on real callables: that is the purpose.  A failure in a
Structure/Surfaces module gets an xfail with a §15 pointer; elsewhere it is
a defect to fix.
"""
from __future__ import annotations

import inspect

import pytest


# ── Discovery ────────────────────────────────────────────────────────────────

def discover() -> list[tuple[str, str, object]]:
    """Every currently-registered @gui_exposed callable (via GUI_REGISTRY).

    Returns list of (short_owner, fn_name, fn) for every entry in the registry.
    Coverage widens automatically as new @gui_exposed callables are added to
    any module in discovery.GUI_MODULES.
    """
    from cryocat.app import discovery

    results: list[tuple[str, str, object]] = []
    for entry in discovery.entries():
        short_owner = entry.owner.split(".")[-1]  # "Motl", "geom", "cryowedge"
        results.append((short_owner, entry.fn.__name__, entry.fn))
    return results


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_controls(rows) -> tuple[list[dict], list]:
    """Walk a form's component tree and collect (id, value) pairs."""
    from tests.app.conftest import collect_id_paths

    ids: list[dict] = []
    values: list = []

    def _walk(node):
        if node is None:
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                _walk(child)
            return
        nid = getattr(node, "id", None)
        if isinstance(nid, dict) and "param" in nid and "tag" in nid:
            ids.append(nid)
            val = getattr(node, "value", None)
            values.append(val)
        _walk(getattr(node, "children", None))

    for row in rows:
        _walk(row)
    return ids, values


# ── Round-trip exemptions ────────────────────────────────────────────────────

# (owner, param_name) → reason.  A default that cannot round-trip must be
# listed here with a reason.  The list may shrink, never grow.
ROUND_TRIP_EXEMPT: dict[tuple[str, str], str] = {
    # numpy array defaults render as placeholder None — cannot round-trip
}


# ── Parametrised test ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "owner, name, fn",
    discover(),
    ids=lambda x: str(x) if not callable(x) else "",
)
def test_form_pipeline(owner: str, name: str, fn) -> None:
    from cryocat.app.formgen import build_form, WIDGET_FACTORIES
    from cryocat.app.apputils import generate_kwargs
    from cryocat.app.pool import PoolState
    from cryocat.utils.classutils import resolve_param_type, TYPE_HANDLERS

    # ── 1. Build the form ─────────────────────────────────────────────────
    rows = build_form(fn)

    # ── 2. Extract controls ───────────────────────────────────────────────
    ids, values = _extract_controls(rows)

    # ── 3. Every resolvable annotation maps to a known widget ─────────────
    try:
        hints = {}
        import typing
        try:
            hints = typing.get_type_hints(fn)
        except Exception:
            pass
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        pytest.skip("Cannot introspect signature")

    gui = getattr(fn, "_gui", {})
    hide = set(gui.get("hide", ())) | {"self"}

    for pname, param in sig.parameters.items():
        if pname in hide:
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        ann = hints.get(pname, param.annotation)
        tag, extra = resolve_param_type(ann)
        assert tag in TYPE_HANDLERS, (
            f"{owner}.{name}: param {pname!r} resolved to unknown tag {tag!r}"
        )
        assert TYPE_HANDLERS[tag]["widget"] in WIDGET_FACTORIES, (
            f"{owner}.{name}: tag {tag!r} maps to unknown widget "
            f"{TYPE_HANDLERS[tag]['widget']!r}"
        )

    # ── 4. One control per surfaced parameter ─────────────────────────────
    # ids may have multiple entries for Tuple (one per slot); deduplicate by param.
    surfaced_params = {cid["param"] for cid in ids}
    expected_params = set()
    for pname, param in sig.parameters.items():
        if pname in hide:
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        expected_params.add(pname)

    missing = expected_params - surfaced_params
    assert not missing, (
        f"{owner}.{name}: parameters {missing} not rendered in form"
    )

    # ── 5. Round-trip fidelity on defaults ────────────────────────────────
    if ids:
        kwargs = generate_kwargs(ids, values, PoolState.empty())

    for pname, param in sig.parameters.items():
        if pname in hide:
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.default is inspect.Parameter.empty:
            # Required parameter — must appear in kwargs (value may be None).
            if ids:
                assert pname in kwargs, (
                    f"{owner}.{name}: required param {pname!r} missing from kwargs"
                )
            continue
        default = param.default
        if default is None:
            continue  # optional with None default — value is None, trivially matches

        exempt_key = (name, pname)
        if exempt_key in ROUND_TRIP_EXEMPT:
            pytest.xfail(reason=ROUND_TRIP_EXEMPT[exempt_key])

        if not ids:
            continue
        got = kwargs.get(pname)
        # Type-coerce the default for comparison (strings, bools, numbers).
        if isinstance(default, bool):
            assert got is True or got is False, (
                f"{owner}.{name}: param {pname!r} default {default!r} did not round-trip; got {got!r}"
            )
            assert got == default, (
                f"{owner}.{name}: param {pname!r} round-trip mismatch: {default!r} → {got!r}"
            )
        elif isinstance(default, (int, float)):
            assert got == pytest.approx(default), (
                f"{owner}.{name}: param {pname!r} round-trip mismatch: {default!r} → {got!r}"
            )
        elif isinstance(default, str):
            assert got == default, (
                f"{owner}.{name}: param {pname!r} round-trip mismatch: {default!r} → {got!r}"
            )

    # ── 6. Metadata shape ─────────────────────────────────────────────────
    gui = getattr(fn, "_gui", getattr(getattr(fn, "__func__", fn), "_gui", None))
    if gui is not None:
        assert gui.get("label"), (
            f"{owner}.{name}: _gui has empty or missing 'label'"
        )
        # Accepted categories — expand when a genuinely new grouping is added.
        # A typo here silently drops an operation from its dropdown group.
        known_categories = {
            "Geometry", "Cleaning", "Classification", "Classes",
            "Alignment", "Analysis", "Symmetry", "Utility",
            "Combine", "Subsets",
            "builder", None,
            "Mask",       # legacy value still in _gui for any builder-tier functions
            "reader",     # file-reading callables surfaced in the data pool
            "surface-op",   # GuiCategory.SURFACE_OP  — mesh/OPC operations on passembly page
            "surface-load", # GuiCategory.SURFACE_LOAD — loaders that create a surface from disk/motl
        }
        cat = gui.get("category")
        assert cat in known_categories, (
            f"{owner}.{name}: _gui 'category' {cat!r} not in known categories "
            f"{known_categories}. Add it or fix the typo."
        )
        motls_spec = gui.get("motls")
        if motls_spec is not None:
            assert "arity" in motls_spec, (
                f"{owner}.{name}: motls spec missing 'arity': {motls_spec!r}"
            )
            assert motls_spec["arity"] in ("pair", "list"), (
                f"{owner}.{name}: motls 'arity' must be 'pair' or 'list', "
                f"got {motls_spec['arity']!r}"
            )
