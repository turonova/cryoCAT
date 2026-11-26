"""Tests for cryocat.app.discovery.

Covers load_registry(), query functions, module-import guarantee, and the
end-to-end guarantee that a new @gui_exposed callable flows through discovery
without touching any app code.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_registry():
    """Return a snapshot of GUI_REGISTRY after load_registry() runs."""
    from cryocat.app import discovery
    from cryocat.utils.classutils import GUI_REGISTRY
    discovery.load_registry()
    return dict(GUI_REGISTRY)


# ---------------------------------------------------------------------------
# load_registry
# ---------------------------------------------------------------------------

def test_load_registry_returns_non_empty_dict():
    """load_registry() returns a non-empty snapshot after importing GUI_MODULES."""
    snapshot = _fresh_registry()
    assert isinstance(snapshot, dict)
    assert len(snapshot) > 0


def test_load_registry_includes_motl_methods():
    """GUI_REGISTRY contains entries whose owner is the Motl class."""
    snapshot = _fresh_registry()
    motl_entries = [e for e in snapshot.values() if "Motl" in e.owner]
    assert len(motl_entries) >= 10, (
        f"Expected ≥10 Motl entries, got {len(motl_entries)}"
    )


def test_load_registry_includes_generate_angles():
    """generate_angles (geom.py) must be registered as a builder."""
    snapshot = _fresh_registry()
    assert "geom.generate_angles" in snapshot
    from cryocat.utils.classutils import GuiCategory
    assert snapshot["geom.generate_angles"].category == GuiCategory.BUILDER


def test_load_registry_includes_generate_wedge_mask():
    """generate_wedge_mask (cryowedge.py) must be a standalone builder."""
    snapshot = _fresh_registry()
    assert "cryowedge.generate_wedge_mask" in snapshot
    entry = snapshot["cryowedge.generate_wedge_mask"]
    from cryocat.utils.classutils import GuiCategory
    assert entry.category == GuiCategory.BUILDER
    assert entry.standalone is True


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def test_single_motl_ops_all_motl_op_with_no_motls():
    from cryocat.app import discovery
    from cryocat.utils.classutils import GuiCategory
    ops = discovery.single_motl_ops()
    assert len(ops) >= 1
    for e in ops:
        assert e.category == GuiCategory.MOTL_OP
        assert e.motls is None


def test_multi_motl_ops_all_have_motls_spec():
    from cryocat.app import discovery
    from cryocat.utils.classutils import GuiCategory
    ops = discovery.multi_motl_ops()
    assert len(ops) >= 1
    for e in ops:
        assert e.category == GuiCategory.MOTL_OP
        assert e.motls is not None
        assert "arity" in e.motls


def test_standalone_builders_all_standalone():
    from cryocat.app import discovery
    from cryocat.utils.classutils import GuiCategory
    builders = discovery.standalone_builders()
    assert len(builders) >= 1
    for e in builders:
        assert e.category == GuiCategory.BUILDER
        assert e.standalone is True


def test_entries_sorted_by_group_order_label():
    from cryocat.app import discovery
    all_entries = discovery.entries()
    sort_keys = [(e.group or "\x7f", e.order, e.label) for e in all_entries]
    assert sort_keys == sorted(sort_keys)


def test_entries_filter_by_category():
    from cryocat.app import discovery
    from cryocat.utils.classutils import GuiCategory
    builders = discovery.entries(category=GuiCategory.BUILDER)
    for e in builders:
        assert e.category == GuiCategory.BUILDER


def test_get_known_key():
    from cryocat.app import discovery
    entry = discovery.get("geom.generate_angles")
    assert entry.label
    assert callable(entry.fn)


def test_get_unknown_key_raises():
    from cryocat.app import discovery
    with pytest.raises(KeyError, match="no_such_key"):
        discovery.get("no_such_key")


def test_gui_ready_known_entry():
    from cryocat.app import discovery
    entry = discovery.get("geom.generate_angles")
    ok, reason = discovery.gui_ready(entry)
    assert ok, f"gui_ready failed: {reason}"


# ---------------------------------------------------------------------------
# Acceptance — new @gui_exposed callable flows end-to-end
# ---------------------------------------------------------------------------

def test_r8_new_callable_appears_in_registry_and_form():
    """Decoration alone is sufficient for full pipeline coverage.

    Steps (automated):
    1. Decorate a trivial function with @gui_exposed.
    2. The entry appears in GUI_REGISTRY immediately.
    3. build_form(entry) produces controls without errors.
    4. generate_kwargs round-trips the default value.
    5. gui_ready reports the entry as renderable.
    """
    from cryocat.utils.classutils import gui_exposed, GUI_REGISTRY, GuiCategory
    from cryocat.app import discovery

    before = set(GUI_REGISTRY.keys())

    @gui_exposed(label="Drop low score", category="Cleaning")
    def _drop_low_score_r8(self, threshold: float = 0.5):
        """Drop subtomograms with score below threshold."""

    after = set(GUI_REGISTRY.keys())
    new_keys = after - before
    assert len(new_keys) == 1
    key = next(iter(new_keys))
    entry = GUI_REGISTRY[key]

    # 2 — correct tier and metadata
    assert entry.label == "Drop low score"
    assert entry.category == GuiCategory.MOTL_OP
    assert entry.group == "Cleaning"

    # 3 — build_form succeeds
    from cryocat.app.formgen import build_form
    rows = build_form(entry)
    assert len(rows) >= 1

    # 4 — generate_kwargs round-trips the default
    from cryocat.app.apputils import generate_kwargs
    from cryocat.app.pool import PoolState
    from tests.app.test_gui_exposed import _extract_controls
    ids_list, values = _extract_controls(rows)
    if ids_list:
        kwargs = generate_kwargs(ids_list, values, PoolState.empty())
        assert "threshold" in kwargs
        assert kwargs["threshold"] == pytest.approx(0.5)

    # 5 — gui_ready
    ok, reason = discovery.gui_ready(entry)
    assert ok, f"gui_ready reported not renderable: {reason}"


# ---------------------------------------------------------------------------
# Every registered entry builds its form
# ---------------------------------------------------------------------------

def _registry_entries():
    from cryocat.app import discovery
    return discovery.entries()


def _resolve_owner(owner: str):
    """Import the module part of *owner* and walk the remaining attribute names."""
    import importlib
    parts = owner.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for name in parts[cut:]:
            obj = getattr(obj, name)
        return obj
    raise ImportError(f"no importable module in {owner!r}")


def _form_params(rows) -> set[str]:
    """Names of the parameters that received a control in a built form."""
    found: set[str] = set()

    def _walk(node):
        if isinstance(node, (list, tuple)):
            for child in node:
                _walk(child)
            return
        if node is None or isinstance(node, (str, int, float)):
            return
        nid = getattr(node, "id", None)
        if isinstance(nid, dict) and "param" in nid:
            found.add(nid["param"])
        _walk(getattr(node, "children", None))

    _walk(rows)
    return found


@pytest.mark.parametrize("entry", _registry_entries(), ids=lambda e: e.key)
def test_registry_entry_builds_its_form(entry):
    """The entry points to an existing callable, its form builds, and every parameter gets a widget."""
    import inspect
    import warnings
    from cryocat.app.formgen import build_form

    owner = _resolve_owner(entry.owner)
    attr = inspect.getattr_static(owner, entry.fn.__name__, None)
    assert attr is not None, f"{entry.owner} has no attribute {entry.fn.__name__!r}"
    target = getattr(attr, "__func__", attr)
    assert inspect.unwrap(target) is inspect.unwrap(entry.fn), (
        f"{entry.owner}.{entry.fn.__name__} is not the registered callable"
    )

    # Parameters supplied outside the form: the path field and the motl pickers.
    supplied = {entry.path_arg} - {""}
    if entry.motls:
        supplied |= {"motl1", "motl2"} if entry.motls.get("arity") == "pair" else {entry.motls["param"]}

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*get_type_hints failed.*")
        rows = build_form(entry, exclude=supplied)

    hidden = set(entry.hide) | supplied | {"self", "cls"}
    expected = {
        name for name, p in inspect.signature(entry.fn).parameters.items()
        if name not in hidden
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    missing = expected - _form_params(rows)
    assert not missing, f"{entry.key}: no widget for {sorted(missing)}"
