"""Tests for the Tango page symmetry controls.

Covers the pure helpers in ``cryocat/app/suite/pages/_tango_symmetry.py``
(GUI_CONVENTIONS §11.1: no Dash, no browser, no mocks) and the layout of the
symmetry widgets in ``ptango.layout`` (§11.2). The helpers translate the
sidebar widgets into the canonical ``(symm, kind)`` pair accepted by
``cryocat.analysis.tango.TwistDescriptor``.
"""
import numpy as np
import pytest

from cryocat.app.suite.pages._tango_symmetry import (
    SYMM_TYPE_OPTIONS,
    build_symm_kwargs,
    kind_control_state,
    kind_options_for_symm,
)
from tests.app.conftest import collect_ids


def _find_by_id(node, target_id):
    """Depth-first search of a Dash tree for the component with ``id == target_id``."""
    if node is None or isinstance(node, (str, int, float, bool)):
        return None
    if isinstance(node, (list, tuple)):
        for child in node:
            found = _find_by_id(child, target_id)
            if found is not None:
                return found
        return None
    if getattr(node, "id", None) == target_id:
        return node
    return _find_by_id(getattr(node, "children", None), target_id)


# ---------------------------------------------------------------------------
# kind_options_for_symm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symm_type, expected",
    [
        # O and I each have two solids; the first one is the default.
        ("O", ["octahedron", "cube"]),
        ("I", ["icosahedron", "dodecahedron"]),
        # T has only the tetrahedron, so there is nothing to choose.
        ("T", []),
        # No symmetry, cyclic symmetry and "no selection yet" offer no solids.
        ("C", []),
        ("None", []),
        (None, []),
    ],
)
def test_kind_options_for_symm(symm_type, expected):
    assert kind_options_for_symm(symm_type) == expected


def test_kind_options_returns_fresh_list():
    # Mutating the returned list must not corrupt the module-level table.
    opts = kind_options_for_symm("O")
    opts.append("bogus")
    assert kind_options_for_symm("O") == ["octahedron", "cube"]


# ---------------------------------------------------------------------------
# kind_control_state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symm_type, expected",
    [
        # Visible with the default solid preselected for O and I.
        ("O", (["octahedron", "cube"], "octahedron", True)),
        ("I", (["icosahedron", "dodecahedron"], "icosahedron", True)),
        # Hidden and cleared for everything else.
        ("T", ([], None, False)),
        ("C", ([], None, False)),
        ("None", ([], None, False)),
        (None, ([], None, False)),
    ],
)
def test_kind_control_state(symm_type, expected):
    assert kind_control_state(symm_type) == expected


# ---------------------------------------------------------------------------
# build_symm_kwargs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symm_type, c_value, kind_value, expected",
    [
        # "None" sentinel (and a missing selection) mean no symmetry at all;
        # stale widget values are ignored.
        ("None", 3, "cube", {"symm": None, "kind": None}),
        (None, None, None, {"symm": None, "kind": None}),
        # Cyclic: the order becomes the canonical "CN" string, kind stays None
        # even if the hidden kind dropdown still holds a value.
        ("C", 2, None, {"symm": "C2", "kind": None}),
        ("C", 5.0, "cube", {"symm": "C5", "kind": None}),
        # Tetrahedral: only one solid, kind is always None.
        ("T", 2, "cube", {"symm": "T", "kind": None}),
        # Octahedral / icosahedral: the chosen solid is passed through.
        ("O", 2, "cube", {"symm": "O", "kind": "cube"}),
        ("O", 2, "octahedron", {"symm": "O", "kind": "octahedron"}),
        ("I", 2, "dodecahedron", {"symm": "I", "kind": "dodecahedron"}),
        # No solid selected yet -> the default solid for that group.
        ("O", 2, None, {"symm": "O", "kind": "octahedron"}),
        ("I", 2, None, {"symm": "I", "kind": "icosahedron"}),
    ],
)
def test_build_symm_kwargs(symm_type, c_value, kind_value, expected):
    assert build_symm_kwargs(symm_type, c_value, kind_value) == expected


@pytest.mark.parametrize(
    "symm_type, c_value, kind_value, match",
    [
        # Legacy free-text words are no longer valid dropdown values.
        ("cube", 2, None, "Unknown symmetry type"),
        ("icosahedron", 2, None, "Unknown symmetry type"),
        # Cyclic order must be an integer >= 2.
        ("C", None, None, "C-symmetry value"),
        ("C", 1, None, "C-symmetry value"),
        ("C", 2.5, None, "C-symmetry value"),
        # A solid from the wrong group is rejected rather than silently used.
        ("O", 2, "dodecahedron", "not valid for symmetry"),
        ("I", 2, "cube", "not valid for symmetry"),
    ],
)
def test_build_symm_kwargs_rejects_invalid(symm_type, c_value, kind_value, match):
    with pytest.raises(ValueError, match=match):
        build_symm_kwargs(symm_type, c_value, kind_value)


def _gui_producible_pairs():
    """Every (symm_type, kind_value, expected_solid) the GUI widgets can emit.

    ``kind_value=None`` stands for "nothing picked yet" (default solid).
    ``expected_solid`` is the ``SymmParticle.kind`` the backend should end up
    with: None for cyclic, "tetrahedron" for T, the chosen/default solid for O/I.
    """
    pairs = [("C", None, None), ("T", None, "tetrahedron")]
    for letter in ("O", "I"):
        options = kind_options_for_symm(letter)
        pairs.append((letter, None, options[0]))
        pairs.extend((letter, k, k) for k in options)
    return pairs


@pytest.mark.parametrize("symm_type, kind_value, expected_solid", _gui_producible_pairs())
def test_build_symm_kwargs_accepted_by_symm_particle(symm_type, kind_value, expected_solid):
    # Cross-check against the library through its public API: every
    # (symm, kind) pair the GUI can produce must construct a real
    # SymmParticle — the class TwistDescriptor ultimately uses for symmetric
    # particles. This covers both the backend's parsing of `symm`/`kind` and
    # the construction of the chosen solid, without needing tomogram data.
    from cryocat.analysis.tango import SymmParticle

    kw = build_symm_kwargs(symm_type, 3, kind_value)
    particle = SymmParticle(np.eye(3), np.zeros(3), symm=kw["symm"], kind=kw["kind"])
    assert particle.kind == expected_solid


def test_symm_type_options_cover_all_backend_paths():
    # Every dropdown option (apart from the "None" sentinel) is exercised by
    # the SymmParticle cross-check above, so none can silently break.
    exercised = {s for s, _, _ in _gui_producible_pairs()}
    assert exercised == set(SYMM_TYPE_OPTIONS) - {"None"}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_layout_symm_controls():
    from cryocat.app.suite.pages import ptango

    # All three symmetry widgets and the kind wrapper div are mounted.
    ids = collect_ids(ptango.layout)
    for cid in ("tango-symm-type", "tango-c-symm-value", "tango-symm-kind", "tango-symm-kind-div"):
        assert cid in ids, f"{cid} missing from ptango.layout"

    # The symmetry dropdown offers the canonical Symmetry letters only.
    symm_dd = _find_by_id(ptango.layout, "tango-symm-type")
    assert symm_dd.options == ["None", "C", "T", "O", "I"]
    assert symm_dd.value == "None"

    # The kind dropdown starts empty and its wrapper starts hidden, matching
    # the default "None" symmetry.
    kind_dd = _find_by_id(ptango.layout, "tango-symm-kind")
    assert kind_dd.options == []
    assert kind_dd.value is None
    assert _find_by_id(ptango.layout, "tango-symm-kind-div").style == {"display": "none"}


def test_kind_callback_registered():
    # The suite app wires tango-symm-type -> kind div style/options/value.
    from cryocat.app.suite.app import app

    outputs = " ".join(app.callback_map.keys())
    assert "tango-symm-kind-div.style" in outputs
    assert "tango-symm-kind.options" in outputs
    assert "tango-symm-kind.value" in outputs
