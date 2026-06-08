"""Tests for the structure-page registry singleton.

The registry holds live :class:`PleomorphicSurface` objects keyed by an
opaque ``surface_id``; the page's ``dcc.Store`` only carries handles.
"""
from __future__ import annotations

import numpy as np
import pytest

from cryocat.app.components import surface_registry as sr
from cryocat.analysis.structure import PleomorphicSurface
from cryocat.core.surface import Mesh, OrientedPointCloud


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_registry():
    """Each test starts from a clean registry."""
    sr.clear_registry()
    yield
    sr.clear_registry()


@pytest.fixture
def tiny_opc():
    """Minimal point cloud (6 points on the coordinate axes) wrapped as PSurf."""
    opc = OrientedPointCloud()
    opc.vertices = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    opc.normals = opc.vertices.copy()
    return PleomorphicSurface(opc)


@pytest.fixture
def tiny_mesh():
    """Minimal mesh (a single triangle) wrapped as PSurf."""
    m = Mesh()
    m.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    m.faces = np.array([[0, 1, 2]], dtype=np.int32)
    return PleomorphicSurface(m)


# ── core operations ──────────────────────────────────────────────────────────

def test_register_and_lookup(tiny_opc):
    sid = sr.register_surface(tiny_opc)
    assert isinstance(sid, str) and sid.startswith("surface-")
    assert sr.get_surface(sid) is tiny_opc


def test_register_yields_distinct_ids(tiny_opc, tiny_mesh):
    a = sr.register_surface(tiny_opc)
    b = sr.register_surface(tiny_mesh)
    assert a != b
    assert sr.get_surface(a) is tiny_opc
    assert sr.get_surface(b) is tiny_mesh


def test_get_surface_missing_returns_none():
    assert sr.get_surface("surface-9999") is None


def test_update_surface_swaps_object(tiny_opc, tiny_mesh):
    sid = sr.register_surface(tiny_opc)
    sr.update_surface(sid, tiny_mesh)
    assert sr.get_surface(sid) is tiny_mesh


def test_update_surface_unknown_id_raises():
    with pytest.raises(KeyError):
        sr.update_surface("surface-9999", None)


def test_remove_surface(tiny_opc):
    sid = sr.register_surface(tiny_opc)
    sr.remove_surface(sid)
    assert sr.get_surface(sid) is None


def test_remove_surface_unknown_is_noop():
    # Must not raise.
    sr.remove_surface("surface-9999")


def test_clear_registry(tiny_opc, tiny_mesh):
    sr.register_surface(tiny_opc)
    sr.register_surface(tiny_mesh)
    assert len(sr.list_surface_ids()) == 2
    sr.clear_registry()
    assert sr.list_surface_ids() == []


def test_list_surface_ids_preserves_insertion_order(tiny_opc, tiny_mesh):
    a = sr.register_surface(tiny_opc)
    b = sr.register_surface(tiny_mesh)
    assert sr.list_surface_ids() == [a, b]


# ── make_handle ──────────────────────────────────────────────────────────────

def test_make_handle_for_mesh(tiny_mesh):
    h = sr.make_handle(tiny_mesh, label="my mesh")
    assert h["representation"] == "mesh"
    assert h["n_elements"] == 3
    assert h["label"] == "my mesh"
    assert h["parent_id"] is None
    assert h["visible"] is True


def test_make_handle_for_point_cloud(tiny_opc):
    h = sr.make_handle(tiny_opc, label="capsid", parent_id="surface-0", visible=False)
    assert h["representation"] == "point_cloud"
    assert h["n_elements"] == 6
    assert h["parent_id"] == "surface-0"
    assert h["visible"] is False


def test_make_handle_handles_unwrapped_visible_flag(tiny_opc):
    # truthy non-bool must be coerced to bool.
    h = sr.make_handle(tiny_opc, label="x", visible=1)
    assert h["visible"] is True
    assert isinstance(h["visible"], bool)
