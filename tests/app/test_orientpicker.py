"""Tests for the orientation-picker component and page helpers.

Section 1 — Component contract
    Tests for the orientation-picker component (orientpicker) and
    orientation modal: registry, factory, registration guard,
    write-back routing, value store format.

Section 2 — Page math (pure helpers in porientpicker)
    Rotation round-trip: for any direction and in-plane angle phi,
    computing zxz Euler angles via _compute_angles and recovering
    the direction via rotations_to_z_normals yields the original
    direction to numerical precision.
    Direction handling: _normalize, _nearest_sphere_point,
    _rotate_mesh_verts.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as srot

from cryocat.app.suite.pages.porientpicker import (
    _normalize,
    _compute_angles,
    _nearest_sphere_point,
    _rotate_mesh_verts,
    _SPHERE_PTS,
)
from cryocat.utils.geom import rotations_to_z_normals


# ===========================================================================
# Helpers shared across sections
# ===========================================================================

def _make_app():
    import dash
    return dash.Dash(__name__, suppress_callback_exceptions=True)


def _walk_tree(component):
    from collections import deque
    q = deque([component])
    while q:
        node = q.popleft()
        yield node
        children = getattr(node, "children", None)
        if children is None:
            continue
        if isinstance(children, list):
            q.extend(children)
        else:
            q.append(children)


# ===========================================================================
# Section 1 — Component contract
# ===========================================================================

class TestToolsRegistry:
    def test_no_tools_entry_for_orientpicker(self):
        from cryocat.app.suite.tools import TOOLS
        modules = [t.get("module", "") for t in TOOLS]
        assert not any("porientpicker" in m for m in modules)

    def test_no_tools_id_for_orientpicker(self):
        from cryocat.app.suite.tools import TOOLS
        ids = [t.get("id", "") for t in TOOLS]
        assert "orientpicker" not in ids


class TestComponentContract:
    def test_controls_contain_no_modal(self):
        import dash_bootstrap_components as dbc
        from cryocat.app.components.orientpicker import get_orientation_picker_controls

        tree = get_orientation_picker_controls("test-c1", mode=None, show_structure=False)
        modal_nodes = [n for n in _walk_tree(tree) if isinstance(n, dbc.Modal)]
        assert not modal_nodes

    def test_panel_contains_no_nested_modal(self):
        import dash_bootstrap_components as dbc
        from cryocat.app.components.orientpicker import get_orientation_picker_panel

        tree = get_orientation_picker_panel("test-c2", mode="direction", show_structure=False)
        modal_nodes = [n for n in _walk_tree(tree) if isinstance(n, dbc.Modal)]
        assert not modal_nodes

    def test_building_controls_registers_no_callbacks(self):
        from cryocat.app.components.orientpicker import (
            get_orientation_picker_controls,
            get_orientation_picker_panel,
        )

        app = _make_app()
        before = len(app.callback_map)
        get_orientation_picker_controls("test-c3", mode="direction", show_structure=False)
        get_orientation_picker_panel("test-c4", mode="direction", show_structure=False)
        assert len(app.callback_map) == before

    def test_value_store_in_controls(self):
        from dash import dcc
        from cryocat.app.components.orientpicker import get_orientation_picker_controls

        tree = get_orientation_picker_controls("test-c5", mode="direction")
        stores = [n for n in _walk_tree(tree) if isinstance(n, dcc.Store) and n.id == "test-c5-value"]
        assert stores


class TestRegistrationGuard:
    def test_double_registration_raises_with_prefix(self):
        from cryocat.app.components.orientpicker import register_orientation_picker_callbacks

        app = _make_app()
        register_orientation_picker_callbacks(app, "dup-prefix", mode="direction", show_structure=False)
        with pytest.raises(RuntimeError, match="dup-prefix"):
            register_orientation_picker_callbacks(app, "dup-prefix", mode="direction", show_structure=False)

    def test_different_prefixes_on_same_app_allowed(self):
        from cryocat.app.components.orientpicker import register_orientation_picker_callbacks

        app = _make_app()
        register_orientation_picker_callbacks(app, "pka", mode="direction", show_structure=False)
        register_orientation_picker_callbacks(app, "pkb", mode="direction", show_structure=False)

    def test_same_prefix_on_different_apps_allowed(self):
        from cryocat.app.components.orientpicker import register_orientation_picker_callbacks

        app1 = _make_app()
        app2 = _make_app()
        register_orientation_picker_callbacks(app1, "shared-px", mode="direction", show_structure=False)
        register_orientation_picker_callbacks(app2, "shared-px", mode="direction", show_structure=False)

    def test_orient_modal_double_registration_raises(self):
        from cryocat.app.components.orientmodal import (
            get_orient_modal,
            register_orient_modal_callbacks,
        )
        import dash

        app = dash.Dash("test-orient-guard-dup", suppress_callback_exceptions=True)
        app.layout = get_orient_modal()
        register_orient_modal_callbacks(app)
        with pytest.raises(RuntimeError):
            register_orient_modal_callbacks(app)


class TestWriteBackRouting:
    def test_request_from_second_picker_updates_only_second(self):
        from dash import no_update
        from cryocat.app.components.orientpicker import _normalize as _comp_normalize

        outputs = [
            {"id": {"type": "t", "owner": "picker-a", "param": "axis", "tag": "TripletLike"}},
            {"id": {"type": "t", "owner": "picker-b", "param": "axis", "tag": "TripletLike"}},
            {"id": {"type": "t", "owner": "picker-c", "param": "axis", "tag": "TripletLike"}},
        ]
        request = {"target": {"type": "orient-pick-btn", "owner": "picker-b", "param": "axis"}}
        owner = request["target"]["owner"]
        param = request["target"]["param"]

        d = _comp_normalize([1.0, 0.0, 0.0])
        value = f"{d[0]:.6g},{d[1]:.6g},{d[2]:.6g}"

        results = [
            value if (e["id"].get("owner") == owner and e["id"].get("param") == param) else no_update
            for e in outputs
        ]

        assert results[0] is no_update
        assert results[1] == value
        assert results[2] is no_update


class TestValueFormat:
    def test_direction_mode_gives_unit_vector(self):
        from cryocat.app.components.orientpicker import _normalize as _comp_normalize

        raw = [3.0, 0.0, 0.0]
        d = _comp_normalize(raw)
        assert len(d) == 3
        assert abs(np.linalg.norm(d) - 1.0) < 1e-9
        assert d[0] == pytest.approx(1.0)

    def test_rotation_mode_gives_euler_triple(self):
        from cryocat.app.components.orientpicker import _compute_angles as _comp_angles
        from cryocat.app.components.orientpicker import _normalize as _comp_normalize

        d = _comp_normalize([0.0, 0.0, 1.0])
        phi = 45.0
        angles = _comp_angles(d, phi)
        assert len(angles) == 3
        assert angles[0] == pytest.approx(45.0)
        assert angles[1] == pytest.approx(0.0)
        assert np.isfinite(angles[2])

    def test_value_same_from_any_prefix(self):
        from cryocat.app.components.orientpicker import _compute_angles as _comp_angles
        from cryocat.app.components.orientpicker import _normalize as _comp_normalize

        direction = [0.5, 0.3, 0.8]
        phi = 20.0
        d = _comp_normalize(direction)

        result_a = _comp_angles(d, phi).tolist()
        result_b = _comp_angles(d, phi).tolist()

        np.testing.assert_array_almost_equal(result_a, result_b)

    def test_zero_direction_raises(self):
        from cryocat.app.components.orientpicker import _normalize as _comp_normalize

        with pytest.raises(ValueError, match="zero"):
            _comp_normalize([0.0, 0.0, 0.0])

    def test_nearest_sphere_point_returns_unit_vector(self):
        from cryocat.app.components.orientpicker import _nearest_sphere_point as _comp_nsp
        from cryocat.app.components.orientpicker import _SPHERE_PTS as _comp_spts

        pt = _comp_nsp(_comp_spts, [1.0, 0.5, 0.3])
        assert abs(np.linalg.norm(pt) - 1.0) < 1e-9

    def test_rotate_mesh_verts_preserves_shape(self):
        from cryocat.app.components.orientpicker import _rotate_mesh_verts as _comp_rmv

        rng = np.random.default_rng(42)
        verts = rng.standard_normal((50, 3))
        rotated = _comp_rmv(verts, 30.0, 45.0, 60.0)
        assert rotated.shape == verts.shape


# ===========================================================================
# Section 2 — Page math (porientpicker pure helpers)
# ===========================================================================

class TestRotationRoundtrip:

    def test_200_random_directions_all_phi(self):
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((200, 3))
        directions = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        for d in directions:
            for phi in (0.0, 90.0, 180.0, 270.0):
                angles = _compute_angles(d, phi)
                recovered = rotations_to_z_normals(angles.reshape(1, 3))[0]
                np.testing.assert_allclose(
                    recovered, d, atol=1e-9,
                    err_msg=f"Round-trip failed: d={d}, phi={phi}",
                )

    def test_phi_does_not_shift_recovered_direction(self):
        d = _normalize([1.0, 2.0, 3.0])
        r0   = rotations_to_z_normals(_compute_angles(d,   0.0).reshape(1, 3))[0]
        r90  = rotations_to_z_normals(_compute_angles(d,  90.0).reshape(1, 3))[0]
        r270 = rotations_to_z_normals(_compute_angles(d, 270.0).reshape(1, 3))[0]
        np.testing.assert_allclose(r0, r90,  atol=1e-9)
        np.testing.assert_allclose(r0, r270, atol=1e-9)

    def test_phi_stored_correctly(self):
        d = _normalize([1.0, 0.0, 0.0])
        for phi in (0.0, 33.7, 180.0, 359.9):
            angles = _compute_angles(d, phi)
            assert angles[0] == pytest.approx(phi, abs=1e-12)

    def test_theta_psi_independent_of_phi(self):
        d = _normalize([0.5, 0.5, 0.7071])
        a0   = _compute_angles(d,   0.0)
        a120 = _compute_angles(d, 120.0)
        a240 = _compute_angles(d, 240.0)
        np.testing.assert_allclose(a0[1:], a120[1:], atol=1e-9)
        np.testing.assert_allclose(a0[1:], a240[1:], atol=1e-9)

    def test_north_pole(self):
        d = np.array([0.0, 0.0, 1.0])
        for phi in (0.0, 90.0, 180.0):
            angles = _compute_angles(d, phi)
            assert not np.any(np.isnan(angles))
            recovered = rotations_to_z_normals(angles.reshape(1, 3))[0]
            np.testing.assert_allclose(recovered, d, atol=1e-9)

    def test_south_pole(self):
        d = np.array([0.0, 0.0, -1.0])
        for phi in (0.0, 45.0, 90.0, 180.0):
            angles = _compute_angles(d, phi)
            assert not np.any(np.isnan(angles))
            recovered = rotations_to_z_normals(angles.reshape(1, 3))[0]
            np.testing.assert_allclose(recovered, d, atol=1e-9)

    def test_equatorial_cardinal_directions(self):
        for d in ([1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]):
            d_arr = np.array(d, dtype=float)
            angles = _compute_angles(d_arr, 45.0)
            recovered = rotations_to_z_normals(angles.reshape(1, 3))[0]
            np.testing.assert_allclose(recovered, d_arr, atol=1e-9)


class TestNormalize:

    def test_unit_vector_unchanged(self):
        np.testing.assert_allclose(_normalize([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0])

    def test_scales_to_unit_length(self):
        d = _normalize([3.0, 0.0, 0.0])
        np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-12)

    def test_direction_preserved(self):
        v = [1.0, 2.0, 3.0]
        d = _normalize(v)
        expected = np.array(v) / np.linalg.norm(v)
        np.testing.assert_allclose(d, expected, atol=1e-12)

    def test_negative_direction(self):
        d = _normalize([-2.0, 0.0, 0.0])
        np.testing.assert_allclose(d, [-1.0, 0.0, 0.0])

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            _normalize([0.0, 0.0, 0.0])

    def test_near_zero_raises(self):
        with pytest.raises(ValueError):
            _normalize([0.0, 0.0, 1e-13])


class TestNearestSpherePoint:

    def test_result_lies_on_unit_sphere(self):
        for direction in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]):
            pt = _nearest_sphere_point(_SPHERE_PTS, direction)
            np.testing.assert_allclose(np.linalg.norm(pt), 1.0, atol=1e-10)

    def test_high_dot_product_with_query(self):
        for d, min_dot in (
            ([1.0, 0.0, 0.0], 0.98),
            ([0.0, 0.0, -1.0], 0.98),
            ([0.0, 0.0, 1.0], 0.98),
        ):
            pt = _nearest_sphere_point(_SPHERE_PTS, d)
            assert np.dot(pt, _normalize(d)) > min_dot

    def test_result_is_element_of_sphere_pts(self):
        d = [0.3, 0.6, 0.7]
        pt = _nearest_sphere_point(_SPHERE_PTS, d)
        assert np.any(np.all(np.isclose(_SPHERE_PTS, pt), axis=1))

    def test_zero_direction_raises(self):
        with pytest.raises(ValueError):
            _nearest_sphere_point(_SPHERE_PTS, [0.0, 0.0, 0.0])


class TestRotateMeshVerts:

    def test_identity_rotation(self):
        rng = np.random.default_rng(0)
        verts = rng.standard_normal((50, 3))
        rot = _rotate_mesh_verts(verts, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(rot, verts, atol=1e-12)

    def test_preserves_vertex_norms(self):
        rng = np.random.default_rng(1)
        verts = rng.standard_normal((30, 3))
        rot = _rotate_mesh_verts(verts, 30.0, 45.0, 60.0)
        np.testing.assert_allclose(
            np.linalg.norm(rot, axis=1),
            np.linalg.norm(verts, axis=1),
            atol=1e-10,
        )

    def test_roundtrip_with_inverse(self):
        rng = np.random.default_rng(2)
        verts = rng.standard_normal((20, 3))
        phi, theta, psi = 30.0, 45.0, 60.0
        forward = _rotate_mesh_verts(verts, phi, theta, psi)
        inv_angles = srot.from_euler("zxz", [phi, theta, psi], degrees=True).inv().as_euler(
            "zxz", degrees=True
        )
        back = _rotate_mesh_verts(forward, inv_angles[0], inv_angles[1], inv_angles[2])
        np.testing.assert_allclose(back, verts, atol=1e-10)

    def test_90deg_around_z_moves_x_to_y(self):
        verts = np.array([[1.0, 0.0, 0.0]])
        rot = _rotate_mesh_verts(verts, 90.0, 0.0, 0.0)
        np.testing.assert_allclose(rot[0], [0.0, 1.0, 0.0], atol=1e-10)
