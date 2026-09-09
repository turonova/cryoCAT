"""Tests for cryocat.utils.symmetry."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as rot
from scipy.linalg import det

from cryocat.utils.symmetry import (
    CyclicGroup,
    DihedralGroup,
    IcosahedralGroup,
    OctahedralGroup,
    TetrahedralGroup,
    compute_conjugation_matrix,
    get_symmetry_angles,
    get_symmetry_rotations,
    _normalize_axis,
)
from cryocat.utils.geom import (
    hausdorff_distance_sphere,
    Tetrahedron,
    Octahedron,
    Icosahedron,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_so3(R: np.ndarray, atol: float = 1e-9) -> bool:
    """True if R is a proper rotation (det=+1, R^T R = I)."""
    return (
        np.allclose(R.T @ R, np.eye(3), atol=atol)
        and abs(det(R) - 1.0) < atol
    )


def _group_is_closed(matrices: np.ndarray, atol: float = 1e-9) -> bool:
    """True if matrices is closed under multiplication."""
    for A in matrices:
        for B in matrices:
            AB = A @ B
            if not any(np.allclose(AB, C, atol=atol) for C in matrices):
                return False
    return True


def _check_symmetry(matrices: np.ndarray, reference_points: np.ndarray, atol: float = 1e-6) -> bool:
    """True if every matrix maps reference_points onto itself (up to permutation)."""
    pts = reference_points / np.linalg.norm(reference_points, axis=1, keepdims=True)
    for R in matrices:
        rotated = (R @ pts.T).T
        if hausdorff_distance_sphere(rotated, pts) > atol:
            return False
    return True


# ---------------------------------------------------------------------------
# _normalize_axis
# ---------------------------------------------------------------------------

class TestNormalizeAxis:
    def test_string_x(self):
        np.testing.assert_allclose(_normalize_axis("x"), [1, 0, 0])

    def test_string_z_upper(self):
        np.testing.assert_allclose(_normalize_axis("Z"), [0, 0, 1])

    def test_array(self):
        v = _normalize_axis([3, 0, 0])
        np.testing.assert_allclose(v, [1, 0, 0])

    def test_bad_string_raises(self):
        with pytest.raises(ValueError, match="Unknown axis"):
            _normalize_axis("w")

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            _normalize_axis([0, 0, 0])


# ---------------------------------------------------------------------------
# compute_conjugation_matrix
# ---------------------------------------------------------------------------

class TestConjugationMatrix:
    def test_same_axis_is_identity(self):
        C = compute_conjugation_matrix("z", "z")
        np.testing.assert_allclose(C, np.eye(3), atol=1e-12)

    def test_z_to_x(self):
        C = compute_conjugation_matrix("z", "x")
        np.testing.assert_allclose(C @ [0, 0, 1], [1, 0, 0], atol=1e-12)

    def test_z_to_y(self):
        C = compute_conjugation_matrix("z", "y")
        np.testing.assert_allclose(C @ [0, 0, 1], [0, 1, 0], atol=1e-12)

    def test_anti_parallel(self):
        C = compute_conjugation_matrix("z", [0, 0, -1])
        np.testing.assert_allclose(C @ [0, 0, 1], [0, 0, -1], atol=1e-12)

    def test_result_is_so3(self):
        C = compute_conjugation_matrix("z", "x")
        assert _is_so3(C)


# ---------------------------------------------------------------------------
# Group order and SO(3) membership
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cls, kwargs, expected_order",
    [
        (CyclicGroup, {"n": 1}, 1),
        (CyclicGroup, {"n": 2}, 2),
        (CyclicGroup, {"n": 6}, 6),
        (DihedralGroup, {"n": 2}, 4),
        (DihedralGroup, {"n": 3}, 6),
        (TetrahedralGroup, {}, 12),
        (OctahedralGroup, {}, 24),
        (IcosahedralGroup, {}, 60),
    ],
)
def test_group_order(cls, kwargs, expected_order):
    group = cls(**kwargs)
    assert len(group.matrices) == expected_order


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (CyclicGroup, {"n": 4}),
        (DihedralGroup, {"n": 3}),
        (TetrahedralGroup, {}),
        (OctahedralGroup, {}),
        (IcosahedralGroup, {}),
    ],
)
def test_all_elements_are_so3(cls, kwargs):
    group = cls(**kwargs)
    assert all(_is_so3(R) for R in group.matrices)


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (CyclicGroup, {"n": 3}),
        (DihedralGroup, {"n": 2}),
        (TetrahedralGroup, {}),
        (OctahedralGroup, {}),
        (IcosahedralGroup, {}),
    ],
)
def test_group_contains_identity(cls, kwargs):
    group = cls(**kwargs)
    assert any(np.allclose(R, np.eye(3)) for R in group.matrices)


# ---------------------------------------------------------------------------
# Closure (only tested for small groups to keep runtime bounded)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (CyclicGroup, {"n": 4}),
        (DihedralGroup, {"n": 3}),
        (TetrahedralGroup, {}),
    ],
)
def test_group_closure(cls, kwargs):
    group = cls(**kwargs)
    assert _group_is_closed(group.matrices)


# ---------------------------------------------------------------------------
# Symmetry correctness: group acts on the corresponding polyhedron vertices
# ---------------------------------------------------------------------------

def test_tetrahedral_acts_on_tetrahedron():
    group = TetrahedralGroup()
    assert _check_symmetry(group.matrices, Tetrahedron().vertices)


def test_octahedral_acts_on_octahedron():
    group = OctahedralGroup()
    assert _check_symmetry(group.matrices, Octahedron().vertices)


def test_icosahedral_acts_on_icosahedron():
    group = IcosahedralGroup()
    assert _check_symmetry(group.matrices, Icosahedron().vertices)


def test_cyclic_c4_acts_on_square():
    group = CyclicGroup(4)
    pts = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)
    assert _check_symmetry(group.matrices, pts)


# ---------------------------------------------------------------------------
# get_symmetry_rotations
# ---------------------------------------------------------------------------

class TestGetSymmetryRotations:
    def test_c5_returns_5_matrices(self):
        mats = get_symmetry_rotations("C5")
        assert mats.shape == (5, 3, 3)

    def test_d3_returns_6_matrices(self):
        mats = get_symmetry_rotations("D3")
        assert mats.shape == (6, 3, 3)

    def test_T_returns_12_matrices(self):
        mats = get_symmetry_rotations("T")
        assert mats.shape == (12, 3, 3)

    def test_O_returns_24_matrices(self):
        mats = get_symmetry_rotations("O")
        assert mats.shape == (24, 3, 3)

    def test_I_returns_60_matrices(self):
        mats = get_symmetry_rotations("I")
        assert mats.shape == (60, 3, 3)

    def test_int_symmetry_is_cyclic(self):
        mats = get_symmetry_rotations(3)
        assert mats.shape == (3, 3, 3)

    def test_identity_is_first(self):
        mats = get_symmetry_rotations("C4")
        np.testing.assert_allclose(mats[0], np.eye(3), atol=1e-12)

    def test_all_so3(self):
        mats = get_symmetry_rotations("O")
        assert all(_is_so3(R) for R in mats)

    def test_axis_reorientation_x(self):
        mats_z = get_symmetry_rotations("C4")
        mats_x = get_symmetry_rotations("C4", axis="x")
        assert mats_x.shape == mats_z.shape
        # All elements must still be SO(3)
        assert all(_is_so3(R) for R in mats_x)

    def test_conjugation_matrix_override(self):
        C = compute_conjugation_matrix("z", "x")
        mats_via_arg = get_symmetry_rotations("C3", axis="x")
        mats_via_mat = get_symmetry_rotations("C3", conjugation_matrix=C)
        np.testing.assert_allclose(mats_via_arg, mats_via_mat, atol=1e-12)

    def test_c2_default_angles(self):
        """C2 rotations around z should be identity and Rz(180°)."""
        mats = get_symmetry_rotations("C2")
        Rz180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        assert any(np.allclose(R, Rz180) for R in mats)


# ---------------------------------------------------------------------------
# get_symmetry_angles
# ---------------------------------------------------------------------------

class TestGetSymmetryAngles:
    def test_shape(self):
        angles = get_symmetry_angles("C3")
        assert angles.shape == (3, 3)

    def test_agrees_with_rotations(self):
        mats = get_symmetry_rotations("T")
        angles = get_symmetry_angles("T")
        reconstructed = rot.from_euler("zxz", angles, degrees=True).as_matrix()
        # Check each reconstructed matrix is in the group
        for R in reconstructed:
            assert any(np.allclose(R, M, atol=1e-9) for M in mats)

    def test_return_df(self):
        df = get_symmetry_angles("C4", return_df=True)
        assert list(df.columns) == ["phi", "theta", "psi"]
        assert len(df) == 4

    def test_radians(self):
        angles_deg = get_symmetry_angles("C3", degrees=True)
        angles_rad = get_symmetry_angles("C3", degrees=False)
        np.testing.assert_allclose(np.deg2rad(angles_deg), angles_rad, atol=1e-12)


# ---------------------------------------------------------------------------
# split_in_asymmetric_subunits – T/O/I subunit counts
# ---------------------------------------------------------------------------

class TestSplitPlatonicGroups:
    @pytest.fixture
    def single_particle(self):
        import pandas as pd
        from cryocat.core.cryomotl import Motl

        data = {
            "shift_x": [0.0], "shift_y": [0.0], "shift_z": [0.0],
            "phi": [0.0], "theta": [0.0], "psi": [0.0],
            "tomo_id": [1], "x": [10.0], "y": [10.0], "z": [10.0],
            "score": [1.0], "subtomo_id": [1],
            "geom1": [0], "geom2": [0], "object_id": [1],
            "subtomo_mean": [0.0], "geom3": [0], "geom4": [0], "geom5": [0],
            "class": [1],
        }
        return Motl(pd.DataFrame(data))

    @pytest.mark.parametrize(
        "sym, expected_count",
        [("T", 12), ("O", 24), ("I", 60)],
    )
    def test_subunit_count(self, single_particle, sym, expected_count):
        result = single_particle.split_in_asymmetric_subunits(sym, [5, 0, 0])
        assert len(result.df) == expected_count

    @pytest.mark.parametrize("sym", ["T", "O", "I"])
    def test_geom2_runs_1_to_n(self, single_particle, sym):
        result = single_particle.split_in_asymmetric_subunits(sym, [5, 0, 0])
        expected_n = {"T": 12, "O": 24, "I": 60}[sym]
        assert set(result.df["geom2"].astype(int)) == set(range(1, expected_n + 1))


# ---------------------------------------------------------------------------
# DihedralGroup orbit tests – staggered 2-fold placement
# ---------------------------------------------------------------------------

class TestDihedralOrbit:
    """Verify the staggered DihedralGroup places 2n distinct positions."""

    @pytest.mark.parametrize("n, step", [(2, 90.0), (6, 30.0)])
    def test_orbit_count_and_spacing(self, n: int, step: float) -> None:
        """Orbit of an in-plane x-shift covers 2n evenly-spaced positions."""
        group = DihedralGroup(n)
        d = 10.0
        shift = np.array([d, 0.0, 0.0])
        centers = np.array([R @ shift for R in group.matrices])

        # Must be 2n distinct positions (no duplicates)
        unique = []
        for c in centers:
            if not any(np.allclose(c, u, atol=1e-9) for u in unique):
                unique.append(c)
        assert len(unique) == 2 * n, f"D{n}: expected {2*n} distinct centers, got {len(unique)}"

        # All centers lie at radius d in the xy-plane
        np.testing.assert_allclose(np.linalg.norm(centers, axis=1), d, atol=1e-9)

        # Azimuthal angles are evenly spaced at 360/(2n) = step degrees
        angles = np.sort(np.degrees(np.arctan2(centers[:, 1], centers[:, 0])) % 360.0)
        diffs = np.diff(np.append(angles, angles[0] + 360.0))
        np.testing.assert_allclose(diffs, step, atol=1e-6)

    @pytest.mark.parametrize("n", [2, 6])
    def test_order_and_so3(self, n: int) -> None:
        group = DihedralGroup(n)
        assert len(group.matrices) == 2 * n
        assert all(_is_so3(R) for R in group.matrices)
