"""Tests for cryocat.utils.symmetry."""

import warnings

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
    SymmGroup,
    SYMMETRY_GROUPS,
    angular_score,
    max_angular_mismatch,
    reduce_angle_grid,
    compute_conjugation_matrix,
    get_symmetry_angles,
    get_symmetry_rotations,
    _normalize_axis,
)
from cryocat.utils.geom import (
    hausdorff_distance_sphere,
    Tetrahedron,
    Octahedron,
    Cube,
    Icosahedron,
    Dodecahedron,
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


def test_octahedral_acts_on_cube():
    # The cube is the dual of the octahedron, so the same 24 rotations must
    # leave its canonical vertices unchanged.
    group = OctahedralGroup()
    assert _check_symmetry(group.matrices, Cube().vertices)


def test_icosahedral_acts_on_dodecahedron():
    # Regression test for the Dodecahedron alignment fix (2026-09-24): the
    # canonical dodecahedron used to be turned 90 deg about z relative to the
    # icosahedron, so only 12 of the 60 icosahedral rotations preserved it.
    group = IcosahedralGroup()
    assert _check_symmetry(group.matrices, Dodecahedron().vertices)


def test_dodecahedron_is_dual_of_icosahedron():
    # Dual alignment: the dodecahedron's 20 vertices must sit on the
    # directions of the icosahedron's 20 face centres (compared on the unit
    # sphere, independent of ordering).
    dod = Dodecahedron().vertices
    ico_faces = Icosahedron().faces
    dod = dod / np.linalg.norm(dod, axis=1, keepdims=True)
    ico_faces = ico_faces / np.linalg.norm(ico_faces, axis=1, keepdims=True)
    assert dod.shape == ico_faces.shape
    assert hausdorff_distance_sphere(dod, ico_faces) < 1e-6


def test_dodecahedron_first_two_vertices_are_adjacent():
    # tango.SymmParticle.max_dissimilarity uses vertices 0 and 1 as a
    # nearest-neighbour pair; the alignment fix must keep that ordering.
    dod = Dodecahedron()
    edges = {tuple(sorted(e)) for e in dod._edge_idx.tolist()}
    assert (0, 1) in edges


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
# get_symmetry_rotations: axis= is ambiguous for D/T/O/I (added 2026-10-07)
# ---------------------------------------------------------------------------

def _same_rotation_set(a, b, atol=1e-8):
    """True if the two (M, 3, 3) stacks hold the same rotations, in any order."""
    return len(a) == len(b) and all(any(np.allclose(x, y, atol=atol) for y in b) for x in a)


class TestAxisAmbiguityWarning:
    """One axis fixes a C_n group but not a D/T/O/I group, whose other axes are then
    placed by a hidden shortest turn. get_symmetry_rotations warns in that case only;
    the returned matrices must be exactly the same as without the warning."""

    @pytest.mark.parametrize("symm", ["D2", "D3", "T", "O", "I"])
    @pytest.mark.parametrize("axis", ["x", "y", [1.0, 1.0, 0.0]])
    def test_non_cyclic_axis_off_z_warns(self, symm, axis):
        # Any axis not along ±z leaves the other symmetry axes undefined → warning.
        with pytest.warns(UserWarning, match="fixes only one axis"):
            get_symmetry_rotations(symm, axis=axis)

    @pytest.mark.parametrize("symm", ["C1", "C4", 6])
    def test_cyclic_axis_does_not_warn(self, symm):
        # For C_n the axis alone defines the group completely: no warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            get_symmetry_rotations(symm, axis="x")

    @pytest.mark.parametrize("symm", ["D3", "T", "O", "I"])
    @pytest.mark.parametrize("axis", ["z", "Z", [0.0, 0.0, 2.0], [0.0, 0.0, -1.0]])
    def test_axis_along_z_does_not_warn(self, symm, axis):
        # Along ±z the result is the canonical group (checked below), so nothing is ambiguous.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mats = get_symmetry_rotations(symm, axis=axis)
        assert _same_rotation_set(mats, get_symmetry_rotations(symm))

    @pytest.mark.parametrize("symm", ["D3", "T", "I"])
    def test_conjugation_matrix_does_not_warn(self, symm):
        # A full orientation is unambiguous; axis is then ignored, even if set.
        C = rot.from_euler("zxz", [30, 40, 50], degrees=True).as_matrix()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            get_symmetry_rotations(symm, axis="x", conjugation_matrix=C)

    @pytest.mark.parametrize("symm", ["D3", "T", "O", "I"])
    def test_warning_does_not_change_output(self, symm):
        # Output equals the explicit shortest-turn conjugation, as before the warning existed.
        Cx = compute_conjugation_matrix("z", "x")
        with pytest.warns(UserWarning):
            mats = get_symmetry_rotations(symm, axis="x")
        np.testing.assert_allclose(mats, get_symmetry_rotations(symm, conjugation_matrix=Cx), atol=1e-12)

    @pytest.mark.parametrize("symm", ["T", "O"])
    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_T_O_axis_xy_gives_canonical_set(self, symm, axis):
        # Documented pitfall: for T/O the shortest turn onto x/y is itself a symmetry of
        # the cube's frame, so the result is the canonical set of rotations.
        with pytest.warns(UserWarning):
            mats = get_symmetry_rotations(symm, axis=axis)
        assert _same_rotation_set(mats, get_symmetry_rotations(symm))

    def test_D3_axis_does_not_fix_side_axes(self):
        # Documented ambiguity: two valid D3 groups with the 3-fold along x (differing by a
        # 30° spin about x) are different sets; axis="x" silently returns only one of them.
        Cx = compute_conjugation_matrix("z", "x")
        spun = rot.from_euler("x", 30, degrees=True).as_matrix() @ Cx
        with pytest.warns(UserWarning):
            mats_axis = get_symmetry_rotations("D3", axis="x")
        mats_spun = get_symmetry_rotations("D3", conjugation_matrix=spun)
        # Both have a 3-fold (120° turn) about x ...
        for mats in (mats_axis, mats_spun):
            rotvecs = rot.from_matrix(mats).as_rotvec()
            assert any(np.allclose(rv, [2 * np.pi / 3, 0, 0], atol=1e-8) for rv in rotvecs)
        # ... but their half-turn axes differ.
        assert not _same_rotation_set(mats_axis, mats_spun)


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



# ===========================================================================
# Link between symmetry groups and Platonic solids (added 2026-09-25)
# ===========================================================================

# All group constructors, used to check behaviour that must hold for every group.
_ALL_GROUPS = [
    (CyclicGroup, {"n": 6}),
    (DihedralGroup, {"n": 4}),
    (TetrahedralGroup, {}),
    (OctahedralGroup, {}),
    (IcosahedralGroup, {}),
]

_R_TEST = rot.from_euler("zxz", [30.0, 50.0, 10.0], degrees=True)


def _same_directions(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> bool:
    """True if *a* and *b* contain the same directions (as sets, on the unit sphere)."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a.shape == b.shape and hausdorff_distance_sphere(a, b) < atol


def _fitted(solid_cls, other_neighbour: bool = False):
    """Solid fitted from vertex 0 and one neighbour of a solid turned by _R_TEST.

    Mimics PolyhedralComplex.fit_geometry: *other_neighbour* picks a different
    neighbouring vertex, which yields a different (but equivalent) rotation.
    """
    true = solid_cls(radius=40.0, R=_R_TEST)
    neighbours = [e[1] if e[0] == 0 else e[0] for e in true._edge_idx if 0 in e]
    v2 = true.vertices[neighbours[1 if other_neighbour else 0]]
    return solid_cls.from_vectors(true.vertices[0], v2)


class TestSymmGroupOrientation:
    @pytest.mark.parametrize("cls, kwargs", _ALL_GROUPS)
    def test_new_group_is_canonical(self, cls, kwargs):
        # A freshly built group is in the textbook orientation.
        g = cls(**kwargs)
        assert np.allclose(g.rotation.as_matrix(), np.eye(3))
        assert g.symbol in "CDTOI"

    @pytest.mark.parametrize("cls, kwargs", _ALL_GROUPS)
    def test_oriented_conjugates_every_rotation(self, cls, kwargs):
        # oriented(R) re-expresses each rotation as R @ g @ R.T and stores R.
        g = cls(**kwargs)
        C = _R_TEST.as_matrix()
        turned = g.oriented(_R_TEST)
        np.testing.assert_allclose(turned.matrices, np.array([C @ m @ C.T for m in g.matrices]), atol=1e-12)
        np.testing.assert_allclose(turned.rotation.as_matrix(), C, atol=1e-12)
        # identity stays first, and the result is still a closed group
        np.testing.assert_allclose(turned.matrices[0], np.eye(3), atol=1e-12)
        assert _group_is_closed(turned.matrices, atol=1e-8)

    def test_oriented_does_not_modify_original(self):
        # The canonical group must not change when a turned copy is made.
        g = IcosahedralGroup()
        before = g.matrices.copy()
        g.oriented(_R_TEST)
        np.testing.assert_array_equal(g.matrices, before)
        assert np.allclose(g.rotation.as_matrix(), np.eye(3))

    def test_oriented_round_trip_and_cumulative(self):
        # Turning by R then by R^-1 gives back the canonical group; turning is cumulative.
        g = OctahedralGroup()
        back = g.oriented(_R_TEST).oriented(_R_TEST.inv())
        np.testing.assert_allclose(back.matrices, g.matrices, atol=1e-12)
        np.testing.assert_allclose(back.rotation.as_matrix(), np.eye(3), atol=1e-12)
        other = rot.from_euler("z", 20.0, degrees=True)
        twice = g.oriented(_R_TEST).oriented(other)
        np.testing.assert_allclose(twice.rotation.as_matrix(), (other * _R_TEST).as_matrix(), atol=1e-12)

    @pytest.mark.parametrize("form", ["rotation", "matrix", "euler"])
    def test_oriented_accepts_rotationlike(self, form):
        # Rotation object, 3x3 matrix and zxz Euler triple (degrees) are equivalent.
        value = {"rotation": _R_TEST, "matrix": _R_TEST.as_matrix(), "euler": [30.0, 50.0, 10.0]}[form]
        np.testing.assert_allclose(
            TetrahedralGroup().oriented(value).matrices, TetrahedralGroup().oriented(_R_TEST).matrices, atol=1e-12
        )

    def test_oriented_rejects_multiple_rotations(self):
        stack = rot.from_euler("zxz", [[0, 0, 0], [10, 20, 30]], degrees=True)
        with pytest.raises(ValueError, match="single rotation"):
            IcosahedralGroup().oriented(stack)

    def test_get_symmetry_rotations_wrapper_matches_explicit_conjugation(self):
        # get_symmetry_rotations now delegates to oriented(); its output must
        # equal the explicit C @ g @ C.T of the previous implementation.
        C = _R_TEST.as_matrix()
        canonical = get_symmetry_rotations("I")
        expected = np.array([C @ m @ C.T for m in canonical])
        np.testing.assert_allclose(get_symmetry_rotations("I", conjugation_matrix=C), expected, atol=1e-12)
        Cx = compute_conjugation_matrix("z", "x")
        expected_x = np.array([Cx @ m @ Cx.T for m in get_symmetry_rotations("D3")])
        np.testing.assert_allclose(get_symmetry_rotations("D3", axis="x"), expected_x, atol=1e-12)


class TestSymmGroupAxes:
    @pytest.mark.parametrize(
        "group, counts",
        [
            (IcosahedralGroup(), {2: 30, 3: 20, 5: 12}),
            (OctahedralGroup(), {2: 12, 3: 8, 4: 6}),
            (TetrahedralGroup(), {2: 6, 3: 8}),
            (DihedralGroup(4), {2: 8, 4: 2}),
            # the z-axis of C6 is reported once, as a 6-fold axis (not also as 2- or 3-fold)
            (CyclicGroup(6), {2: 0, 3: 0, 6: 2}),
        ],
    )
    def test_axis_counts(self, group, counts):
        # Number of axis directions (both ends of every axis) per fold.
        for fold, n in counts.items():
            assert group.axes(fold).shape == (n, 3)

    def test_axes_match_solid_features(self):
        # The spin axes of the Platonic groups are the solids' feature directions.
        assert _same_directions(IcosahedralGroup().axes(5), Icosahedron().vertices)
        assert _same_directions(IcosahedralGroup().axes(3), Icosahedron().faces)
        assert _same_directions(IcosahedralGroup().axes(2), Icosahedron().edges)
        assert _same_directions(IcosahedralGroup().axes(3), Dodecahedron().vertices)
        assert _same_directions(OctahedralGroup().axes(4), Octahedron().vertices)
        assert _same_directions(OctahedralGroup().axes(3), Cube().vertices)

    def test_tetrahedron_three_fold_axes_are_vertices_and_faces(self):
        # Design decision 4: for T, the 8 three-fold directions are the 4
        # vertices plus the 4 face centres, so vertices can't come from axes().
        both = np.vstack((Tetrahedron().vertices, Tetrahedron().faces))
        assert _same_directions(TetrahedralGroup().axes(3), both)

    def test_axes_follow_orientation(self):
        # Turning the group turns its axes the same way.
        turned = IcosahedralGroup().oriented(_R_TEST).axes(5)
        assert _same_directions(turned, _R_TEST.apply(IcosahedralGroup().axes(5)))

    def test_invalid_fold_raises(self):
        with pytest.raises(ValueError, match="fold"):
            OctahedralGroup().axes(1)


class TestSymmGroupOrbit:
    @pytest.mark.parametrize(
        "point, expected",
        [
            (Icosahedron().vertices[0], 12),  # on a 5-fold axis: 60 / 5
            (Icosahedron().faces[0], 20),  # on a 3-fold axis: 60 / 3
            (Icosahedron().edges[0], 30),  # on a 2-fold axis: 60 / 2
            ([0.3, 0.1, 0.9], 60),  # off every axis: one per rotation
        ],
    )
    def test_icosahedral_orbit_sizes(self, point, expected):
        assert len(IcosahedralGroup().orbit(point)) == expected

    def test_orbit_of_vertex_is_the_solid(self):
        # The orbit of one vertex is the whole vertex set of the matching solid.
        assert _same_directions(OctahedralGroup().orbit(Octahedron().vertices[0]), Octahedron().vertices)
        assert _same_directions(TetrahedralGroup().orbit(Tetrahedron().vertices[0]), Tetrahedron().vertices)

    def test_orbit_starts_with_point_and_keeps_length(self):
        p = np.array([2.0, -1.0, 5.0])
        orb = DihedralGroup(3).orbit(p)
        np.testing.assert_allclose(orb[0], p)
        np.testing.assert_allclose(np.linalg.norm(orb, axis=1), np.linalg.norm(p))


class TestToPolyhedron:
    @pytest.mark.parametrize(
        "group, solid_cls",
        [(TetrahedralGroup(), Tetrahedron), (OctahedralGroup(), Octahedron), (IcosahedralGroup(), Icosahedron)],
    )
    def test_default_kind(self, group, solid_cls):
        # Design decision 2: defaults are Tetrahedron / Octahedron / Icosahedron.
        assert type(group.to_polyhedron()) is solid_cls

    @pytest.mark.parametrize("group, kind, solid_cls", [(OctahedralGroup(), "cube", Cube), (IcosahedralGroup(), "Dodecahedron", Dodecahedron)])
    def test_explicit_kind(self, group, kind, solid_cls):
        assert type(group.to_polyhedron(kind=kind)) is solid_cls

    def test_orientation_and_radius(self):
        # The solid is built with the group's orientation and the given radius.
        solid = IcosahedralGroup().oriented(_R_TEST).to_polyhedron(radius=40.0)
        np.testing.assert_allclose(solid.rotation.as_matrix(), _R_TEST.as_matrix(), atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(solid.vertices, axis=1), 40.0)
        assert _same_directions(solid.vertices, Icosahedron(R=_R_TEST).vertices)

    @pytest.mark.parametrize("cls, kwargs", _ALL_GROUPS)
    def test_group_leaves_its_solids_unchanged(self, cls, kwargs):
        # Every solid a (turned) group returns is left unchanged by that group.
        g = cls(**kwargs).oriented(_R_TEST)
        if g.symbol in "CD":
            pytest.skip("cyclic/dihedral groups have no Platonic solid")
        for kind in {"T": ["tetrahedron"], "O": ["octahedron", "cube"], "I": ["icosahedron", "dodecahedron"]}[g.symbol]:
            assert _check_symmetry(g.matrices, g.to_polyhedron(kind=kind).vertices)

    def test_wrong_kind_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            OctahedralGroup().to_polyhedron(kind="icosahedron")

    @pytest.mark.parametrize("group", [CyclicGroup(4), DihedralGroup(3)])
    def test_cyclic_dihedral_have_no_solid(self, group):
        with pytest.raises(ValueError, match="no associated Platonic solid"):
            group.to_polyhedron()


class TestFromPolyhedron:
    @pytest.mark.parametrize(
        "solid_cls, group_cls",
        [
            (Tetrahedron, TetrahedralGroup),
            (Octahedron, OctahedralGroup),
            (Cube, OctahedralGroup),
            (Icosahedron, IcosahedralGroup),
            (Dodecahedron, IcosahedralGroup),
        ],
    )
    def test_group_type_and_orientation(self, solid_cls, group_cls):
        # The group is looked up from the solid's type and turned like the solid.
        solid = _fitted(solid_cls)
        g = SymmGroup.from_polyhedron(solid)
        assert type(g) is group_cls
        np.testing.assert_allclose(g.rotation.as_matrix(), solid.rotation.as_matrix(), atol=1e-12)
        assert _check_symmetry(g.matrices, solid.vertices)

    def test_same_group_for_equivalent_marker_choices(self):
        # Two neighbour choices give different rotations but the same group.
        a = _fitted(Icosahedron)
        b = _fitted(Icosahedron, other_neighbour=True)
        assert not np.allclose(a.rotation.as_matrix(), b.rotation.as_matrix())
        ga, gb = IcosahedralGroup.from_polyhedron(a), IcosahedralGroup.from_polyhedron(b)
        for m in ga.matrices:
            assert any(np.allclose(m, n, atol=1e-9) for n in gb.matrices)

    def test_round_trip_group_to_solid(self):
        # from_polyhedron followed by to_polyhedron reproduces the fitted solid.
        solid = _fitted(Icosahedron)
        back = SymmGroup.from_polyhedron(solid).to_polyhedron(radius=solid.radius)
        assert _same_directions(back.vertices, solid.vertices)

    def test_group_class_mismatch_raises(self):
        with pytest.raises(ValueError, match="belongs to TetrahedralGroup"):
            IcosahedralGroup.from_polyhedron(Tetrahedron())

    def test_non_polyhedron_raises(self):
        with pytest.raises(TypeError, match="Platonic solid"):
            SymmGroup.from_polyhedron(np.eye(3))

    def test_misaligned_solid_fails_invariance_check(self):
        # Design decision 3: a solid whose canonical frame doesn't match the
        # group (here: a dodecahedron turned 90 deg about z, as before the
        # 2026-09-24 alignment fix, but with an identity rotation) is rejected.
        misaligned = Dodecahedron()
        misaligned.vertices = misaligned.vertices @ rot.from_euler("z", 90.0, degrees=True).as_matrix().T
        with pytest.raises(ValueError, match="does not leave its vertices unchanged"):
            SymmGroup.from_polyhedron(misaligned)


# ---------------------------------------------------------------------------
# Group order vs. solid counts (plan point 3: "document the mismatch")
# ---------------------------------------------------------------------------

class TestOrderVersusSolidCounts:
    """Pin the relationship documented in the ``symmetry`` module Notes.

    The group *order* counts rotations (T 12, O 24, I 60); a Platonic solid's
    vertex/edge/face counts are smaller, because every vertex, edge midpoint
    and face centre sits on an ``n``-fold axis and so only lands on
    ``order / n`` distinct places. ``TestSymmGroupOrbit`` already covers the
    icosahedral orbit sizes; here every solid of T, O and I is checked against
    its own declared counts.
    """

    @pytest.mark.parametrize(
        "letter, group_cls, order",
        [("T", TetrahedralGroup, 12), ("O", OctahedralGroup, 24), ("I", IcosahedralGroup, 60)],
    )
    def test_as_symmetry_order_is_group_order(self, letter, group_cls, order):
        # as_symmetry and SymmGroup report the same number: the rotation count.
        from cryocat.utils.geom import as_symmetry

        assert as_symmetry(letter) == (letter, order)
        assert group_cls.order == order
        assert len(group_cls().matrices) == order

    @pytest.mark.parametrize(
        "group_cls, solid_cls, folds",
        [
            # folds = spin-axis fold through (vertex, edge midpoint, face centre)
            (TetrahedralGroup, Tetrahedron, (3, 2, 3)),
            (OctahedralGroup, Octahedron, (4, 2, 3)),
            (OctahedralGroup, Cube, (3, 2, 4)),
            (IcosahedralGroup, Icosahedron, (5, 2, 3)),
            (IcosahedralGroup, Dodecahedron, (3, 2, 5)),
        ],
    )
    def test_solid_counts_are_order_over_fold(self, group_cls, solid_cls, folds):
        group = group_cls()
        solid = solid_cls()
        features = (solid.vertices, solid.edges, solid.faces)
        declared = (solid.n_vertices, solid.n_edges, solid.n_faces)
        for points, n_declared, fold in zip(features, declared, folds):
            # The solid's declared count follows the counting rule order / fold ...
            assert n_declared == group.order // fold
            assert len(points) == n_declared
            # ... exactly `fold` rotations leave one such point in place ...
            n_fixed = sum(np.allclose(m @ points[0], points[0], atol=1e-9) for m in group.matrices)
            assert n_fixed == fold
            # ... and the orbit of that one point is the whole feature set.
            orbit = group.orbit(points[0])
            assert len(orbit) == n_declared
            assert _check_symmetry(group.matrices, points)

    @pytest.mark.parametrize("group_cls", [TetrahedralGroup, OctahedralGroup])
    def test_generic_point_gives_order_copies(self, group_cls):
        # A point off every axis lands on `order` places (I is covered in
        # TestSymmGroupOrbit), as split_in_asymmetric_subunits assumes.
        assert len(group_cls().orbit([0.3, 0.1, 0.9])) == group_cls.order

    @pytest.mark.parametrize("n", [2, 3, 6])
    def test_dihedral_as_symmetry_returns_n_not_group_order(self, n):
        # For "Dn", as_symmetry returns n, while DihedralGroup(n) has 2n rotations.
        from cryocat.utils.geom import as_symmetry

        assert as_symmetry(f"D{n}") == ("D", n)
        assert DihedralGroup(n).order == 2 * n
        assert len(get_symmetry_rotations(f"D{n}")) == 2 * n


# ---------------------------------------------------------------------------
# max_angular_mismatch / angular_score (added 2026-09-29)
# ---------------------------------------------------------------------------

_SCORE_CASES = [("T", None), ("O", "octahedron"), ("O", "cube"), ("I", "icosahedron"), ("I", "dodecahedron")]


def _unit_corners(letter, kind):
    v = SYMMETRY_GROUPS[letter]().to_polyhedron(kind=kind).vertices
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TestMaxAngularMismatch:
    """d_max = the largest possible mismatch: pi/N for cyclic, corner -> nearest
    face centre for T/O/I (the deepest "hole" between corners)."""

    @pytest.mark.parametrize("n", [2, 3, 5, 12])
    def test_cyclic_is_pi_over_n(self, n):
        assert max_angular_mismatch(n) == pytest.approx(np.pi / n)
        assert max_angular_mismatch(f"C{n}") == pytest.approx(np.pi / n)

    @pytest.mark.parametrize(
        "letter, kind, expected_deg",
        [
            ("T", None, 70.5288),
            ("O", None, 54.7356),
            ("O", "cube", 54.7356),
            ("I", None, 37.3774),
            ("I", "dodecahedron", 37.3774),
        ],
    )
    def test_platonic_values(self, letter, kind, expected_deg):
        # Both solids of a group share the same maximum (their holes are the
        # other solid's corners, at the same angle).
        assert np.degrees(max_angular_mismatch(letter, kind)) == pytest.approx(expected_deg, abs=1e-3)

    @pytest.mark.parametrize("letter, kind", _SCORE_CASES)
    def test_no_random_pair_exceeds_it(self, letter, kind):
        # Sampled check that it is an upper bound on the actual mismatch.
        v = _unit_corners(letter, kind)
        d_max = max_angular_mismatch(letter, kind)
        for m in rot.random(500, random_state=11).as_matrix():
            assert hausdorff_distance_sphere(v, v @ m.T) <= d_max + 1e-9

    def test_invalid_inputs(self):
        with pytest.raises(NotImplementedError, match="dihedral"):
            max_angular_mismatch("D4")
        with pytest.raises(ValueError, match="greater than 1"):
            max_angular_mismatch(1)
        with pytest.raises(ValueError, match="only applicable"):
            max_angular_mismatch(4, kind="cube")
        with pytest.raises(ValueError, match="does not match"):
            max_angular_mismatch("O", kind="dodecahedron")


class TestAngularScore:
    """Symmetry-aware similarity of paired orientations in [0, 1]."""

    @pytest.mark.parametrize("letter, kind", _SCORE_CASES)
    def test_symmetry_equivalent_orientations_score_one(self, letter, kind):
        # Any orientation compared with itself turned by one of the group's
        # rotations looks identical: score 1 for every group element.
        base = rot.from_euler("zxz", [10, 60, 5], degrees=True).as_matrix()
        group = SYMMETRY_GROUPS[letter]().matrices
        first = np.repeat(base[None], len(group), axis=0)
        second = np.einsum("ij,njk->nik", base, group)  # base @ g
        np.testing.assert_array_equal(angular_score(first, second, letter, kind=kind), 1.0)

    @pytest.mark.parametrize("letter, kind", _SCORE_CASES)
    def test_matches_hausdorff_reference(self, letter, kind):
        # The batched computation equals the per-pair Hausdorff formula
        # (clamped as documented), also when split into several chunks.
        r1 = rot.random(120, random_state=1).as_matrix()
        r2 = rot.random(120, random_state=2).as_matrix()
        v = _unit_corners(letter, kind)
        d_max = max_angular_mismatch(letter, kind)
        ref = np.array([1 - hausdorff_distance_sphere(v @ a.T, v @ b.T) / d_max for a, b in zip(r1, r2)])
        ref = np.where(ref > 1 - 1e-5, 1.0, np.where(ref < 1e-5, 0.0, ref))
        np.testing.assert_allclose(angular_score(r1, r2, letter, kind=kind, chunk_size=7), ref, atol=1e-12)

    @pytest.mark.parametrize("letter, kind", _SCORE_CASES)
    def test_scores_in_unit_interval(self, letter, kind):
        r1 = rot.random(300, random_state=5)
        r2 = rot.random(300, random_state=6)
        s = angular_score(r1, r2, letter, kind=kind)
        assert s.shape == (300,)
        assert ((s >= 0) & (s <= 1)).all()

    @pytest.mark.parametrize("letter, kind", _SCORE_CASES)
    def test_only_relative_orientation_matters(self, letter, kind):
        # Turning both particles of every pair by the same rotation g leaves
        # the score unchanged.
        r1 = rot.random(50, random_state=7)
        r2 = rot.random(50, random_state=8)
        g = rot.from_euler("zxz", [33, 71, 12], degrees=True)
        np.testing.assert_allclose(
            angular_score(r1, r2, letter, kind=kind), angular_score(g * r1, g * r2, letter, kind=kind), atol=1e-9
        )

    def test_cyclic_delegates_to_geom(self):
        # Cyclic input is scored exactly as geom.angular_score_for_c_symmetry,
        # from the first zxz Euler angle (phi) of each rotation.
        from cryocat.utils.geom import angular_score_for_c_symmetry

        e1 = np.array([[10.0, 30.0, 0.0], [50.0, 80.0, 20.0], [0.0, 45.0, 0.0]])
        e2 = np.array([[40.0, 10.0, 5.0], [175.0, 20.0, 0.0], [72.0, 45.0, 0.0]])
        phi1 = rot.from_euler("zxz", e1, degrees=True).as_euler("zxz")[:, 0]
        phi2 = rot.from_euler("zxz", e2, degrees=True).as_euler("zxz")[:, 0]
        np.testing.assert_allclose(angular_score(e1, e2, "C5"), angular_score_for_c_symmetry(phi1, phi2, 5))

    def test_kind_changes_intermediate_scores(self):
        # Octahedron and cube corners are different marker sets: they agree at
        # 1 (see above) but generic pairs get slightly different scores.
        r1 = rot.random(50, random_state=9)
        r2 = rot.random(50, random_state=10)
        assert not np.allclose(angular_score(r1, r2, "O", kind="octahedron"), angular_score(r1, r2, "O", kind="cube"))

    def test_explicit_max_val_is_used(self):
        r1 = rot.random(20, random_state=12)
        r2 = rot.random(20, random_state=13)
        d_max = max_angular_mismatch("I")
        a = angular_score(r1, r2, "I", max_val=d_max)
        b = angular_score(r1, r2, "I", max_val=2 * d_max)
        # Doubling d_max halves 1 - score (up to clamping near 1).
        mask = a < 1
        np.testing.assert_allclose(1 - b[mask], (1 - a[mask]) / 2, atol=1e-5)

    def test_accepts_single_rotations_and_euler_angles(self):
        s = angular_score([0.0, 0.0, 0.0], [90.0, 45.0, 0.0], "T")
        assert s.shape == (1,)
        same = angular_score(rot.identity(), rot.from_euler("zxz", [90, 45, 0], degrees=True), "T")
        np.testing.assert_allclose(s, same)

    def test_invalid_inputs(self):
        r = rot.random(3, random_state=0)
        with pytest.raises(ValueError, match="same length"):
            angular_score(r, rot.random(2, random_state=1), "T")
        with pytest.raises(NotImplementedError, match="dihedral"):
            angular_score(r, r, "D2")
        with pytest.raises(ValueError, match="only applicable"):
            angular_score(r, r, 3, kind="cube")
        with pytest.raises(ValueError, match="does not match"):
            angular_score(r, r, "I", kind="cube")


# ---------------------------------------------------------------------------
# reduce_angle_grid (added 2026-10-01)
# ---------------------------------------------------------------------------


def _rot_angle_deg(m):
    return np.degrees(np.arccos(np.clip((np.trace(m) - 1) / 2, -1, 1)))


class TestReduceAngleGrid:
    """Keep about one orientation per set of symmetric look-alikes (R ~ R @ g)."""

    def test_identity_always_kept_in_global_mode(self):
        grid = rot.from_matrix(np.stack([np.eye(3)] + list(rot.random(50, random_state=1).as_matrix())))
        assert reduce_angle_grid(grid, "O")[0]

    @pytest.mark.parametrize("symmetry", ["C4", "D3", "T", "O", "I"])
    def test_global_keeps_exactly_one_of_each_exact_orbit(self, symmetry):
        # A grid made of complete symmetry orbits {R @ g}: with no margin,
        # exactly one member of each orbit lies in the kept slice.
        group = get_symmetry_rotations(symmetry)
        seeds = rot.random(20, random_state=2).as_matrix()
        grid = np.einsum("aij,gjk->agik", seeds, group).reshape(-1, 3, 3)
        keep = reduce_angle_grid(grid, symmetry).reshape(len(seeds), len(group))
        assert np.array_equal(keep.sum(axis=1), np.ones(len(seeds)))

    def test_global_kept_orientation_is_closest_to_reference(self):
        # The kept member of an orbit is the copy closest to the reference.
        group = get_symmetry_rotations("T")
        ref = rot.from_euler("zxz", [30, 50, 70], degrees=True).as_matrix()
        seed = rot.from_euler("zxz", [100, 20, -40], degrees=True).as_matrix()
        orbit = np.einsum("ij,gjk->gik", seed, group)
        kept = orbit[reduce_angle_grid(orbit, "T", reference=ref)][0]
        dists = [_rot_angle_deg(ref.T @ m) for m in orbit]
        assert _rot_angle_deg(ref.T @ kept) == pytest.approx(min(dists))

    def test_margin_keeps_more(self):
        grid = rot.random(2000, random_state=3)
        n0 = reduce_angle_grid(grid, "I").sum()
        n5 = reduce_angle_grid(grid, "I", margin_deg=5.0).sum()
        assert n0 < n5 < 2000

    def test_local_drops_only_orientations_with_a_kept_copy(self):
        # Local mode: every dropped orientation has a kept symmetric copy within
        # the tolerance, so nothing that was searched is lost.
        group = get_symmetry_rotations("D3")
        seeds = rot.random(30, random_state=4).as_matrix()
        jitter = rot.from_euler("z", 1.0, degrees=True).as_matrix()
        grid = np.concatenate([seeds, np.einsum("aij,jk,kl->ail", seeds, group[3], jitter)])
        keep = reduce_angle_grid(grid, "D3", local=True, tolerance_deg=2.0)
        for i in np.flatnonzero(~keep):
            copies = np.einsum("ij,gjk->gik", grid[i], group)
            gaps = [min(_rot_angle_deg(k.T @ c) for c in copies) for k in grid[keep]]
            assert min(gaps) <= 2.0 + 1e-6

    def test_local_with_zero_tolerance_keeps_distinct_orientations(self):
        grid = rot.random(100, random_state=5)
        assert reduce_angle_grid(grid, "O", local=True).all()

    def test_c1_keeps_everything(self):
        grid = rot.random(100, random_state=6)
        assert reduce_angle_grid(grid, "C1").all()
        assert reduce_angle_grid(grid, "C1", local=True, tolerance_deg=1.0).all()


# ---------------------------------------------------------------------------
# closest_symmetric_copy (added 2026-10-02)
# ---------------------------------------------------------------------------

from cryocat.utils.symmetry import closest_symmetric_copy, _make_group


def _brute_force_min_angle(r1, r2, symmetry):
    """Reference: smallest angle of R1^-1 @ R2 @ g over all group rotations g, pair by pair.

    Uses scipy's ``magnitude`` (a different formula from the trace used by the
    function under test), so both the minimum and the angle formula are checked.
    """
    gs = rot.from_matrix(get_symmetry_rotations(symmetry))
    return np.array([np.degrees(min((a.inv() * b * g).magnitude() for g in gs)) for a, b in zip(r1, r2)])


class TestClosestSymmetricCopy:
    """For each pair, the closest copy R2 @ g of rotation 2 to rotation 1 (template-side symmetry)."""

    @pytest.mark.parametrize("symm", ["C2", "C3", "C4", "C6", "D2", "D3", "T", "O", "I"])
    def test_matches_brute_force(self, symm):
        # Random pairs: the returned angle is the minimum over all group rotations ...
        r1 = rot.random(100, random_state=10)
        r2 = rot.random(100, random_state=11)
        angle, idx = closest_symmetric_copy(r1, r2, symm)
        np.testing.assert_allclose(angle, _brute_force_min_angle(r1, r2, symm), atol=1e-5)
        # ... and the returned index really is the copy that achieves it
        g = rot.from_matrix(get_symmetry_rotations(symm)[idx])
        np.testing.assert_allclose(np.degrees((r1.inv() * r2 * g).magnitude()), angle, atol=1e-5)

    @pytest.mark.parametrize("symm", ["C4", "D3", "T", "O", "I"])
    def test_symmetric_copies_have_zero_distance(self, symm):
        # R and R @ g look identical for every group rotation g -> distance 0
        gs = get_symmetry_rotations(symm)
        r = rot.random(1, random_state=12).as_matrix()
        r1 = rot.from_matrix(np.repeat(r, len(gs), axis=0))
        r2 = r1 * rot.from_matrix(gs)
        angle, _ = closest_symmetric_copy(r1, r2, symm)
        np.testing.assert_allclose(angle, 0.0, atol=1e-4)

    @pytest.mark.parametrize("symm", ["C3", "D2", "O"])
    def test_never_larger_than_without_symmetry(self, symm):
        # The identity is one of the copies, so the result is at most the C1 angle
        r1 = rot.random(200, random_state=13)
        r2 = rot.random(200, random_state=14)
        angle, _ = closest_symmetric_copy(r1, r2, symm)
        c1_angle = np.degrees((r1.inv() * r2).magnitude())
        assert np.all(angle <= c1_angle + 1e-6)

    def test_c1_is_plain_angle_and_identity_index(self):
        # C1 has only the identity: the plain angle between the rotations, index 0
        r1 = rot.random(50, random_state=15)
        r2 = rot.random(50, random_state=16)
        angle, idx = closest_symmetric_copy(r1, r2, "C1")
        np.testing.assert_allclose(angle, np.degrees((r1.inv() * r2).magnitude()), atol=1e-5)
        assert np.all(idx == 0)

    def test_single_rotations_and_euler_input(self):
        # Single Euler triples (degrees, zxz) are accepted; C4: spins 10 and 100 degrees look the same
        angle, idx = closest_symmetric_copy([10.0, 30.0, 0.0], [100.0, 30.0, 0.0], "C4")
        assert angle.shape == (1,) and idx.shape == (1,)
        np.testing.assert_allclose(angle, 0.0, atol=1e-4)

    def test_unequal_lengths_raise(self):
        # Pairs are compared one by one, so one rotation vs a stack is an error
        with pytest.raises(ValueError, match="same number of rotations"):
            closest_symmetric_copy(rot.random(1, random_state=0), rot.random(3, random_state=1), "C2")

    def test_chunking_does_not_change_result(self):
        # Splitting the work into chunks gives the same answer as one pass
        r1 = rot.random(57, random_state=17)
        r2 = rot.random(57, random_state=18)
        a_full, i_full = closest_symmetric_copy(r1, r2, "O")
        a_chunk, i_chunk = closest_symmetric_copy(r1, r2, "O", chunk_size=10)
        np.testing.assert_array_equal(a_full, a_chunk)
        np.testing.assert_array_equal(i_full, i_chunk)


class TestMakeGroup:
    """``_make_group`` builds the canonical group used by get_symmetry_rotations and reduce_angle_grid."""

    @pytest.mark.parametrize("symm, n_rot", [(4, 4), ("C6", 6), ("D3", 6), ("T", 12), ("O", 24), ("I", 60)])
    def test_order_and_identity_first(self, symm, n_rot):
        # Number of rotations of the group, identity first
        mats = _make_group(symm).matrices
        assert mats.shape == (n_rot, 3, 3)
        np.testing.assert_allclose(mats[0], np.eye(3), atol=1e-12)

    @pytest.mark.parametrize("symm", ["C5", "D4", "T", "O", "I"])
    def test_same_as_get_symmetry_rotations(self, symm):
        # Refactor guard: get_symmetry_rotations (default z axis) returns the same matrices
        np.testing.assert_array_equal(_make_group(symm).matrices, get_symmetry_rotations(symm))
