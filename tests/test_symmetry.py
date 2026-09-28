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
    SymmGroup,
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
