import math
import sys
import types
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.spatial.transform import Rotation as R


from cryocat.analysis import nnana
from cryocat.core import cryomotl
from cryocat.utils import geom, symmetry
from cryocat.analysis.tango import (
    Particle,
    SymmParticle,
    Descriptor,
    TwistDescriptor,
    SHOTDescriptor,
    AlphaComplexDescriptor,
    PLComplexDescriptor,
    AngularScoreNN,
    _check_numeric_param,
    _inplane_angles_from_rotations,
)

# ===========================================================================
# _check_numeric_param (module-level helper, was untestable as a closure)
# ===========================================================================


class TestCheckNumericParam:
    """Direct tests for _check_numeric_param.

    None of these tests would have failed *before this refactor* (the move from
    closure to module level): after the GQ2 numpy fix the logic was already
    correct for all tested inputs.  The value here is testability — the closure
    could not be imported or called directly — and eliminating the per-particle
    reconstruction cost.
    """

    # ── return_int=True (default) ──────────────────────────────────────────

    def test_python_int_returns_int(self):
        assert _check_numeric_param(3, "x") == 3
        assert isinstance(_check_numeric_param(3, "x"), int)

    def test_python_float_truncates_to_int(self):
        assert _check_numeric_param(3.7, "x") == 3
        assert isinstance(_check_numeric_param(3.7, "x"), int)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_numpy_integer_returns_int(self, dtype):
        result = _check_numeric_param(dtype(5), "x")
        assert result == 5
        assert isinstance(result, int)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_numpy_float_casts_to_int(self, dtype):
        result = _check_numeric_param(dtype(7.9), "x")
        assert result == 7
        assert isinstance(result, int)

    # ── return_int=False ───────────────────────────────────────────────────

    def test_python_float_returned_unchanged(self):
        result = _check_numeric_param(2.5, "x", return_int=False)
        assert result == pytest.approx(2.5)

    def test_numpy_float32_returned_as_is(self):
        val = np.float32(1.5)
        result = _check_numeric_param(val, "x", return_int=False)
        assert result == pytest.approx(1.5)

    # ── None ───────────────────────────────────────────────────────────────

    def test_none_returns_none(self):
        assert _check_numeric_param(None, "x") is None
        assert _check_numeric_param(None, "x", return_int=False) is None

    # ── invalid types raise TypeError ──────────────────────────────────────

    def test_string_raises_type_error(self):
        with pytest.raises(TypeError, match="has to be a float or an int"):
            _check_numeric_param("bad", "myfield")

    def test_list_raises_type_error(self):
        with pytest.raises(TypeError):
            _check_numeric_param([1, 2], "x")

    def test_error_message_contains_name(self):
        with pytest.raises(TypeError, match="tomo_id"):
            _check_numeric_param("bad", "tomo_id")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def identity_particle():
    return Particle.identity()


@pytest.fixture
def simple_particle():
    rot = np.eye(3)
    pos = np.array([1.0, 2.0, 3.0])
    return Particle(rot, pos, tomo_id=1, particle_id=5)


@pytest.fixture
def rotated_particle():
    # 90° rotation around z-axis
    angles = np.array([90.0, 0.0, 0.0])
    pos = np.array([0.0, 0.0, 0.0])
    return Particle(angles, pos, degrees=True)


# ===========================================================================
# Particle.__init__
# ===========================================================================


class TestParticleInit:
    def test_euler_angles_degrees(self):
        p = Particle(np.array([90.0, 0.0, 0.0]), np.zeros(3), degrees=True)
        assert p.rotation.shape == (3, 3)
        assert p.position.shape == (3,)

    def test_euler_angles_radians(self):
        angles = np.array([np.pi / 2, 0.0, 0.0])
        p = Particle(angles, np.zeros(3), degrees=False)
        assert p.rotation.shape == (3, 3)

    def test_rotation_matrix(self):
        rot = R.from_euler("zxz", [30, 45, 60], degrees=True).as_matrix()
        p = Particle(rot, np.zeros(3))
        np.testing.assert_allclose(p.rotation, rot, atol=1e-12)

    def test_quaternion(self):
        q = R.from_euler("zxz", [45, 0, 0], degrees=True).as_quat()
        p = Particle(q, np.zeros(3))
        assert p.rotation.shape == (3, 3)

    def test_scipy_rotation_object(self):
        r = R.from_euler("zxz", [10, 20, 30], degrees=True)
        p = Particle(r, np.zeros(3))
        np.testing.assert_allclose(p.rotation, r.as_matrix(), atol=1e-12)

    def test_position_stored_as_1d(self):
        p = Particle(np.eye(3), np.array([[1.0, 2.0, 3.0]]))
        assert p.position.shape == (3,)

    def test_tomo_id_stored_as_int(self):
        p = Particle(np.eye(3), np.zeros(3), tomo_id=3.7)
        assert p.tomo_id == 3
        assert isinstance(p.tomo_id, int)

    def test_particle_id_stored_as_int(self):
        p = Particle(np.eye(3), np.zeros(3), particle_id=7.0)
        assert p.id == 7

    def test_invalid_rotation_raises(self):
        with pytest.raises((ValueError, TypeError)):
            Particle("bad_rotation", np.zeros(3))

    def test_position_wrong_type_raises(self):
        with pytest.raises(TypeError):
            Particle(np.eye(3), [1.0, 2.0, 3.0])

    def test_position_wrong_size_raises(self):
        with pytest.raises(TypeError):
            Particle(np.eye(3), np.array([1.0, 2.0]))

    def test_tomo_id_wrong_type_raises(self):
        with pytest.raises(TypeError):
            Particle(np.eye(3), np.zeros(3), tomo_id="bad")

    def test_particle_id_wrong_type_raises(self):
        with pytest.raises(TypeError):
            Particle(np.eye(3), np.zeros(3), particle_id="bad")

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32])
    def test_tomo_id_numpy_scalar_accepted(self, dtype):
        val = dtype(42)
        p = Particle(np.eye(3), np.zeros(3), tomo_id=val)
        assert p.tomo_id == 42
        assert isinstance(p.tomo_id, int)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32])
    def test_particle_id_numpy_scalar_accepted(self, dtype):
        val = dtype(7)
        p = Particle(np.eye(3), np.zeros(3), particle_id=val)
        assert p.id == 7
        assert isinstance(p.id, int)


# ===========================================================================
# Particle.identity
# ===========================================================================


class TestParticleIdentity:
    def test_rotation_is_eye(self, identity_particle):
        np.testing.assert_allclose(identity_particle.rotation, np.eye(3), atol=1e-12)

    def test_position_is_zero(self, identity_particle):
        np.testing.assert_allclose(identity_particle.position, np.zeros(3), atol=1e-12)


# ===========================================================================
# Particle.__eq__ and __hash__
# ===========================================================================


class TestParticleEq:
    def test_equal_to_self(self, simple_particle):
        assert simple_particle == simple_particle

    def test_identity_equal(self, identity_particle):
        other = Particle.identity()
        assert identity_particle == other

    def test_different_position(self):
        p1 = Particle(np.eye(3), np.array([1.0, 0.0, 0.0]))
        p2 = Particle(np.eye(3), np.array([2.0, 0.0, 0.0]))
        assert not (p1 == p2)

    def test_different_rotation(self):
        p1 = Particle(np.array([0.0, 0.0, 0.0]), np.zeros(3))
        p2 = Particle(np.array([90.0, 0.0, 0.0]), np.zeros(3))
        assert not (p1 == p2)

    def test_invalid_comparison_raises(self, simple_particle):
        with pytest.raises(ValueError):
            simple_particle == "not a particle"

    def test_hashable(self, identity_particle):
        s = {identity_particle}
        assert identity_particle in s


# ===========================================================================
# Particle.inv
# ===========================================================================


class TestParticleInv:
    def test_identity_inv_is_identity(self, identity_particle):
        inv = identity_particle.inv()
        assert inv == identity_particle

    def test_p_times_inv_is_identity(self, simple_particle, identity_particle):
        result = simple_particle * simple_particle.inv()
        assert result == identity_particle

    def test_inv_times_p_is_identity(self, simple_particle, identity_particle):
        result = simple_particle.inv() * simple_particle
        assert result == identity_particle

    def test_double_inv_is_self(self, simple_particle):
        assert simple_particle.inv().inv() == simple_particle


# ===========================================================================
# Particle.__mul__
# ===========================================================================


class TestParticleMul:
    def test_identity_times_p_is_p(self, identity_particle, simple_particle):
        result = identity_particle * simple_particle
        assert result == simple_particle

    def test_p_times_identity_is_p(self, identity_particle, simple_particle):
        result = simple_particle * identity_particle
        assert result == simple_particle

    def test_invalid_mul_raises(self, simple_particle):
        with pytest.raises(ValueError):
            simple_particle * 3.0


# ===========================================================================
# Particle.scale
# ===========================================================================


class TestParticleScale:
    def test_scale_overwrite_true(self):
        p = Particle(np.eye(3), np.array([1.0, 2.0, 3.0]))
        p.scale(2.0, overwrite=True)
        np.testing.assert_allclose(p.position, [2.0, 4.0, 6.0])

    def test_scale_overwrite_false_returns_new(self):
        p = Particle(np.eye(3), np.array([1.0, 2.0, 3.0]))
        p_scaled = p.scale(3.0, overwrite=False)
        np.testing.assert_allclose(p_scaled.position, [3.0, 6.0, 9.0])
        np.testing.assert_allclose(p.position, [1.0, 2.0, 3.0])  # original unchanged

    def test_scale_zero(self):
        p = Particle(np.eye(3), np.array([1.0, 2.0, 3.0]))
        p_scaled = p.scale(0, overwrite=False)
        np.testing.assert_allclose(p_scaled.position, [0.0, 0.0, 0.0])

    def test_scale_invalid_type_raises(self):
        p = Particle(np.eye(3), np.zeros(3))
        with pytest.raises(TypeError):
            p.scale("two")


# ===========================================================================
# Particle.distance
# ===========================================================================


class TestParticleDistance:
    def test_position_distance_zero_same(self, identity_particle):
        d = identity_particle.distance(identity_particle, mode="position")
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_position_distance_euclidean(self):
        p1 = Particle(np.eye(3), np.array([0.0, 0.0, 0.0]))
        p2 = Particle(np.eye(3), np.array([3.0, 4.0, 0.0]))
        d = p1.distance(p2, mode="position")
        assert d == pytest.approx(5.0, rel=1e-6)

    def test_orientation_distance_zero_same(self, identity_particle):
        d = identity_particle.distance(identity_particle, mode="orientation")
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_orientation_distance_positive(self):
        p1 = Particle(np.array([0.0, 0.0, 0.0]), np.zeros(3))
        p2 = Particle(np.array([90.0, 0.0, 0.0]), np.zeros(3))
        d = p1.distance(p2, mode="orientation")
        assert d > 0.0

    def test_orientation_distance_degrees_flag(self):
        p1 = Particle(np.array([0.0, 0.0, 0.0]), np.zeros(3))
        p2 = Particle(np.array([90.0, 0.0, 0.0]), np.zeros(3))
        d_rad = p1.distance(p2, mode="orientation", degrees=False)
        d_deg = p1.distance(p2, mode="orientation", degrees=True)
        assert d_deg == pytest.approx(np.degrees(d_rad), rel=1e-6)

    def test_mixed_distance_zero_same(self, identity_particle):
        d = identity_particle.distance(identity_particle, mode="mixed")
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_invalid_mode_raises(self, identity_particle):
        with pytest.raises(ValueError):
            identity_particle.distance(identity_particle, mode="bad_mode")

    def test_invalid_input_raises(self, simple_particle):
        with pytest.raises(ValueError):
            simple_particle.distance("not a particle")


# ===========================================================================
# Particle.in_plane_angle
# ===========================================================================


class TestParticleInPlaneAngle:
    def test_identity_angle_zero(self, identity_particle):
        angle = identity_particle.in_plane_angle(degrees=True)
        assert angle == pytest.approx(0.0, abs=1e-10)

    def test_90_degree_rotation(self):
        p = Particle(np.array([90.0, 0.0, 0.0]), np.zeros(3), degrees=True)
        angle = p.in_plane_angle(degrees=True)
        assert angle == pytest.approx(90.0, rel=1e-6)

    def test_radians_flag(self):
        p = Particle(np.array([45.0, 0.0, 0.0]), np.zeros(3), degrees=True)
        angle_deg = p.in_plane_angle(degrees=True)
        angle_rad = p.in_plane_angle(degrees=False)
        assert angle_deg == pytest.approx(np.degrees(angle_rad), rel=1e-6)


# ===========================================================================
# Particle.random and Particle.__str__
# ===========================================================================


class TestParticleMisc:
    def test_random_returns_particle(self):
        p = Particle.random((0, 10), (0, 10), (0, 10))
        assert isinstance(p, Particle)
        assert p.rotation.shape == (3, 3)
        assert p.position.shape == (3,)

    def test_str_returns_string(self, simple_particle):
        s = str(simple_particle)
        assert isinstance(s, str)
        assert "Tomogram" in s


# ===========================================================================
# SymmParticle
# ===========================================================================


class TestSymmParticle:
    def test_cyclic_symmetry_integer(self):
        # SymmParticle.__init__ has no degrees param; pass a rotation matrix instead
        sp = SymmParticle(np.eye(3), np.zeros(3), symm=4)
        assert sp.category == 4

    def test_cyclic_symmetry_string(self):
        sp = SymmParticle(np.eye(3), np.zeros(3), symm="c4")
        assert sp.category == 4

    def test_invalid_symmetry_raises(self):
        with pytest.raises(ValueError):
            SymmParticle(np.eye(3), np.zeros(3), symm="invalid_symm")

    def test_max_dissimilarity_cyclic(self):
        n = 6
        sp = SymmParticle(np.eye(3), np.zeros(3), symm=n)
        assert sp.max_dissimilarity() == pytest.approx(np.pi / n, rel=1e-6)

    def test_max_dissimilarity_cyclic_2(self):
        sp = SymmParticle(np.eye(3), np.zeros(3), symm=2)
        assert sp.max_dissimilarity() == pytest.approx(np.pi / 2, rel=1e-6)

    def test_equip_symmetry(self):
        p = Particle(np.array([0.0, 0.0, 0.0]), np.zeros(3))
        sp = SymmParticle.equip_symmetry(p, 4)
        assert isinstance(sp, SymmParticle)
        assert sp.category == 4

    def test_platonic_tetrahedron(self):
        # category is now the Symmetry group letter; kind holds the specific solid.
        sp = SymmParticle(np.eye(3), np.zeros(3), symm="T")
        assert sp.category == "T"
        assert sp.kind == "tetrahedron"

    def test_platonic_octahedron(self):
        sp = SymmParticle(np.eye(3), np.zeros(3), symm="O")
        assert sp.category == "O"
        assert sp.kind == "octahedron"

    def test_cyclic_symmetry_has_no_kind(self):
        sp = SymmParticle(np.eye(3), np.zeros(3), symm=4)
        assert sp.kind is None


# ===========================================================================
# SymmParticle: Symmetry type consistency (added 2026-09-25)
# ===========================================================================


class TestSymmParticleSymmetryType:
    """SymmParticle accepts only the canonical Symmetry type (as_symmetry); the
    legacy free-text words ('tetra', 'octa', 'cube', 'ico', 'dodeca') were
    removed 2026-09-28 and now raise a ValueError naming their canonical
    replacement (see TestLegacySymmWordsRejected below). A `kind` parameter
    (mirrors SymmGroup.to_polyhedron) picks between the two solids sharing a
    group letter (O: octahedron/cube, I: icosahedron/dodecahedron).
    """

    def test_bare_letter_uses_default_kind(self):
        # Default kind matches SymmGroup.to_polyhedron()'s defaults: Octahedron / Icosahedron.
        assert SymmParticle(np.eye(3), np.zeros(3), symm="O").kind == "octahedron"
        assert SymmParticle(np.eye(3), np.zeros(3), symm="I").kind == "icosahedron"

    def test_kind_selects_alternate_solid(self):
        sp_cube = SymmParticle(np.eye(3), np.zeros(3), symm="O", kind="cube")
        sp_octa = SymmParticle(np.eye(3), np.zeros(3), symm="O", kind="octahedron")
        assert sp_cube.category == sp_octa.category == "O"
        assert sp_cube.kind == "cube"
        assert sp_cube.solid.shape != sp_octa.solid.shape  # 8 vs 6 vertices

    def test_kind_for_cyclic_raises(self):
        with pytest.raises(ValueError, match="only applicable to Platonic"):
            SymmParticle(np.eye(3), np.zeros(3), symm=4, kind="cube")

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown kind"):
            SymmParticle(np.eye(3), np.zeros(3), symm="O", kind="not_a_solid")

    def test_dihedral_symmetry_not_supported(self):
        with pytest.raises(NotImplementedError, match="[Dd]ihedral"):
            SymmParticle(np.eye(3), np.zeros(3), symm="D3")

    def test_similarity_symm_requires_matching_kind_not_just_category(self):
        # Same category ("O") but different kind: must be rejected as incomparable,
        # not silently compared (which would fail/mislead since vertex counts differ).
        sp_cube = SymmParticle(np.eye(3), np.zeros(3), symm="O", kind="cube")
        sp_octa = SymmParticle(np.eye(3), np.zeros(3), symm="O", kind="octahedron")
        with pytest.raises(ValueError, match="don't match"):
            sp_cube.similarity_symm(sp_octa)

    def test_equip_symmetry_forwards_kind(self):
        p = Particle(np.array([0.0, 0.0, 0.0]), np.zeros(3))
        sp = SymmParticle.equip_symmetry(p, "O", kind="cube")
        assert sp.category == "O"
        assert sp.kind == "cube"


# ===========================================================================
# SymmParticle: legacy word rejection (added 2026-09-28)
# ===========================================================================


class TestLegacySymmWordsRejected:
    """The free-text words 'tetra'/'octa'/'cube'/'ico'/'dodeca' formerly
    accepted as `symm` (and kept as aliases in the 2026-09-25 refactor, see
    TestSymmParticleSymmetryType) were removed 2026-09-28: tango.py now
    requires the canonical Symmetry form exclusively. Each legacy word must
    raise a ValueError naming its canonical replacement, regardless of
    whether `kind` is also given.
    """

    @pytest.mark.parametrize(
        "legacy, hint",
        [
            ("tetra", "symm='T'"),
            ("octa", "symm='O'"),
            ("cube", "symm='O'"),
            ("ico", "symm='I'"),
            ("dodeca", "symm='I'"),
        ],
    )
    def test_legacy_word_raises_with_migration_hint(self, legacy, hint):
        with pytest.raises(ValueError, match="no longer accepted"):
            SymmParticle(np.eye(3), np.zeros(3), symm=legacy)
        # The error message must name the canonical replacement, not just reject.
        try:
            SymmParticle(np.eye(3), np.zeros(3), symm=legacy)
        except ValueError as exc:
            assert hint in str(exc)

    def test_legacy_word_raises_even_with_matching_kind(self):
        # Legacy-word rejection happens before kind is ever consulted, so a
        # kind that would have been consistent with the old word still raises.
        with pytest.raises(ValueError, match="no longer accepted"):
            SymmParticle(np.eye(3), np.zeros(3), symm="cube", kind="cube")

    def test_legacy_word_raises_through_get_symm_parameters(self):
        # Same rejection must be visible through the TwistDescriptor entry point,
        # not just at the SymmParticle level.
        with pytest.raises(ValueError, match="no longer accepted"):
            TwistDescriptor.get_symm_parameters(_make_single_particle_motl(), symm="ico")


def _make_single_particle_motl():
    """Minimal one-row Motl usable by get_symm_parameters/convert_to_particle_list."""
    from cryocat.core.cryomotl import Motl

    df = pd.DataFrame({col: [0.0] for col in Motl.motl_columns})
    df["subtomo_id"] = [1.0]
    df["tomo_id"] = [1.0]
    return Motl(df)


class TestGetSymmParameters:
    """TwistDescriptor.get_symm_parameters, now Symmetry-typed with a `kind` param."""

    def test_none_symm_returns_none_none(self):
        assert TwistDescriptor.get_symm_parameters(_make_single_particle_motl(), symm=None) == (None, None)

    def test_cyclic_returns_int_category(self):
        max_dis, category = TwistDescriptor.get_symm_parameters(_make_single_particle_motl(), symm=4)
        assert category == 4
        assert max_dis == pytest.approx(np.pi / 4, rel=1e-6)

    def test_platonic_returns_letter_category(self):
        _, category = TwistDescriptor.get_symm_parameters(_make_single_particle_motl(), symm="T")
        assert category == "T"

    def test_kind_selects_alternate_solid(self):
        _, category = TwistDescriptor.get_symm_parameters(_make_single_particle_motl(), symm="O", kind="cube")
        assert category == "O"  # category is the group letter regardless of kind

    def test_kind_forwarded_to_convert_to_particle_list(self):
        # Canonical form + kind must resolve the same way at this level too.
        _, category = TwistDescriptor.get_symm_parameters(_make_single_particle_motl(), symm="I", kind="dodecahedron")
        assert category == "I"


# ===========================================================================
# Descriptor static methods
# ===========================================================================


class TestDescriptor:
    def test_remove_nans_row(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        result = Descriptor.remove_nans(df, "row")
        assert len(result) == 2
        assert not result.isnull().any().any()

    def test_remove_nans_column(self):
        df = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
        result = Descriptor.remove_nans(df, "column")
        assert "a" not in result.columns
        assert "b" in result.columns

    def test_remove_nans_invalid_axis_raises(self):
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError):
            Descriptor.remove_nans(df, "diagonal")

    def test_build_descriptor_feature_map(self):
        desc_list = ["TwistDescriptor", "SHOTDescriptor", "NotADescriptor"]
        feat_list = ["NNCountTwist", "AngularScoreStatsTwist", "CountSHOT"]
        result = Descriptor.build_descriptor_feature_map(desc_list, feat_list)
        assert "TwistDescriptor" in result
        assert "NNCountTwist" in result["TwistDescriptor"]
        assert "AngularScoreStatsTwist" in result["TwistDescriptor"]
        assert "SHOTDescriptor" in result
        assert "CountSHOT" in result["SHOTDescriptor"]
        assert "NotADescriptor" not in result  # excluded — no matching features (empty matches are dropped)

    def test_build_descriptor_feature_map_no_match(self):
        desc_list = ["TwistDescriptor"]
        feat_list = ["SomeOtherFeature"]
        result = Descriptor.build_descriptor_feature_map(desc_list, feat_list)
        assert "TwistDescriptor" not in result

    def test_build_feature_descriptor_map(self):
        feat_list = ["NNCountTwist", "CountSHOT"]
        desc_list = ["TwistDescriptor", "SHOTDescriptor"]
        result = Descriptor.build_feature_descriptor_map(feat_list, desc_list)
        assert result["NNCountTwist"] == "TwistDescriptor"
        assert result["CountSHOT"] == "SHOTDescriptor"


# ===========================================================================
# Particle.tangent_at_identity
# ===========================================================================


class TestParticleTangentAtIdentity:
    def test_identity_returns_zero_vector(self, identity_particle):
        t = identity_particle.tangent_at_identity()
        assert t.shape == (6,)
        np.testing.assert_allclose(t, np.zeros(6), atol=1e-10)

    def test_returns_6d_finite_vector(self, simple_particle):
        t = simple_particle.tangent_at_identity()
        assert t.shape == (6,)
        assert np.isfinite(t).all()


# ===========================================================================
# Particle.twist_vector
# ===========================================================================


class TestParticleTwistVector:
    def test_identity_twist_self_is_zero(self, identity_particle):
        tv = identity_particle.twist_vector(identity_particle)
        np.testing.assert_allclose(tv, np.zeros(6), atol=1e-10)

    def test_returns_6d_vector(self, simple_particle, identity_particle):
        tv = simple_particle.twist_vector(identity_particle)
        assert tv.shape == (6,)
        assert np.isfinite(tv).all()

    def test_invalid_input_raises(self, simple_particle):
        with pytest.raises(ValueError):
            simple_particle.twist_vector("not a particle")


# ===========================================================================
# Particle.tangent_subspace_projection
# ===========================================================================


class TestParticleTangentSubspaceProjection:
    def test_orientation_returns_3d(self, identity_particle, simple_particle):
        proj = identity_particle.tangent_subspace_projection(simple_particle, mode="orientation")
        assert proj.shape == (3,)

    def test_position_returns_3d(self, identity_particle, simple_particle):
        proj = identity_particle.tangent_subspace_projection(simple_particle, mode="position")
        assert proj.shape == (3,)

    def test_mixed_returns_6d(self, identity_particle, simple_particle):
        proj = identity_particle.tangent_subspace_projection(simple_particle, mode="mixed")
        assert proj.shape == (6,)

    def test_invalid_mode_raises(self, identity_particle, simple_particle):
        with pytest.raises(ValueError):
            identity_particle.tangent_subspace_projection(simple_particle, mode="bad_mode")

    def test_invalid_input_raises(self, identity_particle):
        with pytest.raises(ValueError):
            identity_particle.tangent_subspace_projection("not a particle")

    @pytest.mark.parametrize("mode", ["orientation", "position", "mixed"])
    def test_all_valid_modes(self, identity_particle, simple_particle, mode):
        proj = identity_particle.tangent_subspace_projection(simple_particle, mode=mode)
        assert np.isfinite(proj).all()


# ===========================================================================
# Particle.add_noise
# ===========================================================================


class TestParticleAddNoise:
    def test_orientation_noise_returns_particle(self, simple_particle):
        noisy = simple_particle.add_noise(noise_level=0.01, mode="orientation")
        assert isinstance(noisy, Particle)
        assert noisy.rotation.shape == (3, 3)

    def test_position_noise_changes_position(self, simple_particle):
        rng = np.random.default_rng(0)
        np.random.seed(0)
        noisy = simple_particle.add_noise(noise_level=5.0, mode="position")
        assert isinstance(noisy, Particle)

    def test_mixed_noise_returns_particle(self, simple_particle):
        noisy = simple_particle.add_noise(noise_level=0.01, mode="mixed")
        assert isinstance(noisy, Particle)

    def test_invalid_mode_raises(self, simple_particle):
        with pytest.raises(ValueError):
            simple_particle.add_noise(mode="bad_mode")

    @pytest.mark.parametrize("mode", ["orientation", "position", "mixed"])
    def test_valid_modes(self, simple_particle, mode):
        noisy = simple_particle.add_noise(noise_level=0.01, mode=mode)
        assert isinstance(noisy, Particle)


# ===========================================================================
# SymmParticle.similarity_symm
# ===========================================================================


class TestSymmParticleSimilaritySymm:
    def test_self_similarity_is_one(self):
        sp = SymmParticle(np.eye(3), np.zeros(3), symm=4)
        assert sp.similarity_symm(sp) == pytest.approx(1.0, rel=1e-6)

    def test_mismatched_symmetry_raises(self):
        sp4 = SymmParticle(np.eye(3), np.zeros(3), symm=4)
        sp6 = SymmParticle(np.eye(3), np.zeros(3), symm=6)
        with pytest.raises(ValueError):
            sp4.similarity_symm(sp6)

    def test_similarity_between_zero_and_one(self):
        sp1 = SymmParticle(np.eye(3), np.zeros(3), symm=4)
        angles = np.array([30.0, 0.0, 0.0])
        rot = R.from_euler("zxz", angles, degrees=True).as_matrix()
        sp2 = SymmParticle(rot, np.zeros(3), symm=4)
        sim = sp1.similarity_symm(sp2)
        assert 0.0 <= sim <= 1.0

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_identity_similarity_is_one_for_various_n(self, n):
        sp = SymmParticle(np.eye(3), np.zeros(3), symm=n)
        assert sp.similarity_symm(sp) == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.parametrize("kind", ["icosahedron", "dodecahedron"])
    def test_icosahedral_equivalent_orientations_score_one(self, kind):
        # A particle turned by any of the 60 icosahedral symmetry rotations is
        # indistinguishable from the original, so both the icosahedron and the
        # dodecahedron ("I", kind="icosahedron"/"dodecahedron") must score 1.
        # Regression test for the Dodecahedron alignment fix (2026-09-24): the
        # dodecahedron previously scored 1 for only 12 of the 60.
        from cryocat.utils.symmetry import IcosahedralGroup

        ref = SymmParticle(np.eye(3), np.zeros(3), symm="I", kind=kind)
        for g in IcosahedralGroup().matrices:
            other = SymmParticle(g, np.zeros(3), symm="I", kind=kind)
            assert ref.similarity_symm(other) == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# SymmParticle.max_dissimilarity fix and similarity_symm delegation (2026-09-29)
# ===========================================================================

_PLATONIC_CASES = [("T", None), ("O", "octahedron"), ("O", "cube"), ("I", "icosahedron"), ("I", "dodecahedron")]


class TestSymmParticleMaxDissimilarityFix:
    """max_dissimilarity() used "180 deg minus the angle between two hand-picked
    corners", correct only for the tetrahedron. It now returns the true largest
    mismatch (corner -> nearest face centre) via symmetry.max_angular_mismatch.
    """

    @pytest.mark.parametrize(
        "symm, kind, expected_deg",
        [
            ("T", None, 70.5288),  # unchanged by the fix
            ("O", "octahedron", 54.7356),  # was 90.00
            ("O", "cube", 54.7356),  # was 109.47
            ("I", "icosahedron", 37.3774),  # was 116.57
            ("I", "dodecahedron", 37.3774),  # was 138.19
        ],
    )
    def test_platonic_values_are_true_maximum(self, symm, kind, expected_deg):
        sp = SymmParticle(np.eye(3), np.zeros(3), symm=symm, kind=kind)
        assert np.degrees(sp.max_dissimilarity()) == pytest.approx(expected_deg, abs=1e-3)

    @pytest.mark.parametrize("n", [2, 3, 5, 6])
    def test_cyclic_value_unchanged(self, n):
        # pi/n, exactly as before the fix.
        assert SymmParticle(np.eye(3), np.zeros(3), symm=n).max_dissimilarity() == pytest.approx(np.pi / n)

    def test_independent_of_particle_and_custom_rotation(self):
        # d_max is a property of the shape, not of how the particle is turned;
        # the old code read corners from the turned solid, the new one must not
        # depend on rotation or custom_rot at all.
        r = R.from_euler("zxz", [10, 60, 5], degrees=True).as_matrix()
        c = R.from_euler("zxz", [20, 35, 50], degrees=True)
        a = SymmParticle(np.eye(3), np.zeros(3), symm="I", kind="dodecahedron")
        b = SymmParticle(r, np.zeros(3), symm="I", kind="dodecahedron", custom_rot=c)
        assert a.max_dissimilarity() == pytest.approx(b.max_dissimilarity())

    @pytest.mark.parametrize("symm, kind", _PLATONIC_CASES)
    def test_worst_case_now_scores_zero(self, symm, kind):
        # Turning one particle so that a corner lands on the other's nearest
        # face centre is the worst possible mismatch, so the default score is 0.
        # With the old, too-large maximum O/I could never reach 0.
        solid = symmetry.SYMMETRY_GROUPS[symm]().to_polyhedron(kind=kind)
        v = solid.vertices[0] / np.linalg.norm(solid.vertices[0])
        centres = solid.faces / np.linalg.norm(solid.faces, axis=1, keepdims=True)
        f = centres[np.argmax(centres @ v)]
        # Rotation about the axis perpendicular to both, taking v onto f.
        axis = np.cross(v, f)
        turn = R.from_rotvec(axis / np.linalg.norm(axis) * np.arccos(np.clip(v @ f, -1, 1))).as_matrix()
        ref = SymmParticle(np.eye(3), np.zeros(3), symm=symm, kind=kind)
        other = SymmParticle(turn, np.zeros(3), symm=symm, kind=kind)
        assert ref.similarity_symm(other) == pytest.approx(0.0, abs=1e-6)


class TestSymmParticleSimilarityDelegation:
    """similarity_symm now delegates to symmetry.angular_score (single source of
    truth). Scores must equal the previous direct Hausdorff computation on the
    stored solids, apart from the documented clamping to [0, 1].
    """

    @staticmethod
    def _old_score(a, b, max_val):
        # The pre-2026-09-29 implementation, kept here as the reference.
        return 1 - geom.hausdorff_distance_sphere(a.solid, b.solid) / max_val

    # Cyclic cases (4, 6) removed 2026-10-09: for C_n the stored solid is a flat polygon
    # turned by phi, i.e. the phi-only score replaced by the aligned-z-axes score. They are
    # covered by the two cyclic tests below (new reference; old one where it is valid).
    @pytest.mark.parametrize("symm, kind", _PLATONIC_CASES)
    def test_matches_previous_implementation(self, symm, kind):
        rots = R.random(30, random_state=3).as_matrix()
        ref = SymmParticle(rots[0], np.zeros(3), symm=symm, kind=kind)
        for m in rots[1:]:
            other = SymmParticle(m, np.zeros(3), symm=symm, kind=kind)
            expected = np.clip(self._old_score(ref, other, ref.max_dissimilarity()), 0.0, 1.0)
            assert ref.similarity_symm(other) == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize("symm", [4, 6])
    def test_cyclic_matches_aligned_spin(self, symm):
        # Same 30 random orientations as above: the cyclic score equals geom's polygon
        # score for "this particle at 0, the other at the spin left after tilting its
        # z-axis onto this one's".
        rots = R.random(30, random_state=3)
        ref = SymmParticle(rots[0].as_matrix(), np.zeros(3), symm=symm)
        for r in rots[1:]:
            other = SymmParticle(r.as_matrix(), np.zeros(3), symm=symm)
            spin = geom.inplane_angle_after_alignment(rots[0], r)
            expected = geom.angular_score_for_c_symmetry([0.0], spin, symm, ref.max_dissimilarity())[0]
            assert ref.similarity_symm(other) == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize("symm", [4, 6])
    def test_cyclic_same_z_axis_matches_previous_implementation(self, symm):
        # Where the old reference is valid (all particles share one z-axis: fixed theta
        # and psi, random phi), the score must still equal the old flat-polygon score.
        phis = np.random.default_rng(symm).uniform(-180, 180, 30)
        rots = R.from_euler("zxz", np.column_stack([phis, np.full(30, 50.0), np.full(30, 20.0)]), degrees=True)
        ref = SymmParticle(rots[0].as_matrix(), np.zeros(3), symm=symm)
        for r in rots[1:]:
            other = SymmParticle(r.as_matrix(), np.zeros(3), symm=symm)
            expected = np.clip(self._old_score(ref, other, ref.max_dissimilarity()), 0.0, 1.0)
            assert ref.similarity_symm(other) == pytest.approx(expected, abs=1e-5)

    def test_matches_previous_implementation_with_custom_rot(self):
        # Delegation rebuilds each solid's orientation as rotation @ custom_rot;
        # particles may even carry different custom_rot values.
        c1 = R.from_euler("zxz", [20, 35, 50], degrees=True)
        c2 = R.from_euler("z", -90, degrees=True)
        rots = R.random(20, random_state=4).as_matrix()
        ref = SymmParticle(rots[0], np.zeros(3), symm="I", kind="dodecahedron", custom_rot=c1)
        for m in rots[1:]:
            other = SymmParticle(m, np.zeros(3), symm="I", kind="dodecahedron", custom_rot=c2)
            expected = np.clip(self._old_score(ref, other, ref.max_dissimilarity()), 0.0, 1.0)
            assert ref.similarity_symm(other) == pytest.approx(expected, abs=1e-5)

    def test_custom_rot_is_stored_as_matrix(self):
        c = R.from_euler("z", 30, degrees=True)
        assert SymmParticle(np.eye(3), np.zeros(3), symm="O").custom_rot is None
        sp = SymmParticle(np.eye(3), np.zeros(3), symm="O", custom_rot=c)
        np.testing.assert_allclose(sp.custom_rot, c.as_matrix())

    def test_explicit_max_smaller_than_mismatch_clamps_to_zero(self):
        # Documented behaviour change: with a user-given max below the actual
        # mismatch, the old code returned a negative number; now 0.
        ref = SymmParticle(np.eye(3), np.zeros(3), symm=2)
        other = SymmParticle(R.from_euler("z", 80, degrees=True).as_matrix(), np.zeros(3), symm=2)
        assert self._old_score(ref, other, 0.5) < 0
        assert ref.similarity_symm(other, max=0.5) == 0.0

    def test_returns_python_float(self):
        sp = SymmParticle(np.eye(3), np.zeros(3), symm="T")
        assert isinstance(sp.similarity_symm(sp), float)


# ===========================================================================
# SymmParticle custom_rot
# ===========================================================================


class TestSymmParticleCustomRot:
    @pytest.mark.parametrize(
        "symm, kind",
        [("T", None), ("O", "octahedron"), ("O", "cube"), ("I", "icosahedron"), ("I", "dodecahedron")],
    )
    def test_rotation_object_and_matrix_give_same_solid(self, symm, kind):
        # custom_rot must be applied exactly once, whether it is passed as a
        # scipy Rotation or as the equivalent 3x3 matrix. Regression test for
        # the double application of Rotation inputs (fixed 2026-09-24).
        custom = R.from_euler("zxz", [20.0, 35.0, 50.0], degrees=True)
        particle_rot = R.from_euler("zxz", [10.0, 60.0, 5.0], degrees=True).as_matrix()
        sp_obj = SymmParticle(particle_rot, np.zeros(3), symm=symm, kind=kind, custom_rot=custom)
        sp_mat = SymmParticle(particle_rot, np.zeros(3), symm=symm, kind=kind, custom_rot=custom.as_matrix())
        assert np.allclose(sp_obj.solid, sp_mat.solid)

    def test_custom_rot_applied_once(self):
        # With an identity particle rotation, the solid must equal the
        # canonical vertices turned once by custom_rot.
        from cryocat.utils import geom

        custom = R.from_euler("z", 30.0, degrees=True)
        sp = SymmParticle(np.eye(3), np.zeros(3), symm="I", kind="icosahedron", custom_rot=custom)
        expected = geom.Icosahedron().vertices @ custom.as_matrix().T
        assert np.allclose(sp.solid, expected)

    def test_previous_dodecahedron_frame_recoverable(self):
        # The pre-fix 'dodeca' frame (turned 90 deg about z) is documented as
        # recoverable with custom_rot = -90 deg about z. The resulting solid
        # must then no longer be preserved by all 60 icosahedral rotations
        # (only the 12 tetrahedral ones), matching the old behaviour.
        from cryocat.utils.symmetry import IcosahedralGroup

        old_frame = R.from_euler("z", -90.0, degrees=True)
        ref = SymmParticle(np.eye(3), np.zeros(3), symm="I", kind="dodecahedron", custom_rot=old_frame)
        scores = [
            ref.similarity_symm(SymmParticle(g, np.zeros(3), symm="I", kind="dodecahedron", custom_rot=old_frame))
            for g in IcosahedralGroup().matrices
        ]
        assert sum(s == pytest.approx(1.0, abs=1e-6) for s in scores) == 12

    @pytest.mark.parametrize("symm", [4, "C6"])
    def test_custom_rot_for_cyclic_raises_clear_error(self, symm):
        # Regression test (2026-09-29): custom_rot with cyclic symmetry used to
        # fail with an opaque numpy matmul shape error (2D polygon vs 3x3
        # matrix). It now raises a ValueError that names the actual problem.
        with pytest.raises(ValueError, match="only applicable to Platonic"):
            SymmParticle(np.eye(3), np.zeros(3), symm=symm, custom_rot=np.eye(3))


# ===========================================================================
# Descriptor.filter_features
# ===========================================================================


class TestDescriptorFilterFeatures:
    @pytest.fixture
    def desc_with_df(self):
        d = Descriptor()
        d.desc = pd.DataFrame(
            {
                "qp_id": [1, 2, 3],
                "feat_a": [0.1, 0.2, 0.3],
                "feat_b": [1.0, 2.0, 3.0],
            }
        )
        return d

    def test_all_returns_full_df(self, desc_with_df):
        result = desc_with_df.filter_features(desc_with_df.desc, feature_ids="all")
        assert set(result.columns) == set(desc_with_df.desc.columns)

    def test_single_feature_includes_qp_id(self, desc_with_df):
        result = desc_with_df.filter_features(desc_with_df.desc, feature_ids="feat_a")
        assert "feat_a" in result.columns
        assert "qp_id" in result.columns
        assert "feat_b" not in result.columns

    def test_list_of_features_filters(self, desc_with_df):
        result = desc_with_df.filter_features(desc_with_df.desc, feature_ids=["feat_a"])
        assert "feat_a" in result.columns
        assert "feat_b" not in result.columns

    def test_invalid_string_raises(self, desc_with_df):
        with pytest.raises(ValueError):
            desc_with_df.filter_features(desc_with_df.desc, feature_ids="nonexistent_col")

    def test_invalid_type_raises(self, desc_with_df):
        with pytest.raises(ValueError):
            desc_with_df.filter_features(desc_with_df.desc, feature_ids=42)

    def test_list_no_valid_features_raises(self, desc_with_df):
        with pytest.raises(ValueError):
            desc_with_df.filter_features(desc_with_df.desc, feature_ids=["nonexistent"])


# ===========================================================================
# TwistDescriptor.process_tomo_twist
# ===========================================================================


def _make_nn(column_name="tomo_id", feature_value=1, n_pairs=2):
    """Return a blank NearestNeighbors with a minimal df for process_tomo_twist."""
    nn = nnana.NearestNeighbors()
    nn.column_name = column_name
    nn.df = pd.DataFrame(
        {
            column_name: [feature_value] * n_pairs,
            "qp_subtomo_id": list(range(1, n_pairs + 1)),
            "nn_subtomo_id": list(range(n_pairs + 1, 2 * n_pairs + 1)),
            "qp_angles_phi": [0.0] * n_pairs,
            "qp_angles_theta": [0.0] * n_pairs,
            "qp_angles_psi": [0.0] * n_pairs,
            "nn_angles_phi": [90.0] * n_pairs,
            "nn_angles_theta": [45.0] * n_pairs,
            "nn_angles_psi": [0.0] * n_pairs,
            "qp_coord_x": [0.0] * n_pairs,
            "qp_coord_y": [0.0] * n_pairs,
            "qp_coord_z": [0.0] * n_pairs,
            "nn_coord_x": [10.0] * n_pairs,
            "nn_coord_y": [10.0] * n_pairs,
            "nn_coord_z": [10.0] * n_pairs,
        }
    )
    return nn


class TestTwistDescriptorProcessTwist:
    """Regression tests for process_tomo_twist exercising the t_nn.column_name attribute.

    Before the fix in tango.py, this code path contained ``t_nn.feature_id``
    (the old attribute name after nnana.py renamed it to ``column_name``).
    The bug was never caught because no test instantiated TwistDescriptor with
    input_motl + nn_radius, nor called process_tomo_twist directly.
    These tests would have surfaced the AttributeError immediately.
    """

    def test_returns_dataframe(self):
        result = TwistDescriptor.process_tomo_twist(_make_nn())
        assert isinstance(result, pd.DataFrame)

    def test_row_count_matches_input(self):
        result = TwistDescriptor.process_tomo_twist(_make_nn(n_pairs=3))
        assert len(result) == 3

    def test_output_columns_no_symm(self):
        nn = _make_nn(column_name="tomo_id")
        result = TwistDescriptor.process_tomo_twist(nn)
        expected = {
            "qp_id",
            "nn_id",
            "tomo_id",
            "twist_so_x",
            "twist_so_y",
            "twist_so_z",
            "twist_x",
            "twist_y",
            "twist_z",
            "qp_inplane",
            "nn_inplane",
        }
        assert set(result.columns) == expected

    def test_column_name_drives_output_column(self):
        """t_nn.column_name must appear in the output — this catches the feature_id rename."""
        nn = _make_nn(column_name="tomo_id", feature_value=7)
        result = TwistDescriptor.process_tomo_twist(nn)
        assert "tomo_id" in result.columns
        assert (result["tomo_id"] == 7).all()

    def test_custom_column_name_in_output(self):
        """When column_name is not tomo_id the output column name must follow."""
        nn = _make_nn(column_name="object_id", feature_value=42)
        result = TwistDescriptor.process_tomo_twist(nn)
        assert "object_id" in result.columns
        assert "tomo_id" not in result.columns
        assert (result["object_id"] == 42).all()

    def test_twist_vectors_are_finite(self):
        result = TwistDescriptor.process_tomo_twist(_make_nn())
        for col in ["twist_so_x", "twist_so_y", "twist_so_z", "twist_x", "twist_y", "twist_z"]:
            assert np.isfinite(result[col]).all(), f"{col} contains non-finite values"

    def test_cyclic_symm_category_still_computes_angular_score(self):
        # symm_category as an int (cyclic) must keep working exactly as before.
        result = TwistDescriptor.process_tomo_twist(_make_nn(), symm=4, symm_max_value=np.pi / 4, symm_category=4)
        assert "angular_score" in result.columns
        assert np.isfinite(result["angular_score"]).all()

    @pytest.mark.parametrize(
        "letter, kind", [("T", None), ("O", "octahedron"), ("O", "cube"), ("I", "icosahedron"), ("I", "dodecahedron")]
    )
    def test_platonic_symm_category_computes_angular_score(self, letter, kind):
        # Until 2026-09-29 a Platonic symm_category raised NotImplementedError
        # (the guard of 2026-09-25). It now scores the full qp/nn rotations with
        # symmetry.angular_score; the column must equal a direct call on the
        # same rotations (qp: phi=theta=psi=0; nn: phi=90, theta=45, psi=0).
        max_val = symmetry.max_angular_mismatch(letter, kind)
        result = TwistDescriptor.process_tomo_twist(
            _make_nn(), symm=letter, symm_max_value=max_val, symm_category=letter, symm_kind=kind
        )
        expected = symmetry.angular_score([0.0, 0.0, 0.0], [90.0, 45.0, 0.0], letter, kind=kind)[0]
        np.testing.assert_allclose(result["angular_score"].to_numpy(), expected, atol=1e-12)
        assert ((result["angular_score"] >= 0) & (result["angular_score"] <= 1)).all()

    def test_platonic_symm_kind_changes_markers(self):
        # symm_kind must actually reach the score: octahedron and cube corners
        # are different marker sets, so the two scores for this (generic) pair
        # differ, each matching its own direct angular_score call.
        max_val = symmetry.max_angular_mismatch("O")
        octa = TwistDescriptor.process_tomo_twist(
            _make_nn(), symm="O", symm_max_value=max_val, symm_category="O", symm_kind="octahedron"
        )["angular_score"].iloc[0]
        cube = TwistDescriptor.process_tomo_twist(
            _make_nn(), symm="O", symm_max_value=max_val, symm_category="O", symm_kind="cube"
        )["angular_score"].iloc[0]
        assert octa == pytest.approx(symmetry.angular_score([0, 0, 0], [90, 45, 0], "O", kind="octahedron")[0])
        assert cube == pytest.approx(symmetry.angular_score([0, 0, 0], [90, 45, 0], "O", kind="cube")[0])
        assert octa != pytest.approx(cube)

    def test_identity_rotation_zero_so_twist(self):
        nn = _make_nn()
        nn.df["nn_angles_phi"] = 0.0
        nn.df["nn_angles_theta"] = 0.0
        nn.df["nn_angles_psi"] = 0.0
        result = TwistDescriptor.process_tomo_twist(nn)
        np.testing.assert_allclose(result[["twist_so_x", "twist_so_y", "twist_so_z"]].values, 0.0, atol=1e-6)


# ===========================================================================
# One in-plane angle for every C_n scoring path (fixed 2026-10-09)
# ===========================================================================


class TestInplaneAnglesFromRotations:
    """_inplane_angles_from_rotations returns phi read from the rotation, written
    close to the stored phi: stored + wrap(derived - stored)."""

    @staticmethod
    def _call(angles):
        # angles: (N, 3) zxz degrees, used both as the stored values and for the rotations
        angles = np.asarray(angles, dtype=float)
        return _inplane_angles_from_rotations(R.from_euler("zxz", angles, degrees=True), angles[:, 0])

    def test_canonical_angles_kept_exactly(self):
        # Normal particles (theta strictly between 0 and 180): the stored phi is the
        # spin of the rotation, so it must come back bit-identical (rounding absorbed).
        angles = [[10.0, 40.0, 20.0], [-123.456, 89.9, 170.0], [179.0, 1.0, -179.0]]
        np.testing.assert_array_equal(self._call(angles), np.asarray(angles)[:, 0])

    def test_same_spin_written_differently_is_kept(self):
        # 270 and -90 are the same spin: the stored number is kept, not rewritten.
        np.testing.assert_array_equal(self._call([[270.0, 40.0, 0.0]]), [270.0])

    def test_theta_zero_takes_spin_from_psi(self):
        # theta = 0: phi and psi turn about the same axis, so (10, 0, 50) is a 60° spin.
        np.testing.assert_allclose(self._call([[10.0, 0.0, 50.0]]), [60.0], atol=1e-9)

    def test_theta_180_uses_rotation_spin(self):
        # theta = 180: the rotation is described by phi - psi; the result must be the
        # phi scipy reads from the rotation (same spin, up to whole turns).
        angles = np.array([[10.0, 180.0, 50.0]])
        derived = R.from_euler("zxz", angles, degrees=True).as_euler("zxz", degrees=True)[0, 0]
        result = self._call(angles)[0]
        assert np.mod(result - derived + 180.0, 360.0) - 180.0 == pytest.approx(0.0, abs=1e-9)
        assert result != pytest.approx(10.0)

    def test_theta_outside_range_is_corrected(self):
        # (10, -30, 20) is the same rotation as phi = -170 in scipy's ranges.
        np.testing.assert_allclose(self._call([[10.0, -30.0, 20.0]]), [-170.0], atol=1e-9)


class TestCyclicScoreConsistency:
    """Before 2026-10-09, process_tomo_twist and symmetry_statistics scored C_n from
    the stored phi, while SymmParticle.similarity_symm used phi read from the
    rotation; for theta = 0 (or theta outside [0, 180]) they disagreed. All three
    must now give the same result."""

    @staticmethod
    def _nn(qp_angles, nn_angles):
        # One pair; coordinates are irrelevant for the in-plane score.
        nn = _make_nn(n_pairs=1)
        nn.df[["qp_angles_phi", "qp_angles_theta", "qp_angles_psi"]] = [qp_angles]
        nn.df[["nn_angles_phi", "nn_angles_theta", "nn_angles_psi"]] = [nn_angles]
        return nn

    def test_theta_zero_pair_scored_from_true_spin(self):
        # (10, 0, 50) vs (10, 0, 0): 50° apart about the same axis; nearest C4 copy
        # is 40° away, so the score is 1 - 40/45. The old stored-phi path gave 1.
        result = TwistDescriptor.process_tomo_twist(
            self._nn([10.0, 0.0, 50.0], [10.0, 0.0, 0.0]), symm=4, symm_max_value=np.pi / 4, symm_category=4
        )
        assert result["angular_score"].iloc[0] == pytest.approx(1 - 40 / 45)
        assert result["qp_inplane"].iloc[0] == pytest.approx(60.0)
        assert result["nn_inplane"].iloc[0] == pytest.approx(10.0)

    @pytest.mark.parametrize("qp, nn", [([10.0, 0.0, 50.0], [10.0, 0.0, 0.0]), ([10.0, -30.0, 20.0], [35.0, 60.0, 5.0])])
    def test_matches_symm_particle(self, qp, nn):
        # The twist-descriptor score must equal SymmParticle.similarity_symm for the same pair.
        result = TwistDescriptor.process_tomo_twist(self._nn(qp, nn), symm=4, symm_max_value=np.pi / 4, symm_category=4)
        sp_qp = SymmParticle(R.from_euler("zxz", qp, degrees=True), np.zeros(3), symm=4)
        sp_nn = SymmParticle(R.from_euler("zxz", nn, degrees=True), np.zeros(3), symm=4)
        assert result["angular_score"].iloc[0] == pytest.approx(sp_qp.similarity_symm(sp_nn))

    def test_same_orientation_written_two_ways_scores_the_same(self):
        # (10, -30, 20) and its scipy form describe one rotation: score and in-plane
        # angles (as spins) must not depend on how the angles were written.
        written = [10.0, -30.0, 20.0]
        canonical = R.from_euler("zxz", written, degrees=True).as_euler("zxz", degrees=True).tolist()
        other = [35.0, 60.0, 5.0]
        a = TwistDescriptor.process_tomo_twist(self._nn(written, other), symm=3, symm_max_value=np.pi / 3, symm_category=3)
        b = TwistDescriptor.process_tomo_twist(self._nn(canonical, other), symm=3, symm_max_value=np.pi / 3, symm_category=3)
        assert a["angular_score"].iloc[0] == pytest.approx(b["angular_score"].iloc[0])
        spin_diff = a["qp_inplane"].iloc[0] - b["qp_inplane"].iloc[0]
        assert np.mod(spin_diff + 180.0, 360.0) - 180.0 == pytest.approx(0.0, abs=1e-9)

    def test_symmetry_statistics_agrees_with_score(self):
        # symmetry_statistics recomputes C_n scores from qp_inplane/nn_inplane; its C4
        # values must equal the angular_score column computed with C4.
        nn = _make_nn(n_pairs=3)
        nn.df[["qp_angles_phi", "qp_angles_theta", "qp_angles_psi"]] = [[10, 0, 50], [10, -30, 20], [5, 40, 0]]
        nn.df[["nn_angles_phi", "nn_angles_theta", "nn_angles_psi"]] = [[10, 0, 0], [35, 60, 5], [80, 40, 10]]
        df = TwistDescriptor.process_tomo_twist(nn, symm=4, symm_max_value=np.pi / 4, symm_category=4)
        # A descriptor built from the output (distance features not needed here are zero-filled).
        td = TwistDescriptor(input_twist=_minimal_twist_df({c: df[c].tolist() for c in df.columns}))
        fig = td.symmetry_statistics(c_range=[4], plot_graph=False)
        np.testing.assert_allclose(np.asarray(fig.data[0].y, dtype=float), df["angular_score"].to_numpy(), atol=1e-9)

    def test_symmetry_statistics_uses_relative_rotation_only(self):
        # Since 2026-10-09 symmetry_statistics reads the spin from the stored relative
        # rotation (twist_so_*), so it works on a descriptor loaded from a table and
        # ignores the in-plane columns (set to nonsense here on purpose).
        rel = R.random(6, random_state=11)
        rotvec = rel.as_rotvec()
        td = TwistDescriptor(
            input_twist=_minimal_twist_df(
                {
                    "twist_so_x": rotvec[:, 0].tolist(),
                    "twist_so_y": rotvec[:, 1].tolist(),
                    "twist_so_z": rotvec[:, 2].tolist(),
                    "qp_inplane": [123.0] * 6,
                    "nn_inplane": [-45.0] * 6,
                }
            )
        )
        fig = td.symmetry_statistics(c_range=[3], plot_graph=False)
        # Reference: the C3 score of a query point at the identity vs. the neighbour at rel.
        expected = symmetry.angular_score(R.identity(6), rel, 3)
        np.testing.assert_allclose(np.asarray(fig.data[0].y, dtype=float), expected, atol=1e-9)

    def test_inplane_columns_also_corrected_without_symmetry(self):
        # The in-plane columns exist without symmetry too and must hold the same spin.
        result = TwistDescriptor.process_tomo_twist(self._nn([10.0, 0.0, 50.0], [10.0, 0.0, 0.0]))
        assert result["qp_inplane"].iloc[0] == pytest.approx(60.0)

    def test_canonical_input_columns_unchanged(self):
        # _make_nn uses canonical angles (qp phi 0, nn phi 90): stored values are kept exactly.
        result = TwistDescriptor.process_tomo_twist(_make_nn())
        assert (result["qp_inplane"] == 0.0).all()
        assert (result["nn_inplane"] == 90.0).all()


# ── TwistDescriptor.check_twist_columns ──────────────────────────────────────


def _twist_df(extra_cols=(), missing=()):
    cols = [c for c in TwistDescriptor.get_all_feature_ids(symm=False) if c not in missing]
    data = {c: [0.0] for c in cols}
    for c in extra_cols:
        data[c] = [0.0]
    return pd.DataFrame(data)


class TestCheckTwistColumns:
    def test_complete_df_returns_empty(self):
        df = _twist_df()
        assert TwistDescriptor.check_twist_columns(df) == []

    def test_missing_columns_reported(self):
        missing = ["qp_id", "twist_x"]
        df = _twist_df(missing=missing)
        result = TwistDescriptor.check_twist_columns(df)
        assert set(result) == set(missing)

    def test_extra_columns_not_reported(self):
        df = _twist_df(extra_cols=["custom_col"])
        assert TwistDescriptor.check_twist_columns(df) == []

    def test_symm_false_excludes_angular_score(self):
        df = _twist_df()
        assert "angular_score" not in TwistDescriptor.get_all_feature_ids(symm=False)
        assert TwistDescriptor.check_twist_columns(df, symm=False) == []

    def test_symm_true_requires_angular_score(self):
        df = _twist_df()  # no angular_score column
        result = TwistDescriptor.check_twist_columns(df, symm=True)
        assert "angular_score" in result

    def test_symm_true_ok_when_angular_score_present(self):
        df = _twist_df(extra_cols=["angular_score"])
        assert TwistDescriptor.check_twist_columns(df, symm=True) == []

    def test_checker_tracks_get_all_feature_ids(self, monkeypatch):
        sentinel = ["col_a", "col_b"]
        monkeypatch.setattr(TwistDescriptor, "get_all_feature_ids", staticmethod(lambda symm=False: sentinel))
        df_good = pd.DataFrame({"col_a": [1], "col_b": [2]})
        df_bad = pd.DataFrame({"col_a": [1]})
        assert TwistDescriptor.check_twist_columns(df_good) == []
        assert TwistDescriptor.check_twist_columns(df_bad) == ["col_b"]

    def test_empty_df_reports_all_required(self):
        df = pd.DataFrame()
        result = TwistDescriptor.check_twist_columns(df)
        expected = TwistDescriptor.get_all_feature_ids(symm=False)
        assert set(result) == set(expected)


# ── TwistDescriptor radius derivation ────────────────────────────────────────


def _minimal_twist_df(rows=None):
    all_cols = TwistDescriptor.get_all_feature_ids(symm=False)
    n_rows = max((len(v) for v in rows.values()), default=1) if rows else 1
    base = {c: [0.0] * n_rows for c in all_cols if not rows or c not in rows}
    if rows:
        base.update(rows)
    return pd.DataFrame(base)


class TestDeriveNNRadius:
    def test_single_row(self):
        df = _minimal_twist_df({"twist_x": [3.0], "twist_y": [4.0], "twist_z": [0.0]})
        result = TwistDescriptor.derive_nn_radius(df)
        assert math.isclose(result, 5.0, rel_tol=1e-9)

    def test_returns_maximum(self):
        df = _minimal_twist_df(
            {
                "twist_x": [1.0, 0.0, 3.0],
                "twist_y": [0.0, 2.0, 4.0],
                "twist_z": [0.0, 0.0, 0.0],
            }
        )
        assert math.isclose(TwistDescriptor.derive_nn_radius(df), 5.0, rel_tol=1e-9)

    def test_zero_vectors(self):
        df = _minimal_twist_df({"twist_x": [0.0], "twist_y": [0.0], "twist_z": [0.0]})
        assert TwistDescriptor.derive_nn_radius(df) == 0.0

    def test_returns_float(self):
        df = _minimal_twist_df({"twist_x": [1.0], "twist_y": [0.0], "twist_z": [0.0]})
        result = TwistDescriptor.derive_nn_radius(df)
        assert isinstance(result, float)

    def test_all_equal_magnitudes(self):
        df = _minimal_twist_df(
            {
                "twist_x": [3.0, 3.0],
                "twist_y": [4.0, 4.0],
                "twist_z": [0.0, 0.0],
            }
        )
        assert math.isclose(TwistDescriptor.derive_nn_radius(df), 5.0, rel_tol=1e-9)


class TestTwistDescriptorRadiusStorage:
    def _make_td(self, twist_x=3.0, twist_y=4.0, twist_z=0.0, nn_radius=None):
        df = _minimal_twist_df(
            {
                "twist_x": [twist_x],
                "twist_y": [twist_y],
                "twist_z": [twist_z],
            }
        )
        return TwistDescriptor(input_twist=df, nn_radius=nn_radius)

    def test_radius_derived_when_not_given(self):
        td = self._make_td(twist_x=3.0, twist_y=4.0, twist_z=0.0, nn_radius=None)
        assert math.isclose(td.nn_radius, 5.0, rel_tol=1e-9)
        assert td.radius_source == "computed"

    def test_radius_used_as_given_when_provided(self):
        td = self._make_td(nn_radius=42.0)
        assert td.nn_radius == pytest.approx(42.0)
        assert td.radius_source == "given"

    def test_given_radius_overrides_table(self):
        td = self._make_td(twist_x=3.0, twist_y=4.0, twist_z=0.0, nn_radius=99.0)
        assert td.nn_radius == pytest.approx(99.0)

    def test_computed_radius_matches_derive_nn_radius(self):
        df = _minimal_twist_df({"twist_x": [0.0, 6.0], "twist_y": [1.0, 8.0], "twist_z": [0.0, 0.0]})
        expected = TwistDescriptor.derive_nn_radius(df)
        td = TwistDescriptor(input_twist=df.copy(), nn_radius=None)
        assert math.isclose(td.nn_radius, expected, rel_tol=1e-9)
        assert td.radius_source == "computed"

    def test_compute_path_sets_given_source(self):
        from cryocat.core.cryomotl import Motl

        df = pd.DataFrame({col: [1.0, 2.0] for col in Motl.motl_columns})
        df["subtomo_id"] = [1.0, 2.0]
        df["tomo_id"] = [1.0, 1.0]
        motl = Motl(df)

        td = TwistDescriptor(input_motl=motl, nn_radius=50.0, build_unique_desc=False)
        assert td.nn_radius == pytest.approx(50.0)
        assert td.radius_source == "given"

    def test_kind_reaches_symm_particle_through_constructor(self):
        # Regression test (2026-09-28): TwistDescriptor.__init__ never had a
        # `kind` parameter even after get_symm_parameters/
        # get_nn_twist_stats_within_radius gained one, so `kind` could never
        # reach SymmParticle through the class the GUI actually instantiates.
        # Since 2026-09-29 (the process_tomo_twist guard was replaced by the
        # Platonic angular score) the constructor must complete and produce an
        # angular_score column in [0, 1], instead of raising.
        from cryocat.core.cryomotl import Motl

        df = pd.DataFrame({col: [1.0, 2.0] for col in Motl.motl_columns})
        df["subtomo_id"] = [1.0, 2.0]
        df["tomo_id"] = [1.0, 1.0]
        motl = Motl(df)

        td = TwistDescriptor(input_motl=motl, nn_radius=50.0, symm="O", kind="cube", build_unique_desc=False)
        assert "angular_score" in td.df.columns
        assert ((td.df["angular_score"] >= 0) & (td.df["angular_score"] <= 1)).all()

    def test_cyclic_symm_reaches_symm_particle_through_constructor(self):
        # Same entry point, cyclic path: must still compute angular_score end-to-end.
        from cryocat.core.cryomotl import Motl

        df = pd.DataFrame({col: [1.0, 2.0] for col in Motl.motl_columns})
        df["subtomo_id"] = [1.0, 2.0]
        df["tomo_id"] = [1.0, 1.0]
        motl = Motl(df)

        td = TwistDescriptor(input_motl=motl, nn_radius=50.0, symm=4, build_unique_desc=False)
        assert "angular_score" in td.df.columns
        assert np.isfinite(td.df["angular_score"]).all()

    def test_nn_radius_attribute_exists_after_file_load(self, tmp_path):
        df = _minimal_twist_df({"twist_x": [3.0], "twist_y": [4.0], "twist_z": [0.0]})
        csv_path = str(tmp_path / "twist.csv")
        df.to_csv(csv_path, index=False)

        td = TwistDescriptor(input_twist=csv_path, nn_radius=None)
        assert hasattr(td, "nn_radius")
        assert hasattr(td, "radius_source")
        assert td.radius_source in ("computed", "given", "unknown")


# ===========================================================================
# AngularScoreNN (fixed 2026-09-29)
# ===========================================================================


def _scored_twist_descriptor():
    """Two query points with known angular scores for their neighbours."""
    df = _minimal_twist_df(
        {
            "qp_id": [1, 1, 1, 2, 2],
            "nn_id": [10, 11, 12, 20, 21],
            "angular_score": [0.2, 0.9, 0.5, 0.7, 0.1],
        }
    )
    return TwistDescriptor(input_twist=df)


class TestAngularScoreNN:
    """The angular score is a similarity (1 = identical up to symmetry), so the
    filter must keep the neighbours with the HIGHEST scores. Before the fix it
    sorted ascending and kept the least similar ones."""

    def test_keeps_best_fit_per_query_point(self):
        kept = AngularScoreNN(_scored_twist_descriptor(), num_neighbors=1).filter.df
        assert dict(zip(kept["qp_id"], kept["nn_id"])) == {1: 11, 2: 20}  # scores 0.9 and 0.7

    def test_keeps_top_n_in_descending_order(self):
        kept = AngularScoreNN(_scored_twist_descriptor(), num_neighbors=2).filter.df
        qp1 = kept[kept["qp_id"] == 1]["angular_score"].tolist()
        assert qp1 == [0.9, 0.5]  # the 0.2 neighbour is dropped

    def test_num_neighbors_larger_than_available_keeps_all(self):
        kept = AngularScoreNN(_scored_twist_descriptor(), num_neighbors=5).filter.df
        assert len(kept) == 5


# ===========================================================================
# TwistDescriptor.symmetry_statistics c_range (fixed 2026-09-29)
# ===========================================================================


class TestSymmetryStatisticsRange:
    """An integer c_range must include its own value (docstring: "from 2 up to
    and including that value"); before the fix c_range=4 gave only C2, C3."""

    @staticmethod
    def _box_names(fig):
        return [trace.name for trace in fig.data]

    def _descriptor(self):
        td = _scored_twist_descriptor()
        td.df["qp_inplane"] = [0.0, 10.0, 20.0, 30.0, 40.0]
        td.df["nn_inplane"] = [5.0, 50.0, 90.0, 100.0, 170.0]
        return td

    def test_integer_c_range_includes_endpoint(self):
        fig = self._descriptor().symmetry_statistics(c_range=4, plot_graph=False)
        assert self._box_names(fig) == ["2", "3", "4"]

    def test_float_c_range_includes_endpoint(self):
        fig = self._descriptor().symmetry_statistics(c_range=3.0, plot_graph=False)
        assert self._box_names(fig) == ["2", "3"]

    def test_default_range_unchanged(self):
        # Default stays C2..C9.
        fig = self._descriptor().symmetry_statistics(plot_graph=False)
        assert self._box_names(fig) == [str(n) for n in range(2, 10)]

    def test_explicit_range_object_used_as_is(self):
        fig = self._descriptor().symmetry_statistics(c_range=range(3, 6), plot_graph=False)
        assert self._box_names(fig) == ["3", "4", "5"]


# ── TwistDescriptor Integration tests ──────────────────────────────────────

GT_TWIST_DF = pd.read_csv(Path(__file__).parent / "test_data" / "tango" / "gt_twist_vector_r5.csv")


def test_integration_twist_computation():

    input_motl = cryomotl.Motl.load(Path(__file__).parent / "test_data" / "tango" / "motl_cone.em")

    motl = cryomotl.Motl.load(input_motl=input_motl, motl_type="emmotl")
    twist = TwistDescriptor(
        input_motl=motl,
        nn_radius=5,
        column_name="tomo_id",
        symm=None,
        remove_qp=False,
        remove_duplicates=False,
        build_unique_desc=False,
    )
    np.testing.assert_allclose(twist.df.to_numpy(), GT_TWIST_DF.to_numpy(), atol=1e-10)


def test_integration_shot_descriptor():
    desc = SHOTDescriptor(twist_df=GT_TWIST_DF, cone_number=6, shell_number=1, north_pole_axis=None)
    gt_shot = pd.read_csv(Path(__file__).parent / "test_data" / "tango" / "gt_shot.csv")
    np.testing.assert_allclose(desc.desc.to_numpy(), gt_shot.to_numpy(), atol=1e-10)


def test_integration_alpha_complex_descriptor():
    desc = AlphaComplexDescriptor(twist_df=GT_TWIST_DF, alpha_param=200.0)
    gt_alpha = pd.read_csv(Path(__file__).parent / "test_data" / "tango" / "gt_alpha.csv")
    np.testing.assert_allclose(desc.desc.to_numpy(), gt_alpha.to_numpy(), atol=1e-10)


def test_integration_pl_complex_descriptor():
    desc = PLComplexDescriptor(twist_df=GT_TWIST_DF)
    gt_pl = df = pd.read_csv(Path(__file__).parent / "test_data" / "tango" / "gt_pl.csv")
    np.testing.assert_allclose(desc.desc.to_numpy(), gt_pl.to_numpy(), atol=1e-10)


def test_integration_twist_descriptor_from_input():
    desc = TwistDescriptor(input_twist=GT_TWIST_DF)
    gt_twist_desc = pd.read_csv(Path(__file__).parent / "test_data" / "tango" / "gt_twist_desc.csv")
    np.testing.assert_allclose(desc.desc.to_numpy(), gt_twist_desc.to_numpy(), atol=1e-10)


def test_integration_twist_symm_with_integer_tomo_id():
    """TwistDescriptor with symm=3 must not raise when tomo_id dtype is int64.

    The bug: check_type inside Particle.__init__ used isinstance(x, (int, float)),
    which rejects np.int64 and np.float32.  SymmParticle paths (symm != None)
    call convert_to_particle_list → SymmParticle → check_type; the non-symm path
    creates Particle objects directly and hits the same check, so both paths are
    covered by the fix.
    """
    motl = cryomotl.Motl.load(Path(__file__).parent / "test_data" / "tango" / "motl_cone.em")
    motl.df["tomo_id"] = motl.df["tomo_id"].astype(np.int64)

    twist = TwistDescriptor(
        input_motl=motl,
        nn_radius=5,
        column_name="tomo_id",
        symm=3,
        remove_qp=False,
        remove_duplicates=False,
        build_unique_desc=False,
    )

    assert len(twist.df) == 3604
    assert "angular_score" in twist.df.columns
