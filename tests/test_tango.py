import sys
import types
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import LinearSegmentedColormap

# emfile is an optional binary dependency not present in all environments.
# Stub it so that cryomotl (and everything that imports it) can be collected.
if "emfile" not in sys.modules:
    sys.modules["emfile"] = types.ModuleType("emfile")

from cryocat.analysis.tango import (
    Particle,
    SymmParticle,
    Descriptor,
    get_colormap_color,
)


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
        sp = SymmParticle(np.eye(3), np.zeros(3), symm="tetra")
        assert sp.category == "tetrahedron"

    def test_platonic_octahedron(self):
        sp = SymmParticle(np.eye(3), np.zeros(3), symm="octa")
        assert sp.category == "octahedron"


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
# get_colormap_color
# ===========================================================================

class TestGetColormapColor:
    def test_returns_hex_string(self):
        cmap = LinearSegmentedColormap.from_list("test", ["#000000", "#ffffff"])
        color = get_colormap_color(cmap, 0.5)
        assert isinstance(color, str)
        assert color.startswith("#")

    def test_black_at_zero(self):
        cmap = LinearSegmentedColormap.from_list("test", ["#000000", "#ffffff"])
        color = get_colormap_color(cmap, 0.0)
        assert color == "#000000"

    def test_white_at_one(self):
        cmap = LinearSegmentedColormap.from_list("test", ["#000000", "#ffffff"])
        color = get_colormap_color(cmap, 1.0)
        assert color == "#ffffff"
