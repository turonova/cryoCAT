import pytest
import numpy as np
import pandas as pd
from cryocat.surfsamp import *
from cryocat import cryomotl
from cryocat import cryomask
from cryocat import cryomap
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R


@pytest.fixture
def shape():
    shape = load_shapes("./tests/test_data/point_clouds/c10x10.csv")
    return shape


@pytest.fixture
def motl():
    motl = cryomotl.Motl.load("./tests/test_data/point_clouds/motl_c10x10.em")
    return motl


def test_get_oversampling(shape, d=1):
    # check if points are within the given distance
    points, _ = get_oversampling(shape, sampling_distance=d)
    # find distances between all pair of points
    distances = cdist(points, points)
    distances[distances == 0] = np.nan
    min_dist = np.nanmin(distances, axis=1)
    dist_median = np.median(min_dist)
    # Return true as long as the distance median is smaller than the input
    assert d * 1.9 > dist_median


# TODO add test for expand points without moving top and bottom.
def test_expand_points_with_tb(shape, d=3):
    # Checking that the difference in distances between coordinates before and after expansion is consistent with the setting.
    points, normals = get_oversampling(shape, sampling_distance=d)
    mv_points, _ = expand_points(points, normals, distances=d, tb_distances=d)
    distances = np.sqrt(np.sum(np.square(mv_points - points), axis=1))
    # Return true when distance difference are similar to the setting.
    diff_dist = np.absolute(distances - d) < 0.1
    assert diff_dist.all()


@pytest.mark.parametrize(
    "clean_face, exclude_normals", [(1, [1, 0, 0]), (-1, [-1, 0, 0])]
)
def test_rm_points(shape, clean_face, exclude_normals, d=3):
    # After removing flat surface at top and bottom, there should be no normals in [1, 0, 0] or [-1, 0, 0]
    points, normals = get_oversampling(shape, sampling_distance=d)
    _, clean_normals = rm_points(points, normals, rm_surface=clean_face)
    assert not np.all(clean_normals == exclude_normals)


@pytest.mark.parametrize(
    "points, rm_faces, area",
    [
        (
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [0, 1, 1],
                ]
            ),
            0,
            6,
        ),
        (
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]]),
            0,
            1 + ((5 / 4) ** 0.5) + ((5 / 4) ** 0.5),
        ),
        (
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]]),
            -1,
            1 + ((5 / 4) ** 0.5) + ((5 / 4) ** 0.5) - 1,
        ),
        (
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [0, 1, 1],
                ]
            ),
            1,
            5,
        ),
        (
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [0, 1, 1],
                ]
            ),
            -1,
            5,
        ),
        (
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [0, 1, 1],
                ]
            ),
            2,
            4,
        ),
    ],
)
# test with cube and a upside down pyramid
def test_get_surface_area_from_hull(points, rm_faces, area):
    output_area = get_surface_area_from_hull(points, rm_faces=rm_faces)
    assert output_area == area


@pytest.mark.parametrize(
    "points, rm_faces",
    [
        (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]]), 1),
    ],
)
def test_get_surface_area_from_hull_wrong(points, rm_faces):
    # Check error handling when no surface matches the assigned surface.
    with pytest.raises(ValueError):
        get_surface_area_from_hull(points, rm_faces)


def test_reset_normals(shape, motl):
    # Randomize the euler angles in the motl
    random_angles = R.random(len(motl.df)).as_euler("zxz", degrees=True)
    motl_random_angles = motl
    motl_random_angles.fill(
        {
            "angles": pd.DataFrame(
                {
                    "phi": random_angles[:, 0],
                    "psi": random_angles[:, 1],
                    "theta": random_angles[:, 2],
                }
            )
        }
    )
    # Reset the modified motl
    updated_motl = reset_normals(shape, motl_random_angles)
    # Verify if the reset motl matches the input motl.
    assert updated_motl.df.equals(motl.df)


def test_get_sampling_pandas(shape, d=3):
    points, normals = get_sampling_pandas(shape, overSample_dist=d)
    assert np.array_equal(points.columns, ["x", "y", "z"])
    assert np.array_equal(normals.columns, ["phi", "psi", "theta"])


def test_angle_clean(shape, motl):
    # rotation of 180 degrees
    rot = R.from_matrix([[0, 0, -1], [0, -1, 0], [1, 0, 0]])
    motl.apply_rotation(rot)
    points, normals = get_oversampling(shape, sampling_distance=1)
    # clean motl
    clean_motl = angles_clean(
        motl, points, normals, angle_threshold=45, normal_vector=[0, 0, 1]
    )
    # a empty dataframe was expected because all position should be removed after rotating in 180
    assert len(clean_motl.df) == 0


def check_mask(mask1, mask2):
    assert mask1.shape == mask2.shape


# only check the output dimension of volume, using skimage.morphology for all operation, no need for testing.
@pytest.mark.parametrize(
    "r, mode", [(2, "closing"), (2, "opening"), (2, "dilation"), (2, "erosion")]
)
def test_process_mask(r, mode):
    sperical_mask = cryomask.spherical_mask(20, radius=5)
    processed_mask = process_mask(sperical_mask, r, mode)
    check_mask(sperical_mask, processed_mask)


# test with all particles were inside and outside mask area
def test_mask_clean_all_in(motl, r=15):
    df = motl.df.copy(deep=True)
    sperical_mask = cryomask.spherical_mask(11, radius=r)
    clean_motl = mask_clean(motl, sperical_mask)
    check = clean_motl.df == df
    assert check.all(axis=0).all() == True


def test_mask_clean_all_out(motl, r=2):
    df = motl.df.copy(deep=True)
    sperical_mask = cryomask.spherical_mask(11, radius=r)
    clean_motl = mask_clean(motl, sperical_mask)
    assert clean_motl.df.empty


def test_marker_from_centroid(box=10, sper_rad=2, cent_size=1):
    sperical_mask = cryomask.spherical_mask(box, radius=sper_rad)
    mask = marker_from_centroid(sperical_mask, centroid_mark_size=cent_size)
    ground_truth = cryomap.read(
        "./tests/test_data/point_clouds/sphere_centroid_marker_10x10.mrc"
    )
    liken = mask == ground_truth
    assert liken.all()
