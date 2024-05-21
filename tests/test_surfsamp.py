import pytest
import numpy as np
from cryocat.surfsamp import *
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist

@pytest.fixture
def shape():
    shape = load_shapes("./tests/test_data/point_clouds/040_1_shape.csv")
    return shape

def test_get_oversampling(shape, d = 3):
    # check if points are within the given distance
    points, normals = get_oversampling(shape, sampling_distance = d)
    unique_normals = np.unique(normals, axis=0)
    dist_analysis = np.zeros(shape=unique_normals.shape[0])
    # calculating distances between points in separated surface groups
    # Is this really necessary?
    for i,n in enumerate(unique_normals):
        # Extract coordinates with the same normal
        normals_log=normals==n
        points_w_same_normals=points[normals_log.all(axis=1)]
        # Calculate distances between nearest neighbor points
        distances = cdist(points_w_same_normals,points_w_same_normals)
        distances[distances==0] = np.nan
        min_dist = np.nanmin(distances, axis=1)
        # Return False when the most of the distance is much larger than the input parameters
        dist_analysis[i] = sum(min_dist < d*1.5) > points_w_same_normals.shape[0]*0.9
    assert dist_analysis.all()

# TODO add test for expand points without moving top and bottom.
def test_expand_points_with_tb(shape, d = 3):
    # Checking that the difference in distances between coordinates before and after expansion is consistent with the setting.
    points, normals = get_oversampling(shape, sampling_distance = d)
    mv_points, mv_normals = expand_points(points, normals, distances=d, tb_distances=d)
    distances = np.sqrt(np.sum(np.square(mv_points-points), axis=1))
    # Return true when distance difference are similar to the setting.
    diff_dist = distances +- 3 < 0.1
    assert diff_dist.all()

@pytest.mark.parametrize(
    "clean_face, exclude_normals", [(1, [1, 0, 0]), (-1, [-1, 0, 0])]
)
def test_rm_points(shape, clean_face, exclude_normals, d = 3):
    # After removing flat surface at top and bottom, there should be no normals in [1, 0, 0] or [-1, 0, 0]
    points, normals = get_oversampling(shape, sampling_distance = d)
    clean_points, clean_normals = rm_points(points, normals, rm_surface=clean_face)
    assert not np.all(clean_normals == exclude_normals)

@pytest.mark.parametrize(
    "points, rm_faces, area",
    [
        (
            np.array([
                [0,0,0],
                [1,0,0],
                [1,1,0],
                [1,1,1],
                [0,1,0],
                [0,0,1],
                [1,0,1],
                [0,1,1]
            ]),
            0, 6
        ),
        (
            np.array([
                [0,0,0],
                [1,0,0],
                [1,1,0],
                [0,1,0],
                [0.5,0.5,1] 
            ]),
            0, 1+((5/4)**.5)+((5/4)**.5)
        ),
        (
            np.array([
                [0,0,0],
                [1,0,0],
                [1,1,0],
                [0,1,0],
                [0.5,0.5,1] 
            ]),
            1, 1+((5/4)**.5)+((5/4)**.5)-1
        ),
        (
            np.array([
                [0,0,0],
                [1,0,0],
                [1,1,0],
                [1,1,1],
                [0,1,0],
                [0,0,1],
                [1,0,1],
                [0,1,1]
            ]),
            1, 5
        ),
        (
            np.array([
                [0,0,0],
                [1,0,0],
                [1,1,0],
                [1,1,1],
                [0,1,0],
                [0,0,1],
                [1,0,1],
                [0,1,1]
            ]),
            -1, 5
        ),
        (
            np.array([
                [0,0,0],
                [1,0,0],
                [1,1,0],
                [1,1,1],
                [0,1,0],
                [0,0,1],
                [1,0,1],
                [0,1,1]
            ]),
            2, 4
        )
    ]
)
# test with cube and a upside down pyramid
def test_get_surface_area_from_hull(points, rm_faces, area):
    output_area = get_surface_area_from_hull(points, rm_faces = rm_faces)
    assert output_area == area