import numpy as np
import pandas as pd
from math import fabs


def point_dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def get_mesh_vertices(opp_vertex, v1, v2):
    fourth_vertex = opp_vertex + (v1 - v2)
    mesh_corners = np.array((v1, fourth_vertex, v2, opp_vertex))
    return mesh_corners


def get_mesh(vertices, sampling_distance):
    min_dist = point_dist(vertices[0], vertices[1])
    mesh_corners = get_mesh_vertices(vertices[2], vertices[0], vertices[1])

    if min_dist > point_dist(vertices[0], vertices[2]):
        min_dist = point_dist(vertices[0], vertices[2])
        mesh_corners = get_mesh_vertices(vertices[1], vertices[0], vertices[2])

    if min_dist > point_dist(vertices[1], vertices[2]):
        min_dist = point_dist(vertices[1], vertices[2])
        mesh_corners = get_mesh_vertices(vertices[0], vertices[1], vertices[2])

    if min_dist < sampling_distance:
        return vertices

    number_of_samples_long = round(point_dist(mesh_corners[0], mesh_corners[1]) / sampling_distance)

    first_parallel = np.linspace(mesh_corners[0], mesh_corners[1], number_of_samples_long)
    second_parallel = np.linspace(mesh_corners[2], mesh_corners[3], number_of_samples_long)

    ort_dist = point_dist(first_parallel[0], second_parallel[0])
    number_of_samples_short = round(ort_dist / sampling_distance)

    mesh_points = []
    for i in range(len(first_parallel)):
        mesh_points.append(np.linspace(first_parallel[i], second_parallel[i], number_of_samples_short))

    mesh_points = np.concatenate(mesh_points, axis=0)

    return mesh_points


def get_dominant_axis(triangle, point, coord1, coord2):
    u1 = triangle[0][coord1] - triangle[2][coord1]
    u2 = triangle[1][coord1] - triangle[2][coord1]
    u3 = point[coord1] - triangle[0][coord1]
    u4 = point[coord1] - triangle[2][coord1]

    v1 = triangle[0][coord2] - triangle[2][coord2]
    v2 = triangle[1][coord2] - triangle[2][coord2]
    v3 = point[coord2] - triangle[0][coord2]
    v4 = point[coord2] - triangle[2][coord2]

    u = np.array([u1, u2, u3, u4])
    v = np.array([v1, v2, v3, v4])

    return u, v


def compute_barycentric_coord(triangle, point):

    # First, compute two clockwise edge vectors
    d1 = triangle[1] - triangle[0]
    d2 = triangle[2] - triangle[1]

    # Compute surface normal using cross product.  In many cases
    # this step could be skipped, since we would have the surface
    # normal precomputed.  We do not need to normalize it, although
    # if a precomputed normal was normalized, it would be OK.
    triangle_normal = np.cross(d1, d2)

    # Locate dominant axis of normal, and select plane of projection
    if (fabs(triangle_normal[0]) >= fabs(triangle_normal[1])) and (
        fabs(triangle_normal[0]) >= fabs(triangle_normal[2])
    ):

        # Discard x, project onto yz plane
        u, v = get_dominant_axis(triangle, point, 1, 2)

    elif fabs(triangle_normal[1]) >= fabs(triangle_normal[2]):

        # Discard y, project onto xz plane
        u, v = get_dominant_axis(triangle, point, 2, 0)

    else:

        # Discard z, project onto xy plane
        u, v = get_dominant_axis(triangle, point, 0, 1)

    # Compute denominator, check for invalid
    denom = v[0] * u[1] - v[1] * u[0]

    if denom == 0.0:
        return [-1, -1, -1]  # Bogus triangle - probably triangle has zero area

    # Compute barycentric coordinates
    oneOverDenom = 1.0 / denom

    b = np.zeros(3)

    b[0] = (v[3] * u[1] - v[1] * u[3]) * oneOverDenom
    b[1] = (v[0] * u[2] - v[2] * u[0]) * oneOverDenom
    b[2] = 1.0 - b[0] - b[1]

    return b


def point_inside_triangle(point, triangle):
    b = compute_barycentric_coord(triangle, point)

    if np.all(b >= -0.0001) and np.all(b <= 1.0001):
        return True
    else:
        return False
