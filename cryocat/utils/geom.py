import numpy as np
import pandas as pd
import re
from scipy.spatial.transform import Rotation as srot
from cryocat.utils.exceptions import UserInputError
import matplotlib.pyplot as plt
import os
import math
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import splprep, splev
from scipy.optimize import fsolve

from cryocat._types import RotationLike, Symmetry, TripletLike, EulerAngles, ArrayLike


# Constants
ANGLE_DEGREES_TOL = 10e-12
PHI = (1 + np.sqrt(5)) / 2
ROOT3 = np.sqrt(3)
ROOT2 = np.sqrt(2)



class Line:
    """An infinite line in 3-D space defined by a point and a direction.

    Attributes
    ----------
    p : array-like
        A point on the line.
    dir : array-like
        Direction vector of the line (not necessarily unit length).
    """

    def __init__(self, starting_point, line_dir):
        """
        Parameters
        ----------
        starting_point : array-like, shape (3,)
            A point on the line.
        line_dir : array-like, shape (3,)
            Direction vector of the line.
        """
        self.p = starting_point
        self.dir = line_dir


class LineSegment(Line):
    """A finite line segment in 3-D space defined by two endpoints.

    Attributes
    ----------
    p : array-like, shape (3,)
        The start point of the segment.
    dir : ndarray, shape (3,)
        Unit direction vector from ``p`` toward ``p_end``.
    p_end : array-like, shape (3,)
        The end point of the segment.
    length : float
        Euclidean length of the segment.
    """

    def __init__(self, point1, point2):
        """
        Parameters
        ----------
        point1 : array-like, shape (3,)
            Start point of the segment.
        point2 : array-like, shape (3,)
            End point of the segment.
        """
        self.p = point1
        self.dir = normalize_vectors(point2 - point1)
        self.p_end = point2
        self.length = np.linalg.norm(point2 - point1)


class Point3D:
    """A point in 3-D space with NumPy-compatible arithmetic.

    Supports element-wise addition, subtraction, multiplication and division
    with other :class:`Point3D` instances or scalars/arrays.  The object also
    acts as a length-3 array (``__array__``, ``__iter__``, ``__getitem__``),
    so it can be passed directly to NumPy functions.

    Attributes
    ----------
    x, y, z : float
        Coordinate accessors backed by a single ``(3,)`` float64 array.
    """

    def __init__(self, x, y, z):
        """
        Parameters
        ----------
        x : float
            x-coordinate.
        y : float
            y-coordinate.
        z : float
            z-coordinate.
        """
        self._coords = np.array([x, y, z], dtype=float)

    @property
    def x(self):
        return self._coords[0]

    @x.setter
    def x(self, value):
        self._coords[0] = value

    @property
    def y(self):
        return self._coords[1]

    @y.setter
    def y(self, value):
        self._coords[1] = value

    @property
    def z(self):
        return self._coords[2]

    @z.setter
    def z(self, value):
        self._coords[2] = value

    def __getitem__(self, index):
        return self._coords[index]

    def __setitem__(self, index, value):
        self._coords[index] = value

    def __array__(self, dtype=None):
        return self._coords.astype(dtype) if dtype else self._coords

    def __iter__(self):
        return iter(self._coords)

    def __len__(self):
        return len(self._coords)

    def __repr__(self):
        return repr(self._coords)  # will print like a NumPy array

    def _binary_op(self, other, op):
        if isinstance(other, Point3D):
            other = other._coords
        return Point3D(*op(self._coords, other))

    def __add__(self, other):
        return self._binary_op(other, np.add)

    def __sub__(self, other):
        return self._binary_op(other, np.subtract)

    def __mul__(self, other):
        return self._binary_op(other, np.multiply)

    def __truediv__(self, other):
        return self._binary_op(other, np.divide)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return Point3D(*np.subtract(other, self._coords))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return Point3D(*np.divide(other, self._coords))

    def __neg__(self):
        return Point3D(*-self._coords)

    def __eq__(self, other):
        if isinstance(other, Point3D):
            return np.allclose(self._coords, other._coords)
        return False

    def cone_indicator(self, cone_height, cone_radius, axis=None):
        """Check whether the point lies within a cone whose tip is at the origin.

        Parameters
        ----------
        cone_height : float
            Height of the cone along its axis.
        cone_radius : float
            Radius of the cone's base circle.
        axis : array-like, shape (3,), optional
            Axis of revolution. Defaults to ``-np.array([0, 0, 1])``.

        Returns
        -------
        bool
            ``True`` if the point lies inside the cone, ``False`` otherwise.
        """
        if axis is None:
            axis = -np.array([0, 0, 1])

        # cone tip is at origin

        axis = axis / np.linalg.norm(axis)

        cone_slope_angle = np.arctan2(cone_radius, cone_height)

        input_axis_angle = angle_between_n_vectors(self, axis)

        # projection of input point onto axis of revolution
        axis_porjection = np.dot(self, axis) * axis

        # projection of input point onto orthogonal complement of span of axis of revolution
        perp_projection = self - axis_porjection

        height_bool = np.linalg.norm(axis_porjection) <= cone_height

        radial_bool = np.linalg.norm(perp_projection) <= cone_radius

        angular_bool = np.abs(input_axis_angle) <= cone_slope_angle

        return height_bool and radial_bool and angular_bool

    def torus_indicator(self, inner_rad, outer_rad, axis=None):
        """Check whether the point lies within a solid torus centred at the origin.

        Parameters
        ----------
        inner_rad : float
            Inner radius of the solid torus (distance from axis to inner edge).
        outer_rad : float
            Outer radius of the solid torus (distance from axis to outer edge).
        axis : array-like, shape (3,), optional
            Axis of revolution. The torus lies in the plane perpendicular to this
            axis. Defaults to ``np.array([0, 0, 1])``.

        Returns
        -------
        bool
            ``True`` if the point lies inside the solid torus, ``False`` otherwise.
        """
        if axis is None:
            axis = np.array([0, 0, 1])

        # torus revolves around axis in plane through origin

        axis = axis / np.linalg.norm(axis)

        # a solid torus can be described as cartesian product
        # S^1 \times D^2
        # S^1 can be thought of as the central latitude

        rad_center = (outer_rad + inner_rad) / 2

        # radius of D^2 (outer_rad > inner_rad, but let's distrust user anyway):
        disc_rad = np.abs(outer_rad - inner_rad) / 2

        # projection onto orthogonal complement of span of axis
        axis_porjection = np.dot(self, axis) * axis
        perp_projection = self - axis_porjection

        # length of vector connecting input point to projection
        proj_len = np.linalg.norm(axis_porjection)

        # distance between projected point and central S^1:
        circle_in_plane_dist = np.abs(np.linalg.norm(perp_projection) - rad_center)

        # distance between input point and central central S^1:
        circle_dist = np.sqrt(proj_len**2 + circle_in_plane_dist**2)

        # the above distance is an indicator for whether the input point lies within solid torus
        return circle_dist <= disc_rad

    def torus_section_indicator(
        self,
        inner_rad,
        outer_rad,
        cone_radius,
        torus_revolution=None,
        cone_revolution=None,
    ):
        """Check whether the point lies in the intersection of a solid torus and a cone.

        Parameters
        ----------
        inner_rad : float
            Inner radius of the torus.
        outer_rad : float
            Outer radius of the torus (also used as the cone height).
        cone_radius : float
            Radius of the cone's base circle.
        torus_revolution : array-like, shape (3,), optional
            Axis of revolution for the torus. Defaults to ``np.array([0, 0, 1])``.
        cone_revolution : array-like, shape (3,), optional
            Axis of revolution for the cone. Defaults to ``np.array([1, 0, 0])``.

        Returns
        -------
        bool
            ``True`` if the point is inside both the torus and the cone,
            ``False`` otherwise (including when the two axes are parallel).
        """
        if torus_revolution is None:
            torus_revolution = np.array([0, 0, 1])

        if cone_revolution is None:
            cone_revolution = np.array([1, 0, 0])

        # check that axes are not (anti-)parallel
        if not np.allclose(np.zeros(3), np.cross(torus_revolution, cone_revolution)):
            torus_ind = self.torus_indicator(inner_rad, outer_rad, torus_revolution)
            cone_ind = self.cone_indicator(outer_rad, cone_radius, cone_revolution)

            return torus_ind and cone_ind

        else:
            return False


class Triangle:
    """A triangle in 3-D space defined by three :class:`Point3D` vertices.

    Attributes
    ----------
    a, b, c : Point3D
        The three vertices of the triangle.
    """

    def __init__(self, a, b, c):
        """
        Parameters
        ----------
        a : array-like, shape (3,)
            First vertex coordinates.
        b : array-like, shape (3,)
            Second vertex coordinates.
        c : array-like, shape (3,)
            Third vertex coordinates.
        """
        self.a = Point3D(a[0], a[1], a[2])
        self.b = Point3D(b[0], b[1], b[2])
        self.c = Point3D(c[0], c[1], c[2])

    def _side_lengths(self):
        """Return the three side lengths (ab, bc, ca).

        Returns
        -------
        ab, bc, ca : float
            Lengths of the sides opposite vertices c, a, and b respectively.
        """
        ab = np.linalg.norm(self.b - self.a)
        bc = np.linalg.norm(self.c - self.b)
        ca = np.linalg.norm(self.a - self.c)
        return ab, bc, ca

    def area(self):
        """Compute the area of the triangle via the cross-product formula.

        Returns
        -------
        float
            Area of the triangle.
        """
        ab = self.b - self.a
        ac = self.c - self.a
        cross = np.cross(ab, ac)
        return 0.5 * np.linalg.norm(cross)

    def inner_angles(self):
        """Compute the three interior angles of the triangle in degrees.

        Returns
        -------
        angle_A, angle_B, angle_C : float
            Interior angles at vertices A, B, and C in degrees.
        """
        ab = np.array(self.b - self.a)
        bc = np.array(self.c - self.b)
        ca = np.array(self.a - self.c)

        angle_A = angle_between_n_vectors(-ca, ab)
        angle_B = angle_between_n_vectors(-ab, bc)
        angle_C = angle_between_n_vectors(-bc, ca)

        return angle_A, angle_B, angle_C

    def inscribed_circle(self):
        """Compute the inscribed circle (incircle) of the triangle.

        Returns
        -------
        center : Point3D
            Centre of the incircle.
        radius : float
            Inradius of the triangle.
        """
        a, b, c = self._side_lengths()
        p = a + b + c
        center = (a * self.a + b * self.b + c * self.c) / p

        radius = self.area() * 2 / p
        return center, radius

    def circumcircle_radius(self):
        """Compute the circumradius of the triangle.

        Returns
        -------
        float
            Radius of the circumscribed circle.
        """
        a, b, c = self._side_lengths()
        radius = a * b * c / (4 * self.area())

        return radius

    def circumcircle(self):
        """Compute the circumscribed circle (circumcircle) of the triangle.

        Returns
        -------
        center : Point3D or ndarray
            Centre of the circumcircle (``NaN``-filled for degenerate triangles).
        radius : float
            Circumradius (``NaN`` for degenerate triangles).
        """
        ab = np.array(self.b - self.a)
        ac = np.array(self.c - self.a)
        ab_cross_ac = np.cross(ab, ac)
        norm_sq = np.linalg.norm(ab_cross_ac) ** 2

        if norm_sq == 0:
            return np.full(3, np.nan), np.nan  # Degenerate triangle

        ab_sq = np.dot(ab, ab)
        ac_sq = np.dot(ac, ac)
        ab_dot_ac = np.dot(ab, ac)

        # Vector from A to circumcenter
        alpha = ac_sq * np.dot(ab, ac) - ab_sq * np.dot(ac, ac)
        beta = ab_sq * np.dot(ac, ac) - ac_sq * np.dot(ab, ac)
        vector_to_center = (alpha * ab + beta * ac) / (2 * norm_sq)

        center = self.a + vector_to_center

        a, b, c = self._side_lengths()
        radius = a * b * c / (4 * self.area())

        return center, radius

    def __repr__(self):
        return f"Triangle3D({self.a}, {self.b}, {self.c})"


class Matrix:
    """A thin wrapper around a 2-D NumPy array with Lie-group helpers.

    Provides convenience methods for SO(3) / SE(3) checks, Lie-algebra
    computations, and noise addition.

    Attributes
    ----------
    m : ndarray
        The underlying matrix.
    nrow : int
        Number of rows.
    ncol : int
        Number of columns.
    """

    def __init__(self, input_data=None, size=3):
        """
        Parameters
        ----------
        input_data : ndarray, optional
            Pre-built matrix to wrap. If ``None``, an identity matrix of the
            given ``size`` is created.
        size : int, optional
            Edge length for the default identity matrix. Ignored when
            ``input_data`` is provided. Default is 3.

        Raises
        ------
        ValueError
            If ``size`` is not an ``int``.
        """
        if input_data is not None:
            # TODO write properly, i.e. reading from a file, changing dimensions etc.
            self.m = input_data
        else:
            if isinstance(size, int):
                self.m = np.identity(size)
            else:
                raise ValueError(
                    f"The size of the matrix has to be specified as int, instead {type(size)} was provided."
                )

        self.nrow = self.m.shape[0]
        self.ncol = self.m.shape[1]

    def is_SE3(self):
        """Check whether the matrix is a valid SE(3) roto-translation.

        Returns
        -------
        bool
            ``True`` if the upper-left 3×3 block is SO(3), the bottom row is
            ``[0, 0, 0, 1]``, and the matrix is 4×4.
        """
        rotation = self.m[:3, :3]
        if Matrix(rotation).is_SO3():
            return np.allclose(self.m[3, :3], np.zeros(3)) and np.allclose(self.m[3, 3], 1)
        else:
            return False

    def is_SO3(self):
        """
        Boolean function to check whether an input matrix is a rotation matrix.
        """
        det = np.linalg.det(self.m)
        potential_id = self.m @ self.m.T

        return np.allclose(det, 1, atol=0.01) and np.allclose(potential_id, np.identity(3), atol=0.01)

    def dual_basis_so3(self):
        """Return the coordinates of a skew-symmetric matrix in the canonical so(3) basis.

        Extracts the three independent components ``[m[2,1], m[0,2], m[1,0]]``
        that identify the rotation vector corresponding to the skew-symmetric
        matrix.

        Returns
        -------
        ndarray, shape (3,)
            Coefficients ``[ω_x, ω_y, ω_z]`` of the matrix in the canonical
            so(3) basis.
        """
        coefficients = np.asarray([self.m[2, 1], self.m[0, 2], self.m[1, 0]])
        return coefficients

    def dual_basis_se3(self, index=None):
        """Return se(3) coordinates of the matrix, optionally a single component.

        The six canonical se(3) basis coefficients are
        ``[m[2,1], m[0,2], m[1,0], m[0,3], m[1,3], m[2,3]]`` (indices 1–6).

        Parameters
        ----------
        index : int, optional
            If provided (1-based), returns the single coefficient at that
            position. If ``None``, all six coefficients are returned.

        Returns
        -------
        list or float
            All six coefficients when ``index`` is ``None``, or a single
            ``float`` when ``index`` is given.
        """
        # input index is i = 1,2,3,4,5,6
        # input matrix needs to live in se(3)
        coefficients = [self.m[2, 1], self.m[0, 2], self.m[1, 0], self.m[0, 3], self.m[1, 3], self.m[2, 3]]
        if index is None:
            return coefficients
        else:
            return coefficients[index - 1]

    def twist_from_skew_translation(self, translation):
        """Build a twist (se(3) vector) from the skew-symmetric part and a translation.

        Parameters
        ----------
        translation : ndarray, shape (3,)
            Translation component of the twist.

        Returns
        -------
        ndarray, shape (6,)
            Concatenation of the so(3) rotation vector and the translation.
        """
        skew_portion = self.dual_basis_so3()

        twist = np.hstack((skew_portion, translation))

        return twist

    def add_noise_and_project_to_so3(self, noise_level=0.05, degrees=False):
        """Add noise to a matrix and project it back to SO(3) using SVD.
        (Singular value decomp.)
        """
        # Make sure that noise level is small enough
        noise_threshold = np.pi  # injectvity radius of SO(3)

        if degrees:

            noise_level = noise_level * 180 / np.pi  # express in radians

        if noise_level < noise_threshold:
            # Noise matrix
            noise = np.random.randn(3, 3)
            noise = noise / np.linalg.norm(noise, "fro")
            # Add noise
            noisy_matrix = self.m + noise_level * noise
            # Project to the nearest orthogonal matrix with determinant +1
            U, _, Vt = np.linalg.svd(noisy_matrix)
            so3_matrix = U @ Vt
            # Ensure determinant is +1, correct if needed
            if np.linalg.det(so3_matrix) < 0:
                so3_matrix[:, -1] *= -1  # Flip the last column
            return so3_matrix
        else:
            raise ValueError(f"Noise level should be below {noise_threshold} for orientational noise.")

    def matrix_power(self, k):
        """Compute the k-th power of the wrapped square matrix.

        Parameters
        ----------
        k : int
            Non-negative integer exponent.

        Returns
        -------
        ndarray
            The matrix raised to the k-th power (identity for k=0).

        Raises
        ------
        ValueError
            If ``k`` is negative or the matrix is not square.
        TypeError
            If the wrapped matrix is not an ndarray or ``k`` is not an int.
        """
        if isinstance(self.m, np.ndarray) and isinstance(k, int):
            if k >= 0 and self.m.shape[0] == self.m.shape[1]:
                if k == 0:
                    return np.identity(self.m.shape[0])
                elif k > 0:
                    result = np.identity(self.m.shape[0])

                    while k > 0:
                        result = result @ self.m
                        k = k - 1
                    return result
            else:
                raise ValueError("k needs to be non-negative.\nmat has to be square matrix.")
        else:
            raise TypeError("mat has to be np.ndarray.")

    def SE3_cleanup(self):
        """Clean up floating-point rounding errors in an SE(3) matrix.

        Zeros out entries whose absolute value is below 1e-15.

        Returns
        -------
        ndarray or None
            The cleaned-up 4×4 SE(3) matrix, or ``None`` if the matrix is not
            a valid roto-translation (a message is printed in that case).
        """
        rotation_m = Matrix(self.m[:3, :3])
        if rotation_m.is_SE3():

            result = rotation_m.special_euclidean_from_rot_translation(self.m[:3, 3])

            threshold = 1e-15

            result = np.where(np.abs(result) < threshold, 0, result)

            return result

        else:
            print("Input matrix is not rototranslation")

    def special_euclidean_from_rot_translation(self, translation):
        """Combine the wrapped SO(3) matrix with a translation to form an SE(3) matrix.

        Parameters
        ----------
        translation : ndarray, shape (3,)
            Translation vector.

        Returns
        -------
        ndarray, shape (4, 4)
            The corresponding 4×4 homogeneous SE(3) transformation matrix.
        """
        bottom_portion = np.zeros(4)
        bottom_portion[-1] = 1

        top_portion = np.hstack((self.m, translation.reshape(-1, 1)))

        return np.vstack((top_portion, bottom_portion))

    def cone_in_plane_decomp(self):
        """
        Decomposes input rotation matrix into matrices
        cone, in_plane, where cone describes cone-rotation
        and in_plane describes in-plane-rotation.

        It holds: rot_matrix == in_plane @ cone
        """

        eulers = srot.from_matrix(self.m).as_euler("zxz")

        factor_0 = srot.from_rotvec(eulers[-1] * np.array([0, 0, 1])).as_matrix()
        factor_1 = srot.from_rotvec(eulers[1] * factor_0 @ np.array([1, 0, 0])).as_matrix()
        factor_2 = srot.from_rotvec(eulers[0] * factor_1 @ np.array([0, 0, 1])).as_matrix()

        cone = factor_1 @ factor_0  # factor_1 @ factor_0
        in_plane = factor_2

        return cone, in_plane

    def in_plane_angle(self):
        """Extract the in-plane rotation angle (first ZXZ Euler angle) in radians.

        Returns
        -------
        float
            The φ angle (first ZXZ Euler angle) in radians.
        """
        eulers = srot.from_matrix(self.m).as_euler("zxz")

        return eulers[0]


def tetrahedron() -> np.ndarray:
    """Return the four unit-sphere vertices of a regular tetrahedron.

    Returns
    -------
    numpy.ndarray
        ``(4, 3)`` array of vertices on the unit sphere.
    """

    vertices = 1 / ROOT3 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])

    return vertices


def octahedron() -> np.ndarray:
    """Return the six unit-sphere vertices of a regular octahedron.

    Returns
    -------
    numpy.ndarray
        ``(6, 3)`` array of vertices on the unit sphere.
    """

    vertices = np.vstack((np.identity(3), -np.identity(3)))

    return vertices


def cube() -> np.ndarray:
    """Return the eight unit-sphere vertices of a cube.

    Returns
    -------
    numpy.ndarray
        ``(8, 3)`` array of vertices on the unit sphere.
    """

    vertices = (
        1
        / ROOT3
        * np.array([[1, 1, 1], [-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    )

    return vertices


def icosahedron() -> np.ndarray:
    """Return the twelve unit-sphere vertices of a regular icosahedron.

    Returns
    -------
    numpy.ndarray
        ``(12, 3)`` array of vertices on the unit sphere.
    """
    vertices = np.array(
        [
            [0, 1, PHI],
            [0, 1, -PHI],
            [0, -1, PHI],
            [0, -1, -PHI],
            [1, PHI, 0],
            [1, -PHI, 0],
            [-1, PHI, 0],
            [-1, -PHI, 0],
            [PHI, 0, 1],
            [-PHI, 0, 1],
            [PHI, 0, -1],
            [-PHI, 0, -1],
        ]
    )

    return vertices / np.linalg.norm(vertices, axis=1, keepdims=True)


def dodecahedron() -> np.ndarray:
    """Return the twenty unit-sphere vertices of a regular dodecahedron.

    The twelve face-centre vertices are combined with the eight cube vertices
    (see :func:`cube`) to produce all twenty dodecahedron vertices on the unit
    sphere.

    Returns
    -------
    numpy.ndarray
        ``(20, 3)`` array of vertices on the unit sphere.
    """
    vertices = np.array(
        [
            [0, 1 / PHI, PHI],
            [0, -1 / PHI, PHI],
            [0, 1 / PHI, -PHI],
            [0, -1 / PHI, -PHI],
            [1 / PHI, PHI, 0],
            [-1 / PHI, PHI, 0],
            [1 / PHI, -PHI, 0],
            [-1 / PHI, -PHI, 0],
            [PHI, 0, 1 / PHI],
            [-PHI, 0, 1 / PHI],
            [PHI, 0, -1 / PHI],
            [-PHI, 0, -1 / PHI],
        ]
    )

    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    vertices = np.vstack((vertices, cube()))

    return vertices


def n_gon_points(n: int) -> np.ndarray:
    """Return the unit-circle vertices of a regular n-gon in the xy-plane.

    Parameters
    ----------
    n : int
        Number of vertices in the polygon.

    Returns
    -------
    numpy.ndarray
        ``(n, 2)`` array of 2D vertex coordinates on the unit circle.
    """

    coordinates = [np.array([np.cos(2 * np.pi * k / n), np.sin(2 * np.pi * k / n)]) for k in range(n)]

    return np.vstack(coordinates)


def great_circle_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Great-circle (geodesic) distance between two unit-sphere points.

    Parameters
    ----------
    p1 : numpy.ndarray
        First unit-sphere point.
    p2 : numpy.ndarray
        Second unit-sphere point.

    Returns
    -------
    float
        Geodesic angle (radians) between the two points.
    """

    dot_product = np.clip(np.dot(p1, p2), -1.0, 1.0)
    return np.arccos(dot_product)


def min_great_circle_distance(set1: np.ndarray, set2: np.ndarray) -> float:
    """Compute the minimum great-circle distance between two sets of points on S^n.

    Parameters
    ----------
    set1 : numpy.ndarray
        ``(N, d)`` array of unit-sphere points.
    set2 : numpy.ndarray
        ``(M, d)`` array of unit-sphere points.

    Returns
    -------
    float
        Minimum geodesic distance between any pair drawn from the two sets.
    """
    min_distance = np.inf
    for p1 in set1:
        for p2 in set2:
            dist = great_circle_distance(p1, p2)
            min_distance = min(min_distance, dist)

    return min_distance


def great_circle_distance_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances between two sets of points on an n-sphere.

    Parameters
    ----------
    points1 : numpy.ndarray
        ``(N, d)`` array of unit-sphere points.
    points2 : numpy.ndarray
        ``(M, d)`` array of unit-sphere points.

    Returns
    -------
    numpy.ndarray
        ``(N, M)`` matrix of geodesic distances.
    """
    dot_products = np.clip(np.dot(points1, points2.T), -1.0, 1.0)
    return np.arccos(dot_products)


def hausdorff_distance_sphere(set1: np.ndarray, set2: np.ndarray) -> float:
    """Simplified Hausdorff distance between two discrete sets of points on an n-sphere.

    Parameters
    ----------
    set1 : numpy.ndarray
        ``(N, d)`` array of unit-sphere points.
    set2 : numpy.ndarray
        ``(M, d)`` array of unit-sphere points.

    Returns
    -------
    float
        Symmetric Hausdorff distance.
    """
    dist_matrix = great_circle_distance_matrix(set1, set2)

    # Compute directed Hausdorff distances
    dist_hd_ab = np.max(np.min(dist_matrix, axis=1))
    dist_hd_ba = np.max(np.min(dist_matrix, axis=0))

    return max(dist_hd_ab, dist_hd_ba)


def project_points_on_plane_with_preserved_distance(
    starting_point: np.ndarray,
    normal: np.ndarray,
    nn_points: np.ndarray,
) -> np.ndarray:
    """Project approximately coplanar points around a starting_point onto the
    plane perpendicular to normal vector. The distances between projected nearest neighbors and
    starting point are preserved.

    Parameters
    ----------
    starting_point : ndarray
        origin of plane specified by normal vector
    normal : ndarray
        normal vector to plane
    nn_points : ndarray
        nearest neighbors of starting_point

    Returns
    -------
    ndarray
        projected points on plane specified by starting_point and normal
    """

    # Compute the projection of each neighbor point onto the plane defined by the starting point and normal vector
    projection_lengths = np.dot(nn_points - starting_point, normal) / np.linalg.norm(normal)
    projected_points = nn_points - np.outer(projection_lengths, normal)

    # Compute the distances between the neighbor points and the starting point
    distances_to_starting_point = np.linalg.norm(nn_points - starting_point, axis=1)

    # Compute the distances between the projected points and the starting point
    distances_to_projected_point = np.linalg.norm(projected_points - starting_point, axis=1)

    # Compute the adjustment vectors
    adjustment_vectors = projected_points - starting_point

    # Compute the shifted points
    shifted_points = (
        starting_point
        + adjustment_vectors * (distances_to_starting_point / distances_to_projected_point)[:, np.newaxis]
    )

    # distances_to_projected_point = np.linalg.norm(shifted_points - starting_point, axis=1)

    return shifted_points


def align_points_to_xy_plane(
    points_on_plane: np.ndarray,
    plane_normal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Plane is rotated to be aligned with xy-plane.

    Parameters
    ----------
    points_on_plane : ndarray
        coplanar points
    plane_normal : ndarray, optional
        Plane normal. Defaults to None.
        If None, plane normal is estimated from points_on_plane

    Raises
    ------
    ValueError
        One needs at least 3 points to specify a plane if plane normal is not given.

    Returns
    -------
    ndarray (n,3), ndarray (3,3)
        Points in xy-plane, corresponding rotation matrix
    """

    if plane_normal is None:
        if points_on_plane.shape[0] >= 3:
            # Calculate the normal vector of the plane
            v1 = points_on_plane[1] - points_on_plane[0]
            v2 = points_on_plane[2] - points_on_plane[0]
            normal = np.cross(v1, v2)
        else:
            raise ValueError(
                f"The plane has to be specified either by plane normal or at least three points lying on the plane!"
            )
    else:
        normal = plane_normal

    normal = normal / np.linalg.norm(normal)

    # Find the rotation matrix to align normal with z-axis
    z_axis = np.array([0, 0, 1])
    axis = np.cross(normal, z_axis)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.dot(normal, z_axis))
    rotation_matrix = srot.from_rotvec(angle * axis).as_matrix()  # Construct rotation matrix

    # Apply rotation matrix to all points
    rotated_points = np.dot(rotation_matrix, points_on_plane.T).T
    # rotated_points = np.dot(rotation_matrix, plane_normal.T).T

    # return rotated_points[:, 0:2], rotation_matrix
    return rotated_points, rotation_matrix


def spline_sampling(coords: pd.DataFrame, sampling_distance: float) -> np.ndarray:
    """Samples a spline specified by coordinates with a given sampling distance

    Parameters
    ----------
    coords : pandas.DataFrame
        coordinates of the spline (rows are points; uses ``.iloc`` and
        ``.iterrows()`` so a DataFrame is required).
    sampling_distance : float
        sampling frequency in pixels

    Returns
    -------
    ndarray
        coordinates of points on the spline
    """

    # spline = UnivariateSpline(np.arange(0, len(coords), 1), coords.to_numpy())
    spline = InterpolatedUnivariateSpline(np.arange(0, len(coords), 1), coords.to_numpy())

    # Keep track of steps across whole tube
    totalsteps = 0

    for i, row in coords.iterrows():
        if i == 0:
            continue
        # Calculate projected distance between each point
        # TODO: check this issue
        dist = point_pairwise_dist(row, coords.iloc[i - 1])

        # Number of steps between two points; steps are roughly in increments of 1 pixel
        stepnumber = round(dist / sampling_distance)
        # Length of each step
        step = 1 / stepnumber
        # Array to hold fraction of each step between points
        t = np.arrange(i - 1, i, step)  # inclusive end in matlab

        # Evaluate piecewise-polynomial, i.e. spline, at steps 't'.
        # This array contains the Cartesian coordinates of each step

        # Ft(:,totalsteps+1:totalsteps+size(t,2))=ppval(F, t) # TODO
        # scipy.interpolate.NdPPoly
        spline_t = spline(t)

        # Increment the step counter
        totalsteps += len(t)

        return spline_t


def compare_rotations(
    input_rotation_1: RotationLike,
    input_rotation_2: RotationLike,
    cyclic_symmetry: Symmetry = 1,
    rotation_type: str = "all",
) -> tuple[float, float, float] | float:
    """Compare the rotations between two sets of angles.

    Parameters
    ----------
    input_rotation_1 : RotationLike
        The first set of rotations (Euler angles, matrices, quaternions or
        :class:`scipy.spatial.transform.Rotation`).
    input_rotation_2 : RotationLike
        The second set of rotations, same conventions as ``input_rotation_1``.
    cyclic_symmetry : Symmetry, default=1
        Cyclic rotational symmetry specifier (e.g. ``"C5"`` or ``5``); normalized
        via :func:`as_symmetry`.
    rotation_type : {"all", "angular_distance", "cone_distance", "in_plane_distance"}, default="all"
        Selects which distance(s) to return.

    Returns
    -------
    tuple
        A tuple containing the following distances:
        - dist_degrees (float): The overall angular distance between the two sets of angles.
        - dist_degrees_normals (float): The angular distance between the normal vectors of the two sets of angles.
        - dist_degrees_inplane (float): The angular distance within the plane of rotation between the two sets of angles.

    """

    _, cyclic_symmetry = as_symmetry(cyclic_symmetry)

    dist_degrees = angular_distance(input_rotation_1, input_rotation_2, cyclic_symmetry=cyclic_symmetry)[0]
    dist_degrees_normals, dist_degrees_inplane = cone_inplane_distance(input_rotation_1, input_rotation_2, cyclic_symmetry=cyclic_symmetry)

    if rotation_type == "all":
        return dist_degrees, dist_degrees_normals, dist_degrees_inplane
    elif rotation_type == "angular_distance":
        return dist_degrees
    elif rotation_type == "cone_distance":
        return dist_degrees_normals
    elif rotation_type == "in_plane_distance":
        return dist_degrees_inplane
    else:
        raise UserInputError(f"The rotation type {rotation_type} is not supported.")


def euler_angles_to_normals(angles: EulerAngles) -> np.ndarray:
    """Compute normal vectors pointing in z-direction from Euler angles.

    Parameters
    ----------
    angles : EulerAngles
        Single triple ``(3,)`` or a stack ``(N, 3)`` of Euler angles in degrees
        (zxz convention).

    Returns
    -------
    ndarray (n,3)
        Unit length z-normal vectors associated to input Euler angles.
    """
    rotations = srot.from_euler("zxz", angles=angles, degrees=True)
    points = rotations_to_z_normals(rotations)
    n_length = np.linalg.norm(points)
    normalized_normal_vectors = points / n_length

    return normalized_normal_vectors


def normals_to_euler_angles(
    input_normals: np.ndarray | pd.DataFrame,
    output_order: str = "zxz",
) -> np.ndarray:
    """Given normal vectors pointing in z-direction in particle frames,
    compute choice of Euler angles.

    Parameters
    ----------
    input_normals : numpy.ndarray or pandas.DataFrame
        z-normal vectors. DataFrames must have ``x``, ``y``, ``z`` columns.
    output_order : str, optional
        Euler angle convention. Defaults to "zxz".

    Raises
    ------
    UserInputError
        input_normals have to be either pandas dataFrame or numpy array.

    Returns
    -------
    ndarray : (n,3)
        n triplets of Euler angles in choses convention.
    """
    if isinstance(input_normals, pd.DataFrame):
        normals = input_normals.loc[:, ["x", "y", "z"]].values
    elif isinstance(input_normals, np.ndarray):
        normals = input_normals
    else:
        raise UserInputError("The input_normals have to be either pandas dataFrame or numpy array")

    # normalize vectors
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
    theta = np.degrees(np.arctan2(np.sqrt(normals[:, 0] ** 2 + normals[:, 1] ** 2), normals[:, 2]))

    psi = np.degrees(np.arctan2(normals[:, 0], -normals[:, 1])) #90 + np.degrees(np.arctan2(normals[:, 1], normals[:, 0]))
    b_idx = np.where(np.arctan2(normals[:, 1], normals[:, 0]) == 0)
    psi[b_idx] = 0

    phi = np.random.rand(normals.shape[0]) * 360

    if output_order == "zzx":
        angles = np.column_stack((phi, psi, theta))
    else:
        angles = np.column_stack((phi, theta, psi))

    return angles


def quaternion_mult(qs1: np.ndarray, qs2: np.ndarray) -> np.ndarray:
    """Given arrays of quaternions in scalar-last convention, compute
    array of products of unit quaternions.

    Parameters
    ----------
    qs1 : ndarray (n,4)
        n quaternions in scalar-last convention.
    qs2 : ndarray (n,4)
        n quaternions in scalar-last convention.

    Returns
    -------
    ndarray (n,4)
        n quaternions in scalar-last convention.
        Row i is product of qs1[i] and qs2[i].

    """
    mutliplied = []
    for q, q1 in enumerate(qs1):
        q2 = qs2[q, :]
        w = q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2]
        i = q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1]
        j = q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0]
        k = q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3]
        mutliplied.append(np.array([i, j, k, w]))

    return np.vstack(mutliplied)


def quaternion_log(q: np.ndarray) -> np.ndarray:
    """Given array of unit scalar-last quaternions, compute array of
    unit-quaternion logarithms.

    Parameters
    ----------
    q : ndarray (n,4)
        Array of n unit quaternions in scalar-last convention.

    Returns
    -------
    ndarray (n,4)
        Array of n quaternion logarithms.
    """
    v_norm = np.linalg.norm(q[:, :3], axis=1)
    q_norm = np.linalg.norm(q, axis=1)

    tolerance = 10e-14

    new_scalar = []
    new_vector = []

    for i, _ in enumerate(q):
        if q_norm[i] < tolerance:
            # 0 quaternion - undefined
            new_scalar.append(0.0)
            new_vector.append([0.0, 0.0, 0.0])

        elif v_norm[i] < tolerance:
            # real quaternions - no imaginary part
            new_scalar.append(np.log(q_norm[i]))
            new_vector.append([0, 0, 0])
        else:
            vec = q[i, :3] / v_norm[i]
            new_scalar.append(np.log(q_norm[i]))
            vector = np.arccos(q[i, 3] / q_norm[i])
            vector = vec * vector
            new_vector.append(vector)

    new_vector = np.vstack(new_vector)

    new_scalar = np.array(new_scalar).reshape(q.shape[0], 1)
    return np.hstack([new_vector, new_scalar])


def cone_distance(input_rotation_1: RotationLike, input_rotation_2: RotationLike) -> np.ndarray:
    """Compute great-circle distance between z-normals corresponding to orientations
    as represented by input rotations. This corresponds to angular distance between cone-rotation
    portions of respective input rotations.

    Parameters
    ----------
    input_rotation_1 : RotationLike
        Rotation describing orientation of particle (Rotation, Euler triple/stack,
        matrix, or quaternion). Normalized via :func:`as_rotation`.
    input_rotation_2 : RotationLike
        Rotation describing orientation of particle. Normalized via
        :func:`as_rotation`.

    Returns
    -------
    numpy.ndarray
        cone-distance in degrees
    """
    input_rotation_1 = as_rotation(input_rotation_1)
    input_rotation_2 = as_rotation(input_rotation_2)
    point = [0, 0, 1.0]

    vec1 = np.array(input_rotation_1.apply(point), ndmin=2)
    vec2 = np.array(input_rotation_2.apply(point), ndmin=2)

    vec1_n = np.linalg.norm(vec1, axis=1)
    vec1 = vec1 / vec1_n[:, np.newaxis]
    vec2_n = np.linalg.norm(vec2, axis=1)
    vec2 = vec2 / vec2_n[:, np.newaxis]
    cone_angle = np.degrees(np.arccos(np.maximum(np.minimum(np.sum(vec1 * vec2, axis=1), 1.0), -1.0)))

    return cone_angle


def get_axis_from_rotation(input_rotation: RotationLike, axis: str = "z") -> np.ndarray:
    """Given an input rotation, compute the desired unit normal vector
    from the coordinate frame associated to the rotation.

    Parameters
    ----------
    input_rotation : RotationLike
        Rotation describing orientation of particle (Rotation, Euler triple/stack,
        matrix, or quaternion). Normalized via :func:`as_rotation`.
    axis : str, optional
        Desired coordinate direction. Defaults to "z".

    Raises
    ------
    ValueError
        Input must be valid scipy rotation object.

    Returns
    -------
    ndarray
        unit vector
    """

    input_rotation = as_rotation(input_rotation)
    matrix_rep = input_rotation.as_matrix()

    axes_dict = {"x": 0, "y": 1, "z": 2}

    if matrix_rep.shape == (3, 3):  # Single (3, 3) matrix
        ret_axis = matrix_rep[:, axes_dict[axis]]  # Extract column 1 for a single matrix
    elif matrix_rep.shape[1:] == (3, 3):  # Multiple (N, 3, 3) matrices
        ret_axis = matrix_rep[:, :, axes_dict[axis]]  # Extract column 1 for each (N, 3, 3) matrix
    else:
        raise ValueError("Input must be valid scipy rotation object.")

    return ret_axis


def inplane_distance(
    input_rotation_1: RotationLike,
    input_rotation_2: RotationLike,
    convention: str = "zxz",
    degrees: bool = True,
    cyclic_symmetry: Symmetry = 1,
) -> np.ndarray:
    """Compute the angular distance between inplane-rotation portion of two given rotations.

    Parameters
    ----------
    input_rotation_1 : RotationLike
        Rotation describing orientation of particle. Normalized via :func:`as_rotation`.
    input_rotation_2 : RotationLike
        Rotation describing orientation of particle. Normalized via :func:`as_rotation`.
    convention : str, optional
        Euler angle convention. Defaults to "zxz".
    degrees : bool, optional
        Return angular distance in degrees (True) or radians (False). Defaults to True.
    cyclic_symmetry : Symmetry, default=1
        Cyclic rotational symmetry specifier of underlying particles (``"C5"`` or
        ``5``); normalized via :func:`as_symmetry`.

    Returns
    -------
    float
        Angular distance between inplane rotations.
    """
    input_rotation_1 = as_rotation(input_rotation_1, euler_order=convention, degrees=degrees)
    input_rotation_2 = as_rotation(input_rotation_2, euler_order=convention, degrees=degrees)
    _, cyclic_symmetry = as_symmetry(cyclic_symmetry)
    phi1 = np.array(input_rotation_1.as_euler(convention, degrees=degrees), ndmin=2)[:, 0]
    phi2 = np.array(input_rotation_2.as_euler(convention, degrees=degrees), ndmin=2)[:, 0]

    # Remove flot precision errors during conversion
    phi1 = np.where(abs(phi1) < ANGLE_DEGREES_TOL, 0.0, phi1)
    phi2 = np.where(abs(phi2) < ANGLE_DEGREES_TOL, 0.0, phi2)

    # From Scipy the phi is from [-180,180] -> change to [0.0,360]
    phi1 += 180.0
    phi2 += 180.0

    # Get the angular range for symmetry and divide the angles to be only in that range
    if cyclic_symmetry > 1:
        sym_div = 360.0 / cyclic_symmetry
        phi1 = np.mod(phi1, sym_div)
        phi2 = np.mod(phi2, sym_div)

    inplane_angle = np.abs(phi1 - phi2)

    inplane_angle = np.where(inplane_angle > 180.0, np.abs(inplane_angle - 360.0), inplane_angle)

    return inplane_angle


def cone_inplane_distance(
    input_rotation_1: RotationLike,
    input_rotation_2: RotationLike,
    convention: str = "zxz",
    degrees: bool = True,
    cyclic_symmetry: Symmetry = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute angular distance between cone-rotations and inplane-rotations, respectively.

    Parameters
    ----------
    input_rotation_1 : RotationLike
        Rotation describing orientation of particle. Normalized via :func:`as_rotation`.
    input_rotation_2 : RotationLike
        Rotation describing orientation of particle. Normalized via :func:`as_rotation`.
    convention : str, optional
        Euler angle convention. Defaults to "zxz".
    degrees :bool, optional
        Return angular distance in degrees (True) or radians (False). Defaults to True.
    cyclic_symmetry : Symmetry, default=1
        Cyclic rotational symmetry specifier of underlying particles; normalized
        via :func:`as_symmetry`.

    Returns
    -------
    float
        Angular distance between cone-rotations
    float
        angular distance between inplane rotations.
    """
    rot1 = as_rotation(input_rotation_1, euler_order=convention, degrees=degrees)
    rot2 = as_rotation(input_rotation_2, euler_order=convention, degrees=degrees)
    _, cyclic_symmetry = as_symmetry(cyclic_symmetry)

    cone_angle = cone_distance(rot1, rot2)
    inplane_angle = inplane_distance(rot1, rot2, convention, degrees, cyclic_symmetry)

    return cone_angle, inplane_angle


def angular_score_for_c_symmetry(
    inplane_1: ArrayLike,
    inplane_2: ArrayLike,
    cyclic_symmetry: Symmetry,
    max_val: float | None = None,
) -> np.ndarray:
    """
    Computes an angular similarity score for arrays of in-plane angles, based on rotational symmetry.

    Parameters:
    -----------
    inplane_1 : ArrayLike
        First set of in-plane angles (in radians). Normalized via
        :func:`numpy.atleast_1d` / :func:`numpy.asarray`.
    inplane_2 : ArrayLike
        Second set of in-plane angles (in radians). Same handling as ``inplane_1``.
    cyclic_symmetry : Symmetry
        Cyclic symmetry specifier (``"Cn"`` or ``n``); normalized via
        :func:`as_symmetry`. Must specify an order greater than 1.
    max_val : float, optional
        Maximum possible angular distance for normalization.

    Returns:
    --------
    np.ndarray: Array of angular similarity scores in [0, 1].
    """
    _, symm = as_symmetry(cyclic_symmetry)
    if symm <= 1:
        raise ValueError("cyclic_symmetry must specify an order greater than 1.")

    if max_val is None:
        max_val = np.pi / symm

    vertices = n_gon_points(symm)
    inplane_1 = np.atleast_1d(inplane_1)
    inplane_2 = np.atleast_1d(inplane_2)

    if inplane_1.shape != inplane_2.shape:
        raise ValueError("inplane_1 and inplane_2 must have the same shape.")

    scores = []

    for angle1, angle2 in zip(inplane_1, inplane_2):
        rot_1 = np.array([[np.cos(angle1), -np.sin(angle1)], [np.sin(angle1), np.cos(angle1)]])
        rot_2 = np.array([[np.cos(angle2), -np.sin(angle2)], [np.sin(angle2), np.cos(angle2)]])
        vert_1 = vertices @ rot_1.T
        vert_2 = vertices @ rot_2.T

        sim_measure = hausdorff_distance_sphere(vert_1, vert_2)
        out = 1 - sim_measure / max_val

        # Refine the bounds
        if out > 1 - 1e-5:
            out = 1.0
        elif out < 1e-5:
            out = 0.0

        scores.append(out)

    return np.array(scores)


def angular_distance(
    input_rotation_1: RotationLike,
    input_rotation_2: RotationLike,
    convention: str = "zxz",
    degrees: bool = True,
    cyclic_symmetry: Symmetry = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute angular distance between two rotations.
    Formula is based on this post
    https://math.stackexchange.com/questions/90081/quaternion-distance

    Parameters
    ----------
    input_rotation_1 : RotationLike
        Rotation describing orientation of particle. Normalized via :func:`as_rotation`.
    input_rotation_2 : RotationLike
        Rotation describing orientation of particle. Normalized via :func:`as_rotation`.
    convention : str, optional
        Euler angle convention. Defaults to "zxz".
    degrees : bool, optional
        Return angular distance in degrees (True) or radians (False). Defaults to True.
    cyclic_symmetry : Symmetry, default=1
        Cyclic rotational symmetry specifier of underlying particles; normalized
        via :func:`as_symmetry`.

    Returns
    -------
    float
        Angular distance between input rotations.

    Examples
    --------
    >>> rot1 = srot.from_euler("zxz", [0, 0, 0], degrees=True)
    >>> rot2 = srot.from_euler("zxz", [45, 45, 0], degrees=True)
    >>> angular_distance(rot1, rot2)
    45.0
    """

    rot1 = as_rotation(input_rotation_1, euler_order=convention, degrees=degrees)
    rot2 = as_rotation(input_rotation_2, euler_order=convention, degrees=degrees)
    _, cyclic_symmetry = as_symmetry(cyclic_symmetry)

    if cyclic_symmetry > 1:
        angles1 = rot1.as_euler(convention, degrees=degrees)
        angles2 = rot2.as_euler(convention, degrees=degrees)
        sym_div = 360.0 / cyclic_symmetry
        angles1[:, 0] = np.mod(angles1[:, 0], sym_div)
        angles2[:, 0] = np.mod(angles2[:, 0], sym_div)
        rot1 = srot.from_euler(convention, angles1, degrees=degrees)
        rot2 = srot.from_euler(convention, angles2, degrees=degrees)

    q1 = np.array(rot1.as_quat(), ndmin=2)
    q2 = np.array(rot2.as_quat(), ndmin=2)

    if q1.shape != q2.shape:
        print("The size of input rotations differ!!!")
        return

    angle = np.degrees(2 * np.arccos(np.abs(np.sum(q1 * q2, axis=1))))
    angle = angle.astype(float)

    dist = 1 - np.power(np.sum(q1 * q2, 1), 2)

    dist[dist < 10e-8] = 0

    return angle, dist


def number_of_cone_rotations(cone_angle: float, cone_sampling: float) -> int:
    """Calculates the number of rotations required for a sampling process
    of cone-angles based on a sampling interval.

    Parameters
    ----------
    cone_angle : float
        The total cone-angle in degrees.
    cone_sampling : float
        The angular sampling interval in degrees.

    Returns
    -------
    int
        The total number of rotations required for the sampling process.
    """
    # Theta steps
    theta_max = cone_angle / 2
    temp_steps = theta_max / cone_sampling
    theta_array = np.linspace(0, theta_max, round(temp_steps) + 1)
    arc = 2.0 * np.pi * (cone_sampling / 360.0)

    number_of_rotations = 2  # starting and ending angle

    # Generate psi angles
    for i, theta in enumerate(theta_array[(theta_array > 0) & (theta_array < 180)]):
        radius = np.sin(theta * np.pi / 180.0)  # Radius of circle
        circ = 2.0 * np.pi * radius  # Circumference
        number_of_rotations += np.ceil(circ / arc) + 1  # Number of psi steps

    return number_of_rotations


def sample_cone(
    cone_angle: float,
    cone_sampling: float,
    center: TripletLike | None = None,
    radius: float = 1.0,
) -> np.ndarray:
    """Creates an "even" distibution on sphere. Works for tame cases.
    Source:
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/26127012#26127012

    Parameters
    ----------
    cone_angle : float
        Angle for sampling of cone-angles (refers to range of z-normals of particles).
    cone_sampling : float
        Frequency for cone sampling.
    center : TripletLike, optional
        Center of sphere to be sampled (single number broadcast to all axes, or a
        3-element array-like). Normalized via :func:`as_triplet`. Defaults to
        None (origin).
    radius : float, optional
        Radius of sphere to be sampled. Defaults to 1.0.

    Returns
    -------
    ndarray
        Samples on sphere.
    """

    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = as_triplet(center)

    number_of_points = number_of_cone_rotations(cone_angle, cone_sampling)

    # golden angle in radians
    phi = np.pi * (3 - np.sqrt(5))
    cone_size = cone_angle / 180.0

    north_pole = [0.0, 0.0, radius]
    if center is not None:
        north_pole = north_pole + center
    sampled_points = [north_pole]
    for i in np.arange(1, number_of_points, 1):
        # z goes from 1 to -1 for 360 degrees (i.e., a full sphere), is less
        z = 1 - (i / (number_of_points - 1)) * cone_size

        sp_radius = np.sqrt(1 - z * z)

        # golden angle increment
        theta = phi * i
        x = np.cos(theta) * sp_radius
        y = np.sin(theta) * sp_radius

        x = x * radius + center[0]
        y = y * radius + center[1]
        z = z * radius + center[2]

        sampled_points.append(np.array([x, y, z]))

    return np.stack(sampled_points, axis=0)


def generate_angles(
    cone_angle: float,
    cone_sampling: float,
    inplane_angle: float = 360.0,
    inplane_sampling: float | None = None,
    starting_angles: EulerAngles | None = None,
    symmetry: Symmetry = 1,
    angle_order: str = "zxz",
) -> np.ndarray:
    """Compute Euler angles from sample for normal vectors on sphere.
    Sphere sample corresponds to cone-angles.

    Parameters
    ----------
    cone_angle : float
        Angle for sampling of cone-angles (refers to range of z-normals of particles).
    cone_sampling : float
        Frequency for cone sampling.
    inplane_angle : float, optional
        Desired inplane-angles for particle orientations. Defaults to 360.0.
    inplane_sampling : float, optional
        Frequency for sampling of inplane-angles. Defaults to None.
    starting_angles : EulerAngles, optional
        Triplet of Euler angles in convention as specified by ``angle_order``
        (single triple ``(3,)`` or 3-list/tuple). Defaults to None.
    symmetry : Symmetry, default=1
        Rotational symmetry specifier (``"Cn"`` or ``n``); normalized via
        :func:`as_symmetry`.
    angle_order : str, optional
        Convention for Euler angles. Defaults to "zxz".

    Returns
    -------
    ndarray
        Sample of Euler angles.
    """
    _, symmetry = as_symmetry(symmetry)

    points = sample_cone(cone_angle, cone_sampling)
    angles = normals_to_euler_angles(points, output_order=angle_order)
    angles[:, 0] = 0.0

    starting_phi = 0.0

    # if case of no starting angles one can directly do cone_angles = angles
    # but going through rotation object will set the angles to the canonical set
    cone_rotations = srot.from_euler(angle_order, angles=angles, degrees=True)

    if starting_angles is not None:
        starting_angles = np.asarray(starting_angles, dtype=float)
        starting_rot = srot.from_euler(angle_order, angles=starting_angles, degrees=True)
        cone_rotations = starting_rot * cone_rotations  # swapped order w.r.t. the quat_mult in matlab!
        starting_phi = starting_angles[0]

    cone_angles = cone_rotations.as_euler(angle_order, degrees=True)
    cone_angles = cone_angles[:, 1:3]

    # Calculate phi angles
    if inplane_sampling is None:
        inplane_sampling = cone_sampling

    if inplane_angle != 360.0:
        phi_max = min(360.0 / symmetry, inplane_angle)
    else:
        phi_max = inplane_angle / symmetry

    phi_steps = phi_max / inplane_sampling
    phi_array = np.linspace(0, phi_max, round(phi_steps) + 1)
    phi_array = phi_array[:-1]  # Final angle is redundant

    if phi_array.size == 0:
        phi_array = np.array([[0.0]])

    n_phi = np.size(phi_array)
    phi_array = phi_array + starting_phi

    # Generate angle list
    angular_array = np.concatenate(
        [
            np.tile(phi_array[:, np.newaxis], (cone_angles.shape[0], 1)),
            np.repeat(cone_angles, n_phi, axis=0),
        ],
        axis=1,
    )

    return angular_array


def rotations_to_z_normals(
    input_rotation: RotationLike,
    radius: float = 1.0,
) -> np.ndarray:
    """Compute z-normals of input input_rotation.

    Each rotation is applied to the reference vector ``(0, 0, radius)`` to
    obtain a point on (or rescaled from) the unit sphere.

    Parameters
    ----------
    input_rotation : RotationLike
        Orientations. Normalized via :func:`as_rotation`.
    radius : float, default=1.0
        Length of the reference vector. The returned points lie on a sphere
        of this radius.

    Returns
    -------
    numpy.ndarray
        Shape ``(N, 3)``. Z-normal vectors for the input input_rotation.

    See Also
    --------
    cryocat.analysis.visplot.plot_rotation_normals : Plot the resulting
        z-normals as a 3-D scatter.
    """
    input_rotation = as_rotation(input_rotation)
    starting_point = np.array([0.0, 0.0, radius])
    return np.array(input_rotation.apply(starting_point), ndmin=2)


def compute_relative_orientations(
    angles: np.ndarray,
    direction_vectors: np.ndarray,
) -> np.ndarray:
    """Compute orientations relative to a reference frame built from particle 0.

    For each particle, a local right-handed orthonormal frame is constructed
    from its current orientation and its direction-to-target vector::

        v1 = z-normal of the particle's rotation
        v2 = direction_vectors[i] / ||direction_vectors[i]||
        v3 = (v1 Ã v2) / ||v1 Ã v2||

    The reference frame ``W`` is the local frame of particle 0 (columns of
    ``W`` are ``(w1, w2, w3)``); each particle ``i`` gets its rotation as
    ``W Â· V_i`` where ``V_i`` has rows ``(v1, v2, v3)`` of particle ``i``.
    Equivalently, the returned Euler angles describe the rotation that takes
    particle ``i``'s local frame onto particle 0's local frame.

    Row 0 is always ``(0, 0, 0)`` â particle 0 is the reference.

    Parameters
    ----------
    angles : numpy.ndarray
        Shape ``(N, 3)``. Particle orientations as zxz Euler angles in degrees.
    direction_vectors : numpy.ndarray
        Shape ``(N, 3)``. Direction from each particle toward its target
        (centroid, nearest neighbor, ...). Need not be unit-length.

    Returns
    -------
    numpy.ndarray
        Shape ``(N, 3)``. Relative orientations as zxz Euler angles in degrees.

    Notes
    -----
    The algorithm is undefined when a particle's z-normal is parallel to its
    direction vector (the cross product is zero); the resulting Euler angles
    will contain NaN in that case. Callers are expected to filter such pairs
    upstream.
    """
    n = angles.shape[0]
    if direction_vectors.shape != (n, 3):
        raise ValueError(
            f"direction_vectors must have shape ({n}, 3); got {direction_vectors.shape}."
        )

    v1 = euler_angles_to_normals(angles)
    v2 = direction_vectors / np.linalg.norm(direction_vectors, axis=1, keepdims=True)
    v3 = np.cross(v1, v2)
    v3 = v3 / np.linalg.norm(v3, axis=1, keepdims=True)

    # Per-particle V_i has rows (v1, v2, v3); shape (N, 3, 3).
    v_base_mat = np.stack([v1, v2, v3], axis=1)

    # Reference frame W from particle 0 with columns (w1, w2, w3).
    # Equivalently: transpose of V_0 (which has them as rows).
    w_base_mat = v_base_mat[0].T

    # final_mat[i] = W @ V_i, broadcasting (3, 3) against (N, 3, 3).
    final_mat = w_base_mat @ v_base_mat

    return srot.from_matrix(final_mat).as_euler("zxz", degrees=True)


def in_box_bounds(
    coords: np.ndarray,
    box_dims: TripletLike,
    boundary: int = 0,
) -> np.ndarray:
    """Check whether each integer-valued coordinate sits inside a 3-D box.

    A coordinate ``c`` is in-bounds when ``c - boundary >= 0`` and
    ``c + boundary < box_dims[axis]`` for every axis. With ``boundary == 0``
    this is the standard "center is inside the box" check; with
    ``boundary == ceil(box_size / 2)`` it is the "whole subtomogram fits in
    the box" check.

    Parameters
    ----------
    coords : numpy.ndarray
        Shape ``(N, 3)``. Coordinates to test.
    box_dims : TripletLike
        ``(x, y, z)`` box dimensions in voxels. Scalar broadcasts to all three.
    boundary : int, default=0
        Per-axis padding around each coordinate to require inside the box.

    Returns
    -------
    numpy.ndarray
        Boolean mask of shape ``(N,)``.
    """
    dims = as_triplet(box_dims)
    coords = np.asarray(coords)
    coords_min = coords - boundary
    coords_max = coords + boundary
    return (
        (coords_min >= 0).all(axis=1)
        & (coords_max[:, 0] < dims[0])
        & (coords_max[:, 1] < dims[1])
        & (coords_max[:, 2] < dims[2])
    )


def angle_between_vectors(vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
    """Compute the angle (in degrees) between corresponding pairs of vectors in two arrays.

    Parameters
    ----------
    vectors1 : ndarray (n, d)
        Each row represents a d-dimensional vector.
    vectors2 : ndarray (n, d)
        Each row represents a d-dimensional vector.

    Returns
    -------
    ndarray (n,)
        Array containing the angles (in degrees) between corresponding vectors.

    Examples
    --------
    >>> vectors1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> vectors2 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    >>> angle_between_vectors(vectors1, vectors2)
    array([90., 90., 90.])
    """
    dot_products = np.einsum("ij,ij->i", vectors1, vectors2)
    norms1 = np.linalg.norm(vectors1, axis=1)
    norms2 = np.linalg.norm(vectors2, axis=1)

    cosines = dot_products / (norms1 * norms2)
    radians = np.arccos(np.clip(cosines, -1.0, 1.0))
    degrees = np.degrees(radians)

    return degrees


def fill_ellipsoid(
    box_size: TripletLike,
    ellipsoid_parameters: ArrayLike,
) -> np.ndarray:
    """Fills a 3D space defined by `box_size` with a boolean mask where an ellipsoid defined by `ellipsoid_parameters`
    is located.

    Parameters
    ----------
    box_size : TripletLike
        Size of the box in which the ellipsoid will be placed (single int broadcast
        to all three axes, or a 3-element array-like). Normalized via
        :func:`as_triplet`.
    ellipsoid_parameters : array_like
        Coefficients for the general ellipsoid equation:
        Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz + J = 0
        Should contain ten elements corresponding to A, B, C, D, E, F, G, H, I, J respectively.

    Returns
    -------
    numpy.ndarray
        A 3D boolean array where True values represent the points inside or on the surface of the ellipsoid.

    Examples
    --------
    >>> box_size = 10
    >>> ellipsoid_parameters = (1, 1, 1, 0, 0, 0, 0, 0, 0, -100)
    >>> mask = fill_ellipsoid(box_size, ellipsoid_parameters)
    >>> mask.shape
    (10, 10, 10)
    """

    # Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz + J = 0

    box_size = as_triplet(box_size)

    x_array = np.arange(0, box_size[0], 1)
    y_array = np.arange(0, box_size[1], 1)
    z_array = np.arange(0, box_size[2], 1)
    x, y, z = np.meshgrid(x_array, y_array, z_array, indexing="ij")

    A, B, C, D, E, F, G, H, I, J = ellipsoid_parameters
    vals = (
        A * x * x
        + B * y * y
        + C * z * z
        + 2 * D * x * y
        + 2 * E * x * z
        + 2 * F * y * z
        + 2 * G * x
        + 2 * H * y
        + 2 * I * z
        + J
    )

    mask = vals >= 0

    return mask


def fit_ellipsoid(coord: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit an ellipsoid to a set of 3D coordinates. It is based on
    http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit

    Parameters
    ----------
    coord : ndarray
        An array of shape (N, 3) where each row represents the x, y, z coordinates.

    Returns
    -------
    center : ndarray
        The center of the ellipsoid (x, y, z coordinates).
    radii : ndarray
        Radii of the ellipsoid along the principal axes.
    evecs : ndarray
        The eigenvectors corresponding to the principal axes of the ellipsoid.
    v : ndarray
        The 1D array of the ellipsoid parameters used to form the quadratic form.

    Notes
    -----
    This function fits an ellipsoid to a set of points by solving a linear least squares problem to estimate the
    parameters of the ellipsoid's equation in its algebraic form.

    Examples
    --------
    >>> points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> center, radii, evecs, _ = fit_ellipsoid(points)
    >>> center
    array([x_center, y_center, z_center])
    >>> radii
    array([radius_x, radius_y, radius_z])
    >>> evecs
    array([[evec1_x, evec1_y, evec1_z],
           [evec2_x, evec2_y, evec2_z],
           [evec3_x, evec3_y, evec3_z]])
    """

    x = coord[:, 0]
    y = coord[:, 1]
    z = coord[:, 2]
    D = np.array(
        [
            x * x + y * y - 2 * z * z,
            x * x + z * z - 2 * y * y,
            2 * x * y,
            2 * x * z,
            2 * y * z,
            2 * x,
            2 * y,
            2 * z,
            1 - 0 * x,
        ]
    )
    d2 = np.array(x * x + y * y + z * z).T  # rhs for LLSQ
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array(
        [
            [v[0], v[3], v[4], v[6]],
            [v[3], v[1], v[5], v[7]],
            [v[4], v[5], v[2], v[8]],
            [v[6], v[7], v[8], v[9]],
        ]
    )

    center = np.linalg.solve(-A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1.0 / np.abs(evals))
    radii *= np.sign(evals)

    return center, radii, evecs, v


def point_ellipsoid_distance(
    p: ArrayLike,
    params: ArrayLike,
) -> float:
    """Computes the shortest distance from a point p to the surface of an ellipsoid.

    Parameters:
    -----------
    p : ArrayLike
        The 3D point in space. Normalized via :func:`numpy.asarray`.
    params : ArrayLike
        The ellipsoid parameters in the following order:
        ["cx", "cy", "cz", "rx", "ry", "rz",
         "ev1x", "ev1y", "ev1z", "ev2x", "ev2y", "ev2z",
         "ev3x", "ev3y", "ev3z", "p1", ..., "p10"]

    Returns:
    --------
    float
        The shortest distance from the point to the ellipsoid surface.
    """
    p = np.asarray(p)
    params = np.asarray(params)

    # Extract ellipsoid parameters
    center = np.array(params[:3])  # (cx, cy, cz)
    radii = np.array(params[3:6])  # (rx, ry, rz)
    evecs = np.array(params[6:15]).reshape(3, 3)  # 3x3 eigenvector matrix

    # Transform point to local ellipsoid coordinates
    p_local = np.dot(evecs.T, (p - center))

    # Function to solve for lambda (scaling factor)
    def scale_equation(lmbda):
        scaled = p_local / (1 + lmbda)
        return np.sum((scaled / radii) ** 2) - 1

    # Solve for Î» numerically
    lambda_solution = fsolve(scale_equation, 0)[0]

    # Compute closest point on ellipsoid in local space
    closest_local = p_local / (1 + lambda_solution)

    # Transform back to global coordinates
    closest_global = np.dot(evecs, closest_local) + center

    # Compute Euclidean distance from point to closest surface point
    return np.linalg.norm(p - closest_global)


def point_pairwise_dist(coord_1: np.ndarray, coord_2: np.ndarray) -> np.ndarray:
    """Calculate the pairwise Euclidean distance between two sets of coordinates.

    Parameters
    ----------
    coord_1 : ndarray
        An array of shape (N, D) where N is the number of points and D is the dimensionality of each point.
        If N=1, the single point is broadcasted to match the number of points in coord_2.
    coord_2 : ndarray
        An array of shape (M, D) where M is the number of points and D is the dimensionality of each point. If coord_1
        has N>1, then M has to be equal to N.

    Returns
    -------
    pairwise_dist : ndarray
        An array of shape (max(N, M),) containing the Euclidean distances between each pair of points from coord_1
        and coord_2.

    Notes
    -----
    If the input arrays have complex numbers, the distance calculation defaults to 0.0 for those pairs.
    """

    if coord_1.shape[0] == 1 and coord_2.shape[0] != 1:
        coord_1 = np.tile(coord_1, (coord_2.shape[0], 1))

    coord_1 = np.atleast_2d(coord_1)
    coord_2 = np.atleast_2d(coord_2)
    # Squares of the distances
    pairwise_dist = np.linalg.norm(coord_1 - coord_2, axis=1)

    pairwise_dist = np.where(isinstance(pairwise_dist, complex), 0.0, pairwise_dist)

    return pairwise_dist


def area_triangle(coords: np.ndarray) -> float:
    """Calculate the area of a triangle given its vertex coordinates. See
    https://stackoverflow.com/questions/71346322/numpy-area-of-triangle-and-equation-of-a-plane-on-which-triangle-lies-on

    Parameters
    ----------
    coords : ndarray
        An array of shape (3, 3) where each row represents a vertex of the triangle, and each vertex is given by three
        coordinates (x, y, z).

    Returns
    -------
    float
        The area of the triangle.

    Examples
    --------
    >>> coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> area_triangle(coords)
    0.5
    """

    # The cross product of two sides is a normal vector
    triangles = np.cross(coords[:, 1] - coords[:, 0], coords[:, 2] - coords[:, 0])
    # The norm of the cross product of two sides is twice the area
    return np.linalg.norm(triangles) / 2


def ray_ellipsoid_intersection_3d(
    point: ArrayLike,
    normal: ArrayLike,
    ellipsoid_params: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
    """Compute the intersection between a ray starting at point in direction of normal and
    an ellipsoid specified by ellipsoid_params.

    Parameters
    ----------
    point : ArrayLike
        Point in 3D describing origin of ray. Normalized via :func:`numpy.asarray`.
    normal : ArrayLike
        Normal vector describing direction of ray. Normalized via :func:`numpy.asarray`.
    ellipsoid_params : ArrayLike
        Coefficients describing quadratic form of ellipsoid (10 elements).

    Returns
    -------
    tuple (p1, p2, d1, d2, is_inside) with:
        p1: ndarray describing closest intersection, or NaN.
        p2: ndarray describing intersection, or NaN.
        d1: float (distance between point and p1), or NaN.
        d2: float (distance between point and p2), or NaN.
        is_inside: bool, true if point lies inside the ellipsoid.
    """
    point = np.asarray(point)
    normal = np.asarray(normal)

    # Extract line parameters
    x, y, z = point[0], point[1], point[2]
    n1, n2, n3 = normal[0], normal[1], normal[2]

    # Extract ellipsoid parameters
    a, b, c, d, e, f, g, h, i, j = ellipsoid_params

    # Calculate coefficients A, B, C for the quadratic equation
    A = a * n1**2 + b * n2**2 + c * n3**2 + 2 * d * n1 * n2 + 2 * e * n1 * n3 + 2 * f * n3 * n2
    B = 2 * (
        a * x * n1
        + b * y * n2
        + c * z * n3
        + d * x * n2
        + d * y * n1
        + e * x * n3
        + e * z * n1
        + f * z * n2
        + f * y * n3
        + g * n1
        + h * n2
        + i * n3
    )
    C = j + a * x**2 + b * y**2 + c * z**2 + 2 * (i * z + h * y + g * x + d * x * y + e * x * z + f * z * y)

    # Discriminant
    D = B**2 - 4 * A * C

    # Initialize results
    p1, p2 = None, None
    is_inside = False

    if D < 0:  # No intersection
        p1, p2 = np.nan, np.nan
        d1, d2 = np.nan, np.nan
    elif D == 0:  # One intersection point
        t1 = -B / (2 * A)
        p1 = point + t1 * normal
        d1 = np.sign(t1) * np.linalg.norm(p1 - point)  # assigning the correct sign
        p2 = np.nan
        d2 = np.nan
    else:  # Two intersection points
        t1 = (-B + np.sqrt(D)) / (2 * A)
        t2 = (-B - np.sqrt(D)) / (2 * A)

        p1 = point + t1 * normal
        p2 = point + t2 * normal

        d1 = np.sign(t1) * np.linalg.norm(p1 - point)  # assigning the correct sign
        d2 = np.sign(t2) * np.linalg.norm(p2 - point)  # assigning the correct sign

        ps = np.column_stack((p1, p2))
        distances = np.asarray([d1, d2])

        if d1 < 0 and d2 > 0:
            is_inside = True
            pi = 1
        elif d1 > 0 and d2 < 0:
            is_inside = True
            pi = 0
        elif d1 < 0 and d2 < 0:
            pi = np.argmin(abs(distances))
        else:
            pi = np.argmin(distances)

        p1 = ps[:, pi]
        p2 = ps[:, 1 - pi]
        d1 = distances[pi]
        d2 = distances[1 - pi]

    return p1, p2, d1, d2, is_inside


def construct_rays(
    points: np.ndarray,
    normals: np.ndarray,
    ray_length: float | None = None,
    reverse_direction: bool = False,
) -> np.ndarray:
    """
    Build rays from points and normals (N, 6): origin xyz + direction xyz.

    Parameters
    ----------
    points : ndarray (N, 3)
    normals : ndarray (N, 3)
    ray_length : float, optional
        Scale for direction magnitude; if None, effectively infinite.
    reverse_direction : bool
        If True, negate normals before building directions.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float32))
    normals = np.atleast_2d(np.asarray(normals, dtype=np.float32))

    if points.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {points.shape}")
    if normals.shape[1] != 3:
        raise ValueError(f"Normals must have shape (N, 3), got {normals.shape}")
    if points.shape[0] != normals.shape[0]:
        raise ValueError(
            f"Points and normals length mismatch: {points.shape[0]} vs {normals.shape[0]}"
        )

    norm_mag = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(norm_mag < 1e-10):
        raise ValueError("Found zero-length normal vectors")
    normals_normalized = normals / norm_mag

    if reverse_direction:
        normals_normalized = -normals_normalized

    if ray_length is None:
        ray_directions = normals_normalized * 1e10
    else:
        ray_directions = normals_normalized * float(ray_length)

    return np.hstack([points, ray_directions])


def ray_ray_intersection_3d(starting_points, ending_points):
    """Calculate the intersection point and distances from the intersection to each line for a set of 3D rays.

    Parameters
    ----------
    starting_points : ndarray
        An array of shape (N, 3) representing the starting points of N lines in 3D space.
    ending_points : ndarray
        An array of shape (N, 3) representing the ending points of N lines in 3D space.

    Returns
    -------
    P_intersect : ndarray
        A 1D array of shape (3,) containing the coordinates of the intersection point.
    distances : ndarray
        A 1D array of shape (N,) containing the distances from the intersection point to each line.

    Notes
    -----
    This function assumes that all lines are somewhat close to intersecting at a common point and uses a least squares
    approach to find the best intersection point. The function may not be suitable for parallel lines or lines that
    do not converge.
    """

    # N lines described as vectors
    Si = ending_points - starting_points

    # Normalize vectors
    ni = Si / np.linalg.norm(Si, axis=1)[:, np.newaxis]

    nx, ny, nz = ni[:, 0], ni[:, 1], ni[:, 2]

    # Calculate sums
    SXX = np.sum(nx**2 - 1)
    SYY = np.sum(ny**2 - 1)
    SZZ = np.sum(nz**2 - 1)
    SXY = np.sum(nx * ny)
    SXZ = np.sum(nx * nz)
    SYZ = np.sum(ny * nz)

    # Matrix S
    S = np.array([[SXX, SXY, SXZ], [SXY, SYY, SYZ], [SXZ, SYZ, SZZ]])

    CX = np.sum(
        starting_points[:, 0] * (nx**2 - 1) + starting_points[:, 1] * (nx * ny) + starting_points[:, 2] * (nx * nz)
    )
    CY = np.sum(
        starting_points[:, 0] * (nx * ny) + starting_points[:, 1] * (ny**2 - 1) + starting_points[:, 2] * (ny * nz)
    )
    CZ = np.sum(
        starting_points[:, 0] * (nx * nz) + starting_points[:, 1] * (ny * nz) + starting_points[:, 2] * (nz**2 - 1)
    )

    C = np.array([CX, CY, CZ])

    # Solve the system of linear equations
    P_intersect = np.linalg.solve(S, C)

    distances = np.zeros(starting_points.shape[0])

    for i in range(starting_points.shape[0]):
        ui = np.dot(P_intersect - starting_points[i, :], Si[i, :]) / np.dot(Si[i, :], Si[i, :])
        distances[i] = np.linalg.norm(P_intersect - starting_points[i, :] - ui * Si[i, :])

    return P_intersect, distances


def rotate_points_rodrigues(
    P: np.ndarray,
    n0: ArrayLike,
    n1: ArrayLike,
) -> np.ndarray:
    """Rotates points by the rotation defined by two vectors.

    Parameters
    ----------
    P : ndarray
        Array containing point(s) to be rotated. Can be a 1D array for a single point or a 2D array for multiple points.
    n0 : ArrayLike
        Initial vector, before rotation. 3-element array-like; normalized via
        :func:`numpy.asarray`.
    n1 : ArrayLike
        Final vector, after rotation. 3-element array-like; normalized via
        :func:`numpy.asarray`.

    Returns
    -------
    P_rot : ndarray
        Array of rotated points. Same shape as input array P.

    Notes
    -----
    This function computes the rotation matrix that rotates vector n0 to align with vector n1 and applies this rotation
    to point(s) P. The rotation is performed using the Rodrigues' rotation formula, facilitated by scipy's spatial
    transformations.

    Examples
    --------
    >>> P = np.array([1, 0, 0])
    >>> n0 = np.array([1, 0, 0])
    >>> n1 = np.array([0, 1, 0])
    >>> rotate_points_rodrigues(P, n0, n1)
    array([[0., 1., 0.]])
    """

    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[np.newaxis, :]

    n0 = np.asarray(n0)
    n1 = np.asarray(n1)

    # Normalize vectors n0 and n1
    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)

    # Calculate the axis of rotation (k) and angle (theta)
    k = np.cross(n0, n1)
    k = k / np.linalg.norm(k)
    theta = np.arccos(np.clip(np.dot(n0, n1), -1.0, 1.0))  # clip to avoid floating point errors

    # Create the scipy rotation object
    r = srot.from_rotvec(theta * k)

    # Apply rotation to each point in P
    P_rot = r.apply(P)

    return P_rot


def project_3d_points_on_2d_plane_normal_aligned(
    coord: np.ndarray,
    target_direction: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projects 3D points onto a 2D plane that is aligned with a specified normal direction.

    Parameters
    ----------
    coord : ndarray
        An array of shape (N, 3) containing N points in 3D space.
    target_direction : array_like, optional
        A 3-element array specifying the direction to which the normal of the 2D plane should be aligned.
        If None, defaults to [0, 0, 1], aligning the plane with the z-axis. Defaults to None.

    Returns
    -------
    coord_proj : ndarray
        An array of shape (N, 2) containing the 2D coordinates of the projected points.
    coord_mean : ndarray
        A 1D array of length 3 representing the mean of the original coordinates.
    normal : ndarray
        A 1D array of length 3 representing the normal vector of the plane onto which the points are projected.

    Notes
    -----
    The function first centers the points by subtracting their mean. It then uses Singular Value Decomposition (SVD)
    to find the principal components of the points. The smallest singular vector (normal to the plane of best fit)
    is used. The points are then rotated to align this normal with
    the target direction, effectively projecting them onto a new 2D plane.
    """

    coord_mean = coord.mean(axis=0)
    coord_centered = coord - coord_mean
    _, _, V = np.linalg.svd(coord_centered)
    normal = V[2, :]

    if target_direction is None:
        target_direction = [0, 0, 1]
    coord_proj = rotate_points_rodrigues(coord_centered, normal, target_direction)

    return coord_proj, coord_mean, normal


def project_3d_points_on_2d_plane_variance_based(coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projects 3D points onto a 2D plane using variance-based method via Singular Value Decomposition (SVD).

    Parameters
    ----------
    coord : ndarray
        An array of shape (N, 3) where n is the number of 3D points.

    Returns
    -------
    coord_proj : ndarray
        The projected coordinates of the points onto the 2D plane, of shape (N, 2).
    U : ndarray
        The matrix containing the left singular vectors of the decomposition, used to project the points.

    Notes
    -----
    The function performs dimensionality reduction by projecting the original 3D points onto the 2D plane that captures
    the most variance in the data. This is achieved using SVD, which decomposes the input matrix into its singular
    vectors and singular values.

    Examples
    --------
    >>> points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> projected_points, _ = project_3d_points_on_2d_plane_variance_based(points)
    >>> print(projected_points)
    """

    coord = coord.T
    U, _, _ = np.linalg.svd(coord, full_matrices=False)
    coord_proj = np.dot(U.T, coord)

    return coord_proj, U


def fit_circle_3d_lsq(coord: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit a circle to 3D points using least squares optimization.

    Parameters
    ----------
    coord : ndarray
        An array of shape (N, 3) containing the 3D coordinates of the points.

    Returns
    -------
    circle_center : ndarray
        A 1D array of length 3 containing the x, y, z coordinates of the fitted circle's center.
    circle_radius : float
        The radius of the fitted circle.
    residual_error : float
        Sum of the squared residuals of the fit.

    Notes
    -----
    This function projects 3D points onto a 2D plane that is normal to their average normal vector. It then fits a
    circle in 2D and transforms the center back to the 3D space.
    """

    coord_xy, coord_mean, normal = project_3d_points_on_2d_plane_normal_aligned(coord)
    xc, yc, circle_radius, residual_error = fit_circle_2d_lsq(coord_xy[:, 0], coord_xy[:, 1])

    circle_center = rotate_points_rodrigues(np.array([xc, yc, 0]), [0, 0, 1], normal) + coord_mean
    circle_center = circle_center.flatten()

    return circle_center, circle_radius, residual_error


def fit_circle_2d_lsq(
    x: ArrayLike,
    y: ArrayLike,
    w: ArrayLike | None = None,
) -> tuple[float, float, float, float]:
    """Fit a circle to 2D points using the least squares method. The method was taken from
    https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/

    Parameters
    ----------
    x : array_like
        X-coordinates of the points.
    y : array_like
        Y-coordinates of the points.
    w : array_like, optional
        Weights for each point. If provided, must be the same length as `x` and `y`. Defaults to None.

    Returns
    -------
    xc : float
        X-coordinate of the fitted circle's center.
    yc : float
        Y-coordinate of the fitted circle's center.
    r : float
        Radius of the fitted circle.
    error : float
        Sum of the squared residuals of the fit.

    Notes
    -----
    This function fits a circle in 2D to a set of points (x, y) by solving the
    weighted least squares problem if weights `w` are provided. If no weights are
    provided, it solves the ordinary least squares problem.

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([4, 5, 6])
    >>> xc, yc, r, error = fit_circle_2d_lsq(x, y)
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if w is None:
        w = []
    else:
        w = np.asarray(w)

    A = np.array([x, y, np.ones(len(x))]).T
    b = x**2 + y**2

    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W, A)
        b = np.dot(W, b)

    # Solve by method of least squares
    # c = np.linalg.lstsq(A,b,rcond=None)[0]
    c, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Get circle parameters from solution c
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc**2 + yc**2)

    # Calculate least squares error (sum of squared residuals)
    error = np.sum(residuals)

    return xc, yc, r, error


def fit_circle_3d_pratt(coord: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Fit a circle to a set of 3D points using Pratt's method after projecting them onto a 2D plane.

    Parameters
    ----------
    coord : ndarray
        An array of shape (N, 3) containing N points in 3D space.

    Returns
    -------
    circle_center : ndarray
        A 1D array of length 3 representing the center of the circle in 3D space.
    circle_radius : float
        The radius of the fitted circle.
    confidence : int
        Confidence indicator, returns 1 if the center is not at the origin, otherwise -1.

    Notes
    -----
    The function first projects the 3D points onto a 2D plane using a variance-based method. It then applies
    Pratt's method to these 2D points to fit a circle. The center of the circle is then transformed back to
    3D space. The radius is calculated as the mean Euclidean distance from the 3D points to the estimated
    center. The confidence is a simple check on the location of the center.

    Examples
    --------
    >>> points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> center, radius, conf = fit_circle_3d_pratt(points)
    >>> print(center, radius, conf)
    """

    # Projection of coordinates on 2D plane
    coord_proj, U = project_3d_points_on_2d_plane_variance_based(coord)

    # Pratt method
    x = coord_proj[0, :]
    y = coord_proj[1, :]
    a = np.linalg.lstsq(np.vstack([x, y, np.ones_like(x)]).T, -(x**2 + y**2), rcond=None)[0]
    xc = -a[0]
    yc = -a[1]

    center_2d = np.array([xc, yc])

    # 3d center
    c_si = np.concatenate([center_2d, [0]])
    circle_center = np.matmul(U, c_si) / 2.0

    # Compute a radius
    coord = coord.T
    center_coord = np.tile(circle_center, (coord.shape[0], 1))
    euc_dist = np.sqrt(np.sum((coord - center_coord) ** 2, axis=0))
    circle_radius = np.mean(euc_dist)

    # confidence
    if np.all(center_coord == 0):
        confidence = -1
    else:
        confidence = 1

    return circle_center, circle_radius, confidence


def fit_circle_3d_taubin(coord: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Fit a circle to 3D points using Taubin's method projected onto a 2D plane.

    Parameters
    ----------
    coord : ndarray
        An array of shape (N, 3) representing the coordinates of the 3D points.

    Returns
    -------
    circle_center : ndarray
        A 1D array of length 3 representing the center of the fitted circle in 3D space.
    circle_radius : float
        The radius of the fitted circle.
    confidence : float
        A confidence measure for the circle fitting. Returns -1 if the fitting fails.

    Notes
    -----
    The function projects 3D points onto a 2D plane using a variance-based method,
    then fits a circle in 2D using Newton's method. The best fitting circle is selected
    based on the minimum radius criterion among possible circle fits.

    Examples
    --------
    >>> coord = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> center, radius, conf = fit_circle_3d_taubin(coord)
    >>> print(center, radius, conf)
    """

    # Projection of coordinates on 2D plane
    coord_proj, U = project_3d_points_on_2d_plane_variance_based(coord)

    a = np.array([[0, 1], [0, 2], [1, 2]])
    cs = []
    cr = []

    for b in range(a.shape[0]):
        cp = coord_proj[a[b, :], :]
        center_2d, rr, confidence = fit_circle_2d_newton(cp)

        # Check if the fit was successful
        if center_2d[0] == np.inf:
            circle_center = np.inf
            circle_radius = np.inf
            confidence = -1
            return circle_center, circle_radius, confidence

        # Estimated center
        cc = np.zeros(3)
        cc[a[b, 0]] = center_2d[0]
        cc[a[b, 1]] = center_2d[1]
        circle_center = np.matmul(U, cc)

        cs.append(circle_center)
        cr.append(rr)

    circle_radius, c_idx = min(zip(cr, range(len(cr))))

    circle_center = cs[c_idx]

    return circle_center, circle_radius, confidence


def fit_circle_2d_newton(coord: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Fit a circle to 2D data using Newton's method.

    Parameters
    ----------
    coord : ndarray
        An array of shape (N, 2) containing the x and y coordinates of the data points.

    Returns
    -------
    circle_center : ndarray
        A 1D array containing the x and y coordinates of the circle's center.
    circle_radius : float
        The radius of the fitted circle.
    confidence : int
        Confidence level of the fit, 1 if successful, -1 if the fit failed.

    Notes
    -----
    The function implements a numerical method to fit a circle to a set of 2D points by minimizing the algebraic
    distance to the circle. The method used is based on the algebraic form of a circle and Newton's optimization
    method to find the circle parameters that best fit the data.

    Examples
    --------
    >>> points = np.array([[1, 2], [3, 4], [5, 6]])
    >>> center, radius, conf = fit_circle_2d_newton(points)
    >>> print(center, radius, conf)
    """

    XY = coord.T  # Transpose to match MATLAB's column-wise data format

    n = XY.shape[0]  # number of data points
    centroid = np.mean(XY, axis=0)  # the centroid of the data set

    # computing moments (note: all moments will be normed, i.e. divided by n)
    Mxx = 0
    Myy = 0
    Mxy = 0
    Mxz = 0
    Myz = 0
    Mzz = 0

    for i in range(n):
        Xi = XY[i, 0] - centroid[0]  # centering data
        Yi = XY[i, 1] - centroid[1]  # centering data
        Zi = Xi * Xi + Yi * Yi
        Mxy = Mxy + Xi * Yi
        Mxx = Mxx + Xi * Xi
        Myy = Myy + Yi * Yi
        Mxz = Mxz + Xi * Zi
        Myz = Myz + Yi * Zi
        Mzz = Mzz + Zi * Zi

    Mxx = Mxx / n
    Myy = Myy / n
    Mxy = Mxy / n
    Mxz = Mxz / n
    Myz = Myz / n
    Mzz = Mzz / n

    # computing the coefficients of the characteristic polynomial
    Mz = Mxx + Myy
    Cov_xy = Mxx * Myy - Mxy * Mxy
    A3 = 4 * Mz
    A2 = -3 * Mz * Mz - Mzz
    A1 = Mzz * Mz + 4 * Cov_xy * Mz - Mxz * Mxz - Myz * Myz - Mz * Mz * Mz
    A0 = Mxz * Mxz * Myy + Myz * Myz * Mxx - Mzz * Cov_xy - 2 * Mxz * Myz * Mxy + Mz * Mz * Cov_xy
    A22 = A2 + A2
    A33 = A3 + A3 + A3
    xnew = 0
    ynew = 1e20
    epsilon = 1e-12
    IterMax = 20

    # set default to 1
    confidence = 1

    # Newton's method starting at x=0
    for _ in range(IterMax):
        yold = ynew
        ynew = A0 + xnew * (A1 + xnew * (A2 + xnew * A3))
        if abs(ynew) > abs(yold):
            xnew = 0
            confidence = -1
            break
        Dy = A1 + xnew * (A22 + xnew * A33)
        xold = xnew
        xnew = xold - ynew / Dy
        if abs((xnew - xold) / xnew) < epsilon:
            break
        if _ >= IterMax:
            xnew = 0
            confidence = -1
        if xnew < 0:
            xnew = 0
            confidence = -1

    # computing the circle parameters
    DET = xnew * xnew - xnew * Mz + Cov_xy
    Center = np.array([(Mxz * (Myy - xnew) - Myz * Mxy), (Myz * (Mxx - xnew) - Mxz * Mxy)]) / DET / 2
    Par = np.concatenate([Center + centroid, [np.sqrt(np.dot(Center, Center) + Mz)]])
    circle_center = Par[:2]
    circle_radius = Par[2]

    return circle_center, circle_radius, confidence


def normalize_vector(vector: ArrayLike) -> np.ndarray:
    """Normalize a vector.

    Parameters
    ----------
    vector : ArrayLike
        Input vector to be normalized. Normalized via :func:`numpy.asarray`.

    Returns
    -------
    ndarray
        Normalized vector with the same direction but with a norm of 1.

    Examples
    --------
    >>> import numpy as np
    >>> v = np.array([2, 3, 6])
    >>> normalize_vector(v)
    array([0.26726124, 0.40089186, 0.80278373])
    """

    vector = np.asarray(vector)
    return vector / np.linalg.norm(vector)


def normalize_vectors(v: np.ndarray) -> np.ndarray:
    """Normalize each vector, handling both single vectors and arrays of vectors.

    Parameters
    ----------
    v : ndarray (n,d)

    Returns
    -------
    ndarray (n,d)
        Array of normalized vectors.

    Examples
    --------
    >>> import numpy as np
    >>> v = np.array([[1, 2, 3], [4, 5, 6]])
    >>> normalize_vectors(v)
    array([[0.26726124, 0.53452248, 0.80178373],^
              [0.45584231, 0.56980288, 0.68376346]])
    """
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / norm


# TODO: This function should not be necessary due to "angle_between_vectors"
def angle_between_n_vectors(
    v1: np.ndarray,
    v2: np.ndarray,
    degrees: bool = True,
) -> np.ndarray:
    """Compute the angle (in degrees) between corresponding pairs of vectors in two arrays.

    Parameters
    ----------
    v1 : ndarray (n, d)
        Each row represents d-dimensional vector.
    v2 : ndarray (n, d)
        Each row represents d-dimensional vector.

    Returns
    -------
    ndarray (n,)
        Array containing the angles (in degrees) between corresponding vectors.

    Examples
    --------
    >>> import numpy as np
    >>> v1 = np.array([[1, 0, 0], [0, 1, 0]])
    >>> v2 = np.array([[0, 1, 0], [1, 0, 0]])
    >>> angle_between_n_vectors(v1, v2)
    array([90., 90.])
    """

    # Ensure both vectors are normalized
    v1_u = normalize_vectors(v1)
    v2_u = normalize_vectors(v2)

    # Compute dot product element-wise, handling both (3,) and (N, 3) shapes
    dot_product = np.einsum("ij,ij->i", v1_u, v2_u) if v1.ndim > 1 else np.dot(v1_u, v2_u)

    # Clip values to avoid numerical errors outside the valid range for arccos
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    if degrees:
        return np.rad2deg(angle)
    else:
        return angle


def vector_angular_distance(
    v1: ArrayLike,
    v2: ArrayLike,
) -> float:
    """Calculate the angular distance between two vectors in degrees.

    Parameters
    ----------
    v1 : ArrayLike
        First input vector. Normalized via :func:`normalize_vector`.
    v2 : ArrayLike
        Second input vector. Normalized via :func:`normalize_vector`.

    Returns
    -------
    float
        The angular distance between `v1` and `v2` in degrees.

    Examples
    --------
    >>> v1 = [1, 0, 0]
    >>> v2 = [0, 1, 0]
    >>> vector_angular_distance(v1, v2)
    90.0
    """

    v1_u = normalize_vector(v1)
    v2_u = normalize_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def vector_angular_distance_signed(
    u: ArrayLike,
    v: ArrayLike,
    n: ArrayLike | None = None,
) -> float:
    """Compute the signed angular distance between two vectors.

    Parameters
    ----------
    u : ArrayLike
        First input vector.
    v : ArrayLike
        Second input vector.
    n : ArrayLike, optional
        Normal vector to the plane containing `u` and `v`. If not provided, the function
        computes the unsigned angular distance.

    Returns
    -------
    float
        The signed angular distance between vectors `u` and `v`. This is the angle in radians
        between the two vectors, signed according to the direction given by `n`. If `n` is not
        provided, the result is the unsigned angle.

    Notes
    -----
    The signed angle is computed based on the right-hand rule. If `n` is provided, the sign
    of the angle is determined by the direction of `n` relative to the cross product of `u` and `v`.
    If `n` is not provided, the function returns the magnitude of the angle only.

    Examples
    --------
    >>> u = np.array([1, 0, 0])
    >>> v = np.array([0, 1, 0])
    >>> vector_angular_distance_signed(u, v)
    1.5707963267948966  # 90 degrees in radians

    >>> n = np.array([0, 0, 1])
    >>> vector_angular_distance_signed(u, v, n)
    1.5707963267948966  # 90 degrees in radians, positive as per right-hand rule with `n` as z-axis

    >>> n = np.array([0, 0, -1])
    >>> vector_angular_distance_signed(u, v, n)
    -1.5707963267948966  # 90 degrees in radians, negative as per right-hand rule with `n` as -z-axis
    """

    if n is None:
        return np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))
    else:
        return np.arctan2(np.dot(n, np.cross(u, v)), np.dot(u, v))


def oversample_spline(coords: np.ndarray, target_spacing: float) -> np.ndarray:
    """Fit a spline through 3D coordinates and oversample so that the distance between points is approximately `target_spacing`.

    Parameters
    ----------
    coords : ndarray
        Array of shape (n, 3) representing the input points.
    target_spacing : float
        Desired distance between points on the spline.

    Returns
    -------
    ndarray
        Oversampled coordinates along the spline.
    """
    # Fit a parametric spline to the data
    tck, u = splprep(coords.T, s=0)  # `s=0` ensures an exact fit to the input points

    # Compute cumulative arc length
    distances = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
    total_length = np.sum(distances)

    # Determine number of samples based on target spacing
    num_samples = int(total_length / target_spacing) + 1

    # Generate evenly spaced parameter values along the spline
    u_fine = np.linspace(0, 1, num_samples)

    # Evaluate the spline to get oversampled points
    oversampled_points = np.array(splev(u_fine, tck)).T

    return oversampled_points


def distance_array(vol: np.ndarray) -> np.ndarray:
    """Build a cubic grid of Euclidean distances from the centre.

    Parameters
    ----------
    vol : numpy.ndarray
        Reference 3D volume whose first-axis length determines the cubic edge.

    Returns
    -------
    numpy.ndarray
        3D array of distances from the centre voxel.
    """
    shell_grid = np.arange(math.floor(-len(vol[0]) / 2), math.ceil(len(vol[0]) / 2), 1)
    xv, yv, zv = shell_grid, shell_grid, shell_grid
    shell_space = np.meshgrid(xv, yv, zv, indexing="xy")  ## 'ij' denominates matrix indexing, 'xy' cartesian
    distance_v = np.sqrt(shell_space[0] ** 2 + shell_space[1] ** 2 + shell_space[2] ** 2)

    return distance_v


def order_points_on_circle(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order points on a circle based on their angles with respect to the x-axis.

    Parameters
    ----------
    points : numpy.ndarray
        A 2D array of shape (n, 3) where each row represents a point in 3D space (x, y, z).

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (n, 3) containing the input points ordered by their angles
        in the xy-plane, starting from the positive x-axis and moving counterclockwise.
    numpy.ndarray
        A 1D array of shape (n,) containing the order of points.

    Notes
    -----
    This function assumes that the input points are approximately in the xy-plane,
    and it discards the z-coordinate for the ordering process. The points are normalized
    to lie on the unit circle before calculating their angles.
    """

    # Project points onto a plane (optional if already planar)
    # Assuming points lie approximately in the xy-plane, we discard the z-coordinate.
    planar_points = points[:, :2]

    # Normalize points to unit circle
    magnitudes = np.linalg.norm(planar_points, axis=1, keepdims=True)
    normalized_points = planar_points / magnitudes

    # Compute angles with respect to the x-axis
    angles = np.arctan2(normalized_points[:, 1], normalized_points[:, 0])

    # Sort points by angle
    sorted_indices = np.argsort(angles)
    ordered_points = points[sorted_indices]

    return ordered_points, sorted_indices


def cartesian_to_spherical(
    coord: np.ndarray,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    coord : numpy.ndarray
        A 2D array of shape (N, 3*K) where N is the number of points and K is the number of sets of 3D coordinates. Each set of 3D coordinates corresponds to a point in Cartesian space.

    normalize : bool, optional
        If True, the input Cartesian coordinates will be normalized before conversion. Default is True.

    Returns
    -------
    phi : numpy.ndarray
        A 1D array of azimuthal angles (in radians) of shape (M,) where M is the number of valid points after filtering out NaNs.

    theta : numpy.ndarray
        A 1D array of polar angles (in radians) of shape (M,) where M is the number of valid points after filtering out NaNs.

    Raises
    ------
    ValueError
        If the input `coord` does not have the shape (N, 3*K).

    Notes
    -----
    The azimuthal angle phi is computed as the angle in the x-y plane from the positive x-axis, and the polar angle theta is computed from the positive z-axis. The function filters out any points that result in NaN values during the conversion process.
    """

    if coord.ndim != 2 or coord.shape[1] % 3 != 0:
        raise ValueError("Input must have shape (N, 3*K).")

    N, D = coord.shape
    K = D // 3

    # reshape into (N, K, 3)
    vecs = coord.reshape(N, K, 3)

    if normalize:
        norms = np.linalg.norm(vecs, axis=2, keepdims=True)  # (N, K, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            vecs = vecs / norms

    # spherical conversion
    x = vecs[..., 0]
    y = vecs[..., 1]
    z = vecs[..., 2]

    phi = np.arctan2(y, x)  # (N, K)
    theta = np.arccos(np.clip(z, -1.0, 1.0))  # (N, K)

    # remove rows with NaNs (across any K)
    mask = ~np.isnan(phi).any(axis=1) & ~np.isnan(theta).any(axis=1)
    phi = phi[mask]
    theta = theta[mask]

    return phi, theta


# =============================================================================
# Sphere projections (Lambert / stereographic / equidistant)
# =============================================================================


def _spherical_for_projection(coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute (theta, phi) from (N, 3) Cartesian coordinates, preserving rows.

    Unlike :func:`cartesian_to_spherical`, this helper does not drop rows with
    NaN values. NaNs in the output are replaced with ``0.0`` in place so the
    output shape matches the input row count â required by the projection
    functions that index back into the original coordinate array.

    Parameters
    ----------
    coord : numpy.ndarray
        Shape ``(N, 3)``.

    Returns
    -------
    theta : numpy.ndarray
        Inclination angles, shape ``(N,)``.
    phi : numpy.ndarray
        Azimuthal angles, shape ``(N,)``.
    """
    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
    phi = np.arctan2(y, x)
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    np.nan_to_num(phi, copy=False, nan=0.0)
    np.nan_to_num(theta, copy=False, nan=0.0)
    return theta, phi


def project_lambert(coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lambert azimuthal equal-area projection of unit vectors.

    The ``+z`` pole maps to the origin.

    Parameters
    ----------
    coord : numpy.ndarray
        Shape ``(N, 3)``. Vectors are assumed to lie on the unit sphere.

    Returns
    -------
    projection_polar : numpy.ndarray
        Polar coordinates ``(phi, r)`` of projected points, shape ``(N, 2)``.
    projection_xy : numpy.ndarray
        Cartesian ``(x, y)`` coordinates of projected points, shape ``(N, 2)``.

    Notes
    -----
    Per the `Wikipedia article on the Lambert azimuthal equal-area projection
    <https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection>`_,
    the symbols for inclination and azimuth are inverted relative to the
    standard spherical-coordinate convention. The implementation uses
    ``pi - theta`` so that ``+z`` projects to ``(0, 0)``.
    """
    theta, phi = _spherical_for_projection(coord)

    n = phi.shape[0]
    projection_xy = np.zeros((n, 2))
    projection_polar = np.zeros((n, 2))

    # divide by sqrt(2) so the unit sphere projects onto the unit disk
    with np.errstate(divide="ignore", invalid="ignore"):
        projection_xy[:, 0] = coord[:, 0] * np.sqrt(2.0 / (1.0 + coord[:, 2]))
        projection_xy[:, 1] = coord[:, 1] * np.sqrt(2.0 / (1.0 + coord[:, 2]))
    np.nan_to_num(projection_xy, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    projection_polar[:, 0] = phi
    projection_polar[:, 1] = 2.0 * np.cos((np.pi - theta) / 2.0)

    return projection_polar, projection_xy


def project_stereo(coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stereographic projection of unit vectors.

    The ``+z`` pole maps to the origin; the ``-z`` pole maps to infinity.

    Parameters
    ----------
    coord : numpy.ndarray
        Shape ``(N, 3)``.

    Returns
    -------
    projection_polar : numpy.ndarray
        Polar coordinates ``(phi, r)`` of projected points, shape ``(N, 2)``.
    projection_xy : numpy.ndarray
        Cartesian ``(x, y)`` coordinates of projected points, shape ``(N, 2)``.
    """
    theta, phi = _spherical_for_projection(coord)

    n = phi.shape[0]
    projection_xy = np.zeros((n, 2))
    projection_polar = np.zeros((n, 2))

    with np.errstate(divide="ignore", invalid="ignore"):
        projection_xy[:, 0] = coord[:, 0] / (1.0 - coord[:, 2])
        projection_xy[:, 1] = coord[:, 1] / (1.0 - coord[:, 2])
    np.nan_to_num(projection_xy, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # NOTE: https://en.wikipedia.org/wiki/Stereographic_projection uses a
    # different convention from the spherical-coords page. (pi - theta) so
    # that +z projects to (0, 0).
    projection_polar[:, 0] = phi
    with np.errstate(divide="ignore", invalid="ignore"):
        projection_polar[:, 1] = np.sin(np.pi - theta) / (1.0 - np.cos(np.pi - theta))
    np.nan_to_num(projection_polar, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return projection_polar, projection_xy


def project_equidistant(coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthal equidistant projection of unit vectors.

    Parameters
    ----------
    coord : numpy.ndarray
        Shape ``(N, 3)``.

    Returns
    -------
    projection_polar : numpy.ndarray
        Polar coordinates ``(phi, r)`` of projected points, shape ``(N, 2)``.
    projection_xy : numpy.ndarray
        Cartesian ``(x, y)`` coordinates of projected points, shape ``(N, 2)``.

    Notes
    -----
    See https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection.
    """
    theta, phi = _spherical_for_projection(coord)

    n = phi.shape[0]
    projection_xy = np.zeros((n, 2))
    projection_polar = np.zeros((n, 2))

    projection_xy[:, 0] = (np.pi / 2.0 + phi) * np.sin(theta)
    projection_xy[:, 1] = -(np.pi / 2.0 + phi) * np.cos(theta)

    # polar form: r from (x, y) norm, phi from atan2(y, x)
    r_xy = np.linalg.norm(projection_xy, axis=1)
    phi_xy = np.arctan2(projection_xy[:, 1], projection_xy[:, 0])
    np.nan_to_num(phi_xy, copy=False, nan=0.0)

    projection_polar[:, 0] = phi_xy
    projection_polar[:, 1] = r_xy

    return projection_polar, projection_xy


def project_points_on_sphere(
    coord: np.ndarray,
    projection_type: str = "stereo",
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the requested sphere projection.

    Parameters
    ----------
    coord : numpy.ndarray
        Shape ``(N, 3)``.
    projection_type : {'stereo', 'lambert', 'equidistant'}, default="stereo"
        Projection algorithm.

    Returns
    -------
    projection_polar : numpy.ndarray
        Polar coordinates ``(phi, r)``, shape ``(N, 2)``.
    projection_xy : numpy.ndarray
        Cartesian ``(x, y)`` coordinates, shape ``(N, 2)``.

    Raises
    ------
    ValueError
        If *projection_type* is not one of the supported algorithms.
    """
    if projection_type == "stereo":
        return project_stereo(coord)
    if projection_type == "lambert":
        return project_lambert(coord)
    if projection_type == "equidistant":
        return project_equidistant(coord)
    raise ValueError(
        f"Unknown projection_type {projection_type!r}; "
        "expected one of: 'stereo', 'lambert', 'equidistant'."
    )


def create_projection(
    coord: np.ndarray,
    projection_type: str = "stereo",
    split_into_hemispheres: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project 3-D unit vectors onto a 2-D plane, optionally per hemisphere.

    When *split_into_hemispheres* is ``True``, points with ``z >= 0`` and
    ``z < 0`` are projected separately. The southern hemisphere is mirrored
    (``z *= -1``) before projection so that both hemispheres use the same
    ``+z``-pole projection convention.

    Parameters
    ----------
    coord : numpy.ndarray
        Shape ``(N, 3)``.
    projection_type : {'stereo', 'lambert', 'equidistant'}, default="stereo"
        Projection algorithm.
    split_into_hemispheres : bool, default=True
        Project northern and southern hemispheres separately.

    Returns
    -------
    polar_pos : numpy.ndarray
        Polar coordinates of northern-hemisphere points, shape ``(M, 2)``.
    xy_pos : numpy.ndarray
        Cartesian coordinates of northern-hemisphere points, shape ``(M, 2)``.
    polar_neg : numpy.ndarray
        Polar coordinates of southern-hemisphere points, shape ``(K, 2)``.
        Empty array when *split_into_hemispheres* is ``False``.
    xy_neg : numpy.ndarray
        Cartesian coordinates of southern-hemisphere points, shape ``(K, 2)``.
        Empty array when *split_into_hemispheres* is ``False``.
    """
    if split_into_hemispheres:
        coord_pos = coord[coord[:, 2] >= 0]
        coord_neg = coord[coord[:, 2] < 0].copy()  # copy: we mutate below
        coord_neg[:, 2] *= -1

        polar_pos, xy_pos = project_points_on_sphere(coord_pos, projection_type)
        polar_neg, xy_neg = project_points_on_sphere(coord_neg, projection_type)

        return polar_pos, xy_pos, polar_neg, xy_neg

    polar, xy = project_points_on_sphere(coord, projection_type)
    empty = np.empty((0, 2))
    return polar, xy, empty, empty


# =============================================================================
# Triangle mesh sampling and point-in-triangle test
# =============================================================================


def _triangle_mesh_vertices(
    opp_vertex: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> np.ndarray:
    """Compute the four corners of a parallelogram mesh from three triangle vertices.

    Given the vertex opposite to the shortest edge (``v1``â``v2``) and the
    two endpoints of that edge, the fourth corner is constructed so that the
    four points form a parallelogram.

    Parameters
    ----------
    opp_vertex : numpy.ndarray, shape (3,)
        The triangle vertex opposite to the edge ``v1``â``v2``.
    v1 : numpy.ndarray, shape (3,)
        First endpoint of the reference edge.
    v2 : numpy.ndarray, shape (3,)
        Second endpoint of the reference edge.

    Returns
    -------
    numpy.ndarray, shape (4, 3)
        Four corners ``[v1, fourth_vertex, v2, opp_vertex]`` forming a
        parallelogram in 3-D space.
    """
    fourth_vertex = opp_vertex + (v1 - v2)
    return np.array((v1, fourth_vertex, v2, opp_vertex))


def sample_triangle(vertices: np.ndarray, sampling_distance: float) -> np.ndarray:
    """Sample a triangle into a regular grid of 3-D points.

    The function identifies the shortest edge of the triangle, constructs a
    parallelogram mesh around that edge, and samples both directions at the
    given spacing.  If the shortest edge is already shorter than
    ``sampling_distance`` the original three vertices are returned unchanged.

    Parameters
    ----------
    vertices : numpy.ndarray, shape (3, 3)
        The three triangle vertices; each row is ``[x, y, z]``.
    sampling_distance : float
        Desired spacing between sample points.

    Returns
    -------
    numpy.ndarray, shape (N, 3)
        Grid of sampled 3-D points covering the triangle face, or the
        original three vertices when the triangle is too small to subdivide.
    """
    d01 = np.linalg.norm(vertices[0] - vertices[1])
    d02 = np.linalg.norm(vertices[0] - vertices[2])
    d12 = np.linalg.norm(vertices[1] - vertices[2])

    if d01 <= d02 and d01 <= d12:
        min_dist = d01
        mesh_corners = _triangle_mesh_vertices(vertices[2], vertices[0], vertices[1])
    elif d02 <= d12:
        min_dist = d02
        mesh_corners = _triangle_mesh_vertices(vertices[1], vertices[0], vertices[2])
    else:
        min_dist = d12
        mesh_corners = _triangle_mesh_vertices(vertices[0], vertices[1], vertices[2])

    if min_dist < sampling_distance:
        return vertices

    n_long = round(np.linalg.norm(mesh_corners[0] - mesh_corners[1]) / sampling_distance)
    first_parallel = np.linspace(mesh_corners[0], mesh_corners[1], n_long)
    second_parallel = np.linspace(mesh_corners[2], mesh_corners[3], n_long)

    n_short = round(np.linalg.norm(first_parallel[0] - second_parallel[0]) / sampling_distance)

    mesh_points = [np.linspace(first_parallel[i], second_parallel[i], n_short)
                   for i in range(len(first_parallel))]
    return np.concatenate(mesh_points, axis=0)


def _barycentric_coords(triangle: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Compute barycentric coordinates of a point with respect to a triangle.

    Projects the triangle onto the plane that discards the dominant axis of
    the surface normal, then solves for the three barycentric weights.

    Parameters
    ----------
    triangle : numpy.ndarray, shape (3, 3)
        Triangle vertices; each row is ``[x, y, z]``.
    point : numpy.ndarray, shape (3,)
        Query point in 3-D space.

    Returns
    -------
    b : numpy.ndarray, shape (3,)
        Barycentric coordinates ``[b0, b1, b2]``.  Returns ``[-1, -1, -1]``
        if the triangle has zero area.
    """
    from math import fabs

    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[1])
    abs_n = [fabs(normal[0]), fabs(normal[1]), fabs(normal[2])]

    if abs_n[0] >= abs_n[1] and abs_n[0] >= abs_n[2]:
        c1, c2 = 1, 2
    elif abs_n[1] >= abs_n[2]:
        c1, c2 = 2, 0
    else:
        c1, c2 = 0, 1

    u = np.array([triangle[0][c1] - triangle[2][c1],
                  triangle[1][c1] - triangle[2][c1],
                  point[c1] - triangle[0][c1],
                  point[c1] - triangle[2][c1]])
    v = np.array([triangle[0][c2] - triangle[2][c2],
                  triangle[1][c2] - triangle[2][c2],
                  point[c2] - triangle[0][c2],
                  point[c2] - triangle[2][c2]])

    denom = v[0] * u[1] - v[1] * u[0]
    if denom == 0.0:
        return np.array([-1.0, -1.0, -1.0])

    inv = 1.0 / denom
    b = np.zeros(3)
    b[0] = (v[3] * u[1] - v[1] * u[3]) * inv
    b[1] = (v[0] * u[2] - v[2] * u[0]) * inv
    b[2] = 1.0 - b[0] - b[1]
    return b


def point_inside_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    """Test whether a 3-D point lies inside a triangle.

    Uses barycentric coordinates: the point is inside if all three weights
    are in ``[-0.0001, 1.0001]`` (a small tolerance for floating-point
    boundary cases).

    Parameters
    ----------
    point : numpy.ndarray, shape (3,)
        Query point.
    triangle : numpy.ndarray, shape (3, 3)
        Triangle vertices.

    Returns
    -------
    bool
        ``True`` if the point is inside (or on the boundary of) the triangle.
    """
    b = _barycentric_coords(triangle, point)
    return bool(np.all(b >= -0.0001) and np.all(b <= 1.0001))


# ---------------------------------------------------------------------------
# Normalizers for polymorphic inputs
# ---------------------------------------------------------------------------


def as_rotation(
    source: RotationLike,
    *,
    euler_order: str = "zxz",
    degrees: bool = True,
) -> srot:
    """Normalize a rotation-like input to a SciPy Rotation.

    Canonical normalizer for the :data:`cryocat._types.RotationLike` type.
    Any function that accepts a RotationLike should call this at its
    entry point.

    Parameters
    ----------
    source : RotationLike
        One of:

        * :class:`scipy.spatial.transform.Rotation` -- returned as-is
        * Shape ``(3,)`` -- Euler angles (single rotation)
        * Shape ``(N, 3)`` -- stack of Euler angles
        * Shape ``(3, 3)`` -- rotation matrix
        * Shape ``(N, 3, 3)`` -- stack of rotation matrices
        * Shape ``(4,)`` -- quaternion (xyzw)
        * Shape ``(N, 4)`` -- stack of quaternions
    euler_order : str, default="zxz"
        Euler convention for Euler-angle inputs. Standard SciPy
        conventions are accepted.
    degrees : bool, default=True
        Whether the Euler angles are in degrees.

    Returns
    -------
    Rotation
        Equivalent SciPy Rotation object.

    Raises
    ------
    ValueError
        If the shape of `source` cannot be interpreted as a rotation.

    Examples
    --------
    >>> as_rotation([0, 45, 0])             # single Euler triple
    >>> as_rotation(np.eye(3))              # identity matrix
    >>> as_rotation([0, 0, 0, 1])           # identity quaternion
    """
    if isinstance(source, srot):
        return source

    arr = np.asarray(source, dtype=float)

    # 1-D shape (3,): single Euler triple
    if arr.ndim == 1 and arr.shape[0] == 3:
        return srot.from_euler(euler_order, arr, degrees=degrees)
    # 1-D shape (4,): single quaternion
    if arr.ndim == 1 and arr.shape[0] == 4:
        return srot.from_quat(arr)
    # 2-D shape (3, 3): single rotation matrix
    if arr.shape == (3, 3):
        return srot.from_matrix(arr)
    # 2-D shape (N, 3): stack of Euler triples
    if arr.ndim == 2 and arr.shape[1] == 3:
        return srot.from_euler(euler_order, arr, degrees=degrees)
    # 2-D shape (N, 4): stack of quaternions
    if arr.ndim == 2 and arr.shape[1] == 4:
        return srot.from_quat(arr)
    # 3-D shape (N, 3, 3): stack of matrices
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        return srot.from_matrix(arr)

    raise ValueError(
        f"Cannot interpret array of shape {arr.shape} as a rotation. "
        "Expected (3,), (N, 3), (3, 3), (N, 3, 3), (4,), or (N, 4)."
    )


def as_symmetry(source: Symmetry) -> tuple[str, int]:
    """Normalize a symmetry specifier to ``(group, order)``.

    Canonical normalizer for the :data:`cryocat._types.Symmetry` type.
    Consolidates the symmetry-parsing logic previously duplicated in
    :func:`cryomap.rotational_average` and :meth:`Motl.split_in_asymmetric_subunits`.

    Parameters
    ----------
    source : Symmetry
        Either a string like ``"C5"`` or ``"D2"`` (case-insensitive),
        or a bare integer/float ``n`` (interpreted as cyclic Cn).

    Returns
    -------
    group : str
        ``"C"`` for cyclic, ``"D"`` for dihedral.
    order : int
        The order of the symmetry group (n in Cn / Dn).

    Raises
    ------
    ValueError
        If the string does not start with C or D, if no digits are
        present, or if the input type is unsupported.

    Examples
    --------
    >>> as_symmetry("C5")
    ('C', 5)
    >>> as_symmetry("d2")
    ('D', 2)
    >>> as_symmetry(7)
    ('C', 7)
    """
    if isinstance(source, str):
        digits = re.findall(r"\d+", source)
        if not digits:
            raise ValueError(f"No order found in symmetry string {source!r}.")
        order = int(digits[-1])
        letter = source.strip()[0].upper()
        if letter == "C":
            return ("C", order)
        if letter == "D":
            return ("D", order)
        raise ValueError(
            f"Unknown symmetry {source!r}: only C (cyclic) and D (dihedral) are supported."
        )

    # int / float
    if isinstance(source, (int, float, np.integer, np.floating)):
        if source != int(source):
            raise ValueError(f"Symmetry order must be a whole number, got {source}.")
        return ("C", int(source))

    raise ValueError(
        f"Symmetry must be a string ('Cn'/'Dn') or an integer, got {type(source).__name__}."
    )


def as_triplet(
    input_value: TripletLike | None,
    reference_size: TripletLike | None = None,
) -> np.ndarray:
    """Normalize a TripletLike input to a length-3 integer ndarray.

    Canonical normalizer for the :data:`cryocat._types.TripletLike` type.
    Any function that accepts a TripletLike should call this at its entry
    point to get a length-3 ndarray to work with.

    If ``input_value`` is None, ``reference_size`` is normalized the same
    way and then divided by 2 (useful for deriving a centre from a box
    size).

    Parameters
    ----------
    input_value : TripletLike, optional
        A scalar (broadcast to all three axes), 3-tuple/3-list, single-element
        container ``(x,)`` (broadcast), or a length-3 ndarray.
    reference_size : TripletLike, optional
        Fallback used when ``input_value`` is None. Normalized the same way
        and divided by 2. Defaults to None.

    Returns
    -------
    numpy.ndarray
        Length-3 integer ndarray.

    Raises
    ------
    ValueError
        If both ``input_value`` and ``reference_size`` are None, or if a
        container input has length neither 1 nor 3.

    Examples
    --------
    >>> as_triplet(5)
    array([5, 5, 5])
    >>> as_triplet([1, 2, 3])
    array([1, 2, 3])
    >>> as_triplet(None, reference_size=8)
    array([4, 4, 4])
    """

    def format_input(unformatted_value):
        if isinstance(unformatted_value, (tuple, list, np.ndarray)):
            if len(unformatted_value) == 3:
                return np.asarray(unformatted_value).astype(int)
            elif len(unformatted_value) == 1:
                return np.full((3,), unformatted_value).astype(int)
            else:
                raise ValueError("The size have to be a single number or have to have length of 3!")
        elif isinstance(unformatted_value, (float, int)):
            return np.full((3,), unformatted_value).astype(int)

    if input_value is not None:
        size_correct_format = format_input(input_value)
    elif reference_size is not None:
        box_size = format_input(reference_size)
        size_correct_format = box_size // 2
    else:
        raise ValueError("Either input_size or referene_size have to be specified")

    return size_correct_format