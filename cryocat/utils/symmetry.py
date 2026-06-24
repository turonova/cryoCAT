"""Symmetry group rotations for cryo-ET processing.

Provides rotation-matrix representations of crystallographic point groups
(C, D, T, O, I) and utilities for converting them to Euler angles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as rot

from cryocat._types import EulerAngles, Symmetry
from cryocat.utils import geom


_AXIS_MAP: dict[str, np.ndarray] = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}

_PHI = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio


def _normalize_axis(axis: str | np.ndarray) -> np.ndarray:
    """Return a unit-vector for *axis*.

    Parameters
    ----------
    axis : str or ndarray
        One of ``"x"``, ``"y"``, ``"z"`` (case-insensitive) or an
        array-like that will be normalised to unit length.

    Returns
    -------
    numpy.ndarray
        Shape ``(3,)`` unit vector.

    Raises
    ------
    ValueError
        If the string key is unknown or the vector is zero-length.
    """
    if isinstance(axis, str):
        key = axis.strip().lower()
        if key in _AXIS_MAP:
            return _AXIS_MAP[key]
        raise ValueError(f"Unknown axis name {axis!r}; expected 'x', 'y', or 'z'.")
    v = np.asarray(axis, dtype=float).ravel()
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("Axis vector must be non-zero.")
    return v / norm


def compute_conjugation_matrix(
    in_axis: str | np.ndarray = "z",
    out_axis: str | np.ndarray = "x",
) -> np.ndarray:
    """Rotation matrix *C* that maps *in_axis* to *out_axis*.

    Parameters
    ----------
    in_axis : str or ndarray, optional
        Source axis.  Default is ``"z"``.
    out_axis : str or ndarray, optional
        Target axis.  Default is ``"x"``.

    Returns
    -------
    numpy.ndarray
        ``(3, 3)`` rotation matrix satisfying ``C @ in_axis == out_axis``.
    """
    a = _normalize_axis(in_axis)
    b = _normalize_axis(out_axis)
    if np.allclose(a, b):
        return np.eye(3)
    if np.allclose(a, -b):
        # 180° rotation around a perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if not np.allclose(np.abs(a), [1, 0, 0]) else np.array([0.0, 1.0, 0.0])
        return rot.from_rotvec(np.pi * perp).as_matrix()
    cross = np.cross(a, b)
    angle = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    return rot.from_rotvec(angle * cross / np.linalg.norm(cross)).as_matrix()


def _bfs_group(
    generators: list[np.ndarray],
    *,
    atol: float = 1e-9,
) -> np.ndarray:
    """Enumerate all group elements reachable from *generators* by BFS.

    Parameters
    ----------
    generators : list of ndarray
        ``(3, 3)`` rotation matrices that generate the group.
    atol : float, optional
        Absolute tolerance for matrix equality.  Default is ``1e-9``.

    Returns
    -------
    numpy.ndarray
        ``(M, 3, 3)`` array of all group elements, starting with the
        identity.
    """
    identity = np.eye(3)
    elements: list[np.ndarray] = [identity]
    queue: list[np.ndarray] = [identity]

    while queue:
        current = queue.pop(0)
        for gen in generators:
            new = current @ gen
            if not any(np.allclose(new, e, atol=atol) for e in elements):
                elements.append(new)
                queue.append(new)

    return np.array(elements)


class SymmGroup:
    """Base class for a finite point symmetry group acting on SO(3).

    Attributes
    ----------
    order : int
        Number of group elements.
    matrices : numpy.ndarray
        ``(order, 3, 3)`` array of rotation matrices.
    """

    order: int
    matrices: np.ndarray
    _generators: list[np.ndarray]

    def _build(self) -> None:
        """Populate :attr:`matrices` via BFS and validate the group order."""
        self.matrices = _bfs_group(self._generators)
        if len(self.matrices) != self.order:
            raise RuntimeError(
                f"{type(self).__name__}: expected {self.order} elements, "
                f"got {len(self.matrices)}."
            )


class CyclicGroup(SymmGroup):
    """Cyclic symmetry group C_n (n elements, rotations around z-axis).

    Parameters
    ----------
    n : int
        Fold of the cyclic symmetry (n >= 1).
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"Cyclic order must be >= 1, got {n}.")
        self.order = n
        angle = 2.0 * np.pi / n
        self._generators = [rot.from_rotvec(angle * np.array([0.0, 0.0, 1.0])).as_matrix()]
        self._build()


class DihedralGroup(SymmGroup):
    """Dihedral symmetry group D_n (2n elements).

    Parameters
    ----------
    n : int
        Fold of the principal axis (n >= 1).

    Notes
    -----
    The 2-fold axes are placed at the half-step offset (``90/n`` degrees
    from x), giving a staggered (antiprismatic) layout.  An in-plane shift
    along x therefore lands on 2n distinct, evenly-spaced positions at
    ``360 / (2n)`` degree steps.  Elements are sorted by their in-plane
    angle so the output is deterministic and reproduces the classic
    even/odd-interleaved ordering (0°, half-step, step, …).
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"Dihedral order must be >= 1, got {n}.")
        self.order = 2 * n
        angle = 2.0 * np.pi / n
        gen_cn = rot.from_rotvec(angle * np.array([0.0, 0.0, 1.0])).as_matrix()
        # 2-fold at half-step from x → staggered, orbit of x̂ covers 2n distinct spots
        alpha = np.deg2rad(90.0 / n)
        axis = np.array([np.cos(alpha), np.sin(alpha), 0.0])
        gen_c2 = rot.from_rotvec(np.pi * axis).as_matrix()
        self._generators = [gen_cn, gen_c2]
        self._build()
        # Sort by the in-plane angle of R @ x̂ for deterministic ordering
        phi_eff = np.degrees(np.arctan2(self.matrices[:, 1, 0], self.matrices[:, 0, 0])) % 360.0
        self.matrices = self.matrices[np.argsort(phi_eff, kind="stable")]


class TetrahedralGroup(SymmGroup):
    """Proper rotation group T of the tetrahedron (12 elements).

    Generators: C3 around (1,1,1)/√3 and C2 around z.
    """

    order = 12

    def __init__(self) -> None:
        axis_c3 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        gen_c3 = rot.from_rotvec(2.0 * np.pi / 3.0 * axis_c3).as_matrix()
        gen_c2z = rot.from_rotvec(np.pi * np.array([0.0, 0.0, 1.0])).as_matrix()
        self._generators = [gen_c3, gen_c2z]
        self._build()


class OctahedralGroup(SymmGroup):
    """Proper rotation group O of the octahedron/cube (24 elements).

    Generators: C4 around z and C4 around x.
    """

    order = 24

    def __init__(self) -> None:
        gen_c4z = rot.from_rotvec(np.pi / 2.0 * np.array([0.0, 0.0, 1.0])).as_matrix()
        gen_c4x = rot.from_rotvec(np.pi / 2.0 * np.array([1.0, 0.0, 0.0])).as_matrix()
        self._generators = [gen_c4z, gen_c4x]
        self._build()


class IcosahedralGroup(SymmGroup):
    """Proper rotation group I of the icosahedron/dodecahedron (60 elements).

    Generators: C5 around a vertex axis and C3 around the adjacent face normal.
    """

    order = 60

    def __init__(self) -> None:
        # C5 axis: normalised first vertex of the icosahedron [0, 1, φ]
        v_c5 = np.array([0.0, 1.0, _PHI])
        axis_c5 = v_c5 / np.linalg.norm(v_c5)
        gen_c5 = rot.from_rotvec(2.0 * np.pi / 5.0 * axis_c5).as_matrix()

        # C3 axis: centroid of the adjacent face {[0,1,φ], [1,φ,0], [φ,0,1]},
        # which simplifies to (1+φ, 1+φ, 1+φ)/... ∝ (1,1,1)/√3.
        axis_c3 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        gen_c3 = rot.from_rotvec(2.0 * np.pi / 3.0 * axis_c3).as_matrix()

        self._generators = [gen_c5, gen_c3]
        self._build()


SYMMETRY_GROUPS: dict[str, type[SymmGroup]] = {
    "C": CyclicGroup,
    "D": DihedralGroup,
    "T": TetrahedralGroup,
    "O": OctahedralGroup,
    "I": IcosahedralGroup,
}


def get_symmetry_rotations(
    symmetry: Symmetry,
    *,
    axis: str | np.ndarray = "z",
    conjugation_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Return the rotation matrices for a symmetry group.

    Parameters
    ----------
    symmetry : Symmetry
        Symmetry specifier, e.g. ``"C5"``, ``"D3"``, ``"T"``, ``"O"``,
        ``"I"``, or a bare integer (interpreted as cyclic).
    axis : str or ndarray, optional
        Principal symmetry axis.  Default is ``"z"``.  Ignored when
        *conjugation_matrix* is provided.
    conjugation_matrix : ndarray, optional
        Pre-computed ``(3, 3)`` conjugation matrix.  When given, *axis*
        is ignored.

    Returns
    -------
    numpy.ndarray
        ``(M, 3, 3)`` array of rotation matrices.  The identity is
        always the first element.
    """
    group_letter, order = geom.as_symmetry(symmetry)

    cls = SYMMETRY_GROUPS[group_letter]
    if group_letter in ("T", "O", "I"):
        group: SymmGroup = cls()
    else:
        group = cls(order)

    matrices = group.matrices  # (M, 3, 3) around z-axis

    # Apply axis reorientation via conjugation C @ R @ C^T
    if conjugation_matrix is not None:
        C = np.asarray(conjugation_matrix, dtype=float)
    elif isinstance(axis, str) and axis.strip().lower() == "z":
        return matrices
    else:
        C = compute_conjugation_matrix("z", axis)

    return np.array([C @ R @ C.T for R in matrices])


def get_symmetry_angles(
    symmetry: Symmetry,
    *,
    euler_convention: str = "zxz",
    degrees: bool = True,
    return_df: bool = False,
    out_path: str | None = None,
) -> EulerAngles | pd.DataFrame:
    """Return Euler angles for all elements of a symmetry group.

    Parameters
    ----------
    symmetry : Symmetry
        Symmetry specifier (see :func:`get_symmetry_rotations`).
    euler_convention : str, optional
        Euler angle convention passed to
        :meth:`scipy.spatial.transform.Rotation.as_euler`.
        Default is ``"zxz"``.
    degrees : bool, optional
        If ``True`` (default), angles are in degrees; otherwise radians.
    return_df : bool, optional
        If ``True``, return a :class:`pandas.DataFrame` with one column
        per Euler angle; otherwise return a NumPy array.
    out_path : str or Path, optional
        When provided, write the result as a CSV to this path.

    Returns
    -------
    EulerAngles or pandas.DataFrame
        ``(M, 3)`` array of Euler angles, or a DataFrame when
        *return_df* is ``True``.
    """
    matrices = get_symmetry_rotations(symmetry)
    angles = rot.from_matrix(matrices).as_euler(euler_convention, degrees=degrees)

    if return_df or out_path is not None:
        if euler_convention.lower() == "zxz":
            cols = ["phi", "theta", "psi"]
        else:
            cols = [f"e{i}" for i in range(3)]
        df = pd.DataFrame(angles, columns=cols)
        if out_path is not None:
            df.to_csv(out_path, index=False)
        if return_df:
            return df

    return angles
