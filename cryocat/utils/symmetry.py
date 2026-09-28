"""Symmetry group rotations for cryo-ET processing.

Provides rotation-matrix representations of crystallographic point groups
(C, D, T, O, I) and utilities for converting them to Euler angles.

Each group instance also carries an orientation (:attr:`SymmGroup.rotation`)
and is linked to the matching Platonic solid of :mod:`cryocat.utils.geom`
(:meth:`SymmGroup.to_polyhedron`, :meth:`SymmGroup.from_polyhedron`). The
dependency is one-way: this module imports :mod:`~cryocat.utils.geom`, never
the reverse.

Notes
-----
**Group order is not the number of corners of the solid.** The *order* of a
group (:attr:`SymmGroup.order`, and the number returned by
:func:`cryocat.utils.geom.as_symmetry`) counts its *rotations*: 60 for
``"I"``. A :class:`cryocat.utils.geom.Icosahedron` has only 12 corners
(vertices). Both numbers are correct; they count different things.

Apply every rotation of a group to one point and count the distinct places
it lands (:meth:`SymmGroup.orbit`). A point on an ``n``-fold spin axis is
left in place by ``n`` of the rotations, so it lands on only
``order / n`` distinct places; a point off every axis lands on ``order``
places. Corners, edge midpoints and face centres of a solid sit on axes,
hence::

    group  order  solid          vertices    edges      faces
    -----  -----  -------------  ----------  ---------  ---------
    T      12     Tetrahedron     4 = 12/3    6 = 12/2   4 = 12/3
    O      24     Octahedron      6 = 24/4   12 = 24/2   8 = 24/3
    O      24     Cube            8 = 24/3   12 = 24/2   6 = 24/4
    I      60     Icosahedron    12 = 60/5   30 = 60/2  20 = 60/3
    I      60     Dodecahedron   20 = 60/3   30 = 60/2  12 = 60/5

In practice: splitting a particle into asymmetric units
(:meth:`cryocat.core.cryomotl.Motl.split_in_asymmetric_subunits`) gives
``order`` copies (60 for ``"I"``), whereas expanding it onto the corners,
edges or faces of a solid
(:meth:`cryocat.analysis.structure.PolyhedralComplex.expand`) gives the
vertex, edge or face count (12, 30 or 20 for the icosahedron). For ``"Dn"``
note also that :func:`~cryocat.utils.geom.as_symmetry` returns ``n``,
while :class:`DihedralGroup` has ``2 * n`` rotations.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as rot

from cryocat._types import ArrayLike, EulerAngles, RotationLike, Symmetry
from cryocat.utils import geom


_AXIS_MAP: dict[str, np.ndarray] = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}

_PHI = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio

# Platonic solids whose symmetry is described by each group letter. The first
# entry is the default returned by SymmGroup.to_polyhedron().
_SOLIDS: dict[str, tuple[type[geom.Polyhedron], ...]] = {
    "T": (geom.Tetrahedron,),
    "O": (geom.Octahedron, geom.Cube),
    "I": (geom.Icosahedron, geom.Dodecahedron),
}

# Names accepted by the ``kind`` argument of SymmGroup.to_polyhedron().
_SOLID_KINDS: dict[str, type[geom.Polyhedron]] = {
    "tetrahedron": geom.Tetrahedron,
    "octahedron": geom.Octahedron,
    "cube": geom.Cube,
    "icosahedron": geom.Icosahedron,
    "dodecahedron": geom.Dodecahedron,
}


def _as_single_rotation(orientation: RotationLike) -> rot:
    """Normalize *orientation* to a single SciPy Rotation.

    Parameters
    ----------
    orientation : RotationLike
        A :class:`scipy.spatial.transform.Rotation`, a ``(3, 3)`` rotation
        matrix, or any other single-rotation input accepted by
        :func:`cryocat.utils.geom.as_rotation` (Euler triple, quaternion).

    Returns
    -------
    scipy.spatial.transform.Rotation
        A single rotation.

    Raises
    ------
    ValueError
        If *orientation* describes more than one rotation.
    """
    if isinstance(orientation, rot):
        r = orientation
    else:
        arr = np.asarray(orientation, dtype=float)
        # A single orientation given as a 3x3 array is always a matrix here.
        r = rot.from_matrix(arr) if arr.shape == (3, 3) else geom.as_rotation(arr)
    if not r.single:
        raise ValueError(f"Expected a single rotation, got {len(r)} rotations.")
    return r


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

    A freshly constructed group is in the canonical ("textbook") orientation.
    :meth:`oriented` returns the same group turned into another orientation,
    and :meth:`to_polyhedron` / :meth:`from_polyhedron` convert between a
    group and the matching Platonic solid of :mod:`cryocat.utils.geom`.

    Attributes
    ----------
    symbol : str
        Group letter: ``"C"``, ``"D"``, ``"T"``, ``"O"`` or ``"I"``.
    order : int
        Number of group elements (rotations), e.g. 60 for ``"I"``. This is
        not the vertex count of the matching solid (12 for the
        icosahedron); see the module Notes.
    matrices : numpy.ndarray
        ``(order, 3, 3)`` array of rotation matrices, expressed in the
        group's orientation. The identity is always the first element.
    rotation : scipy.spatial.transform.Rotation
        Orientation of the group relative to the canonical one (identity for
        a freshly constructed group).
    """

    symbol: str
    order: int
    matrices: np.ndarray
    rotation: rot
    _generators: list[np.ndarray]

    def _build(self) -> None:
        """Populate :attr:`matrices` via BFS and validate the group order."""
        self.matrices = _bfs_group(self._generators)
        self.rotation = rot.identity()
        if len(self.matrices) != self.order:
            raise RuntimeError(
                f"{type(self).__name__}: expected {self.order} elements, "
                f"got {len(self.matrices)}."
            )

    def oriented(self, orientation: RotationLike) -> "SymmGroup":
        """Return the same group turned by *orientation*.

        Each rotation ``g`` is re-expressed as ``R @ g @ R.T`` ("undo R, apply
        g, redo R"). Turning is cumulative: ``group.oriented(A).oriented(B)``
        has orientation ``B * A``. The group itself is not modified.

        Parameters
        ----------
        orientation : RotationLike
            Single rotation ``R`` (Rotation, 3x3 matrix, Euler triple or
            quaternion; see :func:`cryocat.utils.geom.as_rotation`).

        Returns
        -------
        SymmGroup
            New instance of the same class with turned :attr:`matrices` and
            :attr:`rotation` = ``R * self.rotation``. The identity stays first.

        Raises
        ------
        ValueError
            If *orientation* describes more than one rotation.
        """
        r = _as_single_rotation(orientation)
        C = r.as_matrix()
        turned = copy.copy(self)
        turned.matrices = C @ self.matrices @ C.T
        turned.rotation = r * self.rotation
        return turned

    def axes(self, fold: int, *, atol: float = 1e-6) -> np.ndarray:
        """Return the rotation axes of a given fold, in both directions.

        Rotations are grouped by their axis; an axis shared by ``k``
        non-identity rotations is ``(k + 1)``-fold. For the Platonic groups
        these directions are the matching solid's vertex, edge and face
        directions (e.g. ``IcosahedralGroup().axes(5)`` gives the 12
        icosahedron vertices).

        Parameters
        ----------
        fold : int
            Fold of the axes to return (>= 2).
        atol : float, optional
            Tolerance for treating two axes as the same. Default is ``1e-6``.

        Returns
        -------
        numpy.ndarray
            ``(2 * n_axes, 3)`` unit vectors: each axis as ``+u`` followed by
            ``-u``. Empty ``(0, 3)`` array if the group has no axis of that
            fold.

        Raises
        ------
        ValueError
            If *fold* is smaller than 2.
        """
        if fold < 2:
            raise ValueError(f"fold must be >= 2, got {fold}.")
        found: list[np.ndarray] = []
        counts: list[int] = []
        for m in self.matrices:
            rotvec = rot.from_matrix(m).as_rotvec()
            angle = np.linalg.norm(rotvec)
            if angle < atol:
                continue  # identity
            u = rotvec / angle
            for i, v in enumerate(found):
                if abs(abs(u @ v) - 1.0) < atol:
                    counts[i] += 1
                    break
            else:
                found.append(u)
                counts.append(1)
        selected = [u for u, c in zip(found, counts) if c + 1 == fold]
        if not selected:
            return np.empty((0, 3))
        return np.vstack([np.vstack((u, -u)) for u in selected])

    def orbit(self, point: ArrayLike, *, atol: float = 1e-6) -> np.ndarray:
        """Return the distinct positions of *point* under all group rotations.

        The number of distinct positions is ``order`` divided by the number
        of rotations that leave *point* in place: a point off every symmetry
        axis gives ``order`` positions, a point on an ``n``-fold axis gives
        ``order / n``. This is why a solid's vertex/edge/face counts differ
        from the group order (table in the module Notes).

        Parameters
        ----------
        point : ArrayLike
            ``(3,)`` point, e.g. a subunit shift or a vertex.
        atol : float, optional
            Absolute tolerance (scaled by ``max(1, |point|)``) for treating two
            positions as the same. Default is ``1e-6``.

        Returns
        -------
        numpy.ndarray
            ``(K, 3)`` distinct positions, in the order of first occurrence
            (the first is *point* itself).
        """
        p = np.asarray(point, dtype=float).reshape(3)
        tol = atol * max(1.0, float(np.linalg.norm(p)))
        distinct: list[np.ndarray] = []
        for q in self.matrices @ p:
            if not any(np.linalg.norm(q - d) <= tol for d in distinct):
                distinct.append(q)
        return np.array(distinct)

    def to_polyhedron(self, kind: str | None = None, radius: float = 1.0) -> "geom.Polyhedron":
        """Return the Platonic solid matching this group, in the same orientation.

        Parameters
        ----------
        kind : str, optional
            Which solid to build when the group matches more than one:
            ``"octahedron"`` or ``"cube"`` for O, ``"icosahedron"`` or
            ``"dodecahedron"`` for I, ``"tetrahedron"`` for T. Default is None:
            Tetrahedron for T, Octahedron for O, Icosahedron for I.
        radius : float, optional
            Circumscribed radius of the solid. Default is ``1.0``.

        Returns
        -------
        cryocat.utils.geom.Polyhedron
            Solid with ``R = self.rotation``.

        Raises
        ------
        ValueError
            If the group has no associated Platonic solid (C, D) or *kind* does
            not match the group.

        Notes
        -----
        For T, the vertices always follow :class:`cryocat.utils.geom.Tetrahedron`.
        The tetrahedron flipped through its centre (vertices ``-v``) is left
        unchanged by the same 12 rotations, so the group alone cannot tell
        vertices from face centres; they are never derived from :meth:`axes`.
        """
        solids = _SOLIDS.get(self.symbol)
        if not solids:
            raise ValueError(f"{type(self).__name__} has no associated Platonic solid.")
        if kind is None:
            solid_cls = solids[0]
        else:
            solid_cls = _SOLID_KINDS.get(kind.strip().lower())
            if solid_cls not in solids:
                allowed = ", ".join(repr(s.__name__.lower()) for s in solids)
                raise ValueError(f"kind {kind!r} does not match {type(self).__name__}; expected one of {allowed}.")
        return solid_cls(radius=radius, R=self.rotation)

    @classmethod
    def from_polyhedron(cls, solid: "geom.Polyhedron", *, atol: float = 1e-6) -> "SymmGroup":
        """Return the symmetry group of *solid*, turned the way *solid* is turned.

        The group is looked up from the solid's type (Tetrahedron → T,
        Octahedron/Cube → O, Icosahedron/Dodecahedron → I), oriented by
        ``solid.rotation``, and checked: every rotation must leave the solid's
        vertices unchanged. The result does not depend on which of the
        equivalent rotations ``solid.rotation`` happens to be (e.g. which
        neighbouring vertex was used in :meth:`geom.Polyhedron.from_vectors`).

        Parameters
        ----------
        solid : cryocat.utils.geom.Polyhedron
            A Platonic solid, e.g. ``complex.solid`` after ``fit_geometry``.
        atol : float, optional
            Tolerance (radians, on the unit sphere) of the invariance check.
            Default is ``1e-6``.

        Returns
        -------
        SymmGroup
            The matching group with :attr:`rotation` = ``solid.rotation``.

        Raises
        ------
        TypeError
            If *solid* is not one of the Platonic solids of :mod:`geom`.
        ValueError
            If called on a group class that doesn't match the solid (e.g.
            ``IcosahedralGroup.from_polyhedron(Tetrahedron())``), or if the
            invariance check fails.
        """
        letter = next((s for s, solids in _SOLIDS.items() if isinstance(solid, solids)), None)
        if letter is None:
            raise TypeError(f"Expected a Platonic solid from cryocat.utils.geom, got {type(solid).__name__}.")
        group_cls = SYMMETRY_GROUPS[letter]
        if cls is not SymmGroup and cls is not group_cls:
            raise ValueError(f"{type(solid).__name__} belongs to {group_cls.__name__}, not {cls.__name__}.")
        group = group_cls().oriented(solid.rotation)
        verts = solid.vertices / np.linalg.norm(solid.vertices, axis=1, keepdims=True)
        for m in group.matrices:
            if geom.hausdorff_distance_sphere(verts @ m.T, verts) > atol:
                raise ValueError(
                    f"{group_cls.__name__} in the orientation of the given {type(solid).__name__} "
                    "does not leave its vertices unchanged; the solid's canonical frame and the "
                    "group's canonical frame do not match."
                )
        return group


class CyclicGroup(SymmGroup):
    """Cyclic symmetry group C_n (n elements, rotations around z-axis).

    Parameters
    ----------
    n : int
        Fold of the cyclic symmetry (n >= 1).
    """

    symbol = "C"

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

    symbol = "D"

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

    The same 12 rotations also leave the tetrahedron flipped through its
    centre unchanged; :meth:`to_polyhedron` therefore always follows the
    vertices of :class:`cryocat.utils.geom.Tetrahedron`.
    """

    symbol = "T"
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

    symbol = "O"
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

    symbol = "I"
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

    # Axis reorientation via conjugation C @ R @ C^T, implemented once in
    # SymmGroup.oriented().
    if conjugation_matrix is not None:
        return group.oriented(np.asarray(conjugation_matrix, dtype=float)).matrices
    if isinstance(axis, str) and axis.strip().lower() == "z":
        return group.matrices  # (M, 3, 3) around z-axis
    return group.oriented(compute_conjugation_matrix("z", axis)).matrices


def get_symmetry_angles(
    symmetry: Symmetry,
    *,
    euler_convention: str = "zxz",
    degrees: bool = True,
    conjugation_matrix: np.ndarray | None = None,
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
    conjugation_matrix : ndarray, optional
        Pre-computed ``(3, 3)`` conjugation matrix.  When given, *axis*
        is ignored.
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
    matrices = get_symmetry_rotations(symmetry, conjugation_matrix= conjugation_matrix)
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
