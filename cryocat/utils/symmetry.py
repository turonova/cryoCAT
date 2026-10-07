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

**Canonical orientations.** The groups are built in fixed orientations,
identical to ChimeraX's defaults (``sym`` / ``measure symmetry``) except for
D_n (checked against the ChimeraX source, 2026-10-01)::

    group  canonical axes                                     ChimeraX
    -----  -------------------------------------------------  ---------------------------
    C_n    n-fold along z                                     Cn
    D_n    n-fold along z; 2-folds at 90/n degrees from x     Dn turned by 90/n about z
    T      2-folds along x, y, z                              T, orientation 222
    O      4-folds along x, y, z                              O
    I      2-folds along x, y, z; 5-folds in the yz-plane     I, orientation 222

Functions that assume a symmetric map or template is in this orientation
(e.g. :func:`cryocat.utils.geom.generate_angles`) need the map aligned
first; see the Notes of that function.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as rot

from cryocat._types import ArrayLike, EulerAngles, RotationLike, Symmetry
from cryocat.utils import geom


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

    Thin wrapper around :func:`cryocat.utils.geom.unit_axis` (single
    implementation; kept for the existing call sites in this module).

    Parameters
    ----------
    axis : str or ndarray
        One of ``"x"``, ``"y"``, ``"z"`` (case-insensitive) or a
        3-element array-like that will be normalised to unit length.

    Returns
    -------
    numpy.ndarray
        Shape ``(3,)`` unit vector.

    Raises
    ------
    ValueError
        If the string key is unknown, the vector does not have 3 elements,
        or it is zero-length.
    """
    return geom.unit_axis(axis)


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

    This differs from ChimeraX, which places a 2-fold axis along x: this
    group equals ChimeraX's D_n turned by ``90/n`` degrees about z.
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


def _make_group(symmetry: Symmetry) -> SymmGroup:
    """Build the canonical :class:`SymmGroup` for a symmetry specifier.

    Parameters
    ----------
    symmetry : Symmetry
        Symmetry specifier (``"C5"``, ``"D3"``, ``"T"``, ``"O"``, ``"I"`` or
        an integer, interpreted as cyclic); normalized via
        :func:`cryocat.utils.geom.as_symmetry`.

    Returns
    -------
    SymmGroup
        The group in its canonical orientation (identity first in ``matrices``).
    """
    letter, order = geom.as_symmetry(symmetry)
    cls = SYMMETRY_GROUPS[letter]
    return cls() if letter in ("T", "O", "I") else cls(order)


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
        Direction onto which the group's z-axis is moved: ``"x"``, ``"y"``,
        ``"z"`` or a vector. Default is ``"z"`` (canonical orientation).
        Fully defines the orientation only for cyclic symmetry (see Notes).
        Ignored when *conjugation_matrix* is provided.
    conjugation_matrix : ndarray, optional
        Pre-computed ``(3, 3)`` rotation matrix ``C`` giving the full
        orientation of the group: every rotation ``g`` is returned as
        ``C @ g @ C.T``. When given, *axis* is ignored. Use this (not
        *axis*) to orient D/T/O/I groups.

    Returns
    -------
    numpy.ndarray
        ``(M, 3, 3)`` array of rotation matrices.  The identity is
        always the first element.

    Warns
    -----
    UserWarning
        If *axis* is used with a non-cyclic group (D/T/O/I) and does not lie
        along ``±z``, since the result then depends on a hidden choice (see
        Notes). The returned matrices are not affected by the warning.

    Notes
    -----
    One axis fixes the orientation of a C_n group completely, since all its
    rotations turn about that axis. D/T/O/I groups have further symmetry axes
    (e.g. the half-turn axes lying flat around the main axis of D_n); knowing
    where one axis points does not say where the others are, much as knowing
    where a fan's shaft points does not say where its blades are.

    *axis* moves the group by the shortest turn taking z onto *axis*
    (:func:`compute_conjugation_matrix`, e.g. 90 degrees about y for
    ``"x"``), so the remaining axes land wherever that turn puts them. The
    result is always a complete, valid group with its z-axis along *axis*,
    but it matches a given reference only if the reference's other axes
    happen to lie there too. For T and O, *axis* ``"x"`` or ``"y"`` returns
    the same set of rotations as the canonical group (that turn is itself a
    symmetry of the cube's frame), so it only matches a canonically oriented
    reference.

    For D/T/O/I, pass the full orientation as *conjugation_matrix* instead,
    e.g. ``SymmGroup.from_polyhedron(solid).rotation.as_matrix()`` for a
    fitted solid (the same orientation used by ``symmetry_orientation`` in
    :meth:`cryocat.core.cryomotl.Motl.split_in_asymmetric_subunits`).
    """
    group = _make_group(symmetry)

    # Axis reorientation via conjugation C @ R @ C^T, implemented once in
    # SymmGroup.oriented().
    if conjugation_matrix is not None:
        return group.oriented(np.asarray(conjugation_matrix, dtype=float)).matrices
    if isinstance(axis, str) and axis.strip().lower() == "z":
        return group.matrices  # (M, 3, 3) around z-axis
    # Along ±z the canonical group is returned, so only other axes are ambiguous.
    if group.symbol != "C" and not np.isclose(abs(_normalize_axis(axis)[2]), 1.0):
        warnings.warn(
            f"axis={axis!r} fixes only one axis of the {group.symbol} group; its other symmetry axes "
            "are placed by the shortest turn from z and may not match the reference. "
            "Pass the full orientation as conjugation_matrix instead.",
            UserWarning,
            stacklevel=2,
        )
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


def _check_scorable(symmetry: Symmetry, kind: str | None) -> tuple[str, int]:
    """Parse *symmetry* and reject combinations the angular score does not support.

    Parameters
    ----------
    symmetry : Symmetry
        Symmetry specifier.
    kind : str or None
        Solid name; only valid for T/O/I.

    Returns
    -------
    tuple of (str, int)
        Group letter and order, as returned by :func:`cryocat.utils.geom.as_symmetry`.

    Raises
    ------
    ValueError
        If a cyclic order is below 2, or *kind* is given for cyclic symmetry.
    NotImplementedError
        For dihedral symmetry.
    """
    letter, order = geom.as_symmetry(symmetry)
    if letter == "D":
        raise NotImplementedError(f"Angular score is not implemented for dihedral symmetry ({symmetry!r}).")
    if letter == "C":
        if order <= 1:
            raise ValueError("Cyclic symmetry must specify an order greater than 1.")
        if kind is not None:
            raise ValueError(f"kind={kind!r} is only applicable to T/O/I symmetry, not cyclic.")
    return letter, order


def max_angular_mismatch(symmetry: Symmetry, kind: str | None = None) -> float:
    """Largest possible mismatch (radians) between two turned copies of a symmetric shape.

    This is the normaliser ``d_max`` of :func:`angular_score`: the score is
    ``1 - d / d_max``, so it only spans 0 to 1 if ``d_max`` is the true
    worst case.

    Parameters
    ----------
    symmetry : Symmetry
        ``"CN"``/int (N > 1), ``"T"``, ``"O"`` or ``"I"``.
    kind : str, optional
        For O/I, which solid's corners are used (see
        :meth:`SymmGroup.to_polyhedron`). Default is the group's default solid.

    Returns
    -------
    float
        ``pi / N`` for cyclic C_N; for T/O/I the angle from a corner of the
        solid to the centre of the nearest face.

    Raises
    ------
    ValueError
        For a cyclic order below 2, *kind* given for cyclic symmetry, or a
        *kind* that doesn't match the group.
    NotImplementedError
        For dihedral symmetry.

    Notes
    -----
    The mismatch between two copies is the angle of their worst-matched
    corner. It is largest when a corner of one copy lands in the deepest
    "hole" between the corners of the other: halfway between two
    neighbouring corners of a polygon (``pi / N``), or the centre of a face
    of a solid. For T/O/I this gives 70.53°, 54.74° and 37.38°; both solids
    of a group (octahedron/cube, icosahedron/dodecahedron) have the same
    value.
    """
    letter, order = _check_scorable(symmetry, kind)
    if letter == "C":
        return np.pi / order
    solid = SYMMETRY_GROUPS[letter]().to_polyhedron(kind=kind)
    corner = solid.vertices[0] / np.linalg.norm(solid.vertices[0])
    centres = solid.faces / np.linalg.norm(solid.faces, axis=1, keepdims=True)
    return float(np.arccos(np.clip(centres @ corner, -1.0, 1.0)).min())


def angular_score(
    rotations_1: RotationLike,
    rotations_2: RotationLike,
    symmetry: Symmetry,
    kind: str | None = None,
    max_val: float | None = None,
    *,
    chunk_size: int = 10000,
) -> np.ndarray:
    """Symmetry-aware similarity of paired orientations, from 0 (most different) to 1 (identical).

    Marker points with the particle's symmetry (a regular polygon for
    C_N, the corners of the matching Platonic solid for T/O/I) are turned
    by each rotation of a pair; the score is ``1 - d / d_max``, where ``d``
    is the angle of the worst-matched marker
    (:func:`cryocat.utils.geom.hausdorff_distance_sphere`) and ``d_max``
    the largest possible value of ``d`` (:func:`max_angular_mismatch`).
    Orientations that differ only by a symmetry rotation score 1.

    Parameters
    ----------
    rotations_1 : RotationLike
        First orientation of each pair (``N`` rotations, or one). Normalized
        via :func:`cryocat.utils.geom.as_rotation` (Euler angles in degrees,
        ``zxz``).
    rotations_2 : RotationLike
        Second orientation of each pair, same length as *rotations_1*.
    symmetry : Symmetry
        ``"CN"``/int (N > 1), ``"T"``, ``"O"`` or ``"I"``.
    kind : str, optional
        For O/I, which solid's corners are the markers: ``"octahedron"``
        (default) or ``"cube"``, ``"icosahedron"`` (default) or
        ``"dodecahedron"``. Scores for the two solids of a group agree at 0
        and 1 but differ slightly in between. Not applicable to cyclic
        symmetry.
    max_val : float, optional
        Normaliser ``d_max`` in radians. Default is
        :func:`max_angular_mismatch` for the given symmetry and kind.
    chunk_size : int, optional
        Number of pairs processed at once (bounds memory use). Default 10000.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` scores. As in
        :func:`cryocat.utils.geom.angular_score_for_c_symmetry`, values within
        ``1e-5`` of 1 are set to 1 and values below ``1e-5`` to 0.

    Raises
    ------
    ValueError
        If the two inputs have different lengths, for a cyclic order below 2,
        *kind* given for cyclic symmetry, or a *kind* that doesn't match the
        group.
    NotImplementedError
        For dihedral symmetry.

    Notes
    -----
    Cyclic symmetry is delegated to
    :func:`cryocat.utils.geom.angular_score_for_c_symmetry`, which compares
    only the in-plane angle (the first ``zxz`` Euler angle, ``phi``, of each
    rotation). T/O/I use the full 3D rotations, since these particles look
    the same after rotations about several different axes. The score
    depends only on how the two orientations differ: turning both by the
    same rotation leaves it unchanged.
    """
    letter, order = _check_scorable(symmetry, kind)
    m1 = geom.as_rotation(rotations_1).as_matrix().reshape(-1, 3, 3)
    m2 = geom.as_rotation(rotations_2).as_matrix().reshape(-1, 3, 3)
    if len(m1) != len(m2):
        raise ValueError(f"rotations_1 and rotations_2 must have the same length, got {len(m1)} and {len(m2)}.")

    if letter == "C":
        phi_1 = rot.from_matrix(m1).as_euler("zxz")[:, 0]
        phi_2 = rot.from_matrix(m2).as_euler("zxz")[:, 0]
        return geom.angular_score_for_c_symmetry(phi_1, phi_2, order, max_val)

    if max_val is None:
        max_val = max_angular_mismatch(symmetry, kind)
    vertices = SYMMETRY_GROUPS[letter]().to_polyhedron(kind=kind).vertices
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    distances = np.empty(len(m1))
    for start in range(0, len(m1), chunk_size):
        stop = start + chunk_size
        a = np.einsum("kj,nij->nki", vertices, m1[start:stop])  # (n, k, 3) corners turned by each rotation
        b = np.einsum("kj,nij->nki", vertices, m2[start:stop])
        angles = np.arccos(np.clip(np.einsum("nik,njk->nij", a, b), -1.0, 1.0))  # (n, k, k)
        distances[start:stop] = np.maximum(angles.min(axis=2).max(axis=1), angles.min(axis=1).max(axis=1))

    scores = 1.0 - distances / max_val
    scores[scores > 1 - 1e-5] = 1.0
    scores[scores < 1e-5] = 0.0
    return scores


def _rotation_angles_deg(traces: np.ndarray) -> np.ndarray:
    """Rotation angle (degrees) of rotation matrices given their traces."""
    return np.degrees(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)))


def closest_symmetric_copy(
    rotations_1: RotationLike,
    rotations_2: RotationLike,
    symmetry: Symmetry,
    *,
    chunk_size: int = 20000,
) -> tuple[np.ndarray, np.ndarray]:
    """Find, for each pair, the symmetric copy of rotation 2 closest to rotation 1.

    A particle with symmetry *symmetry* looks identical in orientations ``R``
    and ``R @ g`` for every group rotation ``g`` (template-side symmetry, as for
    particle-list angles; see :func:`reduce_angle_grid`). For each pair
    ``(R1, R2)`` this returns the smallest rotation angle between ``R1`` and
    any copy ``R2 @ g``, and which ``g`` achieves it. The angle of a rotation
    is read from its matrix trace, so no Euler angles are involved.

    Parameters
    ----------
    rotations_1 : RotationLike
        First rotation(s); normalized via :func:`cryocat.utils.geom.as_rotation`
        (Euler angles in degrees, ``zxz``).
    rotations_2 : RotationLike
        Second rotation(s), same number as *rotations_1* (compared pair by pair).
    symmetry : Symmetry
        Symmetry of the particle, in the canonical frame of this module (see
        :func:`reduce_angle_grid` Notes). Any group (C, D, T, O, I).
    chunk_size : int, optional
        Number of pairs processed at once (bounds memory use).

    Returns
    -------
    angle : numpy.ndarray
        Shape ``(N,)``. Smallest rotation angle in degrees, in [0, 180].
    copy_index : numpy.ndarray
        Shape ``(N,)``, int. Index into ``group.matrices`` (identity = 0) of the
        closest copy, i.e. the closest copy is ``R2 @ matrices[copy_index]``.

    Raises
    ------
    ValueError
        If the two inputs hold a different number of rotations.
    """
    mats_1 = geom.as_rotation(rotations_1).as_matrix().reshape(-1, 3, 3)
    mats_2 = geom.as_rotation(rotations_2).as_matrix().reshape(-1, 3, 3)
    if len(mats_1) != len(mats_2):
        raise ValueError(
            f"The inputs must hold the same number of rotations, got {len(mats_1)} and {len(mats_2)}."
        )
    g_mats = _make_group(symmetry).matrices
    rel = np.einsum("nji,njk->nik", mats_1, mats_2)  # R1^T @ R2

    n = len(rel)
    angle = np.empty(n)
    copy_index = np.empty(n, dtype=int)
    for start in range(0, n, chunk_size):
        # trace(rel @ g) for every pair and every group rotation
        copy_angles = _rotation_angles_deg(np.einsum("nij,gji->ng", rel[start : start + chunk_size], g_mats))
        copy_index[start : start + chunk_size] = copy_angles.argmin(axis=1)
        angle[start : start + chunk_size] = copy_angles.min(axis=1)
    return angle, copy_index


def reduce_angle_grid(
    rotations: RotationLike,
    symmetry: Symmetry,
    *,
    reference: RotationLike | None = None,
    local: bool = False,
    margin_deg: float = 0.0,
    tolerance_deg: float = 0.0,
    chunk_size: int = 20000,
) -> np.ndarray:
    """Select one orientation per set of symmetric look-alikes in an angle grid.

    A particle with symmetry *symmetry* looks identical in orientations ``R``
    and ``R @ g`` for every group rotation ``g`` (the template's content is
    turned by ``R``, as for particle-list angles). Searching both is wasted
    work; this function returns a mask keeping (about) one of each.

    Parameters
    ----------
    rotations : RotationLike
        The unreduced grid of orientations, e.g. from
        :func:`cryocat.utils.geom.generate_angles` with ``symmetry="C1"``
        (Euler angles in degrees, ``zxz``).
    symmetry : Symmetry
        Symmetry of the template, in the canonical frame of this module (see
        Notes). Any group (C, D, T, O, I).
    reference : RotationLike, optional
        Orientation around which the kept slice is centred (e.g. the starting
        orientation of a local search). Default is no rotation.
    local : bool, optional
        False (default) for a grid covering all orientations: keep the
        orientations in one slice of orientation space (fundamental zone)
        around *reference*. True for a grid covering only part of it (a local
        search): drop an orientation only if a symmetric copy of it, within
        *tolerance_deg*, is kept, so nothing that was searched is lost.
    margin_deg : float, optional
        Global mode: also keep orientations up to this angle beyond the edge
        of the slice. Grid points rarely line up across the slice's edges, so
        a margin of about half the sampling step keeps the coverage at the
        edges as good as in the unreduced grid. Default is 0.
    tolerance_deg : float, optional
        Local mode: how close (degrees) a kept symmetric copy must be for an
        orientation to be dropped. Default is 0 (only exact copies).
    chunk_size : int, optional
        Number of orientations processed at once (bounds memory use).

    Returns
    -------
    numpy.ndarray
        Boolean mask of length ``len(rotations)``; True = keep.

    Notes
    -----
    Canonical frames (identical to ChimeraX's default orientations, except
    D_n): C_n and D_n have the n-fold axis along z; D_n has its 2-fold axes
    in the xy-plane at ``90/n`` degrees from x (ChimeraX's D_n turned by
    ``90/n`` degrees about z); T has 2-fold axes along x, y, z (ChimeraX
    ``T`` ``222``); O has 4-fold axes along x, y, z; I has 2-fold axes along
    x, y, z with 5-fold axes in the yz-plane (ChimeraX ``I`` ``222``).
    """
    g_mats = _make_group(symmetry).matrices  # identity first
    mats = geom.as_rotation(rotations).as_matrix().reshape(-1, 3, 3)
    if reference is not None:
        ref = _as_single_rotation(reference).as_matrix()
        mats = np.einsum("ji,njk->nik", ref, mats)  # orientation relative to the reference
    n = len(mats)

    # For every orientation: rotation angle of each symmetric copy R @ g, and the closest copy.
    copy_angles_min = np.empty(n)
    own_angle = np.empty(n)
    closest = np.empty(n, dtype=int)
    for start in range(0, n, chunk_size):
        angles = _rotation_angles_deg(np.einsum("nij,gji->ng", mats[start : start + chunk_size], g_mats))
        own_angle[start : start + chunk_size] = angles[:, 0]
        copy_angles_min[start : start + chunk_size] = angles.min(axis=1)
        closest[start : start + chunk_size] = angles.argmin(axis=1)

    if not local:
        return own_angle <= copy_angles_min + margin_deg

    # Local: fold every orientation onto its copy closest to the reference, then
    # drop an orientation only if an earlier, kept one folds onto (almost) the same place.
    from scipy.spatial import cKDTree

    folded = np.einsum("nij,njk->nik", mats, g_mats[closest])
    quats = rot.from_matrix(folded).as_quat()
    quats *= np.where(quats[:, 3:4] < 0, -1.0, 1.0)  # q and -q are the same rotation
    radius = 2.0 * np.sin(np.radians(tolerance_deg) / 4.0) + 1e-12  # quaternion distance for that angle
    tree = cKDTree(np.vstack([quats, -quats]))
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if keep[i]:
            for j in tree.query_ball_point(quats[i], radius):
                j %= n
                if j > i:
                    keep[j] = False
    return keep
