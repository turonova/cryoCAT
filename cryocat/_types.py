"""Type aliases for cryoCAT.

Single source of truth for the polymorphic input/output types used across the
package. Goals:

1. Stop repeating ``str | Path | np.ndarray | ...`` unions in every signature.
2. Document the *intent* of a polymorphic input ("a map source", not just
   "a string or an ndarray").
3. Give the GUI form generator and the future agent layer a stable vocabulary:
   any parameter typed ``MapSource`` gets the same widget; any parameter typed
   ``MotlType`` is a dropdown of fixed options.

Aliases use the PEP 695 ``type`` statement (Python 3.12+), which evaluates
its right-hand side lazily.

Conventions
-----------
* ``XxxSource`` -- a polymorphic input that needs normalizing via a
  matching ``as_xxx()`` or ``xxx_load()`` helper at the function boundary.
* ``XxxLike`` -- a polymorphic input that is structurally one thing in
  several shapes (e.g. ``RotationLike``: Rotation, matrix, Euler triple).
* Plain ``Xxx`` -- a concrete shape with a well-defined dtype/shape contract.

Notes on runtime checks
-----------------------
These are static-typing constructs. ``isinstance(x, MapSource)`` does NOT
work and will raise. To check membership at runtime, test against the
concrete classes::

    if isinstance(source, np.ndarray):
        ...
    elif isinstance(source, (str, os.PathLike)):
        ...

------------------------------------------------------------------------
Directory: where every alias and normalizer lives
------------------------------------------------------------------------

Aliases defined IN THIS FILE
............................

Paths
    PathOrStr               -- str | os.PathLike

Generic arrays and lists
    ArrayLike               -- numpy.typing.ArrayLike re-export
    ListLike[T]            -- T | list[T] | tuple[T, ...]

Shapes and sizes
    TripletLike             -- int | float | 3-tuple | 3-list | 3-ndarray

Rotations and angles
    RotationLike            -- Rotation | matrix | quaternion | Euler
    EulerAngles             -- (3,) or (N, 3) array of degrees

3D map data
    MapSource               -- ndarray | PathOrStr
    TiltStack               -- ndarray | PathOrStr (2D stacks)

Per-tomogram metadata sources
    TomoDimensions          -- many forms; normalize with ioutils.dimensions_load
    TomoList                -- many forms; normalize with ioutils.tlt_load

Generic structured-data sources
    DataFrameSource         -- df | path | ndarray (for ioutils.df_load)
    DictSource              -- path | dict (for ioutils.dict_load)

Symmetry
    Symmetry                -- str ('C5') or int (5)

Literals (good for GUI dropdowns)
    MotlType                -- emmotl, relion, relion5, ...
    MotlColumn              -- the 20 fixed motl columns
    BoundaryType            -- center, whole
    CTFFileType             -- gctf, ctffind4, warp, relion

Generic
    FeatureName             -- str (looser than MotlColumn)
    ColumnNames             -- list[str] | None


Aliases defined ELSEWHERE (class-dependent; defined next to their class)
......................................................................

    Point3DLike             -> cryocat.utils.geom  (next to Point3D class)
    MotlSource              -> cryocat.core.cryomotl (next to Motl class)


Normalizers (boundary helpers that take a polymorphic input -> canonical form)
..............................................................................

    MapSource               -> cryocat.core.cryomap.read(x)
    TiltStack               -> cryocat.core.tiltstack helpers
    RotationLike            -> cryocat.utils.geom.as_rotation(x)
    Symmetry                -> cryocat.utils.geom.as_symmetry(x)
    TripletLike             -> cryocat.utils.geom.as_triplet(x)
    ListLike               -> cryocat.utils.classutils.as_list(x)
    MotlSource              -> cryocat.core.cryomotl.as_motl(x)
    TomoDimensions          -> cryocat.utils.ioutils.dimensions_load(x)
    TomoList                -> cryocat.utils.ioutils.tlt_load(x)
    DataFrameSource         -> cryocat.utils.ioutils.df_load(x)
    DictSource              -> cryocat.utils.ioutils.dict_load(x)
"""

from os import PathLike as _PathLike
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial.transform import Rotation


# ===========================================================================
# Paths
# ===========================================================================

type PathOrStr = str | _PathLike[str]
"""Anything you can pass to ``pathlib.Path()`` or ``open()``.

Covers plain strings, :class:`pathlib.Path`, and any object implementing
``__fspath__``. Normalize at the function boundary with ``Path(x)`` or
``os.fspath(x)``.
"""

# ===========================================================================
# Generic numpy arrays and lists
# ===========================================================================

type ArrayLike = npt.ArrayLike
"""Any object that can be coerced into a numpy ndarray.

Re-export of :data:`numpy.typing.ArrayLike`. Covers ndarrays, scalars,
nested lists/tuples of numbers, and anything else accepted by
``np.asarray``. Use when a parameter only needs to be array-coercible
and the specific shape/dtype is unimportant or documented separately in
the function's own docstring.

Normalize at the function boundary with ``np.asarray(x)`` (or
``np.asarray(x, dtype=...)`` to pin the dtype).
"""

type ListLike[T] = T | list[T] | tuple[T, ...]
"""A single value or a sequence of values of the same type.

PEP 695 generic alias. Use for parameters where the user can pass either
one item or many. Normalize at the boundary with
:func:`cryocat.utils.classutils.as_list`.

Examples
--------
    def crop(tomo_ids: ListLike[int]) -> ...:
    def load_masks(paths: ListLike[PathOrStr]) -> ...:
"""

# ===========================================================================
# Shapes, sizes, point arrays
# ===========================================================================

type _Num = int | float
type TripletLike = (
    _Num
    | tuple[_Num, _Num, _Num]
    | list[_Num]
    | npt.NDArray
)
"""Anything coercible to a length-3 ndarray.

A scalar (int or float) is broadcast to all three elements; a 3-tuple /
3-list / shape-``(3,)`` ndarray is used directly. Mixed-numeric inputs
like ``(64, 64.0, 64)`` are valid. Use for ``box_size``, ``mask_size``,
``new_size``, ``center``, ``shift``, ``radii``, and any other 3-element
quantity that can be specified compactly.

Normalize with :func:`cryocat.utils.geom.as_triplet`.
"""


# ===========================================================================
# Rotations and angles
# ===========================================================================

type RotationLike = Rotation | npt.NDArray | tuple[float, float, float] | list[float]
"""Anything coercible to a SciPy Rotation.

Accepted shapes:
    * :class:`scipy.spatial.transform.Rotation` -- passthrough
    * ``(3,)`` array/tuple/list -- Euler angles
    * ``(N, 3)`` ndarray -- stack of Euler angles
    * ``(3, 3)`` ndarray -- rotation matrix
    * ``(N, 3, 3)`` ndarray -- stack of rotation matrices
    * ``(4,)`` / ``(N, 4)`` ndarray -- quaternion(s)

Normalize with :func:`cryocat.utils.geom.as_rotation`.
"""

type EulerAngles = npt.NDArray[np.floating] | tuple[float, float, float] | list[float]
"""Euler angles in degrees.

Either a single triple (shape ``(3,)`` / 3-tuple / 3-list) or a stack
(shape ``(N, 3)``). Convention is normally zxz unless the function says
otherwise. Distinct from a generic 3-vector because the GUI/agent should
treat these as angles (degree spinners) rather than coordinates.
"""


# ===========================================================================
# Maps, masks, stacks (3D map data; ndarray-or-path)
# ===========================================================================

type MapSource = npt.NDArray | PathOrStr
"""A 3D map: ndarray or path to a map file (.mrc, .em, ...).

The single polymorphic input for any 3D map data -- tomograms, references,
reconstructions, density maps, and binary masks. The distinction between a
"volume" and a "mask" is purely semantic; both are loaded the same way and
share the same structural type.

Normalize with :func:`cryocat.core.cryomap.read`. Common parameter names:
``input_map``, ``mask``, ``input_mask``, ``ref``, ``dist_mask``.
"""

type TiltStack = npt.NDArray | PathOrStr
"""A 2D image stack: ndarray or path to a stack file (.mrc, .st, ...).

Used by :mod:`cryocat.core.tiltstack`. Distinct from :data:`MapSource`
because the semantics differ (tilt series vs. reconstruction) even though
the I/O is similar.
"""


# ===========================================================================
# Per-tomogram metadata sources (call a *_load helper to normalize)
# ===========================================================================

type TomoDimensions = (
    PathOrStr | pd.DataFrame | npt.NDArray | tuple[int, int, int] | list[int]
)
"""Per-tomogram x/y/z dimensions.

Normalize with :func:`cryocat.utils.ioutils.dimensions_load`.
"""

type TomoList = PathOrStr | int | list[int] | npt.NDArray[np.integer]
"""Identifiers of one or more tomograms.

Normalize with :func:`cryocat.utils.ioutils.tlt_load`.
"""



# ===========================================================================
# Generic structured-data sources (used by I/O helpers in ioutils.py)
# ===========================================================================

type DataFrameSource = pd.DataFrame | PathOrStr | npt.NDArray
"""Generic tabular input.

Normalize with :func:`cryocat.utils.ioutils.df_load`.
"""

type DictSource = PathOrStr | dict
"""Generic dict/JSON input.

Normalize with :func:`cryocat.utils.ioutils.dict_load`.
"""


# ===========================================================================
# Symmetry
# ===========================================================================

type Symmetry = str | int
"""Point-group symmetry specifier.

Accepts a string like ``"C5"``, ``"D2"`` or a bare integer ``5`` (which
is interpreted as cyclic Cn for that order).

Normalize with :func:`cryocat.utils.geom.as_symmetry`.
"""


# ===========================================================================
# String enums (Literal types) -- great for GUI dropdowns
# ===========================================================================

type MotlType = Literal[
    "emmotl", "relion", "relion5", "relion5_1", "stopgap", "dynamo", "mod"
]
"""Recognized motl file-format identifiers used by :meth:`Motl.load`."""

type MotlColumn = Literal[
    "score", "geom1", "geom2", "subtomo_id", "tomo_id", "object_id",
    "subtomo_mean", "x", "y", "z", "shift_x", "shift_y", "shift_z",
    "geom3", "geom4", "geom5", "phi", "psi", "theta", "class",
]
"""The fixed 20-column motl schema (see :attr:`Motl.motl_columns`).

Use this when a parameter must name one of the standard columns
(e.g. ``metric_id="score"``). For parameters that may also name a
user-added column, use :data:`FeatureName` instead.
"""

type BoundaryType = Literal["center", "whole"]
"""How a particle is considered to fall within a mask boundary.

* ``"center"`` -- only the particle center is checked
* ``"whole"`` -- the full subtomogram box must lie inside (requires box_size)
"""

type CTFFileType = Literal["gctf", "ctffind4", "warp", "relion"]
"""CTF estimation file format (input to :func:`defocus_load`)."""



# ===========================================================================
# Generic
# ===========================================================================

type FeatureName = str
"""Name of a Motl column used for grouping/filtering.

Looser than :data:`MotlColumn` because callers can group by user-added
columns. Use :data:`MotlColumn` when the value must be one of the 20
standard fields.
"""

type ColumnNames = list[str] | None
"""Optional column-name list for DataFrame construction.

Used by :func:`df_load` and similar helpers. ``None`` means "use the
default integer-index columns" or "the file already has a header row".
"""
