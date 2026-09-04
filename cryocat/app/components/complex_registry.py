"""Server-side registry for live complex instances (SymmetricComplex / PolyhedralComplex).

Complex objects hold large in-memory data structures that cannot be serialised
into a ``dcc.Store``.  This module keeps them server-side and exposes lightweight
JSON handles for Dash to carry across callbacks.

Public API
----------
* :data:`registry` — ``Registry`` instance; ids are ``complex-1``, ``complex-2``, …
* :class:`ComplexHandle` — the dcc.Store-safe handle dataclass.
* :func:`make_handle` — build a handle from a live complex instance.
* :func:`reconstruct` — rebuild the live complex from a handle + a pool motl.
"""
from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from cryocat.app.components.registry import Registry

if TYPE_CHECKING:
    from cryocat.analysis.structure import SymmetricComplex

# 1-based ids match the motl pool convention.
registry: Registry = Registry("complex", start=1)


@dataclass
class ComplexHandle:
    """Lightweight, JSON-serialisable handle stored in a dcc.Store pool."""

    complex_id: str          # "complex-1"
    label: str
    cls: str                 # "IcosahedralComplex"
    symmetry: str            # e.g. "I", "C8", "D6"; empty for polyhedral subclasses
    n_subunits: int
    n_objects: int
    motl_links: dict         # {"source": "motl_3"} — provenance link
    affiliation_column: str
    order_column: str
    tomo_id_column: str
    geometry_fitted: bool = False    # True when fit_geometry has been called
    radius: float | None = None      # circumscribed-sphere radius in voxels
    init_kwargs: dict = field(default_factory=dict)  # extra init params (JSON-safe)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ComplexHandle":
        return cls(**d)


def make_handle(
    cpx,
    complex_id: str,
    label: str,
    motl_links: dict,
    init_kwargs: dict,
) -> dict:
    """Build the dcc.Store-safe handle dict for a live complex instance.

    Parameters
    ----------
    cpx:
        The live complex (SymmetricComplex / PolyhedralComplex subclass).
    complex_id:
        The stable key returned by :meth:`Registry.add`.
    label:
        Human-readable label shown in the complex list.
    motl_links:
        Role-keyed motl links, e.g. ``{"source": "motl_3"}``.
    init_kwargs:
        Extra constructor keyword arguments (beyond ``motl``), JSON-serialisable.
    """
    cls_name = type(cpx).__name__
    symmetry = getattr(cpx, "fold", "") or ""
    group = getattr(cpx, "group", "")
    if group and symmetry:
        symmetry_str = f"{group}{symmetry}"
    elif symmetry:
        symmetry_str = str(symmetry)
    else:
        # PolyhedralComplex: read _symmetry class attr
        symmetry_str = getattr(type(cpx), "_symmetry", "") or ""

    n_subunits = int(getattr(cpx, "n_subunits", 0) or 0)

    try:
        tomo_col = getattr(cpx, "tomo_id_column", "tomo_id") or "tomo_id"
        aff_col = getattr(cpx, "affiliation_column", "object_id") or "object_id"
        n_objects = int(cpx.motl.df[tomo_col].nunique()) if hasattr(cpx, "motl") else 0
    except Exception:
        n_objects = 0

    solid = getattr(cpx, "solid", None)
    geometry_fitted = solid is not None
    radius = float(solid.radius) if geometry_fitted else None

    handle = ComplexHandle(
        complex_id=complex_id,
        label=label,
        cls=cls_name,
        symmetry=symmetry_str,
        n_subunits=n_subunits,
        n_objects=n_objects,
        motl_links=motl_links,
        affiliation_column=getattr(cpx, "affiliation_column", "object_id") or "object_id",
        order_column=getattr(cpx, "order_column", "geom1") or "geom1",
        tomo_id_column=getattr(cpx, "tomo_id_column", "tomo_id") or "tomo_id",
        geometry_fitted=geometry_fitted,
        radius=radius,
        init_kwargs=init_kwargs,
    )
    return handle.to_dict()


def reconstruct(handle_dict: dict, motl: Any) -> Any:
    """Rebuild the live complex from a handle dict + a Motl instance.

    Used when the server-side registry has been cleared (e.g. hot-reload)
    and the complex must be re-created from the persisted handle.

    Parameters
    ----------
    handle_dict:
        A dict previously produced by :func:`make_handle`.
    motl:
        The pool motl (already loaded as a :class:`~cryocat.core.cryomotl.Motl`).
    """
    handle = ComplexHandle.from_dict(handle_dict)
    mod = importlib.import_module("cryocat.analysis.structure")
    cls = getattr(mod, handle.cls)
    kwargs: dict[str, Any] = {
        "affiliation_column": handle.affiliation_column,
        "order_column": handle.order_column,
        "tomo_id_column": handle.tomo_id_column,
        **handle.init_kwargs,
    }
    # PolyhedralComplex subclasses do not accept a symmetry argument.
    if handle.symmetry:
        base_names = {"PolyhedralComplex", "TetrahedralComplex",
                      "OctahedralComplex", "IcosahedralComplex"}
        if handle.cls not in base_names:
            kwargs["symmetry"] = handle.symmetry
    return cls(motl, **kwargs)
