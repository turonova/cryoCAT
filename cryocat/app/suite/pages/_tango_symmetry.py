"""Symmetry-control helpers for the Tango page (``ptango.py``).

Pure functions only — no Dash imports, no callbacks. They translate the state
of the Tango sidebar symmetry widgets (``tango-symm-type``,
``tango-c-symm-value``, ``tango-symm-kind``) into the canonical
``(symm, kind)`` arguments accepted by
:class:`cryocat.analysis.tango.TwistDescriptor`, where ``symm`` follows
:data:`cryocat._types.Symmetry` (``"T"``, ``"O"``, ``"I"`` or ``"CN"``) and
``kind`` names the specific solid for O/I.
"""
from __future__ import annotations

#: Options of the ``tango-symm-type`` dropdown. ``"None"`` is a string sentinel
#: (Dash dropdowns cannot hold a Python ``None`` option value cleanly).
SYMM_TYPE_OPTIONS: list[str] = ["None", "C", "T", "O", "I"]

#: Selectable solids per Platonic group letter; the first entry is the default.
#: "T" is absent on purpose: the tetrahedron is its only solid, so no choice
#: is offered and ``kind`` stays None.
_KIND_OPTIONS: dict[str, list[str]] = {
    "O": ["octahedron", "cube"],
    "I": ["icosahedron", "dodecahedron"],
}


def kind_options_for_symm(symm_type: str | None) -> list[str]:
    """Return the solid ("kind") choices offered for a symmetry-type selection.

    Parameters
    ----------
    symm_type : str or None
        Value of the ``tango-symm-type`` dropdown (one of
        :data:`SYMM_TYPE_OPTIONS`, or None before any selection).

    Returns
    -------
    list of str
        ``["octahedron", "cube"]`` for ``"O"``, ``["icosahedron",
        "dodecahedron"]`` for ``"I"``, and an empty list otherwise (no choice
        needed for None, cyclic or tetrahedral symmetry). The first entry is
        the default solid.
    """
    return list(_KIND_OPTIONS.get(symm_type or "", []))


def kind_control_state(symm_type: str | None) -> tuple[list[str], str | None, bool]:
    """Compute options, default value and visibility of the kind dropdown.

    Parameters
    ----------
    symm_type : str or None
        Value of the ``tango-symm-type`` dropdown.

    Returns
    -------
    options : list of str
        Choices for the ``tango-symm-kind`` dropdown (see
        :func:`kind_options_for_symm`).
    value : str or None
        Default selection — the first option, or None when there are none.
    show : bool
        True when the kind dropdown should be visible (only for ``"O"``/``"I"``).
    """
    options = kind_options_for_symm(symm_type)
    return options, (options[0] if options else None), bool(options)


def build_symm_kwargs(
    symm_type: str | None,
    c_symm_value: int | float | None,
    kind_value: str | None,
) -> dict[str, str | None]:
    """Translate the symmetry widgets into ``TwistDescriptor`` keyword arguments.

    Parameters
    ----------
    symm_type : str or None
        Value of the ``tango-symm-type`` dropdown: ``"None"`` (or None) for no
        symmetry, ``"C"`` for cyclic, ``"T"``/``"O"``/``"I"`` for Platonic.
    c_symm_value : int, float or None
        Value of the ``tango-c-symm-value`` input — the cyclic order N. Only
        read when ``symm_type == "C"``.
    kind_value : str or None
        Value of the ``tango-symm-kind`` dropdown. Only read when
        ``symm_type`` is ``"O"`` or ``"I"``; ignored otherwise (the dropdown
        may still hold a stale value while hidden). When None for O/I, the
        default solid (first entry of :func:`kind_options_for_symm`) is used.

    Returns
    -------
    dict
        ``{"symm": <Symmetry or None>, "kind": <str or None>}``, where
        ``symm`` is None, ``"CN"``, ``"T"``, ``"O"`` or ``"I"``, and ``kind``
        is None unless ``symm`` is ``"O"`` or ``"I"``.

    Raises
    ------
    ValueError
        If ``symm_type`` is not one of :data:`SYMM_TYPE_OPTIONS`, if the cyclic
        order is missing, non-integer or below 2, or if ``kind_value`` is not a
        valid solid for the chosen group.
    """
    if symm_type is None or symm_type == "None":
        return {"symm": None, "kind": None}
    if symm_type not in SYMM_TYPE_OPTIONS:
        raise ValueError(f"Unknown symmetry type {symm_type!r}; expected one of {SYMM_TYPE_OPTIONS}.")
    if symm_type == "C":
        if c_symm_value is None or float(c_symm_value) != int(c_symm_value) or int(c_symm_value) < 2:
            raise ValueError(f"C-symmetry value must be an integer >= 2, got {c_symm_value!r}.")
        return {"symm": f"C{int(c_symm_value)}", "kind": None}
    options = kind_options_for_symm(symm_type)
    if not options:
        return {"symm": symm_type, "kind": None}
    kind = kind_value or options[0]
    if kind not in options:
        raise ValueError(f"kind={kind!r} is not valid for symmetry {symm_type!r}; expected one of {options}.")
    return {"symm": symm_type, "kind": kind}
