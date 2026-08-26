"""Class/introspection utilities + the annotation-driven type system.

This module is the single type authority shared by the Dash form generator
(``cryocat.app.formgen``) and the CLI (``cryocat.cli``):

* :func:`gui_exposed` — decorator marking a method as GUI-exposable and
  carrying presentation metadata.
* :func:`resolve_param_type` — maps a parameter *annotation* to a handler tag.
* :data:`TYPE_HANDLERS` — central table: tag -> {widget descriptor, GUI value
  parser, argparse spec}. ``widget`` values are plain string descriptors; the
  app realizes them into Dash components. **Nothing here imports Dash.**
"""

import importlib
import inspect
import sys
import typing
import types as _stdtypes
from collections.abc import Callable
from dataclasses import dataclass, field as _field
from enum import StrEnum
from typing import Any, Literal

from numpydoc.docscrape import NumpyDocString

from cryocat._types import ListLike


def filter_strings(
    input_list: list[str],
    filter_contains: str | list[str] | None = None,
    filter_exclude: str | list[str] | None = None,
) -> list[str]:
    """Filter a list of strings based on inclusion and exclusion criteria.

    Parameters
    ----------
    input_list : list of str
        The list of strings to filter.
    filter_contains : str or list of str, optional
        Substrings that must be present in the string for inclusion.
        If None, no inclusion filtering is applied.
    filter_exclude : str or list of str, optional
        Substrings that must be absent from the string for inclusion.
        If None, no exclusion filtering is applied.

    Returns
    -------
    list of str
        Filtered list of strings that meet the specified criteria.
    """
    # Normalize filters to lists
    if filter_contains is None:
        filter_contains = []
    elif isinstance(filter_contains, str):
        filter_contains = [filter_contains]

    if filter_exclude is None:
        filter_exclude = []
    elif isinstance(filter_exclude, str):
        filter_exclude = [filter_exclude]

    filtered = []
    for item in input_list:
        if filter_contains and not any(substr in item for substr in filter_contains):
            continue
        if any(substr in item for substr in filter_exclude):
            continue
        filtered.append(item)

    return filtered


def get_class_names_by_parent(
    parent_class_name: str,
    module_name: str,
    filter_contains: str | list[str] | None = None,
    filter_exclude: str | list[str] | None = None,
) -> list[str]:
    """Get class names that are subclasses of a specified parent class in a given module.

    Parameters
    ----------
    parent_class_name : str
        Name of the parent class to check against for subclasses.
    module_name : str
        Name of the module to inspect (e.g., 'my_module.submodule').
    filter_contains : str or list of str, optional
        Substrings that must be present in a class name to be included.
        If ``None``, no inclusion filtering is applied.
    filter_exclude : str or list of str, optional
        Substrings that must be absent from a class name to be included.
        If ``None``, no exclusion filtering is applied.

    Returns
    -------
    list of str
        A list of class names that are subclasses of the specified parent class,
        excluding the parent class itself and only including classes defined in
        the specified module.

    Raises
    ------
    ValueError
        If ``parent_class_name`` is not a valid class in the given module.
    """
    module = sys.modules.get(module_name)
    if module is None:
        module = importlib.import_module(module_name)

    # Get the actual class object from the module using its name
    parent_class = getattr(module, parent_class_name, None)
    if parent_class is None or not inspect.isclass(parent_class):
        raise ValueError(f"'{parent_class_name}' is not a valid class in module '{module_name}'.")

    # Now find all subclasses of parent_class
    class_names = []
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if issubclass(cls, parent_class) and cls is not parent_class and cls.__module__ == module.__name__:
            class_names.append(name)

    class_names = filter_strings(class_names, filter_contains=filter_contains, filter_exclude=filter_exclude)
    return class_names


def get_classes_from_names(
    class_names: str | type | list[str | type],
    module_name: str,
) -> type | list[type]:
    """Convert class names to actual class objects from a specified module.

    Parameters
    ----------
    class_names : str, type, or list of str/type
        Class names as strings or actual class objects. If strings, they will be
        looked up in the specified module. If class objects, they are returned as-is.
    module_name : str
        Name of the module where classes should be looked up (only used for string names).

    Returns
    -------
    type or list of type
        Class object(s) corresponding to the input names.
    """
    if not isinstance(class_names, list):
        if isinstance(class_names, str):
            module = importlib.import_module(module_name)
            return getattr(module, class_names)
        else:
            return class_names

    if all(isinstance(name, type) for name in class_names):  # list case
        # All items are already class objects
        return class_names
    else:
        module = importlib.import_module(module_name)
        return [getattr(module, name) for name in class_names]


def get_class_names_by_prefix(prefix: str) -> list[str]:
    """Get class names in the current module that start with a specified prefix.

    Parameters
    ----------
    prefix : str
        The prefix to filter class names.

    Returns
    -------
    list of str
        A list of class names that start with the given prefix and are defined in the current module.

    Examples
    --------
    >>> get_class_names_by_prefix('My')
    ['MyClass1', 'MyClass2']
    """

    current_module = sys.modules[__name__]
    class_names = [
        name
        for name, cls in inspect.getmembers(current_module, inspect.isclass)
        if name.startswith(prefix) and cls.__module__ == __name__
    ]
    return class_names


# ===========================================================================
# GuiCategory / GuiEntry — unified GUI registry types (Phase 4 / doc 4)
# ===========================================================================

class GuiCategory(StrEnum):
    """Closed set of GUI tiers.  Extend only when a third tier is genuinely needed."""
    MOTL_OP = "motl-op"
    BUILDER  = "builder"
    READER   = "reader"   # file-reading callables surfaced in the data pool


@dataclass(frozen=True)
class GuiEntry:
    """Immutable descriptor for one ``@gui_exposed`` callable.

    The ``key`` is the stable registry identifier derived from owner + name
    (e.g. ``"Motl.clean_by_distance"``, ``"geom.generate_angles"``).
    ``fn`` and ``motls`` are excluded from ``__hash__`` so the entry can be
    stored in sets/dict-keys even though ``fn`` is a callable and ``motls``
    is a mutable dict.
    """
    key:      str
    fn:       Callable        = _field(hash=False)
    label:    str             = ""
    category: GuiCategory     = GuiCategory.MOTL_OP
    owner:    str             = ""
    kind:     str             = "function"
    group:    str             = ""
    order:    int             = 100
    hide:     frozenset[str]  = _field(default_factory=frozenset)
    output:   str | None      = None
    motls:    dict | None     = _field(default=None, hash=False)
    standalone: bool          = False
    preview:  str | None      = None
    returns:  str | None      = None
    path_arg:   str             = ""           # name of the path parameter (hidden from form)
    extensions: tuple[str, ...] = ()           # accepted file extensions for the path field


GUI_REGISTRY: dict[str, GuiEntry] = {}
"""Single source of truth for every ``@gui_exposed`` callable.

Populated at decoration time as each annotated module is imported.
Read via :mod:`cryocat.app.discovery` — never walk ``inspect.getmembers``
or read this dict directly in app code.
"""

_VALID_MOTLS_KEYS: frozenset[str] = frozenset({"arity", "ordered", "main_first", "param"})


# ===========================================================================
# @gui_exposed — mark a method as GUI-exposable + carry presentation metadata
# ===========================================================================


def _infer_returns(fn: Callable) -> str | None:
    """Infer the 'returns' dispatch kind from the return-type annotation.

    Returns ``None`` when the annotation is absent or ambiguous (e.g. a tuple
    or ndarray) — the caller must supply an explicit ``returns=`` override.
    """
    try:
        sig = inspect.signature(fn)
        ret = sig.return_annotation
    except Exception:
        return None
    if ret is inspect.Parameter.empty:
        return None
    if ret is None or ret is type(None):
        return "none"
    # Forward-reference string (e.g. "cryomotl.Motl")
    if isinstance(ret, str):
        if ret in ("None", "none"):
            return "none"
        if "Motl" in ret:
            return "motl"
        if "DataFrame" in ret:
            return "dataframe"
        return None
    # list[Motl] or list["Motl"] -> "motl_group"
    import types as _types
    origin = typing.get_origin(ret)
    if origin is list:
        args = typing.get_args(ret)
        if args:
            arg_str = str(args[0])
            if "Motl" in arg_str:
                return "motl_group"
    # Actual type object
    name = getattr(ret, "__name__", "")
    if name in ("Motl", "EmMotl", "RelionMotl", "RelionMotlv5",
                "RelionMotlv5_1", "StopgapMotl", "DynamoMotl", "ModMotl"):
        return "motl"
    if name == "DataFrame":
        return "dataframe"
    # Fallback: string representation for e.g. union types
    ret_str = str(ret)
    if "Motl" in ret_str and "tuple" not in ret_str and "list" not in ret_str:
        return "motl"
    return None


def gui_exposed(
    _fn: Callable | None = None,
    *,
    label: str | None = None,
    category: str | None = None,
    group: str = "",
    order: int = 100,
    hide: tuple[str, ...] = (),
    output: str | None = None,
    motls: dict | None = None,
    standalone: bool = False,
    preview: str | None = None,
    returns: str | None = None,
    path_arg: str = "",
    extensions: tuple[str, ...] = (),
):
    """Mark a callable as GUI-exposable and register it in :data:`GUI_REGISTRY`.

    Parameters
    ----------
    label : str, optional
        Display name in the operation dropdown. Defaults to the function name.
    category : str, optional
        Grouping for the dropdown (e.g. ``"Cleaning"``, ``"Geometry"``).
        Use ``"builder"`` for standalone value-producing functions.
        ``None`` = ungrouped motl-op.
    hide : tuple of str, optional
        Parameter names the GUI/CLI should NOT surface (beyond ``self`` /
        ``cls``, which are always hidden).
    output : str or None, optional
        What the callable returns — ``"motl"``, ``"figure"``,
        ``"dataframe"``, ``"map"``, etc.  Used by the GUI to route the result.
    motls : dict, optional
        Marks a MULTI-motl operation.  Required key ``"arity"``:
        ``"pair"`` (exactly 2) or ``"list"`` (N >= 2).  Optional keys:
        ``"ordered"``, ``"main_first"``, ``"param"``.
        ``None`` (default) => single-motl operation.
    standalone : bool, default ``False``
        When ``True`` and ``category="builder"``, the function is listed on
        the Utilities page as its own panel.
    preview : str, optional
        Preview plot style string interpreted by the builder panel component.

    Notes
    -----
    Malformed metadata (bad motls spec, empty label) raises :exc:`ValueError`
    **at decoration time**, naming the offending callable.

    A bare ``@gui_exposed`` (no parentheses) is accepted.  Stack
    ``@classmethod`` **below** so ``_gui`` lands on the underlying function
    and the collector can read it through ``__func__``.

    Re-decorating the same key with identical essential metadata is a no-op
    (safe for hot-reload / module reimport).  Conflicting metadata raises
    :exc:`RuntimeError`.
    """

    def wrap(fn):
        # Unwrap classmethod/staticmethod so we can read qualname/module.
        if isinstance(fn, classmethod):
            target = fn.__func__
            kind = "classmethod"
        elif isinstance(fn, staticmethod):
            target = fn.__func__
            kind = "staticmethod"
        else:
            target = fn
            params_list = list(inspect.signature(target).parameters.keys())
            if params_list and params_list[0] == "self":
                kind = "method"
            elif params_list and params_list[0] == "cls":
                kind = "classmethod"
            else:
                kind = "function"

        gui_label = label or target.__name__.replace("_", " ").capitalize()
        if not gui_label:
            raise ValueError(
                f"@gui_exposed on {target.__qualname__!r}: label must be non-empty"
            )

        # Validate motls spec eagerly.
        if motls is not None:
            if "arity" not in motls:
                raise ValueError(
                    f"@gui_exposed on {target.__qualname__!r}: motls spec missing 'arity'"
                )
            if motls["arity"] not in ("pair", "list"):
                raise ValueError(
                    f"@gui_exposed on {target.__qualname__!r}: "
                    f"motls['arity'] must be 'pair' or 'list', got {motls['arity']!r}"
                )
            unknown = set(motls) - _VALID_MOTLS_KEYS
            if unknown:
                raise ValueError(
                    f"@gui_exposed on {target.__qualname__!r}: "
                    f"unknown motls keys {unknown}"
                )

        # Derive tier and display group.
        # `group` (explicit) takes precedence; fall back to `category` for legacy callers.
        if category == "builder":
            gui_category = GuiCategory.BUILDER
            gui_group: str = group  # explicit group name, or "" for ungrouped
        elif category == "reader":
            gui_category = GuiCategory.READER
            gui_group = group
        else:
            gui_category = GuiCategory.MOTL_OP
            gui_group = group or (category or "")

        # Derive stable key and owner from qualname / module.
        qualname = target.__qualname__
        module   = target.__module__ or ""
        if "<locals>" in qualname:
            # Nested / closure functions (test helpers, lambdas).
            # Use the sanitized full qualname so every such function is unique.
            sanitized = qualname.replace("<locals>.", "").replace("<locals>", "")
            mod_short = module.rsplit(".", 1)[-1] if module else ""
            owner_str = module
            reg_key   = f"{mod_short}.{sanitized}" if mod_short else sanitized
        elif "." in qualname:
            class_part  = qualname.rsplit(".", 1)[0]
            short_class = class_part.split(".")[-1]
            owner_str   = f"{module}.{class_part}" if module else class_part
            reg_key     = f"{short_class}.{target.__name__}"
        else:
            mod_short = module.rsplit(".", 1)[-1] if module else ""
            owner_str = module
            reg_key   = f"{mod_short}.{target.__name__}" if mod_short else target.__name__

        returns_val = returns if returns is not None else _infer_returns(target)

        entry = GuiEntry(
            key=reg_key,
            fn=target,
            label=gui_label,
            category=gui_category,
            owner=owner_str,
            kind=kind,
            group=gui_group,
            order=order,
            hide=frozenset(hide),
            output=output,
            motls=motls,
            standalone=standalone,
            preview=preview,
            returns=returns_val,
            path_arg=path_arg,
            extensions=tuple(extensions),
        )

        existing = GUI_REGISTRY.get(reg_key)
        if existing is not None:
            if existing.label != gui_label or existing.category != gui_category:
                raise RuntimeError(
                    f"@gui_exposed: conflicting registration for {reg_key!r}; "
                    f"existing label={existing.label!r} category={existing.category!r}, "
                    f"new label={gui_label!r} category={gui_category!r}"
                )
        else:
            GUI_REGISTRY[reg_key] = entry

        # Keep fn._gui for backward compatibility (formgen.build_form reads it).
        target._gui = {
            "label": gui_label,
            "category": category,
            "hide": set(hide) | {"self", "cls"},
            "output": output,
            "motls": motls,
            "standalone": standalone,
            "preview": preview,
        }
        return fn

    return wrap(_fn) if _fn is not None else wrap


# ===========================================================================
# Annotation -> handler-tag resolver
# ===========================================================================

# PEP-695 ``type X = ...`` aliases handled directly by tag (== alias name).
_ALIAS_TAGS = {
    "MapSource", "DataSource", "TiltStack", "TomoList", "TomoDimensions",
    "TripletLike", "EulerAngles", "ListLike", "Symmetry", "ArrayLike",
    "RotationLike", "PathOrStr",
}
# PEP-695 aliases whose value is a Literal[...] — resolved to ("Literal", choices).
_LITERAL_ALIASES = {
    "MotlType", "MotlColumn", "BoundaryType", "CTFFileType", "NNType",
    "RotationDistanceType", "ProjectionType", "WedgeMaskMethod",
}


def resolve_param_type(annotation: Any) -> tuple[str, dict]:
    """Map a parameter annotation to a handler tag + extras.

    Parameters
    ----------
    annotation : Any
        Python type annotation (a class, a PEP-695 ``type X = ...`` alias, a
        ``Literal[...]``, an ``X | None`` / ``X | None`` union, or
        :data:`inspect.Parameter.empty` for an unannotated parameter).

    Returns
    -------
    tuple of (str, dict)
        ``(tag, extra)``; ``extra`` carries e.g. ``{"choices": [...]}`` for
        Literals or ``{"length": int, "elem": str}`` for fixed-length numeric
        tuples.

    Notes
    -----
    Resolution rules:

    * ``None`` / empty annotation        -> ``("str", {})``
    * ``X | None`` / ``X | None``     -> unwrap to ``X``, then resolve
    * ``Literal[...]``                   -> ``("Literal", {"choices": [...]})``
    * a PEP-695 alias in the known set   -> ``(alias name, {})``
    * bare ``bool``/``int``/``float``/``str`` -> ``(name, {})``
    * anything else                      -> ``("str", {})``
    """
    if annotation is inspect.Parameter.empty or annotation is None:
        return ("str", {})

    origin = typing.get_origin(annotation)

    # X | None / X | None — unwrap NoneType and resolve the remainder.
    if origin is typing.Union or origin is _stdtypes.UnionType:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return resolve_param_type(args[0])
        for a in args:  # several non-None members: take the first recognised one
            tag, extra = resolve_param_type(a)
            if tag != "str":
                return (tag, extra)
        return ("str", {})

    # Literal[...] used directly.
    if origin is typing.Literal:
        return ("Literal", {"choices": list(typing.get_args(annotation))})

    # tuple[float, ...] / tuple[float, float] / tuple[int, int] — fixed-length
    # numeric tuple. We surface this as a composite "Tuple" widget so the form
    # renders one number field per slot and the parser builds a python tuple.
    # The element type and length come from the annotation's args.
    if origin in (tuple,):
        args = list(typing.get_args(annotation))
        # Reject heterogeneous / Ellipsis tuples — they'd need a different
        # widget. Fall through to str for now (so behavior is unchanged for
        # anything we don't explicitly support).
        if args and Ellipsis not in args and len(set(args)) == 1 and args[0] in (int, float):
            return ("Tuple", {"length": len(args), "elem": args[0].__name__})

    # Subscripted PEP-695 alias, e.g. ListLike[int] — origin is the alias itself.
    if isinstance(origin, typing.TypeAliasType) and origin.__name__ in _ALIAS_TAGS:
        alias_name = origin.__name__
        if alias_name == "ListLike":
            # Carry the element type tag so the widget factory can branch.
            args = typing.get_args(annotation)
            if args:
                elem_tag, _ = resolve_param_type(args[0])
            else:
                elem_tag = "str"
            return ("ListLike", {"elem_tag": elem_tag})
        return (alias_name, {})

    # Bare PEP-695 alias, e.g. MapSource / Symmetry / MotlType.
    if isinstance(annotation, typing.TypeAliasType):
        name = annotation.__name__
        if name in _ALIAS_TAGS:
            return (name, {})
        if name in _LITERAL_ALIASES:
            choices = list(typing.get_args(annotation.__value__))
            return ("Literal", {"choices": choices})
        # Unknown alias — fall back to resolving its underlying value.
        return resolve_param_type(annotation.__value__)

    # Bare builtins.
    if annotation in (bool, int, float, str):
        return (annotation.__name__, {})

    return ("str", {})


# ===========================================================================
# Value parsers (GUI value -> python) and argparse helpers (CLI str -> python)
# ===========================================================================

def _coerce_scalar(x: str) -> int | float | str:
    """Best-effort int -> float -> str coercion of a single token."""
    x = x.strip()
    try:
        return int(x)
    except ValueError:
        pass
    try:
        return float(x)
    except ValueError:
        return x


def _parse_number_list(v: Any) -> Any:
    """``"1,2,3"`` -> ``[1, 2, 3]`` (per-element int/float autodetect);
    a single token -> a scalar. Non-string input is returned unchanged."""
    if v is None or v == "":
        return None
    if not isinstance(v, str):
        return v
    parts = [p for p in (tok.strip() for tok in v.split(",")) if p != ""]
    if not parts:
        return None
    vals = [_coerce_scalar(p) for p in parts]
    return vals[0] if len(vals) == 1 else vals


def _parse_path(v: Any) -> str | None:
    return v or None


def _parse_bool(v: Any) -> bool:
    return v in (True, "True", "true", "1", 1)


def _parse_int(v: Any) -> int | None:
    """GUI number field -> int. Empty/None -> None; floats and numeric strings
    are coerced via float() to tolerate things like ``"33.0"``."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return None
    return int(float(v))


def _parse_float(v: Any) -> float | None:
    """GUI number field -> float, with empty/None tolerated."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return None
    return float(v)


def _parse_triplet(v: Any) -> Any:
    """A GUI triplet field (``"64,64,64"`` or ``"64"``). The receiving function
    normalizes with :func:`cryocat.utils.geom.as_triplet`."""
    return _parse_number_list(v)


def _parse_listlike(v: Any) -> Any:
    """A GUI csv/text field (``"1,2,3"``). The receiving function normalizes
    with :func:`cryocat.utils.classutils.as_list`."""
    return _parse_number_list(v)


def _parse_literal(v: Any, choices: list | None = None) -> Any:
    """A GUI dropdown value. If the matching choice is non-string typed,
    coerce ``v`` to that choice's type."""
    if v is None or not choices:
        return v
    for c in choices:
        if str(c) == str(v):
            return c
    return v


def _parse_tuple(v: Any, elem: str = "float") -> tuple | None:
    """Composite fixed-length numeric tuple field.

    The formgen widget renders one number input per slot and stores the slots
    as a list under a single id (see :func:`cryocat.app.formgen._tuple_field`).
    This parser coerces that list to a Python tuple with the requested element
    type. ``None`` / empty / mismatched-length payloads return ``None`` so the
    function falls back to its default.
    """
    if v is None or v == "":
        return None
    if not isinstance(v, (list, tuple)) or len(v) == 0:
        return None
    coerced = []
    cast = float if elem == "float" else int
    for item in v:
        if item is None or item == "":
            return None
        try:
            coerced.append(cast(item))
        except (TypeError, ValueError):
            return None
    return tuple(coerced)


# argparse ``type=`` helpers (CLI string -> python value).
def _parse_str(v: Any) -> str | None:
    """GUI text field -> str. None or empty string stays None."""
    if v is None or v == "":
        return None
    return str(v)


def _arg_bool(s: str) -> bool:
    return _parse_bool(s)


def _arg_triplet(s: str) -> Any:
    return _parse_number_list(s)


def _arg_listlike(s: str) -> Any:
    return _parse_number_list(s)


# ===========================================================================
# The central type -> handler table
# ===========================================================================
# Each entry: widget descriptor (string, never a Dash object) | GUI value parser
# | argparse spec for add_argument. The app's formgen maps the widget string to
# an actual component; render and parse both read this one table so they cannot
# drift.
TYPE_HANDLERS = {
    # tag             widget        parse (GUI value -> py)   argparse spec
    "MapSource":      {"widget": "path",     "parse": _parse_path,     "argparse": {"type": str}},
    "DataSource":     {"widget": "path",     "parse": _parse_path,     "argparse": {"type": str}},
    "TiltStack":      {"widget": "path",     "parse": _parse_path,     "argparse": {"type": str}},
    "TomoDimensions": {"widget": "path",     "parse": _parse_path,     "argparse": {"type": str}},
    "TomoList":       {"widget": "text",     "parse": _parse_listlike, "argparse": {"type": _arg_listlike}},
    "ArrayLike":      {"widget": "csv_text", "parse": _parse_listlike, "argparse": {"type": _arg_listlike}},
    "TripletLike":    {"widget": "triplet",  "parse": _parse_triplet,  "argparse": {"type": _arg_triplet}},
    "EulerAngles":    {"widget": "triplet",  "parse": _parse_triplet,  "argparse": {"type": _arg_triplet}},
    "ListLike":       {"widget": "listlike", "parse": _parse_listlike, "argparse": {"type": _arg_listlike}},
    "PathOrStr":      {"widget": "path",     "parse": _parse_path,     "argparse": {"type": str}},
    "Symmetry":       {"widget": "text",     "parse": _parse_str,      "argparse": {"type": str}},
    "RotationLike":   {"widget": "rotation", "parse": _parse_str,      "argparse": {"type": str}},
    "Literal":        {"widget": "dropdown", "parse": _parse_literal,  "argparse": {"type": str}},
    "Tuple":          {"widget": "tuple",    "parse": _parse_tuple,    "argparse": {"type": _arg_listlike}},
    "bool":           {"widget": "bool",     "parse": _parse_bool,     "argparse": {"type": _arg_bool}},
    "int":            {"widget": "number",   "parse": _parse_int,      "argparse": {"type": int}},
    "float":          {"widget": "number",   "parse": _parse_float,    "argparse": {"type": float}},
    "str":            {"widget": "text",     "parse": _parse_str,      "argparse": {"type": str}},
}


# ===========================================================================
# Docstring -> parameter descriptions (help text / tooltips only)
# ===========================================================================

def _clean_desc(text: str) -> str:
    """Strip reST cross-reference role markers from help text."""
    for role in (":meth:", ":func:", ":class:", ":data:", ":attr:", ":mod:"):
        text = text.replace(role, "")
    return text.strip()


def process_method_docstring(
    path_to_method: Any,
    method_name: str | None = None,
) -> dict[str, str]:
    """Extract parameter *descriptions* from a method's numpy-style docstring.

    Types, required/default, and choices are NOT taken from the docstring any
    more — those come from the function signature + :func:`resolve_param_type`.
    This supplies help text / tooltips only.

    Parameters
    ----------
    path_to_method : module, class, or callable
        Either a module/class containing the method (then pass ``method_name``),
        or the callable itself (then leave ``method_name`` as None).
    method_name : str, optional
        Name of the method to process. When None, ``path_to_method`` is treated
        as the callable directly.

    Returns
    -------
    dict of str -> str
        ``{param_name: description}``. Empty when the docstring has no
        Parameters section.
    """
    if method_name is None:
        method_obj = path_to_method
    else:
        method_obj = inspect.getattr_static(path_to_method, method_name)
    docstring = inspect.getdoc(method_obj)
    if not docstring:
        return {}

    np_doc = NumpyDocString(docstring)
    descriptions = {}
    for p in np_doc["Parameters"]:
        # NumpyDocString param: (name, type_desc, description_lines). The name
        # field can be "name : type" or "name: type"; keep only the name.
        raw_name = p[0].split(":")[0].strip()
        descriptions[raw_name] = _clean_desc(" ".join(p[2]))
    return descriptions


def as_list[T](x: ListLike[T]) -> list[T]:
    """Wrap a scalar in a list, or convert a sequence to a list.

    Strings and bytes are treated as scalars (wrapped, not iterated).
    Existing lists are returned without copying.

    Examples
    --------
    >>> as_list(5)
    [5]
    >>> as_list([1, 2, 3])
    [1, 2, 3]
    >>> as_list("hello")
    ['hello']
    >>> as_list((1, 2, 3))
    [1, 2, 3]
    """
    if isinstance(x, list):
        return x
    if isinstance(x, (str, bytes)) or not hasattr(x, "__iter__"):
        return [x]
    return list(x)
