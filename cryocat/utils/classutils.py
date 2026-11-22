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

import argparse
import ast
import importlib
import inspect
import pydoc
import re
import sys
import textwrap
import typing
import types as _stdtypes
from collections.abc import Callable
from dataclasses import dataclass, field as _field
from enum import StrEnum
from functools import lru_cache
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
    READER      = "reader"       # file-reading callables surfaced in the data pool
    SURFACE_OP  = "surface-op"   # operations on existing Mesh / OPC surfaces
    SURFACE_LOAD = "surface-load" # loaders that create a new surface from disk / motl
    PLEOMORPHIC_OP = "pleomorphic-op"  # operations on PleomorphicAssembly assemblies


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
        elif category == "surface-op":
            gui_category = GuiCategory.SURFACE_OP
            gui_group = group
        elif category == "surface-load":
            gui_category = GuiCategory.SURFACE_LOAD
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
# @cli_exposed — mark a callable as a command-line subcommand
# ===========================================================================

_CLI_OUTPUTS: frozenset[str | None] = frozenset({None, "motl", "csv", "map"})


def cli_exposed(
    _fn: Callable | None = None,
    *,
    hide: tuple[str, ...] = (),
    output: str | None = None,
    required: tuple[str, ...] = (),
):
    """Mark a callable as a subcommand of the command line of its module (:mod:`cryocat.cli`).

    Independent of :func:`gui_exposed`: a callable is on the command line
    only if it carries this decorator.  The callable is returned unchanged;
    the metadata is stored on it as ``_cli``.

    Parameters
    ----------
    hide : tuple of str, optional
        Parameter names the command line does not offer (beyond ``self`` and
        ``cls``, which are always hidden).
    output : {None, "motl", "csv", "map"}, optional
        What the command line writes after the call.  ``None`` writes
        nothing: the callable writes through its own parameter, or prints.
        ``"motl"`` adds a required ``--output_motl`` and writes the returned
        motl there, or, for a method that returns ``None``, the motl it
        changed in place.  ``"csv"`` adds a required ``--output_csv`` and
        writes the returned DataFrame with ``to_csv(index=False)``.  ``"map"``
        adds a required ``--output_map`` and writes the returned array with
        :func:`cryocat.core.cryomap.write`.
    required : tuple of str, optional
        Parameter names the command line offers as required although their
        signature has a default.

    Notes
    -----
    An unknown *output*, or a name in *required* that is not a parameter of
    the callable, raises :exc:`ValueError` at decoration time, naming the
    callable.  A bare ``@cli_exposed`` (no parentheses) is accepted.
    Stack ``@classmethod`` / ``@staticmethod`` **below** so ``_cli`` lands on
    the underlying function, as with :func:`gui_exposed`.
    """

    def wrap(fn):
        target = fn.__func__ if isinstance(fn, (classmethod, staticmethod)) else fn
        if output not in _CLI_OUTPUTS:
            raise ValueError(
                f"@cli_exposed on {target.__qualname__!r}: output must be one of None, 'motl', 'csv', 'map'; got {output!r}"
            )
        unknown = set(required) - set(inspect.signature(target).parameters)
        if unknown:
            raise ValueError(
                f"@cli_exposed on {target.__qualname__!r}: required names {sorted(unknown)} are not parameters"
            )
        target._cli = {"hide": set(hide) | {"self", "cls"}, "output": output, "required": frozenset(required)}
        return fn

    return wrap(_fn) if _fn is not None else wrap


# ===========================================================================
# Annotation -> handler-tag resolver
# ===========================================================================

# PEP-695 ``type X = ...`` aliases handled directly by tag (== alias name).
_ALIAS_TAGS = {
    "MapSource", "DataSource", "DataPoolEntry", "MotlSource", "TiltStack", "TomoList", "TomoDimensions",
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

    # list[MotlSource] → multi-select pool-motl picker.
    if origin is list:
        args = typing.get_args(annotation)
        if args and isinstance(args[0], typing.TypeAliasType) and args[0].__name__ == "MotlSource":
            return ("MotlSource", {"multi": True})

    # Bare builtins.
    if annotation in (bool, int, float, str):
        return (annotation.__name__, {})

    return ("str", {})


# ===========================================================================
# Derived parameters — options read from signatures, docstrings and source
# ===========================================================================
#
# Used by the shared motl IO of cryocat.core.cryomotl and by cryocat.cli to
# offer the parameters of constructors and writers, and of the functions a
# callable passes ``**kwargs`` on to, without listing them by hand.


@dataclass(frozen=True)
class DerivedParameter:
    """One parameter of a function or class, with a type derived for it.

    ``source`` says where ``annotation`` came from: ``"annotation"``,
    ``"docstring"`` (the numpydoc type), ``"default"`` (the type of a non-None
    default) or ``"text"`` (nothing gave a type).
    """

    name: str
    annotation: Any
    default: Any
    description: str
    source: str
    owner: Callable


@dataclass(frozen=True)
class ForwardedKeywords:
    """A parameter a function passes on as ``**name``, where it goes, and the options this gives.

    ``targets`` are the functions or classes the function's own calls pass it
    to; ``chained`` the functions reached after them, through a single target
    that passes its own ``**kwargs`` on.  ``unresolved`` says why the chain
    stops before a function with an explicit signature, or is None.
    """

    name: str
    targets: tuple[Callable, ...]
    chained: tuple[Callable, ...]
    options: dict[str, DerivedParameter]
    unresolved: str | None


def source_tree(fn: Callable) -> ast.Module:
    """The :mod:`ast` tree of the source of *fn* (a function, method or class)."""
    return ast.parse(textwrap.dedent(inspect.getsource(fn)))


def _namespace(obj: Any) -> dict:
    """The globals of the module that defines *obj*."""
    return vars(sys.modules[obj.__module__])


def documented_type(doc_type: str, namespace: dict) -> Any:
    """The type a numpydoc entry names (``float, optional`` → ``float``), a union for ``a or b``, or None.

    Names are evaluated in *namespace*, dotted names (``pandas.DataFrame``)
    located by import path; a name that is neither (``array-like``) is left out.
    """
    spec = re.sub(r",\s*(optional|default\s*[=:].*)$", "", doc_type.strip())
    spec = re.sub(r",\s*(optional|default\s*[=:].*)$", "", spec)
    members = []
    for token in re.split(r"\s+or\s+|\s*\|\s*", spec):
        try:
            member = eval(token, namespace)  # noqa: S307 - library docstring, not user input
        except Exception:
            member = pydoc.locate(token)
        if inspect.isclass(member) or isinstance(member, (typing.TypeAliasType, _stdtypes.GenericAlias)):
            members.append(member)
    if not members:
        return None
    return members[0] if len(members) == 1 else typing.Union[tuple(members)]


def positional_parameters(fn: Callable, count: int) -> set[str]:
    """The names of the first *count* parameters of *fn* (``self`` / ``cls`` aside): the ones a call fills by position."""
    names = [n for n in inspect.signature(fn).parameters if n not in ("self", "cls")]
    return set(names[:count])


def derived_parameters(fn: Callable, skip: set[str]) -> dict[str, DerivedParameter]:
    """The named parameters of *fn* (a class: its constructor) except *skip*, with derived types.

    The type is the annotation; else the numpydoc type of the parameter's
    docstring entry; else the type of a non-None default; else ``str``.
    """
    doc = inspect.getdoc(fn) or ""
    documented = {p.name.split(":")[0].strip(): p.type for p in NumpyDocString(doc)["Parameters"]} if doc else {}
    descriptions = process_method_docstring(fn)
    namespace = _namespace(fn)
    parameters = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name in skip or name in ("self", "cls") or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation, source = param.annotation, "annotation"
        if annotation is inspect.Parameter.empty:
            annotation, source = documented_type(documented.get(name, ""), namespace), "docstring"
            if annotation is None and param.default not in (None, inspect.Parameter.empty):
                annotation, source = type(param.default), "default"
            if annotation is None:
                annotation, source = str, "text"
        while isinstance(annotation, str):
            annotation = eval(annotation, namespace)  # noqa: S307 - library source, not user input
        parameters[name] = DerivedParameter(name, annotation, param.default, descriptions.get(name, ""), source, fn)
    return parameters


def merged_parameter(declarations: list[DerivedParameter]) -> DerivedParameter:
    """One parameter declared by several functions: the union of their types, the first description."""
    annotations = list(dict.fromkeys(d.annotation for d in declarations))
    annotation = annotations[0] if len(annotations) == 1 else typing.Union[tuple(annotations)]
    first = declarations[0]
    return DerivedParameter(first.name, annotation, first.default, first.description, first.source, first.owner)


def _unpacks(call: ast.Call, name: str) -> bool:
    """True when *call* passes *name* on as ``**name`` or inside an unpacked dict literal ``**{..., **name}``."""
    for keyword in call.keywords:
        if keyword.arg is not None:
            continue
        if isinstance(keyword.value, ast.Name) and keyword.value.id == name:
            return True
        if isinstance(keyword.value, ast.Dict) and any(
            key is None and isinstance(value, ast.Name) and value.id == name
            for key, value in zip(keyword.value.keys, keyword.value.values)
        ):
            return True
    return False


def explicit_keywords(call: ast.Call) -> set[str]:
    """The names *call* passes itself: its keywords and the constant keys of a dict literal it unpacks."""
    names = {k.arg for k in call.keywords if k.arg is not None}
    for keyword in call.keywords:
        if keyword.arg is None and isinstance(keyword.value, ast.Dict):
            names |= {k.value for k in keyword.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return names


def assigned_classes(fn: Callable, tree: ast.AST) -> dict[str, list[type]]:
    """The classes each variable of *fn* is assigned an instance of (``var = Cls(...)``)."""
    namespace = _namespace(fn)
    assigned: dict[str, list[type]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            cls = namespace.get(node.value.func.id)
            if inspect.isclass(cls):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned.setdefault(target.id, []).append(cls)
    return assigned


def call_targets(fn: Callable, call: ast.Call, assigned: dict[str, list[type]]) -> tuple[Callable, ...] | str:
    """The cryoCAT functions or classes *call* in *fn* reaches, or why they cannot be determined.

    Resolved: a name of *fn*'s module (``write(...)``, ``Cls(...)``), a module
    or class attribute (``cryomap.write(...)``, ``Motl.load(...)``), a method
    of ``self`` / ``cls``, of a variable assigned an instance of a class (see
    :func:`assigned_classes`), or of an instance built in place
    (``Cls(...).write_out(...)``).
    """
    namespace = _namespace(fn)
    func = call.func
    if isinstance(func, ast.Name):
        candidates = [namespace.get(func.id)]
    elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        base = func.value.id
        owner = namespace.get(fn.__qualname__.split(".")[0])
        if base in ("self", "cls") and inspect.isclass(owner):
            candidates = [getattr(owner, func.attr, None)]
        elif base in assigned:
            candidates = [getattr(c, func.attr) for c in assigned[base] if callable(getattr(c, func.attr, None))]
        elif inspect.ismodule(namespace.get(base)) or inspect.isclass(namespace.get(base)):
            candidates = [getattr(namespace[base], func.attr, None)]
        else:
            return f"{ast.unparse(func)} is called through the variable {base!r}"
    elif (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Name)
          and inspect.isclass(namespace.get(func.value.func.id))):
        candidates = [getattr(namespace[func.value.func.id], func.attr, None)]
    else:
        return f"{ast.unparse(func)} cannot be resolved"
    if not candidates or any(c is None or not callable(c) for c in candidates):
        return f"{ast.unparse(func)} cannot be resolved"
    for candidate in candidates:
        module = getattr(candidate, "__module__", None) or ""
        if module != "cryocat" and not module.startswith("cryocat."):
            return f"{ast.unparse(func)} is outside cryoCAT ({module or 'built-in'})"
    return tuple(candidates)


def _is_dict(node: ast.AST) -> bool:
    """True for a dict literal or a ``dict(...)`` call."""
    return isinstance(node, ast.Dict) or (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict"
    )


def _dict_keys(node: ast.AST) -> set[str]:
    """The constant keys a dict literal or ``dict(...)`` call writes."""
    if isinstance(node, ast.Dict):
        return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return {k.arg for k in node.keywords if k.arg is not None}


def _merges(node: ast.AST, name: str) -> bool:
    """True when a dict literal or ``dict(...)`` call merges *name* in (``{**name}``, ``dict(name)``, ``dict(**name)``)."""
    if isinstance(node, ast.Dict):
        return any(
            key is None and isinstance(value, ast.Name) and value.id == name for key, value in zip(node.keys, node.values)
        )
    return any(isinstance(a, ast.Name) and a.id == name for a in node.args) or any(
        k.arg is None and isinstance(k.value, ast.Name) and k.value.id == name for k in node.keywords
    )


def _local_dicts(tree: ast.AST, name: str) -> dict[str, set[str]]:
    """The local dicts *name* is merged into, with the constant keys written into each.

    A local dict is a name bound to a dict literal, ``dict(...)`` or ``{}``;
    *name* is merged in through the binding (``{**name}``, ``dict(name)``) or
    ``local.update(name)``.  Its constant keys come from the binding, from
    ``local.update(...)`` and from ``local["key"] = ...``.
    """
    bound: dict[str, set[str]] = {}
    carries: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and _is_dict(node.value)):
            local = node.targets[0].id
            bound.setdefault(local, set()).update(_dict_keys(node.value))
            if _merges(node.value, name):
                carries.add(local)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name) and node.func.value.id in bound):
            local = node.func.value.id
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id == name:
                    carries.add(local)
                elif _is_dict(arg):
                    bound[local] |= _dict_keys(arg)
                    if _merges(arg, name):
                        carries.add(local)
            bound[local] |= {k.arg for k in node.keywords if k.arg is not None}
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in bound
                        and isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str)):
                    bound[target.value.id].add(target.slice.value)
    return {local: bound[local] for local in carries}


def _reach(fn: Callable, tree: ast.AST, name: str) -> tuple[list[tuple[Callable, ast.Call]], str | None, set[str]]:
    """Every (target, call) of the calls in *fn* that pass *name* on, or why one cannot be determined.

    A call passes *name* on when it unpacks it, or unpacks a local dict it is
    merged into (see :func:`_local_dicts`).  The third value holds the constant
    keys of those local dicts: names *fn* passes itself.
    """
    assigned = assigned_classes(fn, tree)
    carriers = _local_dicts(tree, name)
    reached, keys = [], set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        through = [local for local in carriers if _unpacks(node, local)]
        if not (_unpacks(node, name) or through):
            continue
        targets = call_targets(fn, node, assigned)
        if isinstance(targets, str):
            return [], targets, set()
        reached += [(target, node) for target in targets]
        for local in through:
            keys |= carriers[local]
    return reached, None, keys


@lru_cache(maxsize=None)
def forwarded_keywords(fn: Callable) -> dict[str, ForwardedKeywords]:
    """The parameters *fn* passes on as ``**name``, by name, with where they go and the options this gives.

    The options are the named parameters of every target, except those the
    calls fill by position or name themselves, or write as constant keys of a
    local dict the keywords are merged into first.  A single target that
    passes its own ``**kwargs`` on is followed the same way, as long as that
    again reaches a single cryoCAT function; the names passed explicitly along
    the chain stay excluded.  The returned dict is shared: do not change it.
    """
    tree = source_tree(fn)
    result = {}
    for name in inspect.signature(fn).parameters:
        reached, unresolved, explicit = _reach(fn, tree, name)
        if not reached and unresolved is None:
            continue
        for target, call in reached:
            explicit |= explicit_keywords(call)
        declarations: dict[str, list[DerivedParameter]] = {}

        def declare(target: Callable, calls: list[ast.Call]) -> None:
            skip = set(explicit)
            for call in calls:
                skip |= positional_parameters(target, len(call.args))
            for option in derived_parameters(target, skip).values():
                declarations.setdefault(option.name, []).append(option)

        targets = tuple(dict.fromkeys(target for target, _ in reached))
        for target in targets:
            declare(target, [call for t, call in reached if t is target])

        chained: list[Callable] = []
        current = targets[0] if len(targets) == 1 else None
        while current is not None and unresolved is None:
            var_keyword = [p.name for p in inspect.signature(current).parameters.values() if p.kind == p.VAR_KEYWORD]
            if not var_keyword:
                break
            inner, unresolved, keys = _reach(current, source_tree(current), var_keyword[0])
            if unresolved is not None:
                break
            explicit |= keys
            following = tuple(dict.fromkeys(target for target, _ in inner))
            if not following:
                unresolved = f"{current.__qualname__} passes **{var_keyword[0]} on only through a variable"
            elif len(following) > 1:
                names = ", ".join(t.__qualname__ for t in following)
                unresolved = f"{current.__qualname__} passes **{var_keyword[0]} on to more than one target ({names})"
            else:
                for _, call in inner:
                    explicit |= explicit_keywords(call)
                current = following[0]
                chained.append(current)
                declare(current, [call for _, call in inner])
                continue
            break

        options = {n: merged_parameter(d) for n, d in declarations.items()}
        result[name] = ForwardedKeywords(name, targets, tuple(chained), options, unresolved)
    return result


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


def _parse_symmetry(v: Any) -> str | int | None:
    """GUI text field -> Symmetry (int or str).
    Int-coercible inputs (e.g. '1', 5) return as int; otherwise kept as str."""
    if v is None or v == "":
        return None
    if isinstance(v, int):
        return v
    s = str(v).strip()
    try:
        return int(s)
    except ValueError:
        return s


def _parse_data_pool_entry(value: Any):
    """Pool-table picker value (data_id string) -> server-side DataFrame payload.

    The lazy import means this function is only reachable after the GUI app has
    started; calling it from a pure-library context returns None.
    """
    if not value:
        return None
    try:
        from cryocat.app.datapool import get_payload as _get_payload
        return _get_payload(str(value))
    except Exception:
        return None


def _parse_motl_source(value: Any):
    """Pool-motl picker value (motl_id string or list of ids) -> Motl or list[Motl].

    The lazy import means this function is only reachable after the GUI app has
    started; calling it from a pure-library context returns None.
    """
    if not value:
        return None
    if isinstance(value, list):
        try:
            from cryocat.app.pool import get_rows as _get_rows
            from cryocat.core.cryomotl import EmMotl
            result = [EmMotl(_get_rows(str(v))) for v in value if v]
            return result if result else None
        except Exception:
            return None
    try:
        from cryocat.app.pool import get_rows as _get_rows
        from cryocat.core.cryomotl import EmMotl
        return EmMotl(_get_rows(str(value)))
    except Exception:
        return None


def _arg_bool(s: str) -> bool:
    """``true``/``false`` (any case) or ``1``/``0``; anything else is an argparse error."""
    value = s.strip().lower()
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false or 1/0, got {s!r}")


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
    "MapSource":      {"widget": "path",       "parse": _parse_path,            "argparse": {"type": str}},
    "DataSource":     {"widget": "path",       "parse": _parse_path,            "argparse": {"type": str}},
    "DataPoolEntry":  {"widget": "pool_table", "parse": _parse_data_pool_entry, "argparse": {"type": str}},
    "MotlSource":     {"widget": "pool_motl",  "parse": _parse_motl_source,     "argparse": {"type": str}},
    "TiltStack":      {"widget": "path",     "parse": _parse_path,     "argparse": {"type": str}},
    "TomoDimensions": {"widget": "path",     "parse": _parse_path,     "argparse": {"type": str}},
    "TomoList":       {"widget": "text",     "parse": _parse_listlike, "argparse": {"type": _arg_listlike}},
    "ArrayLike":      {"widget": "csv_text", "parse": _parse_listlike, "argparse": {"type": _arg_listlike}},
    "TripletLike":    {"widget": "triplet",  "parse": _parse_triplet,  "argparse": {"type": _arg_triplet}},
    "EulerAngles":    {"widget": "triplet",  "parse": _parse_triplet,  "argparse": {"type": _arg_triplet}},
    "ListLike":       {"widget": "listlike", "parse": _parse_listlike, "argparse": {"type": _arg_listlike}},
    "PathOrStr":      {"widget": "path",     "parse": _parse_path,     "argparse": {"type": str}},
    # Normalized tag used by formgen.build_form for all path-widget params.
    "path":           {"widget": "path",     "parse": _parse_path,     "argparse": {"type": str}},
    "Symmetry":       {"widget": "text",     "parse": _parse_symmetry, "argparse": {"type": str}},
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
