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

from numpydoc.docscrape import NumpyDocString

from cryocat._types import ListLike


def filter_strings(input_list, filter_contains=None, filter_exclude=None):
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


def get_class_names_by_parent(parent_class_name: str, module_name: str, filter_contains=None, filter_exclude=None):
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


def get_classes_from_names(class_names, module_name):
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


def get_class_names_by_prefix(prefix):
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
# @gui_exposed — mark a method as GUI-exposable + carry presentation metadata
# ===========================================================================

def gui_exposed(_fn=None, *, label=None, category=None, hide=(), output=None):
    """Mark a method as GUI-exposable and carry presentation metadata.

    Parameters
    ----------
    label : str, optional
        Display name in the operation dropdown. Defaults to the function name.
    category : str, optional
        Grouping for the dropdown (e.g. "Cleaning", "Geometry"). None = ungrouped.
    hide : iterable of str, optional
        Parameter names the GUI/CLI should NOT surface (beyond ``self``, which is
        always hidden).
    output : {"motl", "figure", "dataframe", None}, optional
        What the method returns, so the GUI knows how to route the result
        ("motl" -> wire to send-to-editor; "figure"/"dataframe" -> display;
        None -> in-place / no surfaced result).

    Notes
    -----
    A function is GUI-exposed iff ``getattr(fn, "_gui", None) is not None``.
    A bare ``@gui_exposed`` (no parentheses) works.
    """

    def wrap(fn):
        fn._gui = {
            "label": label or fn.__name__.replace("_", " ").capitalize(),
            "category": category,
            "hide": set(hide) | {"self"},
            "output": output,
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
}
# PEP-695 aliases whose value is a Literal[...] — resolved to ("Literal", choices).
_LITERAL_ALIASES = {"MotlType", "MotlColumn", "BoundaryType", "CTFFileType"}


def resolve_param_type(annotation) -> tuple[str, dict]:
    """Map a parameter annotation to a handler tag + extras.

    Returns ``(tag, extra)``; ``extra`` carries e.g. ``{"choices": [...]}`` for
    Literals. Rules:

    * None / empty annotation        -> ``("str", {})``
    * ``Optional[X]`` / ``X | None`` -> unwrap to X, then resolve
    * ``Literal[...]``               -> ``("Literal", {"choices": [...]})``
    * a PEP-695 alias in the known set -> ``(alias name, {})``
    * bare ``bool``/``int``/``float``/``str`` -> ``(name, {})``
    * anything else                  -> ``("str", {})``
    """
    if annotation is inspect.Parameter.empty or annotation is None:
        return ("str", {})

    origin = typing.get_origin(annotation)

    # Optional[X] / X | None — unwrap NoneType and resolve the remainder.
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

    # Subscripted PEP-695 alias, e.g. ListLike[int] — origin is the alias itself.
    if isinstance(origin, typing.TypeAliasType) and origin.__name__ in _ALIAS_TAGS:
        return (origin.__name__, {})

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

def _coerce_scalar(x):
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


def _parse_number_list(v):
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


def _parse_path(v):
    return v or None


def _parse_bool(v):
    return v in (True, "True", "true", "1", 1)


def _parse_triplet(v):
    """A GUI triplet field (``"64,64,64"`` or ``"64"``). The receiving function
    normalizes with :func:`cryocat.utils.geom.as_triplet`."""
    return _parse_number_list(v)


def _parse_listlike(v):
    """A GUI csv/text field (``"1,2,3"``). The receiving function normalizes
    with :func:`cryocat.utils.classutils.as_list`."""
    return _parse_number_list(v)


def _parse_literal(v, choices=None):
    """A GUI dropdown value. If the matching choice is non-string typed,
    coerce ``v`` to that choice's type."""
    if v is None or not choices:
        return v
    for c in choices:
        if str(c) == str(v):
            return c
    return v


# argparse ``type=`` helpers (CLI string -> python value).
def _arg_bool(s):
    return _parse_bool(s)


def _arg_triplet(s):
    return _parse_number_list(s)


def _arg_listlike(s):
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
    "ListLike":       {"widget": "csv_text", "parse": _parse_listlike, "argparse": {"type": _arg_listlike}},
    "Symmetry":       {"widget": "text",     "parse": str,             "argparse": {"type": str}},
    "Literal":        {"widget": "dropdown", "parse": _parse_literal,  "argparse": {"type": str}},
    "bool":           {"widget": "bool",     "parse": _parse_bool,     "argparse": {"type": _arg_bool}},
    "int":            {"widget": "number",   "parse": int,             "argparse": {"type": int}},
    "float":          {"widget": "number",   "parse": float,           "argparse": {"type": float}},
    "str":            {"widget": "text",     "parse": str,             "argparse": {"type": str}},
}


# ===========================================================================
# Docstring -> parameter descriptions (help text / tooltips only)
# ===========================================================================

def _clean_desc(text):
    """Strip reST cross-reference role markers from help text."""
    for role in (":meth:", ":func:", ":class:", ":data:", ":attr:", ":mod:"):
        text = text.replace(role, "")
    return text.strip()


def process_method_docstring(path_to_method, method_name=None):
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
    dict
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
