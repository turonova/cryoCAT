"""Command-line interfaces for cryoCAT: one command per module, one subcommand per exposed callable.

Each command ``cc_<module>`` offers every callable of its module that carries
:func:`cryocat.utils.classutils.cli_exposed`: the public functions, and the
public methods of the classes defined in the module.  The subcommand is the
Python name of the callable and the options are its parameter names::

    cc_cryomotl split_in_asymmetric_subunits --input_motl in.em --output_motl out.em --symmetry C8

Argument parsers are built from function *signatures* — parameter types via
:func:`cryocat.utils.classutils.resolve_param_type`, required/optional and
defaults from the signature, help text from the docstring — routed through the
same :data:`cryocat.utils.classutils.TYPE_HANDLERS` table the GUI form
generator uses.  There is one type authority (annotations) and one table.

Motl methods read ``--input_motl`` in the format ``--input_type`` and write
``--output_motl`` in the format ``--output_type``, and the motl converters
take the options of the motl constructors and writers they pass their
``load_kwargs`` / ``write_kwargs`` on to.  All of this goes through the shared
motl IO of :mod:`cryocat.core.cryomotl`; its options are offered as
``--input_<name>`` (reading) and ``--output_<name>`` (writing).
"""

import argparse
import ast
import functools
import inspect
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from cryocat.analysis import tmana as tmana_module
from cryocat.core import cryomap as cryomap_module
from cryocat.core import cryomask as cryomask_module
from cryocat.core import cryomotl as cryomotl_module
from cryocat.core import cryowedge as cryowedge_module
from cryocat.core import mdoc as mdoc_module
from cryocat.core import tiltstack as tiltstack_module
from cryocat.utils import geom as geom_module
from cryocat.utils.classutils import (
    TYPE_HANDLERS,
    DerivedParameter,
    call_targets,
    forwarded_keywords,
    process_method_docstring,
    resolve_param_type,
    source_tree,
)

_LIST_TAGS = frozenset({"ListLike", "ArrayLike", "TripletLike"})
"""Handler tags whose values are given as several tokens (``nargs="+"``)."""

_PATH_TAGS = frozenset({"PathOrStr", "path", "MapSource", "DataSource", "TiltStack"})
"""Handler tags of the path types: a union of one of them with a number takes a number or a path."""



@dataclass(frozen=True)
class _Command:
    """One subcommand: the exposed callable, the class that owns it (``None`` for a function) and how it is called."""

    name: str
    fn: Callable
    owner: type | None
    kind: str
    hide: frozenset[str]
    output: str | None
    required: frozenset[str]

    @property
    def qualname(self) -> str:
        return self.fn.__qualname__


@dataclass(frozen=True)
class _MotlOption:
    """One ``--input_<name>`` / ``--output_<name>`` option of the shared motl IO.

    ``targets`` maps each motl type (for a converter: each dict parameter)
    that takes the option to where a given value goes: ``"load"``,
    ``"constructor"`` or ``"writer"``.
    """

    name: str
    annotation: Any
    help: str
    owner: Callable
    targets: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class _Invocation:
    """What a parsed command line calls, built before anything is read or run.

    ``read`` is ``(input_motl, input_type, Motl.load options)`` for a motl
    method, ``(input_mdoc, "mdoc", {})`` for an :class:`~cryocat.core.mdoc.Mdoc`
    method, and ``write`` ``(output_type, constructor options, writer
    options)``, each ``None`` where the command reads or writes no motl.
    ``writes_itself`` is true when the input and output types are the same:
    the loaded motl, as the method returns or changes it, writes itself with
    the writer options.
    """

    kwargs: dict
    destination: str | None
    read: tuple[str, str, dict] | None
    write: tuple[str, dict, dict] | None
    writes_itself: bool


class _OneOrSeveral(argparse.Action):
    """Store one token as a single value and several tokens as a list."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values[0] if len(values) == 1 else values)


def _summary(obj: Any) -> str:
    """First paragraph of a docstring — used as subcommand and command help."""
    doc = inspect.getdoc(obj) or ""
    para = []
    for line in doc.splitlines():
        if not line.strip():
            break
        para.append(line.strip())
    return " ".join(para)


def _exposed(module: types.ModuleType) -> dict[str, _Command]:
    """Every ``@cli_exposed`` function of *module* and method of a class defined in it, by name.

    Raises
    ------
    ValueError
        If two exposed callables of the module share a name, or an exposed
        instance method belongs to a class that is neither a motl class nor
        :class:`~cryocat.core.mdoc.Mdoc`.
    """
    commands: dict[str, _Command] = {}

    def add(name: str, fn: Callable, owner: type | None, kind: str) -> None:
        command = _Command(name, fn, owner, kind, frozenset(fn._cli["hide"]), fn._cli["output"], fn._cli["required"])
        if command.name in commands:
            raise ValueError(
                f"{module.__name__}: {commands[command.name].qualname} and {command.qualname} "
                f"are both exposed as the subcommand {command.name!r}"
            )
        commands[command.name] = command

    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) and obj.__module__ == module.__name__ and hasattr(obj, "_cli"):
            add(name, obj, None, "function")
        elif inspect.isclass(obj) and obj.__module__ == module.__name__:
            for member_name, member in vars(obj).items():
                fn = member.__func__ if isinstance(member, (classmethod, staticmethod)) else member
                if member_name.startswith("_") or not inspect.isfunction(fn) or not hasattr(fn, "_cli"):
                    continue
                kind = (
                    "classmethod" if isinstance(member, classmethod)
                    else "staticmethod" if isinstance(member, staticmethod)
                    else "method"
                )
                if kind == "method" and not issubclass(obj, (cryomotl_module.Motl, mdoc_module.Mdoc)):
                    raise ValueError(
                        f"{fn.__qualname__}: instance methods are exposed only for motl classes and Mdoc, "
                        "which the command line loads from --input_motl or --input_mdoc"
                    )
                add(member_name, fn, obj, kind)
    return dict(sorted(commands.items()))


def _evaluated(annotation: Any, fn: Callable) -> Any:
    """*annotation* evaluated in the namespace of *fn*.

    Under ``from __future__ import annotations`` every annotation is a string,
    and one written in quotes (``"MotlSource"``) is a string again after the
    first evaluation.
    """
    while isinstance(annotation, str):
        annotation = eval(annotation, fn.__globals__)  # noqa: S307 - library source, not user input
    return annotation


def _members(annotation: Any) -> list[Any]:
    """The members of a union other than ``None``; a non-union annotation as its only member."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return [a for a in typing.get_args(annotation) if a is not type(None)]
    return [annotation]


def _is_text(annotation: Any) -> bool:
    """True when *annotation* really is text (or absent), not a type that only fell back to the ``str`` tag."""
    if annotation is inspect.Parameter.empty:
        return True
    for member in _members(annotation):
        if member is str:
            return True
        if isinstance(member, typing.TypeAliasType) and _is_text(member.__value__):
            return True
    return False


def _number_or_path(token: str, numbers: tuple[type, ...]) -> int | float | str:
    """*token* as the first of *numbers* (``int``, ``float``) it parses as, otherwise as a path string."""
    for number in numbers:
        try:
            return number(token)
        except ValueError:
            pass
    return token


def _argument(fn: Callable, name: str, annotation: Any) -> dict:
    """The ``add_argument`` type, ``nargs``, ``choices`` and ``action`` for parameter *name* of *fn*.

    Raises
    ------
    ValueError
        If the parameter's type has no command-line form.
    """
    annotation = _evaluated(annotation, fn)
    members = _members(annotation)

    # X | list[X]: one token is a single value, several a list.
    lists = [m for m in members if typing.get_origin(m) is list and typing.get_args(m)]
    if len(members) == 2 and len(lists) == 1 and typing.get_args(lists[0]) == tuple(m for m in members if m is not lists[0]):
        spec = _argument(fn, name, typing.get_args(lists[0])[0])
        if "nargs" in spec:
            raise ValueError(f"{fn.__qualname__}: parameter {name!r} is a nested list, which has no command-line form")
        return {**spec, "nargs": "+", "action": _OneOrSeveral}

    # A number or a path: a token that parses as a number is that number, any other a path.
    numeric = tuple(n for n in (int, float) if n in members)
    if numeric and any(resolve_param_type(m)[0] in _PATH_TAGS for m in members):
        return {"type": functools.partial(_number_or_path, numbers=numeric)}

    # int | float: a float.
    if set(members) == {int, float}:
        return {"type": float}

    if len(members) == 1 and typing.get_origin(members[0]) in (list, tuple) and resolve_param_type(members[0])[0] == "str":
        elements = [a for a in typing.get_args(members[0]) if a is not Ellipsis]
        spec = _argument(fn, name, elements[0] if len(set(elements)) == 1 else Any)
        if "nargs" in spec:
            raise ValueError(f"{fn.__qualname__}: parameter {name!r} is a nested list, which has no command-line form")
        return {**spec, "nargs": "+"}

    tag, extra = resolve_param_type(annotation)
    if tag == "str" and not _is_text(annotation):
        raise ValueError(f"{fn.__qualname__}: parameter {name!r} of type {annotation!r} has no command-line form")
    spec = dict(TYPE_HANDLERS[tag]["argparse"])
    if tag == "Literal":
        spec["choices"] = extra["choices"]
        spec["type"] = functools.partial(TYPE_HANDLERS["Literal"]["parse"], choices=extra["choices"])
    elif tag == "ListLike":
        spec["type"] = TYPE_HANDLERS[extra["elem_tag"]]["argparse"]["type"]
        spec["nargs"] = "+"
    elif tag == "Tuple":
        spec["type"] = TYPE_HANDLERS[extra["elem"]]["argparse"]["type"]
        spec["nargs"] = extra["length"]
    elif tag in _LIST_TAGS or (tag == "MotlSource" and extra.get("multi")):
        spec["nargs"] = "+"
    elif tag == "TomoDimensions":
        # CLI-only: a comma-separated token ("5760,4092,1800") is parsed into
        # [x, y, z] (or [tomo_id, x, y, z]) ints, the same `list` case
        # ioutils.dimensions_load() already accepts from Python, so its own
        # 1x3/Nx4 shape check validates the result; a token with no comma
        # that isn't numeric is kept as a path, as before (same mechanism
        # already used for TomoList, which has the same path-or-list-of-ints
        # shape). The shared TYPE_HANDLERS entry (GUI path widget) is
        # untouched.
        spec["type"] = TYPE_HANDLERS["ArrayLike"]["argparse"]["type"]
    elif tag == "RotationLike":
        # CLI-only: a comma-separated token ("10,20,30") is parsed into a
        # list of numbers -- the (3,) Euler-angle case of geom.as_rotation().
        # Matrices/quaternions/stacks aren't meaningfully typable as CLI
        # tokens, so that's the only shape this covers. The shared
        # TYPE_HANDLERS entry (GUI rotation widget) is untouched.
        spec["type"] = TYPE_HANDLERS["ArrayLike"]["argparse"]["type"]
    return spec


def _motl_options(declarations: list[tuple[str, str, DerivedParameter]]) -> dict[str, _MotlOption]:
    """Options by name from ``(target, destination, parameter)`` declarations of derived parameters.

    A declaration whose type has no command-line form (a ``dict``, a
    DataFrame) is not offered; an option declared several times takes the
    union of their types.
    """
    grouped: dict[str, list[tuple[str, str, DerivedParameter]]] = {}
    for target, destination, parameter in declarations:
        try:
            _argument(parameter.owner, parameter.name, parameter.annotation)
        except ValueError:
            continue
        grouped.setdefault(parameter.name, []).append((target, destination, parameter))
    options = {}
    for name, entries in grouped.items():
        annotations = tuple(dict.fromkeys(p.annotation for _, _, p in entries))
        targets: dict[str, tuple[str, ...]] = {}
        for target, destination, _ in entries:
            targets[target] = targets.get(target, ()) + (destination,)
        first = entries[0][2]
        options[name] = _MotlOption(
            name, annotations[0] if len(annotations) == 1 else typing.Union[annotations], first.description,
            first.owner, targets,
        )
    return options


@functools.cache
def _read_options() -> dict[str, _MotlOption]:
    """The ``--input_<name>`` options of motl methods: what :meth:`Motl.load` takes per motl type."""
    return _motl_options([
        (motl_type, "load", parameter)
        for motl_type in cryomotl_module.motl_io_types()
        for parameter in cryomotl_module.motl_read_parameters(motl_type).values()
    ])


@functools.cache
def _write_options() -> dict[str, _MotlOption]:
    """The ``--output_<name>`` options of motl outputs: the writer's constructor and ``write_out`` options per type."""
    declarations = []
    for motl_type in cryomotl_module.motl_io_types():
        constructor, writer = cryomotl_module.motl_write_parameters(motl_type)
        declarations += [(motl_type, "constructor", p) for p in constructor.values()]
        declarations += [(motl_type, "writer", p) for p in writer.values()]
    return _motl_options(declarations)


@dataclass(frozen=True)
class _Forwarded:
    """The options of a parameter a callable passes on as ``**name`` (see :func:`~cryocat.utils.classutils.forwarded_keywords`).

    ``flags`` maps each option to its command-line flag (without ``--``).  For
    the callable's own ``**kwargs`` (``unpacked``) a given option becomes one
    of its keyword arguments; for a dict parameter it goes into that dict, and
    ``prefix`` is ``"input"`` (constructors) or ``"output"``.
    """

    parameter: str
    unpacked: bool
    prefix: str | None
    options: dict[str, _MotlOption]
    flags: dict[str, str]


def _loads_mdoc(command: _Command) -> bool:
    """True for an :class:`~cryocat.core.mdoc.Mdoc` method, called on the mdoc loaded from ``--input_mdoc``."""
    return command.kind == "method" and issubclass(command.owner, mdoc_module.Mdoc)


def _added_names(command: _Command) -> set[str]:
    """The options the command line adds to *command* besides its parameters (without ``--``)."""
    names = {"help", "h"}
    if _loads_mdoc(command):
        names.add("input_mdoc")
    elif command.kind == "method":
        names |= {"input_motl", "input_type"} | {f"input_{n}" for n in _read_options()}
    if command.output == "motl":
        names |= {"output_motl", "output_type"} | {f"output_{n}" for n in _write_options()}
    elif command.output is not None:
        names.add(f"output_{command.output}")
    return names


@functools.cache
def _forwarded(command: _Command) -> dict[str, _Forwarded]:
    """The parameters *command* passes on as ``**name`` to cryoCAT functions, with their options and flags.

    A dict parameter (a converter's ``load_kwargs`` / ``write_kwargs``) is
    offered as ``--input_<name>`` when it goes to classes (constructors) and
    as ``--output_<name>`` otherwise.  The callable's own ``**kwargs`` is
    offered as ``--<name>``, or ``--<kwargs>_<name>`` where ``<name>`` clashes
    with an option the command line adds.  A name that is also one of the
    callable's own parameters is not offered: Python binds it to that
    parameter, so it can never reach the ``**kwargs``.
    """
    parameters = inspect.signature(command.fn).parameters
    added = _added_names(command)
    forwarded = {}
    for name, keywords in forwarded_keywords(command.fn).items():
        options = _motl_options([(name, "forwarded", p) for p in keywords.options.values()])
        unpacked = parameters[name].kind == inspect.Parameter.VAR_KEYWORD
        if unpacked:
            options = {n: o for n, o in options.items() if n not in parameters}
        if not options:
            continue
        if unpacked:
            prefix = None
            flags = {n: f"{name}_{n}" if n in added else n for n in options}
        else:
            prefix = "input" if all(inspect.isclass(t) for t in keywords.targets) else "output"
            flags = {n: f"{prefix}_{n}" for n in options}
        forwarded[name] = _Forwarded(name, unpacked, prefix, options, flags)
    return forwarded


@functools.cache
def _version_parameter(command: _Command) -> str | None:
    """The parameter of a converter that sets the RELION version, or None.

    Read from the converter's source with :mod:`ast`, as the names it passes
    itself are: the parameter it passes as ``version=`` to a class, or
    compares in an ``if`` whose branch builds a class with a dict it passes on.

    Raises
    ------
    ValueError
        If more than one parameter qualifies.
    """
    dicts = {name for name, f in _forwarded(command).items() if not f.unpacked}
    if not dicts:
        return None
    fn = command.fn
    own = set(inspect.signature(fn).parameters)
    tree = source_tree(fn)

    def class_calls(nodes: list[ast.AST]) -> list[ast.Call]:
        calls = []
        for node in nodes:
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                    targets = call_targets(fn, call, {})
                    if not isinstance(targets, str) and inspect.isclass(targets[0]):
                        calls.append(call)
        return calls

    found = set()
    for call in class_calls([tree]):
        for keyword in call.keywords:
            if keyword.arg == "version" and isinstance(keyword.value, ast.Name) and keyword.value.id in own:
                found.add(keyword.value.id)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id in own):
            continue
        if any(
            k.arg is None and isinstance(k.value, ast.Name) and k.value.id in dicts
            for call in class_calls(node.body) for k in call.keywords
        ):
            found.add(node.test.left.id)
    if len(found) > 1:
        raise ValueError(f"{fn.__qualname__}: several parameters set the RELION version: {sorted(found)}")
    return next(iter(found), None)


def _given_options(
    parser: argparse.ArgumentParser,
    args: dict,
    prefix: str,
    options: dict[str, _MotlOption],
    target: str,
    flags: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Pop the options given in *args* (flag ``<prefix>_<name>``, or ``flags[name]``); their values by destination for *target*."""
    by_destination: dict[str, dict] = {}
    for name, option in options.items():
        flag = flags[name] if flags else f"{prefix}_{name}"
        if flag not in args:
            continue
        value = args.pop(flag)
        if target not in option.targets:
            parser.error(f"--{flag} applies only to --{prefix}_type {' or '.join(option.targets)}")
        for destination in option.targets[target]:
            by_destination.setdefault(destination, {})[name] = value
    return by_destination


def _add_subcommand(subparsers: argparse._SubParsersAction, command: _Command) -> None:
    """Add the subcommand for *command*, one option per visible parameter plus the options the CLI adds."""
    summary = _summary(command.fn)
    parser = subparsers.add_parser(command.name, description=summary, help=summary, add_help=False)
    required = parser.add_argument_group("Required arguments")
    optional = parser.add_argument_group("Optional arguments")
    formats = parser.add_argument_group("Motl format arguments")
    forwarding = parser.add_argument_group("Forwarded arguments")
    descriptions = process_method_docstring(command.fn)

    added: dict[str, tuple[argparse._ArgumentGroup, dict]] = {}
    motl_types = cryomotl_module.motl_io_types()

    def add_options(
        prefix: str,
        options: dict[str, _MotlOption],
        typed: bool,
        flags: dict[str, str] | None = None,
        group: argparse._ArgumentGroup = formats,
    ) -> None:
        for option in options.values():
            only = ""
            if typed and len(option.targets) < len(motl_types):
                only = f" (--{prefix}_type {' or '.join(option.targets)})"
            added[flags[option.name] if flags else f"{prefix}_{option.name}"] = (group, {
                **_argument(option.owner, option.name, option.annotation),
                "default": argparse.SUPPRESS,
                "help": option.help + only,
            })

    def add_type(prefix: str, given: str) -> None:
        added[f"{prefix}_type"] = (formats, {"choices": motl_types, "default": "emmotl", "help": f"Format of --{given}."})

    if _loads_mdoc(command):
        added["input_mdoc"] = (required, {"required": True, "type": str, "help": "Mdoc file the method is called on."})
    elif command.kind == "method":
        added["input_motl"] = (required, {"required": True, "type": str, "help": "Motl file the method is called on."})
        add_type("input", "input_motl")
        add_options("input", _read_options(), typed=True)
    if command.output == "motl":
        added["output_motl"] = (required, {"required": True, "type": str, "help": "Path the resulting motl is written to."})
        add_type("output", "output_motl")
        add_options("output", _write_options(), typed=True)
    elif command.output == "csv":
        added["output_csv"] = (required, {"required": True, "type": str, "help": "Path the resulting table is written to, as CSV."})
    elif command.output == "map":
        added["output_map"] = (required, {"required": True, "type": str, "help": "Path the resulting map is written to."})
    elif command.output == "mdoc":
        added["output_mdoc"] = (required, {"required": True, "type": str, "help": "Path the resulting mdoc is written to."})
    forwarded = _forwarded(command)
    for passed in forwarded.values():
        add_options("", passed.options, typed=False, flags=passed.flags, group=forwarding)

    for name, param in inspect.signature(command.fn).parameters.items():
        if name in command.hide or name in forwarded or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in added:
            raise ValueError(f"{command.qualname}: parameter {name!r} clashes with the option the command line adds")
        kwargs = {"help": descriptions.get(name, ""), "dest": name, **_argument(command.fn, name, param.annotation)}
        if "choices" not in kwargs:
            kwargs["metavar"] = ("value_" + name).upper()
        if param.default is inspect.Parameter.empty or name in command.required:
            required.add_argument(f"--{name}", required=True, **kwargs)
        else:
            optional.add_argument(f"--{name}", default=param.default, **kwargs)

    for name, (group, kwargs) in added.items():
        if "choices" not in kwargs:
            kwargs["metavar"] = ("value_" + name).upper()
        group.add_argument(f"--{name}", dest=name, **kwargs)
    # Re-add help (suppressed above via add_help=False so it lands in the group).
    optional.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit.")


def build_parser(module: types.ModuleType) -> tuple[argparse.ArgumentParser, dict[str, _Command]]:
    """Build the parser of the command for *module*: one subcommand per ``@cli_exposed`` callable.

    Parameters
    ----------
    module : module
        The module whose exposed functions, and exposed methods of the
        classes defined in it, become subcommands.

    Returns
    -------
    argparse.ArgumentParser
        The parser, with one subcommand per exposed callable.
    dict of str to _Command
        The exposed callables by subcommand name.

    Raises
    ------
    ValueError
        If two exposed callables share a name, or a visible parameter has a
        type with no command-line form; the message names the callable and
        the parameter.
    """
    short = module.__name__.rsplit(".", 1)[-1]
    commands = _exposed(module)
    parser = argparse.ArgumentParser(
        prog=f"cc_{short}",
        description=f"{_summary(module)} For help on a subcommand run cc_{short} <subcommand> --help.",
    )
    subparsers = parser.add_subparsers(title="Subcommands", dest="subcommand", required=True, metavar="subcommand")
    for command in commands.values():
        _add_subcommand(subparsers, command)
    return parser, commands


def _invocation(parser: argparse.ArgumentParser, command: _Command, args: dict) -> _Invocation:
    """The call that the parsed *args* (without ``subcommand``) describe for *command*; *args* is not changed."""
    args = dict(args)
    destination = args.pop(f"output_{command.output}") if command.output else None

    read = None
    if _loads_mdoc(command):
        read = (args.pop("input_mdoc"), "mdoc", {})
    elif command.kind == "method":
        input_motl, input_type = args.pop("input_motl"), args.pop("input_type")
        loaded = _given_options(parser, args, "input", _read_options(), input_type)
        read = (input_motl, input_type, loaded.get("load", {}))
    write = None
    writes_itself = False
    if command.output == "motl":
        output_type = args.pop("output_type")
        writes_itself = read is not None and read[1] == output_type
        if writes_itself:
            # Checked against the loaded motl's own write_out after loading (see _check_self_write);
            # an option no writer takes is an error here.
            writer_kwargs = {}
            for name, option in _write_options().items():
                if f"output_{name}" not in args:
                    continue
                if not any("writer" in destinations for destinations in option.targets.values()):
                    parser.error(f"--output_{name} does not apply when the motl writes itself (--input_type and "
                                 f"--output_type are both {output_type})")
                writer_kwargs[name] = args.pop(f"output_{name}")
            write = (output_type, {}, writer_kwargs)
        else:
            written = _given_options(parser, args, "output", _write_options(), output_type)
            write = (output_type, written.get("constructor", {}), written.get("writer", {}))
    version_parameter = _version_parameter(command)
    version = args.get(version_parameter) if version_parameter else None
    relion_type = cryomotl_module.relion_version_to_type(version) if version is not None else None
    for name, passed in _forwarded(command).items():
        given = _given_options(parser, args, "", passed.options, name, passed.flags)
        values = {k: v for destination in given.values() for k, v in destination.items()}
        if passed.unpacked:
            args.update(values)
            continue
        if relion_type is not None:
            accepted = (
                cryomotl_module.motl_read_parameters(relion_type) if passed.prefix == "input"
                else cryomotl_module.motl_write_parameters(relion_type)[1]
            )
            for option in values:
                if option not in accepted:
                    parser.error(f"--{passed.flags[option]} does not apply to --{version_parameter} {version}")
        if values:
            args[name] = values
    return _Invocation(args, destination, read, write, writes_itself)


def _check_self_write(parser: argparse.ArgumentParser, invocation: _Invocation, motl: cryomotl_module.Motl) -> None:
    """For a self-write, reject a given ``--output_*`` option the ``write_out`` of the loaded *motl* does not take."""
    if not invocation.writes_itself:
        return
    accepted = inspect.signature(type(motl).write_out).parameters
    for name in invocation.write[2]:
        if name not in accepted:
            parser.error(f"--output_{name} does not apply to the loaded {type(motl).__name__}, which writes itself")


def _write_output(
    command: _Command, result: Any, called_on: cryomotl_module.Motl | None, invocation: _Invocation
) -> None:
    """Write what *command* returned (or, for a method returning ``None``, the motl it changed)."""
    path = invocation.destination
    if command.output == "motl":
        if isinstance(result, cryomotl_module.Motl):
            motl = result
        elif result is None and called_on is not None:
            motl = called_on
        else:
            raise TypeError(f"{command.qualname} returned {type(result).__name__}, not a motl to write")
        output_type, constructor_kwargs, writer_kwargs = invocation.write
        if invocation.writes_itself and type(motl) is type(called_on):
            motl.write_out(path, **writer_kwargs)
            return
        # A method that returns a motl of another class (e.g. a plain Motl) is written as for differing types.
        extra = cryomotl_module.motl_extra_table(motl) if invocation.writes_itself else None
        writer = cryomotl_module.build_motl_writer(output_type, motl.df, extra, **constructor_kwargs)
        writer.write_out(path, **writer_kwargs)
    elif command.output == "csv":
        if not isinstance(result, pd.DataFrame):
            raise TypeError(f"{command.qualname} returned {type(result).__name__}, not a DataFrame to write as CSV")
        result.to_csv(path, index=False)
    elif command.output == "map":
        if not isinstance(result, np.ndarray):
            raise TypeError(f"{command.qualname} returned {type(result).__name__}, not an array to write as a map")
        cryomap_module.write(result, path)
    elif command.output == "mdoc":
        if isinstance(result, mdoc_module.Mdoc):
            result.write(path)
        elif result is None and isinstance(called_on, mdoc_module.Mdoc):
            called_on.write(path)
        else:
            raise TypeError(f"{command.qualname} returned {type(result).__name__}, not an mdoc to write")


def run(module: types.ModuleType) -> None:
    """Parse the command line for *module*, call the chosen callable and write its output.

    Parameters
    ----------
    module : module
        The module whose command is run (see :func:`build_parser`).
    """
    parser, commands = build_parser(module)
    args = vars(parser.parse_args())
    command = commands[args.pop("subcommand")]
    invocation = _invocation(parser, command, args)

    called_on = None
    if invocation.read is not None:
        input_path, input_type, load_kwargs = invocation.read
        if input_type == "mdoc":
            called_on = mdoc_module.Mdoc(input_path)
        else:
            called_on = cryomotl_module.Motl.load(input_path, input_type, **load_kwargs)
            _check_self_write(parser, invocation, called_on)
        result = getattr(called_on, command.name)(**invocation.kwargs)
    elif command.owner is not None:
        result = getattr(command.owner, command.name)(**invocation.kwargs)
    else:
        result = command.fn(**invocation.kwargs)
    _write_output(command, result, called_on, invocation)


def cryomotl() -> None:
    """``cc_cryomotl``: the command line of :mod:`cryocat.core.cryomotl`."""
    run(cryomotl_module)


def cryomap() -> None:
    """``cc_cryomap``: the command line of :mod:`cryocat.core.cryomap`."""
    run(cryomap_module)


def cryomask() -> None:
    """``cc_cryomask``: the command line of :mod:`cryocat.core.cryomask`."""
    run(cryomask_module)


def tiltstack() -> None:
    """``cc_tiltstack``: the command line of :mod:`cryocat.core.tiltstack`."""
    run(tiltstack_module)


def cryowedge() -> None:
    """``cc_cryowedge``: the command line of :mod:`cryocat.core.cryowedge`."""
    run(cryowedge_module)


def mdoc() -> None:
    """``cc_mdoc``: the command line of :mod:`cryocat.core.mdoc`."""
    run(mdoc_module)


def tmana() -> None:
    """``cc_tmana``: the command line of :mod:`cryocat.analysis.tmana`."""
    run(tmana_module)


def geom() -> None:
    """``cc_geom``: the command line of :mod:`cryocat.utils.geom`."""
    run(geom_module)
