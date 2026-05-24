"""Command-line interfaces for cryoCAT.

Argument parsers are built from function *signatures* — parameter types via
:func:`cryocat.utils.classutils.resolve_param_type`, required/optional and
defaults from the signature, help text from the docstring — routed through the
same :data:`cryocat.utils.classutils.TYPE_HANDLERS` table the GUI form
generator uses. There is one type authority (annotations) and one table.
"""

import argparse
import inspect

from cryocat.utils import wedgeutils
from cryocat.analysis import tmana
from cryocat.utils.classutils import resolve_param_type, process_method_docstring, TYPE_HANDLERS


def _summary(fn):
    """First paragraph of a function's docstring — used as subcommand help."""
    doc = inspect.getdoc(fn) or ""
    para = []
    for line in doc.splitlines():
        if not line.strip():
            break
        para.append(line.strip())
    return " ".join(para)


def add_params_from_signature(subparsers, function_name, fn):
    """Add an argparse subcommand for ``fn``, building each argument from the
    function signature via :func:`resolve_param_type` + :data:`TYPE_HANDLERS`.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object to add the new parser to.
    function_name : str
        Name of the subcommand.
    fn : callable
        The function whose signature drives the arguments.
    """
    summary = _summary(fn)
    parser = subparsers.add_parser(function_name, description=summary, help=summary, add_help=False)

    required = parser.add_argument_group("Required arguments")
    optional = parser.add_argument_group("Optional arguments")

    descriptions = process_method_docstring(fn)
    gui = getattr(fn, "_gui", None)
    hide = set(gui["hide"]) if gui else {"self"}

    for name, param in inspect.signature(fn).parameters.items():
        if name in hide:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        tag, extra = resolve_param_type(param.annotation)
        spec = TYPE_HANDLERS[tag]["argparse"]
        is_required = param.default is inspect.Parameter.empty

        kwargs = {
            "help": descriptions.get(name, ""),
            "dest": name,
            "type": spec["type"],
            "metavar": ("value_" + name).upper(),
        }
        if tag == "Literal" and extra.get("choices"):
            kwargs["choices"] = extra["choices"]

        if is_required:
            required.add_argument(f"--{name}", required=True, **kwargs)
        else:
            optional.add_argument(f"--{name}", default=param.default, **kwargs)

    # Re-add help (suppressed above via add_help=False so it lands in the group).
    optional.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit."
    )


def parse_arguments(f_dict, description):
    """Build a parser with one subcommand per entry of ``f_dict`` and dispatch
    the parsed arguments to the matching function.

    Parameters
    ----------
    f_dict : dict
        Mapping of subcommand name -> function object.
    description : str
        Description for the main argument parser.
    """
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(title="Options", dest="option")

    for key, fn in f_dict.items():
        add_params_from_signature(subparsers, key, fn)

    args = parser.parse_args()

    for key, fn in f_dict.items():
        if args.option == key:
            del args.option
            fn(**vars(args))


def wedge_list():
    """Provides a command-line interface with subcommands for
    generating wedge lists using different methods and formats. It supports
    both Stopgap batch and regular Stopgap formats.

    Subcommands
    -----------
    stopgap_batch : function
        Create wedge lists in Stopgap batch format.
    stopgap : function
        Create wedge lists in regular Stopgap format.

    Examples
    --------
    >>> wedge_list()  # Called from command line
    # To get help: wedge_list --help
    # To get help on specific subcommand: wedge_list stopgap --help
    """
    f_dict = {"stopgap_batch": wedgeutils.create_wedge_list_sg_batch, "stopgap": wedgeutils.create_wedge_list_sg}
    description = (
        "Function to create wedge lists in different formats. For help on specific option run wedge_list option --help"
    )
    parse_arguments(f_dict, description=description)


def tm_ana():
    """Provides a command-line interface for particle extraction
    from template matching results.

    Subcommands
    -----------
    extract_particles : function
        Extract particles from template matching results.

    Examples
    --------
    >>> tm_ana()  # Called from command line
    # To get help: tm_ana --help
    # To get help on specific subcommand: tm_ana extract_particles --help
    """
    f_dict = {"extract_particles": tmana.scores_extract_particles}
    description = (
        "Function to extract particles from template matching. For help on specific option run tm_ana option --help"
    )
    parse_arguments(f_dict, description=description)
