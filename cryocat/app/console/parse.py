"""Console command parser and validator.

Converts a single line of user input into a :class:`Command` — a frozen
dataclass that carries the desugared Python source, AST node, and command
kind so execution can proceed without re-parsing.

All errors that the user should see are raised as :class:`ConsoleSyntaxError`
(bad Python) or :class:`ConsoleRejected` (disallowed construct).

Public API
----------
* :class:`ConsoleSyntaxError`
* :class:`ConsoleRejected`
* :class:`Command`
* :func:`parse`
* :func:`validate`
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Literal

from cryocat.app.console.sugar import desugar


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConsoleSyntaxError(ValueError):
    """The command text is not syntactically valid Python after desugaring."""

    def __init__(self, msg: str, *, line: int | None = None, col: int | None = None):
        self.line = line
        self.col = col
        super().__init__(msg)


class ConsoleRejected(ValueError):
    """The command is syntactically valid but uses a disallowed construct."""


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Command:
    """A parsed, validated console command ready for execution.

    Attributes
    ----------
    raw : str
        Original user input (stripped).
    src : str
        Desugared Python source (``#n`` → ``motl_n``); this is what the
        script projection records verbatim.
    kind : Literal["expr", "assign", "pool_assign", "directive"]
        *expr*        — bare expression; result bound to ``_``, pool unchanged.
        *assign*      — ``name = expr``; name bound in console locals.
        *pool_assign* — ``motl_N = expr``; replaces pool entry *N*.
        *directive*   — ``help``, ``vars``, ``history``, or ``clear``.
    target : str | None
        For *assign*: the local variable name.
        For *pool_assign*: the digit string *N* (e.g. ``"3"`` for ``motl_3``).
        For *directive*: optional argument (e.g. key passed to ``help``).
        ``None`` for *expr*.
    directive : str | None
        The directive keyword for *directive* commands; ``None`` otherwise.
    node : ast.Module
        Parsed AST module; body[0] is the single statement.
        For *directive* commands this is a placeholder empty module.
    """

    raw: str
    src: str
    kind: Literal["expr", "assign", "pool_assign", "directive"]
    target: str | None
    directive: str | None
    node: ast.Module


# ---------------------------------------------------------------------------
# Rejected node types
# ---------------------------------------------------------------------------

_REJECTED_TYPES: list = [
    ast.Import, ast.ImportFrom,
    ast.FunctionDef, ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.While, ast.For, ast.AsyncFor,
    ast.Try,
    ast.With, ast.AsyncWith,
    ast.Lambda,
]
try:
    _REJECTED_TYPES.append(ast.TryStar)   # Python 3.11+
except AttributeError:
    pass
_REJECTED_TYPES_TUPLE = tuple(_REJECTED_TYPES)

_BANNED_NAMES: frozenset[str] = frozenset({
    "__builtins__", "__globals__", "__class__", "__import__",
    "__loader__", "__spec__",
})

_DIRECTIVES: frozenset[str] = frozenset({"help", "vars", "history", "clear"})

_POOL_VAR_RE = re.compile(r"^motl_(\d+)$")


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def validate(node: ast.Module) -> None:
    """Walk *node* and raise :class:`ConsoleRejected` on any disallowed construct.

    Checks performed:
    * No import / def / class / lambda / loop / try / with statements.
    * No attribute access where the attribute name starts with ``_``.
    * No reference to ``__builtins__``, ``__globals__``, or ``__class__``.
    """
    for n in ast.walk(node):
        if isinstance(n, _REJECTED_TYPES_TUPLE):
            tname = type(n).__name__
            raise ConsoleRejected(f"{tname} is not allowed in console commands")
        if isinstance(n, ast.Attribute) and n.attr.startswith("_"):
            raise ConsoleRejected(
                f"access to private/dunder attribute {n.attr!r} is not allowed"
            )
        if isinstance(n, ast.Name) and n.id in _BANNED_NAMES:
            raise ConsoleRejected(f"access to {n.id!r} is not allowed")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse(text: str) -> Command:
    """Parse a console command line into a :class:`Command`.

    Parameters
    ----------
    text : str
        Raw user input (may have leading/trailing whitespace).

    Returns
    -------
    Command

    Raises
    ------
    ConsoleSyntaxError
        If the desugared text is not valid Python.
    ConsoleRejected
        If the text contains a disallowed Python construct.
    """
    raw = text.strip()
    if not raw:
        raise ConsoleSyntaxError("empty command")

    # -- Directives (checked before desugaring) --------------------------------
    for d in _DIRECTIVES:
        if raw == d or raw.startswith(d + " "):
            arg = raw[len(d):].strip() or None
            return Command(
                raw=raw,
                src=raw,
                kind="directive",
                target=arg,
                directive=d,
                node=ast.Module(body=[], type_ignores=[]),
            )

    # -- Desugar #n → motl_n --------------------------------------------------
    src = desugar(raw)

    # -- Python parse ----------------------------------------------------------
    try:
        node = ast.parse(src, mode="exec")
    except SyntaxError as exc:
        raise ConsoleSyntaxError(
            str(exc), line=exc.lineno, col=exc.offset
        ) from exc

    if len(node.body) != 1:
        raise ConsoleRejected("only a single statement is allowed per command")

    stmt = node.body[0]

    # -- Validate (before kind detection so errors are clear) ------------------
    validate(node)

    # -- Kind detection --------------------------------------------------------
    if isinstance(stmt, ast.Expr):
        return Command(raw=raw, src=src, kind="expr", target=None, directive=None, node=node)

    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            raise ConsoleRejected(
                "only simple single-name assignments are supported (e.g. x = expr)"
            )
        target_name = stmt.targets[0].id
        m = _POOL_VAR_RE.match(target_name)
        if m:
            # motl_N = expr  →  pool assign
            return Command(
                raw=raw, src=src, kind="pool_assign",
                target=m.group(1), directive=None, node=node,
            )
        return Command(
            raw=raw, src=src, kind="assign",
            target=target_name, directive=None, node=node,
        )

    raise ConsoleRejected(f"unsupported statement type: {type(stmt).__name__}")
