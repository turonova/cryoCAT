"""Console variable registration — one register/unregister pair for all sites.

Every module that adds or removes a name from the interactive console namespace
calls these two functions instead of importing and mutating _CONSOLE_LOCALS
directly.  The late import keeps the circular-dependency surface minimal.
"""
from __future__ import annotations

from typing import Any


def register_console_var(name: str, value: Any) -> None:
    """Bind *name* → *value* in the console namespace."""
    from cryocat.app.console.execute import _CONSOLE_LOCALS
    _CONSOLE_LOCALS[name] = value


def unregister_console_var(name: str) -> None:
    """Remove *name* from the console namespace; no-op when absent."""
    from cryocat.app.console.execute import _CONSOLE_LOCALS
    _CONSOLE_LOCALS.pop(name, None)
