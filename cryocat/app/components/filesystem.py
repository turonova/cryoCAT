"""Server-side filesystem helpers for the file-browser modal.

No Dash imports.  All functions are pure so they can be unit-tested
without spinning up a Dash app.

Security note: this module exposes the filesystem the app process can already
read.  See GUI_CONVENTIONS.md §12 — the app must be bound to 127.0.0.1
(accessed via SSH forwarding) not 0.0.0.0 (reachable on a shared login node).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Entry:
    name: str
    path: str          # absolute, native-separator
    is_dir: bool
    size: int | None   # bytes; None for directories
    mtime: float | None


def list_dir(
    path: str | Path,
    *,
    extensions: tuple[str, ...] = (),
    show_hidden: bool = False,
) -> tuple[list[Entry], str | None]:
    """List *path*: directories first, then files, both sorted case-insensitively.

    Returns ``(entries, error)``.  A ``PermissionError`` or missing path is
    returned as an error string, never raised.  Entries that themselves are
    unreadable are listed but flagged ``size=None``.

    Extension matching is case-insensitive; accepts ``".em"`` and ``"em"`` alike.
    Directories are never filtered by extension.
    """
    p = Path(path).resolve()

    exts: tuple[str, ...] = tuple(
        ("." + e.lstrip(".")).lower() for e in extensions if e
    )

    try:
        raw = list(p.iterdir())
    except PermissionError as exc:
        return [], str(exc)
    except FileNotFoundError:
        return [], f"Directory not found: {p}"
    except NotADirectoryError:
        return [], f"Not a directory: {p}"
    except OSError as exc:
        return [], str(exc)

    dirs: list[Entry] = []
    files: list[Entry] = []

    for child in raw:
        name = child.name
        if not show_hidden and name.startswith("."):
            continue

        try:
            is_dir = child.is_dir()
        except OSError:
            continue  # skip entries the OS cannot inspect (e.g. Windows junctions without access)

        if not is_dir and exts and child.suffix.lower() not in exts:
            continue

        try:
            st = child.stat()
            size = st.st_size if not is_dir else None
            mtime = st.st_mtime
        except OSError:
            size = None
            mtime = None

        try:
            abs_path = str(child.resolve())
        except OSError:
            abs_path = str(child)

        entry = Entry(name=name, path=abs_path, is_dir=is_dir, size=size, mtime=mtime)
        (dirs if is_dir else files).append(entry)

    dirs.sort(key=lambda e: e.name.lower())
    files.sort(key=lambda e: e.name.lower())

    return dirs + files, None


def breadcrumbs(path: str | Path) -> list[tuple[str, str]]:
    """Return ``[(display_name, absolute_path), …]`` from root to *path*.

    The root entry uses the drive/root as display name.
    ``breadcrumbs("/")`` returns exactly one entry and does not loop.
    """
    p = Path(path).resolve()
    parts: list[tuple[str, str]] = []

    current = p
    while True:
        display = current.name if current.name else str(current)
        parts.append((display, str(current)))
        parent = current.parent
        if parent == current:
            break
        current = parent

    parts.reverse()
    return parts


def resolve_input(text: str) -> tuple[str, str | None]:
    """Expand ``~`` and ``$VARS``, make absolute, normalise.

    Returns ``(resolved_path, error_or_None)``.  On success ``error`` is
    ``None``; on failure ``resolved_path`` is the original text.
    """
    if not text or not text.strip():
        return "", "No path entered"

    text = text.strip()

    try:
        expanded = os.path.expandvars(os.path.expanduser(text))
        resolved = str(Path(expanded).resolve())
        return resolved, None
    except Exception as exc:
        return text, str(exc)


def validate(
    path: str,
    *,
    mode: str,
    extensions: tuple[str, ...] = (),
) -> str | None:
    """Return a human-readable problem string, or ``None`` if *path* is acceptable.

    Modes:

    * ``"open"``      — *path* must be an existing file.
    * ``"directory"`` — *path* must be an existing directory.
    * ``"save"``      — parent directory must exist; file need not (warns if it
      does, but still returns ``None``).

    Extension checking applies only to ``"open"`` mode.
    """
    p = Path(path)

    if mode == "open":
        if not p.exists():
            return f"File not found: {path}"
        if p.is_dir():
            return f"Path is a directory, not a file: {path}"
        exts = tuple(("." + e.lstrip(".")).lower() for e in extensions if e)
        if exts and p.suffix.lower() not in exts:
            return f"Extension {p.suffix!r} not in {exts}"
        return None

    if mode == "directory":
        if not p.exists():
            return f"Directory not found: {path}"
        if not p.is_dir():
            return f"Path is a file, not a directory: {path}"
        return None

    if mode == "save":
        parent = p.parent
        if not parent.exists():
            return f"Parent directory does not exist: {parent}"
        if p.is_dir():
            return f"Path is an existing directory — provide a filename: {path}"
        # existing file: warn but accept (return None so caller may still proceed)
        return None

    return f"Unknown mode: {mode!r}"
