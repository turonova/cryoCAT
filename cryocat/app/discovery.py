"""GUI registry discovery — the single query layer over :data:`GUI_REGISTRY`.

Usage pattern
-------------
``server.py`` calls :func:`load_registry` **once** before importing any app
module that reads the registry at import time.  App code then queries via the
functions below; it must never walk ``inspect.getmembers`` or read
``GUI_REGISTRY`` directly.

    from cryocat.app import discovery
    discovery.load_registry()           # called once from server.py
    ops = discovery.single_motl_ops()   # list[GuiEntry], sorted by label

Adding a new GUI-exposed callable
----------------------------------
1. Decorate it with ``@gui_exposed`` in one of the modules listed in
   :data:`GUI_MODULES` (or add the module to that tuple).
2. That's it.  ``load_registry`` imports the module; the decorator fires;
   the entry appears in every query function automatically.
"""
from __future__ import annotations

import importlib

from cryocat.utils.classutils import GUI_REGISTRY, GuiCategory, GuiEntry

# Modules that carry ``@gui_exposed`` decorators.  Importing them causes the
# decorators to fire and populate GUI_REGISTRY.  A module listed here that
# cannot be imported raises ImportError immediately (hard error, not a warning).
GUI_MODULES: tuple[str, ...] = (
    "cryocat.core.cryomotl",
    "cryocat.utils.geom",
    "cryocat.core.cryowedge",
    "cryocat.analysis.structure",
    "cryocat.utils.ioutils",
    "cryocat.core.cryomap",
)

_loaded: bool = False


def load_registry(extra_modules: tuple[str, ...] = ()) -> dict[str, GuiEntry]:
    """Import every module in :data:`GUI_MODULES` so their decorators fire.

    Idempotent: a second call with no ``extra_modules`` returns immediately.
    Unimportable modules raise :exc:`ImportError` naming the offending module.

    Parameters
    ----------
    extra_modules:
        Additional module names to import (used by tests to register fixture
        modules without modifying :data:`GUI_MODULES`).

    Returns
    -------
    dict[str, GuiEntry]
        Snapshot of :data:`GUI_REGISTRY` after loading.
    """
    global _loaded
    for mod_name in GUI_MODULES + extra_modules:
        try:
            importlib.import_module(mod_name)
        except ImportError as exc:
            raise ImportError(
                f"GUI_MODULES lists {mod_name!r} but it cannot be imported: {exc}"
            ) from exc
    _loaded = True
    return dict(GUI_REGISTRY)


def _ensure_loaded() -> None:
    if not _loaded:
        load_registry()


# ---------------------------------------------------------------------------
# Query functions — all sorted by label for deterministic UI order
# ---------------------------------------------------------------------------

def entries(
    *,
    category: GuiCategory | None = None,
    owner: str | None = None,
) -> list[GuiEntry]:
    """All registered entries, optionally filtered by category and/or owner."""
    _ensure_loaded()
    result = list(GUI_REGISTRY.values())
    if category is not None:
        result = [e for e in result if e.category == category]
    if owner is not None:
        result = [e for e in result if e.owner == owner]
    return sorted(result, key=lambda e: (e.group or "\x7f", e.order, e.label))


def single_motl_ops() -> list[GuiEntry]:
    """Motl operations that accept a single motl (``motls`` spec is ``None``)."""
    _ensure_loaded()
    return sorted(
        (
            e for e in GUI_REGISTRY.values()
            if e.category == GuiCategory.MOTL_OP and e.motls is None
        ),
        key=lambda e: (e.group or "\x7f", e.order, e.label),
    )


def multi_motl_ops() -> list[GuiEntry]:
    """Motl operations that accept multiple motls (``motls`` spec is set)."""
    _ensure_loaded()
    return sorted(
        (
            e for e in GUI_REGISTRY.values()
            if e.category == GuiCategory.MOTL_OP and e.motls is not None
        ),
        key=lambda e: (e.group or "\x7f", e.order, e.label),
    )


def readers() -> list[GuiEntry]:
    """File-reading callables registered with ``category="reader"``."""
    _ensure_loaded()
    return sorted(
        (e for e in GUI_REGISTRY.values() if e.category == GuiCategory.READER),
        key=lambda e: (e.group or "\x7f", e.order, e.label),
    )


def standalone_builders() -> list[GuiEntry]:
    """Builder functions flagged ``standalone=True`` (appear on Utilities page)."""
    _ensure_loaded()
    return sorted(
        (
            e for e in GUI_REGISTRY.values()
            if e.category == GuiCategory.BUILDER and e.standalone
        ),
        key=lambda e: (e.group or "\x7f", e.order, e.label),
    )


def get(key: str) -> GuiEntry:
    """Return the entry for ``key`` or raise :exc:`KeyError`."""
    _ensure_loaded()
    try:
        return GUI_REGISTRY[key]
    except KeyError:
        raise KeyError(f"No GUI entry registered for key {key!r}") from None


def entries_for_class(
    cls: type,
    *,
    stop_classes: tuple[type, ...] = (),
) -> list[GuiEntry]:
    """Collect entries for *cls* and its ancestors, respecting the MRO.

    Iterates the MRO and collects entries whose ``owner`` matches each class in
    turn, stopping before any class listed in *stop_classes* (or ``object``).
    A method that is overridden in a subclass appears only once under the
    subclass's key (MRO order ensures first-match wins).

    Parameters
    ----------
    cls:
        The concrete class to collect entries for.
    stop_classes:
        Ancestor classes at which MRO traversal stops (exclusive).

    Returns
    -------
    list[GuiEntry]
        Merged, deduplicated, label-sorted list of entries.
    """
    _ensure_loaded()
    seen_names: set[str] = set()
    result: list[GuiEntry] = []
    for klass in cls.__mro__:
        if klass is object:
            break
        if any(klass is sc for sc in stop_classes):
            break
        mod = getattr(klass, "__module__", "") or ""
        owner = f"{mod}.{klass.__qualname__}" if mod else klass.__qualname__
        for e in GUI_REGISTRY.values():
            if e.owner != owner:
                continue
            fn_name = e.fn.__name__
            if fn_name not in seen_names:
                seen_names.add(fn_name)
                result.append(e)
    return sorted(result, key=lambda e: (e.group or "\x7f", e.order, e.label))


def gui_ready(entry: GuiEntry) -> tuple[bool, str]:
    """Check whether ``entry`` can be rendered into a form without errors.

    Returns
    -------
    (renderable, reason)
        ``renderable`` is ``True`` when every non-hidden parameter maps to a
        known widget.  ``reason`` is empty on success or explains the gap.
    """
    import inspect
    import typing
    from cryocat.utils.classutils import resolve_param_type, TYPE_HANDLERS

    hide = entry.hide | {"self", "cls"}
    try:
        sig = inspect.signature(entry.fn)
        hints = typing.get_type_hints(entry.fn)
    except Exception as exc:
        return False, f"cannot introspect signature: {exc}"

    for pname, param in sig.parameters.items():
        if pname in hide:
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        ann = hints.get(pname, param.annotation)
        tag, _ = resolve_param_type(ann)
        if tag not in TYPE_HANDLERS:
            return False, f"param {pname!r} resolved to unknown tag {tag!r}"
        if TYPE_HANDLERS[tag]["widget"] not in _KNOWN_WIDGETS:
            return False, (
                f"param {pname!r} tag {tag!r} maps to unknown widget "
                f"{TYPE_HANDLERS[tag]['widget']!r}"
            )
    return True, ""


# Widgets realised by formgen.WIDGET_FACTORIES — kept in sync with that dict.
# A mismatch here surfaces via gui_ready() / test_gui_exposed.py.
_KNOWN_WIDGETS: frozenset[str] = frozenset({
    "path", "triplet", "csv_text", "text",
    "number", "bool", "dropdown", "rotation", "tuple", "listlike",
})


# ---------------------------------------------------------------------------
# CLI: python -m cryocat.app.discovery --report
# ---------------------------------------------------------------------------

def _report() -> None:
    load_registry()
    motl_ops   = single_motl_ops()
    multi_ops  = multi_motl_ops()
    builders   = standalone_builders()

    print(f"GUI_REGISTRY: {len(GUI_REGISTRY)} entries total")
    print(f"  single-motl ops : {len(motl_ops)}")
    print(f"  multi-motl ops  : {len(multi_ops)}")
    print(f"  standalone bldrs: {len(builders)}")
    print()

    for section, lst in (
        ("Single-motl ops", motl_ops),
        ("Multi-motl ops",  multi_ops),
        ("Standalone builders", builders),
    ):
        sep = "-" * max(0, 40 - len(section))
        print(f"-- {section} -{sep}-")
        for e in lst:
            ready, reason = gui_ready(e)
            flag = "OK" if ready else "!!"
            extra = f"  [{reason}]" if reason else ""
            print(f"  {flag} {e.key:<40} {e.label}{extra}")
        print()


if __name__ == "__main__":
    import sys
    if "--report" in sys.argv:
        _report()
    else:
        print("Usage: python -m cryocat.app.discovery --report")
