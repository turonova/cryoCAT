"""AZ1 — Every plain-string callback dependency must name a layout component.

Removing a ``dcc.Store`` (or any other component) silently unregisters every
callback that names it as ``Input``, ``State``, or ``Output``, with no error
and no log message.  This test catches that class of bug for all plain-string
component ids.

Pattern-matching ids (dicts with ALL/MATCH/ALLSMALLER) are covered separately
by ``test_pattern_callbacks``; this test is for plain string ids only.

Dynamic component ids
---------------------
The only legitimate exception is a component whose id is produced exclusively
by a callback (e.g. an AG Grid node mounted into a static container div by a
render callback).  These are declared in each page module's ``DYNAMIC_IDS``
list as ``(container_id, component_id)`` pairs.  This test reads those lists
directly rather than keeping a separate copy — the two were the same fact and
had drifted out of sync.  ``test_app_ids.py`` already verifies that every
declared container_id exists in the static layout.
"""
from __future__ import annotations

import json

import pytest

from tests.app.conftest import collect_ids


# ---------------------------------------------------------------------------
# Dynamic component ids — read from each page module's DYNAMIC_IDS list
# ---------------------------------------------------------------------------

def _load_dynamic_modules():
    """Lazy import of all page modules that declare DYNAMIC_IDS."""
    from cryocat.app.suite.pages import passembly, pmotl, pnn, ptango, pdatapool, pgraphs
    return (passembly, pmotl, pnn, ptango, pdatapool, pgraphs)


def _dynamic_component_ids() -> set[str]:
    """Return every component id declared as the second element of any DYNAMIC_IDS entry."""
    ids: set[str] = set()
    for mod in _load_dynamic_modules():
        for _container_id, component_id in getattr(mod, "DYNAMIC_IDS", []):
            ids.add(component_id)
    return ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_pattern_string(raw: object) -> bool:
    """True if *raw* is a JSON-encoded dict (pattern-matching id)."""
    return isinstance(raw, str) and raw.startswith("{")


def _is_pattern_dict(raw: object) -> bool:
    """True if *raw* is a Python dict (pattern-matching id)."""
    return isinstance(raw, dict)


def _string_deps_with_callback(app) -> list[tuple[str, str, str]]:
    """Return [(component_id, callback_name, role)] for every plain-string dep.

    Roles are ``'input'``, ``'state'``, or ``'output'``.
    Pattern-matching ids (dict form or JSON-encoded ``{`` strings) are skipped;
    they are verified by ``test_pattern_callbacks``.
    """
    result: list[tuple[str, str, str]] = []

    # Inputs and State come from _callback_list (all registered callbacks).
    for item in app._callback_list:
        output_key = item.get("output", "")
        cb_entry = app.callback_map.get(output_key, {})
        cb_fn = cb_entry.get("callback")
        cb_name = getattr(cb_fn, "__name__", "<unknown>") if cb_fn else "<unknown>"

        for role in ("inputs", "state"):
            for dep in item.get(role, []):
                if not isinstance(dep, dict):
                    continue
                raw = dep.get("id")
                if not isinstance(raw, str):
                    continue  # Python dict id — pattern; skip
                if _is_pattern_string(raw):
                    continue  # JSON-encoded dict — pattern; skip
                result.append((raw, cb_name, role.rstrip("s")))  # "inputs"→"input"

    # Outputs come from callback_map (unique per output spec).
    for key, entry in app.callback_map.items():
        cb_fn = entry.get("callback")
        cb_name = getattr(cb_fn, "__name__", "<unknown>") if cb_fn else "<unknown>"
        out = entry.get("output")
        for o in (out if isinstance(out, list) else [out]):
            cid = getattr(o, "component_id", None)
            if cid is None:
                continue
            if _is_pattern_dict(cid):
                continue  # pattern dict; skip
            if isinstance(cid, str) and _is_pattern_string(cid):
                continue  # JSON-encoded pattern; skip
            if isinstance(cid, str):
                result.append((cid, cb_name, "output"))

    return result


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_every_string_dep_names_a_layout_component(suite_app):
    """Every plain-string Input/State/Output must name a component in the layout.

    Components produced only by a callback (e.g. dynamically mounted AG Grid
    nodes) are the sole legitimate exception and must be listed explicitly in
    ``_CALLBACK_CREATED_IDS`` with a written reason.

    This test detects the class of bug where removing a ``dcc.Store`` silently
    kills every callback that depends on it (confirmed: AX2, where removing
    ``{prefix}-value`` unregistered ``_plot`` in pgraphs.py with no error).
    """
    layout_ids = collect_ids(suite_app.layout)
    string_layout_ids: set[str] = {nid for nid in layout_ids if isinstance(nid, str)}
    allowed: set[str] = string_layout_ids | _dynamic_component_ids()

    violations: list[str] = []
    for comp_id, cb_name, role in _string_deps_with_callback(suite_app):
        if comp_id not in allowed:
            violations.append(f"  [{role}] {comp_id!r}  →  callback {cb_name!r}")

    assert not violations, (
        f"{len(violations)} plain-string callback dep(s) reference a component "
        f"absent from the layout and the DYNAMIC_IDS declarations:\n"
        + "\n".join(sorted(set(violations)))
    )


def test_callback_created_ids_are_referenced(suite_app):
    """Every id declared in a DYNAMIC_IDS list must be referenced by at least one callback.

    Stale entries silently paper over future bugs — an id removed from both the
    layout and from callbacks should be removed from its DYNAMIC_IDS list too.
    """
    referenced: set[str] = {
        comp_id
        for comp_id, _cb, _role in _string_deps_with_callback(suite_app)
    }
    declared = _dynamic_component_ids()
    stale = sorted(declared - referenced)
    assert not stale, (
        "DYNAMIC_IDS contains component ids not referenced by any callback "
        "(remove the stale entries from the relevant page module's DYNAMIC_IDS):\n"
        + "\n".join(f"  {sid!r}" for sid in stale)
    )
