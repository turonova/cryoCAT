"""Pattern-matching callback key sets must match component key sets.

A pattern-matching callback that uses a different number of keys than its
target components lands in a separate Dash internal bucket and is silently
dead (confirmed from Dash 4.1.0 `dependencies.py`: key-set mismatch →
immediate False from `id_matches`, so the callback never fires and its output
is always discarded).

This test catches that class of bug:

  for every pattern P registered by a callback:
    if P's ``type`` value appears in any static or built-on-demand component:
      assert P's key set is in the set of key sets that components of
      that type actually use.

AP1 coverage
------------
In addition to the static layout, the test calls every dynamic form builder
directly — passembly loading / operation forms, the save-dialog writer forms,
pmemthick stage forms, and pnn forms — so their id-types enter the check.
A bug that was previously invisible because the type was absent from the
static layout now causes a test failure.

Design rationale
----------------
Only patterns whose ``type`` string IS present in the static layout *or* is
produced by a called builder are checked.  Patterns for types that cannot
be reached (pool-picker items, file-browser entries, …) are skipped
automatically via ``type not in comp_keys_by_type``.

Patterns whose ``type`` is itself a wildcard (``ALL``, ``MATCH``) are skipped
because they can match any type and cannot be checked this way.

Dash 4.1.0 serialises ALL/MATCH/ALLSMALLER as JSON arrays in
``_callback_list``; Python dict ids from ``callback_map`` carry the real
Python objects.  Both forms are handled by ``_is_wildcard_val``.
"""
from __future__ import annotations

import json

import pytest

from tests.app.conftest import collect_ids

# ---------------------------------------------------------------------------
# Wildcard detection helpers
# ---------------------------------------------------------------------------

try:
    from dash import ALL as _ALL, MATCH as _MATCH, ALLSMALLER as _ALLSMALLER
    _PY_WILDCARDS = {_ALL, _MATCH, _ALLSMALLER}
except ImportError:
    _PY_WILDCARDS: set = set()

_JSON_WILDCARDS = (["ALL"], ["MATCH"], ["ALLSMALLER"])


def _is_wildcard_val(v) -> bool:
    """True if *v* is a Dash wildcard (ALL, MATCH, ALLSMALLER)."""
    if isinstance(v, list) and v in _JSON_WILDCARDS:
        return True
    try:
        return v in _PY_WILDCARDS
    except TypeError:
        return False


def _parse_id(raw) -> dict | None:
    """Return a parsed dict if *raw* is a pattern dict or JSON dict-string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.startswith("{"):
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None


def _is_pattern(id_dict: dict) -> bool:
    """True if *id_dict* contains at least one wildcard value."""
    return any(_is_wildcard_val(v) for v in id_dict.values())


def _pattern_key(id_dict: dict) -> tuple:
    """Stable hashable key representing a unique pattern (key-set + fixed values)."""
    parts = []
    for k in sorted(id_dict):
        v = id_dict[k]
        parts.append((k, "__WILD__" if _is_wildcard_val(v) else v))
    return tuple(parts)


# ---------------------------------------------------------------------------
# Collect pattern deps from _callback_list and callback_map
# ---------------------------------------------------------------------------

def _all_pattern_deps(app) -> list[tuple[str, dict]]:
    """Return [(role, id_dict)] for all pattern-matching deps.

    In Dash 4.1.0:
    * ``_callback_list`` entries carry "inputs" and "state" as lists of
      ``{"id": ..., "property": ...}`` dicts, but "output" as a plain string.
    * ``callback_map`` carries Output objects with ``.component_id``.

    role is one of 'output', 'input', 'state'.
    """
    result = []
    # inputs and state come from _callback_list
    for item in app._callback_list:
        for role in ("inputs", "state"):
            for dep in item.get(role, []):
                if not isinstance(dep, dict):
                    continue
                raw = dep.get("id")
                d = _parse_id(raw)
                if d is None or not _is_pattern(d):
                    continue
                canonical = "input" if role == "inputs" else "state"
                result.append((canonical, d))
    # outputs come from callback_map
    for entry in app.callback_map.values():
        out = entry["output"]
        for o in (out if isinstance(out, list) else [out]):
            cid = getattr(o, "component_id", None)
            d = _parse_id(cid)
            if d is None or not _is_pattern(d):
                continue
            result.append(("output", d))
    return result


# ---------------------------------------------------------------------------
# AP1: dynamic-form builder coverage
# ---------------------------------------------------------------------------

def _dynamic_form_ids() -> tuple[dict[str, set[frozenset]], set[str]]:
    """Call every dynamic-form builder directly and collect component key sets.

    Returns
    -------
    extra : dict[str, set[frozenset]]
        {id_type: set_of_frozensets_of_keys} — merged across all built forms.
        Types already in the static layout get redundant entries (same key sets);
        types absent from the static layout gain their first entry, enabling
        the pattern check for those types.
    hint_owners : set[str]
        Owner values found in path-exists-hint components in the built forms.
        Used to update ``comp_hint_owners`` so the AO2 skip can distinguish
        forms that actually have path parameters.
    """
    from cryocat.app import formgen

    extra: dict[str, set[frozenset]] = {}
    hint_owners: set[str] = set()

    def _add(rows_or_component) -> None:
        for nid in collect_ids(rows_or_component):
            if not isinstance(nid, tuple):
                continue
            d = dict(nid)
            t = d.get("type")
            if t:
                extra.setdefault(t, set()).add(frozenset(d.keys()))
            if t == "path-exists-hint":
                owner = d.get("owner")
                if owner is not None:
                    hint_owners.add(owner)

    # ── passembly: loading forms ─────────────────────────────────────────────
    # These are rendered dynamically per selected loader; absent from the static
    # layout.  Each entry in LOAD_OPS maps to a build_form call with id_extra
    # {"op": load_id}, producing 5-key IDs {type, owner, op, param, tag}.
    from cryocat.app.suite.pages.passembly import LOAD_OPS, _LOAD_ID_TYPE
    for load_id, op in LOAD_OPS.items():
        _add(formgen.build_form(
            op["method"],
            id_type=_LOAD_ID_TYPE,
            id_extra={"op": load_id},
            exclude=op.get("exclude", []),
        ))

    # ── passembly: operation forms ───────────────────────────────────────────
    # Rendered dynamically per selected operation.  Mesh ops look up the
    # GuiEntry by key (reg_key == op_id); parametric ops use getattr on
    # ParametricSurface.  Alpha-shape and intersection use custom UIs with no
    # formgen form and are skipped.
    from cryocat.app.suite.pages.passembly import OPERATIONS, _OP_ID_TYPE
    from cryocat.app import discovery as _disc
    from cryocat.analysis.assembly import ParametricSurface
    for op_id, op in OPERATIONS.items():
        cat = op.get("category")
        if cat == "mesh":
            entry = _disc.get(op_id)
            _add(formgen.build_form(
                entry,
                id_type=_OP_ID_TYPE,
                id_extra={"op": op_id},
            ))
        elif cat == "parametric":
            method = getattr(ParametricSurface, op["method_name"])
            exclude = ["input_motl", "output_path"] + list(op.get("extra_pickers", []))
            _add(formgen.build_form(
                method,
                id_type=_OP_ID_TYPE,
                id_extra={"op": op_id},
                exclude=exclude,
            ))
        # "alpha_shape" and "intersection" → custom UIs, no formgen form

    # ── save-dialog: writer forms ─────────────────────────────────────────────
    # The _on_format callback renders writer params when format changes; absent
    # from the static layout.  Only "stopgap" and "relion" produce form rows.
    # "emmotl" and "dynamo" return [] (no writer params beyond output_path).
    from cryocat.core.cryomotl import StopgapMotl, RelionMotl
    from cryocat.app.components.savedialog import _RELION_WRITER_EXCLUDE
    from cryocat.app.suite.motlsidebar import N_SLOTS

    for prefix in ["me-batch-save"] + [f"me-{i}-save" for i in range(N_SLOTS)]:
        id_type = f"{prefix}-writer-param"
        _add(formgen.build_form(StopgapMotl.write_out, id_type=id_type,
                                exclude=("output_path",)))
        _add(formgen.build_form(RelionMotl.write_out, id_type=id_type,
                                exclude=_RELION_WRITER_EXCLUDE))

    # ── pmemthick: stage forms (static layout — explicit builder check) ───────
    # All four _stage_form(...) calls appear in the layout function; they are
    # already in the static layout.  Building them here verifies the builders
    # are consistent with what the layout renders.
    from cryocat.app.suite.pages.pmemthick import (
        _stage_form, _PIPELINE_ID_TYPE, _ANALYZER_PREFIX,
        _STAGE_GENERAL, _STAGE_SURFACE, _STAGE_MATCH, _STAGE_PROFILE,
    )
    from cryocat.app.components.memthick_widgets import get_analyzer_subform
    for stage_params in (_STAGE_GENERAL, _STAGE_SURFACE, _STAGE_MATCH, _STAGE_PROFILE):
        _add(_stage_form(stage_params))
    _add(get_analyzer_subform(_ANALYZER_PREFIX))

    # ── pnn: forms (static layout — explicit builder check) ───────────────────
    # All three build_form calls appear in the pnn layout; they are already in
    # the static layout.  Building them here verifies key-set consistency.
    from cryocat.analysis.nnana import NearestNeighbors
    for cls_name, fn, excl in [
        ("nn-params",     NearestNeighbors,
         ["input_data", "nn_type", "exclude_column_name"]),
        ("nn-angular",    NearestNeighbors.get_angular_distances, []),
        ("nn-pp-angular", NearestNeighbors.get_angular_distances, []),
    ]:
        _add(formgen.build_form(fn, id_type="nn-forms-params",
                                id_extra={"cls_name": cls_name}, exclude=excl))

    # ── pcomplexes: init forms ────────────────────────────────────────────────
    # _update_init_form renders build_form(cls, id_type=_INIT, id_extra={}) for
    # the selected class; absent from the static layout.  Build all six classes
    # to cover the full union of possible parameter key sets.
    from cryocat.app.suite.pages.pcomplexes import COMPLEX_CLASSES, _INIT, _METH
    for cls_name, cls in COMPLEX_CLASSES.items():
        exclude = ["motl", "symmetry"] if cls_name == "NPC" else ["motl"]
        _add(formgen.build_form(cls, id_type=_INIT, id_extra={}, exclude=exclude))

    # ── pcomplexes: method forms ──────────────────────────────────────────────
    # _update_meth_form renders build_form(entry, id_type=_METH, id_extra={})
    # for the selected operation; absent from the static layout.  Collect
    # entries for every class to cover the full union of method signatures.
    seen_keys: set[str] = set()
    for cls in COMPLEX_CLASSES.values():
        for entry in _disc.entries_for_class(cls):
            if entry.key in seen_keys:
                continue
            seen_keys.add(entry.key)
            _add(formgen.build_form(entry, id_type=_METH, id_extra={}))

    # ── cluster: method-param forms ───────────────────────────────────────────
    # _on_cluster_select renders build_form(fn, id_type="cluster-method-param",
    # id_extra={"owner": prefix, "method": method_name}, exclude=[...]) for
    # the selected clustering algorithm; absent from the static layout.
    from cryocat.analysis import clustering as _clust
    for _method_name, _fn in [
        ("kmeans", _clust.kmeans_cluster),
        ("dbscan", _clust.dbscan_cluster),
    ]:
        _add(formgen.build_form(
            _fn,
            id_type="cluster-method-param",
            id_extra={"owner": "_ap1_", "method": _method_name},
            exclude=["input_df", "feature_ids", "id_columns", "nan_drop", "pca_dict"],
        ))

    # ── motl editor: single-op and multi-motl param forms ────────────────────
    # generate_op_form renders build_form(getattr(Motl, method), id_type="me-op-param").
    # generate_multi_motl_form renders build_form(fn, id_type="me-multi-param").
    # Both absent from the static layout; build all gui-exposed Motl methods to
    # cover the full union of parameter key sets.
    from cryocat.core.cryomotl import Motl as _Motl
    _seen_me: set[str] = set()
    for _entry in _disc.entries_for_class(_Motl):
        if _entry.key in _seen_me:
            continue
        _seen_me.add(_entry.key)
        _add(formgen.build_form(_entry.fn, id_type="me-op-param"))
        _add(formgen.build_form(_entry.fn, id_type="me-multi-param"))

    # ── tango: support, descriptor, and feature param forms ──────────────────
    # _show_support_form, _show_desc_form, _show_feat_form render build_form
    # with id_extra={"cls_name": ...}; all absent from the static layout.
    # Build all non-Custom subclasses to cover the full union.
    from cryocat.utils.classutils import (
        get_class_names_by_parent as _gcnbp,
        get_classes_from_names as _gcfn,
    )
    from cryocat.analysis.tango import Descriptor as _TangoDesc
    _TANGO_MOD = "cryocat.analysis.tango"
    _tango_descs = _gcnbp("Descriptor", _TANGO_MOD)
    _tango_supps = _gcnbp("Support", _TANGO_MOD)
    _tango_feats = _gcnbp("Feature", _TANGO_MOD)
    _feat_desc_map = _TangoDesc.build_feature_descriptor_map(_tango_feats, _tango_descs)

    for _cls_name in _tango_supps:
        _add(formgen.build_form(
            _gcfn(_cls_name, _TANGO_MOD),
            id_type="tango-support-params",
            id_extra={"cls_name": _cls_name},
            exclude=["twist_desc"],
        ))

    for _cls_name in _tango_descs:
        if _cls_name == "CustomDescriptor":
            continue
        _excl = (
            ["input_twist", "input_motl", "nn_radius", "column_name",
             "symm", "remove_qp", "remove_duplicates", "build_unique_desc"]
            if _cls_name == "TwistDescriptor"
            else ["twist_df", "build_unique_desc"]
        )
        _add(formgen.build_form(
            _gcfn(_cls_name, _TANGO_MOD),
            id_type="tango-desc-params",
            id_extra={"cls_name": _cls_name},
            exclude=_excl,
        ))

    for _feat_name, _parent_desc in _feat_desc_map.items():
        if _parent_desc == "TwistDescriptor":
            continue
        _add(formgen.build_form(
            _gcfn(_parent_desc, _TANGO_MOD),
            id_type="tango-feat-params",
            id_extra={"cls_name": _feat_name},
            exclude=["twist_df", "build_unique_desc"],
        ))

    return extra, hint_owners


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_pattern_key_sets_match_components(suite_app):
    """For every pattern-matching dep whose 'type' appears in the static layout
    or is produced by a called builder, the pattern's key set must equal a key
    set that components of that type actually use.

    Remaining skips (AS2)
    ---------------------
    1. Wildcard type (``ALL``/``MATCH``): cannot be checked by type.
    2. Type absent from all sources: pool-picker items, file-browser entries,
       and other fully-dynamic types whose builders cannot be called without
       server state (e.g. per-group pool rows).  These are skipped automatically
       via ``type_val not in comp_keys_by_type``.
    3. path-exists-hint with extra keys, and NO hint component carries extra
       keys that are a non-empty subset of the pattern's extra keys.  This
       indicates the source form type has no path-tagged parameters at all;
       ``register_form_callbacks`` registers the hint callback unconditionally
       (like the slot/elem callback), so an inert registration is by design.
       When a subset match DOES exist the source form type has path params,
       meaning a key-set mismatch here is a dead callback — let the main
       check catch it.
    4. 6-key tuple-slot patterns (slot + elem): these are registered
       unconditionally to handle Tuple-type parameters, but not all form types
       have Tuple-type parameters.  When the directly-built form produces no
       6-key components (no Tuple params in scope), the pattern remains
       unverifiable.  This is NOT a key-set mismatch — the pattern is correct
       for the kind of control it targets; no Tuple params just means it will
       never fire for the current signature.
    """
    # ── Static-layout component IDs ───────────────────────────────────────────
    all_ids = collect_ids(suite_app.layout)
    comp_keys_by_type: dict[str, set[frozenset]] = {}
    for nid in all_ids:
        if not isinstance(nid, tuple):
            continue
        d = dict(nid)
        t = d.get("type")
        if t:
            comp_keys_by_type.setdefault(t, set()).add(frozenset(d.keys()))

    # ── AP1: merge dynamically-built form IDs ─────────────────────────────────
    dyn_keys, _ = _dynamic_form_ids()
    for t, key_sets in dyn_keys.items():
        comp_keys_by_type.setdefault(t, set()).update(key_sets)

    seen: set[tuple] = set()
    failures: list[str] = []

    for role, id_dict in _all_pattern_deps(suite_app):
        type_val = id_dict.get("type")

        # Skip 1: type itself is a wildcard (matches all types; uncheckable)
        if type_val is None or _is_wildcard_val(type_val):
            continue

        pkey = _pattern_key(id_dict)
        if pkey in seen:
            continue
        seen.add(pkey)

        # Skip 2: type absent from all sources → fully dynamic / unreachable
        if type_val not in comp_keys_by_type:
            continue

        # Skip 4: 6-key tuple-slot patterns — registered unconditionally;
        # only fire when the form has Tuple-type parameters.  When the built
        # forms produce no 6-key components for this type, the pattern is
        # correct but unverifiable against a concrete component.
        if "slot" in id_dict and "elem" in id_dict:
            if frozenset(id_dict.keys()) not in comp_keys_by_type[type_val]:
                continue

        # Skip 3: path-exists-hint with extra keys, and no hint component
        # shares those same extra keys (subset check).  When no hint component
        # has extra keys that are a non-empty subset of the pattern's extras,
        # the source form type has no path-tagged parameters; the registration
        # is inert by design.  When a subset match exists, the source form type
        # DOES have path params, so a key-set mismatch is a dead callback —
        # the main check below catches it.
        # id_type is structural (present in every form-level hint); not a
        # discriminating "extra".  Including it prevents false-positive subset
        # matches from unrelated form types (e.g. passembly hints matching a
        # pnn pattern, letting an inert registration slip past Skip 3).
        _HINT_BASE = frozenset({"type", "owner", "param", "tag", "id_type"})
        if type_val == "path-exists-hint" and not frozenset(id_dict.keys()) <= _HINT_BASE:
            pattern_extra = frozenset(id_dict.keys()) - _HINT_BASE
            if not any(
                0 < len(ks - _HINT_BASE) and (ks - _HINT_BASE) <= pattern_extra
                for ks in comp_keys_by_type.get("path-exists-hint", set())
            ):
                continue

        pattern_keyset = frozenset(id_dict.keys())
        component_keysets = comp_keys_by_type[type_val]

        if pattern_keyset not in component_keysets:
            failures.append(
                f"[{role}] pattern {id_dict!r}\n"
                f"  key set  : {sorted(pattern_keyset)}\n"
                f"  component: {[sorted(k) for k in component_keysets]}"
            )

    assert not failures, (
        f"{len(failures)} pattern(s) have key sets that do not match "
        f"any static or built component of their type:\n"
        + "\n".join(f"  {f}" for f in failures)
    )
