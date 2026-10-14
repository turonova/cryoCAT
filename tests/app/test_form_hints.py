"""DD1 — every callable that formgen.build_form renders has resolvable type hints.

A function whose typing.get_type_hints raises causes build_form to fall back to
an empty hints dict.  Every parameter in the resulting form is rendered as a
plain text field with no type coercion — booleans always read True, integers
stay as strings, Literal dropdowns become free-text inputs.  This is not
cosmetic.

The test is parametrised by (source, function) so a failure names the exact
offender.  The environment must have the app's full dependency set available
because get_type_hints resolves into the live runtime namespace.
"""
from __future__ import annotations

import importlib
import inspect
import typing

import pytest

# ---------------------------------------------------------------------------
# Boot: import every module that registers GUI_REGISTRY entries so that the
# @gui_exposed decorators fire before we read the registry.
# ---------------------------------------------------------------------------

for _mod_name in [
    "cryocat.analysis.nnana",
    "cryocat.analysis.pana",
    "cryocat.analysis.sta",
    "cryocat.analysis.tango",
    "cryocat.analysis.visplot",
    "cryocat.core.cryomotl",
    "cryocat.core.cryomap",
    "cryocat.core.surface",
    "cryocat.analysis.assembly",
    "cryocat.analysis.clustering",
]:
    importlib.import_module(_mod_name)

from cryocat.utils.classutils import (
    GUI_REGISTRY,
    get_class_names_by_parent,
    get_classes_from_names,
)


# ---------------------------------------------------------------------------
# Helper: mirror what formgen.build_form does.
# For a class, build_form calls get_type_hints on __init__ (not the class),
# because get_type_hints(cls) only covers class-level annotations and misses
# every __init__ parameter annotation.
# ---------------------------------------------------------------------------

def _hints_target(fn):
    return fn.__init__ if inspect.isclass(fn) else fn


# ---------------------------------------------------------------------------
# Source 1: GUI_REGISTRY
# All entries whose .fn was decorated with @gui_exposed.
# ---------------------------------------------------------------------------

_registry_cases: list[pytest.param] = [
    pytest.param(_hints_target(entry.fn), id=f"registry::{key}")
    for key, entry in sorted(GUI_REGISTRY.items())
]


# ---------------------------------------------------------------------------
# Source 2: LOAD_OPS in passembly.py
# Hand-written dict mapping loader IDs to method callables; every entry is
# passed to formgen.build_form regardless of kind.
# ---------------------------------------------------------------------------

from cryocat.app.suite.pages.passembly import LOAD_OPS  # noqa: E402

_load_ops_cases: list[pytest.param] = [
    pytest.param(_hints_target(entry["method"]), id=f"LOAD_OPS::{key}")
    for key, entry in LOAD_OPS.items()
]


# ---------------------------------------------------------------------------
# Source 3: hardcoded form callables outside GUI_REGISTRY
# Each entry below corresponds to an explicit build_form(fn, ...) call in
# the app source.  A source here means at least one page will render a form
# from this callable at runtime.
#
# Pages → callables:
#   pnn.py              NearestNeighbors (class → __init__), get_angular_distances, trace_chains
#   psta.py             generate_angles
#   ppana.py            generate_wedge_mask, generate_angles
#   anglesbuilder.py    generate_angles (same function)
#   putilities.py       generate_wedge_mask (same function, via discovery entry)
#   tablecluster.py     kmeans_cluster, dbscan_cluster
#   savedialog.py       StopgapMotl.write_out, RelionMotl.write_out
#   memthick_widgets.py IntensityProfileAnalyzer (class → __init__)
#   pmemthick.py        run_full_pipeline
#   pcomplexes.py       CnComplex / DnComplex / NPC / TetrahedralComplex /
#                       OctahedralComplex / IcosahedralComplex (all via COMPLEX_CLASSES)
#   ptango.py           every Descriptor and Feature subclass (via get_class_names_by_parent)
# ---------------------------------------------------------------------------

from cryocat.analysis.nnana import NearestNeighbors, trace_chains  # noqa: E402
from cryocat.utils.geom import generate_angles  # noqa: E402
from cryocat.core.cryowedge import generate_wedge_mask  # noqa: E402
from cryocat.analysis import clustering as _cm  # noqa: E402
from cryocat.core.cryomotl import StopgapMotl, RelionMotl  # noqa: E402
from cryocat.analysis.memthick import IntensityProfileAnalyzer, run_full_pipeline  # noqa: E402
from cryocat.app.suite.pages.pcomplexes import COMPLEX_CLASSES  # noqa: E402

_descriptor_names = get_class_names_by_parent("Descriptor", "cryocat.analysis.tango")
_feature_names = get_class_names_by_parent("Feature", "cryocat.analysis.tango")

_hardcoded_cases: list[pytest.param] = [
    pytest.param(NearestNeighbors.__init__, id="pnn::NearestNeighbors.__init__"),
    pytest.param(NearestNeighbors.get_angular_distances,
                 id="pnn::NearestNeighbors.get_angular_distances"),
    pytest.param(trace_chains, id="pnn::trace_chains"),
    pytest.param(generate_angles, id="psta_ppana_angles::generate_angles"),
    pytest.param(generate_wedge_mask, id="ppana_putilities::generate_wedge_mask"),
    pytest.param(_cm.kmeans_cluster, id="tablecluster::kmeans_cluster"),
    pytest.param(_cm.dbscan_cluster, id="tablecluster::dbscan_cluster"),
    pytest.param(StopgapMotl.write_out, id="savedialog::StopgapMotl.write_out"),
    pytest.param(RelionMotl.write_out, id="savedialog::RelionMotl.write_out"),
    pytest.param(IntensityProfileAnalyzer.__init__,
                 id="memthick_widgets::IntensityProfileAnalyzer.__init__"),
    pytest.param(run_full_pipeline, id="pmemthick::run_full_pipeline"),
] + [
    pytest.param(cls.__init__, id=f"pcomplexes::{name}.__init__")
    for name, cls in COMPLEX_CLASSES.items()
] + [
    pytest.param(
        get_classes_from_names(name, "cryocat.analysis.tango").__init__,
        id=f"ptango::Descriptor::{name}.__init__",
    )
    for name in _descriptor_names
] + [
    pytest.param(
        get_classes_from_names(name, "cryocat.analysis.tango").__init__,
        id=f"ptango::Feature::{name}.__init__",
    )
    for name in _feature_names
]


# ---------------------------------------------------------------------------
# Deduplicate: some functions appear in multiple sources (e.g. generate_angles
# is called from psta, ppana, and anglesbuilder with the same callable object).
# Keep one parametrised case per unique function; use the first id encountered.
# ---------------------------------------------------------------------------

_seen: set[int] = set()
_all_cases: list[pytest.param] = []
for _p in _registry_cases + _load_ops_cases + _hardcoded_cases:
    fn = _p.values[0]
    fid = id(fn)
    if fid not in _seen:
        _seen.add(fid)
        _all_cases.append(_p)


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", _all_cases)
def test_form_callable_hints_resolvable(fn):
    """typing.get_type_hints must not raise on any callable formgen renders.

    A failure means formgen.build_form fell back to an empty hints dict for
    this callable.  Every parameter in that form is rendered as a plain text
    field — booleans always read True, integers stay strings, Literal
    dropdowns become free-text inputs.  This is not cosmetic.

    Fix: remove inner string quotes from return annotations under PEP 563
    (``-> "Foo" | None`` → ``-> Foo | None``), or qualify TYPE_CHECKING-only
    names via their module (``MotlSource`` → ``cryomotl.MotlSource``).
    """
    try:
        typing.get_type_hints(fn)
    except Exception as exc:
        fn_name = getattr(fn, "__qualname__", None) or repr(fn)
        pytest.fail(
            f"typing.get_type_hints({fn_name!r}) raised — "
            f"the form renders untyped and values are not coerced.\n"
            f"{type(exc).__name__}: {exc}"
        )
