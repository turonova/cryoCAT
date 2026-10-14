"""Tests for the membrane thickness page and its support modules.

Covers:
- Analysis helpers: resolve_thickness_csv, parse_membrane_names, motl_to_pool_rows,
  labelled_motl_payload.
- Membrane results registry: register, get, remove, make_handle.
- Tuple-widget pipeline: resolve_param_type, TYPE_HANDLERS["Tuple"], generate_kwargs
  round-trip.
- Composite widget readers in memthick_widgets: read_label_dict, read_per_membrane_mode,
  read_analyzer_kwargs.
- Code generator in _memthick_codegen: render_pipeline_py, render_pipeline_ipynb,
  render_slurm_wrapper, end-to-end form-state to Python script.
"""
from __future__ import annotations

import ast
import json
import typing
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cryocat.utils.classutils import resolve_param_type, TYPE_HANDLERS
from cryocat.app.apputils import generate_kwargs
from cryocat.app.pool import PoolState
from cryocat.app.components import memthick_registry as mreg
from cryocat.app.components import memthick_widgets as mw
from cryocat.app.components.memthick_widgets import read_label_dict
from cryocat.app.suite.pages import _memthick_analysis as analysis_helpers
from cryocat.app.suite.pages import _memthick_codegen as codegen

_PS = PoolState.empty()


# ===========================================================================
# Analysis helpers
# ===========================================================================

def test_resolve_thickness_csv_default_suffix():
    p = analysis_helpers.resolve_thickness_csv(
        "/work/outputs", "2140_z150to400_segmented", "IMM",
    )
    assert p == Path("/work/outputs") / "2140_z150to400_segmented_IMM_thickness.csv"


def test_resolve_thickness_csv_custom_suffix():
    p = analysis_helpers.resolve_thickness_csv(
        "/work/outputs", "seg", "OMM", suffix="int_profiles.pkl",
    )
    assert p == Path("/work/outputs") / "seg_OMM_int_profiles.pkl"


def test_resolve_thickness_csv_accepts_pathlike():
    base = Path("/work/outputs")
    p = analysis_helpers.resolve_thickness_csv(base, "seg", "ER")
    assert p.parent == base
    assert p.name == "seg_ER_thickness.csv"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ER, IMM, OMM", ["ER", "IMM", "OMM"]),
        ("ER\nIMM\nOMM", ["ER", "IMM", "OMM"]),
        ("ER IMM OMM", ["ER", "IMM", "OMM"]),
        ("ER; IMM; OMM", ["ER", "IMM", "OMM"]),
        ("", []),
        (None, []),
        ("ER, ER, IMM", ["ER", "IMM"]),     # dedup
        ("  ER ,  ,  IMM ", ["ER", "IMM"]), # empty entries dropped
    ],
)
def test_parse_membrane_names(text, expected):
    assert analysis_helpers.parse_membrane_names(text) == expected


def test_motl_to_pool_rows_from_motl_like():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    motl = SimpleNamespace(df=df)
    rows = analysis_helpers.motl_to_pool_rows(motl)
    assert rows == [{"x": 1, "y": 3}, {"x": 2, "y": 4}]


def test_motl_to_pool_rows_accepts_dataframe_directly():
    df = pd.DataFrame({"x": [9]})
    rows = analysis_helpers.motl_to_pool_rows(df)
    assert rows == [{"x": 9}]


def test_motl_to_pool_rows_none_returns_empty():
    assert analysis_helpers.motl_to_pool_rows(None) == []


def test_labelled_motl_payload_round_trips():
    df1 = pd.DataFrame({"x": [1]})
    df2 = pd.DataFrame({"x": [2]})
    motls = {"IMM": (SimpleNamespace(df=df1), SimpleNamespace(df=df2))}
    payload = analysis_helpers.labelled_motl_payload(motls)
    assert payload == {"IMM": {"surface1": [{"x": 1}], "surface2": [{"x": 2}]}}


# ===========================================================================
# Membrane results registry
# ===========================================================================

@pytest.fixture(autouse=True)
def _clean_registry():
    mreg.registry.clear()
    yield
    mreg.registry.clear()


def _make_bundle(name="IMM", n_rows=10, modes=None):
    """Build a minimal MembraneResults for registry tests."""
    md = SimpleNamespace(thickness_df=pd.DataFrame({"x": list(range(n_rows))}))
    boundary_info = {
        "n_resolved": n_rows - 2,
        "n_unresolved": 2,
        "n_finite_inflection_thickness_nm": n_rows - 3,
        "by_detection_mode": modes or {"max_max": 5, "max_anchor": 3, "minima_only": 2},
    }
    return mreg.MembraneResults(
        membrane=name,
        membrane_data=md,
        thickness_results=None,
        profile_results=None,
        boundary_info=boundary_info,
        thickness_csv=f"/tmp/{name}_thickness.csv",
        pixel_size_nm=0.272,
    )


def test_register_and_get_results():
    bundle = _make_bundle("IMM")
    rid = mreg.registry.add(bundle)
    assert rid.startswith("memthick-")
    assert mreg.registry.get(rid) is bundle


def test_register_ids_are_unique():
    a = mreg.registry.add(_make_bundle("IMM"))
    b = mreg.registry.add(_make_bundle("OMM"))
    assert a != b
    assert set(mreg.registry.keys()) == {a, b}


def test_remove_results():
    rid = mreg.registry.add(_make_bundle("IMM"))
    mreg.registry.remove(rid)
    assert mreg.registry.get(rid) is None
    # No-op for unknown id.
    mreg.registry.remove("memthick-doesnotexist")


def test_get_results_returns_none_for_missing():
    assert mreg.registry.get("memthick-doesnotexist") is None


def test_make_handle_shape():
    bundle = _make_bundle("OMM", n_rows=42)
    handle = mreg.make_handle(bundle)
    assert handle == {
        "membrane": "OMM",
        "n_rows": 42,
        "n_resolved": 40,
        "n_unresolved": 2,
        "n_finite_inflection_thickness_nm": 39,
        "by_detection_mode": {"max_max": 5, "max_anchor": 3, "minima_only": 2},
        "pixel_size_nm": 0.272,
        "thickness_csv": "/tmp/OMM_thickness.csv",
    }


def test_make_handle_tolerates_missing_boundary_info():
    bundle = mreg.MembraneResults(
        membrane="ER",
        membrane_data=SimpleNamespace(thickness_df=pd.DataFrame({"x": [1]})),
        thickness_results=None,
        profile_results=None,
        boundary_info={},
        thickness_csv="/tmp/ER.csv",
    )
    handle = mreg.make_handle(bundle)
    assert handle["n_rows"] == 1
    assert handle["n_resolved"] == 0
    assert handle["by_detection_mode"] == {}
    assert handle["pixel_size_nm"] is None


# ===========================================================================
# Tuple-widget pipeline
# ===========================================================================

def test_resolve_tuple_float_pair():
    tag, extra = resolve_param_type(tuple[float, float])
    assert tag == "Tuple"
    assert extra == {"length": 2, "elem": "float"}


def test_resolve_tuple_int_triple():
    tag, extra = resolve_param_type(tuple[int, int, int])
    assert tag == "Tuple"
    assert extra == {"length": 3, "elem": "int"}


def test_resolve_heterogeneous_tuple_falls_through():
    # tuple[int, str] is not a uniform numeric tuple -> resolver should not claim it.
    tag, _ = resolve_param_type(tuple[int, str])
    assert tag != "Tuple"


def test_tuple_handler_entry_shape():
    entry = TYPE_HANDLERS["Tuple"]
    assert entry["widget"] == "tuple"
    assert callable(entry["parse"])
    assert "type" in entry["argparse"]


@pytest.mark.parametrize(
    "values, elem, expected",
    [
        ([1.5, 2.5], "float", (1.5, 2.5)),
        ([3, 4, 5], "int", (3, 4, 5)),
        ([None, 2.5], "float", None),
        ([], "float", None),
        (None, "float", None),
    ],
)
def test_parse_tuple(values, elem, expected):
    assert TYPE_HANDLERS["Tuple"]["parse"](values, elem=elem) == expected


def test_generate_kwargs_reassembles_tuple_slots():
    ids = [
        {"type": "ipa", "param": "minima_search_nm", "tag": "Tuple", "slot": 1, "elem": "float"},
        {"type": "ipa", "param": "minima_search_nm", "tag": "Tuple", "slot": 0, "elem": "float"},
    ]
    values = [4.2, 2.9]
    out = generate_kwargs(ids, values, _PS)
    assert out == {"minima_search_nm": (2.9, 4.2)}


def test_generate_kwargs_mixed_scalar_and_tuple():
    ids = [
        {"type": "ipa", "param": "n_jobs", "tag": "int"},
        {"type": "ipa", "param": "minima_search_nm", "tag": "Tuple", "slot": 0, "elem": "float"},
        {"type": "ipa", "param": "minima_search_nm", "tag": "Tuple", "slot": 1, "elem": "float"},
    ]
    values = [4, 3.0, 4.0]
    out = generate_kwargs(ids, values, _PS)
    assert out == {"n_jobs": 4, "minima_search_nm": (3.0, 4.0)}


# ===========================================================================
# Composite widget readers
# ===========================================================================

def test_read_label_dict_basic():
    rows = [
        {"name": "outer", "id": 1},
        {"name": "inner", "id": 2},
    ]
    assert mw.read_label_dict(rows) == {"outer": 1, "inner": 2}


def test_read_label_dict_skips_partial_rows():
    rows = [
        {"name": "", "id": 1},          # blank name -> skipped
        {"name": "outer", "id": None},  # missing id -> skipped
        {"name": "inner", "id": "3"},   # str id coerced
        {"name": "  spaced  ", "id": 4},
    ]
    assert mw.read_label_dict(rows) == {"inner": 3, "spaced": 4}


def test_read_per_membrane_mode_single():
    # Toggle off -> the override list is ignored.
    out = mw.read_per_membrane_mode(False, "closed", [], [])
    assert out == "closed"


def test_read_per_membrane_mode_per_label_dict():
    out = mw.read_per_membrane_mode(
        True, "planar",
        [{"type": "memthick-mode-per-label-mode", "label": "outer"},
         {"type": "memthick-mode-per-label-mode", "label": "inner"}],
        ["closed", "planar"],
    )
    assert out == {"outer": "closed", "inner": "planar"}


def test_read_per_membrane_mode_empty_list_falls_back_to_single():
    out = mw.read_per_membrane_mode(True, "planar", [], [])
    assert out == "planar"


def test_read_analyzer_kwargs_filters_empties():
    ids = [
        {"type": "x", "param": "smooth_sigma_intensity_profiles", "tag": "float"},
        {"type": "x", "param": "n_jobs", "tag": "int"},
        {"type": "x", "param": "minima_search_nm", "tag": "Tuple", "slot": 0, "elem": "float"},
        {"type": "x", "param": "minima_search_nm", "tag": "Tuple", "slot": 1, "elem": "float"},
    ]
    values = [0.7, None, 2.8, 4.1]
    out = mw.read_analyzer_kwargs(ids, values, _PS)
    assert out == {"smooth_sigma_intensity_profiles": 0.7, "minima_search_nm": (2.8, 4.1)}


# ===========================================================================
# read_label_dict — extended pure-function tests
# ===========================================================================

class TestReadLabelDict:
    def test_basic(self):
        rows = [
            {"name": "membrane_1", "id": "1"},
            {"name": "membrane_2", "id": "2"},
        ]
        result = read_label_dict(rows)
        assert result == {"membrane_1": 1, "membrane_2": 2}

    def test_none_input(self):
        assert read_label_dict(None) == {}

    def test_empty_list(self):
        assert read_label_dict([]) == {}

    def test_blank_name_dropped(self):
        rows = [{"name": "", "id": "1"}, {"name": "  ", "id": "2"}, {"name": "ok", "id": "3"}]
        result = read_label_dict(rows)
        assert "ok" in result
        assert len(result) == 1

    def test_missing_id_dropped(self):
        rows = [{"name": "no_id"}, {"name": "has_id", "id": "5"}]
        result = read_label_dict(rows)
        assert "has_id" in result
        assert "no_id" not in result

    def test_non_numeric_id_dropped(self):
        rows = [{"name": "bad", "id": "abc"}, {"name": "good", "id": "3"}]
        result = read_label_dict(rows)
        assert "good" in result
        assert "bad" not in result

    def test_duplicate_name_last_wins(self):
        rows = [
            {"name": "mem", "id": "1"},
            {"name": "mem", "id": "99"},
        ]
        result = read_label_dict(rows)
        assert result["mem"] == 99

    def test_none_id_dropped(self):
        rows = [{"name": "x", "id": None}, {"name": "y", "id": "2"}]
        result = read_label_dict(rows)
        assert "y" in result
        assert "x" not in result


# ===========================================================================
# Code generator
# ===========================================================================

def test_render_py_is_valid_python():
    py = codegen.render_pipeline_py({
        "segmentation_map": "/x/seg.mrc",
        "output_path": "/x/out",
        "analyzer": {"smooth_sigma_intensity_profiles": 0.7, "minima_search_nm": (2.8, 4.1)},
    })
    ast.parse(py)
    assert "from cryocat.analysis import memthick" in py
    assert "IntensityProfileAnalyzer" in py
    assert "analyzer=analyzer" in py
    # Path params hoisted to named top-level variables.
    assert "segmentation_path = '/x/seg.mrc'" in py
    assert "output_path = '/x/out'" in py
    assert "segmentation_map=segmentation_path" in py


def test_render_py_omits_analyzer_when_unspecified():
    py = codegen.render_pipeline_py({"segmentation_map": "/x/seg.mrc"})
    ast.parse(py)
    assert "IntensityProfileAnalyzer" not in py
    assert "analyzer" not in py


def test_render_py_indented_dict_for_membrane_labels():
    py = codegen.render_pipeline_py({
        "segmentation_map": "/x/seg.mrc",
        "membrane_labels": {"outer": 1, "inner": 2},
    })
    ast.parse(py)
    # Dict with > 1 entry should be rendered as a multi-line literal.
    assert "'outer': 1" in py
    assert "'inner': 2" in py


def test_render_ipynb_structure():
    nb = codegen.render_pipeline_ipynb({
        "segmentation_map": "/x/seg.mrc",
        "analyzer": {"smooth_sigma_intensity_profiles": 0.7},
    })
    assert nb["nbformat"] == 4
    cell_types = [c["cell_type"] for c in nb["cells"]]
    assert cell_types[0] == "markdown"
    # imports + paths + analyzer + run -> 4 code cells minimum.
    assert cell_types.count("code") >= 4
    # Notebook must be JSON-serialisable.
    json.dumps(nb)


def test_render_ipynb_json_round_trips():
    text = codegen.render_pipeline_ipynb_json({"segmentation_map": "/x/seg.mrc"})
    parsed = json.loads(text)
    assert parsed["nbformat"] == 4


def test_wrap_slurm_emits_directives_in_order():
    out = codegen.render_slurm_wrapper(
        "/scratch/run.py",
        cluster_params={"--mem": "32G", "-N": 1},
        module_loads=["cryocat/1.0", "cuda/12.1"],
    )
    assert out.startswith("#!/bin/bash\n")
    assert "#SBATCH --mem=32G" in out
    assert "#SBATCH -N 1" in out
    assert "module load cryocat/1.0" in out
    assert "module load cuda/12.1" in out
    assert out.rstrip().endswith("python /scratch/run.py")


def test_wrap_slurm_no_params_is_valid():
    out = codegen.render_slurm_wrapper("/scratch/run.py")
    assert out.splitlines() == ["#!/bin/bash", "python /scratch/run.py"]


# ===========================================================================
# End-to-end: form-state -> generated script
# ===========================================================================

def test_round_trip_form_state_to_python_script():
    """The whole pipeline: form ids/values -> kwargs -> rendered .py is valid."""
    pipe_ids = [
        {"type": "memthick-pipe-param", "param": "segmentation_map", "tag": "MapSource"},
        {"type": "memthick-pipe-param", "param": "radius_hit", "tag": "float"},
    ]
    pipe_vals = ["/scratch/seg.mrc", 3.5]
    an_ids = [
        {"type": "memthick-analyzer", "param": "smooth_sigma_intensity_profiles",
         "tag": "float"},
        {"type": "memthick-analyzer", "param": "minima_search_nm",
         "tag": "Tuple", "slot": 0, "elem": "float"},
        {"type": "memthick-analyzer", "param": "minima_search_nm",
         "tag": "Tuple", "slot": 1, "elem": "float"},
    ]
    an_vals = [0.8, 2.5, 4.0]

    pipeline_kwargs = generate_kwargs(pipe_ids, pipe_vals, _PS)
    pipeline_kwargs = {k: v for k, v in pipeline_kwargs.items() if v not in (None, "", [])}
    analyzer_kwargs = mw.read_analyzer_kwargs(an_ids, an_vals, _PS)

    py = codegen.render_pipeline_py({**pipeline_kwargs, "analyzer": analyzer_kwargs})
    ast.parse(py)
    # segmentation_map is hoisted to a top variable now.
    assert "segmentation_path = '/scratch/seg.mrc'" in py
    assert "segmentation_map=segmentation_path" in py
    assert "radius_hit=3.5" in py
    assert "smooth_sigma_intensity_profiles=0.8" in py
    assert "minima_search_nm=(2.5, 4.0)" in py
    assert "analyzer=analyzer" in py
