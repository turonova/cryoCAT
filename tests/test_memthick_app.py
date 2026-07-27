"""Tests for the Memthick-tab support modules.

Covers the three layers added in Phase M1 that are pure-Python and worth
locking down with snapshots:

* Tuple-widget pipeline (``resolve_param_type`` + ``_parse_tuple`` +
  ``TYPE_HANDLERS["Tuple"]`` + ``generate_kwargs`` round-trip).
* Composite-widget readers in
  :mod:`cryocat.app.components.memthick_widgets`.
* Pure code generator in
  :mod:`cryocat.app.suite.pages._memthick_codegen`.

The Dash callbacks themselves aren't exercised here — that would require a
running app; they're covered by the smoke-mount step.
"""
from __future__ import annotations

import ast
import json
import typing

import pytest

from cryocat.utils.classutils import resolve_param_type, TYPE_HANDLERS
from cryocat.app.apputils import generate_kwargs
from cryocat.app.components import memthick_widgets as mw
from cryocat.app.suite.pages import _memthick_codegen as codegen


# ── Tuple-widget pipeline ───────────────────────────────────────────────────


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
    out = generate_kwargs(ids, values)
    assert out == {"minima_search_nm": (2.9, 4.2)}


def test_generate_kwargs_mixed_scalar_and_tuple():
    ids = [
        {"type": "ipa", "param": "n_jobs", "tag": "int"},
        {"type": "ipa", "param": "minima_search_nm", "tag": "Tuple", "slot": 0, "elem": "float"},
        {"type": "ipa", "param": "minima_search_nm", "tag": "Tuple", "slot": 1, "elem": "float"},
    ]
    values = [4, 3.0, 4.0]
    out = generate_kwargs(ids, values)
    assert out == {"n_jobs": 4, "minima_search_nm": (3.0, 4.0)}


# ── Composite-widget readers ────────────────────────────────────────────────


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
    out = mw.read_analyzer_kwargs(ids, values)
    assert out == {"smooth_sigma_intensity_profiles": 0.7, "minima_search_nm": (2.8, 4.1)}


# ── Code generator ──────────────────────────────────────────────────────────


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


# ── End-to-end: form-state -> generated script ──────────────────────────────


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

    pipeline_kwargs = generate_kwargs(pipe_ids, pipe_vals)
    pipeline_kwargs = {k: v for k, v in pipeline_kwargs.items() if v not in (None, "", [])}
    analyzer_kwargs = mw.read_analyzer_kwargs(an_ids, an_vals)

    py = codegen.render_pipeline_py({**pipeline_kwargs, "analyzer": analyzer_kwargs})
    ast.parse(py)
    # segmentation_map is hoisted to a top variable now.
    assert "segmentation_path = '/scratch/seg.mrc'" in py
    assert "segmentation_map=segmentation_path" in py
    assert "radius_hit=3.5" in py
    assert "smooth_sigma_intensity_profiles=0.8" in py
    assert "minima_search_nm=(2.5, 4.0)" in py
    assert "analyzer=analyzer" in py
