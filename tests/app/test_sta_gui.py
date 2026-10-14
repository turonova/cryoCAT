"""GUI-logic tests for cryocat/app/suite/pages/psta.py.

Tests pure helper functions that are used by callbacks — no running Dash app required.
"""
from __future__ import annotations

from pathlib import Path

from cryocat.app.suite.pages.psta import (
    _build_sta_run_from_form,
    _validate_sep_ranges,
    _expand_sep_ranges,
)


# ── _build_sta_run_from_form ─────────────────────────────────────────────────


def _form_kwargs(tmp_path: Path) -> dict:
    sub = tmp_path / "subtomos"
    sub.mkdir()
    return {
        "output_base": str(tmp_path / "out"),
        "folder_name": "run01",
        "n_runs": 1,
        "subtomo_path": str(sub),
        "run_mode": "singleref",
        "motl_paths_text": None,
        "cls_paths": None,
        "iter_n": 10,
        "mask": None,
        "ccmask": None,
        "wedgelist": None,
        "ref_files_text": None,
    }


def test_build_sta_run_success(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    sta_run, motl_paths, errors = _build_sta_run_from_form(**kwargs)
    assert errors == []
    assert sta_run is not None
    assert str(sta_run.folder_name) == "run01"
    assert sta_run.run_mode == "singleref"
    assert len(sta_run.schedule) == 1


def test_build_sta_run_missing_output_base(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["output_base"] = ""
    _, _, errors = _build_sta_run_from_form(**kwargs)
    assert any("output base" in e.lower() for e in errors)


def test_build_sta_run_missing_folder_name(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["folder_name"] = None
    _, _, errors = _build_sta_run_from_form(**kwargs)
    assert any("folder name" in e.lower() for e in errors)


def test_build_sta_run_missing_subtomo_path(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["subtomo_path"] = None
    _, _, errors = _build_sta_run_from_form(**kwargs)
    assert any("subtomogram" in e.lower() or "subtomo" in e.lower() for e in errors)


def test_build_sta_run_motl_paths_from_cls(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["cls_paths"] = ["/tmp/motl_mr1.star", "/tmp/motl_mr2.star"]
    sta_run, motl_paths, errors = _build_sta_run_from_form(**kwargs)
    assert errors == []
    assert len(motl_paths) == 2


def test_build_sta_run_motl_paths_from_text(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["motl_paths_text"] = "/tmp/a.star, /tmp/b.star"
    sta_run, motl_paths, errors = _build_sta_run_from_form(**kwargs)
    assert errors == []
    assert len(motl_paths) == 2


def test_build_sta_run_cls_paths_take_priority_over_text(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["cls_paths"] = ["/cls/motl.star"]
    kwargs["motl_paths_text"] = "/text/motl.star"
    _, motl_paths, _ = _build_sta_run_from_form(**kwargs)
    assert len(motl_paths) == 1
    assert "cls" in str(motl_paths[0])


def test_build_sta_run_base_params_from_paths(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["mask"] = "/masks/mask.em"
    kwargs["ccmask"] = "/masks/ccmask.em"
    kwargs["wedgelist"] = "/lists/wedge.star"
    sta_run, _, _ = _build_sta_run_from_form(**kwargs)
    bp = sta_run.base_params
    assert bp.get("mask name") == "/masks/mask.em"
    assert bp.get("cc mask name") == "/masks/ccmask.em"
    assert bp.get("wedge list") == "/lists/wedge.star"


def test_build_sta_run_n_runs(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["n_runs"] = 4
    sta_run, _, _ = _build_sta_run_from_form(**kwargs)
    assert sta_run.n_runs == 4


def test_build_sta_run_references_from_text(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["ref_files_text"] = "/ref/ref_1.em, /ref/ref_2.em"
    sta_run, _, _ = _build_sta_run_from_form(**kwargs)
    assert len(sta_run.references) == 2


def test_build_sta_run_multiple_errors_reported(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["output_base"] = ""
    kwargs["folder_name"] = ""
    kwargs["subtomo_path"] = ""
    _, _, errors = _build_sta_run_from_form(**kwargs)
    assert len(errors) >= 3


# ── _validate_sep_ranges ─────────────────────────────────────────────────────


def test_validate_sep_ranges_valid_contiguous():
    rows = [
        {"from_iter": 1, "to_iter": 5},
        {"from_iter": 6, "to_iter": 10},
    ]
    errors = _validate_sep_ranges(rows, 10)
    assert errors == []


def test_validate_sep_ranges_gap():
    rows = [
        {"from_iter": 1, "to_iter": 4},
        {"from_iter": 6, "to_iter": 10},
    ]
    errors = _validate_sep_ranges(rows, 10)
    assert any("gap" in e.lower() for e in errors)


def test_validate_sep_ranges_overlap():
    rows = [
        {"from_iter": 1, "to_iter": 6},
        {"from_iter": 5, "to_iter": 10},
    ]
    errors = _validate_sep_ranges(rows, 10)
    assert any("overlap" in e.lower() for e in errors)


def test_validate_sep_ranges_bad_start():
    rows = [
        {"from_iter": 2, "to_iter": 10},
    ]
    errors = _validate_sep_ranges(rows, 10)
    assert any("start at 1" in e for e in errors)


def test_validate_sep_ranges_bad_end():
    rows = [
        {"from_iter": 1, "to_iter": 8},
    ]
    errors = _validate_sep_ranges(rows, 10)
    assert any("end at 10" in e for e in errors)


# ── _expand_sep_ranges ───────────────────────────────────────────────────────


def test_expand_sep_ranges_constant():
    rows = [{"from_iter": 1, "to_iter": 5, "cone_angle": "30", "cone_sampling": "5",
             "inplane_angle": "360", "inplane_sampling": "5"}]
    fields = ["cone_angle", "cone_sampling", "inplane_angle", "inplane_sampling"]
    result = _expand_sep_ranges(rows, fields, iter_n=5, start_idx=1)
    assert result["cone_angle"] == "30 30 30 30 30"
    assert result["cone_sampling"] == "5 5 5 5 5"


def test_expand_sep_ranges_two_ranges():
    rows = [
        {"from_iter": 1, "to_iter": 3, "cone_angle": "30", "cone_sampling": "5",
         "inplane_angle": "360", "inplane_sampling": "5"},
        {"from_iter": 4, "to_iter": 5, "cone_angle": "10", "cone_sampling": "2",
         "inplane_angle": "360", "inplane_sampling": "2"},
    ]
    fields = ["cone_angle", "cone_sampling", "inplane_angle", "inplane_sampling"]
    result = _expand_sep_ranges(rows, fields, iter_n=5, start_idx=1)
    assert result["cone_angle"] == "30 30 30 10 10"
    assert result["cone_sampling"] == "5 5 5 2 2"
