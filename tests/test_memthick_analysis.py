"""Tests for the Memthick M2 (Analyze) support code.

Pure helpers + registry only — the ``memthick_analyze_plot`` plot functions
are pre-existing tutorial code (no new tests per spec). Per-callback Dash
behaviour is exercised by the smoke-mount in the page module.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cryocat.app.components import memthick_registry as mreg
from cryocat.app.suite.pages import _memthick_analysis as analysis_helpers


# ── resolve_thickness_csv ───────────────────────────────────────────────────


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


# ── parse_membrane_names ────────────────────────────────────────────────────


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


# ── motl_to_pool_rows ───────────────────────────────────────────────────────


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


# ── memthick_registry ───────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_registry():
    mreg.clear_registry()
    yield
    mreg.clear_registry()


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
    rid = mreg.register_results(bundle)
    assert rid.startswith("memthick-")
    assert mreg.get_results(rid) is bundle


def test_register_ids_are_unique():
    a = mreg.register_results(_make_bundle("IMM"))
    b = mreg.register_results(_make_bundle("OMM"))
    assert a != b
    assert set(mreg.list_ids()) == {a, b}


def test_remove_results():
    rid = mreg.register_results(_make_bundle("IMM"))
    mreg.remove_results(rid)
    assert mreg.get_results(rid) is None
    # No-op for unknown id.
    mreg.remove_results("memthick-doesnotexist")


def test_get_results_returns_none_for_missing():
    assert mreg.get_results("memthick-doesnotexist") is None


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
