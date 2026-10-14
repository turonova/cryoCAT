"""Tests for memthick.py — membrane thickness analysis pipeline.

Covers:
- String / name utilities (pure, no I/O)
- Surface processing helpers
- CPU point matching and one-to-one assignment
- Matching statistics generation and saving
- IntensityProfileAnalyzer detect pipeline
- Integration: process_membrane_segmentation on a synthetic flat slab
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cryocat.core import cryomap
from cryocat.analysis.memthick import (
    _matched_table_base_name,
    _matched_point_geometry_counts,
    _infer_membrane_suffix_from_csv,
    _sanitize_log_fragment,
    pipeline_analysis_log_filename,
    create_vertex_volume,
    is_surface_point,
    _filter_to_segmentation_boundary,
    process_matches_cpu2cpu,
    measure_thickness_cpu,
    generate_matching_statistics,
    save_matching_statistics,
    IntensityProfileAnalyzer,
    process_membrane_segmentation,
)



# =============================================================================
# A. String / name utilities
# =============================================================================

@pytest.mark.parametrize("stem,expected", [
    ("foo_matched_points_2to1",  "foo"),
    ("foo_matched_points",       "foo"),
    ("foo_thickness_2to1",       "foo"),
    ("foo_thickness",            "foo"),
    ("foo_membrane_thickness",   "foo_membrane"),
    ("foo_bar",                  "foo_bar"),   # no recognised suffix → unchanged
    ("just_stem",                "just_stem"),
])
def test_matched_table_base_name(stem, expected):
    fake_path = Path(f"/some/dir/{stem}.csv")
    assert _matched_table_base_name(fake_path) == expected


def test_sanitize_log_fragment_spaces_and_slashes():
    assert " " not in _sanitize_log_fragment("hello world")
    assert "/" not in _sanitize_log_fragment("a/b/c")
    assert "\\" not in _sanitize_log_fragment("a\\b")


def test_sanitize_log_fragment_empty_returns_memthick():
    assert _sanitize_log_fragment("") == "memthick"
    assert _sanitize_log_fragment("   ") == "memthick"


def test_sanitize_log_fragment_no_double_underscores():
    result = _sanitize_log_fragment("a  b  c")
    assert "__" not in result


def test_pipeline_analysis_log_filename_all_parts(tmp_path):
    seg = tmp_path / "seg_file.mrc"
    tomo = tmp_path / "tomo_file.mrc"
    seg.touch(); tomo.touch()
    name = pipeline_analysis_log_filename(str(seg), str(tomo), membrane_labels=["OMM", "IMM"])
    assert name.endswith("_analysis.log")
    assert "seg_file" in name
    assert "tomo_file" in name
    # labels are sorted and joined
    assert "IMM" in name and "OMM" in name


def test_pipeline_analysis_log_filename_no_parts():
    name = pipeline_analysis_log_filename()
    assert name == "memthick_analysis.log"


def test_infer_membrane_suffix_from_csv(tmp_path):
    seg = tmp_path / "2738_seg.mrc"
    seg.touch()
    csv = tmp_path / "2738_seg_OMM_matched_points.csv"
    result = _infer_membrane_suffix_from_csv(csv, seg)
    assert result == "OMM"


def test_infer_membrane_suffix_no_match(tmp_path):
    seg = tmp_path / "seg.mrc"
    seg.touch()
    csv = tmp_path / "completely_different_matched_points.csv"
    result = _infer_membrane_suffix_from_csv(csv, seg)
    assert result is None


def test_matched_point_geometry_counts_valid():
    df = pd.DataFrame({
        "x1_voxel": [1.0, 2.0],
        "y1_voxel": [1.0, 2.0],
        "z1_voxel": [1.0, 2.0],
        "x2_voxel": [4.0, 5.0],
        "y2_voxel": [4.0, 5.0],
        "z2_voxel": [4.0, 5.0],
    })
    n_rows, n_valid = _matched_point_geometry_counts(df)
    assert n_rows == 2
    assert n_valid == 2


def test_matched_point_geometry_counts_with_nan():
    df = pd.DataFrame({
        "x1_voxel": [1.0, np.nan],
        "y1_voxel": [1.0, 2.0],
        "z1_voxel": [1.0, 2.0],
        "x2_voxel": [4.0, 5.0],
        "y2_voxel": [4.0, 5.0],
        "z2_voxel": [4.0, 5.0],
    })
    n_rows, n_valid = _matched_point_geometry_counts(df)
    assert n_rows == 2
    assert n_valid == 1


def test_matched_point_geometry_counts_missing_cols():
    df = pd.DataFrame({"a": [1, 2]})
    assert _matched_point_geometry_counts(df) is None


# =============================================================================
# B. Surface processing helpers
# =============================================================================

def test_create_vertex_volume_marks_positions():
    vertices = np.array([[2, 3, 4], [5, 5, 5], [1, 1, 1]], dtype=float)
    vol = create_vertex_volume(vertices, (10, 10, 10))
    assert vol[2, 3, 4] == 1
    assert vol[5, 5, 5] == 1
    assert vol[1, 1, 1] == 1
    assert vol.sum() == 3


def test_create_vertex_volume_out_of_bounds_ignored():
    vertices = np.array([[0, 0, 0], [20, 20, 20]], dtype=float)  # second is OOB
    vol = create_vertex_volume(vertices, (10, 10, 10))
    assert vol.sum() == 1


def test_is_surface_point_on_boundary():
    # Inner 3×3×3 block is True; surrounding shell is False.
    # Voxel [1,2,2] is inside the block and its x-1=0 neighbour is outside → surface point.
    seg = np.zeros((5, 5, 5), dtype=bool)
    seg[1:4, 1:4, 1:4] = True
    assert is_surface_point([1, 2, 2], seg)


def test_is_surface_point_interior_not_surface():
    # All-ones 5×5×5: centre [2,2,2] has all 6 neighbours inside → not a surface point
    seg = np.ones((5, 5, 5), dtype=bool)
    assert not is_surface_point([2, 2, 2], seg)


def test_is_surface_point_not_in_segmentation():
    seg = np.zeros((5, 5, 5), dtype=bool)
    assert not is_surface_point([2, 2, 2], seg)


def test_filter_to_segmentation_boundary_keeps_boundary_only():
    seg = np.zeros((10, 10, 10), dtype=bool)
    seg[2:8, 2:8, 2:8] = True

    # [4,4,4] is interior (all 6 face neighbours are inside)
    # [2,4,4] is on the boundary (has at least one outside face neighbour)
    vertices = np.array([[4, 4, 4], [2, 4, 4]], dtype=float)
    normals = np.ones((2, 3), dtype=float)

    bv, bn = _filter_to_segmentation_boundary(vertices, normals, seg)
    assert len(bv) == 1
    assert list(bv[0]) == [2, 4, 4]


def test_filter_to_segmentation_boundary_empty_input():
    seg = np.zeros((5, 5, 5), dtype=bool)
    vertices = np.zeros((0, 3), dtype=float)
    normals = np.zeros((0, 3), dtype=float)
    bv, bn = _filter_to_segmentation_boundary(vertices, normals, seg)
    assert len(bv) == 0


# =============================================================================
# C. CPU point matching
# =============================================================================

@pytest.fixture
def flat_bilayer():
    """Two 5×5 grids separated by 5 voxels along z, normals pointing inward."""
    xs, ys = np.meshgrid(np.arange(5), np.arange(5))
    xs, ys = xs.ravel().astype(float), ys.ravel().astype(float)
    n = len(xs)
    zeros = np.zeros(n)
    ones = np.ones(n)

    p1 = np.column_stack([xs, ys, zeros])           # surface 1 at z = 0
    p2 = np.column_stack([xs, ys, ones * 5.0])      # surface 2 at z = 5

    points = np.vstack([p1, p2]).astype(np.float32)
    normals = np.vstack([
        np.column_stack([zeros, zeros,  ones]),      # S1 normals: +z
        np.column_stack([zeros, zeros, -ones]),      # S2 normals: -z
    ]).astype(np.float32)

    s1 = np.array([True] * n + [False] * n)
    s2 = np.array([False] * n + [True] * n)
    return points, normals, s1, s2, n


def test_measure_thickness_cpu_recovers_separation(flat_bilayer):
    points, normals, s1, s2, n = flat_bilayer
    pixel_size = 0.5  # nm/vox → expected distance = 5 × 0.5 = 2.5 nm

    results, valid, pairs = measure_thickness_cpu(
        points, normals, s1, s2, pixel_size=pixel_size,
        max_distance_nm=4.0, max_angle_degrees=5.0, direction="1to2",
    )
    # Every surface-1 point should be matched
    assert valid[:n].all(), "Not all surface-1 points were matched"
    np.testing.assert_allclose(results[valid], 2.5, atol=0.01)


def test_measure_thickness_cpu_direction_2to1(flat_bilayer):
    points, normals, s1, s2, n = flat_bilayer
    pixel_size = 0.5

    results, valid, pairs = measure_thickness_cpu(
        points, normals, s1, s2, pixel_size=pixel_size,
        max_distance_nm=4.0, max_angle_degrees=5.0, direction="2to1",
    )
    # With direction="2to1", surface-2 points (indices n:2n) are the source
    assert valid[n:].all(), "Not all surface-2 points were matched in 2to1 direction"
    np.testing.assert_allclose(results[valid], 2.5, atol=0.01)


def test_process_matches_cpu2cpu_one_to_one():
    # Points 0 and 1 (surface-1) both want surface-2 target 100; point 0 is closer.
    flat_matches = [(1.0, 0, 100), (2.0, 1, 100), (3.0, 2, 200)]
    results, valid, pairs = process_matches_cpu2cpu(flat_matches, n_points=300, pixel_size=1.0)

    assert valid[0], "Closest match (point 0) should be valid"
    assert not valid[1], "Farther competitor (point 1) should lose the target"
    assert valid[2], "Uncontested match (point 2) should be valid"


def test_process_matches_cpu2cpu_converts_to_physical_units():
    flat_matches = [(2.0, 0, 1)]   # 2 vox distance
    results, valid, _ = process_matches_cpu2cpu(flat_matches, n_points=5, pixel_size=0.5)
    assert valid[0]
    np.testing.assert_allclose(results[0], 1.0, atol=1e-5)   # 2 × 0.5 = 1.0 nm


def test_process_matches_cpu2cpu_empty():
    results, valid, pairs = process_matches_cpu2cpu([], n_points=10, pixel_size=1.0)
    assert not valid.any()
    assert results.sum() == 0.0


# =============================================================================
# D. Matching statistics
# =============================================================================

@pytest.fixture
def simple_stats_inputs():
    thickness = np.array([2.0, 2.5, 3.0, 2.2, 2.8])
    valid = np.ones(5, dtype=bool)
    points = np.random.default_rng(42).random((5, 3))
    s1 = np.array([True, True, True, False, False])
    s2 = np.array([False, False, False, True, True])
    return thickness, valid, points, s1, s2


def test_generate_matching_statistics_required_keys(simple_stats_inputs):
    thickness, valid, points, s1, s2 = simple_stats_inputs
    stats = generate_matching_statistics(thickness, valid, points, s1, s2, pixel_size=0.5)
    for key in (
        "total_points", "surface1_points", "surface2_points",
        "valid_measurements", "coverage_percentage",
        "mean_thickness", "std_thickness", "median_thickness",
        "min_thickness", "max_thickness",
        "thickness_histogram", "spatial_distribution",
    ):
        assert key in stats, f"Missing key: {key}"


def test_generate_matching_statistics_values(simple_stats_inputs):
    thickness, valid, points, s1, s2 = simple_stats_inputs
    stats = generate_matching_statistics(thickness, valid, points, s1, s2, pixel_size=0.5)
    assert stats["total_points"] == 5
    assert stats["surface1_points"] == 3
    assert stats["surface2_points"] == 2
    assert stats["valid_measurements"] == 5
    np.testing.assert_allclose(stats["mean_thickness"], np.mean(thickness), atol=1e-5)


def test_save_matching_statistics_creates_file(tmp_path, simple_stats_inputs):
    thickness, valid, points, s1, s2 = simple_stats_inputs
    stats = generate_matching_statistics(thickness, valid, points, s1, s2, pixel_size=0.5)
    out = tmp_path / "stats.txt"
    save_matching_statistics(stats, out)
    assert out.exists()
    content = out.read_text()
    assert "Mean distance" in content
    assert "Coverage" in content


def test_save_matching_statistics_with_params(tmp_path, simple_stats_inputs):
    thickness, valid, points, s1, s2 = simple_stats_inputs
    stats = generate_matching_statistics(thickness, valid, points, s1, s2, pixel_size=0.5)
    params = {"max_distance_nm": 8.0, "direction": "1to2"}
    out = tmp_path / "stats_params.txt"
    save_matching_statistics(stats, out, matching_params=params)
    content = out.read_text()
    assert "Parameters" in content
    assert "max_distance_nm" in content


# =============================================================================
# E. Intensity profile analysis
# =============================================================================

@pytest.fixture
def bilayer_profile_dict():
    """Idealised M-shaped bilayer intensity profile, pixel_size=0.5 nm.

    Shape: two minima at ±4 vox from centre, central maximum at 0,
    outward maxima at ±10 vox.
    """
    pixel_size = 0.5
    n = 80
    xs = np.linspace(-15.0, 15.0, n)   # voxel positions relative to midpoint

    intensity = (
        -np.exp(-0.5 * ((xs - 4) ** 2 / 1.5 ** 2))   # right minimum
        - np.exp(-0.5 * ((xs + 4) ** 2 / 1.5 ** 2))  # left minimum
        + 0.7 * np.exp(-0.5 * (xs ** 2 / 2.0 ** 2))  # central max
        + 0.5 * np.exp(-0.5 * ((xs - 10) ** 2 / 2.5 ** 2))  # right outward max
        + 0.5 * np.exp(-0.5 * ((xs + 10) ** 2 / 2.5 ** 2))  # left outward max
    )

    midpoint = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    start = midpoint + xs[0] * direction
    end = midpoint + xs[-1] * direction
    p1 = midpoint + (-4.0) * direction   # matched point 1 projection position
    p2 = midpoint + (4.0) * direction    # matched point 2 projection position

    return {
        "p1": p1,
        "p2": p2,
        "midpoint": midpoint,
        "start": start,
        "end": end,
        "profile": intensity.astype(np.float32),
        "pixel_size": pixel_size,
    }


def test_detect_single_profile_resolves_clean_bilayer(bilayer_profile_dict):
    analyzer = IntensityProfileAnalyzer(smooth_sigma_intensity_profiles=0.5)
    result = analyzer._detect_single_profile(bilayer_profile_dict)
    assert result["resolved"], f"Profile not resolved: {result.get('failure_reason')}"
    # Inflection-based thickness should be in a plausible biological range
    assert np.isfinite(result["membrane_thickness_nm"])
    assert 1.0 < result["membrane_thickness_nm"] < 10.0


def test_detect_single_profile_bad_pixel_size(bilayer_profile_dict):
    prof = {**bilayer_profile_dict, "pixel_size": 0.0}
    result = IntensityProfileAnalyzer()._detect_single_profile(prof)
    assert not result["resolved"]
    assert result["failure_reason"] == "invalid_pixel_size"


def test_detect_single_profile_negative_pixel_size(bilayer_profile_dict):
    prof = {**bilayer_profile_dict, "pixel_size": -1.0}
    result = IntensityProfileAnalyzer()._detect_single_profile(prof)
    assert not result["resolved"]


def test_detect_single_profile_invalid_axis(bilayer_profile_dict):
    prof = {**bilayer_profile_dict, "p1": bilayer_profile_dict["p2"]}  # zero-length direction
    result = IntensityProfileAnalyzer()._detect_single_profile(prof)
    assert not result["resolved"]


def test_analyzer_detect_returns_required_keys(bilayer_profile_dict):
    profiles = [bilayer_profile_dict] * 3
    df = pd.DataFrame({
        "match_distance_nm": [4.0] * 3,
        "x1_voxel": [0.0] * 3, "y1_voxel": [0.0] * 3, "z1_voxel": [-4.0] * 3,
        "x2_voxel": [0.0] * 3, "y2_voxel": [0.0] * 3, "z2_voxel": [ 4.0] * 3,
    })
    results = IntensityProfileAnalyzer().detect(
        profiles, df, profile_half_width_nm=8.0, max_distance_nm=12.0
    )
    for key in ("boundary_results", "resolved_thickness_df", "membrane_thickness_df",
                "statistics", "parameters"):
        assert key in results, f"Missing key: {key}"


def test_analyzer_detect_statistics_fields(bilayer_profile_dict):
    profiles = [bilayer_profile_dict] * 4
    df = pd.DataFrame({
        "match_distance_nm": [4.0] * 4,
        "x1_voxel": [0.0] * 4, "y1_voxel": [0.0] * 4, "z1_voxel": [-4.0] * 4,
        "x2_voxel": [0.0] * 4, "y2_voxel": [0.0] * 4, "z2_voxel": [ 4.0] * 4,
    })
    results = IntensityProfileAnalyzer().detect(profiles, df, max_distance_nm=12.0)
    stats = results["statistics"]
    assert stats["total_profiles"] == 4
    assert "profiles_resolved" in stats
    assert "resolution_rate" in stats


# =============================================================================
# F. Integration: process_membrane_segmentation
# =============================================================================

def test_process_membrane_segmentation_flat_slab(tmp_path):
    """Flat 5-voxel-thick slab in a 40×40×40 volume → two bilayer surfaces, CSV written."""
    seg = np.zeros((40, 40, 40), dtype=np.float32)
    seg[15:20, 5:35, 5:35] = 1.0   # slab occupying z-slices 15..19

    mrc_path = str(tmp_path / "seg.mrc")
    cryomap.write(seg, mrc_path, transpose=False, pixel_size=5.0)  # 5 Å = 0.5 nm

    result = process_membrane_segmentation(
        mrc_path,
        output_path=str(tmp_path),
        membrane_labels={"mem": 1},
        refine_normals=False,    # skip refinement to keep the test fast
    )

    assert result is not None, "process_membrane_segmentation returned None"
    assert "mem" in result, "Expected 'mem' key in result dict"

    csv_path = result["mem"]
    assert os.path.exists(csv_path), f"Output CSV not found: {csv_path}"

    df = pd.read_csv(csv_path)
    required_cols = {"x_voxel", "y_voxel", "z_voxel", "surface1", "surface2"}
    assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"

    # Both surfaces should have vertices
    assert df["surface1"].astype(bool).any(), "No vertices assigned to surface 1"
    assert df["surface2"].astype(bool).any(), "No vertices assigned to surface 2"


def test_process_membrane_segmentation_missing_label(tmp_path):
    """Label not present in segmentation → membrane skipped, result may be empty dict."""
    seg = np.zeros((20, 20, 20), dtype=np.float32)
    mrc_path = str(tmp_path / "empty_seg.mrc")
    cryomap.write(seg, mrc_path, transpose=False, pixel_size=5.0)

    result = process_membrane_segmentation(
        mrc_path,
        output_path=str(tmp_path),
        membrane_labels={"missing": 99},
        refine_normals=False,
    )
    # Should return an empty dict (label not found), not None
    assert result is not None
    assert "missing" not in result
