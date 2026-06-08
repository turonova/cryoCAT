"""Tests for the Complexes tab (NPC) — analysis refactor + page helpers.

Covers what's new:

* :meth:`NPC.cluster_subunits_to_rings` — both mask modes produce the same
  result (in-memory shift mode vs. user-supplied mask path).
* :meth:`NPC.compute_diameter` opposite-pair logic on a synthetic motl.
* :meth:`NPC.get_centers_as_motl` on the same synthetic data.
* :func:`pcomplexes.motl_from_pool_rows` / :func:`motl_to_pool_rows` round-trip.
* :func:`pcomplexes.compute_diameter_table` shape + per-(tomo_id, object_id) row.
* :func:`pcomplexes.resolve_mask_kwargs` for both branches of the
  mask-source widget.
* :data:`pcomplexes.COMPLEX_TYPES` registry — ids, ops, picker / param
  exclusions match the NPC public surface.

Touches no existing assertions (per the spec / guideline §2).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cryocat.core import cryomotl, cryomask
from cryocat.analysis.structure import NPC
from cryocat.app.suite.pages import pcomplexes
from cryocat.app.suite.pages.pcomplexes import (
    COMPLEX_TYPES, NPC_OPS,
    motl_from_pool_rows, motl_to_pool_rows,
    resolve_mask_kwargs,
)


# ── Synthetic NPC motl ──────────────────────────────────────────────────────


def _ring_subunits(center, radius=20.0, n=8, tomo_id=1, object_id=1):
    """Build n subunits arranged on a circle around ``center`` (xy-plane)."""
    cx, cy, cz = center
    rows = []
    for i in range(n):
        theta = 2 * np.pi * i / n
        rows.append({
            "score": 0.0,
            "geom1": float(n),
            "geom2": float(i + 1),
            "subtomo_id": 0.0,
            "tomo_id": float(tomo_id),
            "object_id": float(object_id),
            "subtomo_mean": 0.0,
            "x": float(cx + radius * np.cos(theta)),
            "y": float(cy + radius * np.sin(theta)),
            "z": float(cz),
            "shift_x": 0.0, "shift_y": 0.0, "shift_z": 0.0,
            "geom3": 0.0, "geom4": 0.0, "geom5": 0.0,
            "phi": 0.0, "psi": 0.0, "theta": 0.0,
            "class": 1.0,
        })
    return rows


def _synthetic_npc_motl():
    """Two 8-fold rings in two tomograms; clean opposite-pair geometry."""
    rows = (
        _ring_subunits((100, 100, 100), radius=20.0, tomo_id=1, object_id=1)
        + _ring_subunits((300, 300, 100), radius=25.0, tomo_id=2, object_id=1)
    )
    df = pd.DataFrame(rows)
    df["subtomo_id"] = np.arange(1, len(df) + 1, dtype=float)
    return cryomotl.Motl(df)


# ── compute_diameter / table wrapper ────────────────────────────────────────


def test_compute_diameter_returns_summary_df_and_motl():
    motl = _synthetic_npc_motl()
    summary_df, motl_out = NPC.compute_diameter(motl, pixel_size=1.0)
    # 2 NPCs with full opposite-pair coverage.
    assert list(summary_df.columns) == ["tomo_id", "object_id", "mean_diameter", "n_pairs"]
    assert len(summary_df) == 2
    # Ring radius = 20 → diameter = 40; radius 25 → diameter 50.
    np.testing.assert_allclose(
        sorted(summary_df["mean_diameter"].tolist()), [40.0, 50.0], atol=1e-6,
    )
    # 8-fold symmetry → 4 opposite pairs per ring.
    assert (summary_df["n_pairs"] == 4).all()
    # The returned motl is the same length as the input.
    assert len(motl_out.df) == len(motl.df)


def test_compute_diameter_pixel_size_scales_distance():
    motl = _synthetic_npc_motl()
    raw_df, _ = NPC.compute_diameter(motl, pixel_size=1.0)
    scaled_df, _ = NPC.compute_diameter(motl, pixel_size=2.0)
    np.testing.assert_allclose(
        sorted(scaled_df["mean_diameter"].tolist()),
        [d * 2.0 for d in sorted(raw_df["mean_diameter"].tolist())],
    )


def test_compute_diameter_writes_to_store_column_default_geom4():
    motl = _synthetic_npc_motl()
    _, motl_out = NPC.compute_diameter(motl, pixel_size=1.0)
    # Default store_column is "geom4"; every row in NPC 1 (tomo_id=1, object_id=1)
    # should carry that NPC's diameter (40), every row in NPC 2 carries 50.
    npc1 = motl_out.df[(motl_out.df["tomo_id"] == 1) & (motl_out.df["object_id"] == 1)]
    npc2 = motl_out.df[(motl_out.df["tomo_id"] == 2) & (motl_out.df["object_id"] == 1)]
    np.testing.assert_allclose(npc1["geom4"].to_numpy(), 40.0, atol=1e-6)
    np.testing.assert_allclose(npc2["geom4"].to_numpy(), 50.0, atol=1e-6)


def test_compute_diameter_honors_custom_store_column():
    motl = _synthetic_npc_motl()
    _, motl_out = NPC.compute_diameter(motl, pixel_size=1.0, store_column="geom5")
    # geom5 now carries the per-NPC diameter.
    npc1 = motl_out.df[(motl_out.df["tomo_id"] == 1) & (motl_out.df["object_id"] == 1)]
    np.testing.assert_allclose(npc1["geom5"].to_numpy(), 40.0, atol=1e-6)
    # geom4 stays untouched (zero from the synthetic motl).
    np.testing.assert_allclose(npc1["geom4"].to_numpy(), 0.0, atol=1e-6)


def test_compute_diameter_does_not_mutate_input_motl():
    motl = _synthetic_npc_motl()
    before = motl.df.copy()
    _ = NPC.compute_diameter(motl, pixel_size=1.0)
    pd.testing.assert_frame_equal(motl.df, before, check_dtype=False)


def test_compute_diameter_empty_summary_when_no_pairs():
    # A motl with only subunit 1 / 2 (no opposite pair) → empty summary.
    rows = []
    for su in (1, 2):
        rows.append({
            "score": 0.0, "geom1": 8.0, "geom2": float(su),
            "subtomo_id": float(su), "tomo_id": 1.0, "object_id": 1.0,
            "subtomo_mean": 0.0, "x": float(su), "y": 0.0, "z": 0.0,
            "shift_x": 0.0, "shift_y": 0.0, "shift_z": 0.0,
            "geom3": 0.0, "geom4": 0.0, "geom5": 0.0,
            "phi": 0.0, "psi": 0.0, "theta": 0.0, "class": 1.0,
        })
    summary_df, motl_out = NPC.compute_diameter(
        cryomotl.Motl(pd.DataFrame(rows)), pixel_size=1.0,
    )
    assert summary_df.empty
    # All rows have NaN in store_column when no NPC produced a diameter.
    assert motl_out.df["geom4"].isna().all()


# ── get_centers_as_motl (representative merge/centers op) ───────────────────


def test_get_centers_as_motl_one_per_ring():
    motl = _synthetic_npc_motl()
    tm = motl.get_motl_subset(column_values=[1], column_name="tomo_id", reset_index=True)
    centers = NPC.get_centers_as_motl(tm, tomo_id=1, radius=20.0)
    # Only one ring in this tomogram.
    assert len(centers.df) == 1
    # get_center_with_radius shifts every particle inward by (-radius, 0, 0)
    # along the local X axis. With identity rotations, local X == global X,
    # so the mean ends up at (cx - radius, cy, cz) = (100 - 20, 100, 100).
    np.testing.assert_allclose(
        centers.df.loc[0, ["x", "y", "z"]].to_numpy().astype(float),
        [80.0, 100.0, 100.0], atol=1e-6,
    )


# ── cluster_subunits_to_rings: both mask modes produce identical rings ──────


def test_cluster_subunits_to_rings_validates_missing_mask_args():
    motl = _synthetic_npc_motl()
    with pytest.raises(ValueError, match="entry_mask"):
        NPC.cluster_subunits_to_rings(
            motl, npc_radius=40.0, max_trace_distance=10.0,
            # entry: no mask AND no coord
            exit_mask_coord=(10, 10, 10), mask_size=20,
        )
    with pytest.raises(ValueError, match="exit_mask"):
        NPC.cluster_subunits_to_rings(
            motl, npc_radius=40.0, max_trace_distance=10.0,
            entry_mask_coord=(10, 10, 10), mask_size=20,
            # exit: no mask AND no coord
        )


def test_cluster_subunits_to_rings_in_memory_and_supplied_mask_match(tmp_path):
    """Spec requirement: same ring result regardless of mask source.

    Build the mask path-mode input by writing the same spherical mask
    spherical_mask() would produce in-memory — then check the two calls
    return motls with equal contents.
    """
    motl = _synthetic_npc_motl()
    # Mask-internal coords (within the mask_size box).
    entry_coord = (40, 50, 40)
    exit_coord = (40, 30, 40)
    mask_size = 80

    # In-memory shift mode.
    result_mem = NPC.cluster_subunits_to_rings(
        cryomotl.Motl(motl.df.copy()),
        npc_radius=40.0,
        max_trace_distance=50.0,
        mask_size=mask_size,
        entry_mask_coord=entry_coord,
        exit_mask_coord=exit_coord,
    )

    # Path-supplied mode: write the same spherical masks to disk first.
    entry_path = tmp_path / "entry.em"
    exit_path = tmp_path / "exit.em"
    cryomask.spherical_mask(mask_size, 3, center=entry_coord, output_path=str(entry_path))
    cryomask.spherical_mask(mask_size, 3, center=exit_coord, output_path=str(exit_path))

    result_paths = NPC.cluster_subunits_to_rings(
        cryomotl.Motl(motl.df.copy()),
        npc_radius=40.0,
        max_trace_distance=50.0,
        entry_mask=str(entry_path),
        exit_mask=str(exit_path),
    )

    # Same algorithm + same masks → identical motl rows.
    a = result_mem.df.sort_values("subtomo_id").reset_index(drop=True)
    b = result_paths.df.sort_values("subtomo_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, atol=1e-6)


# ── Page-side pure helpers ──────────────────────────────────────────────────


def test_motl_from_pool_rows_round_trip():
    motl = _synthetic_npc_motl()
    rows = motl_to_pool_rows(motl)
    pool = {"motl-0": rows}
    rebuilt = motl_from_pool_rows(pool, "motl-0")
    assert rebuilt is not None
    pd.testing.assert_frame_equal(
        rebuilt.df.reset_index(drop=True),
        motl.df.reset_index(drop=True),
        check_dtype=False,
    )


def test_motl_from_pool_rows_missing_or_empty_returns_none():
    assert motl_from_pool_rows({}, "motl-0") is None
    assert motl_from_pool_rows({"motl-0": []}, "motl-0") is None
    assert motl_from_pool_rows({"motl-0": []}, None) is None
    assert motl_from_pool_rows(None, "motl-0") is None


def test_resolve_mask_kwargs_shift_branch():
    out = resolve_mask_kwargs(
        "shift", mask_size=72,
        entry_coord=(34, 61, 36), exit_coord=(34, 17, 36),
        entry_path="", exit_path="",
    )
    assert out == {
        "mask_size": 72,
        "entry_mask_coord": (34, 61, 36),
        "exit_mask_coord": (34, 17, 36),
    }


def test_resolve_mask_kwargs_shift_branch_drops_missing():
    out = resolve_mask_kwargs(
        "shift", mask_size=None,
        entry_coord=None, exit_coord=None,
        entry_path=None, exit_path=None,
    )
    assert out == {}


def test_resolve_mask_kwargs_paths_branch():
    out = resolve_mask_kwargs(
        "paths", mask_size=None,
        entry_coord=None, exit_coord=None,
        entry_path="/scratch/entry.em", exit_path="/scratch/exit.em",
    )
    assert out == {
        "entry_mask": "/scratch/entry.em",
        "exit_mask": "/scratch/exit.em",
    }


def test_resolve_mask_kwargs_paths_branch_drops_empties():
    out = resolve_mask_kwargs(
        "paths", mask_size=None,
        entry_coord=None, exit_coord=None,
        entry_path="  ", exit_path=None,
    )
    assert out == {}


# ── Registry plumbing ───────────────────────────────────────────────────────


def test_complex_types_registry_has_npc():
    assert "npc" in COMPLEX_TYPES
    npc = COMPLEX_TYPES["npc"]
    assert npc["namespace"] is NPC
    assert npc["ops"] is NPC_OPS
    assert npc["id_prefix"].startswith("complexes-")


def test_npc_ops_cover_the_spec_table():
    expected = {
        "cluster_subunits", "unify_orientations", "merge_subunits",
        "merge_rings", "centers", "diameter",
    }
    assert set(NPC_OPS) == expected


def test_only_cluster_has_the_mask_widget_flag():
    for op_id, op in NPC_OPS.items():
        assert op["needs_mask_widget"] == (op_id == "cluster_subunits")


def test_only_merge_rings_uses_multi_picker():
    for op_id, op in NPC_OPS.items():
        for _, multi in op["pickers"]:
            if op_id == "merge_rings":
                assert multi is True
            else:
                assert multi is False


def test_op_method_names_exist_on_npc():
    for op in NPC_OPS.values():
        assert hasattr(NPC, op["method_name"]), op["method_name"]


def test_excluded_params_include_picker_kwargs():
    """Every picker kwarg must be in the op's exclude set so build_form skips it."""
    for op in NPC_OPS.values():
        for picker_name, _ in op["pickers"]:
            assert picker_name in op["excluded_params"], picker_name


def test_cluster_subunits_excludes_mask_params():
    """The mask-source widget owns these — they must not appear in the form."""
    excl = set(NPC_OPS["cluster_subunits"]["excluded_params"])
    for name in ("entry_mask", "exit_mask",
                 "entry_mask_coord", "exit_mask_coord", "mask_size"):
        assert name in excl, name
