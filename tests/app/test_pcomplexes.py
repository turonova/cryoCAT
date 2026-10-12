"""Tests for the Complexes tab — analysis helpers + registry-driven page.

Covers:

* :meth:`NPC.cluster_subunits_to_rings` — both mask modes produce the same
  result (in-memory shift mode vs. user-supplied mask path).
* :meth:`NPC.compute_diameter` opposite-pair logic on a synthetic motl.
* :meth:`NPC.get_centers_as_motl` on the same synthetic data.
* :func:`pcomplexes.motl_from_pool_rows` / :func:`motl_to_pool_rows` round-trip.
* :data:`pcomplexes.COMPLEX_CLASSES` — expected concrete classes present.
* :func:`~cryocat.app.discovery.entries_for_class` for NPC — operations match
  the exposed method surface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryocat.core import cryomotl, cryomask
from cryocat.analysis.assembly import NPC, CnComplex
from cryocat.app import discovery
from cryocat.app.suite.pages import pcomplexes
from cryocat.app.suite.pages.pcomplexes import (
    COMPLEX_CLASSES,
    motl_from_pool_rows, motl_to_pool_rows,
)
from cryocat.app.pool import PoolState, insert_motl



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
    state = PoolState.from_stores({}, {}, 0)
    state, motl_id = insert_motl(state, motl.df)
    rebuilt = motl_from_pool_rows(motl_id)
    assert rebuilt is not None
    pd.testing.assert_frame_equal(
        rebuilt.df.reset_index(drop=True),
        motl.df.reset_index(drop=True),
        check_dtype=False,
    )


def test_motl_from_pool_rows_missing_or_empty_returns_none():
    assert motl_from_pool_rows("nonexistent-motl-id") is None
    assert motl_from_pool_rows(None) is None


# ── Registry-driven page plumbing ───────────────────────────────────────────


def test_complex_classes_has_all_concrete_types():
    expected = {"CnComplex", "DnComplex", "NPC",
                "TetrahedralComplex", "OctahedralComplex", "IcosahedralComplex"}
    assert set(COMPLEX_CLASSES) == expected


def test_complex_classes_values_are_the_actual_classes():
    assert COMPLEX_CLASSES["NPC"] is NPC
    assert COMPLEX_CLASSES["CnComplex"] is CnComplex


def test_entries_for_class_npc_includes_instance_methods():
    """entries_for_class(NPC) must surface at least the expected instance ops."""
    entries = discovery.entries_for_class(NPC)
    labels = {e.label for e in entries}
    required = {
        "Create affiliation", "Get centers as motl", "Occupancy",
        "Clean per object", "Merge subunits",
        "Circumference", "Assign subunit order", "Get object stats",
        "Unify NN orientations",
    }
    missing = required - labels
    assert not missing, f"Missing entries: {missing}"


def test_entries_for_class_npc_no_duplicate_labels():
    entries = discovery.entries_for_class(NPC)
    labels = [e.label for e in entries]
    assert len(labels) == len(set(labels)), f"Duplicate labels: {labels}"


def test_entries_for_class_returns_infer_kind():
    """Every entry for NPC has a known returns kind."""
    valid = {"motl", "dataframe", "none", "features", "object", None}
    for e in discovery.entries_for_class(NPC):
        assert e.returns in valid, f"{e.label!r} has unknown returns={e.returns!r}"


def test_entries_for_class_icosahedral_includes_geometry_methods():
    """IcosahedralComplex entries must include the new Y1 geometry methods."""
    from cryocat.analysis.assembly import IcosahedralComplex
    entries = discovery.entries_for_class(IcosahedralComplex)
    labels = {e.label for e in entries}
    required = {
        "Fit geometry from markers",
        "Feature vectors",
        "Write features to CMM",
        "Expand to subparticles",
        "Assign subunit order",
    }
    missing = required - labels
    assert not missing, f"Missing entries: {missing}"


def test_entries_for_class_sorted_by_group_order():
    """entries_for_class must return entries sorted by (group, order, label)."""
    from cryocat.analysis.assembly import IcosahedralComplex
    entries = discovery.entries_for_class(IcosahedralComplex)
    sort_keys = [(e.group or "\x7f", e.order, e.label) for e in entries]
    assert sort_keys == sorted(sort_keys)


def test_make_handle_geometry_fitted_false_initially():
    """A freshly constructed complex has geometry_fitted=False in its handle."""
    import pandas as pd
    from cryocat.core.cryomotl import Motl
    from cryocat.analysis.assembly import IcosahedralComplex
    from cryocat.app.components import complex_registry as cr
    rows = _ring_subunits((100, 100, 100), n=6)
    df = pd.DataFrame(rows)
    df["subtomo_id"] = range(1, len(df) + 1)
    motl = Motl(df)
    ico = IcosahedralComplex(motl)
    handle = cr.make_handle(ico, "complex-1", "ico_1 (IcosahedralComplex)", "motl-1", {})
    assert handle["geometry_fitted"] is False
    assert handle["radius"] is None
