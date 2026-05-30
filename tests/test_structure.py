import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from cryocat.core import cryomotl
from cryocat.analysis import structure

DATA_DIR = Path(__file__).parent / "test_data" / "structure_data"


def test_unify_nn_orientations():
    ir_motl = cryomotl.Motl.load(str(DATA_DIR / "ir_input.em"))
    result = structure.NPC.unify_nn_orientations(ir_motl, dist_threshold=10000)
    gt = cryomotl.Motl.load(str(DATA_DIR / "gt_ir_flipped.em"))

    result_df = result.df.sort_values("subtomo_id").reset_index(drop=True)
    gt_df = gt.df.sort_values("subtomo_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(result_df, gt_df, check_dtype=False, atol=1e-4)


def test_cluster_subunits_to_rings():
    result = structure.NPC.cluster_subunits_to_rings(
        input_motl_path=str(DATA_DIR / "gt_ir_flipped.em"),
        mask_size=72,
        entry_mask_coord=(34, 61, 36),
        exit_mask_coord=(34, 17, 36),
        npc_radius=55,
        max_trace_distance=5,
        min_trace_distance=0,
    )
    gt = cryomotl.Motl.load(str(DATA_DIR / "gt_ir_merged.em"))

    result_df = result.df.sort_values("subtomo_id").reset_index(drop=True)
    gt_df = gt.df.sort_values("subtomo_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(result_df, gt_df, check_dtype=False, atol=1e-4)


def _make_toy_chain_motl():
    """Two chains of 3 particles each in one tomogram, with exit coordinates."""
    rows = []
    for chain_id in (1, 2):
        for order in (1, 2, 3):
            rows.append({
                "score": 0.0, "geom1": 0.0, "geom2": float(order),
                "subtomo_id": float(chain_id * 10 + order),
                "tomo_id": 1.0, "object_id": float(chain_id),
                "subtomo_mean": 0.0,
                "x": float(order), "y": float(chain_id), "z": 0.0,
                "shift_x": 0.0, "shift_y": 0.0, "shift_z": 0.0,
                "geom3": 0.0, "geom4": 0.0, "geom5": 0.0,
                "phi": 0.0, "psi": 0.0, "theta": 0.0, "class": 1.0,
                "exit_x": float(order) + 0.5, "exit_y": float(chain_id), "exit_z": 0.0,
            })
    df = pd.DataFrame(rows)
    m = cryomotl.Motl()
    m.df = df
    return m


def test_get_chain_stats_no_keyerror():
    chain = structure.Chain(
        traced_motl=_make_toy_chain_motl(), pixel_size=1.0,
        column_name="tomo_id", chain_id_col="object_id", order_id_col="geom2",
    )
    stats = chain.get_chain_stats(min_chain_size=2)
    assert len(stats) == 4


def test_get_chain_stats_chain_size_is_particle_count():
    chain = structure.Chain(
        traced_motl=_make_toy_chain_motl(), pixel_size=1.0,
        column_name="tomo_id", chain_id_col="object_id", order_id_col="geom2",
    )
    stats = chain.get_chain_stats(min_chain_size=2)
    assert set(stats["chain_size"].unique()) == {3.0}


def test_get_chain_stats_rot_unit_vectors():
    chain = structure.Chain(
        traced_motl=_make_toy_chain_motl(), pixel_size=1.0,
        column_name="tomo_id", chain_id_col="object_id", order_id_col="geom2",
    )
    stats = chain.get_chain_stats(min_chain_size=2)
    rot = stats[["rot_x", "rot_y", "rot_z"]].values.astype(float)
    np.testing.assert_allclose(np.linalg.norm(rot, axis=1), 1.0, atol=1e-6)
