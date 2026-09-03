"""Source-motl overlap helpers shared by pnn.py and ptango.py.

Pure functions only — no Dash imports, no callbacks.  The id-overlap check is
the single authoritative implementation for W2 of LOADED_TABLE_MOTL_LINK.md.
"""
from __future__ import annotations

import pandas as pd


def check_motl_overlap(
    df: pd.DataFrame,
    id_col: str,
    motl_id: str,
) -> tuple[int, int, str]:
    """Return (matched, total, message) for a table-vs-motl id-overlap check.

    Parameters
    ----------
    df:
        Loaded table DataFrame.
    id_col:
        Column in *df* holding particle ids (e.g. ``"qp_subtomo_id"``).
    motl_id:
        Pool id of the candidate source motl (e.g. ``"motl_2"``).

    Returns
    -------
    matched:
        Number of ids in ``df[id_col]`` present in the motl's ``subtomo_id``.
    total:
        Total distinct ids in ``df[id_col]``.
    message:
        One-line status string with a ✓ / ⚠ / ✗ indicator.
    """
    from cryocat.app.pool import get_rows, PoolPayloadMissing

    try:
        motl_rows = get_rows(motl_id)
    except PoolPayloadMissing:
        return 0, 0, f"Motl {motl_id!r} not found in pool — reload it first."

    motl_df = pd.DataFrame(motl_rows)
    motl_ids = set(motl_df["subtomo_id"].dropna().astype(float).astype(int))

    table_ids = df[id_col].dropna().astype(float).astype(int)
    total = len(table_ids)
    matched = int(table_ids.isin(motl_ids).sum())

    if matched == 0:
        msg = f"0 of {total:,} ids matched {motl_id} ✗ no overlap"
    elif matched == total:
        msg = f"{matched:,} of {total:,} ids matched {motl_id} ✓"
    else:
        msg = f"{matched:,} of {total:,} ids matched {motl_id} ⚠ likely the wrong motl"

    return matched, total, msg


def has_source_motl(ref: dict | None) -> bool:
    """True when *ref* has a non-None ``source_motl_id``.

    Distinguishes three cases:
    * key absent (computed table — no restriction) → True
    * ``source_motl_id = None`` (loaded without source) → False
    * ``source_motl_id = "motl_N"`` (loaded with source) → True
    """
    if not isinstance(ref, dict):
        return False
    if "source_motl_id" not in ref:
        return True
    return ref["source_motl_id"] is not None
