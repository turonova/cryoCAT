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


def ordered_selection_to_motl_links(
    selected: list[str] | str | None,
) -> dict[str, list[str]] | None:
    """Build motl_links from an ordered selection.

    First id is the query; remaining ids are the neighbours.  A single id is a
    self-comparison: the same motl in both roles.  Returns None when empty.
    """
    if not selected:
        return None
    if isinstance(selected, str):
        selected = [selected]
    neighbours = list(selected[1:]) if len(selected) > 1 else [selected[0]]
    return {"query": [selected[0]], "neighbour": neighbours}


def get_motl_role_id(links: dict | None, role: str) -> str | None:
    """Return the first id for *role* from *links*, or None.

    Handles both list values (canonical) and bare strings (legacy).
    """
    val = (links or {}).get(role)
    if isinstance(val, list):
        return val[0] if val else None
    return val or None


def get_motl_role_ids(links: dict | None, role: str) -> list[str]:
    """Return all ids for *role* from *links* as a list; empty when absent."""
    val = (links or {}).get(role)
    if isinstance(val, list):
        return val
    return [val] if val else []


def get_motl_link(ref: dict | None, role: str) -> str | None:
    """Return the first motl id for *role* from ``ref["motl_links"]``, or None."""
    if not isinstance(ref, dict):
        return None
    return get_motl_role_id(ref.get("motl_links"), role)


def has_source_motl(ref: dict | None) -> bool:
    """True when *ref* carries at least one motl link in ``motl_links``."""
    if not isinstance(ref, dict):
        return False
    return bool(ref.get("motl_links"))
