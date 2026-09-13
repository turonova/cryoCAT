"""Pure helpers shared by the Structure page's intersection panel.

Kept separate from :mod:`cryocat.app.suite.pages.pstructure` so the helpers
stay easy to unit-test without spinning up a Dash app. The helpers
deliberately take and return plain Python / numpy / pandas values; the page
threads them between the registry, the motl pool, and the result store.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cryocat.core.cryomotl import Motl


def motl_from_rows(rows: list[dict]) -> Motl:
    """Reconstruct a :class:`Motl` from pool-store rows.

    The pool-motls store carries
    ``{motl_id: [row, row, ...]}`` where each row is a dict (the
    ``DataFrame.to_dict("records")`` round-trip). This helper builds a fresh
    :class:`Motl` from such a list.

    Parameters
    ----------
    rows : list of dict
        Per-particle rows; every row must carry the standard motl columns
        (``x``/``y``/``z``, ``shift_x``/``shift_y``/``shift_z``, ``phi`` etc.).

    Returns
    -------
    cryocat.core.cryomotl.Motl
        A fresh motl wrapping ``pd.DataFrame(rows)``.

    Raises
    ------
    ValueError
        If ``rows`` is empty or None.
    """
    if not rows:
        raise ValueError("Cannot build a Motl from empty rows.")
    return Motl(pd.DataFrame(rows))


def subset_motl_rows(rows: list[dict], particle_indices) -> list[dict]:
    """Keep the rows whose row-position appears in ``particle_indices``.

    The intersection workflow's hit table reports a 0-based ``particle_id``
    per hit row -- those are positional indices into the original motl rows
    (not the motl's ``subtomo_id`` column). This helper filters by position
    and de-duplicates while preserving first-seen order.

    Parameters
    ----------
    rows : list of dict
        Original motl rows from the pool.
    particle_indices : array-like of int
        Positional indices to keep. Out-of-range or duplicate indices are
        silently dropped / de-duplicated.

    Returns
    -------
    list of dict
        The matching subset (possibly empty).
    """
    idx = np.asarray(list(particle_indices), dtype=int)
    if idx.size == 0:
        return []
    # Drop out-of-range and dedupe while preserving order.
    in_range = idx[(idx >= 0) & (idx < len(rows))]
    seen: set = set()
    out: list[dict] = []
    for i in in_range:
        if i in seen:
            continue
        seen.add(int(i))
        out.append(rows[int(i)])
    return out


def hits_summary_dataframe(data: dict) -> pd.DataFrame:
    """Build a small summary table from an :meth:`intersection_data` result.

    The ``region_summary`` slot of the result is already a DataFrame when
    ``surface_radii`` were provided; this helper handles the case where the
    user ran without radii (no region summary) by falling back to overall
    hit-count statistics so the UI always has something to show.

    Parameters
    ----------
    data : dict
        Output of :meth:`cryocat.analysis.structure.PleomorphicSurface.intersection_data`.

    Returns
    -------
    pandas.DataFrame
        Either the ``region_summary`` slot, or a one-row fallback with the
        total hit count and median source-target distance.
    """
    rs = data.get("region_summary")
    if isinstance(rs, pd.DataFrame) and not rs.empty:
        return rs
    hits = data.get("hits")
    if isinstance(hits, pd.DataFrame) and not hits.empty:
        return pd.DataFrame({
            "region": ["all hits"],
            "n_hits": [len(hits)],
            "median_distance_nm": [float(hits["distance_nm"].median())],
        })
    return pd.DataFrame({"region": [], "n_hits": []})
