"""Pure helpers for the Memthick Analyze section (M2). No Dash.

Three small, unit-testable pieces:

* :func:`resolve_thickness_csv` — mirrors the notebook's ``out(membrane,
  suffix)`` helper. Given an output folder, a segmentation base name, and a
  membrane name, returns the canonical path of the thickness CSV. The
  Analyze panel feeds these into
  :func:`memthick_analyze_plot.load_membrane_data` (which then
  auto-discovers the matching ``*_int_profiles.pkl`` and stats files).

* :func:`parse_membrane_names` — turn the user's free-text input ("ER, IMM,
  OMM" / one-per-line / whitespace-separated) into a clean list.

* :func:`motl_to_pool_rows` — convert a :class:`cryocat.core.cryomotl.Motl`
  (or anything with a ``.df``) into the list-of-row-dicts shape the suite
  pool's ``motlsink`` expects.

Kept Dash-free so the unit tests don't have to mock callbacks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_thickness_csv(
    output_path: str | Path,
    seg_base: str,
    membrane: str,
    suffix: str = "thickness.csv",
) -> Path:
    """Return ``<output_path>/<seg_base>_<membrane>_<suffix>``.

    Mirrors the ``out(membrane, suffix)`` lambda in
    ``memthick_analysis.ipynb`` (cell 5). The Analyze panel uses this to
    build the path it hands to
    :func:`memthick_analyze_plot.load_membrane_data`; the rest of the
    discovery (``*_int_profiles.pkl``, ``*_boundary_stats.txt``) happens
    inside that loader via ``auto_discover_related_files=True``.

    Parameters
    ----------
    output_path : str or Path
        Pipeline output folder.
    seg_base : str
        Segmentation base name (matches the M1 segmentation filename stem
        and the pipeline's ``segbase`` everywhere downstream).
    membrane : str
        Membrane name as used at pipeline-run time (e.g. ``"IMM"``).
    suffix : str
        File suffix to append. Defaults to the thickness CSV — the loader
        derives the pkl / stats paths from this one.
    """
    return Path(output_path) / f"{seg_base}_{membrane}_{suffix}"


def parse_membrane_names(text: str | None) -> list[str]:
    """Parse a free-text input listing membrane names into a clean list.

    Accepts comma- / semicolon- / newline- / whitespace-separated input.
    Empty entries are dropped; duplicates are kept in first-occurrence
    order. The output preserves user-visible spelling (case, hyphens).
    """
    if not text:
        return []
    # Normalise separators to whitespace, then split.
    cleaned = (
        str(text)
        .replace(",", " ")
        .replace(";", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
    )
    seen: set[str] = set()
    out: list[str] = []
    for part in cleaned.split():
        name = part.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def motl_to_pool_rows(motl: Any) -> list[dict]:
    """Convert a :class:`cryocat.core.cryomotl.Motl` to the pool's row format.

    The suite pool stores motls as ``list[dict]`` rows
    (see :mod:`cryocat.app.components.motlsink`). This helper hands the
    underlying DataFrame through ``df.to_dict("records")`` and gracefully
    accepts an already-DataFrame or already-list input so the caller doesn't
    need to branch.

    Returns ``[]`` for ``None`` / empty inputs so the page can safely call
    this without guards.
    """
    if motl is None:
        return []
    df = getattr(motl, "df", motl)
    if hasattr(df, "to_dict"):
        return list(df.to_dict("records"))
    if isinstance(motl, list):
        return list(motl)
    return []


def labelled_motl_payload(
    motls_per_membrane: dict[str, tuple[Any, Any]],
) -> dict[str, Any]:
    """Assemble the payload the Send-to-editor store carries.

    ``motls_per_membrane[<membrane>] = (motl1, motl2)`` (the return shape of
    :func:`memthick_analyze_plot.create_thickness_motls`).
    The page picks which surface to send via a sidebar dropdown and builds
    one ``list[dict]`` of rows. We pre-package both surfaces here so the
    callback's job stays trivial.
    """
    return {
        membrane: {
            "surface1": motl_to_pool_rows(s1),
            "surface2": motl_to_pool_rows(s2),
        }
        for membrane, (s1, s2) in motls_per_membrane.items()
    }
