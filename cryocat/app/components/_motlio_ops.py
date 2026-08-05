"""Pure-function helpers for motlio.

Extracted from motlio.py so callback bodies can be thin (§3).
Phase 10 (Z5): dcc.Upload and base64/tempfile helpers removed; files are
loaded directly from server-side paths.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from cryocat.core.cryomotl import Motl


# ── File extension → motl type ───────────────────────────────────────────────

def motl_type_from_path(path: str) -> str | None:
    """Return the motl type implied by *path*'s extension, or None."""
    ext = os.path.splitext(path)[-1].lower()
    return {
        ".em": "emmotl",
        ".star": "relion",
        ".csv": "stopgap",
        ".tbl": "dynamo",
    }.get(ext)


# ── Relion version → internal type string ────────────────────────────────────

def relion_version_to_type(version: float) -> str:
    """Map a Relion version float to the Motl.load type string."""
    if version == 5.0:
        return "relion5"
    if version == 5.1:
        return "relion5_1"
    return "relion"


# ── Motl load ─────────────────────────────────────────────────────────────────

def load_motl_from_path(
    path: str,
    motl_type: str,
    rln_version: float | None,
    rln_pixelsize: float | None,
    rln_binning: float | None,
    rln_tomoformat: str | None,
    rln_subtomoformat: str | None,
    rln_tomos: str | None,
) -> tuple:
    """Load a Motl from a server-side *path*; return stores tuple.

    Returns (table_data, extra_data, relion_optics, relion_tomos, motl_type, relion_params).

    For Relion 5 the tomogram STAR content (``rln_tomos``) is still supplied
    as a string from the tomos-store and written to a temp file, because
    ``Motl.load`` requires a file path for ``input_tomograms``.
    """
    from cryocat.app.components.filesystem import resolve_input

    resolved, err = resolve_input(path)
    if err:
        raise ValueError(f"Cannot resolve path {path!r}: {err}")

    extra_data = None
    relion_optics = None
    relion_tomos_out = None
    relion_params = None

    if motl_type != "relion":
        motl = Motl.load(resolved, motl_type)
        if motl_type == "stopgap":
            extra_data = motl.sg_df.to_dict("records")
        elif motl_type == "dynamo":
            extra_data = motl.dynamo_df.to_dict("records")
    else:
        rln_kwargs: dict = {}
        actual_type = relion_version_to_type(rln_version) if rln_version else "relion"

        if actual_type == "relion" and rln_version is not None:
            rln_kwargs["version"] = rln_version

        for val, kw in (
            (rln_pixelsize, "pixel_size"),
            (rln_binning, "binning"),
            (rln_tomoformat, "tomo_format"),
            (rln_subtomoformat, "subtomo_format"),
        ):
            if val:
                rln_kwargs[kw] = val

        tomo_tmp_path = None
        if actual_type in ("relion5", "relion5_1") and rln_tomos:
            suffix = os.path.splitext(resolved)[-1] or ".star"
            tomo_tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, mode="w", encoding="utf-8", newline="\n"
            )
            tomo_tmp.write(rln_tomos)
            tomo_tmp.close()
            tomo_tmp_path = tomo_tmp.name
            rln_kwargs["input_tomograms"] = tomo_tmp_path

        try:
            motl = Motl.load(resolved, actual_type, **rln_kwargs)
        finally:
            if tomo_tmp_path and os.path.exists(tomo_tmp_path):
                os.remove(tomo_tmp_path)

        motl_type = actual_type
        extra_data = motl.relion_df.to_dict("records")
        relion_optics = motl.optics_data.to_dict("records") if motl.optics_data is not None else None

        if actual_type in ("relion5", "relion5_1") and hasattr(motl, "tomo_df"):
            relion_tomos_out = motl.tomo_df.to_dict("records")

        relion_params = {
            "version": rln_version,
            "pixel_size": getattr(motl, "pixel_size", None),
            "binning": getattr(motl, "binning", None),
            "tomo_format": rln_tomoformat or "",
            "subtomo_format": rln_subtomoformat or "",
        }

    table_data = motl.df.fillna(0.0).to_dict("records")
    return table_data, extra_data, relion_optics, relion_tomos_out, motl_type, relion_params


# ── Motl save helpers ─────────────────────────────────────────────────────────

def validate_save(motl_type: str, path: str, rln_kwargs: dict) -> list[str]:
    """Return user-visible validation messages; empty list means OK.

    Checks that are now done UP FRONT instead of failing half-way through save.
    """
    msgs: list[str] = []
    if not path:
        msgs.append("Specify an output filename.")
    if not motl_type:
        msgs.append("Select an output type.")
    if motl_type == "relion":
        version = rln_kwargs.get("rln_version")
        if version == 5.0 and not rln_kwargs.get("rln_tomos"):
            msgs.append("Relion 5.0 save requires a tomogram file.")
    return msgs


def save_kwargs_from_store(rln_state: dict | None, rln_tomos_orig) -> dict:
    """Assemble save_motl kwargs from the rln-value store and inherited tomos."""
    rln = rln_state or {}
    return {
        "rln_tomos": rln.get("tomos") or rln_tomos_orig,
        "rln_binning": rln.get("binning"),
        "rln_pixel_size": rln.get("pixel_size"),
        "rln_tomo_format": rln.get("tomo_format"),
        "rln_subtomo_format": rln.get("subtomo_format"),
        "rln_version": rln.get("version"),
        "rln_use_original": rln.get("use_original", False),
    }


def load_kwargs_from_store(rln_value: dict | None) -> dict:
    """Unpack rln-value store into load_motl_from_path keyword args."""
    rln = rln_value or {}
    return {
        "rln_version": rln.get("version"),
        "rln_pixelsize": rln.get("pixel_size"),
        "rln_binning": rln.get("binning"),
        "rln_tomoformat": rln.get("tomo_format"),
        "rln_subtomoformat": rln.get("subtomo_format"),
    }


def filter_by_class(
    motl_df: pd.DataFrame,
    column_id: str,
    class_filter: list,
    checklist_options: list,
    results_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """Apply class-based filtering for the full-save path.  Returns None on no-op."""
    class_map = results_df.set_index("subtomo_id")["class"]
    motl_df = motl_df.copy()
    motl_df[column_id] = np.nan
    motl_df[column_id] = motl_df["subtomo_id"].map(class_map)

    if len(checklist_options) == 0:
        return None
    if len(checklist_options) == 1:
        if checklist_options[0] == "Drop unassigned entries":
            if class_filter:
                motl_df = motl_df[motl_df[column_id] != 0]
            # else: keep all rows
        else:
            motl_df = motl_df[motl_df[column_id].isin([int(x) for x in class_filter])]
    else:
        motl_df = motl_df[motl_df[column_id].isin([int(x) for x in class_filter])]

    return motl_df.dropna(subset=[column_id])
