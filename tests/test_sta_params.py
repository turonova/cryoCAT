"""Tests for the path-resolution layer on top of StaParameters / Stopgap / NovaSta.

Covers the rules the user spelled out:

* STOPGAP -- ``get_motl_base_name`` resolves to
  ``rootdir/lists/<motl name><separator>``. ``working_dir`` overrides the
  ``rootdir`` column; the ``lists/`` subdirectory is still appended.
* novaSTA -- the stored ``motl`` column carries the (possibly relative,
  possibly absolute) full path. ``working_dir`` joins onto relative paths
  and replaces the directory portion of absolute paths.
* Auxiliary resolvers (wedge_list, mask, ccmask, ref_base) mirror the same
  per-format conventions.

Touches no existing assertions (guideline §2).
"""
from __future__ import annotations

import pandas as pd
import pytest

from cryocat.analysis.sta import (
    StopgapParams, NovaStaParams,
    _apply_working_dir,
)


# ── _apply_working_dir contract (novaSTA-style) ──────────────────────────────


@pytest.mark.parametrize(
    "path, working_dir, expected_suffix",
    [
        ("./ddd",            None,        "./ddd"),
        ("./ddd",            "/scratch",  "ddd"),
        ("/gg/cc/motl_base", "/scratch",  "motl_base"),
        ("/gg/cc/motl_base", None,        "/gg/cc/motl_base"),
        ("motl_base",        "/scratch",  "motl_base"),
        ("motl_base",        None,        "motl_base"),
        ("../foo/bar",       "/scratch",  "bar"),  # only basename joined? no — see below
    ],
)
def test_apply_working_dir_contract(path, working_dir, expected_suffix):
    """All transformations end with the expected basename / relative tail.

    Use suffix comparison so platform separators don't matter.
    """
    out = _apply_working_dir(path, working_dir)
    if working_dir is None:
        # Identity when no override is given.
        assert out == path
        return
    # For relative paths the join keeps the relative tail; for absolute paths
    # we strip down to the basename.
    if expected_suffix == "bar":
        # ../foo/bar is relative; the join preserves the whole tail.
        assert out.replace("\\", "/").endswith("../foo/bar")
    elif expected_suffix == "ddd":
        # ./ddd is relative; the leading ./ is stripped by PurePosixPath but
        # the tail is preserved.
        assert out.replace("\\", "/").endswith("/ddd") or out.replace("\\", "/").endswith("ddd")
    elif expected_suffix == "motl_base":
        assert out.replace("\\", "/").endswith("/motl_base") or out.endswith("motl_base")


def test_apply_working_dir_absolute_replaces_dir():
    """Absolute paths have their directory replaced with working_dir."""
    out = _apply_working_dir("/gg/cc/motl_base", "/scratch").replace("\\", "/")
    assert out == "/scratch/motl_base"


def test_apply_working_dir_relative_joins_onto_dir():
    """Relative paths get working_dir prepended; tail preserved verbatim."""
    out = _apply_working_dir("./ddd", "/scratch").replace("\\", "/")
    # PurePosixPath normalises ./ddd -> ddd, then Path(/scratch) / ddd.
    assert out == "/scratch/ddd"


def test_apply_working_dir_none_is_identity():
    assert _apply_working_dir("./anything", None) == "./anything"
    assert _apply_working_dir("/absolute/path", None) == "/absolute/path"
    assert _apply_working_dir("bare_name", None) == "bare_name"


# ── STOPGAP: rootdir + lists/ resolution ────────────────────────────────────


def _stopgap_df():
    """Minimal STOPGAP params DataFrame with the columns the resolvers read."""
    return pd.DataFrame({
        "rootdir": ["/work/run42"],
        "motl name": ["allmotl_lt"],
        "wedgelist name": ["wedge_list_noInterpol.star"],
        "mask name": ["mask_64px.em"],
        "ccmask name": ["cc_mask_64px.em"],
        "ref name": ["pent_b2_64px_ref"],
        "iteration": [1],
    })


def test_stopgap_motl_base_name_uses_rootdir_and_lists():
    sg = StopgapParams(_stopgap_df())
    out = sg.get_motl_base_name(separator="_").replace("\\", "/")
    assert out == "/work/run42/lists/allmotl_lt_"


def test_stopgap_motl_base_name_working_dir_overrides_rootdir():
    sg = StopgapParams(_stopgap_df())
    out = sg.get_motl_base_name(separator="_", working_dir="/scratch/altrun").replace("\\", "/")
    # working_dir replaces /work/run42; the /lists/ subdir is preserved.
    assert out == "/scratch/altrun/lists/allmotl_lt_"


def test_stopgap_motl_base_name_no_rootdir_falls_back_to_bare_name():
    df = _stopgap_df()
    df["rootdir"] = [None]
    sg = StopgapParams(df)
    assert sg.get_motl_base_name(separator="_") == "allmotl_lt_"


def test_stopgap_resolve_wedge_list_in_lists_subdir():
    sg = StopgapParams(_stopgap_df())
    out = sg.resolve_wedge_list().replace("\\", "/")
    assert out == "/work/run42/lists/wedge_list_noInterpol.star"


def test_stopgap_resolve_mask_in_masks_subdir():
    sg = StopgapParams(_stopgap_df())
    out = sg.resolve_mask().replace("\\", "/")
    assert out == "/work/run42/masks/mask_64px.em"


def test_stopgap_resolve_ccmask_in_masks_subdir():
    sg = StopgapParams(_stopgap_df())
    out = sg.resolve_ccmask().replace("\\", "/")
    assert out == "/work/run42/masks/cc_mask_64px.em"


def test_stopgap_resolve_ref_base_in_refs_subdir():
    sg = StopgapParams(_stopgap_df())
    out = sg.resolve_ref_base(separator="_").replace("\\", "/")
    # Downstream code appends <iter>.em.
    assert out == "/work/run42/refs/pent_b2_64px_ref_"


def test_stopgap_resolvers_honour_working_dir_override():
    sg = StopgapParams(_stopgap_df())
    assert sg.resolve_wedge_list("/alt").replace("\\", "/") == "/alt/lists/wedge_list_noInterpol.star"
    assert sg.resolve_mask("/alt").replace("\\", "/") == "/alt/masks/mask_64px.em"
    assert sg.resolve_ccmask("/alt").replace("\\", "/") == "/alt/masks/cc_mask_64px.em"
    assert sg.resolve_ref_base("/alt").replace("\\", "/") == "/alt/refs/pent_b2_64px_ref_"


# ── novaSTA: motl-column-as-path resolution ─────────────────────────────────


def _novasta_df(motl_value):
    return pd.DataFrame({
        "motl": [motl_value],
        "wedge list": ["../wedges/wedge_list.star"],
        "mask": ["/abs/path/mask.em"],
        "cc mask": ["ccmask.em"],
        "ref": ["../ref_base"],
        "iteration": [1],
    })


def test_novasta_motl_base_name_passes_through_when_no_override():
    nv = NovaStaParams(_novasta_df("../virion_motl_cleaned"))
    out = nv.get_motl_base_name(separator="_")
    assert out == "../virion_motl_cleaned_"


def test_novasta_motl_base_name_relative_path_joins_onto_working_dir():
    nv = NovaStaParams(_novasta_df("./virion_motl_cleaned"))
    out = nv.get_motl_base_name(separator="_", working_dir="/scratch/run").replace("\\", "/")
    # PurePosixPath strips the ./ from ./virion_motl_cleaned.
    assert out == "/scratch/run/virion_motl_cleaned_"


def test_novasta_motl_base_name_absolute_path_replaces_dir():
    nv = NovaStaParams(_novasta_df("/gg/cc/virion_motl_cleaned"))
    out = nv.get_motl_base_name(separator="_", working_dir="/scratch/run").replace("\\", "/")
    assert out == "/scratch/run/virion_motl_cleaned_"


def test_novasta_motl_base_name_bare_name_assumes_working_dir():
    nv = NovaStaParams(_novasta_df("virion_motl_cleaned"))
    out = nv.get_motl_base_name(separator="_", working_dir="/scratch/run").replace("\\", "/")
    assert out == "/scratch/run/virion_motl_cleaned_"


def test_novasta_resolvers_apply_working_dir_per_column():
    nv = NovaStaParams(_novasta_df("./motl_base"))
    # Relative wedge list: joined onto working_dir.
    out_w = nv.resolve_wedge_list("/scratch").replace("\\", "/")
    assert out_w.endswith("/scratch/../wedges/wedge_list.star") or out_w.endswith("/scratch/wedges/wedge_list.star") \
        or out_w == "/scratch/../wedges/wedge_list.star"
    # Absolute mask: directory replaced.
    assert nv.resolve_mask("/scratch").replace("\\", "/") == "/scratch/mask.em"
    # Bare cc mask: joined.
    assert nv.resolve_ccmask("/scratch").replace("\\", "/") == "/scratch/ccmask.em"
    # Reference base with separator.
    out_r = nv.resolve_ref_base("/scratch").replace("\\", "/")
    assert out_r.endswith("ref_base_")


def test_novasta_resolvers_pass_through_without_override():
    nv = NovaStaParams(_novasta_df("./motl_base"))
    assert nv.resolve_wedge_list() == "../wedges/wedge_list.star"
    assert nv.resolve_mask() == "/abs/path/mask.em"
    assert nv.resolve_ccmask() == "ccmask.em"
    assert nv.resolve_ref_base() == "../ref_base_"


# ── evaluate_*_from_params plumbs working_dir to the resolver ────────────────


def test_evaluate_from_params_accepts_working_dir(monkeypatch):
    """Confirm the kwarg flows from the public entry point down to the resolver."""
    from cryocat.analysis import sta as sta_mod
    captured: dict = {}

    def _fake_evaluate_alignment(base, start_it, end_it, **kwargs):
        captured["base"] = base
        captured["start_it"] = start_it
        captured["end_it"] = end_it
        captured["motl_type"] = kwargs.get("motl_type")
        return [pd.DataFrame()]

    monkeypatch.setattr(sta_mod, "evaluate_alignment", _fake_evaluate_alignment)

    sg = StopgapParams(_stopgap_df())
    sta_mod.evaluate_alignment_from_params(sg, working_dir="/scratch/altrun")
    assert captured["base"].replace("\\", "/") == "/scratch/altrun/lists/allmotl_lt_"
    assert captured["motl_type"] == "stopgap"


def test_compute_alignment_statistics_from_params_accepts_working_dir(monkeypatch):
    from cryocat.analysis import sta as sta_mod
    captured: dict = {}

    def _fake_compute(base, start_it, end_it, **kwargs):
        captured["base"] = base
        captured["motl_type"] = kwargs.get("motl_type")
        return pd.DataFrame()

    monkeypatch.setattr(sta_mod, "compute_alignment_statistics", _fake_compute)

    nv = NovaStaParams(_novasta_df("/gg/cc/virion_motl"))
    sta_mod.compute_alignment_statistics_from_params(nv, working_dir="/scratch/run")
    assert captured["base"].replace("\\", "/") == "/scratch/run/virion_motl_"
    assert captured["motl_type"] == "emmotl"
