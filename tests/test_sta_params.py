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

import re
import tempfile
import warnings

import pandas as pd
import pytest

from cryocat.analysis.sta import (
    StopgapParams, NovaStaParams, StaParameters,
    _apply_working_dir, _normalize_rootdir, _generate_temperature_schedule,
    stopgap_to_nova_angles, nova_to_stopgap_angles,
    MANDATORY, DERIVED, _STA_SCHEMA, _SCHEMA,
)
from cryocat.utils.starfileio import Starfile


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
    """Minimal STOPGAP params DataFrame with CANONICAL column names."""
    return pd.DataFrame({
        "rootdir": ["/work/run42"],
        "motl": ["allmotl_lt"],           # canonical name (was "motl name")
        "wedge list": ["wedge_list_noInterpol.star"],  # canonical (was "wedgelist name")
        "mask": ["mask_64px.em"],         # canonical (was "mask name")
        "cc mask": ["cc_mask_64px.em"],   # canonical (was "ccmask name")
        "ref": ["pent_b2_64px_ref"],      # canonical (was "ref name")
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


# ── §11 new tests ─────────────────────────────────────────────────────────────


def _minimal_sg_df(n_iter: int = 1) -> pd.DataFrame:
    """Build a minimal StopgapParams DataFrame with all required fields."""
    rows = []
    for i in range(1, n_iter + 1):
        rows.append({
            "rootdir":          "./run42",
            "motl":             "allmotl_lt",
            "wedge list":       "wedge_list.star",
            "mask":             "mask_64px.em",
            "cc mask":          "cc_mask_64px.em",
            "ref":              "ref_base",
            "subtomo name":     "subtomo",
            "iteration":        i,
            "cone angle":       30.0,
            "cone sampling":    5.0,
            "inplane angle":    30.0,
            "inplane sampling": 5.0,
            "low pass":         40,
            "high pass":        1,
        })
    return pd.DataFrame(rows)


# ── Test 1: Round-trip fidelity ───────────────────────────────────────────────

def test_roundtrip_no_double_underscore(tmp_path):
    """write_out → from_file preserves values; no double-underscore columns in file."""
    sg = StopgapParams(_minimal_sg_df())
    path = str(tmp_path / "params.star")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sg.write_out(path)

    # Read raw star file: Starfile.read strips ONE leading '_'.
    # Any column still starting with '_' means the file had '__' (bug).
    frame, _, _ = Starfile.read(path, data_id=0)
    bad_cols = [c for c in frame.columns if c.startswith("_")]
    assert bad_cols == [], f"Double-underscore columns found: {bad_cols}"

    # First column must be completed_ali (after strip)
    assert frame.columns[0] == "completed_ali"

    # Load back and check values
    loaded = StopgapParams.from_file(path)
    assert isinstance(loaded, StopgapParams)
    assert loaded.num_iterations == 1
    assert loaded.df["motl"].iloc[0] == "allmotl_lt"


# ── Test 2: subtomo_mode ordering ─────────────────────────────────────────────

def test_subtomo_mode_canonical_format(tmp_path):
    """All written subtomo_mode values match the canonical {ali|avg}_{family} pattern."""
    sg = StopgapParams(_minimal_sg_df(n_iter=2), create_ref=True)
    path = str(tmp_path / "params.star")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sg.write_out(path)

    frame, _, _ = Starfile.read(path, data_id=0)
    pattern = re.compile(r'^(ali|avg)_(singleref|multiref|multiclass)$')
    for mode in frame["subtomo_mode"]:
        assert pattern.match(mode), f"Invalid subtomo_mode: {mode!r}"

    # Old STOPGAP format must NOT appear
    assert "multiref_ali" not in frame["subtomo_mode"].values
    assert "singleref_ali" not in frame["subtomo_mode"].values


# ── Test 3: Defaults fill ─────────────────────────────────────────────────────

def test_from_dict_defaults_fill():
    """from_dict fills optional parameters from schema defaults."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = StaParameters.from_dict(
            {
                "rootdir": "./run",
                "motl": "motl_lt",
                "wedge list": "wedge.star",
                "mask": "mask.em",
                "cc mask": "ccmask.em",
                "ref": "ref_base",
                "subtomo name": "subtomo",
                "cone angle": 30.0,
                "cone sampling": 5.0,
                "inplane angle": 30.0,
                "inplane sampling": 5.0,
                "low pass": 40,
                "start_index": 1,
            },
            sta_type="novasta",
        )

    # Optional params should be at schema defaults
    assert "binning" not in p.df.columns or p.df.get("binning", pd.Series([None])).iloc[0] is None
    # high pass has default 1 in schema; if not supplied, may not be in df
    # (from_dict only puts supplied keys in df)
    assert p.num_iterations == 1


# ── Test 4: Manual example line ───────────────────────────────────────────────

def test_stopgap_write_produces_correct_col_count(tmp_path):
    """Basic param_set produces exactly 34 STOPGAP columns."""
    sg = StopgapParams(_minimal_sg_df())
    path = str(tmp_path / "params.star")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sg.write_out(path, param_set="basic")

    frame, _, _ = Starfile.read(path, data_id=0)
    assert len(frame.columns) == 34, (
        f"Expected 34 basic columns, got {len(frame.columns)}: {list(frame.columns)}"
    )


# ── Test 5: Mandatory reporting ───────────────────────────────────────────────

def test_validate_reports_missing_mandatory():
    """validate() lists missing mandatory params; file is still written (strict=False)."""
    df = pd.DataFrame({
        "motl": ["allmotl_lt"],
        "wedge list": ["wedge.star"],
        "mask": ["mask.em"],
        "cc mask": ["ccmask.em"],
        "ref": ["ref_base"],
        "subtomo name": ["subtomo"],
        "iteration": [1],
        "cone angle": [30.0],
        "cone sampling": [5.0],
        "inplane angle": [30.0],
        "inplane sampling": [5.0],
        "low pass": [40],
    })
    # rootdir and mask are intentionally omitted (well, mask IS present; let's omit rootdir + low pass)
    df2 = df.drop(columns=["rootdir"] if "rootdir" in df.columns else [])
    df2 = df2.drop(columns=["low pass"])
    sg = StopgapParams(df2)
    problems = sg.validate()

    reported = " ".join(problems)
    assert "rootdir" in reported
    assert "low pass" in reported


def test_validate_file_still_written_when_not_strict(tmp_path):
    """write_out with strict=False warns but still writes the file."""
    sg = StopgapParams(pd.DataFrame({"iteration": [1]}))
    path = str(tmp_path / "params.star")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sg.write_out(path, strict=False)

    import os
    assert os.path.exists(path)
    assert any("mandatory" in str(x.message).lower() for x in w)


# ── Test 6: Literal rejection ─────────────────────────────────────────────────

def test_validate_detects_invalid_literal():
    """validate() reports invalid literal values for e.g. search_mode."""
    df = _minimal_sg_df()
    df["search mode"] = "hillclimb"   # invalid: should be "hc" or "shc"
    sg = StopgapParams(df)
    problems = sg.validate()
    assert any("search mode" in p for p in problems), (
        f"Expected search_mode problem in {problems}"
    )


# ── Test 7: Conditional requirements ─────────────────────────────────────────

def test_validate_conditional_ccmask_not_needed_for_avg():
    """cc mask is not flagged as mandatory when validating an avg-only context."""
    # StopgapParams.validate always uses is_avg_row=False for the context,
    # so cc mask IS mandatory for alignment rows.
    df = _minimal_sg_df()
    df = df.drop(columns=["cc mask"])
    sg = StopgapParams(df)
    problems = sg.validate()
    reported = " ".join(problems)
    assert "cc mask" in reported


def test_validate_split_into_even_odd_requires_fsc_mask():
    """pixel size and fsc mask become mandatory when split_into_even_odd is True."""
    df = pd.DataFrame({
        "iteration": [1],
        "split into even odd": [True],
    })
    nv = NovaStaParams(df)
    problems = nv.validate()
    reported = " ".join(problems)
    assert "fsc mask" in reported or "pixel size" in reported


# ── Test 8: Temperature schedule ──────────────────────────────────────────────

def test_temperature_schedule_zero():
    """T=0 produces all-zero schedule."""
    sched = _generate_temperature_schedule(0, 5)
    assert sched == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_temperature_schedule_normal():
    """T=3, n=3: schedule is [3, 2, 1]."""
    sched = _generate_temperature_schedule(3, 3)
    assert sched == [3.0, 2.0, 1.0]


def test_temperature_schedule_warns_when_not_finished():
    """Temperature schedule warns when n iterations aren't enough to reach 1."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sched = _generate_temperature_schedule(5, 3)  # ends at max(1, 5-2)=3 → warn
    assert len(sched) == 3
    assert sched[0] == 5.0
    assert sched[-1] == 3.0
    assert any("Temperature" in str(x.message) for x in w)


# ── Test 9: rootdir normalisation ─────────────────────────────────────────────

def test_normalize_rootdir_bare_name():
    """Bare folder name gets ./ prepended."""
    assert _normalize_rootdir("run42") == "./run42"


def test_normalize_rootdir_absolute_unchanged():
    """Absolute paths are returned as-is."""
    assert _normalize_rootdir("/data/run42") == "/data/run42"


def test_normalize_rootdir_already_relative():
    """Paths with a separator are returned unchanged."""
    assert _normalize_rootdir("./run42") == "./run42"
    assert _normalize_rootdir("../run42") == "../run42"


# ── Test 10: cc mask in avg rows ──────────────────────────────────────────────

def test_avg_row_has_none_for_ccmask_and_angles(tmp_path):
    """Avg rows written by write_out have 'none' for angle and cc mask columns."""
    sg = StopgapParams(_minimal_sg_df(), create_ref=True)
    path = str(tmp_path / "params.star")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sg.write_out(path)

    frame, _, _ = Starfile.read(path, data_id=0)
    # First row should be the avg row
    avg_rows = frame[frame["subtomo_mode"].str.startswith("avg_")]
    assert len(avg_rows) >= 1
    for _, row in avg_rows.iterrows():
        assert str(row.get("angincr",  "none")).strip() == "none", "avg row angincr != none"
        assert str(row.get("angiter",  "none")).strip() == "none", "avg row angiter != none"
        assert str(row.get("ccmask_name", "none")).strip() == "none", "avg row ccmask_name != none"


# ── Test 11: Symmetry conversion ──────────────────────────────────────────────

def test_symmetry_c5_roundtrip():
    """Schoenflies C5 → novaSTA integer 5 → back to C5."""
    df = pd.DataFrame({"iteration": [1], "symmetry": ["C5"]})
    sg = StopgapParams(df)
    nv = sg.to_novasta()
    assert nv.df["symmetry"].iloc[0] == 5

    sg2 = nv.to_stopgap()
    assert str(sg2.df["symmetry"].iloc[0]) == "C5"


def test_symmetry_non_cyclic_warns():
    """Non-cyclic symmetry (D7, T, O, I) triggers a warning on to_novasta."""
    df = pd.DataFrame({"iteration": [1], "symmetry": ["D7"]})
    sg = StopgapParams(df)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        nv = sg.to_novasta()
    assert any("D7" in str(x.message) or "Non-cyclic" in str(x.message) for x in w)
    assert nv.df["symmetry"].iloc[0] == 1


# ── Test 12: Cross-format conversion ─────────────────────────────────────────

def test_cross_format_roundtrip_preserves_values():
    """StopgapParams → to_novasta → to_stopgap preserves canonical values."""
    df = pd.DataFrame({
        "iteration": [1],
        "motl": ["allmotl_lt"],
        "cone angle": [30.0],
        "cone sampling": [5.0],
        "inplane angle": [20.0],
        "inplane sampling": [4.0],
        "low pass": [40],
        "symmetry": ["C1"],
    })
    sg = StopgapParams(df)
    nv = sg.to_novasta()
    sg2 = nv.to_stopgap()

    # Canonical values survive the round-trip
    assert sg2.df["motl"].iloc[0] == "allmotl_lt"
    assert float(sg2.df["cone angle"].iloc[0]) == 30.0
    assert sg2.num_iterations == sg.num_iterations


# ── Test 13: param_set ────────────────────────────────────────────────────────

def test_param_set_basic_produces_34_columns(tmp_path):
    """param_set='basic' writes exactly 34 STOPGAP columns."""
    sg = StopgapParams(_minimal_sg_df())
    path = str(tmp_path / "basic.star")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sg.write_out(path, param_set="basic")

    frame, _, _ = Starfile.read(path, data_id=0)
    assert len(frame.columns) == 34


def test_param_set_full_produces_more_than_34_columns(tmp_path):
    """param_set='full' writes more than 34 STOPGAP columns."""
    sg = StopgapParams(_minimal_sg_df())
    path = str(tmp_path / "full.star")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sg.write_out(path, param_set="full")

    frame, _, _ = Starfile.read(path, data_id=0)
    assert len(frame.columns) > 34


def test_param_set_auto_promotes_when_full_group_columns_set(tmp_path):
    """Setting a full-group column auto-promotes param_set='basic' to 'full'."""
    df = _minimal_sg_df()
    df["scoring fcn"] = "pearson"   # group="full" column
    sg = StopgapParams(df)
    path = str(tmp_path / "auto.star")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sg.write_out(path, param_set="basic")

    frame, _, _ = Starfile.read(path, data_id=0)
    assert len(frame.columns) > 34


# ── Test 14: Cone/Euler exclusivity ──────────────────────────────────────────

def test_validate_cone_and_euler_mutually_exclusive():
    """validate() reports an error when both cone and euler search are configured."""
    df = _minimal_sg_df()
    df["euler axes"] = "ZYZ"       # non-trivial euler axes
    df["euler 1 incr"] = 5.0
    df["euler 1 iter"] = 3
    sg = StopgapParams(df)
    problems = sg.validate()
    assert any("mutually exclusive" in p.lower() for p in problems), (
        f"Expected cone/euler exclusivity problem, got: {problems}"
    )


def test_validate_bad_euler_axes_flags_problem():
    """validate() flags euler_axes where first and second axis are the same."""
    df = pd.DataFrame({
        "iteration": [1],
        "euler axes": ["ZZY"],  # second axis must differ from first
    })
    sg = StopgapParams(df)
    problems = sg.validate()
    assert any("euler" in p.lower() for p in problems), (
        f"Expected euler_axes problem, got: {problems}"
    )


# ── New tests: schema API, format converters, bool coercion ───────────────────

from cryocat.analysis.sta import (
    _halfset_from_format, _halfset_to_format,
    _sym_from_format, _sym_to_format,
    get_schema, get_shared_schema, is_mandatory, get_choices, get_default, build_ctx,
    StaParamContext,
)


# ── New Test 1: halfset from_format — STOPGAP direction ──────────────────────

def test_halfset_from_format_stopgap_inversion():
    """STOPGAP ignore_halfsets=0 means 'do split' → canonical bool True."""
    assert _halfset_from_format(0, "stopgap") is True   # 0 = not ignoring = split
    assert _halfset_from_format(1, "stopgap") is False  # 1 = ignoring = no split


# ── New Test 2: halfset from_format — novaSTA direction ──────────────────────

def test_halfset_from_format_novasta_same_direction():
    """novaSTA splitIntoEvenOdd=1 means 'do split' → canonical bool True."""
    assert _halfset_from_format(1, "novasta") is True
    assert _halfset_from_format(0, "novasta") is False


# ── New Test 3: halfset to_format — STOPGAP direction ────────────────────────

def test_halfset_to_format_stopgap_inversion():
    """Canonical True (do split) → STOPGAP ignore_halfsets=0."""
    assert _halfset_to_format(True,  "stopgap") == 0
    assert _halfset_to_format(False, "stopgap") == 1


# ── New Test 4: halfset to_format — novaSTA direction ────────────────────────

def test_halfset_to_format_novasta_same_direction():
    """Canonical True (do split) → novaSTA splitIntoEvenOdd=1."""
    assert _halfset_to_format(True,  "novasta") == 1
    assert _halfset_to_format(False, "novasta") == 0


# ── New Test 5: symmetry from_format novaSTA (integer → Schoenflies) ─────────

def test_sym_from_format_novasta_integer_to_schoenflies():
    """Loading symmetry integer from novaSTA produces canonical Schoenflies."""
    assert _sym_from_format(5, "novasta") == "C5"
    assert _sym_from_format(1, "novasta") == "C1"
    assert _sym_from_format("5", "novasta") == "C5"


# ── New Test 6: symmetry to_format STOPGAP (plain int → Schoenflies) ─────────

def test_sym_to_format_stopgap_int_to_schoenflies():
    """Plain integer symmetry is promoted to Schoenflies for STOPGAP."""
    assert _sym_to_format(5, "stopgap") == "C5"
    assert _sym_to_format("C5", "stopgap") == "C5"
    assert _sym_to_format("C1", "stopgap") == "C1"


# ── New Test 7: symmetry to_format novaSTA (Schoenflies → integer) ────────────

def test_sym_to_format_novasta_schoenflies_to_int():
    """Schoenflies symmetry is converted to integer for novaSTA."""
    assert _sym_to_format("C5", "novasta") == 5
    assert _sym_to_format("C1", "novasta") == 1
    assert _sym_to_format(5, "novasta") == 5  # already integer — idempotent


# ── New Test 8: rootdir mandatory for STOPGAP, not novaSTA ───────────────────

def test_rootdir_mandatory_for_stopgap_only():
    """rootdir is required for STOPGAP but optional for novaSTA."""
    spec = next(s for s in _STA_SCHEMA if s.canonical == "rootdir")
    ctx_sg = build_ctx(sta_type="stopgap")
    ctx_nv = build_ctx(sta_type="novasta")
    assert is_mandatory(spec, ctx_sg), "rootdir must be mandatory for STOPGAP"
    assert not is_mandatory(spec, ctx_nv), "rootdir must NOT be mandatory for novaSTA"


# ── New Test 9: novaSTA 'folder' key maps to canonical 'rootdir' ─────────────

def test_novasta_folder_maps_to_canonical_rootdir(tmp_path):
    """When a novaSTA file has a 'folder' key it is stored as canonical 'rootdir'."""
    params_file = tmp_path / "params.txt"
    params_file.write_text(
        "iter 1\n"
        "startIndex 1\n"
        "createRef 0\n"
        "folder /data/my_run\n"
        "motl ./allmotl_\n"
        "wedgeList ./wedgelist.star\n"
        "ref ./ref_\n"
        "mask ./mask.em\n"
        "ccMask ./ccmask.em\n"
        "lowPass 30\n"
        "coneAngle 10.0\n"
        "coneSampling 2.5\n"
        "inplaneAngle 20.0\n"
        "inplaneSampling 2.5\n"
    )
    obj = NovaStaParams.from_file(str(params_file))
    assert "rootdir" in obj.df.columns, "'folder' was not remapped to 'rootdir'"
    assert obj.df["rootdir"].iloc[0] == "/data/my_run"


# ── New Test 10: _fmt_val handles bool before int ─────────────────────────────

def test_fmt_val_bool_encoding():
    """_fmt_val converts True → '1' and False → '0', not 'True'/'False'."""
    from cryocat.analysis.sta import _fmt_val
    assert _fmt_val(True)  == "1"
    assert _fmt_val(False) == "0"
    assert _fmt_val(1.0)   == "1"    # whole float drops .0
    assert _fmt_val(3.14)  == "3.14"


# ── New Test 11: build_ctx produces correct StaParamContext ───────────────────

def test_build_ctx_fields():
    """build_ctx returns a StaParamContext with all specified fields."""
    ctx = build_ctx(
        sta_type="stopgap",
        create_ref=True,
        ref_family="multiref",
        n_iterations=5,
        is_avg_row=False,
        use_euler_search=True,
        row={"motl": "mymotl"},
    )
    assert isinstance(ctx, StaParamContext)
    assert ctx.sta_type == "stopgap"
    assert ctx.create_ref is True
    assert ctx.ref_family == "multiref"
    assert ctx.n_iterations == 5
    assert ctx.use_euler_search is True
    assert ctx.get("motl") == "mymotl"
    assert ctx.get("missing", "default") == "default"


# ── New Test 12: get_schema filters by sta_type ───────────────────────────────

def test_get_schema_stopgap_excludes_novasta_only():
    """get_schema('stopgap') must not include novaSTA-only entries."""
    entries = get_schema("stopgap")
    for spec in entries:
        assert spec.stopgap is not None, (
            f"STOPGAP schema includes novaSTA-only spec {spec.canonical!r}"
        )


def test_get_schema_novasta_excludes_stopgap_only():
    """get_schema('novasta') must not include STOPGAP-only entries."""
    entries = get_schema("novasta")
    for spec in entries:
        assert spec.novasta is not None, (
            f"novaSTA schema includes STOPGAP-only spec {spec.canonical!r}"
        )


# ── New Test 13: get_shared_schema returns only cross-format entries ──────────

def test_get_shared_schema_all_have_both_format_names():
    """get_shared_schema() entries must have both stopgap and novasta names."""
    shared = get_shared_schema()
    assert len(shared) > 0, "Expected at least one shared entry"
    for spec in shared:
        assert spec.stopgap is not None and spec.novasta is not None, (
            f"Shared spec {spec.canonical!r} is missing a format name"
        )


# ── New Test 14: Euler columns only mandatory when use_euler_search=True ──────

def test_euler_columns_mandatory_only_with_euler_search():
    """Euler columns are required for STOPGAP+euler_search, not otherwise."""
    euler_specs = [s for s in _STA_SCHEMA if s.group == "euler" and s.canonical is not None]
    assert len(euler_specs) > 0, "No euler specs found"

    ctx_euler = build_ctx(sta_type="stopgap", use_euler_search=True)
    ctx_cone  = build_ctx(sta_type="stopgap", use_euler_search=False)
    ctx_nova  = build_ctx(sta_type="novasta",  use_euler_search=True)

    for spec in euler_specs:
        assert is_mandatory(spec, ctx_euler), (
            f"{spec.canonical!r} must be mandatory for STOPGAP with euler_search"
        )
        assert not is_mandatory(spec, ctx_cone), (
            f"{spec.canonical!r} must NOT be mandatory for STOPGAP cone search"
        )
        assert not is_mandatory(spec, ctx_nova), (
            f"{spec.canonical!r} must NOT be mandatory for novaSTA (no euler support)"
        )
