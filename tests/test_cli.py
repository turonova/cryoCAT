"""Tests for cryocat.cli: the ``cc_<module>`` commands, at parse level only (nothing is loaded or processed)."""
import argparse
import sys
import tomllib
from pathlib import Path

import pytest

from cryocat import cli

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# {"cc_cryomotl": "cryomotl", ...}: the command line entries and their entry functions in cryocat.cli.
COMMANDS = {
    name: target.split(":")[1]
    for name, target in tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]["scripts"].items()
    if name.startswith("cc_")
}


def _module(command):
    return getattr(cli, f"{COMMANDS[command]}_module")


def _subcommands(command):
    """The subparsers of *command* by subcommand name, and the exposed callables behind them."""
    parser, exposed = cli.build_parser(_module(command))
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return action.choices, exposed


def _action(subparser, dest):
    return next(a for a in subparser._actions if a.dest == dest)


def _run(command, argv, monkeypatch):
    """Call the entry function of *command* with *argv*; return the exit code."""
    monkeypatch.setattr(sys, "argv", [command, *argv])
    with pytest.raises(SystemExit) as exit_info:
        getattr(cli, COMMANDS[command])()
    return exit_info.value.code


def _parse(command, argv):
    parser, _ = cli.build_parser(_module(command))
    return parser.parse_args(argv)


SUBCOMMANDS = [(command, name) for command in COMMANDS for name in _subcommands(command)[1]]
MOTL_METHODS = [name for name, c in _subcommands("cc_cryomotl")[1].items() if c.kind == "method"]
MOTL_OUTPUTS = [name for name, c in _subcommands("cc_cryomotl")[1].items() if c.output == "motl"]
MAP_OUTPUTS = [(command, name) for command in COMMANDS for name, c in _subcommands(command)[1].items() if c.output == "map"]


# ---------------------------------------------------------------------------
# Parsers and help
# ---------------------------------------------------------------------------


def test_all_module_commands_are_entries():
    assert set(COMMANDS) == {
        "cc_cryomotl", "cc_cryomap", "cc_cryomask", "cc_tiltstack", "cc_cryowedge", "cc_mdoc", "cc_tmana", "cc_geom",
    }


@pytest.mark.parametrize("command", COMMANDS)
def test_parser_builds(command):
    parser, exposed = cli.build_parser(_module(command))
    assert parser.prog == command
    assert exposed


@pytest.mark.parametrize("command", COMMANDS)
def test_command_help(command, monkeypatch, capsys):
    assert _run(command, ["--help"], monkeypatch) == 0
    assert capsys.readouterr().out.startswith(f"usage: {command}")


@pytest.mark.parametrize("command, subcommand", SUBCOMMANDS)
def test_subcommand_help(command, subcommand, monkeypatch, capsys):
    assert _run(command, [subcommand, "--help"], monkeypatch) == 0
    assert capsys.readouterr().out.startswith(f"usage: {command} {subcommand}")


def test_missing_subcommand_exits(monkeypatch):
    assert _run("cc_cryomap", [], monkeypatch) == 2


# ---------------------------------------------------------------------------
# Boolean options
# ---------------------------------------------------------------------------

_REMOVE_IMAGES = ["remove_images", "--input_mdoc", "in.mdoc", "--idx_to_remove", "idx.txt"]


@pytest.mark.parametrize("token, expected", [("true", True), ("TRUE", True), ("false", False), ("1", True), ("0", False)])
def test_bool_option_accepts(token, expected):
    args = _parse("cc_mdoc", [*_REMOVE_IMAGES, "--numbered_from_1", token])
    assert args.numbered_from_1 is expected


@pytest.mark.parametrize("token", ["yes", "2"])
def test_bool_option_rejects(token, monkeypatch):
    assert _run("cc_mdoc", [*_REMOVE_IMAGES, "--numbered_from_1", token], monkeypatch) == 2


# ---------------------------------------------------------------------------
# Numbers: int | float, and typed loader options
# ---------------------------------------------------------------------------


def test_int_or_float_accepts_fraction():
    args = _parse("cc_cryomask", ["spherical_shell_mask", "--mask_size", "10", "--shell_thickness", "1.5"])
    assert args.shell_thickness == 1.5


# ---------------------------------------------------------------------------
# Required parameters
# ---------------------------------------------------------------------------

MOTL_CONVERTERS = (
    "emmotl2relion", "relion2emmotl", "stopgap2emmotl", "emmotl2stopgap",
    "relion2stopgap", "stopgap2relion", "emmotl2mod", "mod2emmotl",
)


@pytest.mark.parametrize("subcommand", MOTL_CONVERTERS)
def test_converter_output_path_is_required(subcommand):
    assert _action(_subcommands("cc_cryomotl")[0][subcommand], "output_path").required


def test_required_name_must_be_a_parameter():
    from cryocat.utils.classutils import cli_exposed

    with pytest.raises(ValueError, match="not parameters"):
        cli_exposed(required=("missing",))(lambda present=1: None)


# ---------------------------------------------------------------------------
# A number or a path
# ---------------------------------------------------------------------------

_DECONVOLVE = ["deconvolve", "--tilt_stack", "stack.mrc", "--pixel_size_a", "1.0"]


@pytest.mark.parametrize("token, expected", [("3", 3), ("2.5", 2.5), ("-1.5", -1.5),("defocus.txt", "defocus.txt")])
def test_number_or_path(token, expected):
    defocus = _parse("cc_tiltstack", [*_DECONVOLVE, "--defocus", token]).defocus
    assert defocus == expected
    assert type(defocus) is type(expected)


# ---------------------------------------------------------------------------
# Motl formats and map output
# ---------------------------------------------------------------------------

_MOTL_TYPES = ("emmotl", "relion", "relion5", "relion5_1", "stopgap", "dynamo", "mod")


@pytest.mark.parametrize("subcommand", MOTL_METHODS)
def test_motl_method_has_input_type(subcommand):
    action = _action(_subcommands("cc_cryomotl")[0][subcommand], "input_type")
    assert action.default == "emmotl"
    assert tuple(action.choices) == _MOTL_TYPES


@pytest.mark.parametrize("subcommand", MOTL_OUTPUTS)
def test_motl_output_has_output_type(subcommand):
    action = _action(_subcommands("cc_cryomotl")[0][subcommand], "output_type")
    assert action.default == "emmotl"
    assert tuple(action.choices) == _MOTL_TYPES


def test_motl_methods_write_motls():
    assert set(MOTL_METHODS) <= set(MOTL_OUTPUTS) | {"save_to", "split_by_feature", "write_to_model_file"}


def _invocation(command, argv):
    parser, exposed = cli.build_parser(_module(command))
    args = vars(parser.parse_args(argv))
    return cli._invocation(parser, exposed[args.pop("subcommand")], args)


_FLIP = ["flip_handedness", "--input_motl", "in.star", "--output_motl", "out.star"]


@pytest.mark.parametrize("option, value", [("--input_pixel_size", 2.0), ("--input_binning", 2.0), ("--input_version", 4.0)])
def test_relion_read_option_is_a_number(option, value):
    args = _parse("cc_cryomotl", [*_FLIP, "--input_type", "relion", option, "2" if value == 2.0 else "4"])
    assert getattr(args, option[2:]) == value
    assert isinstance(getattr(args, option[2:]), float)


def test_read_option_reaches_motl_load():
    invocation = _invocation("cc_cryomotl", [*_FLIP, "--input_type", "relion5", "--input_input_tomograms", "t.star"])
    assert invocation.read == ("in.star", "relion5", {"input_tomograms": "t.star"})


def test_read_option_of_another_type_is_rejected(monkeypatch):
    assert _run("cc_cryomotl", [*_FLIP, "--input_type", "emmotl", "--input_pixel_size", "2"], monkeypatch) == 2


def test_write_options_go_to_constructor_and_writer():
    invocation = _invocation("cc_cryomotl", [
        *_FLIP, "--output_type", "relion5_1", "--output_binning", "4", "--output_subtomo_size", "64",
    ])
    assert invocation.write == ("relion5_1", {"binning": 4.0}, {"binning": 4.0, "subtomo_size": 64})


def test_relion5_1_writes_without_a_version_option(monkeypatch):
    assert _run("cc_cryomotl", [*_FLIP, "--output_type", "relion5_1", "--output_version", "4"], monkeypatch) == 2


_EMMOTL2RELION = ["emmotl2relion", "--input_motl", "in.em", "--relion_version", "4.0", "--output_path", "out.star"]


@pytest.mark.parametrize("subcommand, present, absent", [
    ("emmotl2relion", {"input_pixel_size", "input_binning", "input_input_tomograms", "output_write_optics", "output_binning"},
     {"input_version", "load_kwargs", "write_kwargs"}),
    ("stopgap2relion", {"input_pixel_size", "input_binning", "output_subtomo_size"}, {"input_version", "load_kwargs"}),
    ("relion2emmotl", {"input_tomo_format", "input_input_tomograms", "pixel_size", "binning"},
     {"input_pixel_size", "input_binning", "input_version", "load_kwargs"}),
    ("relion2stopgap", {"input_pixel_size", "input_binning", "reset_index"}, {"input_version", "load_kwargs"}),
    ("emmotl2stopgap", {"reset_index"}, {"input_pixel_size", "output_binning"}),
])
def test_converter_options(subcommand, present, absent):
    dests = {a.dest for a in _subcommands("cc_cryomotl")[0][subcommand]._actions}
    assert present <= dests
    assert not absent & dests


def test_converter_options_reach_their_kwargs():
    invocation = _invocation("cc_cryomotl", [
        *_EMMOTL2RELION, "--input_binning", "2", "--input_tomo_format", "TS_{}", "--output_write_optics", "false",
    ])
    assert invocation.kwargs["load_kwargs"] == {"binning": 2.0, "tomo_format": "TS_{}"}
    assert invocation.kwargs["write_kwargs"] == {"write_optics": False}


@pytest.mark.parametrize("option", [["--input_tomo_idx", "idx.txt"], ["--output_convert", "true"]])
def test_converter_option_of_another_relion_version_is_rejected(option, monkeypatch):
    assert _run("cc_cryomotl", [*_EMMOTL2RELION, *option], monkeypatch) == 2


def test_converter_option_of_its_relion_version_is_accepted():
    invocation = _invocation("cc_cryomotl", [
        "emmotl2relion", "--input_motl", "in.em", "--relion_version", "5.0", "--output_path", "out.star",
        "--input_input_tomograms", "t.star",
    ])
    assert invocation.kwargs["load_kwargs"] == {"input_tomograms": "t.star"}


def test_same_type_writes_itself():
    invocation = _invocation("cc_cryomotl", [
        *_FLIP, "--input_type", "relion5", "--output_type", "relion5", "--output_subtomo_size", "64",
    ])
    assert invocation.writes_itself
    assert invocation.write == ("relion5", {}, {"subtomo_size": 64})


def test_different_types_build_a_writer():
    invocation = _invocation("cc_cryomotl", [*_FLIP, "--input_type", "relion", "--output_type", "relion5_1"])
    assert not invocation.writes_itself


def test_writes_itself_rejects_constructor_only_option(monkeypatch):
    argv = [*_FLIP, "--input_type", "relion5", "--output_type", "relion5", "--output_input_tomograms", "t.star"]
    assert _run("cc_cryomotl", argv, monkeypatch) == 2


def test_version_parameter_is_derived():
    import re

    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert not re.search(r"\brelion_version\b", source)
    commands = _subcommands("cc_cryomotl")[1]
    checked = {name for name in MOTL_CONVERTERS if cli._version_parameter(commands[name]) is not None}
    assert checked == {"emmotl2relion", "relion2emmotl", "relion2stopgap", "stopgap2relion"}
    assert {cli._version_parameter(commands[name]) for name in checked} == {"relion_version"}


# ---------------------------------------------------------------------------
# Forwarded keyword arguments
# ---------------------------------------------------------------------------


def _forwarded_flags(command, subcommand):
    exposed = _subcommands(command)[1][subcommand]
    return {flag for passed in cli._forwarded(exposed).values() for flag in passed.flags.values()}


def test_em2mrc_offers_write_options():
    dests = {a.dest for a in _subcommands("cc_cryomap")[0]["em2mrc"]._actions}
    assert {"pixel_size", "transpose", "overwrite"} <= dests
    assert "output_kwargs" not in dests


def test_forwarded_option_reaches_call_keywords():
    invocation = _invocation("cc_cryomap", [
        "em2mrc", "--input_path", "in.em", "--output_path", "out.mrc", "--pixel_size", "2.5", "--transpose", "false",
    ])
    assert invocation.kwargs["pixel_size"] == 2.5
    assert invocation.kwargs["transpose"] is False


def test_forwarded_option_not_given_is_not_passed():
    invocation = _invocation("cc_cryomap", ["em2mrc", "--input_path", "in.em", "--output_path", "out.mrc"])
    assert not {"pixel_size", "transpose", "overwrite"} & set(invocation.kwargs)


def test_name_passed_by_the_function_is_not_offered():
    # deconvolve passes pixel_size=pixel_size_a to cryomap.write itself.
    assert _forwarded_flags("cc_cryomap", "deconvolve") == {"transpose", "overwrite"}


def test_name_of_own_parameter_is_not_offered():
    # lowpass has its own pixel_size, so pixel_size can never reach **output_kwargs.
    assert _forwarded_flags("cc_cryomap", "lowpass") == {"transpose", "overwrite"}


@pytest.mark.parametrize("command, subcommand, flags", [
    ("cc_cryomap", "em2mrc", {"pixel_size", "transpose", "overwrite"}),
    ("cc_cryomask", "spherical_mask", {"pixel_size", "transpose", "overwrite"}),
    ("cc_cryomask", "generate_mask", {"pixel_size", "transpose", "overwrite"}),
    ("cc_cryomotl", "save_to", {"motl_type"}),
    ("cc_tmana", "scores_extract_particles", {"relion_version"}),
    ("cc_tiltstack", "bin", {"pixel_size", "overwrite"}),
])
def test_forwarding_functions_offer_target_options(command, subcommand, flags):
    assert _forwarded_flags(command, subcommand) == flags


_BIN = ["bin", "--tilt_stack", "stack.mrc", "--binning_factor", "2", "--output_path", "out.mrc"]


def test_tiltstack_reaches_write_through_local_dict():
    # TiltStack.write_out merges **output_kwargs into a local dict and passes that on to cryomap.write.
    from cryocat.core import cryomap
    from cryocat.core.tiltstack import TiltStack

    exposed = _subcommands("cc_tiltstack")[1]["bin"]
    keywords = cli.forwarded_keywords(exposed.fn)["output_kwargs"]
    assert keywords.targets == (TiltStack.write_out,)
    assert keywords.chained == (cryomap.write,)
    assert keywords.unresolved is None


def test_tiltstack_forwarded_option_reaches_call_keywords():
    invocation = _invocation("cc_tiltstack", [*_BIN, "--pixel_size", "1.7", "--overwrite", "false"])
    assert invocation.kwargs["pixel_size"] == 1.7
    assert invocation.kwargs["overwrite"] is False


def test_constant_key_of_local_dict_is_not_offered():
    # write_kw = {"data_type": ..., "transpose": False} in TiltStack.write_out.
    dests = {a.dest for a in _subcommands("cc_tiltstack")[0]["bin"]._actions}
    assert "transpose" not in dests
    assert "data_type" not in dests


def test_converter_flags_unchanged():
    assert _forwarded_flags("cc_cryomotl", "emmotl2relion") == {
        "input_pixel_size", "input_binning", "input_tomo_format", "input_subtomo_format", "input_input_tomograms",
        "input_tomo_idx", "output_write_optics", "output_tomo_format", "output_subtomo_format",
        "output_use_original_entries", "output_keep_all_entries", "output_version", "output_binning",
        "output_pixel_size", "output_optics_data", "output_subtomo_size", "output_convert",
    }
    assert _forwarded_flags("cc_cryomotl", "relion2emmotl") == {
        "input_tomo_format", "input_subtomo_format", "input_input_tomograms", "input_tomo_idx",
    }


def _in_memory_relion(cls, **kwargs):
    from cryocat.core import cryomotl

    df = cryomotl.Motl.create_empty_motl_df()
    df.loc[0] = 0.0
    df.loc[0, ["subtomo_id", "tomo_id", "class"]] = [1, 1, 1]
    return cls(df, **kwargs)


def _self_write(motl, argv, monkeypatch):
    """Run a relion5_1 self-write of flip_handedness on *motl*; return the exit code and the calls made."""
    calls = []
    monkeypatch.setattr(cli.cryomotl_module.Motl, "load", classmethod(lambda cls, *a, **k: motl))
    monkeypatch.setattr(motl, "flip_handedness", lambda **kwargs: calls.append(("method", kwargs)))
    monkeypatch.setattr(motl, "write_out", lambda path, **kwargs: calls.append(("write_out", path, kwargs)))
    monkeypatch.setattr(sys, "argv", ["cc_cryomotl", *_FLIP, "--input_type", "relion5_1", "--output_type", "relion5_1", *argv])
    try:
        cli.cryomotl()
    except SystemExit as exit_info:
        return exit_info.code, calls
    return 0, calls


def test_self_write_accepts_option_of_loaded_writer(monkeypatch):
    import pandas as pd
    from cryocat.core.cryomotl import RelionMotlv5

    tomograms = pd.DataFrame({"rlnTomoName": ["TS_1"], "rlnTomoSizeX": [100], "rlnTomoSizeY": [100], "rlnTomoSizeZ": [50]})
    motl = _in_memory_relion(RelionMotlv5, input_tomograms=tomograms, binning=2.0, version=5.1)
    code, calls = _self_write(motl, ["--output_convert", "true"], monkeypatch)
    assert code == 0
    assert calls[-1] == ("write_out", "out.star", {"convert": True})


def test_self_write_rejects_option_before_the_method_runs(monkeypatch):
    from cryocat.core.cryomotl import RelionMotl

    motl = _in_memory_relion(RelionMotl, version=5.1)
    code, calls = _self_write(motl, ["--output_convert", "true"], monkeypatch)
    assert code == 2
    assert calls == []


def test_converter_without_options_passes_no_kwargs():
    invocation = _invocation("cc_cryomotl", _EMMOTL2RELION)
    assert "load_kwargs" not in invocation.kwargs
    assert "write_kwargs" not in invocation.kwargs


@pytest.mark.parametrize("command, subcommand", MAP_OUTPUTS)
def test_map_output_is_required(command, subcommand):
    assert _action(_subcommands(command)[0][subcommand], "output_map").required


def test_map_outputs():
    assert set(MAP_OUTPUTS) == {
        ("cc_cryomap", name)
        for name in ("binarize", "normalize", "recenter", "normalize_under_mask", "pad", "symmetrize_volume")
    }
