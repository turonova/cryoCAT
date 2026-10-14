"""Tests for cryocat.cli."""
import argparse
import pytest

from cryocat.cli import add_params_from_signature


# ---------------------------------------------------------------------------
# add_params_from_signature
# ---------------------------------------------------------------------------

def _sample_function(output_path: str, scale: float = 1.0):
    """Sample function for CLI testing.

    Parameters
    ----------
    output_path : str
        Path to the output file.
    scale : float, default=1.0
        Scale factor applied to the data.
    """
    pass


def _build(fn=_sample_function, name="sample"):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    add_params_from_signature(subparsers, name, fn)
    return parser


def test_add_params_creates_subcommand():
    parser = _build()
    args = parser.parse_args(["sample", "--output_path", "out.mrc"])
    assert args.output_path == "out.mrc"


def test_add_params_optional_has_default():
    parser = _build()
    args = parser.parse_args(["sample", "--output_path", "out.mrc"])
    assert args.scale == pytest.approx(1.0)


def test_add_params_optional_overridable():
    parser = _build()
    args = parser.parse_args(["sample", "--output_path", "out.mrc", "--scale", "2.5"])
    assert args.scale == pytest.approx(2.5)


def test_add_params_typed_value_conversion():
    # The `float` annotation routes through TYPE_HANDLERS to argparse type=float,
    # so the parsed value is a real float, not a string.
    parser = _build()
    args = parser.parse_args(["sample", "--output_path", "out.mrc", "--scale", "3"])
    assert isinstance(args.scale, float)


def test_add_params_required_missing_exits():
    parser = _build()
    with pytest.raises(SystemExit):
        parser.parse_args(["sample"])


# ---------------------------------------------------------------------------
# CLI entry points (smoke tests)
# ---------------------------------------------------------------------------

def test_wedge_list_exits_without_args():
    with pytest.raises(SystemExit):
        from cryocat.cli import wedge_list
        wedge_list()


def test_tm_ana_exits_without_args():
    with pytest.raises(SystemExit):
        from cryocat.cli import tm_ana
        tm_ana()
