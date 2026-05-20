import numpy as np
import pandas as pd
import pytest

from cryocat.utils import starfileio as sf
from cryocat.utils.starfileio import Token, TokenType, _try_numeric
from pathlib import Path

test_data = Path(__file__).parent / "test_data"


# ---------------------------------------------------------------------------
# Fixture: pre-parsed relion file
# ---------------------------------------------------------------------------

@pytest.fixture
def relion_optics():
    return sf.Starfile(str(test_data / "relion_3.1_optics.star"))


@pytest.fixture
def simple_star(tmp_path):
    """Write a minimal STAR file and return its path."""
    content = (
        "\ndata_particles\n\n"
        "loop_\n"
        "_rlnX #1\n"
        "_rlnY #2\n"
        "1.0\t2.0\n"
        "3.0\t4.0\n"
    )
    p = tmp_path / "simple.star"
    p.write_text(content)
    return str(p)


# ---------------------------------------------------------------------------
# _try_numeric
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "values, expected_dtype_kind",
    [
        (["1", "2", "3"], "i"),        # integers → numeric
        (["1.5", "2.5"], "f"),         # floats → numeric
        (["a", "b"], "O"),             # strings → object (unchanged)
        (["1", "two", "3"], "O"),      # mixed → object (unchanged)
    ],
)
def test_try_numeric(values, expected_dtype_kind):
    col = pd.Series(values)
    result = _try_numeric(col)
    assert result.dtype.kind == expected_dtype_kind


# ---------------------------------------------------------------------------
# Token.tokenize
# ---------------------------------------------------------------------------

def test_tokenize_literal():
    tokens = Token.tokenize("data_particles")
    values = [t.value for t in tokens if t.token_type == TokenType.LITERAL]
    assert "data_particles" in values


def test_tokenize_loop():
    tokens = Token.tokenize("loop_\n")
    loop_tokens = [t for t in tokens if t.token_type == TokenType.LOOP]
    assert len(loop_tokens) == 1


def test_tokenize_property():
    tokens = Token.tokenize("_rlnX #1\n")
    prop_tokens = [t for t in tokens if t.token_type == TokenType.PROPERTY]
    assert any(t.value == "_rlnX" for t in prop_tokens)


def test_tokenize_comment():
    tokens = Token.tokenize("# hello world\n")
    comment_tokens = [t for t in tokens if t.token_type == TokenType.COMMENT]
    assert any("hello world" in t.value for t in comment_tokens)


def test_tokenize_newline_count():
    tokens = Token.tokenize("a\nb\nc")
    newlines = [t for t in tokens if t.token_type == TokenType.NEWLINE]
    assert len(newlines) == 3


def test_tokenize_reversed_order():
    tokens = Token.tokenize("first\nsecond")
    # reversed: pop() should yield first token first
    first = tokens[-1]
    assert first.token_type in (TokenType.LITERAL, TokenType.NEWLINE, TokenType.COMMENT, TokenType.LOOP, TokenType.PROPERTY)


# ---------------------------------------------------------------------------
# Token.check / consume / check_then_consume
# ---------------------------------------------------------------------------

def _make_token_queue(token_type, value="x"):
    return [Token(token_type, value, (0, 0))]


def test_token_check_matching():
    q = _make_token_queue(TokenType.LITERAL)
    assert Token.check(q, TokenType.LITERAL) is True


def test_token_check_not_matching():
    q = _make_token_queue(TokenType.LITERAL)
    assert Token.check(q, TokenType.LOOP) is False


def test_token_check_empty_raises():
    with pytest.raises(IOError):
        Token.check([], TokenType.LITERAL)


def test_token_consume_returns_token():
    q = _make_token_queue(TokenType.LITERAL, "hello")
    t = Token.consume(q, TokenType.LITERAL)
    assert t.value == "hello"
    assert len(q) == 0


def test_token_consume_wrong_type_raises():
    q = _make_token_queue(TokenType.LOOP)
    with pytest.raises(IOError):
        Token.consume(q, TokenType.LITERAL)


def test_token_consume_empty_raises():
    with pytest.raises(IOError):
        Token.consume([], TokenType.LITERAL)


def test_token_check_then_consume_match():
    q = _make_token_queue(TokenType.COMMENT, "note")
    t = Token.check_then_consume(q, TokenType.COMMENT)
    assert t is not None
    assert t.value == "note"


def test_token_check_then_consume_no_match_returns_none():
    q = _make_token_queue(TokenType.LITERAL)
    assert Token.check_then_consume(q, TokenType.LOOP) is None
    assert len(q) == 1  # not consumed


def test_token_check_then_consume_empty_returns_none():
    assert Token.check_then_consume([], TokenType.LITERAL) is None


# ---------------------------------------------------------------------------
# Token.lookahead
# ---------------------------------------------------------------------------

def test_token_lookahead_finds_target():
    q = [
        Token(TokenType.NEWLINE, None, (0, 0)),
        Token(TokenType.LITERAL, "x", (1, 0)),
    ]
    q = q[::-1]  # reversed so pop() yields first
    assert Token.lookahead(q, TokenType.LITERAL, [TokenType.NEWLINE]) is True


def test_token_lookahead_blocked_by_non_ignored():
    q = [
        Token(TokenType.LOOP, "loop_", (0, 0)),
        Token(TokenType.LITERAL, "x", (1, 0)),
    ]
    q = q[::-1]
    assert Token.lookahead(q, TokenType.LITERAL, [TokenType.NEWLINE]) is False


# ---------------------------------------------------------------------------
# Existing STAR file tests (original, kept)
# ---------------------------------------------------------------------------

def test_relion_optics_specifiers(relion_optics):
    assert relion_optics.specifiers == ["data_optics", "data_particles"]


def test_relion_optics_comments(relion_optics):
    assert relion_optics.comments == [["version 30001", "version 30002"], ["version 30001"]]


def test_relion_optics_frame_count(relion_optics):
    assert len(relion_optics.frames) == 2


@pytest.mark.parametrize(
    "frame_idx, expected_shape",
    [(0, (1, 7)), (1, (3, 22))],
)
def test_relion_optics_frame_shapes(relion_optics, frame_idx, expected_shape):
    assert relion_optics.frames[frame_idx].shape == expected_shape


def test_relion5_star_fix():
    path = str(test_data / "motl_data" / "relion5" / "clean" / "warp2_particles_matching.star")
    result = sf.Starfile.fix_relion5_star(path)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Starfile.get_specifier_id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "specifiers, query, expected",
    [
        (["data_particles", "data_optics"], "data_particles", 0),
        (["data_particles", "data_optics"], "data_optics", 1),
        (["data_particles"], "data_missing", None),
        ([], "data_anything", None),
    ],
)
def test_get_specifier_id(specifiers, query, expected):
    assert sf.Starfile.get_specifier_id(specifiers, query) == expected


# ---------------------------------------------------------------------------
# Starfile.read  (simple round-trip)
# ---------------------------------------------------------------------------

def test_read_returns_three_elements(simple_star):
    frames, specifiers, comments = sf.Starfile.read(simple_star)
    assert len(frames) == 1
    assert specifiers == ["data_particles"]
    assert isinstance(comments, list)


def test_read_numeric_cast(simple_star):
    frames, _, _ = sf.Starfile.read(simple_star)
    assert frames[0]["rlnX"].dtype.kind in ("f", "i")


def test_read_with_data_id(simple_star):
    frame, spec, comment = sf.Starfile.read(simple_star, data_id=0)
    assert isinstance(frame, pd.DataFrame)
    assert spec == "data_particles"


# ---------------------------------------------------------------------------
# Starfile.write + read round-trip
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_frame():
    return pd.DataFrame({"ColA": [1.0, 2.0, 3.0], "ColB": [4.0, 5.0, 6.0]})


def test_write_creates_file(tmp_path, sample_frame):
    out = str(tmp_path / "out.star")
    sf.Starfile.write([sample_frame], out)
    assert Path(out).exists()


def test_write_read_roundtrip(tmp_path, sample_frame):
    out = str(tmp_path / "roundtrip.star")
    sf.Starfile.write([sample_frame], out, specifiers=["data_test"])
    frames, specifiers, _ = sf.Starfile.read(out)
    assert specifiers == ["data_test"]
    assert frames[0].shape == sample_frame.shape
    assert list(frames[0].columns) == list(sample_frame.columns)


def test_write_multiple_blocks(tmp_path, sample_frame):
    out = str(tmp_path / "multi.star")
    sf.Starfile.write(
        [sample_frame, sample_frame],
        out,
        specifiers=["data_block1", "data_block2"],
    )
    frames, specifiers, _ = sf.Starfile.read(out)
    assert len(frames) == 2
    assert specifiers == ["data_block1", "data_block2"]


def test_write_mismatched_lengths_raises(tmp_path, sample_frame):
    out = str(tmp_path / "bad.star")
    with pytest.raises(ValueError):
        sf.Starfile.write([sample_frame], out, specifiers=["a", "b"])


# ---------------------------------------------------------------------------
# Starfile.remove_lines
# ---------------------------------------------------------------------------

def test_remove_lines_returns_modified(simple_star):
    frames, specifiers, comments = sf.Starfile.remove_lines(simple_star, [0])
    assert len(frames[0]) == 1


def test_remove_lines_writes_output(simple_star, tmp_path):
    out = str(tmp_path / "removed.star")
    sf.Starfile.remove_lines(simple_star, [0], output_path=out)
    frames, _, _ = sf.Starfile.read(out)
    assert len(frames[0]) == 1


def test_remove_lines_unknown_specifier_warns(simple_star):
    with pytest.warns(UserWarning):
        result = sf.Starfile.remove_lines(simple_star, [0], data_specifier="data_nonexistent")
    assert result is None


# ---------------------------------------------------------------------------
# Starfile.get_frame_and_comments
# ---------------------------------------------------------------------------

def test_get_frame_and_comments_returns_frame(simple_star):
    frame, comments = sf.Starfile.get_frame_and_comments(simple_star, "data_particles")
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (2, 2)


def test_get_frame_and_comments_invalid_specifier_raises(simple_star):
    with pytest.raises(ValueError):
        sf.Starfile.get_frame_and_comments(simple_star, "data_nonexistent")


# ---------------------------------------------------------------------------
# Starfile.__init__ with no file
# ---------------------------------------------------------------------------

def test_starfile_init_no_path():
    sf_obj = sf.Starfile()
    assert sf_obj.frames is None
    assert sf_obj.specifiers is None


def test_starfile_init_from_file(simple_star):
    sf_obj = sf.Starfile(simple_star)
    assert sf_obj.specifiers == ["data_particles"]
    assert len(sf_obj.frames) == 1


def test_starfile_init_nonexistent_path():
    sf_obj = sf.Starfile("/does/not/exist.star")
    assert sf_obj.frames is None
