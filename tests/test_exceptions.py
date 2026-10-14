"""Tests for cryocat.utils.exceptions."""
import pytest

from cryocat.utils.exceptions import MotlException, ProcessError, UserInputError


# ---------------------------------------------------------------------------
# MotlException
# ---------------------------------------------------------------------------

def test_motl_exception_message_stored():
    exc = MotlException("test message")
    assert exc.message == "test message"


def test_motl_exception_str_with_message():
    assert str(MotlException("custom")) == "custom"


def test_motl_exception_str_no_message():
    assert "Unspecified error" in str(MotlException())


def test_motl_exception_no_message_attribute_is_none():
    assert MotlException().message is None


def test_motl_exception_is_exception_subclass():
    assert issubclass(MotlException, Exception)


def test_motl_exception_can_be_raised_and_caught():
    with pytest.raises(MotlException):
        raise MotlException("boom")


# ---------------------------------------------------------------------------
# UserInputError
# ---------------------------------------------------------------------------

def test_user_input_error_inherits_motl():
    assert issubclass(UserInputError, MotlException)


def test_user_input_error_str_with_message():
    assert str(UserInputError("bad value")) == "bad value"


def test_user_input_error_str_no_message():
    assert "Incorrect input" in str(UserInputError())


def test_user_input_error_can_be_raised_and_caught():
    with pytest.raises(UserInputError):
        raise UserInputError("invalid")


def test_user_input_error_caught_as_motl():
    with pytest.raises(MotlException):
        raise UserInputError("invalid")


def test_user_input_error_caught_as_exception():
    with pytest.raises(Exception):
        raise UserInputError("invalid")


# ---------------------------------------------------------------------------
# ProcessError
# ---------------------------------------------------------------------------

def test_process_error_inherits_motl():
    assert issubclass(ProcessError, MotlException)


def test_process_error_str_with_message():
    assert str(ProcessError("step failed")) == "step failed"


def test_process_error_str_no_message():
    assert "Failed to finish" in str(ProcessError())


def test_process_error_can_be_raised_and_caught():
    with pytest.raises(ProcessError):
        raise ProcessError("pipeline error")


def test_process_error_caught_as_motl():
    with pytest.raises(MotlException):
        raise ProcessError("pipeline error")


def test_process_error_caught_as_exception():
    with pytest.raises(Exception):
        raise ProcessError("pipeline error")


# ---------------------------------------------------------------------------
# Parametrized cross-type checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls, default_fragment", [
    (MotlException, "Unspecified error"),
    (UserInputError, "Incorrect input"),
    (ProcessError, "Failed to finish"),
])
def test_default_message_content(cls, default_fragment):
    assert default_fragment in str(cls())


@pytest.mark.parametrize("cls", [MotlException, UserInputError, ProcessError])
def test_custom_message_passthrough(cls):
    msg = f"{cls.__name__} custom message"
    assert str(cls(msg)) == msg


@pytest.mark.parametrize("cls", [MotlException, UserInputError, ProcessError])
def test_message_attribute_stored(cls):
    exc = cls("hello")
    assert exc.message == "hello"
