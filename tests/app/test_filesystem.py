"""Z8 — Tests for cryocat.app.components.filesystem (pure layer).

All tests run against tmp_path so no real filesystem side-effects occur.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from cryocat.app.components.filesystem import (
    Entry,
    breadcrumbs,
    list_dir,
    resolve_input,
    validate,
)


# ── list_dir ─────────────────────────────────────────────────────────────────

def test_dirs_before_files(tmp_path):
    (tmp_path / "z_file.em").touch()
    (tmp_path / "a_dir").mkdir()
    entries, err = list_dir(tmp_path)
    assert err is None
    assert entries[0].is_dir
    assert not entries[1].is_dir


def test_sorted_case_insensitively(tmp_path):
    (tmp_path / "Banana.em").touch()
    (tmp_path / "apple.em").touch()
    (tmp_path / "Cherry.em").touch()
    entries, _ = list_dir(tmp_path)
    names = [e.name.lower() for e in entries]
    assert names == sorted(names)


def test_extension_filter_with_dot(tmp_path):
    (tmp_path / "data.em").touch()
    (tmp_path / "other.star").touch()
    entries, _ = list_dir(tmp_path, extensions=(".em",))
    assert all(e.name.endswith(".em") or e.is_dir for e in entries)
    names = [e.name for e in entries]
    assert "data.em" in names
    assert "other.star" not in names


def test_extension_filter_without_dot(tmp_path):
    """Extensions given without leading dot are normalised."""
    (tmp_path / "data.em").touch()
    (tmp_path / "other.star").touch()
    entries, _ = list_dir(tmp_path, extensions=("em",))
    names = [e.name for e in entries]
    assert "data.em" in names
    assert "other.star" not in names


def test_extension_filter_case_insensitive(tmp_path):
    (tmp_path / "DATA.EM").touch()
    entries, _ = list_dir(tmp_path, extensions=(".em",))
    assert any(e.name == "DATA.EM" for e in entries)


def test_extension_filter_does_not_remove_dirs(tmp_path):
    (tmp_path / "subdir").mkdir()
    (tmp_path / "ignored.txt").touch()
    entries, _ = list_dir(tmp_path, extensions=(".em",))
    dir_entries = [e for e in entries if e.is_dir]
    assert any(e.name == "subdir" for e in dir_entries)


def test_missing_directory_returns_error(tmp_path):
    entries, err = list_dir(tmp_path / "no_such_dir")
    assert err is not None
    assert entries == []


@pytest.mark.skipif(sys.platform == "win32", reason="chmod 000 not meaningful on Windows")
def test_unreadable_directory_returns_error(tmp_path):
    if os.getuid() == 0:
        pytest.skip("Running as root — PermissionError won't fire")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        entries, err = list_dir(locked)
        assert err is not None
        assert entries == []
    finally:
        locked.chmod(0o755)


def test_hidden_files_excluded_by_default(tmp_path):
    (tmp_path / ".hidden").touch()
    (tmp_path / "visible.em").touch()
    entries, _ = list_dir(tmp_path)
    names = [e.name for e in entries]
    assert ".hidden" not in names
    assert "visible.em" in names


def test_hidden_files_shown_when_requested(tmp_path):
    (tmp_path / ".hidden").touch()
    entries, _ = list_dir(tmp_path, show_hidden=True)
    names = [e.name for e in entries]
    assert ".hidden" in names


# ── breadcrumbs ──────────────────────────────────────────────────────────────

def test_breadcrumbs_root_is_one_entry():
    root = Path("/").resolve()
    crumbs = breadcrumbs(str(root))
    assert len(crumbs) >= 1
    # Must not loop — the root entry's path equals str(root)
    assert crumbs[0][1] == str(root)


def test_breadcrumbs_path_sequence(tmp_path):
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    crumbs = breadcrumbs(str(sub))
    paths = [c[1] for c in crumbs]
    # Must end at the resolved sub path
    assert paths[-1] == str(sub.resolve())
    # Must be strictly ascending prefixes
    for i in range(len(paths) - 1):
        assert paths[i] in paths[i + 1] or paths[i + 1].startswith(paths[i])


# ── resolve_input ─────────────────────────────────────────────────────────────

def test_resolve_home(tmp_path):
    resolved, err = resolve_input("~")
    assert err is None
    assert resolved == str(Path.home().resolve())


def test_resolve_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_DIR_CRYOCAT", str(tmp_path))
    resolved, err = resolve_input("$TEST_DIR_CRYOCAT")
    assert err is None
    assert resolved == str(tmp_path.resolve())


def test_resolve_relative_becomes_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved, err = resolve_input(".")
    assert err is None
    assert Path(resolved).is_absolute()


def test_resolve_empty_returns_error():
    _, err = resolve_input("")
    assert err is not None


def test_resolve_whitespace_only():
    _, err = resolve_input("   ")
    assert err is not None


# ── validate ─────────────────────────────────────────────────────────────────

def test_validate_open_existing_file(tmp_path):
    f = tmp_path / "data.em"
    f.touch()
    assert validate(str(f), mode="open") is None


def test_validate_open_missing_file(tmp_path):
    assert validate(str(tmp_path / "missing.em"), mode="open") is not None


def test_validate_open_directory_rejected(tmp_path):
    assert validate(str(tmp_path), mode="open") is not None


def test_validate_open_extension_check(tmp_path):
    f = tmp_path / "data.star"
    f.touch()
    msg = validate(str(f), mode="open", extensions=(".em",))
    assert msg is not None


def test_validate_directory_existing(tmp_path):
    assert validate(str(tmp_path), mode="directory") is None


def test_validate_directory_file_rejected(tmp_path):
    f = tmp_path / "file.em"
    f.touch()
    assert validate(str(f), mode="directory") is not None


def test_validate_directory_missing(tmp_path):
    assert validate(str(tmp_path / "no_such"), mode="directory") is not None


def test_validate_save_existing_file_accepted(tmp_path):
    """save mode warns but accepts an existing file (returns None)."""
    f = tmp_path / "output.em"
    f.touch()
    assert validate(str(f), mode="save") is None


def test_validate_save_new_path_accepted(tmp_path):
    assert validate(str(tmp_path / "new_output.em"), mode="save") is None


def test_validate_save_missing_parent(tmp_path):
    msg = validate(str(tmp_path / "no_such" / "output.em"), mode="save")
    assert msg is not None


# ── Symlink ───────────────────────────────────────────────────────────────────

@pytest.mark.skipif(sys.platform == "win32", reason="symlinks require elevated rights on Windows")
def test_symlinked_directory_is_enterable(tmp_path):
    real = tmp_path / "real_dir"
    real.mkdir()
    (real / "file.em").touch()
    link = tmp_path / "link_dir"
    link.symlink_to(real)

    entries, err = list_dir(link)
    assert err is None
    assert any(e.name == "file.em" for e in entries)

    # Displayed path is stable (resolves through symlink)
    link_crumbs = breadcrumbs(str(link))
    assert link_crumbs[-1][1] == str(link.resolve())


# ── Component-level: get_path_field registers no callbacks ────────────────────

def test_get_path_field_no_callbacks():
    import dash
    from cryocat.app.components.pathfield import get_path_field

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    before = len(app._callback_list)
    get_path_field("test-prefix", mode="open", kind="motl")
    after = len(app._callback_list)
    assert after == before, "get_path_field must register no callbacks"
