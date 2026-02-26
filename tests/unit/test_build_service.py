from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from app.services.build_service import (
    BuildError,
    build_llama_cpp,
    clone_or_update_llama_cpp,
    get_quantize_binary,
    setup_venv,
)

LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp.git"


# ---------------------------------------------------------------------------
# clone_or_update_llama_cpp — fresh clone (no .git dir)
# ---------------------------------------------------------------------------

@patch("subprocess.run")
def test_clone_when_dest_does_not_exist(mock_run, tmp_path):
    dest = str(tmp_path / "llama.cpp")
    mock_run.return_value = MagicMock(returncode=0)
    clone_or_update_llama_cpp(dest, LLAMA_CPP_URL)
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "git"
    assert "clone" in cmd


@patch("subprocess.run")
def test_clone_passes_url(mock_run, tmp_path):
    dest = str(tmp_path / "llama.cpp")
    mock_run.return_value = MagicMock(returncode=0)
    clone_or_update_llama_cpp(dest, LLAMA_CPP_URL)
    cmd = mock_run.call_args[0][0]
    assert LLAMA_CPP_URL in cmd


@patch("subprocess.run")
def test_clone_passes_dest(mock_run, tmp_path):
    dest = str(tmp_path / "llama.cpp")
    mock_run.return_value = MagicMock(returncode=0)
    clone_or_update_llama_cpp(dest, LLAMA_CPP_URL)
    cmd = mock_run.call_args[0][0]
    assert dest in cmd


@patch("subprocess.run")
def test_clone_nonzero_exit_raises_build_error(mock_run, tmp_path):
    dest = str(tmp_path / "llama.cpp")
    mock_run.return_value = MagicMock(returncode=1, stderr=b"fatal: repo not found")
    with pytest.raises(BuildError):
        clone_or_update_llama_cpp(dest, LLAMA_CPP_URL)


# ---------------------------------------------------------------------------
# clone_or_update_llama_cpp — existing repo (has .git)
# ---------------------------------------------------------------------------

@patch("subprocess.run")
def test_pulls_when_git_dir_exists(mock_run, tmp_path):
    (tmp_path / ".git").mkdir()
    mock_run.return_value = MagicMock(returncode=0)
    clone_or_update_llama_cpp(str(tmp_path), LLAMA_CPP_URL)
    cmd = mock_run.call_args[0][0]
    assert "pull" in cmd


@patch("subprocess.run")
def test_pull_does_not_clone_when_git_exists(mock_run, tmp_path):
    (tmp_path / ".git").mkdir()
    mock_run.return_value = MagicMock(returncode=0)
    clone_or_update_llama_cpp(str(tmp_path), LLAMA_CPP_URL)
    cmd = mock_run.call_args[0][0]
    assert "clone" not in cmd


@patch("subprocess.run")
def test_pull_nonzero_exit_raises_build_error(mock_run, tmp_path):
    (tmp_path / ".git").mkdir()
    mock_run.return_value = MagicMock(returncode=1, stderr=b"network error")
    with pytest.raises(BuildError):
        clone_or_update_llama_cpp(str(tmp_path), LLAMA_CPP_URL)


@patch("subprocess.run")
def test_pull_passes_dest_to_git_c(mock_run, tmp_path):
    (tmp_path / ".git").mkdir()
    mock_run.return_value = MagicMock(returncode=0)
    clone_or_update_llama_cpp(str(tmp_path), LLAMA_CPP_URL)
    cmd = mock_run.call_args[0][0]
    assert str(tmp_path) in cmd


# ---------------------------------------------------------------------------
# build_llama_cpp
# ---------------------------------------------------------------------------

@patch("subprocess.run")
def test_build_runs_cmake_configure(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    build_llama_cpp(str(tmp_path / "llama"), str(tmp_path / "build"))
    all_cmds = [c[0][0] for c in mock_run.call_args_list]
    assert any("cmake" in str(c) for c in all_cmds)


@patch("subprocess.run")
def test_build_calls_cmake_at_least_twice(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    build_llama_cpp(str(tmp_path / "llama"), str(tmp_path / "build"))
    assert mock_run.call_count >= 2


@patch("subprocess.run")
def test_build_passes_release_config(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    build_llama_cpp(str(tmp_path / "llama"), str(tmp_path / "build"))
    all_args = " ".join(
        str(arg)
        for c in mock_run.call_args_list
        for arg in c[0][0]
    )
    assert "Release" in all_args


@patch("subprocess.run")
def test_build_passes_build_dir(mock_run, tmp_path):
    build_dir = str(tmp_path / "build")
    mock_run.return_value = MagicMock(returncode=0)
    build_llama_cpp(str(tmp_path / "llama"), build_dir)
    all_args = " ".join(
        str(arg)
        for c in mock_run.call_args_list
        for arg in c[0][0]
    )
    assert build_dir in all_args


@patch("subprocess.run")
def test_build_configure_failure_raises_build_error(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=1, stderr=b"CMake error")
    with pytest.raises(BuildError):
        build_llama_cpp(str(tmp_path / "llama"), str(tmp_path / "build"))


@patch("subprocess.run")
def test_build_compile_failure_raises_build_error(mock_run, tmp_path):
    mock_run.side_effect = [
        MagicMock(returncode=0),               # configure succeeds
        MagicMock(returncode=1, stderr=b"compile error"),  # build fails
    ]
    with pytest.raises(BuildError):
        build_llama_cpp(str(tmp_path / "llama"), str(tmp_path / "build"))


@patch("subprocess.run")
def test_build_error_message_mentions_exit_code(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=2, stderr=b"bad")
    with pytest.raises(BuildError, match="2"):
        build_llama_cpp(str(tmp_path / "llama"), str(tmp_path / "build"))


# ---------------------------------------------------------------------------
# get_quantize_binary
# ---------------------------------------------------------------------------

def test_get_binary_finds_plain_binary(tmp_path):
    binary = tmp_path / "llama-quantize"
    binary.touch()
    result = get_quantize_binary(str(tmp_path))
    assert result == str(binary)


def test_get_binary_finds_exe_in_build_dir(tmp_path):
    binary = tmp_path / "llama-quantize.exe"
    binary.touch()
    result = get_quantize_binary(str(tmp_path))
    assert "llama-quantize" in result


def test_get_binary_finds_windows_release_subdir(tmp_path):
    release = tmp_path / "Release"
    release.mkdir()
    binary = release / "llama-quantize.exe"
    binary.touch()
    result = get_quantize_binary(str(tmp_path))
    assert "llama-quantize" in result


def test_get_binary_release_subdir_takes_priority_over_plain(tmp_path):
    # Release/llama-quantize.exe should be found before llama-quantize
    release = tmp_path / "Release"
    release.mkdir()
    win_binary = release / "llama-quantize.exe"
    win_binary.touch()
    plain = tmp_path / "llama-quantize"
    plain.touch()
    result = get_quantize_binary(str(tmp_path))
    assert "Release" in result


def test_get_binary_finds_bin_release_subdir(tmp_path):
    """Windows CMake puts the exe in bin/Release/ inside the build dir."""
    bin_release = tmp_path / "bin" / "Release"
    bin_release.mkdir(parents=True)
    binary = bin_release / "llama-quantize.exe"
    binary.touch()
    result = get_quantize_binary(str(tmp_path))
    assert "llama-quantize" in result


def test_get_binary_raises_build_error_when_nothing_found(tmp_path):
    with pytest.raises(BuildError):
        get_quantize_binary(str(tmp_path))


def test_get_binary_error_mentions_build_dir(tmp_path):
    with pytest.raises(BuildError, match=str(tmp_path).replace("\\", "\\\\")):
        get_quantize_binary(str(tmp_path))


def test_get_binary_result_is_string(tmp_path):
    binary = tmp_path / "llama-quantize"
    binary.touch()
    result = get_quantize_binary(str(tmp_path))
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# setup_venv
# ---------------------------------------------------------------------------

@patch("subprocess.run")
def test_setup_venv_creates_venv(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    setup_venv(str(tmp_path / "llama"), str(tmp_path / "venv"))
    first_cmd = mock_run.call_args_list[0][0][0]
    assert "venv" in " ".join(str(a) for a in first_cmd)


@patch("subprocess.run")
def test_setup_venv_installs_requirements_when_file_exists(mock_run, tmp_path):
    llama_dir = tmp_path / "llama"
    llama_dir.mkdir()
    (llama_dir / "requirements.txt").touch()
    mock_run.return_value = MagicMock(returncode=0)
    setup_venv(str(llama_dir), str(tmp_path / "venv"))
    all_args = " ".join(
        str(arg)
        for c in mock_run.call_args_list
        for arg in c[0][0]
    )
    assert "requirements" in all_args


@patch("subprocess.run")
def test_setup_venv_skips_requirements_when_file_absent(mock_run, tmp_path):
    llama_dir = tmp_path / "llama"
    llama_dir.mkdir()
    # No requirements.txt present
    mock_run.return_value = MagicMock(returncode=0)
    setup_venv(str(llama_dir), str(tmp_path / "venv"))
    all_args = " ".join(
        str(arg)
        for c in mock_run.call_args_list
        for arg in c[0][0]
    )
    assert "requirements" not in all_args


@patch("subprocess.run")
def test_setup_venv_installs_huggingface_hub(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    setup_venv(str(tmp_path / "llama"), str(tmp_path / "venv"))
    all_args = " ".join(
        str(arg)
        for c in mock_run.call_args_list
        for arg in c[0][0]
    )
    assert "huggingface_hub" in all_args


@patch("subprocess.run")
def test_setup_venv_returns_python_path(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    result = setup_venv(str(tmp_path / "llama"), str(tmp_path / "venv"))
    assert "python" in result.lower()


@patch("subprocess.run")
def test_setup_venv_python_path_is_inside_venv(mock_run, tmp_path):
    venv_dir = str(tmp_path / "venv")
    mock_run.return_value = MagicMock(returncode=0)
    result = setup_venv(str(tmp_path / "llama"), venv_dir)
    assert result.startswith(venv_dir)


@patch("subprocess.run")
def test_setup_venv_failure_raises_build_error(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=1, stderr=b"venv creation failed")
    with pytest.raises(BuildError):
        setup_venv(str(tmp_path / "llama"), str(tmp_path / "venv"))


# ---------------------------------------------------------------------------
# BuildError
# ---------------------------------------------------------------------------

def test_build_error_is_exception():
    assert issubclass(BuildError, Exception)


def test_build_error_carries_message():
    err = BuildError("cmake exploded")
    assert "cmake exploded" in str(err)
