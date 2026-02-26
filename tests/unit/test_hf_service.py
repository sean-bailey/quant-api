from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.hf_service import (
    DownloadError,
    InvalidCredentialsError,
    UploadError,
    clone_model,
    create_output_repo,
    upload_file,
    validate_credentials,
)


# ---------------------------------------------------------------------------
# validate_credentials
# ---------------------------------------------------------------------------

@patch("app.services.hf_service.HfApi")
def test_validate_credentials_returns_true_on_success(mock_cls):
    mock_cls.return_value.whoami.return_value = {"name": "testuser"}
    assert validate_credentials("valid_token") is True


@patch("app.services.hf_service.HfApi")
def test_validate_credentials_passes_token_to_whoami(mock_cls):
    mock_cls.return_value.whoami.return_value = {"name": "testuser"}
    validate_credentials("mytoken")
    mock_cls.return_value.whoami.assert_called_once_with(token="mytoken")


@patch("app.services.hf_service.HfApi")
def test_validate_credentials_raises_invalid_credentials_on_error(mock_cls):
    mock_cls.return_value.whoami.side_effect = Exception("401 Unauthorized")
    with pytest.raises(InvalidCredentialsError):
        validate_credentials("bad_token")


@patch("app.services.hf_service.HfApi")
def test_validate_credentials_error_message_is_propagated(mock_cls):
    mock_cls.return_value.whoami.side_effect = Exception("Token is invalid")
    with pytest.raises(InvalidCredentialsError, match="Token is invalid"):
        validate_credentials("bad_token")


@patch("app.services.hf_service.HfApi")
def test_validate_credentials_creates_fresh_api_instance(mock_cls):
    mock_cls.return_value.whoami.return_value = {"name": "u"}
    validate_credentials("tok")
    mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# create_output_repo
# ---------------------------------------------------------------------------

@patch("app.services.hf_service.HfApi")
def test_create_output_repo_calls_create_repo_with_correct_id(mock_cls):
    create_output_repo("meta-llama/Llama-3.2-1B-Instruct", "testuser", "tok")
    mock_cls.return_value.create_repo.assert_called_once_with(
        repo_id="testuser/Llama-3.2-1B-Instruct-GGUF",
        repo_type="model",
        private=False,
        exist_ok=True,
        token="tok",
    )


@patch("app.services.hf_service.HfApi")
def test_create_output_repo_returns_repo_id(mock_cls):
    result = create_output_repo("org/ModelName", "alice", "tok")
    assert result == "alice/ModelName-GGUF"


@patch("app.services.hf_service.HfApi")
def test_create_output_repo_strips_source_org(mock_cls):
    result = create_output_repo("some-org/SomeModel", "myuser", "tok")
    assert "some-org" not in result
    assert result.startswith("myuser/")


@patch("app.services.hf_service.HfApi")
def test_create_output_repo_uses_exist_ok(mock_cls):
    create_output_repo("a/b", "user", "tok")
    call_kwargs = mock_cls.return_value.create_repo.call_args.kwargs
    assert call_kwargs["exist_ok"] is True


@patch("app.services.hf_service.HfApi")
def test_create_output_repo_sets_model_type(mock_cls):
    create_output_repo("a/b", "user", "tok")
    call_kwargs = mock_cls.return_value.create_repo.call_args.kwargs
    assert call_kwargs["repo_type"] == "model"


@patch("app.services.hf_service.HfApi")
def test_create_output_repo_error_raises_upload_error(mock_cls):
    mock_cls.return_value.create_repo.side_effect = Exception("network error")
    with pytest.raises(UploadError):
        create_output_repo("a/b", "user", "tok")


# ---------------------------------------------------------------------------
# clone_model
# ---------------------------------------------------------------------------

@patch("app.services.hf_service.snapshot_download")
def test_clone_model_calls_snapshot_download(mock_dl, tmp_path):
    mock_dl.return_value = str(tmp_path / "ModelA")
    clone_model("org/ModelA", tmp_path, "tok")
    mock_dl.assert_called_once()


@patch("app.services.hf_service.snapshot_download")
def test_clone_model_passes_repo_id(mock_dl, tmp_path):
    mock_dl.return_value = str(tmp_path)
    clone_model("org/ModelA", tmp_path, "tok")
    assert mock_dl.call_args.kwargs["repo_id"] == "org/ModelA"


@patch("app.services.hf_service.snapshot_download")
def test_clone_model_passes_token(mock_dl, tmp_path):
    mock_dl.return_value = str(tmp_path)
    clone_model("org/ModelA", tmp_path, "mytoken")
    assert mock_dl.call_args.kwargs["token"] == "mytoken"


@patch("app.services.hf_service.snapshot_download")
def test_clone_model_local_dir_uses_model_name_not_org(mock_dl, tmp_path):
    mock_dl.return_value = str(tmp_path / "ModelA")
    clone_model("org/ModelA", tmp_path, "tok")
    local_dir = mock_dl.call_args.kwargs["local_dir"]
    assert "ModelA" in local_dir
    assert "org" not in Path(local_dir).name


@patch("app.services.hf_service.snapshot_download")
def test_clone_model_returns_path_object(mock_dl, tmp_path):
    target = tmp_path / "ModelA"
    target.mkdir()
    mock_dl.return_value = str(target)
    result = clone_model("org/ModelA", tmp_path, "tok")
    assert isinstance(result, Path)
    assert result == target


@patch("app.services.hf_service.snapshot_download")
def test_clone_model_error_raises_download_error(mock_dl, tmp_path):
    mock_dl.side_effect = Exception("connection refused")
    with pytest.raises(DownloadError):
        clone_model("org/ModelA", tmp_path, "tok")


@patch("app.services.hf_service.snapshot_download")
def test_clone_model_error_message_contains_repo(mock_dl, tmp_path):
    mock_dl.side_effect = Exception("connection refused")
    with pytest.raises(DownloadError, match="org/ModelA"):
        clone_model("org/ModelA", tmp_path, "tok")


# ---------------------------------------------------------------------------
# upload_file
# ---------------------------------------------------------------------------

@patch("app.services.hf_service.HfApi")
def test_upload_file_calls_hf_api_upload(mock_cls, tmp_path):
    f = tmp_path / "model-Q4_K_M.gguf"
    f.touch()
    upload_file(f, "user/model-GGUF", "model-Q4_K_M.gguf", "tok")
    mock_cls.return_value.upload_file.assert_called_once()


@patch("app.services.hf_service.HfApi")
def test_upload_file_passes_path_in_repo(mock_cls, tmp_path):
    f = tmp_path / "model-Q4_K_M.gguf"
    f.touch()
    upload_file(f, "user/model-GGUF", "model-Q4_K_M.gguf", "tok")
    call_kwargs = mock_cls.return_value.upload_file.call_args.kwargs
    assert call_kwargs["path_in_repo"] == "model-Q4_K_M.gguf"


@patch("app.services.hf_service.HfApi")
def test_upload_file_passes_repo_id(mock_cls, tmp_path):
    f = tmp_path / "file.gguf"
    f.touch()
    upload_file(f, "user/model-GGUF", "file.gguf", "tok")
    call_kwargs = mock_cls.return_value.upload_file.call_args.kwargs
    assert call_kwargs["repo_id"] == "user/model-GGUF"


@patch("app.services.hf_service.HfApi")
def test_upload_file_passes_token(mock_cls, tmp_path):
    f = tmp_path / "file.gguf"
    f.touch()
    upload_file(f, "user/model-GGUF", "file.gguf", "secrettoken")
    call_kwargs = mock_cls.return_value.upload_file.call_args.kwargs
    assert call_kwargs["token"] == "secrettoken"


@patch("app.services.hf_service.HfApi")
def test_upload_file_sets_model_repo_type(mock_cls, tmp_path):
    f = tmp_path / "file.gguf"
    f.touch()
    upload_file(f, "user/model-GGUF", "file.gguf", "tok")
    call_kwargs = mock_cls.return_value.upload_file.call_args.kwargs
    assert call_kwargs["repo_type"] == "model"


@patch("app.services.hf_service.HfApi")
def test_upload_file_error_raises_upload_error(mock_cls, tmp_path):
    f = tmp_path / "file.gguf"
    f.touch()
    mock_cls.return_value.upload_file.side_effect = Exception("network timeout")
    with pytest.raises(UploadError):
        upload_file(f, "user/model-GGUF", "file.gguf", "tok")


@patch("app.services.hf_service.HfApi")
def test_upload_file_error_message_contains_filename(mock_cls, tmp_path):
    f = tmp_path / "model-Q4_K_M.gguf"
    f.touch()
    mock_cls.return_value.upload_file.side_effect = Exception("timeout")
    with pytest.raises(UploadError, match="model-Q4_K_M.gguf"):
        upload_file(f, "user/model-GGUF", "model-Q4_K_M.gguf", "tok")


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

def test_invalid_credentials_error_is_exception():
    assert issubclass(InvalidCredentialsError, Exception)


def test_download_error_is_exception():
    assert issubclass(DownloadError, Exception)


def test_upload_error_is_exception():
    assert issubclass(UploadError, Exception)
