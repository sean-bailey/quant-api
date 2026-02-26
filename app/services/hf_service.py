from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InvalidCredentialsError(Exception):
    """Raised when HuggingFace credentials are rejected."""


class DownloadError(Exception):
    """Raised when cloning a source model from HuggingFace fails."""


class UploadError(Exception):
    """Raised when creating a repo or uploading a file to HuggingFace fails."""


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def validate_credentials(token: str) -> bool:
    """Verify that *token* is accepted by HuggingFace.

    Returns True on success; raises InvalidCredentialsError on any failure.
    """
    try:
        HfApi().whoami(token=token)
        return True
    except Exception as e:
        raise InvalidCredentialsError(str(e)) from e


def create_output_repo(source_repo: str, username: str, token: str) -> str:
    """Create (or ensure existence of) the destination GGUF repo.

    The repo is named ``{username}/{ModelName}-GGUF`` where *ModelName* is
    the name portion of *source_repo* (without the org prefix).

    Returns the full ``owner/repo`` repo ID string.
    """
    model_name = source_repo.split("/", 1)[1]
    repo_id = f"{username}/{model_name}-GGUF"
    try:
        HfApi().create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            exist_ok=True,
            token=token,
        )
    except Exception as e:
        raise UploadError(f"Failed to create repo {repo_id}: {e}") from e
    return repo_id


def clone_model(repo: str, dest_dir: Path, token: str) -> Path:
    """Download all model files from *repo* into ``dest_dir/<ModelName>``.

    Uses ``huggingface_hub.snapshot_download`` so no git/LFS tooling is
    required on the host machine.

    Returns the Path to the downloaded model directory.
    """
    model_name = repo.split("/", 1)[1]
    local_dir = dest_dir / model_name
    try:
        result = snapshot_download(
            repo_id=repo,
            local_dir=str(local_dir),
            token=token,
        )
    except Exception as e:
        raise DownloadError(f"Failed to download {repo}: {e}") from e
    return Path(result)


def upload_file(
    local_path: Path,
    repo_id: str,
    path_in_repo: str,
    token: str,
) -> None:
    """Upload a single file to a HuggingFace model repository.

    Raises UploadError on any failure.
    """
    try:
        HfApi().upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
    except Exception as e:
        raise UploadError(
            f"Failed to upload {local_path.name} to {repo_id}: {e}"
        ) from e
