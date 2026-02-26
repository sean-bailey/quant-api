"""End-to-end test for the full quantize-and-upload pipeline.

Requires real HuggingFace credentials supplied as environment variables:

    HF_TOKEN       — a write-access token for the target account
    HF_USERNAME    — the username that owns the token

Run with (PowerShell):

    $env:HF_TOKEN="hf_xxx"; $env:HF_USERNAME="yourname"; py -m pytest -m e2e -v

Skip e2e in normal runs:

    py -m pytest -m "not e2e"

Skipped automatically when HF_TOKEN is not set so that the suite remains
clean in CI environments that have not been provisioned with real credentials.

The test uses ``HuggingFaceTB/SmolLM2-135M`` — a 135 M-parameter
LlamaForCausalLM model that converts cleanly — and requests only the
``Q4_K_M`` quantisation type to minimise wall-clock time.

After the test completes (pass or fail) it attempts to delete the output
repository from HuggingFace so the account is not left cluttered.
"""

import os
import time

import pytest
from fastapi.testclient import TestClient
from huggingface_hub import HfApi

from app.dependencies import get_settings, get_store
from app.main import app
from app.storage.job_store import JobStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TINY_REPO = "HuggingFaceTB/SmolLM2-135M"
_MODEL_NAME = _TINY_REPO.split("/", 1)[1]  # "SmolLM2-135M"
_QUANT_TYPES = ["Q4_K_M"]
_POLL_INTERVAL_S = 10
_TIMEOUT_S = 600  # 10 minutes — llama.cpp build dominates on first run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_credentials():
    """Return (hf_token, hf_username) from env, or (None, None) if absent."""
    token = os.environ.get("HF_TOKEN")
    username = os.environ.get("HF_USERNAME")
    return token, username


def _delete_repo_if_exists(repo_id: str, token: str) -> None:
    """Best-effort delete of *repo_id* from HuggingFace."""
    try:
        api = HfApi(token=token)
        api.delete_repo(repo_id=repo_id, repo_type="model")
    except Exception:
        pass  # not fatal — cleanup is best-effort


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hf_credentials():
    """Yield (token, username) or skip the module if env vars are absent."""
    token, username = _get_credentials()
    if not token:
        pytest.skip("HF_TOKEN not set — skipping e2e tests")
    if not username:
        pytest.skip("HF_USERNAME not set — skipping e2e tests")
    yield token, username


@pytest.fixture(scope="module")
def e2e_client(hf_credentials):
    """TestClient wired to a fresh JobStore (no dependency on global singleton)."""
    store = JobStore()
    app.dependency_overrides[get_store] = lambda: store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_full_pipeline_completes_successfully(e2e_client, hf_credentials):
    """Submit a real job, poll until terminal, assert success."""
    token, username = hf_credentials

    payload = {
        "hf_repo": _TINY_REPO,
        "hf_username": username,
        "hf_token": token,
        "quant_types": _QUANT_TYPES,
        "keep_files": False,
    }

    # ---- submit --------------------------------------------------------
    post_resp = e2e_client.post("/jobs", json=payload)
    assert post_resp.status_code == 202, f"Expected 202, got: {post_resp.text}"
    job_id = post_resp.json()["job_id"]
    assert job_id, "Response must include a non-empty job_id"

    expected_output_repo = f"{username}/{_MODEL_NAME}-GGUF"

    # ---- poll until terminal or timeout --------------------------------
    deadline = time.monotonic() + _TIMEOUT_S
    final_data = None

    while time.monotonic() < deadline:
        poll_resp = e2e_client.get(f"/jobs/{job_id}")
        assert poll_resp.status_code == 200
        data = poll_resp.json()
        status = data["status"]

        if status in ("done", "failed"):
            final_data = data
            break

        time.sleep(_POLL_INTERVAL_S)
    else:
        pytest.fail(
            f"Pipeline did not reach a terminal state within {_TIMEOUT_S}s. "
            f"Last status: {data.get('status')!r}. "
            f"Progress: {data.get('progress')}"
        )

    # ---- cleanup (best-effort, runs regardless of assertion outcome) ----
    _delete_repo_if_exists(expected_output_repo, token)

    # ---- assertions ----------------------------------------------------
    assert final_data["status"] == "done", (
        f"Pipeline failed. Error: {final_data.get('error')!r}. "
        f"Progress:\n" + "\n".join(final_data.get("progress", []))
    )
    assert final_data["output_repo"] == expected_output_repo, (
        f"output_repo mismatch: {final_data['output_repo']!r}"
    )
    assert final_data["error"] is None


@pytest.mark.e2e
def test_full_pipeline_response_fields_present(e2e_client, hf_credentials):
    """All expected response fields are present in the final job record."""
    token, username = hf_credentials

    payload = {
        "hf_repo": _TINY_REPO,
        "hf_username": username,
        "hf_token": token,
        "quant_types": _QUANT_TYPES,
        "keep_files": False,
    }

    post_resp = e2e_client.post("/jobs", json=payload)
    assert post_resp.status_code == 202
    job_id = post_resp.json()["job_id"]

    deadline = time.monotonic() + _TIMEOUT_S
    data = {}

    while time.monotonic() < deadline:
        poll_resp = e2e_client.get(f"/jobs/{job_id}")
        data = poll_resp.json()
        if data["status"] in ("done", "failed"):
            break
        time.sleep(_POLL_INTERVAL_S)

    _delete_repo_if_exists(
        f"{username}/{_MODEL_NAME}-GGUF", token
    )

    required_fields = {"job_id", "status", "hf_repo", "hf_username", "created_at", "updated_at", "progress", "error", "output_repo"}
    for field in required_fields:
        assert field in data, f"Missing field: {field!r}"


@pytest.mark.e2e
def test_full_pipeline_token_never_appears_in_responses(e2e_client, hf_credentials):
    """The HF token must never be surfaced in any API response."""
    token, username = hf_credentials

    payload = {
        "hf_repo": _TINY_REPO,
        "hf_username": username,
        "hf_token": token,
        "quant_types": _QUANT_TYPES,
        "keep_files": False,
    }

    post_resp = e2e_client.post("/jobs", json=payload)
    assert token not in post_resp.text, "Token must not appear in POST response"

    job_id = post_resp.json()["job_id"]

    deadline = time.monotonic() + _TIMEOUT_S
    final_text = ""

    while time.monotonic() < deadline:
        poll_resp = e2e_client.get(f"/jobs/{job_id}")
        final_text = poll_resp.text
        if poll_resp.json()["status"] in ("done", "failed"):
            break
        time.sleep(_POLL_INTERVAL_S)

    _delete_repo_if_exists(
        f"{username}/{_MODEL_NAME}-GGUF", token
    )

    assert token not in final_text, "Token must not appear in GET /jobs/{id} response"

    list_resp = e2e_client.get("/jobs")
    assert token not in list_resp.text, "Token must not appear in GET /jobs response"


@pytest.mark.e2e
def test_full_pipeline_progress_log_is_non_empty(e2e_client, hf_credentials):
    """A completed job must have at least one progress log entry."""
    token, username = hf_credentials

    payload = {
        "hf_repo": _TINY_REPO,
        "hf_username": username,
        "hf_token": token,
        "quant_types": _QUANT_TYPES,
        "keep_files": False,
    }

    post_resp = e2e_client.post("/jobs", json=payload)
    job_id = post_resp.json()["job_id"]

    deadline = time.monotonic() + _TIMEOUT_S
    data = {}

    while time.monotonic() < deadline:
        poll_resp = e2e_client.get(f"/jobs/{job_id}")
        data = poll_resp.json()
        if data["status"] in ("done", "failed"):
            break
        time.sleep(_POLL_INTERVAL_S)

    _delete_repo_if_exists(
        f"{username}/{_MODEL_NAME}-GGUF", token
    )

    assert len(data.get("progress", [])) > 0, "Progress log must not be empty"


@pytest.mark.e2e
def test_full_pipeline_stream_endpoint_works(e2e_client, hf_credentials):
    """SSE stream for a completed job returns event: status with done."""
    token, username = hf_credentials

    payload = {
        "hf_repo": _TINY_REPO,
        "hf_username": username,
        "hf_token": token,
        "quant_types": _QUANT_TYPES,
        "keep_files": False,
    }

    post_resp = e2e_client.post("/jobs", json=payload)
    job_id = post_resp.json()["job_id"]

    deadline = time.monotonic() + _TIMEOUT_S
    data = {}

    while time.monotonic() < deadline:
        poll_resp = e2e_client.get(f"/jobs/{job_id}")
        data = poll_resp.json()
        if data["status"] in ("done", "failed"):
            break
        time.sleep(_POLL_INTERVAL_S)

    _delete_repo_if_exists(
        f"{username}/{_MODEL_NAME}-GGUF", token
    )

    # Only assert stream behaviour if the job succeeded
    if data.get("status") == "done":
        stream_resp = e2e_client.get(f"/jobs/{job_id}/stream")
        assert stream_resp.status_code == 200
        assert "text/event-stream" in stream_resp.headers["content-type"]
        assert "event: status" in stream_resp.text
        assert "done" in stream_resp.text
