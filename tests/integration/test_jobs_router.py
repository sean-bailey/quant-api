import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_store
from app.main import app
from app.storage.job_store import JobStore

VALID_PAYLOAD = {
    "hf_repo": "meta-llama/Llama-3.2-1B",
    "hf_username": "testuser",
    "hf_token": "hf_abc123",
}


# ---------------------------------------------------------------------------
# Fixtures — fresh store + mocked pipeline per test
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> JobStore:
    return JobStore()


@pytest.fixture
def mock_pipeline_run(mocker):
    """Prevent the real pipeline from running during router tests."""
    return mocker.patch("app.routers.jobs.pipeline.run")


@pytest.fixture
def client(store, mock_pipeline_run):
    app.dependency_overrides[get_store] = lambda: store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /jobs — success cases
# ---------------------------------------------------------------------------

def test_post_jobs_returns_202(client):
    response = client.post("/jobs", json=VALID_PAYLOAD)
    assert response.status_code == 202


def test_post_jobs_response_contains_job_id(client):
    response = client.post("/jobs", json=VALID_PAYLOAD)
    assert "job_id" in response.json()


def test_post_jobs_job_id_is_non_empty_string(client):
    response = client.post("/jobs", json=VALID_PAYLOAD)
    job_id = response.json()["job_id"]
    assert isinstance(job_id, str) and len(job_id) > 0


def test_post_jobs_returns_pending_status(client):
    response = client.post("/jobs", json=VALID_PAYLOAD)
    assert response.json()["status"] == "pending"


def test_post_jobs_response_contains_hf_repo(client):
    response = client.post("/jobs", json=VALID_PAYLOAD)
    assert response.json()["hf_repo"] == "meta-llama/Llama-3.2-1B"


def test_post_jobs_response_contains_hf_username(client):
    response = client.post("/jobs", json=VALID_PAYLOAD)
    assert response.json()["hf_username"] == "testuser"


def test_post_jobs_response_contains_empty_progress(client):
    response = client.post("/jobs", json=VALID_PAYLOAD)
    assert response.json()["progress"] == []


def test_post_jobs_triggers_pipeline_as_background_task(client, mock_pipeline_run):
    client.post("/jobs", json=VALID_PAYLOAD)
    mock_pipeline_run.assert_called_once()


def test_post_jobs_pipeline_called_with_correct_repo(client, mock_pipeline_run):
    client.post("/jobs", json=VALID_PAYLOAD)
    _, call_kwargs = mock_pipeline_run.call_args
    call_args = mock_pipeline_run.call_args[0]
    request_arg = call_args[1]
    assert request_arg.hf_repo == "meta-llama/Llama-3.2-1B"


def test_post_jobs_two_submissions_create_two_jobs(client):
    resp1 = client.post("/jobs", json=VALID_PAYLOAD)
    resp2 = client.post("/jobs", json=VALID_PAYLOAD)
    assert resp1.json()["job_id"] != resp2.json()["job_id"]


# ---------------------------------------------------------------------------
# POST /jobs — validation failures
# ---------------------------------------------------------------------------

def test_post_jobs_missing_hf_repo_returns_422(client):
    response = client.post("/jobs", json={"hf_username": "u", "hf_token": "t"})
    assert response.status_code == 422


def test_post_jobs_missing_hf_username_returns_422(client):
    response = client.post("/jobs", json={"hf_repo": "a/b", "hf_token": "t"})
    assert response.status_code == 422


def test_post_jobs_missing_hf_token_returns_422(client):
    response = client.post("/jobs", json={"hf_repo": "a/b", "hf_username": "u"})
    assert response.status_code == 422


def test_post_jobs_invalid_repo_no_slash_returns_422(client):
    response = client.post("/jobs", json={**VALID_PAYLOAD, "hf_repo": "noslash"})
    assert response.status_code == 422


def test_post_jobs_invalid_repo_empty_owner_returns_422(client):
    response = client.post("/jobs", json={**VALID_PAYLOAD, "hf_repo": "/model"})
    assert response.status_code == 422


def test_post_jobs_invalid_repo_two_slashes_returns_422(client):
    response = client.post("/jobs", json={**VALID_PAYLOAD, "hf_repo": "a/b/c"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /jobs — token security
# ---------------------------------------------------------------------------

def test_post_jobs_token_not_in_response_body(client):
    payload = {**VALID_PAYLOAD, "hf_token": "supersecrettoken"}
    response = client.post("/jobs", json=payload)
    assert "supersecrettoken" not in response.text


def test_post_jobs_token_keyword_not_in_response_body(client):
    response = client.post("/jobs", json=VALID_PAYLOAD)
    data = response.json()
    assert "hf_token" not in data
    assert "token" not in data


# ---------------------------------------------------------------------------
# GET /jobs
# ---------------------------------------------------------------------------

def test_get_jobs_returns_200(client):
    response = client.get("/jobs")
    assert response.status_code == 200


def test_get_jobs_empty_store_returns_empty_list(client):
    response = client.get("/jobs")
    assert response.json() == {"jobs": []}


def test_get_jobs_returns_jobs_key(client):
    response = client.get("/jobs")
    assert "jobs" in response.json()


def test_get_jobs_after_one_post_returns_one_job(client):
    client.post("/jobs", json=VALID_PAYLOAD)
    response = client.get("/jobs")
    assert len(response.json()["jobs"]) == 1


def test_get_jobs_after_two_posts_returns_two_jobs(client):
    client.post("/jobs", json=VALID_PAYLOAD)
    client.post("/jobs", json=VALID_PAYLOAD)
    response = client.get("/jobs")
    assert len(response.json()["jobs"]) == 2


def test_get_jobs_each_entry_has_job_id(client):
    client.post("/jobs", json=VALID_PAYLOAD)
    jobs = client.get("/jobs").json()["jobs"]
    assert all("job_id" in j for j in jobs)


def test_get_jobs_each_entry_has_status(client):
    client.post("/jobs", json=VALID_PAYLOAD)
    jobs = client.get("/jobs").json()["jobs"]
    assert all("status" in j for j in jobs)


def test_get_jobs_token_not_in_list_response(client):
    client.post("/jobs", json={**VALID_PAYLOAD, "hf_token": "supersecrettoken"})
    response = client.get("/jobs")
    assert "supersecrettoken" not in response.text


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}
# ---------------------------------------------------------------------------

def test_get_job_by_id_returns_200(client):
    post_resp = client.post("/jobs", json=VALID_PAYLOAD)
    job_id = post_resp.json()["job_id"]
    assert client.get(f"/jobs/{job_id}").status_code == 200


def test_get_job_by_id_returns_correct_job_id(client):
    post_resp = client.post("/jobs", json=VALID_PAYLOAD)
    job_id = post_resp.json()["job_id"]
    response = client.get(f"/jobs/{job_id}")
    assert response.json()["job_id"] == job_id


def test_get_job_by_id_returns_hf_repo(client):
    post_resp = client.post("/jobs", json=VALID_PAYLOAD)
    job_id = post_resp.json()["job_id"]
    assert client.get(f"/jobs/{job_id}").json()["hf_repo"] == "meta-llama/Llama-3.2-1B"


def test_get_job_by_id_returns_hf_username(client):
    post_resp = client.post("/jobs", json=VALID_PAYLOAD)
    job_id = post_resp.json()["job_id"]
    assert client.get(f"/jobs/{job_id}").json()["hf_username"] == "testuser"


def test_get_job_by_id_returns_progress_list(client):
    post_resp = client.post("/jobs", json=VALID_PAYLOAD)
    job_id = post_resp.json()["job_id"]
    data = client.get(f"/jobs/{job_id}").json()
    assert "progress" in data
    assert isinstance(data["progress"], list)


def test_get_job_by_id_returns_error_field(client):
    post_resp = client.post("/jobs", json=VALID_PAYLOAD)
    job_id = post_resp.json()["job_id"]
    data = client.get(f"/jobs/{job_id}").json()
    assert "error" in data


def test_get_job_by_id_returns_output_repo_field(client):
    post_resp = client.post("/jobs", json=VALID_PAYLOAD)
    job_id = post_resp.json()["job_id"]
    data = client.get(f"/jobs/{job_id}").json()
    assert "output_repo" in data


def test_get_job_unknown_id_returns_404(client):
    assert client.get("/jobs/nonexistent-id").status_code == 404


def test_get_job_unknown_id_has_detail(client):
    response = client.get("/jobs/no-such-job")
    assert "detail" in response.json()


def test_get_job_token_not_in_detail_response(client):
    post_resp = client.post("/jobs", json={**VALID_PAYLOAD, "hf_token": "supersecrettoken"})
    job_id = post_resp.json()["job_id"]
    response = client.get(f"/jobs/{job_id}")
    assert "supersecrettoken" not in response.text


def test_get_job_different_jobs_return_different_ids(client):
    id1 = client.post("/jobs", json=VALID_PAYLOAD).json()["job_id"]
    id2 = client.post("/jobs", json=VALID_PAYLOAD).json()["job_id"]
    assert client.get(f"/jobs/{id1}").json()["job_id"] != client.get(f"/jobs/{id2}").json()["job_id"]


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/stream
# ---------------------------------------------------------------------------

def test_stream_unknown_job_returns_404(client):
    assert client.get("/jobs/nonexistent-id/stream").status_code == 404


def test_stream_returns_200_for_valid_job(client):
    post_resp = client.post("/jobs", json=VALID_PAYLOAD)
    job_id = post_resp.json()["job_id"]
    assert client.get(f"/jobs/{job_id}/stream").status_code == 200


def test_stream_returns_event_stream_content_type(client):
    post_resp = client.post("/jobs", json=VALID_PAYLOAD)
    job_id = post_resp.json()["job_id"]
    response = client.get(f"/jobs/{job_id}/stream")
    assert "text/event-stream" in response.headers["content-type"]


def test_stream_contains_status_event(client):
    post_resp = client.post("/jobs", json=VALID_PAYLOAD)
    job_id = post_resp.json()["job_id"]
    response = client.get(f"/jobs/{job_id}/stream")
    assert "event: status" in response.text


def test_stream_status_event_contains_job_status_value(client):
    post_resp = client.post("/jobs", json=VALID_PAYLOAD)
    job_id = post_resp.json()["job_id"]
    response = client.get(f"/jobs/{job_id}/stream")
    assert "pending" in response.text


def test_stream_yields_existing_log_lines(client, store):
    """If a job already has progress lines, the stream yields them."""
    post_resp = client.post("/jobs", json=VALID_PAYLOAD)
    job_id = post_resp.json()["job_id"]
    # Inject a log line directly into the store
    import asyncio
    asyncio.get_event_loop().run_until_complete(
        store.append_log(job_id, "step 1 complete")
    )
    response = client.get(f"/jobs/{job_id}/stream")
    assert "step 1 complete" in response.text
