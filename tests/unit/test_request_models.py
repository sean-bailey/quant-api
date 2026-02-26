import pytest
from pydantic import ValidationError

from app.models.job import Job, JobStatus
from app.models.request import JobListResponse, JobResponse, QuantizeRequest


# ---------------------------------------------------------------------------
# QuantizeRequest — required fields
# ---------------------------------------------------------------------------

def test_quantize_request_requires_hf_repo():
    with pytest.raises(ValidationError):
        QuantizeRequest(hf_username="user", hf_token="tok")


def test_quantize_request_requires_hf_username():
    with pytest.raises(ValidationError):
        QuantizeRequest(hf_repo="a/b", hf_token="tok")


def test_quantize_request_requires_hf_token():
    with pytest.raises(ValidationError):
        QuantizeRequest(hf_repo="a/b", hf_username="user")


# ---------------------------------------------------------------------------
# QuantizeRequest — hf_repo format validation
# ---------------------------------------------------------------------------

def test_hf_repo_without_slash_raises():
    with pytest.raises(ValidationError):
        QuantizeRequest(hf_repo="noslash", hf_username="u", hf_token="tok")


def test_hf_repo_with_empty_owner_raises():
    with pytest.raises(ValidationError):
        QuantizeRequest(hf_repo="/modelname", hf_username="u", hf_token="tok")


def test_hf_repo_with_empty_name_raises():
    with pytest.raises(ValidationError):
        QuantizeRequest(hf_repo="owner/", hf_username="u", hf_token="tok")


def test_hf_repo_with_two_slashes_raises():
    with pytest.raises(ValidationError):
        QuantizeRequest(hf_repo="a/b/c", hf_username="u", hf_token="tok")


def test_hf_repo_with_whitespace_raises():
    with pytest.raises(ValidationError):
        QuantizeRequest(hf_repo="owner /name", hf_username="u", hf_token="tok")


def test_hf_repo_simple_valid_format_accepted():
    req = QuantizeRequest(hf_repo="org/model", hf_username="u", hf_token="tok")
    assert req.hf_repo == "org/model"


def test_hf_repo_real_world_format_accepted():
    req = QuantizeRequest(
        hf_repo="meta-llama/Llama-3.2-1B-Instruct",
        hf_username="u",
        hf_token="tok",
    )
    assert req.hf_repo == "meta-llama/Llama-3.2-1B-Instruct"


def test_hf_repo_with_dots_and_underscores_accepted():
    req = QuantizeRequest(hf_repo="my_org/my.model_v2", hf_username="u", hf_token="tok")
    assert req.hf_repo == "my_org/my.model_v2"


# ---------------------------------------------------------------------------
# QuantizeRequest — hf_token security
# ---------------------------------------------------------------------------

def test_hf_token_not_in_model_dump_json():
    req = QuantizeRequest(hf_repo="a/b", hf_username="u", hf_token="supersecret")
    assert "supersecret" not in req.model_dump_json()


def test_hf_token_not_in_str():
    req = QuantizeRequest(hf_repo="a/b", hf_username="u", hf_token="supersecret")
    assert "supersecret" not in str(req)


def test_hf_token_not_in_repr():
    req = QuantizeRequest(hf_repo="a/b", hf_username="u", hf_token="supersecret")
    assert "supersecret" not in repr(req)


def test_hf_token_value_accessible_via_get_secret_value():
    req = QuantizeRequest(hf_repo="a/b", hf_username="u", hf_token="mytoken123")
    assert req.hf_token.get_secret_value() == "mytoken123"


# ---------------------------------------------------------------------------
# QuantizeRequest — optional field defaults
# ---------------------------------------------------------------------------

def test_quant_types_defaults_to_none():
    req = QuantizeRequest(hf_repo="a/b", hf_username="u", hf_token="tok")
    assert req.quant_types is None


def test_threads_defaults_to_none():
    req = QuantizeRequest(hf_repo="a/b", hf_username="u", hf_token="tok")
    assert req.threads is None


def test_keep_files_defaults_to_false():
    req = QuantizeRequest(hf_repo="a/b", hf_username="u", hf_token="tok")
    assert req.keep_files is False


def test_imatrix_defaults_to_none():
    req = QuantizeRequest(hf_repo="a/b", hf_username="u", hf_token="tok")
    assert req.imatrix is None


# ---------------------------------------------------------------------------
# QuantizeRequest — optional fields can be set
# ---------------------------------------------------------------------------

def test_quant_types_can_be_set():
    req = QuantizeRequest(
        hf_repo="a/b", hf_username="u", hf_token="tok",
        quant_types=["Q4_K_M", "Q5_K_M"],
    )
    assert req.quant_types == ["Q4_K_M", "Q5_K_M"]


def test_threads_can_be_set():
    req = QuantizeRequest(hf_repo="a/b", hf_username="u", hf_token="tok", threads=8)
    assert req.threads == 8


def test_keep_files_can_be_true():
    req = QuantizeRequest(hf_repo="a/b", hf_username="u", hf_token="tok", keep_files=True)
    assert req.keep_files is True


def test_imatrix_can_be_set():
    req = QuantizeRequest(
        hf_repo="a/b", hf_username="u", hf_token="tok",
        imatrix="/path/to/imatrix.bin",
    )
    assert req.imatrix == "/path/to/imatrix.bin"


# ---------------------------------------------------------------------------
# JobResponse — built from Job
# ---------------------------------------------------------------------------

def test_job_response_from_job_preserves_job_id():
    job = Job(hf_repo="a/b", hf_username="user")
    resp = JobResponse.from_job(job)
    assert resp.job_id == job.job_id


def test_job_response_from_job_preserves_status():
    job = Job(hf_repo="a/b", hf_username="user")
    resp = JobResponse.from_job(job)
    assert resp.status == JobStatus.PENDING


def test_job_response_from_job_preserves_hf_repo():
    job = Job(hf_repo="org/ModelX", hf_username="user")
    resp = JobResponse.from_job(job)
    assert resp.hf_repo == "org/ModelX"


def test_job_response_from_job_preserves_hf_username():
    job = Job(hf_repo="a/b", hf_username="alice")
    resp = JobResponse.from_job(job)
    assert resp.hf_username == "alice"


def test_job_response_from_job_preserves_progress():
    job = Job(hf_repo="a/b", hf_username="user", progress=["step 1", "step 2"])
    resp = JobResponse.from_job(job)
    assert resp.progress == ["step 1", "step 2"]


def test_job_response_from_job_preserves_error():
    job = Job(hf_repo="a/b", hf_username="user", error="it broke")
    resp = JobResponse.from_job(job)
    assert resp.error == "it broke"


def test_job_response_from_job_preserves_output_repo():
    job = Job(hf_repo="a/b", hf_username="user", output_repo="user/b-GGUF")
    resp = JobResponse.from_job(job)
    assert resp.output_repo == "user/b-GGUF"


def test_job_response_has_no_token_attribute():
    job = Job(hf_repo="a/b", hf_username="user")
    resp = JobResponse.from_job(job)
    assert not hasattr(resp, "hf_token")


def test_job_response_json_contains_no_token_key():
    job = Job(hf_repo="a/b", hf_username="user")
    resp = JobResponse.from_job(job)
    assert "token" not in resp.model_dump_json()


def test_job_response_json_contains_status():
    job = Job(hf_repo="a/b", hf_username="user")
    resp = JobResponse.from_job(job)
    assert "pending" in resp.model_dump_json()


# ---------------------------------------------------------------------------
# JobListResponse
# ---------------------------------------------------------------------------

def test_job_list_response_empty_jobs():
    resp = JobListResponse(jobs=[])
    assert resp.jobs == []


def test_job_list_response_with_jobs():
    job1 = Job(hf_repo="a/b", hf_username="user")
    job2 = Job(hf_repo="c/d", hf_username="user")
    resp = JobListResponse(jobs=[JobResponse.from_job(job1), JobResponse.from_job(job2)])
    assert len(resp.jobs) == 2


def test_job_list_response_jobs_have_correct_repos():
    job1 = Job(hf_repo="org/ModelA", hf_username="user")
    job2 = Job(hf_repo="org/ModelB", hf_username="user")
    resp = JobListResponse(jobs=[JobResponse.from_job(job1), JobResponse.from_job(job2)])
    repos = {j.hf_repo for j in resp.jobs}
    assert repos == {"org/ModelA", "org/ModelB"}
