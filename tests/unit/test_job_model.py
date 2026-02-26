import pytest

from app.models.job import Job, JobStatus, compute_output_repo


# --- Job field defaults ---

def test_job_default_status_is_pending():
    job = Job(hf_repo="a/b", hf_username="user")
    assert job.status == JobStatus.PENDING


def test_job_default_progress_is_empty():
    job = Job(hf_repo="a/b", hf_username="user")
    assert job.progress == []


def test_job_default_error_is_none():
    job = Job(hf_repo="a/b", hf_username="user")
    assert job.error is None


def test_job_default_output_repo_is_none():
    job = Job(hf_repo="a/b", hf_username="user")
    assert job.output_repo is None


def test_job_id_is_auto_generated():
    job = Job(hf_repo="a/b", hf_username="user")
    assert job.job_id is not None
    assert len(job.job_id) > 0


def test_two_jobs_have_different_ids():
    job1 = Job(hf_repo="a/b", hf_username="user")
    job2 = Job(hf_repo="a/b", hf_username="user")
    assert job1.job_id != job2.job_id


def test_job_stores_hf_repo_and_username():
    job = Job(hf_repo="meta-llama/Llama-3.2-1B", hf_username="alice")
    assert job.hf_repo == "meta-llama/Llama-3.2-1B"
    assert job.hf_username == "alice"


def test_job_has_created_at_and_updated_at():
    job = Job(hf_repo="a/b", hf_username="user")
    assert job.created_at is not None
    assert job.updated_at is not None


# --- compute_output_repo ---

def test_output_repo_uses_username_and_model_name():
    result = compute_output_repo("meta-llama/Llama-3.2-1B-Instruct", "myuser")
    assert result == "myuser/Llama-3.2-1B-Instruct-GGUF"


def test_output_repo_strips_source_org():
    result = compute_output_repo("bigscience/bloom", "alice")
    assert result == "alice/bloom-GGUF"


def test_output_repo_appends_gguf_suffix():
    result = compute_output_repo("org/SomeModel", "bob")
    assert result.endswith("-GGUF")


def test_output_repo_owner_is_username_not_source_org():
    result = compute_output_repo("source-org/ModelName", "dest-user")
    assert result.startswith("dest-user/")
    assert "source-org" not in result


# --- JobStatus transitions ---

def test_pending_can_transition_to_building():
    assert JobStatus.PENDING.can_transition_to(JobStatus.BUILDING)


def test_building_can_transition_to_downloading():
    assert JobStatus.BUILDING.can_transition_to(JobStatus.DOWNLOADING)


def test_downloading_can_transition_to_converting():
    assert JobStatus.DOWNLOADING.can_transition_to(JobStatus.CONVERTING)


def test_converting_can_transition_to_quantizing():
    assert JobStatus.CONVERTING.can_transition_to(JobStatus.QUANTIZING)


def test_quantizing_can_transition_to_uploading():
    assert JobStatus.QUANTIZING.can_transition_to(JobStatus.UPLOADING)


def test_uploading_can_transition_to_done():
    assert JobStatus.UPLOADING.can_transition_to(JobStatus.DONE)


def test_done_cannot_transition_to_any_status():
    for status in JobStatus:
        assert not JobStatus.DONE.can_transition_to(status)


def test_failed_cannot_transition_to_any_status():
    for status in JobStatus:
        assert not JobStatus.FAILED.can_transition_to(status)


def test_any_active_status_can_transition_to_failed():
    active = [
        JobStatus.PENDING,
        JobStatus.BUILDING,
        JobStatus.DOWNLOADING,
        JobStatus.CONVERTING,
        JobStatus.QUANTIZING,
        JobStatus.UPLOADING,
    ]
    for status in active:
        assert status.can_transition_to(JobStatus.FAILED), f"{status} should allow → FAILED"


def test_done_cannot_transition_back_to_pending():
    assert not JobStatus.DONE.can_transition_to(JobStatus.PENDING)


def test_status_values_are_lowercase_strings():
    assert JobStatus.PENDING.value == "pending"
    assert JobStatus.DONE.value == "done"
    assert JobStatus.FAILED.value == "failed"
