import asyncio

import pytest

from app.models.job import JobStatus
from app.storage.job_store import JobNotFoundError, JobStore


@pytest.fixture
def store() -> JobStore:
    return JobStore()


# --- create ---

async def test_create_returns_job_with_pending_status(store):
    job = await store.create("meta-llama/Llama-3.2-1B", "alice")
    assert job.status == JobStatus.PENDING


async def test_create_stores_hf_repo(store):
    job = await store.create("meta-llama/Llama-3.2-1B", "alice")
    assert job.hf_repo == "meta-llama/Llama-3.2-1B"


async def test_create_stores_hf_username(store):
    job = await store.create("meta-llama/Llama-3.2-1B", "alice")
    assert job.hf_username == "alice"


async def test_create_returns_unique_ids(store):
    job1 = await store.create("a/b", "user")
    job2 = await store.create("a/b", "user")
    assert job1.job_id != job2.job_id


async def test_create_returns_independent_copy(store):
    """Mutating the returned job's progress must not affect the stored job.

    If create() returns a shallow copy, both progress lists are the same object
    and pipeline-style double-appending (job.progress.append + store.append_log)
    would cause every log line to appear twice.
    """
    job = await store.create("a/b", "user")
    job.progress.append("external mutation")
    stored = await store.get(job.job_id)
    assert stored.progress == [], "Stored job progress was mutated by the returned copy"


# --- get ---

async def test_get_returns_correct_job(store):
    created = await store.create("a/b", "user")
    fetched = await store.get(created.job_id)
    assert fetched.job_id == created.job_id
    assert fetched.hf_repo == "a/b"


async def test_get_nonexistent_job_raises_job_not_found(store):
    with pytest.raises(JobNotFoundError):
        await store.get("nonexistent-id")


async def test_job_not_found_error_contains_job_id(store):
    with pytest.raises(JobNotFoundError) as exc_info:
        await store.get("missing-abc")
    assert "missing-abc" in str(exc_info.value)


# --- list_all ---

async def test_list_all_empty_store_returns_empty_list(store):
    jobs = await store.list_all()
    assert jobs == []


async def test_list_all_returns_all_created_jobs(store):
    await store.create("a/b", "user")
    await store.create("c/d", "user")
    jobs = await store.list_all()
    assert len(jobs) == 2


async def test_list_all_returns_correct_repos(store):
    await store.create("org/ModelA", "user")
    await store.create("org/ModelB", "user")
    repos = {j.hf_repo for j in await store.list_all()}
    assert repos == {"org/ModelA", "org/ModelB"}


# --- update_status ---

async def test_update_status_persists(store):
    job = await store.create("a/b", "user")
    await store.update_status(job.job_id, JobStatus.BUILDING)
    updated = await store.get(job.job_id)
    assert updated.status == JobStatus.BUILDING


async def test_update_status_advances_updated_at(store):
    job = await store.create("a/b", "user")
    original_ts = job.updated_at
    await asyncio.sleep(0.01)
    await store.update_status(job.job_id, JobStatus.BUILDING)
    updated = await store.get(job.job_id)
    assert updated.updated_at > original_ts


async def test_update_status_nonexistent_job_raises(store):
    with pytest.raises(JobNotFoundError):
        await store.update_status("bad-id", JobStatus.BUILDING)


# --- append_log ---

async def test_append_log_accumulates_lines(store):
    job = await store.create("a/b", "user")
    await store.append_log(job.job_id, "step 1 done")
    await store.append_log(job.job_id, "step 2 done")
    updated = await store.get(job.job_id)
    assert updated.progress == ["step 1 done", "step 2 done"]


async def test_append_log_first_line_on_empty_progress(store):
    job = await store.create("a/b", "user")
    await store.append_log(job.job_id, "hello")
    updated = await store.get(job.job_id)
    assert updated.progress == ["hello"]


async def test_append_log_nonexistent_job_raises(store):
    with pytest.raises(JobNotFoundError):
        await store.append_log("bad-id", "some line")


# --- set_error ---

async def test_set_error_persists_message(store):
    job = await store.create("a/b", "user")
    await store.set_error(job.job_id, "something went wrong")
    updated = await store.get(job.job_id)
    assert updated.error == "something went wrong"


async def test_set_error_nonexistent_job_raises(store):
    with pytest.raises(JobNotFoundError):
        await store.set_error("bad-id", "error")


# --- set_output_repo ---

async def test_set_output_repo_persists(store):
    job = await store.create("a/b", "user")
    await store.set_output_repo(job.job_id, "alice/b-GGUF")
    updated = await store.get(job.job_id)
    assert updated.output_repo == "alice/b-GGUF"


async def test_set_output_repo_nonexistent_job_raises(store):
    with pytest.raises(JobNotFoundError):
        await store.set_output_repo("bad-id", "user/repo-GGUF")
