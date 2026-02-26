import pytest

from app.models.job import JobStatus
from app.models.request import QuantizeRequest
from app.services.job_service import create_job, get_job, list_jobs
from app.storage.job_store import JobNotFoundError, JobStore


@pytest.fixture
def store() -> JobStore:
    return JobStore()


@pytest.fixture
def quant_request() -> QuantizeRequest:
    return QuantizeRequest(
        hf_repo="meta-llama/Llama-3.2-1B",
        hf_username="alice",
        hf_token="hf_test",
    )


# ---------------------------------------------------------------------------
# create_job
# ---------------------------------------------------------------------------

async def test_create_job_returns_pending_job(store, quant_request):
    job = await create_job(quant_request, store)
    assert job.status == JobStatus.PENDING


async def test_create_job_stores_hf_repo(store, quant_request):
    job = await create_job(quant_request, store)
    assert job.hf_repo == "meta-llama/Llama-3.2-1B"


async def test_create_job_stores_hf_username(store, quant_request):
    job = await create_job(quant_request, store)
    assert job.hf_username == "alice"


async def test_create_job_does_not_store_token(store, quant_request):
    job = await create_job(quant_request, store)
    assert not hasattr(job, "hf_token")


async def test_create_job_assigns_unique_ids(store, quant_request):
    job1 = await create_job(quant_request, store)
    job2 = await create_job(quant_request, store)
    assert job1.job_id != job2.job_id


async def test_create_job_is_retrievable_from_store(store, quant_request):
    job = await create_job(quant_request, store)
    fetched = await store.get(job.job_id)
    assert fetched.job_id == job.job_id


# ---------------------------------------------------------------------------
# get_job
# ---------------------------------------------------------------------------

async def test_get_job_returns_correct_job(store, quant_request):
    created = await create_job(quant_request, store)
    fetched = await get_job(created.job_id, store)
    assert fetched.job_id == created.job_id
    assert fetched.hf_repo == created.hf_repo


async def test_get_job_raises_job_not_found_for_unknown_id(store):
    with pytest.raises(JobNotFoundError):
        await get_job("does-not-exist", store)


async def test_get_job_reflects_status_updates(store, quant_request):
    job = await create_job(quant_request, store)
    await store.update_status(job.job_id, JobStatus.BUILDING)
    fetched = await get_job(job.job_id, store)
    assert fetched.status == JobStatus.BUILDING


# ---------------------------------------------------------------------------
# list_jobs
# ---------------------------------------------------------------------------

async def test_list_jobs_returns_empty_list_initially(store):
    jobs = await list_jobs(store)
    assert jobs == []


async def test_list_jobs_returns_all_created_jobs(store, quant_request):
    await create_job(quant_request, store)
    await create_job(quant_request, store)
    jobs = await list_jobs(store)
    assert len(jobs) == 2


async def test_list_jobs_returns_correct_repos(store):
    req_a = QuantizeRequest(hf_repo="org/ModelA", hf_username="u", hf_token="t")
    req_b = QuantizeRequest(hf_repo="org/ModelB", hf_username="u", hf_token="t")
    await create_job(req_a, store)
    await create_job(req_b, store)
    jobs = await list_jobs(store)
    repos = {j.hf_repo for j in jobs}
    assert repos == {"org/ModelA", "org/ModelB"}
