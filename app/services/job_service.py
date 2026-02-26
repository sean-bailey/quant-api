from app.models.job import Job
from app.models.request import QuantizeRequest
from app.storage.job_store import JobStore


async def create_job(request: QuantizeRequest, store: JobStore) -> Job:
    """Create a new pending job from *request* and persist it to *store*.

    The token is never stored — it lives only in the in-flight request.
    """
    return await store.create(request.hf_repo, request.hf_username)


async def get_job(job_id: str, store: JobStore) -> Job:
    """Fetch a job by ID.

    Raises JobNotFoundError if *job_id* is not in the store.
    Callers (routers) are responsible for mapping this to HTTP 404.
    """
    return await store.get(job_id)


async def list_jobs(store: JobStore) -> list[Job]:
    """Return all jobs currently in the store."""
    return await store.list_all()
