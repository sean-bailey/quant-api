from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.config import Settings
from app.dependencies import get_settings, get_store
from app.models.request import JobListResponse, JobResponse, QuantizeRequest
from app.services import job_service
from app.storage.job_store import JobNotFoundError, JobStore
from app.workers import pipeline

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", status_code=202, response_model=JobResponse)
async def submit_job(
    request: QuantizeRequest,
    background_tasks: BackgroundTasks,
    store: JobStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
):
    """Accept a quantization request and start the pipeline in the background.

    Returns 202 Accepted immediately with the new job details.
    Poll ``GET /jobs/{job_id}`` or stream ``GET /jobs/{job_id}/stream``
    to follow progress.
    """
    job = await job_service.create_job(request, store)
    background_tasks.add_task(pipeline.run, job, request, store, settings)
    return JobResponse.from_job(job)


@router.get("", response_model=JobListResponse)
async def list_jobs(store: JobStore = Depends(get_store)):
    """Return a summary of every job in the store."""
    jobs = await job_service.list_jobs(store)
    return JobListResponse(jobs=[JobResponse.from_job(j) for j in jobs])


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, store: JobStore = Depends(get_store)):
    """Return full details (including progress log) for a single job."""
    try:
        job = await job_service.get_job(job_id, store)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JobResponse.from_job(job)


@router.get("/{job_id}/stream")
async def stream_job_logs(job_id: str, store: JobStore = Depends(get_store)):
    """Server-Sent Events stream of log lines for a running or completed job.

    Yields all existing ``data:`` lines immediately, then closes with a
    terminal ``event: status`` event carrying the current job status value.
    Clients that need live updates should reconnect periodically until they
    receive ``event: status`` with ``data: done`` or ``data: failed``.
    """
    try:
        job = await job_service.get_job(job_id, store)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")

    async def event_gen():
        for line in job.progress:
            yield f"data: {line}\n\n"
        yield f"event: status\ndata: {job.status.value}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
