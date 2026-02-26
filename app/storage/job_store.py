import asyncio
from datetime import datetime, timezone

from app.models.job import Job, JobStatus


class JobNotFoundError(Exception):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"Job not found: {job_id}")
        self.job_id = job_id


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self, hf_repo: str, hf_username: str) -> Job:
        job = Job(hf_repo=hf_repo, hf_username=hf_username)
        async with self._lock:
            self._jobs[job.job_id] = job
        return job.model_copy(deep=True)

    async def get(self, job_id: str) -> Job:
        async with self._lock:
            self._raise_if_missing(job_id)
            return self._jobs[job_id].model_copy(deep=True)

    async def list_all(self) -> list[Job]:
        async with self._lock:
            return [j.model_copy(deep=True) for j in self._jobs.values()]

    async def update_status(self, job_id: str, status: JobStatus) -> Job:
        async with self._lock:
            self._raise_if_missing(job_id)
            job = self._jobs[job_id]
            job.status = status
            job.updated_at = datetime.now(timezone.utc)
            return job.model_copy()

    async def append_log(self, job_id: str, line: str) -> None:
        async with self._lock:
            self._raise_if_missing(job_id)
            job = self._jobs[job_id]
            job.progress.append(line)
            job.updated_at = datetime.now(timezone.utc)

    async def set_error(self, job_id: str, error: str) -> None:
        async with self._lock:
            self._raise_if_missing(job_id)
            job = self._jobs[job_id]
            job.error = error
            job.updated_at = datetime.now(timezone.utc)

    async def set_output_repo(self, job_id: str, output_repo: str) -> None:
        async with self._lock:
            self._raise_if_missing(job_id)
            job = self._jobs[job_id]
            job.output_repo = output_repo
            job.updated_at = datetime.now(timezone.utc)

    def _raise_if_missing(self, job_id: str) -> None:
        if job_id not in self._jobs:
            raise JobNotFoundError(job_id)
