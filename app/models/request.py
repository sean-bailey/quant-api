import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, SecretStr, field_validator

from app.models.job import Job, JobStatus

_HF_REPO_RE = re.compile(r"^[^/\s]+/[^/\s]+$")


class QuantizeRequest(BaseModel):
    hf_repo: str
    hf_username: str
    hf_token: SecretStr
    quant_types: Optional[list[str]] = None
    threads: Optional[int] = None
    keep_files: bool = False
    imatrix: Optional[str] = None

    @field_validator("hf_repo")
    @classmethod
    def validate_hf_repo_format(cls, v: str) -> str:
        if not _HF_REPO_RE.match(v):
            raise ValueError(
                "hf_repo must be in 'owner/name' format with no spaces "
                "(e.g. 'meta-llama/Llama-3.2-1B-Instruct')"
            )
        return v


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    hf_repo: str
    hf_username: str
    created_at: datetime
    updated_at: datetime
    progress: list[str]
    error: Optional[str]
    output_repo: Optional[str]

    @classmethod
    def from_job(cls, job: Job) -> "JobResponse":
        return cls(
            job_id=job.job_id,
            status=job.status,
            hf_repo=job.hf_repo,
            hf_username=job.hf_username,
            created_at=job.created_at,
            updated_at=job.updated_at,
            progress=job.progress,
            error=job.error,
            output_repo=job.output_repo,
        )


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
