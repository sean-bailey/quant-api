from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    BUILDING = "building"
    DOWNLOADING = "downloading"
    CONVERTING = "converting"
    QUANTIZING = "quantizing"
    UPLOADING = "uploading"
    DONE = "done"
    FAILED = "failed"

    def can_transition_to(self, next_status: JobStatus) -> bool:
        return next_status in _VALID_TRANSITIONS.get(self, set())


_VALID_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.PENDING:     {JobStatus.BUILDING,    JobStatus.FAILED},
    JobStatus.BUILDING:    {JobStatus.DOWNLOADING, JobStatus.FAILED},
    JobStatus.DOWNLOADING: {JobStatus.CONVERTING,  JobStatus.FAILED},
    JobStatus.CONVERTING:  {JobStatus.QUANTIZING,  JobStatus.FAILED},
    JobStatus.QUANTIZING:  {JobStatus.UPLOADING,   JobStatus.FAILED},
    JobStatus.UPLOADING:   {JobStatus.DONE,        JobStatus.FAILED},
    JobStatus.DONE:        set(),
    JobStatus.FAILED:      set(),
}


def compute_output_repo(hf_repo: str, username: str) -> str:
    """Return the destination HF repo name: {username}/{ModelName}-GGUF."""
    model_name = hf_repo.split("/", 1)[1]
    return f"{username}/{model_name}-GGUF"


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.PENDING
    hf_repo: str
    hf_username: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    progress: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    output_repo: Optional[str] = None
