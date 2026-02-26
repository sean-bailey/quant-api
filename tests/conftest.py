import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_store
from app.main import app
from app.storage.job_store import JobStore


@pytest.fixture
def client() -> TestClient:
    """Base client fixture — fresh store, no dependency on pipeline mock.

    Used by Phase 1 health tests.  Router tests define their own ``client``
    fixture with a mocked pipeline to prevent background tasks from running.
    """
    store = JobStore()
    app.dependency_overrides[get_store] = lambda: store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
