from app.config import Settings
from app.config import settings as _default_settings
from app.storage.job_store import JobStore

# Singleton store for the lifetime of the process.
_store = JobStore()


def get_store() -> JobStore:
    """FastAPI dependency — returns the process-level JobStore.

    Override in tests via ``app.dependency_overrides[get_store] = lambda: fresh_store``.
    """
    return _store


def get_settings() -> Settings:
    """FastAPI dependency — returns the process-level Settings.

    Override in tests via ``app.dependency_overrides[get_settings] = lambda: custom_settings``.
    """
    return _default_settings
