from pathlib import Path

import pytest

from app.config import Settings
from app.models.job import Job, JobStatus
from app.models.request import QuantizeRequest
from app.services.build_service import BuildError
from app.services.hf_service import DownloadError, InvalidCredentialsError, UploadError
from app.services.quant_service import ConversionError
from app.storage.job_store import JobStore
from app.workers import pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> JobStore:
    return JobStore()


@pytest.fixture
def quant_request() -> QuantizeRequest:
    return QuantizeRequest(
        hf_repo="meta-llama/Llama-3.2-1B",
        hf_username="testuser",
        hf_token="hf_testtoken",
        quant_types=["Q4_K_M"],  # single type keeps tests fast
    )


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    return Settings(
        models_dir=str(tmp_path / "models"),
        llama_cpp_dir=str(tmp_path / "llama.cpp"),
        build_dir=str(tmp_path / "build"),
        venv_path=str(tmp_path / "venv"),
        threads=4,
    )


async def _make_job(store: JobStore, hf_repo: str = "meta-llama/Llama-3.2-1B") -> Job:
    """Create a job in the store and return it."""
    return await store.create(hf_repo, "testuser")


def _patch_all(mocker, model_dir: Path):
    """Patch every external service call to succeed silently."""
    mocker.patch("app.workers.pipeline.validate_credentials", return_value=True)
    mocker.patch("app.workers.pipeline.clone_or_update_llama_cpp")
    mocker.patch("app.workers.pipeline.build_llama_cpp")
    mocker.patch("app.workers.pipeline.get_quantize_binary", return_value="/mock/llama-quantize")
    mocker.patch("app.workers.pipeline.setup_venv", return_value="/mock/python")
    mocker.patch("app.workers.pipeline.create_output_repo", return_value="testuser/Llama-3.2-1B-GGUF")
    mocker.patch("app.workers.pipeline.clone_model", return_value=model_dir)
    mocker.patch("app.workers.pipeline.is_multimodal", return_value=False)
    mocker.patch("app.workers.pipeline.convert_to_gguf")
    mocker.patch("app.workers.pipeline.clean_source_files", return_value=0)
    mocker.patch("app.workers.pipeline.quantize_single", return_value="ok")
    mocker.patch("app.workers.pipeline.upload_file")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_happy_path_job_reaches_done(mocker, store, quant_request, test_settings, tmp_path):
    job = await _make_job(store)
    model_dir = Path(test_settings.models_dir) / "Llama-3.2-1B"
    _patch_all(mocker, model_dir)

    await pipeline.run(job, quant_request, store, test_settings)

    assert job.status == JobStatus.DONE


async def test_happy_path_store_reflects_done(mocker, store, quant_request, test_settings, tmp_path):
    job = await _make_job(store)
    model_dir = Path(test_settings.models_dir) / "Llama-3.2-1B"
    _patch_all(mocker, model_dir)

    await pipeline.run(job, quant_request, store, test_settings)

    stored = await store.get(job.job_id)
    assert stored.status == JobStatus.DONE


async def test_happy_path_sets_output_repo(mocker, store, quant_request, test_settings, tmp_path):
    job = await _make_job(store)
    model_dir = Path(test_settings.models_dir) / "Llama-3.2-1B"
    _patch_all(mocker, model_dir)

    await pipeline.run(job, quant_request, store, test_settings)

    assert job.output_repo == "testuser/Llama-3.2-1B-GGUF"


async def test_happy_path_progress_contains_log_entries(mocker, store, quant_request, test_settings, tmp_path):
    job = await _make_job(store)
    model_dir = Path(test_settings.models_dir) / "Llama-3.2-1B"
    _patch_all(mocker, model_dir)

    await pipeline.run(job, quant_request, store, test_settings)

    assert len(job.progress) > 0


async def test_happy_path_progress_contains_phase_markers(mocker, store, quant_request, test_settings, tmp_path):
    job = await _make_job(store)
    model_dir = Path(test_settings.models_dir) / "Llama-3.2-1B"
    _patch_all(mocker, model_dir)

    await pipeline.run(job, quant_request, store, test_settings)

    combined = "\n".join(job.progress)
    assert "credentials" in combined.lower()
    assert "llama.cpp" in combined.lower()
    assert "downloading" in combined.lower() or "Downloading" in combined
    assert "converting" in combined.lower() or "Converting" in combined
    assert "quantiz" in combined.lower()


async def test_happy_path_does_not_raise(mocker, store, quant_request, test_settings, tmp_path):
    job = await _make_job(store)
    model_dir = Path(test_settings.models_dir) / "Llama-3.2-1B"
    _patch_all(mocker, model_dir)
    # Should complete silently — no exception should propagate
    await pipeline.run(job, quant_request, store, test_settings)


# ---------------------------------------------------------------------------
# Status transitions during happy path
# ---------------------------------------------------------------------------

async def test_pipeline_transitions_through_building(mocker, store, quant_request, test_settings, tmp_path):
    """Verify BUILDING status is set (captured via store side-effect)."""
    job = await _make_job(store)
    model_dir = Path(test_settings.models_dir) / "Llama-3.2-1B"
    observed = []

    original_update = store.update_status

    async def tracking_update(job_id, status):
        observed.append(status)
        return await original_update(job_id, status)

    store.update_status = tracking_update
    _patch_all(mocker, model_dir)

    await pipeline.run(job, quant_request, store, test_settings)

    assert JobStatus.BUILDING in observed
    assert JobStatus.DOWNLOADING in observed
    assert JobStatus.CONVERTING in observed
    assert JobStatus.QUANTIZING in observed
    assert JobStatus.UPLOADING in observed
    assert JobStatus.DONE in observed


# ---------------------------------------------------------------------------
# Failure cases — each service error → FAILED
# ---------------------------------------------------------------------------

async def test_invalid_credentials_transitions_to_failed(mocker, store, quant_request, test_settings):
    job = await _make_job(store)
    mocker.patch(
        "app.workers.pipeline.validate_credentials",
        side_effect=InvalidCredentialsError("bad token"),
    )

    await pipeline.run(job, quant_request, store, test_settings)

    assert job.status == JobStatus.FAILED


async def test_invalid_credentials_sets_error_message(mocker, store, quant_request, test_settings):
    job = await _make_job(store)
    mocker.patch(
        "app.workers.pipeline.validate_credentials",
        side_effect=InvalidCredentialsError("bad token"),
    )

    await pipeline.run(job, quant_request, store, test_settings)

    assert job.error is not None
    assert "bad token" in job.error


async def test_build_failure_transitions_to_failed(mocker, store, quant_request, test_settings):
    job = await _make_job(store)
    mocker.patch("app.workers.pipeline.validate_credentials", return_value=True)
    mocker.patch(
        "app.workers.pipeline.clone_or_update_llama_cpp",
        side_effect=BuildError("git clone failed"),
    )

    await pipeline.run(job, quant_request, store, test_settings)

    assert job.status == JobStatus.FAILED


async def test_download_failure_transitions_to_failed(mocker, store, quant_request, test_settings, tmp_path):
    job = await _make_job(store)
    mocker.patch("app.workers.pipeline.validate_credentials", return_value=True)
    mocker.patch("app.workers.pipeline.clone_or_update_llama_cpp")
    mocker.patch("app.workers.pipeline.build_llama_cpp")
    mocker.patch("app.workers.pipeline.get_quantize_binary", return_value="/mock/bin")
    mocker.patch("app.workers.pipeline.setup_venv", return_value="/mock/python")
    mocker.patch("app.workers.pipeline.create_output_repo", return_value="u/r-GGUF")
    mocker.patch(
        "app.workers.pipeline.clone_model",
        side_effect=DownloadError("network timeout"),
    )

    await pipeline.run(job, quant_request, store, test_settings)

    assert job.status == JobStatus.FAILED


async def test_conversion_failure_transitions_to_failed(mocker, store, quant_request, test_settings, tmp_path):
    job = await _make_job(store)
    model_dir = Path(test_settings.models_dir) / "Llama-3.2-1B"
    mocker.patch("app.workers.pipeline.validate_credentials", return_value=True)
    mocker.patch("app.workers.pipeline.clone_or_update_llama_cpp")
    mocker.patch("app.workers.pipeline.build_llama_cpp")
    mocker.patch("app.workers.pipeline.get_quantize_binary", return_value="/mock/bin")
    mocker.patch("app.workers.pipeline.setup_venv", return_value="/mock/python")
    mocker.patch("app.workers.pipeline.create_output_repo", return_value="u/r-GGUF")
    mocker.patch("app.workers.pipeline.clone_model", return_value=model_dir)
    mocker.patch("app.workers.pipeline.is_multimodal", return_value=False)
    mocker.patch(
        "app.workers.pipeline.convert_to_gguf",
        side_effect=ConversionError("bf16 conversion failed"),
    )

    await pipeline.run(job, quant_request, store, test_settings)

    assert job.status == JobStatus.FAILED


async def test_upload_failure_transitions_to_failed(mocker, store, quant_request, test_settings, tmp_path):
    job = await _make_job(store)
    model_dir = Path(test_settings.models_dir) / "Llama-3.2-1B"
    mocker.patch("app.workers.pipeline.validate_credentials", return_value=True)
    mocker.patch("app.workers.pipeline.clone_or_update_llama_cpp")
    mocker.patch("app.workers.pipeline.build_llama_cpp")
    mocker.patch("app.workers.pipeline.get_quantize_binary", return_value="/mock/bin")
    mocker.patch("app.workers.pipeline.setup_venv", return_value="/mock/python")
    mocker.patch("app.workers.pipeline.create_output_repo", return_value="u/r-GGUF")
    mocker.patch("app.workers.pipeline.clone_model", return_value=model_dir)
    mocker.patch("app.workers.pipeline.is_multimodal", return_value=False)
    mocker.patch("app.workers.pipeline.convert_to_gguf")
    mocker.patch("app.workers.pipeline.clean_source_files", return_value=0)
    mocker.patch("app.workers.pipeline.quantize_single", return_value="ok")
    mocker.patch(
        "app.workers.pipeline.upload_file",
        side_effect=UploadError("S3 connection reset"),
    )

    await pipeline.run(job, quant_request, store, test_settings)

    assert job.status == JobStatus.FAILED


async def test_any_failure_stores_error_in_store(mocker, store, quant_request, test_settings):
    job = await _make_job(store)
    mocker.patch(
        "app.workers.pipeline.validate_credentials",
        side_effect=InvalidCredentialsError("expired token"),
    )

    await pipeline.run(job, quant_request, store, test_settings)

    stored = await store.get(job.job_id)
    assert stored.error is not None


async def test_pipeline_never_raises_on_error(mocker, store, quant_request, test_settings):
    job = await _make_job(store)
    mocker.patch(
        "app.workers.pipeline.validate_credentials",
        side_effect=RuntimeError("unexpected crash"),
    )
    # Must not propagate any exception
    await pipeline.run(job, quant_request, store, test_settings)
    assert job.status == JobStatus.FAILED


# ---------------------------------------------------------------------------
# quantize_single "failed" is non-fatal
# ---------------------------------------------------------------------------

async def test_quantize_failure_does_not_stop_pipeline(mocker, store, quant_request, test_settings, tmp_path):
    """A single variant failing should not fail the whole job."""
    multi_req = QuantizeRequest(
        hf_repo="meta-llama/Llama-3.2-1B",
        hf_username="testuser",
        hf_token="hf_testtoken",
        quant_types=["Q4_K_M", "Q5_K_M"],
    )
    job = await _make_job(store)
    model_dir = Path(test_settings.models_dir) / "Llama-3.2-1B"
    _patch_all(mocker, model_dir)
    # First type fails, second succeeds
    mocker.patch(
        "app.workers.pipeline.quantize_single",
        side_effect=["failed", "ok"],
    )

    await pipeline.run(job, multi_req, store, test_settings)

    assert job.status == JobStatus.DONE


async def test_quantize_failure_is_logged(mocker, store, quant_request, test_settings, tmp_path):
    job = await _make_job(store)
    model_dir = Path(test_settings.models_dir) / "Llama-3.2-1B"
    _patch_all(mocker, model_dir)
    mocker.patch("app.workers.pipeline.quantize_single", return_value="failed")

    await pipeline.run(job, quant_request, store, test_settings)

    assert any("failed" in line for line in job.progress)


# ---------------------------------------------------------------------------
# Multimodal path
# ---------------------------------------------------------------------------

async def test_multimodal_model_triggers_mmproj_creation(mocker, store, quant_request, test_settings, tmp_path):
    job = await _make_job(store)
    model_dir = Path(test_settings.models_dir) / "Llama-3.2-1B"
    _patch_all(mocker, model_dir)
    mocker.patch("app.workers.pipeline.is_multimodal", return_value=True)
    mock_mmproj = mocker.patch("app.workers.pipeline.create_mmproj")

    await pipeline.run(job, quant_request, store, test_settings)

    mock_mmproj.assert_called_once()


async def test_non_multimodal_model_skips_mmproj(mocker, store, quant_request, test_settings, tmp_path):
    job = await _make_job(store)
    model_dir = Path(test_settings.models_dir) / "Llama-3.2-1B"
    _patch_all(mocker, model_dir)
    mocker.patch("app.workers.pipeline.is_multimodal", return_value=False)
    mock_mmproj = mocker.patch("app.workers.pipeline.create_mmproj")

    await pipeline.run(job, quant_request, store, test_settings)

    mock_mmproj.assert_not_called()


# ---------------------------------------------------------------------------
# Service call verification
# ---------------------------------------------------------------------------

async def test_pipeline_validates_credentials_with_token(mocker, store, quant_request, test_settings, tmp_path):
    job = await _make_job(store)
    model_dir = Path(test_settings.models_dir) / "Llama-3.2-1B"
    _patch_all(mocker, model_dir)
    mock_validate = mocker.patch("app.workers.pipeline.validate_credentials", return_value=True)

    await pipeline.run(job, quant_request, store, test_settings)

    mock_validate.assert_called_once_with("hf_testtoken")


async def test_pipeline_calls_quantize_for_each_requested_type(mocker, store, test_settings, tmp_path):
    multi_req = QuantizeRequest(
        hf_repo="meta-llama/Llama-3.2-1B",
        hf_username="testuser",
        hf_token="hf_testtoken",
        quant_types=["Q4_K_M", "Q5_K_M", "Q8_0"],
    )
    job = await store.create("meta-llama/Llama-3.2-1B", "testuser")
    model_dir = Path(test_settings.models_dir) / "Llama-3.2-1B"
    _patch_all(mocker, model_dir)
    mock_quant = mocker.patch("app.workers.pipeline.quantize_single", return_value="ok")

    await pipeline.run(job, multi_req, store, test_settings)

    assert mock_quant.call_count == 3


async def test_pipeline_uses_settings_threads_when_request_threads_is_none(
    mocker, store, quant_request, test_settings, tmp_path
):
    job = await _make_job(store)
    model_dir = Path(test_settings.models_dir) / "Llama-3.2-1B"
    _patch_all(mocker, model_dir)
    mock_quant = mocker.patch("app.workers.pipeline.quantize_single", return_value="ok")

    await pipeline.run(job, quant_request, store, test_settings)

    _, kwargs = mock_quant.call_args
    assert kwargs.get("threads") == test_settings.threads or mock_quant.call_args[0][4] == test_settings.threads
