import shutil
from pathlib import Path

from app.config import Settings
from app.models.job import Job, JobStatus
from app.models.request import QuantizeRequest
from app.services.build_service import (
    build_llama_cpp,
    clone_or_update_llama_cpp,
    get_quantize_binary,
    setup_venv,
)
from app.services.hf_service import (
    clone_model,
    create_output_repo,
    upload_file,
    validate_credentials,
)
from app.services.quant_service import (
    QUANT_TYPES,
    clean_source_files,
    convert_to_gguf,
    create_mmproj,
    is_multimodal,
    quantize_single,
)
from app.storage.job_store import JobStore


async def run(
    job: Job,
    request: QuantizeRequest,
    store: JobStore,
    settings: Settings,
) -> None:
    """End-to-end quantization pipeline.

    Mutates *job* in-place and mirrors every state change into *store* so
    the REST API always reflects the current status.

    Never raises — all exceptions are caught, written to ``job.error``, and
    the job transitions to FAILED.
    """
    token = request.hf_token.get_secret_value()
    quant_types = request.quant_types or QUANT_TYPES
    threads = request.threads or settings.threads

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def log(msg: str) -> None:
        job.progress.append(msg)
        await store.append_log(job.job_id, msg)

    async def transition(status: JobStatus) -> None:
        job.status = status
        await store.update_status(job.job_id, status)

    async def fail(err: Exception) -> None:
        msg = str(err)
        job.error = msg
        await store.set_error(job.job_id, msg)
        job.status = JobStatus.FAILED
        await store.update_status(job.job_id, JobStatus.FAILED)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    try:
        # Step 1 — Validate credentials
        await log("[1/9] Validating HuggingFace credentials")
        validate_credentials(token)

        # Step 2 — Clone / build llama.cpp
        await transition(JobStatus.BUILDING)
        await log("[2/9] Setting up llama.cpp")
        clone_or_update_llama_cpp(settings.llama_cpp_dir, settings.llama_cpp_url)
        build_llama_cpp(settings.llama_cpp_dir, settings.build_dir)
        binary = get_quantize_binary(settings.build_dir)

        # Step 3 — Python venv
        await log("[3/9] Setting up Python venv")
        python = setup_venv(settings.llama_cpp_dir, settings.venv_path)

        # Step 4 — Create destination repo on HuggingFace
        await log("[4/9] Creating output repo on HuggingFace")
        repo_id = create_output_repo(request.hf_repo, request.hf_username, token)
        job.output_repo = repo_id
        await store.set_output_repo(job.job_id, repo_id)

        # Step 5 — Download source model
        await transition(JobStatus.DOWNLOADING)
        await log(f"[5/9] Downloading {request.hf_repo}")
        models_dir = Path(settings.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        model_dir = clone_model(request.hf_repo, models_dir, token)

        # Step 6 — Convert to GGUF BF16 + optional mmproj
        await transition(JobStatus.CONVERTING)
        await log("[6/9] Converting to GGUF BF16")
        model_name = request.hf_repo.split("/", 1)[1]
        base_gguf = models_dir / f"{model_name}-BF16.gguf"
        multimodal = is_multimodal(model_dir)
        convert_to_gguf(python, settings.llama_cpp_dir, str(model_dir), str(base_gguf))

        mmproj_file: Path | None = None
        if multimodal:
            await log("[6/9] Multimodal model detected — creating mmproj")
            mmproj_file = models_dir / f"{model_name}-mmproj.gguf"
            create_mmproj(python, settings.llama_cpp_dir, str(model_dir), str(mmproj_file))

        freed = clean_source_files(model_dir)
        await log(f"[6/9] Freed {freed / (1024 ** 3):.2f} GB of source weights")

        # Step 7 — Quantize + upload each variant
        await transition(JobStatus.QUANTIZING)
        await log(f"[7/9] Quantizing {len(quant_types)} variant(s)")

        for qt in quant_types:
            out_file = models_dir / f"{model_name}-{qt}.gguf"
            result = quantize_single(
                binary=binary,
                base_model=str(base_gguf),
                output=str(out_file),
                quant_type=qt,
                threads=threads,
                imatrix=request.imatrix,
            )
            await log(f"  {qt}: {result}")
            if result == "ok":
                upload_file(out_file, repo_id, out_file.name, token)
                await log(f"  {qt}: uploaded")
                if not request.keep_files:
                    out_file.unlink(missing_ok=True)

        # Step 8 — Upload base model, mmproj, README
        await transition(JobStatus.UPLOADING)
        await log("[8/9] Uploading base model and README")

        if base_gguf.exists():
            upload_file(base_gguf, repo_id, base_gguf.name, token)
            if not request.keep_files:
                base_gguf.unlink(missing_ok=True)

        if mmproj_file and mmproj_file.exists():
            upload_file(mmproj_file, repo_id, mmproj_file.name, token)

        readme = model_dir / "README.md"
        if readme.exists():
            upload_file(readme, repo_id, "README.md", token)

        # Step 9 — Optional local cleanup
        if not request.keep_files:
            await log("[9/9] Cleaning up local files")
            if model_dir.exists():
                shutil.rmtree(str(model_dir), ignore_errors=True)

        await transition(JobStatus.DONE)
        await log("Pipeline complete!")

    except Exception as exc:  # noqa: BLE001
        await fail(exc)
