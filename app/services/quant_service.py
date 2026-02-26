import json
import subprocess
from pathlib import Path
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConversionError(Exception):
    """Raised when HF → GGUF conversion fails."""


class QuantizationError(Exception):
    """Raised for fatal quantization-level errors (not per-type failures)."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All supported quantization types — order mirrors the original shell scripts.
QUANT_TYPES: list[str] = [
    # Standard
    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "F16",
    # K-quants
    "Q2_K", "Q2_K_S", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K",
    # I-quants
    "IQ1_S", "IQ1_M",
    "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
    "IQ4_NL", "IQ4_XS",
    # Ternary
    "TQ1_0", "TQ2_0",
    # BFloat16
    "BF16",
]

# Types that require an importance matrix file to produce useful output.
REQUIRES_IMATRIX: set[str] = {
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "Q2_K_S",
}

# Keys in config.json whose presence signals a multimodal/vision model.
_VISION_CONFIG_KEYS: frozenset[str] = frozenset({
    "vision_config", "visual_config", "image_encoder",
    "vision_tower", "visual_encoder",
})

# README keywords (checked case-insensitively) that suggest a vision model.
_README_VISION_KEYWORDS: frozenset[str] = frozenset({
    "vision", "multimodal", "vlm",
})

# File extensions deleted by clean_source_files.
_SOURCE_EXTENSIONS: frozenset[str] = frozenset({
    ".safetensors", ".bin", ".pt", ".pth",
})


# ---------------------------------------------------------------------------
# Multimodal detection
# ---------------------------------------------------------------------------

def is_multimodal(model_dir: Path) -> bool:
    """Return True if the model in *model_dir* appears to be multimodal/vision.

    Checks (in order):
    1. config.json for known vision-related keys
    2. Presence of preprocessor_config.json
    3. README.md for vision-related keywords (case-insensitive)
    """
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if any(k in config for k in _VISION_CONFIG_KEYS):
                return True
        except (json.JSONDecodeError, OSError):
            pass

    if (model_dir / "preprocessor_config.json").exists():
        return True

    readme_path = model_dir / "README.md"
    if readme_path.exists():
        try:
            text = readme_path.read_text(encoding="utf-8").lower()
            if any(kw in text for kw in _README_VISION_KEYWORDS):
                return True
        except OSError:
            pass

    return False


# ---------------------------------------------------------------------------
# Internal subprocess helper
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> None:
    """Run *cmd*; raise ConversionError on non-zero exit."""
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = (
            result.stderr.decode(errors="replace")
            if isinstance(result.stderr, bytes)
            else str(result.stderr)
        )
        raise ConversionError(
            f"Command '{cmd[0]}' failed (exit {result.returncode}): {stderr}"
        )


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_to_gguf(
    python: str,
    llama_dir: str,
    model_dir: str,
    out_file: str,
) -> None:
    """Convert a HuggingFace model directory to GGUF BF16 format.

    Calls ``convert_hf_to_gguf.py --outtype bf16 --outfile <out_file>``.
    Raises ConversionError on failure.
    """
    script = str(Path(llama_dir) / "convert_hf_to_gguf.py")
    _run([python, script, model_dir, "--outtype", "bf16", "--outfile", out_file])


def create_mmproj(
    python: str,
    llama_dir: str,
    model_dir: str,
    out_file: str,
) -> None:
    """Create the vision-projector (mmproj) GGUF for multimodal models.

    Calls ``convert_hf_to_gguf.py --mmproj --outfile <out_file>``.
    Raises ConversionError on failure.
    """
    script = str(Path(llama_dir) / "convert_hf_to_gguf.py")
    _run([python, script, "--mmproj", model_dir, "--outfile", out_file])


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_single(
    binary: str,
    base_model: str,
    output: str,
    quant_type: str,
    threads: int,
    imatrix: Optional[str] = None,
) -> Literal["ok", "skipped", "failed"]:
    """Quantize *base_model* to *quant_type* and write to *output*.

    Returns:
    - ``"skipped"`` — output already exists, or the type requires an
      importance matrix that was not supplied.
    - ``"ok"``      — quantization succeeded.
    - ``"failed"``  — llama-quantize exited non-zero.  Does **not** raise,
      so the pipeline can continue with remaining types.
    """
    if Path(output).exists():
        return "skipped"

    if quant_type in REQUIRES_IMATRIX and imatrix is None:
        return "skipped"

    cmd = [binary]
    if imatrix is not None:
        cmd += ["--imatrix", imatrix]
    cmd += [base_model, output, quant_type, str(threads)]

    result = subprocess.run(cmd, capture_output=True)
    return "ok" if result.returncode == 0 else "failed"


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def clean_source_files(model_dir: Path) -> int:
    """Delete raw model weight files from *model_dir* (non-recursive).

    Removes files with extensions: ``.safetensors``, ``.bin``, ``.pt``, ``.pth``.
    All other files (config, tokenizer, README, GGUF, …) are preserved.

    Returns the total bytes freed.
    """
    freed = 0
    for path in model_dir.iterdir():
        if path.is_file() and path.suffix in _SOURCE_EXTENSIONS:
            freed += path.stat().st_size
            path.unlink()
    return freed
