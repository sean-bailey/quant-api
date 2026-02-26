import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class BuildError(Exception):
    """Raised when cloning, building, or configuring llama.cpp fails."""


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> None:
    """Run *cmd* as a subprocess; raise BuildError on non-zero exit."""
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = (
            result.stderr.decode(errors="replace")
            if isinstance(result.stderr, bytes)
            else str(result.stderr)
        )
        raise BuildError(
            f"Command '{cmd[0]}' failed (exit {result.returncode}): {stderr}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clone_or_update_llama_cpp(dest: str, url: str) -> None:
    """Clone *url* into *dest*, or pull if *dest* is already a git repo."""
    dest_path = Path(dest)
    if (dest_path / ".git").exists():
        _run(["git", "-C", dest, "pull"])
    else:
        _run(["git", "clone", url, dest])


def build_llama_cpp(llama_dir: str, build_dir: str) -> None:
    """Configure and compile llama.cpp with CMake (Release build).

    Runs two commands:
    1. ``cmake -B <build_dir> -S <llama_dir> -DCMAKE_BUILD_TYPE=Release``
    2. ``cmake --build <build_dir> --config Release --parallel``
    """
    _run([
        "cmake",
        "-B", build_dir,
        "-S", llama_dir,
        "-DCMAKE_BUILD_TYPE=Release",
    ])
    _run([
        "cmake",
        "--build", build_dir,
        "--config", "Release",
        "--parallel",
    ])


def get_quantize_binary(build_dir: str) -> str:
    """Return the path to the ``llama-quantize`` binary inside *build_dir*.

    Checks candidate locations in priority order so that platform-specific
    CMake output layouts (Windows Release sub-directory, plain Linux path)
    are all handled without requiring a ``sys.platform`` branch here.

    Raises BuildError if the binary cannot be found.
    """
    build_path = Path(build_dir)
    candidates = [
        build_path / "Release" / "llama-quantize.exe",        # Windows CMake Release (flat)
        build_path / "bin" / "Release" / "llama-quantize.exe",  # Windows CMake bin/Release subdir
        build_path / "llama-quantize.exe",                     # Windows flat
        build_path / "llama-quantize",                         # Linux / macOS
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise BuildError(
        f"llama-quantize binary not found in {build_dir}. "
        "Run build_llama_cpp() first and ensure the build succeeded."
    )


def setup_venv(llama_dir: str, venv_path: str) -> str:
    """Create a Python venv at *venv_path* and install llama.cpp dependencies.

    Steps:
    1. Create the venv with ``sys.executable -m venv <venv_path>``
    2. Install ``<llama_dir>/requirements.txt`` if present
    3. Install ``huggingface_hub``

    Returns the path to the Python interpreter inside the created venv.
    """
    # 1. Create the venv
    _run([sys.executable, "-m", "venv", venv_path])

    # Resolve pip / python paths relative to the venv
    venv = Path(venv_path)
    if sys.platform == "win32":
        pip = str(venv / "Scripts" / "pip")
        python = str(venv / "Scripts" / "python")
    else:
        pip = str(venv / "bin" / "pip")
        python = str(venv / "bin" / "python")

    # 2. Install llama.cpp requirements if present
    req_file = Path(llama_dir) / "requirements.txt"
    if req_file.exists():
        _run([pip, "install", "-r", str(req_file)])

    # 3. Ensure huggingface_hub is available in the venv
    _run([pip, "install", "huggingface_hub"])

    return python
