import pytest

from app.config import Settings


def test_default_models_dir():
    s = Settings()
    assert s.models_dir == "models"


def test_default_llama_cpp_dir():
    s = Settings()
    assert s.llama_cpp_dir == "llama.cpp"


def test_default_llama_cpp_url_points_to_ggml_org():
    s = Settings()
    assert "ggml-org/llama.cpp" in s.llama_cpp_url


def test_default_threads_is_positive():
    s = Settings()
    assert s.threads >= 1


def test_default_venv_path():
    s = Settings()
    assert s.venv_path == "venv"


def test_default_build_dir():
    s = Settings()
    assert s.build_dir == "build/bin"


def test_env_var_overrides_models_dir(monkeypatch):
    monkeypatch.setenv("QUANT_MODELS_DIR", "/custom/models")
    s = Settings()
    assert s.models_dir == "/custom/models"


def test_env_var_overrides_threads(monkeypatch):
    monkeypatch.setenv("QUANT_THREADS", "16")
    s = Settings()
    assert s.threads == 16


def test_env_var_overrides_llama_cpp_dir(monkeypatch):
    monkeypatch.setenv("QUANT_LLAMA_CPP_DIR", "/opt/llama.cpp")
    s = Settings()
    assert s.llama_cpp_dir == "/opt/llama.cpp"


def test_default_host():
    s = Settings()
    assert s.host == "0.0.0.0"


def test_default_port():
    s = Settings()
    assert s.port == 8000


def test_env_var_overrides_host(monkeypatch):
    monkeypatch.setenv("QUANT_HOST", "127.0.0.1")
    s = Settings()
    assert s.host == "127.0.0.1"


def test_env_var_overrides_port(monkeypatch):
    monkeypatch.setenv("QUANT_PORT", "9000")
    s = Settings()
    assert s.port == 9000
