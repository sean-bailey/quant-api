import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QUANT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    models_dir: str = "models"
    llama_cpp_url: str = "https://github.com/ggml-org/llama.cpp.git"
    llama_cpp_dir: str = "llama.cpp"
    build_dir: str = "build/bin"
    venv_path: str = "venv"
    threads: int = os.cpu_count() or 4
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
