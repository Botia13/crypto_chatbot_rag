from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"

    openrouter_api_key: str = Field(repr=False)

    qdrant_packaged_path: Path = (
        PROJECT_ROOT / "artifacts" / "qdrant_runtime"
    )

    qdrant_collection: str = (
        "sec_filings__chunk-500__overlap-120__encoding-cl100k_base__"
        "embedding-openai-text-embedding-3-small"
    )

    max_question_length: int = 2000
    max_history_messages: int = 10
    max_concurrent_requests: int = 3
    max_generation_tokens: int = 700
    request_timeout_seconds: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()