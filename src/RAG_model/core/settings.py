from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
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

    openrouter_api_key: SecretStr = Field(repr=False)

    qdrant_packaged_path: Path = (
        PROJECT_ROOT / "artifacts" / "qdrant_runtime"
    )

    qdrant_collection: str = (
        "sec_filings__chunk-500__overlap-120__encoding-cl100k_base__"
        "embedding-openai-text-embedding-3-small"
    )

    max_question_length: int = Field(default=2000, ge=1, le=20_000)
    max_history_messages: int = Field(default=10, ge=0, le=100)
    max_concurrent_requests: int = Field(default=3, ge=1, le=100)
    max_generation_tokens: int = Field(default=700, ge=1, le=10_000)
    max_query_rewrite_tokens: int = Field(default=128, ge=1, le=1_000)
    request_timeout_seconds: float = Field(default=60.0, gt=0, le=600)


@lru_cache
def get_settings() -> Settings:
    return Settings()
