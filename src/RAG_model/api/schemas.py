from typing import Literal

from pydantic import BaseModel, Field, field_validator

from RAG_model.core.settings import get_settings


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message content must not be empty.")
        if len(value) > get_settings().max_question_length:
            raise ValueError("Message content exceeds the configured limit.")
        return value


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be empty.")
        if len(value) > get_settings().max_question_length:
            raise ValueError("Question exceeds the configured limit.")
        return value

    @field_validator("history")
    @classmethod
    def validate_history(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if len(value) > get_settings().max_history_messages:
            raise ValueError("History exceeds the configured message limit.")
        return value


class Citation(BaseModel):
    chunk_id: str
    ticker: str | None = None
    filing_date: str | None = None
    section: str | None = None
    source_url: str | None = None


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[Citation]
    citation_ids: list[str]
    usage: dict
    timings_ms: dict
    retrieval: dict
    validation: dict
    system: dict
    retrieved_chunks: list[dict]
