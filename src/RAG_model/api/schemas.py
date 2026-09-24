from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)


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