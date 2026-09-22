from types import SimpleNamespace

import pandas as pd
import pytest

import RAG_model.answer.answer as answer_module
from RAG_model.evaluation import evaluation
from RAG_model.ingestion.config import BASELINE_RUN_CONFIG
from RAG_model.ingestion.embedding import create_embeddings
from RAG_model.ingestion.vector_database import index_is_ready


class RecordingEmbeddingClient:
    def __init__(self):
        self.inputs = []
        self.embeddings = self

    def create(self, *, model, input):
        self.inputs.extend(input)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index), 1.0])
                for index, _ in enumerate(input)
            ]
        )


def test_document_metadata_is_included_in_embedding_input():
    client = RecordingEmbeddingClient()
    chunk = {
        "chunk_id": "accession*item-1*text-0000",
        "chunk_text": "The sponsor fee accrues daily.",
        "ticker": "FETH",
        "company_name": "Fidelity Ethereum Fund",
        "form_type": "10-K",
        "filing_date": "2025-03-20",
        "period_end": "2024-12-31",
        "accession_number": "0000950170-25-039374",
        "chunk_title": "Business",
        "section_key": "item-1",
        "chunk_type": "text",
        "table_title": None,
    }

    result = create_embeddings(
        [chunk], "embedding-model", 10, "v1", embedding_client=client
    )

    embedded_text = client.inputs[0]
    assert "Ticker: FETH" in embedded_text
    assert "Company: Fidelity Ethereum Fund" in embedded_text
    assert "Accession number: 0000950170-25-039374" in embedded_text
    assert "Content:\nThe sponsor fee accrues daily." in embedded_text
    assert result["embedded_chunks"][0]["chunk_text"] == chunk["chunk_text"]


def test_document_metrics_use_unique_document_rank():
    ranked_documents = evaluation.unique_ranked_documents(
        ["doc-a", "doc-a", " DOC-B ", "doc-b"]
    )

    assert ranked_documents == ["doc-a", "doc-b"]
    assert evaluation.reciprocal_rank(ranked_documents, "doc-b") == 0.5


def test_multi_document_metrics_require_and_rank_every_expected_document():
    ranked_documents = ["doc-a", "doc-x", "doc-b"]
    expected_documents = evaluation.parse_expected_documents("doc-a; doc-b")

    assert expected_documents == ["doc-a", "doc-b"]
    assert evaluation.mean_reciprocal_rank(
        ranked_documents, expected_documents
    ) == (1.0 + (1 / 3)) / 2


def test_retrieval_only_skips_generation(monkeypatch):
    retrieved = {
        "User Question": "question",
        "Retrieved Chunk texts": [
            {
                "chunk_id": "chunk-1",
                "accession_number": " DOC-1 ",
                "score": 0.9,
            }
        ],
        "Similarity Scores": [{"chunk_id": "chunk-1", "score": 0.9}],
        "Latency": {
            "retrieval": 5.0,
            "prompt_building": 0.0,
            "generation": 0.0,
            "total": 5.0,
        },
        "Token_usage": {
            "embedding_input_tokens": 2,
            "embedding_total_tokens": 2,
            "generation_input_tokens": 0,
            "generation_output_tokens": 0,
            "generation_total_tokens": 0,
            "rag_total_tokens": 2,
        },
    }
    monkeypatch.setattr(evaluation, "retrieve", lambda *args, **kwargs: retrieved)
    monkeypatch.setattr(
        evaluation,
        "answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("generation must not run")
        ),
    )

    result = evaluation.evaluate_question_deterministic(
        {
            "question_id": 1,
            "question": "question",
            "answerable": True,
            "expected_document": "doc-1",
            "reference_answer": "answer",
            "category": "factual",
            "difficulty": "medium",
            "evaluation_focus": "retrieval",
        },
        {
            "experiment_name": "test",
            "pipeline_version": "v1",
            "retrieval_k": 5,
            "embedding_model": "embedding-model",
            "generation_model": "generation-model",
            "prompt_version": "v1",
        },
        qdrant_client=object(),
        retrieval_only=True,
    )

    assert result["document_hit_at_k"] is True
    assert result["reciprocal_rank"] == 1.0
    assert result["retrieved_accession_numbers"] == ["doc-1"]
    assert result["retrieved_ranked_documents"] == ["doc-1"]
    assert result["answer"] is None
    assert result["citations_valid"] is None
    assert result["token_usage"]["generation_total_tokens"] == 0


def test_multi_document_question_requires_all_documents(monkeypatch):
    retrieved = {
        "Retrieved Chunk texts": [
            {"chunk_id": "a-1", "accession_number": "doc-a", "score": 0.9}
        ],
        "Latency": {
            "retrieval": 1.0,
            "prompt_building": 0.0,
            "generation": 0.0,
            "total": 1.0,
        },
        "Token_usage": {
            "embedding_input_tokens": 1,
            "embedding_total_tokens": 1,
            "generation_input_tokens": 0,
            "generation_output_tokens": 0,
            "generation_total_tokens": 0,
            "rag_total_tokens": 1,
        },
    }
    monkeypatch.setattr(evaluation, "retrieve", lambda *args, **kwargs: retrieved)
    result = evaluation.evaluate_question_deterministic(
        {
            "question_id": 17,
            "question": "Compare A and B",
            "answerable": True,
            "expected_document": "doc-a; doc-b",
            "reference_answer": "comparison",
            "category": "Comparison",
            "difficulty": "hard",
            "evaluation_focus": "multi-document",
        },
        {
            "experiment_name": "test",
            "pipeline_version": "v1",
            "retrieval_k": 5,
            "embedding_model": "embedding-model",
        },
        qdrant_client=object(),
        retrieval_only=True,
    )

    assert result["document_hit_at_k"] is False
    assert result["document_recall_at_k"] == 0.5
    assert result["reciprocal_rank"] == 0.5


def test_vector_index_is_prepared_once_per_collection(monkeypatch):
    calls = []
    monkeypatch.setattr(evaluation, "ingestion", lambda config: calls.append(config))
    evaluation._PREPARED_COLLECTIONS.clear()
    base = {"collection_name": "shared-index"}

    evaluation.prepare_vector_index({**base, "generation_model": "model-a"})
    evaluation.prepare_vector_index({**base, "generation_model": "model-b"})

    assert len(calls) == 1


def test_existing_index_must_match_embedding_input_version():
    class FakeQdrant:
        def collection_exists(self, name):
            return True

        def get_collection(self, name):
            return SimpleNamespace(points_count=10)

        def scroll(self, **kwargs):
            assert kwargs["scroll_filter"] is not None
            return [SimpleNamespace(payload={"embedding_input_version": "old"})], None

    assert index_is_ready(FakeQdrant(), "index", "metadata-v1") is False


def test_all_index_points_must_have_current_embedding_input_version():
    class FakeQdrant:
        def collection_exists(self, name):
            return True

        def get_collection(self, name):
            return SimpleNamespace(points_count=10)

        def scroll(self, **kwargs):
            return [], None

    assert index_is_ready(FakeQdrant(), "index", "metadata-v1") is True


def test_answerable_string_false_is_not_treated_as_true():
    assert evaluation.is_answerable("false") is False
    assert evaluation.is_answerable("TRUE") is True


def test_summary_mrr_and_hit_rate_ignore_unanswerable_questions():
    frame = pd.DataFrame(
        [
            {
                "answerable": True,
                "document_hit_at_k": True,
                "document_recall_at_k": 1.0,
                "reciprocal_rank": 1.0,
                "difficulty": "medium",
            },
            {
                "answerable": True,
                "document_hit_at_k": False,
                "document_recall_at_k": 0.0,
                "reciprocal_rank": 0.0,
                "difficulty": "hard",
            },
            {
                "answerable": False,
                "document_hit_at_k": None,
                "document_recall_at_k": None,
                "reciprocal_rank": None,
                "difficulty": "hard",
            },
        ]
    )
    for column in (
        "ragas_answer_correct", "citations_valid", "ragas_faithfulness",
        "ragas_factual_correctness", "ragas_context_precision",
        "abstention_correct", "total_latency_ms", "rag_total_tokens",
    ):
        frame[column] = None

    summary = evaluation.summarize_evaluation(frame)
    overall = summary.loc[summary["scope"] == "All questions"].iloc[0]

    assert overall["document_hit_rate_at_k"] == 0.5
    assert overall["mean_document_recall_at_k"] == 0.5
    assert overall["mrr"] == 0.5


def test_notebook_config_object_is_accepted():
    class NotebookConfig:
        def get_dictionary(self):
            return {"collection_name": "index", "retrieval_k": 5}

    assert evaluation.normalize_run_config(NotebookConfig()) == {
        "collection_name": "index",
        "retrieval_k": 5,
    }


class ListVector(list):
    def tolist(self):
        return list(self)


class FakeRetrievalEmbeddingClient:
    def __init__(self):
        self.calls = []
        self.embeddings = self

    def create(self, *, model, input):
        self.calls.append({"model": model, "input": input})
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])],
            usage=SimpleNamespace(prompt_tokens=7, total_tokens=7),
        )


class FakeBm25:
    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        yield SimpleNamespace(
            indices=ListVector([1, 2]),
            values=ListVector([0.4, 0.6]),
        )


class FakeQdrant:
    def __init__(self, *, result_groups=(), catalog=()):
        self.result_groups = list(result_groups)
        self.catalog = list(catalog)
        self.query_calls = []
        self.scroll_calls = []

    def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        payloads = self.result_groups[len(self.query_calls) - 1]
        return SimpleNamespace(
            points=[
                SimpleNamespace(payload=payload, score=1.0 - index / 100)
                for index, payload in enumerate(payloads)
            ]
        )

    def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)
        return [SimpleNamespace(payload=payload) for payload in self.catalog], None


def retrieval_chunk(
    chunk_id,
    ticker,
    *,
    form_type="10-K",
    period_end="2024-12-31",
):
    return {
        "chunk_id": chunk_id,
        "accession_number": f"accession-{ticker.lower()}",
        "chunk_text": f"Content for {chunk_id}",
        "ticker": ticker,
        "form_type": form_type,
        "filing_date": "2025-03-20",
        "period_end": period_end,
        "chunk_title": "Business",
        "source_url": "https://www.sec.gov/example",
        "section_type": "item",
        "section_part": None,
        "section_key": "item-1",
        "table_type": None,
        "table_title": None,
    }


@pytest.fixture
def fake_retrieval_models(monkeypatch):
    embedding_client = FakeRetrievalEmbeddingClient()
    bm25 = FakeBm25()
    monkeypatch.setattr(answer_module, "client", embedding_client)
    monkeypatch.setattr(answer_module, "_bm25", bm25)
    return embedding_client, bm25


def test_single_scope_filters_both_prefetches_and_fusion(fake_retrieval_models):
    client = FakeQdrant(result_groups=[[retrieval_chunk("feth-1", "FETH")]])
    question = (
        "What is FETH's Sponsor Fee?\n"
        "Source scope: FETH 10-K for the reporting period ending 2024-12-31."
    )

    chunks, usage = answer_module.fetch_context(
        question, 20, "embedding-model", "collection", client
    )

    assert [item["chunk_id"] for item in chunks] == ["feth-1"]
    assert chunks[0]["form_type"] == "10-K"
    assert chunks[0]["period_end"] == "2024-12-31"
    assert usage == {"input_tokens": 7, "total_tokens": 7}
    assert len(client.query_calls) == 1
    call = client.query_calls[0]
    assert call["limit"] == 20
    assert all(prefetch.limit == 20 for prefetch in call["prefetch"])
    assert all(
        prefetch.filter == call["query_filter"] for prefetch in call["prefetch"]
    )
    serialized_filter = str(call["query_filter"])
    for expected in (
        "ticker", "FETH", "form_type", "10-K", "period_end", "2024-12-31"
    ):
        assert expected in serialized_filter
    assert len(fake_retrieval_models[0].calls) == 1
    assert fake_retrieval_models[1].calls == [question]


def test_multi_ticker_searches_share_embedding_and_merge_round_robin(
    fake_retrieval_models,
):
    client = FakeQdrant(result_groups=[
        [retrieval_chunk("feth-1", "FETH"), retrieval_chunk("feth-2", "FETH")],
        [retrieval_chunk("fbtc-1", "FBTC"), retrieval_chunk("fbtc-2", "FBTC")],
    ])
    question = (
        "Compare FETH and FBTC Sponsor Fees.\n"
        "Source scope: FETH 10-K for the reporting period ending 2024-12-31; "
        "FBTC 10-K for the reporting period ending 2024-12-31."
    )

    chunks, _ = answer_module.fetch_context(
        question, 20, "embedding-model", "collection", client
    )

    assert [item["chunk_id"] for item in chunks] == [
        "feth-1", "fbtc-1", "feth-2", "fbtc-2"
    ]
    assert [call["limit"] for call in client.query_calls] == [10, 10]
    assert len(fake_retrieval_models[0].calls) == 1
    assert fake_retrieval_models[1].calls == [question]


def test_duplicate_chunk_ids_keep_the_first_occurrence(fake_retrieval_models):
    repeated = retrieval_chunk("feth-1", "FETH")
    client = FakeQdrant(result_groups=[[
        repeated,
        {**repeated, "chunk_text": "Duplicate copy"},
        retrieval_chunk("feth-2", "FETH"),
    ]])
    question = (
        "What is FETH's Sponsor Fee?\n"
        "Source scope: FETH 10-K for the reporting period ending 2024-12-31."
    )

    chunks, _ = answer_module.fetch_context(
        question, 20, "embedding-model", "collection", client
    )

    assert [item["chunk_id"] for item in chunks] == ["feth-1", "feth-2"]
    assert chunks[0]["chunk_text"] == "Content for feth-1"


def test_missing_filing_skips_embedding_and_search(fake_retrieval_models):
    client = FakeQdrant(catalog=[retrieval_chunk("historical", "FBTC")])

    chunks, usage = answer_module.fetch_context(
        "What was FBTC's balance sheet for Q2 2028?",
        20,
        "embedding-model",
        "collection",
        client,
    )

    assert chunks == []
    assert usage == {"input_tokens": 0, "total_tokens": 0}
    assert len(client.scroll_calls) == 1
    assert client.query_calls == []
    assert fake_retrieval_models[0].calls == []
    assert fake_retrieval_models[1].calls == []


def test_scope_violation_fails_instead_of_leaking_wrong_filing(
    fake_retrieval_models,
):
    client = FakeQdrant(result_groups=[[retrieval_chunk("wrong-1", "FBTC")]])
    question = (
        "What is FETH's Sponsor Fee?\n"
        "Source scope: FETH 10-K for the reporting period ending 2024-12-31."
    )

    with pytest.raises(AssertionError, match="Filter violation"):
        answer_module.fetch_context(
            question, 20, "embedding-model", "collection", client
        )


def test_raw_retrieval_preserves_order_without_calling_reranker(monkeypatch):
    expected = [
        {"chunk_id": "first", "score": 0.8},
        {"chunk_id": "second", "score": 0.7},
    ]
    monkeypatch.setattr(
        answer_module,
        "fetch_context",
        lambda *args, **kwargs: (expected, {"input_tokens": 2, "total_tokens": 2}),
    )
    monkeypatch.setattr(
        answer_module,
        "rerank",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reranker must remain inactive")
        ),
    )
    config = {
        "candidate_k": 20,
        "retrieval_k": 20,
        "embedding_model": "embedding-model",
        "collection_name": "collection",
        "rerank": False,
    }

    result = answer_module.retrieve("question", config, object())

    assert result["Retrieved Chunk texts"] == expected
    assert result["Similarity Scores"] == [
        {"chunk_id": "first", "hybrid_score": 0.8, "rerank_score": None},
        {"chunk_id": "second", "hybrid_score": 0.7, "rerank_score": None},
    ]


def test_baseline_uses_the_existing_experiment_collection():
    assert BASELINE_RUN_CONFIG["collection_name"] == (
        "sec_filings__chunk-500__overlap-120__encoding-cl100k_base__"
        "embedding-openai-text-embedding-3-small"
    )
