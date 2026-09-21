"""Pure deterministic result metrics; no model, client, or ingestion imports."""
import re
import pandas as pd
from RAG_model.model_analysis_notebooks.utils.evidence import evaluate_claim_evidence

ABSTENTION = "I could not find enough evidence in the retrieved SEC filings."


def score_answer(question, chunks, answer_text):
    """Score an already generated answer; never retrieve or use gold in a prompt."""
    answerable = is_answerable(question['answerable'])
    documents = unique_ranked_documents(c.get('accession_number') for c in chunks)
    expected = parse_expected_documents(question['expected_document']) if answerable else []
    matched = set(expected) & set(documents)
    claims, evidence = evaluate_claim_evidence(
        question.get('required_claims', []), question.get('reference_evidence', []), chunks
    ) if answerable else ([], [])
    legacy, legacy_evidence = evaluate_claim_evidence(
        question.get('required_claims', []), question.get('reference_evidence', []),
        chunks, allow_adjacent=False
    ) if answerable else ([], [])
    citations = re.findall(r'\[SOURCE:\s*([^\]]+)\]', answer_text or '')
    ids = {c['chunk_id'] for c in chunks}
    invalid = [c for c in citations if c not in ids]
    abstained = answer_text.strip() == ABSTENTION
    return {
        'document_hit_at_k': set(expected).issubset(documents) if answerable else None,
        'document_recall_at_k': len(matched)/len(set(expected)) if expected else None,
        'reciprocal_rank': mean_reciprocal_rank(documents, expected) if answerable else None,
        'required_claim_count': len(claims),
        'supported_claim_count': sum(c['claim_complete'] for c in claims),
        'legacy_supported_claim_count': sum(c['claim_complete'] for c in legacy),
        'claim_evidence_recall_at_k': sum(c['claim_complete'] for c in claims)/len(claims) if claims else None,
        'legacy_claim_evidence_recall_at_k': sum(c['claim_complete'] for c in legacy)/len(legacy) if legacy else None,
        'citations': citations, 'invalid_citations': invalid,
        'citations_valid': not invalid and (bool(citations) or not answerable),
        'abstention_correct': abstained if not answerable else None,
        'incorrect_abstention': abstained if answerable else None,
        'claims': claims, 'evidence': evidence,
        'legacy_claims': legacy, 'legacy_evidence': legacy_evidence,
    }

def reciprocal_rank (retrieved_accession_numbers, expected_accession_number):
    """ 
    Return 1/rank for the first relevant retrieved result.
    Rank starts at 1, return 0 if it was not retrieved
    """
    
    for rank, accesion_number in enumerate(retrieved_accession_numbers,start=1):
        if accesion_number==expected_accession_number:
            return 1/rank

    return 0.0


def mean_reciprocal_rank(
    retrieved_accession_numbers: list[str],
    expected_accession_numbers: list[str],
) -> float:
    """Average reciprocal rank across every document required by a question."""
    if not expected_accession_numbers:
        return 0.0
    return sum(
        reciprocal_rank(retrieved_accession_numbers, expected)
        for expected in expected_accession_numbers
    ) / len(expected_accession_numbers)


def normalize_accession_number(value) -> str:
    """Normalize IDs read from CSV/Qdrant before metric comparison."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def parse_expected_documents(value) -> list[str]:
    """Parse one or more semicolon-delimited expected accession numbers."""
    if isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        if value is None or pd.isna(value):
            return []
        raw_values = str(value).split(";")
    return [
        normalized
        for item in raw_values
        if (normalized := normalize_accession_number(item))
    ]


def unique_ranked_documents(retrieved_accession_numbers) -> list[str]:
    """Collapse repeated chunks while preserving document retrieval order."""
    return list(dict.fromkeys(
        normalized
        for value in retrieved_accession_numbers
        if (normalized := normalize_accession_number(value))
    ))


def is_answerable(value) -> bool:
    """Convert CSV and Python boolean representations without truthy strings."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
        raise ValueError(f"Unknown answerable value: {value!r}")
    if value is None or pd.isna(value):
        return False
    return bool(value)
