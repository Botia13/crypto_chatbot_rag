"""Deterministic evidence scoring, with no model or application imports."""
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

CLAIM_EVIDENCE_MATCH_THRESHOLD = 0.85
EVIDENCE_METRIC_VERSION = "adjacent-pair-v1"


def load_evaluation_questions(path=None):
    from RAG_model.ingestion.config import PROJECT_ROOT, questions_file
    csv_path = Path(path) if path is not None else PROJECT_ROOT / "data/rag_evaluation" / questions_file
    questions = pd.read_csv(csv_path)
    for field in ("required_claims", "reference_evidence"):
        questions[field] = questions[field].apply(json.loads)
    return questions


def normalize_evidence_tokens(text):
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    return re.sub(r"[^\w]+", " ", text).split()


def normalize_evidence_accession(value):
    return re.sub(r"\D", "", str(value or ""))


def evidence_passage_coverage(supporting_passage, chunk_text):
    evidence = normalize_evidence_tokens(supporting_passage)
    if not evidence:
        return 0.0
    matcher = SequenceMatcher(None, evidence, normalize_evidence_tokens(chunk_text), autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(evidence)


def narrative_position(chunk):
    """Require the index's unambiguous accession*section*text-N identity."""
    match = re.fullmatch(r"([^*]+)\*([^*]+)\*text-(\d+)", str(chunk.get("chunk_id", "")))
    if not match or chunk.get("chunk_type", "text") != "text":
        return None
    accession, section, index = match.groups()
    if normalize_evidence_accession(accession) != normalize_evidence_accession(chunk.get("accession_number")):
        return None
    if chunk.get("section_key") and chunk["section_key"] != section:
        return None
    return normalize_evidence_accession(accession), section, int(index)


def best_evidence_match(evidence, chunks, match_threshold=CLAIM_EVIDENCE_MATCH_THRESHOLD, *, allow_adjacent=True):
    accession = normalize_evidence_accession(evidence["accession"])
    eligible = [(rank, c) for rank, c in enumerate(chunks, 1)
                if accession and normalize_evidence_accession(c.get("accession_number")) == accession]
    best = {"coverage": 0.0, "matched": False, "chunk_ids": [], "ranks": [], "support_type": None}

    def consider(items, kind):
        text = "\n".join(str(c.get("chunk_text") or "") for _, c in items)
        coverage = evidence_passage_coverage(evidence["supporting_passage"], text)
        if coverage > best["coverage"]:
            best.update(coverage=coverage, matched=coverage >= match_threshold,
                        chunk_ids=[c.get("chunk_id") for _, c in items],
                        ranks=[rank for rank, _ in items], support_type=kind)

    for item in eligible:
        consider([item], "single")
    if allow_adjacent and not best["matched"]:
        positions = {pos: item for item in eligible if (pos := narrative_position(item[1])) is not None}
        for (acc, section, index), first in positions.items():
            second = positions.get((acc, section, index + 1))
            if second:
                consider([first, second], "adjacent_pair")
    return best


def evaluate_claim_evidence(required_claims, reference_evidence, retrieved_chunks,
                            match_threshold=CLAIM_EVIDENCE_MATCH_THRESHOLD, *, allow_adjacent=True):
    evidence_rows = []
    for number, evidence in enumerate(reference_evidence, 1):
        evidence_rows.append({"evidence_number": number, "claim_index": int(evidence["claim_index"]),
                              "expected_accession": evidence["accession"],
                              **best_evidence_match(evidence, retrieved_chunks, match_threshold,
                                                    allow_adjacent=allow_adjacent)})
    claim_rows = []
    for index, claim in enumerate(required_claims, 1):
        assigned = [e for e in evidence_rows if e["claim_index"] == index]
        claim_rows.append({"claim_index": index, "claim": claim,
                           "required_evidence_count": len(assigned),
                           "claim_complete": bool(assigned) and all(e["matched"] for e in assigned)})
    return claim_rows, evidence_rows


def calculate_claim_evidence_recall(required_claims, reference_evidence, retrieved_chunks,
                                    match_threshold=CLAIM_EVIDENCE_MATCH_THRESHOLD, *, allow_adjacent=True):
    if not required_claims:
        return None
    claims, _ = evaluate_claim_evidence(required_claims, reference_evidence, retrieved_chunks,
                                        match_threshold, allow_adjacent=allow_adjacent)
    return sum(row["claim_complete"] for row in claims) / len(claims)
