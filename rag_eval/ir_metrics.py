from __future__ import annotations

import re

from .adapter import RAGObservation
from .dataset import GoldenCase

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_COVERAGE_HIT_THRESHOLD = 0.35


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _coverage(reference: str, retrieved_texts: list[str]) -> float:
    gold = _tokens(reference)
    if not gold:
        return 0.0
    pooled: set[str] = set()
    for chunk in retrieved_texts:
        pooled |= _tokens(chunk)
    return len(gold & pooled) / len(gold)


def _first_hit_rank(observation: RAGObservation, case: GoldenCase) -> int | None:
    if case.expected_doc_ids:
        wanted = {doc_id.lower() for doc_id in case.expected_doc_ids}
        for rank, doc_id in enumerate(observation.retrieved_ids, start=1):
            if doc_id.lower() in wanted:
                return rank

    needles: list[str] = []
    needles.extend(case.expected_context)
    needles.extend(case.expected_keywords)
    if not needles:
        return None

    for rank, chunk in enumerate(observation.retrieved_texts, start=1):
        chunk_lower = chunk.lower()
        chunk_tokens = _tokens(chunk)
        for needle in needles:
            if not needle.strip():
                continue
            if needle.lower() in chunk_lower:
                return rank
            if _coverage(needle, [chunk]) >= _COVERAGE_HIT_THRESHOLD and chunk_tokens:
                return rank
    return None


def compute_ir_metrics(case: GoldenCase, observation: RAGObservation) -> dict[str, float]:
    """Lexical / ID-based retrieval quality. Complements DeepEval's LLM judges."""
    k = max(len(observation.retrieved_texts), 1)
    if case.unanswerable:
        return {}

    rank = _first_hit_rank(observation, case)
    hit = 1.0 if rank is not None else 0.0
    mrr = (1.0 / rank) if rank else 0.0

    if case.expected_context:
        context_recall = sum(_coverage(ref, observation.retrieved_texts) for ref in case.expected_context) / len(
            case.expected_context
        )
    elif case.expected_keywords:
        context_recall = hit
    else:
        context_recall = 0.0

    if case.expected_keywords:
        pooled = " ".join(observation.retrieved_texts).lower()
        keyword_hits = sum(1 for keyword in case.expected_keywords if keyword.lower() in pooled)
        keyword_recall = keyword_hits / len(case.expected_keywords)
    else:
        keyword_recall = hit

    return {
        "hit_at_k": hit,
        "mrr_at_k": mrr,
        "context_recall_at_k": context_recall,
        "keyword_recall": keyword_recall,
        "k": float(k),
    }


def refusal_score(case: GoldenCase, answer: str) -> dict[str, float] | None:
    if not case.unanswerable:
        return None
    lowered = (answer or "").lower()
    refused = any(marker in lowered for marker in ("i don't know", "i do not know", "not contained", "cannot answer"))
    return {"refusal_accuracy": 1.0 if refused else 0.0}
