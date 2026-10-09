from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from app.rag_pipeline import LatencyMetrics, RAGService


def document_id(doc: Document) -> str:
    metadata = doc.metadata or {}
    for key in ("id", "doc_id", "source"):
        value = metadata.get(key)
        if value:
            return str(value)
    digest = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class RAGObservation:
    """Normalized RAG output so any pipeline can be scored with this harness."""

    question: str
    answer: str
    retrieved_texts: list[str]
    retrieved_ids: list[str]
    latency: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_docs(
        cls,
        question: str,
        answer: str,
        docs: list[Document],
        latency: LatencyMetrics | dict[str, float] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> "RAGObservation":
        from app.rag_pipeline import LatencyMetrics

        if isinstance(latency, LatencyMetrics):
            latency_dict = latency.to_dict()
        else:
            latency_dict = dict(latency or {})
        return cls(
            question=question,
            answer=answer,
            retrieved_texts=[doc.page_content for doc in docs],
            retrieved_ids=[document_id(doc) for doc in docs],
            latency=latency_dict,
            metadata=metadata or {},
        )


class RAGSystem(Protocol):
    """Minimal contract another RAG pipeline must satisfy to reuse this eval pack."""

    def retrieve(self, question: str) -> RAGObservation:
        """Return retrieved chunks. Answer may be empty in retriever-only runs."""

    def answer(self, question: str) -> RAGObservation:
        """Return generated answer plus the context that was actually used."""


class LocalRAGAdapter:
    """Adapter for this repo's RAGService. Swap this class for another stack."""

    def __init__(self, service: RAGService):
        self.service = service

    def retrieve(self, question: str) -> RAGObservation:
        started = time.perf_counter()
        docs = self.service.get_relevant_context(question)
        retrieval_ms = (time.perf_counter() - started) * 1000.0
        return RAGObservation.from_docs(
            question=question,
            answer="",
            docs=docs,
            latency={
                "retrieval_ms": retrieval_ms,
                "rerank_ms": 0.0,
                "generation_ms": 0.0,
                "ttft_ms": retrieval_ms,
                "total_ms": retrieval_ms,
            },
        )

    def answer(self, question: str) -> RAGObservation:
        answer, docs, metrics = self.service.answer(question)
        return RAGObservation.from_docs(
            question=question,
            answer=answer,
            docs=docs,
            latency=metrics,
        )
