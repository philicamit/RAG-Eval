"""Reusable RAG evaluation harness built on DeepEval plus lexical retrieval metrics."""

from .adapter import LocalRAGAdapter, RAGObservation, RAGSystem
from .dataset import GoldenCase, load_golden_dataset

__all__ = [
    "GoldenCase",
    "LocalRAGAdapter",
    "RAGObservation",
    "RAGSystem",
    "load_golden_dataset",
]
