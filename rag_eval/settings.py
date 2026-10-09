from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalSettings:
    """Evaluation-only settings. RAG runtime still comes from app.config.Settings."""

    dataset_path: Path
    output_dir: Path
    eval_model: str
    metric_mode: str
    max_cases: int | None
    max_concurrent: int
    retrieval_threshold: float
    generation_threshold: float
    correctness_threshold: float
    min_pass_rate: float
    latency_p95_ms: float | None
    include_reason: bool

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "EvalSettings":
        root = project_root or Path(__file__).resolve().parent.parent
        max_cases_raw = os.getenv("EVAL_MAX_CASES", "").strip()
        latency_raw = os.getenv("EVAL_LATENCY_P95_MS", "").strip()
        return cls(
            dataset_path=Path(
                os.getenv("EVAL_DATASET_PATH", str(root / "rag_eval" / "datasets" / "got_golden.jsonl"))
            ),
            output_dir=Path(os.getenv("EVAL_OUTPUT_DIR", str(root / "rag_eval" / "reports"))),
            eval_model=os.getenv("EVAL_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
            metric_mode=os.getenv("EVAL_METRIC_MODE", "all").strip().lower(),
            max_cases=int(max_cases_raw) if max_cases_raw else None,
            max_concurrent=int(os.getenv("EVAL_MAX_CONCURRENT", "4")),
            retrieval_threshold=float(os.getenv("EVAL_RETRIEVAL_THRESHOLD", "0.6")),
            generation_threshold=float(os.getenv("EVAL_GENERATION_THRESHOLD", "0.6")),
            correctness_threshold=float(os.getenv("EVAL_CORRECTNESS_THRESHOLD", "0.6")),
            min_pass_rate=float(os.getenv("EVAL_MIN_PASS_RATE", "0.7")),
            latency_p95_ms=float(latency_raw) if latency_raw else None,
            include_reason=os.getenv("EVAL_INCLUDE_REASON", "true").strip().lower() in {"1", "true", "yes"},
        )
