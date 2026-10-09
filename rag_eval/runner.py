from __future__ import annotations

import os
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .adapter import LocalRAGAdapter, RAGObservation, RAGSystem
from .dataset import GoldenCase, load_golden_dataset
from .deepeval_metrics import (
    build_deepeval_metrics,
    extract_metric_rows,
    run_deepeval,
    to_llm_test_case,
)
from .ir_metrics import compute_ir_metrics, refusal_score
from .report import build_summary, write_reports
from .settings import EvalSettings


def _disable_deepeval_telemetry() -> None:
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    os.environ.setdefault("ERROR_REPORTING", "0")
    os.environ.setdefault("DEEPEVAL_NO_INSPECT_PROMPT", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")


def collect_observation(system: RAGSystem, question: str, metric_mode: str) -> RAGObservation:
    if metric_mode == "retrieval":
        return system.retrieve(question)
    return system.answer(question)


def _score_batch(
    indexed_cases: list[tuple[int, GoldenCase, RAGObservation]],
    settings: EvalSettings,
    include_reference_metrics: bool,
) -> dict[int, dict]:
    if not indexed_cases:
        return {}
    metrics = build_deepeval_metrics(settings, include_reference_metrics=include_reference_metrics)
    if not metrics:
        return {}
    test_cases = [to_llm_test_case(case, observation) for _, case, observation in indexed_cases]
    result = run_deepeval(test_cases, metrics, settings.max_concurrent)
    rows = extract_metric_rows(result)
    scored: dict[int, dict] = {}
    for (index, _, _), row in zip(indexed_cases, rows):
        scored[index] = row
    return scored


def evaluate_rag(
    system: RAGSystem,
    goldens: list[GoldenCase],
    settings: EvalSettings,
    rag_config: dict[str, object] | None = None,
) -> dict:
    _disable_deepeval_telemetry()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    observations: list[tuple[int, GoldenCase, RAGObservation | None, str | None]] = []

    for index, case in enumerate(goldens):
        try:
            observation = collect_observation(system, case.input, settings.metric_mode)
            observations.append((index, case, observation, None))
        except Exception as exc:
            observations.append((index, case, None, f"{exc}\n{traceback.format_exc()}"))

    successful = [(i, case, obs) for i, case, obs, error in observations if obs is not None and error is None]
    answerable = [(i, case, obs) for i, case, obs in successful if not case.unanswerable]
    unanswerable = [(i, case, obs) for i, case, obs in successful if case.unanswerable]

    deepeval_by_index: dict[int, dict] = {}
    deepeval_by_index.update(_score_batch(answerable, settings, include_reference_metrics=True))
    if settings.metric_mode in {"all", "generation"}:
        deepeval_by_index.update(_score_batch(unanswerable, settings, include_reference_metrics=True))
    elif unanswerable:
        deepeval_by_index.update(_score_batch(unanswerable, settings, include_reference_metrics=False))

    case_rows: list[dict] = []
    for index, case, observation, error in observations:
        if error or observation is None:
            case_rows.append(
                {
                    "id": case.id,
                    "input": case.input,
                    "expected_output": case.expected_output,
                    "unanswerable": case.unanswerable,
                    "tags": case.tags,
                    "passed": False,
                    "error": error,
                    "metrics": {},
                    "latency": {},
                    "eval_cost_usd": 0.0,
                }
            )
            continue

        metrics: dict[str, dict] = {}
        ir_scores = compute_ir_metrics(case, observation)
        for name, score in ir_scores.items():
            if name == "k":
                continue
            metrics[name] = {"score": score, "success": None, "reason": "lexical/id retrieval metric"}
        refused = refusal_score(case, observation.answer)
        if refused:
            for name, score in refused.items():
                metrics[name] = {
                    "score": score,
                    "success": score >= 1.0,
                    "reason": "deterministic refusal check",
                }

        deepeval_row = deepeval_by_index.get(index, {})
        metrics.update(deepeval_row.get("metrics") or {})
        deepeval_pass = deepeval_row.get("success")
        refusal_ok = True if not case.unanswerable else (refused or {}).get("refusal_accuracy", 0.0) >= 1.0
        passed = (True if deepeval_pass is None else bool(deepeval_pass)) and refusal_ok and not error

        case_rows.append(
            {
                "id": case.id,
                "input": case.input,
                "expected_output": case.expected_output,
                "actual_output": observation.answer,
                "retrieval_context": observation.retrieved_texts,
                "retrieved_ids": observation.retrieved_ids,
                "unanswerable": case.unanswerable,
                "tags": case.tags,
                "passed": passed,
                "error": None,
                "metrics": metrics,
                "latency": observation.latency,
                "eval_cost_usd": float(deepeval_row.get("eval_cost_usd") or 0.0),
            }
        )

    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dataset_path": str(settings.dataset_path),
            "metric_mode": settings.metric_mode,
            "eval_model": settings.eval_model,
            "retrieval_threshold": settings.retrieval_threshold,
            "generation_threshold": settings.generation_threshold,
            "correctness_threshold": settings.correctness_threshold,
            "min_pass_rate": settings.min_pass_rate,
            "latency_p95_ms": settings.latency_p95_ms,
            "rag": rag_config or {},
        },
        "cases": case_rows,
    }
    payload["summary"] = build_summary(case_rows, settings)
    return payload


def evaluate_local_pipeline(settings: EvalSettings | None = None) -> dict:
    from app.config import Settings
    from app.rag_pipeline import RAGService

    eval_settings = settings or EvalSettings.from_env()
    rag_settings = Settings.from_env()
    rag_settings.validate()
    goldens = load_golden_dataset(eval_settings.dataset_path, eval_settings.max_cases)
    adapter = LocalRAGAdapter(RAGService(rag_settings))
    payload = evaluate_rag(
        adapter,
        goldens,
        eval_settings,
        rag_config={
            "model": rag_settings.model,
            "embedding_model": rag_settings.embedding_model,
            "retriever_k": rag_settings.retriever_k,
            "bm25_weight": rag_settings.bm25_weight,
            "vector_weight": rag_settings.vector_weight,
            "rerank_top_n": rag_settings.rerank_top_n,
        },
    )
    json_path, md_path = write_reports(payload, eval_settings.output_dir)
    payload["report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    return payload


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "reports"
