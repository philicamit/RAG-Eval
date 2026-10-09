from __future__ import annotations

from typing import Any

from .adapter import RAGObservation
from .dataset import GoldenCase
from .settings import EvalSettings


def _geval_params():
    try:
        from deepeval.test_case import SingleTurnParams

        return [
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ]
    except ImportError:
        from deepeval.test_case import LLMTestCaseParams

        return [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ]


def build_deepeval_metrics(settings: EvalSettings, include_reference_metrics: bool) -> list[Any]:
    """DeepEval RAG metrics. Precision/recall need expected_output (LLM-as-judge)."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval,
    )

    model = settings.eval_model
    retrieval_threshold = settings.retrieval_threshold
    generation_threshold = settings.generation_threshold
    include_reason = settings.include_reason
    metrics: list[Any] = []

    if settings.metric_mode in {"all", "retrieval"}:
        metrics.append(
            _build_metric(
                ContextualRelevancyMetric,
                threshold=retrieval_threshold,
                model=model,
                include_reason=include_reason,
            )
        )
        if include_reference_metrics:
            metrics.extend(
                [
                    _build_metric(
                        ContextualPrecisionMetric,
                        threshold=retrieval_threshold,
                        model=model,
                        include_reason=include_reason,
                    ),
                    _build_metric(
                        ContextualRecallMetric,
                        threshold=retrieval_threshold,
                        model=model,
                        include_reason=include_reason,
                    ),
                ]
            )

    if settings.metric_mode in {"all", "generation"}:
        metrics.extend(
            [
                _build_metric(
                    FaithfulnessMetric,
                    threshold=generation_threshold,
                    model=model,
                    include_reason=include_reason,
                ),
                _build_metric(
                    AnswerRelevancyMetric,
                    threshold=generation_threshold,
                    model=model,
                    include_reason=include_reason,
                ),
            ]
        )
        if include_reference_metrics:
            metrics.append(
                _build_metric(
                    GEval,
                    name="Correctness",
                    evaluation_steps=[
                        "Compare the actual output to the expected output.",
                        "Ignore wording differences when the same facts are present.",
                        "Penalize missing key facts and contradictions.",
                        "If the expected output is a refusal, pass only when the actual output also refuses.",
                    ],
                    evaluation_params=_geval_params(),
                    threshold=settings.correctness_threshold,
                    model=model,
                )
            )
    return metrics


def _build_metric(metric_cls, **kwargs):
    optional = ["include_reason", "model", "evaluation_params", "evaluation_steps", "name"]
    current = dict(kwargs)
    while True:
        try:
            return metric_cls(**current)
        except TypeError as exc:
            dropped = None
            for key in optional:
                if key in current and key in str(exc):
                    dropped = key
                    break
            if dropped is None:
                raise
            current.pop(dropped)


def to_llm_test_case(case: GoldenCase, observation: RAGObservation):
    import inspect

    from deepeval.test_case import LLMTestCase

    kwargs: dict[str, Any] = {
        "input": case.input,
        "actual_output": observation.answer or " ",
        "retrieval_context": observation.retrieved_texts or [" "],
        "expected_output": case.expected_output or None,
        "context": case.expected_context or None,
        "name": case.id,
        "tags": case.tags or None,
        "metadata": {
            "case_id": case.id,
            "unanswerable": case.unanswerable,
            "tags": case.tags,
            "retrieved_ids": observation.retrieved_ids,
        },
        "additional_metadata": {
            "case_id": case.id,
            "unanswerable": case.unanswerable,
            "retrieved_ids": observation.retrieved_ids,
        },
    }
    accepted = set(inspect.signature(LLMTestCase).parameters)
    return LLMTestCase(**{key: value for key, value in kwargs.items() if key in accepted and value is not None})


def run_deepeval(test_cases: list[LLMTestCase], metrics: list[Any], max_concurrent: int) -> Any:
    from deepeval import evaluate

    kwargs: dict[str, Any] = {
        "test_cases": test_cases,
        "metrics": metrics,
    }
    try:
        from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig

        kwargs["async_config"] = AsyncConfig(run_async=True, max_concurrent=max_concurrent)
        kwargs["error_config"] = ErrorConfig(ignore_errors=True, skip_on_missing_params=True)
        kwargs["cache_config"] = CacheConfig(write_cache=False, use_cache=False)
        display_kwargs = {"print_results": False, "inspect_after_run": False, "show_indicator": False}
        try:
            kwargs["display_config"] = DisplayConfig(**display_kwargs)
        except TypeError:
            display_kwargs.pop("inspect_after_run", None)
            kwargs["display_config"] = DisplayConfig(**display_kwargs)
    except ImportError:
        kwargs["run_async"] = True

    return evaluate(**kwargs)


def extract_metric_rows(evaluation_result: Any) -> list[dict[str, Any]]:
    if evaluation_result is None:
        return []
    test_results = getattr(evaluation_result, "test_results", None)
    if test_results is None and isinstance(evaluation_result, dict):
        test_results = evaluation_result.get("test_results") or evaluation_result.get("testResults")
    if not test_results:
        return []

    rows: list[dict[str, Any]] = []
    for result in test_results:
        metrics_data = getattr(result, "metrics_data", None)
        if metrics_data is None:
            metrics_data = getattr(result, "metricsData", None)
        if metrics_data is None and isinstance(result, dict):
            metrics_data = result.get("metrics_data") or result.get("metricsData") or []

        metric_scores: dict[str, Any] = {}
        passed = True
        cost = 0.0
        for item in metrics_data or []:
            name = str(getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else "metric"))
            score = getattr(item, "score", None) if not isinstance(item, dict) else item.get("score")
            success = getattr(item, "success", None) if not isinstance(item, dict) else item.get("success")
            reason = getattr(item, "reason", None) if not isinstance(item, dict) else item.get("reason")
            error = getattr(item, "error", None) if not isinstance(item, dict) else item.get("error")
            eval_cost = getattr(item, "evaluation_cost", None) if not isinstance(item, dict) else item.get("evaluationCost")
            if eval_cost:
                cost += float(eval_cost)
            if success is False:
                passed = False
            metric_scores[name] = {
                "score": None if score is None else float(score),
                "success": success,
                "reason": reason,
                "error": error,
            }
        rows.append(
            {
                "success": bool(getattr(result, "success", passed)),
                "metrics": metric_scores,
                "eval_cost_usd": cost,
            }
        )
    return rows
