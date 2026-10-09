from __future__ import annotations

from statistics import mean

from .settings import EvalSettings


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def evaluate_gates(summary: dict, settings: EvalSettings) -> dict:
    failures: list[str] = []
    pass_rate = float(summary.get("pass_rate") or 0.0)
    if pass_rate < settings.min_pass_rate:
        failures.append(
            f"pass_rate {pass_rate:.3f} is below EVAL_MIN_PASS_RATE {settings.min_pass_rate:.3f}"
        )

    metric_means: dict[str, float] = summary.get("metric_means") or {}
    retrieval_metrics = {
        "Contextual Relevancy",
        "Contextual Precision",
        "Contextual Recall",
        "hit_at_k",
        "context_recall_at_k",
    }
    generation_metrics = {
        "Faithfulness",
        "Answer Relevancy",
        "Correctness",
        "refusal_accuracy",
    }

    def _matches(name: str, candidates: set[str]) -> bool:
        return name in candidates or any(name.startswith(candidate) for candidate in candidates)

    for name, value in metric_means.items():
        if _matches(name, retrieval_metrics) and value < settings.retrieval_threshold:
            failures.append(
                f"{name} mean {value:.3f} is below retrieval threshold {settings.retrieval_threshold:.3f}"
            )
        if _matches(name, generation_metrics) and value < settings.generation_threshold:
            failures.append(
                f"{name} mean {value:.3f} is below generation threshold {settings.generation_threshold:.3f}"
            )

    latency = summary.get("latency") or {}
    p95 = latency.get("p95_total_ms")
    if settings.latency_p95_ms is not None and p95 is not None and p95 > settings.latency_p95_ms:
        failures.append(
            f"p95 total latency {p95:.1f}ms exceeds EVAL_LATENCY_P95_MS {settings.latency_p95_ms:.1f}ms"
        )

    return {"passed": not failures, "failures": failures}


def mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None
