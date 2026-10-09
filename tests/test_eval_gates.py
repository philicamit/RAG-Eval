from rag_eval.gates import evaluate_gates, percentile
from rag_eval.report import build_summary
from rag_eval.settings import EvalSettings


def _settings(**overrides) -> EvalSettings:
    values = dict(
        dataset_path="unused.jsonl",
        output_dir="unused",
        eval_model="gpt-4o-mini",
        metric_mode="all",
        max_cases=None,
        max_concurrent=2,
        retrieval_threshold=0.6,
        generation_threshold=0.6,
        correctness_threshold=0.6,
        min_pass_rate=0.7,
        latency_p95_ms=2000.0,
        include_reason=True,
    )
    values.update(overrides)
    return EvalSettings(**values)


def test_percentile_interpolates():
    assert percentile([10, 20, 30, 40], 0.5) == 25


def test_gates_fail_on_pass_rate_and_latency():
    cases = [
        {
            "passed": True,
            "latency": {"total_ms": 100, "retrieval_ms": 10, "generation_ms": 80, "ttft_ms": 20},
            "metrics": {"Faithfulness": {"score": 0.9}},
            "eval_cost_usd": 0.01,
        },
        {
            "passed": False,
            "latency": {"total_ms": 5000, "retrieval_ms": 10, "generation_ms": 80, "ttft_ms": 20},
            "metrics": {"Faithfulness": {"score": 0.2}},
            "eval_cost_usd": 0.01,
        },
    ]
    summary = build_summary(cases, _settings())
    assert summary["pass_rate"] == 0.5
    assert summary["gates"]["passed"] is False
    joined = " ".join(summary["gates"]["failures"])
    assert "pass_rate" in joined
    assert "latency" in joined.lower() or "p95" in joined


def test_gates_pass_when_scores_and_latency_are_healthy():
    cases = [
        {
            "passed": True,
            "latency": {"total_ms": 400, "retrieval_ms": 40, "generation_ms": 300, "ttft_ms": 80},
            "metrics": {"Faithfulness": {"score": 0.9}, "hit_at_k": {"score": 1.0}},
            "eval_cost_usd": 0.0,
        }
    ]
    settings = _settings(min_pass_rate=0.5, latency_p95_ms=1000.0)
    result = evaluate_gates(build_summary(cases, settings), settings)
    assert result["passed"] is True
    assert result["failures"] == []
