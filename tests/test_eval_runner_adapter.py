from rag_eval.adapter import RAGObservation
from rag_eval.dataset import GoldenCase
from rag_eval.runner import evaluate_rag
from rag_eval.settings import EvalSettings


class FakeRAG:
    def retrieve(self, question: str) -> RAGObservation:
        return RAGObservation(
            question=question,
            answer="",
            retrieved_texts=["Eddard Stark is the Lord of Winterfell and Warden of the North."],
            retrieved_ids=["doc-1"],
            latency={"retrieval_ms": 12.0, "rerank_ms": 1.0, "generation_ms": 0.0, "ttft_ms": 12.0, "total_ms": 12.0},
        )

    def answer(self, question: str) -> RAGObservation:
        obs = self.retrieve(question)
        obs.answer = "Eddard Stark is Lord of Winterfell."
        obs.latency = {
            "retrieval_ms": 12.0,
            "rerank_ms": 1.0,
            "generation_ms": 80.0,
            "ttft_ms": 20.0,
            "total_ms": 100.0,
        }
        return obs


def test_evaluate_rag_with_fake_system_skips_deepeval(monkeypatch):
    monkeypatch.setattr("rag_eval.runner._score_batch", lambda *args, **kwargs: {})

    settings = EvalSettings(
        dataset_path="unused.jsonl",
        output_dir="unused",
        eval_model="gpt-4o-mini",
        metric_mode="all",
        max_cases=None,
        max_concurrent=1,
        retrieval_threshold=0.5,
        generation_threshold=0.5,
        correctness_threshold=0.5,
        min_pass_rate=0.5,
        latency_p95_ms=None,
        include_reason=False,
    )
    goldens = [
        GoldenCase(
            id="ned",
            input="Who is Ned Stark?",
            expected_output="Eddard Stark is Lord of Winterfell.",
            expected_keywords=["Winterfell", "Eddard"],
            expected_context=["Eddard Stark is the Lord of Winterfell and Warden of the North."],
        )
    ]
    payload = evaluate_rag(FakeRAG(), goldens, settings)
    assert payload["summary"]["n_cases"] == 1
    assert payload["cases"][0]["metrics"]["hit_at_k"]["score"] == 1.0
    assert payload["cases"][0]["latency"]["total_ms"] == 100.0
    assert "report_paths" not in payload
