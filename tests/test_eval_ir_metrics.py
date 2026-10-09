from rag_eval.adapter import RAGObservation
from rag_eval.dataset import GoldenCase
from rag_eval.ir_metrics import compute_ir_metrics, refusal_score


def test_hit_mrr_and_keyword_recall_on_ranked_context():
    case = GoldenCase(
        id="t1",
        input="Who is Ned?",
        expected_output="Eddard Stark is Lord of Winterfell.",
        expected_context=["Eddard Stark is the Lord of Winterfell and Warden of the North."],
        expected_keywords=["Eddard", "Winterfell"],
    )
    observation = RAGObservation(
        question=case.input,
        answer="Ned Stark rules Winterfell.",
        retrieved_texts=[
            "Jaime Lannister is a knight of the Kingsguard.",
            "Eddard Stark is the Lord of Winterfell and Warden of the North.",
        ],
        retrieved_ids=["a", "b"],
    )
    scores = compute_ir_metrics(case, observation)
    assert scores["hit_at_k"] == 1.0
    assert scores["mrr_at_k"] == 0.5
    assert scores["keyword_recall"] == 1.0
    assert scores["context_recall_at_k"] > 0.5


def test_miss_when_context_is_irrelevant():
    case = GoldenCase(
        id="t2",
        input="Who is Ned?",
        expected_output="Eddard Stark is Lord of Winterfell.",
        expected_keywords=["Winterfell"],
        expected_context=["Eddard Stark is Lord of Winterfell."],
    )
    observation = RAGObservation(
        question=case.input,
        answer="I don't know.",
        retrieved_texts=["The weather in King's Landing is humid."],
        retrieved_ids=["x"],
    )
    scores = compute_ir_metrics(case, observation)
    assert scores["hit_at_k"] == 0.0
    assert scores["mrr_at_k"] == 0.0


def test_refusal_score_only_for_unanswerable_cases():
    case = GoldenCase(
        id="u1",
        input="What is Apple stock?",
        expected_output="I don't know. Please try asking another question.",
        unanswerable=True,
    )
    assert refusal_score(case, "I don't know. Please try asking another question.")["refusal_accuracy"] == 1.0
    assert refusal_score(case, "Apple closed at 190.")["refusal_accuracy"] == 0.0
    answerable = GoldenCase(id="a1", input="Who is Ned?", expected_output="Ned Stark")
    assert refusal_score(answerable, "Ned Stark") is None
