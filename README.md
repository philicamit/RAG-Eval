# RAG Eval

A standalone evaluation harness for RAG systems. This package measures retrieval quality, answer quality, refusal behavior, and latency using golden datasets and DeepEval metrics.

## Features
- Retrieval metrics: hit@k, MRR, context recall, keyword recall
- Generation metrics: faithfulness, answer relevancy, correctness
- Refusal checks for unanswerable cases
- Quality gates and pass-rate thresholds
- JSON and Markdown reporting

## Quick start

```bash
python -m pip install -r requirements.txt
python -m rag_eval --dataset rag_eval/datasets/got_golden.jsonl --output-dir reports
```

## Repository contents
- `rag_eval/` — evaluation package
- `datasets/` — golden cases
- `tests/` — evaluation tests
- `reports/` — generated outputs
