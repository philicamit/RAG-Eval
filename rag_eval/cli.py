from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .runner import evaluate_local_pipeline
from .settings import EvalSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run production-style RAG evaluation (DeepEval + retrieval IR metrics)."
    )
    parser.add_argument("--dataset", type=Path, help="JSONL or JSON golden dataset path.")
    parser.add_argument("--output-dir", type=Path, help="Directory for JSON and Markdown reports.")
    parser.add_argument(
        "--mode",
        choices=["all", "retrieval", "generation"],
        help="Which metric family to run. retrieval skips the generator LLM.",
    )
    parser.add_argument("--max-cases", type=int, help="Limit the number of golden cases.")
    parser.add_argument("--eval-model", help="Judge model for DeepEval LLM-as-judge metrics.")
    parser.add_argument("--min-pass-rate", type=float, help="Quality-gate minimum pass rate.")
    parser.add_argument(
        "--fail-under-gates",
        action="store_true",
        default=True,
        help="Exit with code 1 when quality gates fail (default).",
    )
    parser.add_argument(
        "--no-fail-under-gates",
        action="store_true",
        help="Always exit 0 after writing the report.",
    )
    return parser


def _configure_stdio() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _require_deepeval() -> None:
    try:
        import deepeval  # noqa: F401
    except ModuleNotFoundError:
        print(
            "DeepEval is not installed for this Python interpreter:\n"
            f"  {sys.executable}\n"
            "Install it into the SAME environment you use to run eval:\n"
            f'  "{sys.executable}" -m pip install -r requirements.txt\n'
            "Then run:\n"
            f'  "{sys.executable}" -m rag_eval',
            file=sys.stderr,
        )
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    _require_deepeval()
    args = build_parser().parse_args(argv)
    settings = EvalSettings.from_env()
    settings = EvalSettings(
        dataset_path=args.dataset or settings.dataset_path,
        output_dir=args.output_dir or settings.output_dir,
        eval_model=args.eval_model or settings.eval_model,
        metric_mode=(args.mode or settings.metric_mode).lower(),
        max_cases=args.max_cases if args.max_cases is not None else settings.max_cases,
        max_concurrent=settings.max_concurrent,
        retrieval_threshold=settings.retrieval_threshold,
        generation_threshold=settings.generation_threshold,
        correctness_threshold=settings.correctness_threshold,
        min_pass_rate=args.min_pass_rate if args.min_pass_rate is not None else settings.min_pass_rate,
        latency_p95_ms=settings.latency_p95_ms,
        include_reason=settings.include_reason,
    )

    payload = evaluate_local_pipeline(settings)
    summary = payload.get("summary") or {}
    print(json.dumps({"run_id": payload.get("run_id"), "summary": summary, "reports": payload.get("report_paths")}, indent=2))

    fail_on_gates = not args.no_fail_under_gates
    gates_passed = bool((summary.get("gates") or {}).get("passed"))
    if fail_on_gates and not gates_passed:
        failures = (summary.get("gates") or {}).get("failures") or []
        print(
            "Evaluation finished and reports were written. "
            "Exit code 1 is a quality-gate result, not a crash.",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print("Re-run with --no-fail-under-gates to always exit 0.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
