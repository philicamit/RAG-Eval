from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .gates import evaluate_gates, mean_or_none, percentile
from .settings import EvalSettings


def build_summary(cases: list[dict], settings: EvalSettings) -> dict:
    n_cases = len(cases)
    n_passed = sum(1 for case in cases if case.get("passed"))
    totals = [float((case.get("latency") or {}).get("total_ms") or 0.0) for case in cases]
    retrievals = [float((case.get("latency") or {}).get("retrieval_ms") or 0.0) for case in cases]
    generations = [float((case.get("latency") or {}).get("generation_ms") or 0.0) for case in cases]
    ttfts = [float((case.get("latency") or {}).get("ttft_ms") or 0.0) for case in cases]

    metric_values: dict[str, list[float]] = {}
    for case in cases:
        for name, payload in (case.get("metrics") or {}).items():
            score = payload.get("score") if isinstance(payload, dict) else payload
            if score is None:
                continue
            metric_values.setdefault(name, []).append(float(score))

    summary = {
        "n_cases": n_cases,
        "n_passed": n_passed,
        "pass_rate": (n_passed / n_cases) if n_cases else 0.0,
        "metric_means": {name: mean(values) for name, values in metric_values.items()},
        "latency": {
            "mean_total_ms": mean_or_none(totals),
            "p50_total_ms": percentile(totals, 0.50) if totals else None,
            "p95_total_ms": percentile(totals, 0.95) if totals else None,
            "mean_retrieval_ms": mean_or_none(retrievals),
            "mean_generation_ms": mean_or_none(generations),
            "mean_ttft_ms": mean_or_none(ttfts),
        },
        "eval_cost_usd": sum(float(case.get("eval_cost_usd") or 0.0) for case in cases),
        "error_count": sum(1 for case in cases if case.get("error")),
    }
    summary["gates"] = evaluate_gates(summary, settings)
    return summary


def write_reports(run_payload: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_payload.get("run_id", stamp)
    json_path = output_dir / f"eval_{run_id}.json"
    md_path = output_dir / f"eval_{run_id}.md"
    json_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(run_payload), encoding="utf-8")
    latest_json = output_dir / "latest.json"
    latest_md = output_dir / "latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return json_path, md_path


def _to_markdown(payload: dict) -> str:
    summary = payload.get("summary") or {}
    gates = summary.get("gates") or {}
    lines = [
        f"# RAG evaluation report (`{payload.get('run_id')}`)",
        "",
        f"- Time (UTC): {payload.get('timestamp')}",
        f"- Mode: `{payload.get('config', {}).get('metric_mode')}`",
        f"- Dataset: `{payload.get('config', {}).get('dataset_path')}`",
        f"- Judge model: `{payload.get('config', {}).get('eval_model')}`",
        f"- Cases: {summary.get('n_cases')} (passed {summary.get('n_passed')})",
        f"- Pass rate: {float(summary.get('pass_rate') or 0):.3f}",
        f"- Eval cost (USD): {float(summary.get('eval_cost_usd') or 0):.4f}",
        f"- Quality gates: {'PASS' if gates.get('passed') else 'FAIL'}",
        "",
        "## Metric means",
        "",
    ]
    means = summary.get("metric_means") or {}
    if means:
        lines.extend(["| Metric | Mean |", "| --- | ---: |"])
        for name, value in sorted(means.items()):
            lines.append(f"| {name} | {value:.3f} |")
    else:
        lines.append("_No metric scores recorded._")

    latency = summary.get("latency") or {}
    lines.extend(
        [
            "",
            "## Latency",
            "",
            f"- Mean total: {_fmt_ms(latency.get('mean_total_ms'))}",
            f"- p50 total: {_fmt_ms(latency.get('p50_total_ms'))}",
            f"- p95 total: {_fmt_ms(latency.get('p95_total_ms'))}",
            f"- Mean retrieval: {_fmt_ms(latency.get('mean_retrieval_ms'))}",
            f"- Mean generation: {_fmt_ms(latency.get('mean_generation_ms'))}",
            f"- Mean TTFT: {_fmt_ms(latency.get('mean_ttft_ms'))}",
            "",
        ]
    )
    if gates.get("failures"):
        lines.append("## Gate failures")
        lines.append("")
        for failure in gates["failures"]:
            lines.append(f"- {failure}")
        lines.append("")

    lines.extend(["## Cases", ""])
    for case in payload.get("cases") or []:
        status = "PASS" if case.get("passed") else "FAIL"
        lines.append(f"### {case.get('id')} — {status}")
        lines.append("")
        lines.append(f"- Question: {case.get('input')}")
        if case.get("error"):
            lines.append(f"- Error: {case['error']}")
        for name, payload_metric in (case.get("metrics") or {}).items():
            if not isinstance(payload_metric, dict):
                lines.append(f"- {name}: {payload_metric}")
                continue
            score = payload_metric.get("score")
            score_text = "n/a" if score is None else f"{float(score):.3f}"
            lines.append(f"- {name}: {score_text}")
        lines.append("")
    return "\n".join(lines)


def _fmt_ms(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f} ms"
