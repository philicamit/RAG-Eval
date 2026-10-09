from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GoldenCase:
    id: str
    input: str
    expected_output: str
    expected_context: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    expected_doc_ids: list[str] = field(default_factory=list)
    unanswerable: bool = False
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict) -> "GoldenCase":
        question = str(raw.get("input") or raw.get("question") or "").strip()
        if not question:
            raise ValueError("Golden case is missing `input`.")
        case_id = str(raw.get("id") or raw.get("case_id") or question[:48])
        return cls(
            id=case_id,
            input=question,
            expected_output=str(raw.get("expected_output") or raw.get("expected_answer") or "").strip(),
            expected_context=_as_str_list(raw.get("expected_context") or raw.get("reference_contexts")),
            expected_keywords=_as_str_list(raw.get("expected_keywords")),
            expected_doc_ids=_as_str_list(raw.get("expected_doc_ids")),
            unanswerable=bool(raw.get("unanswerable", False)),
            tags=_as_str_list(raw.get("tags")),
        )


def _as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"Expected a string or list of strings, got {type(value).__name__}")


def load_golden_dataset(path: str | Path, max_cases: int | None = None) -> list[GoldenCase]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {dataset_path}")

    cases: list[GoldenCase] = []
    if dataset_path.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL on line {line_number} of {dataset_path}") from exc
            cases.append(GoldenCase.from_dict(payload))
    else:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        rows = payload["cases"] if isinstance(payload, dict) and "cases" in payload else payload
        if not isinstance(rows, list):
            raise ValueError("JSON dataset must be a list or an object with a `cases` list.")
        cases = [GoldenCase.from_dict(row) for row in rows]

    if max_cases is not None:
        cases = cases[: max(0, max_cases)]
    if not cases:
        raise ValueError(f"Golden dataset is empty: {dataset_path}")
    return cases
