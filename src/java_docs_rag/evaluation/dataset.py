"""Developer A: load a versioned evaluation dataset."""

import json
from dataclasses import fields
from pathlib import Path

from java_docs_rag.common.models import EvaluationQuestion


def load_dataset(path: Path) -> list[EvaluationQuestion]:
    """Load a UTF-8 JSON array matching EvaluationQuestion exactly.

    Reject unknown/missing fields, duplicate question IDs and invalid field types
    with ValueError. Filesystem errors propagate to the caller.
    """
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Dataset must contain valid UTF-8 JSON") from exc
    if not isinstance(records, list):
        raise ValueError("Dataset must be a JSON array")
    required = {item.name for item in fields(EvaluationQuestion)}
    seen: set[str] = set()
    questions = []
    for index, record in enumerate(records):
        prefix = f"Dataset record {index}"
        if not isinstance(record, dict) or set(record) != required:
            raise ValueError(f"{prefix} must have exactly the EvaluationQuestion fields")
        for name in ("question_id", "question", "category"):
            if not isinstance(record[name], str) or not record[name].strip():
                raise ValueError(f"{prefix}: {name} must be a non-blank string")
        expected = record["expected_answer"]
        if expected is not None and (not isinstance(expected, str) or not expected.strip()):
            raise ValueError(f"{prefix}: expected_answer must be a non-blank string or null")
        if not isinstance(record["answerable"], bool):
            raise ValueError(f"{prefix}: answerable must be a boolean")
        relevant = record["relevant_document_ids"]
        if not isinstance(relevant, list) or any(
            not isinstance(value, str) or not value.strip() for value in relevant
        ):
            raise ValueError(
                f"{prefix}: relevant_document_ids must be an array of non-blank strings"
            )
        if record["question_id"] in seen:
            raise ValueError(f"{prefix}: duplicate question_id {record['question_id']!r}")
        seen.add(record["question_id"])
        questions.append(EvaluationQuestion(**record))
    return questions
