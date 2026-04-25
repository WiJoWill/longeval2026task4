"""Validation for Task 4 TREC RAG JSONL records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_METADATA_FIELDS = ("team_id", "run_id", "type", "narrative", "narrative_id")


class ValidationError(ValueError):
    """Raised when an output record violates the expected format."""


def validate_record(record: dict[str, Any]) -> None:
    """Validate a single Task 4 output record."""

    if not isinstance(record, dict):
        raise ValidationError("Record must be a JSON object")
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValidationError("Record metadata must be an object")
    for field_name in REQUIRED_METADATA_FIELDS:
        if field_name not in metadata or metadata[field_name] in (None, ""):
            raise ValidationError(f"Missing metadata.{field_name}")

    references = record.get("references")
    if not isinstance(references, list) or not all(isinstance(item, str) for item in references):
        raise ValidationError("references must be a list of document ID strings")
    if len(references) != len(set(references)):
        raise ValidationError("references must not contain duplicate document IDs")

    answer = record.get("answer")
    if not isinstance(answer, list):
        raise ValidationError("answer must be a list")
    for answer_index, item in enumerate(answer):
        if not isinstance(item, dict):
            raise ValidationError(f"answer[{answer_index}] must be an object")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValidationError(f"answer[{answer_index}].text must be a non-empty string")
        citations = item.get("citations")
        if not isinstance(citations, list) or not all(isinstance(citation, int) for citation in citations):
            raise ValidationError(f"answer[{answer_index}].citations must be a list of integers")
        for citation in citations:
            if citation < 0 or citation >= len(references):
                raise ValidationError(
                    f"answer[{answer_index}] citation {citation} is outside references length {len(references)}"
                )


def validate_jsonl(path: str | Path) -> list[str]:
    """Validate a run file, returning human-readable errors."""

    errors: list[str] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                validate_record(record)
            except Exception as exc:
                errors.append(f"{path}:{line_no}: {exc}")
    return errors
