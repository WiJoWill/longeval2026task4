"""Utilities for repairing provided baseline runs into valid Task 4 JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data_loader import load_queries
from .evaluator import FILLER_PATTERNS
from .output_writer import write_jsonl
from .validator import validate_record


def repair_run(
    input_run_path: str | Path,
    query_path: str | Path,
    output_run_path: str | Path,
    team_id: str = "our_team",
    run_id: str = "generated_responses_repaired_v1",
    run_type: str = "automatic",
    max_answer_sentences: int = 5,
) -> dict[str, Any]:
    """Repair a baseline run by filling metadata and dropping obvious filler."""

    query_map = {query.query_id: query for query in load_queries(query_path)}
    repaired_records: list[dict[str, Any]] = []

    for record in _read_jsonl(input_run_path):
        metadata = dict(record.get("metadata", {}))
        query_id = str(metadata.get("narrative_id", ""))
        query = query_map.get(query_id)
        references = [str(item) for item in record.get("references", [])]
        answer = repair_answer_items(record.get("answer", []), len(references), max_answer_sentences=max_answer_sentences)

        repaired = {
            "metadata": {
                "team_id": team_id,
                "run_id": run_id,
                "type": run_type,
                "narrative": query.text if query else metadata.get("narrative", ""),
                "narrative_id": query_id,
            },
            "references": references,
            "answer": answer,
        }
        validate_record(repaired)
        repaired_records.append(repaired)

    write_jsonl(repaired_records, output_run_path)
    return {
        "records_written": len(repaired_records),
        "output_run_path": str(output_run_path),
    }


def repair_answer_items(
    answer_items: list[dict[str, Any]],
    num_references: int,
    max_answer_sentences: int = 5,
) -> list[dict[str, Any]]:
    """Drop filler/uncited items and keep a compact valid answer list."""

    repaired: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for item in answer_items:
        text = str(item.get("text", "")).strip()
        if not text or _is_filler_text(text):
            continue
        citations = item.get("citations", [])
        if not isinstance(citations, list):
            continue
        clean_citations: list[int] = []
        for citation in citations:
            if isinstance(citation, int) and 0 <= citation < num_references and citation not in clean_citations:
                clean_citations.append(citation)
        if not clean_citations:
            continue
        signature = (text.lower(), tuple(clean_citations))
        if signature in seen:
            continue
        seen.add(signature)
        repaired.append({"text": text, "citations": clean_citations})
        if len(repaired) >= max_answer_sentences:
            break
    if not repaired:
        repaired.append({"text": "No supported answer could be generated from the provided candidate documents.", "citations": []})
    return repaired


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _is_filler_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in FILLER_PATTERNS)
