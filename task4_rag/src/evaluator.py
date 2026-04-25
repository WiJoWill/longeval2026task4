"""Evaluation helpers for LongEval-RAG runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data_loader import load_queries
from .validator import ValidationError, validate_record


FILLER_PATTERNS = (
    re.compile(r"^\s*i will next answer\b", flags=re.IGNORECASE),
    re.compile(r"^\s*therefore\b", flags=re.IGNORECASE),
    re.compile(r"^\s*we have to consider\b", flags=re.IGNORECASE),
)


@dataclass(frozen=True)
class EvaluationSummary:
    """Compact report for one run file."""

    report: dict[str, Any]


def analyze_run(
    run_path: str | Path,
    query_path: str | Path,
) -> EvaluationSummary:
    """Analyze a TREC RAG run against the query-docid file."""

    run_records = _read_jsonl(run_path)
    queries = load_queries(query_path)
    query_map = {query.query_id: query for query in queries}

    invalid_records = 0
    validation_errors: list[str] = []
    answer_items = 0
    answer_items_without_citations = 0
    filler_items = 0
    total_references = 0
    total_unique_cited_references = 0
    reference_exact_match = 0
    reference_set_match = 0
    reference_subset_match = 0

    seen_query_ids: set[str] = set()
    per_record: list[dict[str, Any]] = []

    for record in run_records:
        metadata = record.get("metadata", {})
        query_id = str(metadata.get("narrative_id", ""))
        seen_query_ids.add(query_id)
        references = [str(item) for item in record.get("references", [])]
        answer = record.get("answer", [])
        total_references += len(references)

        try:
            validate_record(record)
        except ValidationError as exc:
            invalid_records += 1
            validation_errors.append(f"{query_id}: {exc}")

        cited_reference_ids = set()
        for item in answer:
            answer_items += 1
            citations = item.get("citations", [])
            if not citations:
                answer_items_without_citations += 1
            text = str(item.get("text", "")).strip()
            if _is_filler_text(text):
                filler_items += 1
            for citation_index in citations:
                if isinstance(citation_index, int) and 0 <= citation_index < len(references):
                    cited_reference_ids.add(references[citation_index])

        total_unique_cited_references += len(cited_reference_ids)

        expected = query_map.get(query_id)
        expected_doc_ids = expected.candidate_doc_ids if expected else []
        if expected_doc_ids and references == expected_doc_ids:
            reference_exact_match += 1
        if expected_doc_ids and set(references) == set(expected_doc_ids):
            reference_set_match += 1
        if expected_doc_ids and set(references).issubset(set(expected_doc_ids)):
            reference_subset_match += 1

        per_record.append(
            {
                "query_id": query_id,
                "num_references": len(references),
                "num_answer_items": len(answer),
                "num_items_without_citations": sum(1 for item in answer if not item.get("citations")),
                "num_unique_cited_references": len(cited_reference_ids),
                "reference_exact_match": bool(expected_doc_ids and references == expected_doc_ids),
                "reference_set_match": bool(expected_doc_ids and set(references) == set(expected_doc_ids)),
                "reference_subset_match": bool(expected_doc_ids and set(references).issubset(set(expected_doc_ids))),
            }
        )

    expected_query_ids = set(query_map)
    missing_query_ids = sorted(expected_query_ids - seen_query_ids)
    extra_query_ids = sorted(seen_query_ids - expected_query_ids)

    num_records = len(run_records)
    report = {
        "run_path": str(run_path),
        "query_path": str(query_path),
        "expected_queries": len(queries),
        "records_in_run": num_records,
        "missing_queries": len(missing_query_ids),
        "extra_queries": len(extra_query_ids),
        "missing_query_ids_sample": missing_query_ids[:10],
        "extra_query_ids_sample": extra_query_ids[:10],
        "invalid_records": invalid_records,
        "validation_errors_sample": validation_errors[:10],
        "reference_exact_match_rate": _ratio(reference_exact_match, num_records),
        "reference_set_match_rate": _ratio(reference_set_match, num_records),
        "reference_subset_match_rate": _ratio(reference_subset_match, num_records),
        "avg_references_per_record": _ratio(total_references, num_records),
        "avg_answer_items_per_record": _ratio(answer_items, num_records),
        "avg_unique_cited_references_per_record": _ratio(total_unique_cited_references, num_records),
        "answer_items_without_citations": answer_items_without_citations,
        "answer_item_empty_citation_rate": _ratio(answer_items_without_citations, answer_items),
        "filler_answer_items": filler_items,
        "filler_answer_item_rate": _ratio(filler_items, answer_items),
        "per_record_sample": per_record[:10],
    }
    return EvaluationSummary(report=report)


def write_report(summary: EvaluationSummary, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary.report, indent=2), encoding="utf-8")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _is_filler_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in FILLER_PATTERNS)


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)
