"""RAG-specific quality diagnostics for Task 4 runs.

These metrics are proxy diagnostics, not official judged scores. They follow the
RAG evaluation framing of context relevance, answer faithfulness, answer
relevance, and RGB/RECALL-style robustness abilities.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .data_loader import Document, load_documents, load_queries
from .preprocess import normalize_text, tokenize
from .query_expansion import extract_keywords


RELEVANCE_THRESHOLD = 0.16
FAITHFULNESS_THRESHOLD = 0.22
GENERIC_TERMS = {
    "analysis",
    "approach",
    "based",
    "class",
    "data",
    "document",
    "effect",
    "evidence",
    "framework",
    "information",
    "method",
    "model",
    "paper",
    "result",
    "study",
    "system",
    "using",
}
INTEGRATION_HINTS = {
    "and",
    "both",
    "between",
    "compare",
    "comparison",
    "earlier",
    "later",
    "multiple",
    "relationship",
    "while",
}
REJECTION_PATTERNS = (
    re.compile(r"\binsufficient information\b", flags=re.IGNORECASE),
    re.compile(r"\bno supported answer\b", flags=re.IGNORECASE),
    re.compile(r"\bcannot answer\b", flags=re.IGNORECASE),
)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*(?:%|kg|km|mm|cm|m|g|mg|ml|s|sec|min|h|hr|year|years)?\b", re.IGNORECASE)


def analyze_rag_quality(
    run_path: str | Path,
    query_path: str | Path,
    documents_path: str | Path,
    doc_text_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Analyze retrieval-generation quality with reference-free proxy metrics."""

    run_records = _read_jsonl(run_path)
    queries = load_queries(query_path)
    query_map = {query.query_id: query for query in queries}
    wanted_doc_ids = _collect_doc_ids(run_records, query_map)
    documents = load_documents(documents_path, allowed_doc_ids=wanted_doc_ids, text_fields=doc_text_fields)
    doc_map = {document.doc_id: document for document in documents}

    per_record = []
    for record in run_records:
        metadata = record.get("metadata", {})
        query_id = str(metadata.get("narrative_id", ""))
        query = query_map.get(query_id)
        query_text = query.text if query else str(metadata.get("narrative", ""))
        references = [str(item) for item in record.get("references", [])]
        answer = record.get("answer", [])
        per_record.append(_score_record(query_id, query_text, references, answer, doc_map))

    return {
        "run_path": str(run_path),
        "query_path": str(query_path),
        "documents_path": str(documents_path),
        "metric_type": "reference_free_rag_proxy",
        "notes": [
            "These are diagnostic proxies, not official judged Task 4 labels.",
            "Context relevance and answer relevance use keyword overlap.",
            "Faithfulness estimates whether answer terms are supported by cited documents.",
            "RGB/RECALL-style scores adapt robustness ideas to this candidate-constrained task.",
        ],
        "quality_scores": _quality_scores(per_record),
        "rgb_like_abilities": _rgb_scores(per_record),
        "recall_like_counterfactual_risk": _recall_scores(per_record),
        "per_record_sample": per_record[:10],
    }


def write_rag_quality_report(report: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _score_record(
    query_id: str,
    query_text: str,
    references: list[str],
    answer: list[dict[str, Any]],
    doc_map: dict[str, Document],
) -> dict[str, Any]:
    query_terms = _content_terms(query_text)
    reference_scores = []
    relevant_reference_ids = set()
    cited_reference_ids = set()

    for index, doc_id in enumerate(references):
        doc = doc_map.get(doc_id)
        doc_text = _document_text(doc) if doc else ""
        relevance = _overlap_ratio(query_terms, _content_terms(doc_text))
        if relevance >= RELEVANCE_THRESHOLD:
            relevant_reference_ids.add(doc_id)
        reference_scores.append(
            {
                "reference_index": index,
                "doc_id": doc_id,
                "context_relevance": relevance,
                "title": doc.title if doc else "",
            }
        )

    answer_item_scores = []
    answer_text_parts = []
    unsupported_items = 0
    answer_numbers = 0
    supported_numbers = 0

    for item in answer:
        text = normalize_text(str(item.get("text", "")))
        answer_text_parts.append(text)
        citations = [citation for citation in item.get("citations", []) if isinstance(citation, int)]
        cited_docs = [references[citation] for citation in citations if 0 <= citation < len(references)]
        cited_reference_ids.update(cited_docs)
        cited_text = " ".join(_document_text(doc_map.get(doc_id)) for doc_id in cited_docs)
        faithfulness = _overlap_ratio(_content_terms(text), _content_terms(cited_text))
        if faithfulness < FAITHFULNESS_THRESHOLD:
            unsupported_items += 1
        numbers = NUMBER_RE.findall(text)
        answer_numbers += len(numbers)
        supported_numbers += sum(1 for number in numbers if number in cited_text)
        answer_item_scores.append(
            {
                "text": text[:220],
                "citations": citations,
                "faithfulness_proxy": faithfulness,
                "supported_number_rate": _ratio(sum(1 for number in numbers if number in cited_text), len(numbers)),
            }
        )

    cited_relevant = cited_reference_ids & relevant_reference_ids
    noisy_cited = cited_reference_ids - relevant_reference_ids
    answer_text = " ".join(answer_text_parts)
    answer_relevance = _overlap_ratio(query_terms, _content_terms(answer_text))
    no_relevant_context = not relevant_reference_ids
    rejected = _is_rejection(answer_text)
    integration_needed = _needs_integration(query_text)

    return {
        "query_id": query_id,
        "context_relevance": _avg(item["context_relevance"] for item in reference_scores),
        "context_precision_proxy": _ratio(len(relevant_reference_ids), len(references)),
        "answer_faithfulness_proxy": _avg(item["faithfulness_proxy"] for item in answer_item_scores),
        "answer_relevance_proxy": answer_relevance,
        "unsupported_answer_item_rate": _ratio(unsupported_items, len(answer_item_scores)),
        "noise_robustness_proxy": 1.0 - _ratio(len(noisy_cited), len(cited_reference_ids)),
        "negative_rejection_applicable": no_relevant_context,
        "negative_rejection_success": bool(no_relevant_context and rejected),
        "information_integration_applicable": integration_needed,
        "information_integration_proxy": _ratio(len(cited_reference_ids), 2) if integration_needed else None,
        "counterfactual_risk_proxy": 1.0 - _ratio(supported_numbers, answer_numbers),
        "num_references": len(references),
        "num_cited_references": len(cited_reference_ids),
        "num_relevant_references": len(relevant_reference_ids),
        "num_noisy_cited_references": len(noisy_cited),
        "reference_scores": reference_scores[:10],
        "answer_item_scores": answer_item_scores[:10],
    }


def _quality_scores(records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "context_relevance": _avg(record["context_relevance"] for record in records),
        "context_precision_proxy": _avg(record["context_precision_proxy"] for record in records),
        "answer_faithfulness_proxy": _avg(record["answer_faithfulness_proxy"] for record in records),
        "answer_relevance_proxy": _avg(record["answer_relevance_proxy"] for record in records),
        "unsupported_answer_item_rate": _avg(record["unsupported_answer_item_rate"] for record in records),
    }


def _rgb_scores(records: list[dict[str, Any]]) -> dict[str, float]:
    negative_records = [record for record in records if record["negative_rejection_applicable"]]
    integration_records = [record for record in records if record["information_integration_applicable"]]
    return {
        "noise_robustness_proxy": _avg(record["noise_robustness_proxy"] for record in records),
        "negative_rejection_coverage": _ratio(len(negative_records), len(records)),
        "negative_rejection_success_rate": _avg(record["negative_rejection_success"] for record in negative_records),
        "information_integration_coverage": _ratio(len(integration_records), len(records)),
        "information_integration_proxy": _avg(
            min(float(record["information_integration_proxy"] or 0.0), 1.0) for record in integration_records
        ),
    }


def _recall_scores(records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "numeric_claim_support_rate": 1.0 - _avg(record["counterfactual_risk_proxy"] for record in records),
        "counterfactual_risk_proxy": _avg(record["counterfactual_risk_proxy"] for record in records),
        "high_risk_record_rate": _avg(record["counterfactual_risk_proxy"] > 0.0 for record in records),
    }


def _collect_doc_ids(run_records: list[dict[str, Any]], query_map: dict[str, Any]) -> set[str]:
    doc_ids: set[str] = set()
    for record in run_records:
        doc_ids.update(str(item) for item in record.get("references", []))
        query_id = str(record.get("metadata", {}).get("narrative_id", ""))
        query = query_map.get(query_id)
        if query:
            doc_ids.update(query.candidate_doc_ids)
    return doc_ids


def _document_text(document: Document | None) -> str:
    if document is None:
        return ""
    return normalize_text(f"{document.title} {document.text}")


def _content_terms(text: str) -> set[str]:
    terms = set(extract_keywords(text, limit=80))
    terms.update(token for token in tokenize(text) if len(token) >= 4 and token not in GENERIC_TERMS)
    return {term for term in terms if term not in GENERIC_TERMS}


def _overlap_ratio(left_terms: set[str], right_terms: set[str]) -> float:
    if not left_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms)


def _needs_integration(query_text: str) -> bool:
    query_tokens = set(tokenize(query_text))
    return bool(query_tokens & INTEGRATION_HINTS)


def _is_rejection(text: str) -> bool:
    return any(pattern.search(text) for pattern in REJECTION_PATTERNS)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _avg(values: Any) -> float:
    items = [float(value) for value in values]
    return _ratio(sum(items), len(items))


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)
