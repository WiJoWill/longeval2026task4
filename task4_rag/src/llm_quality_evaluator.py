"""Optional LLM-as-judge evaluation for Task 4 RAG runs.

This evaluator is intentionally separate from the deterministic proxy
evaluator. It uses an external model to judge relevance, faithfulness, citation
quality, and robustness from a compact query/evidence/answer packet.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .data_loader import Document, load_documents, load_queries
from .preprocess import normalize_text


SCORE_FIELDS = (
    "context_relevance",
    "answer_relevance",
    "faithfulness",
    "completeness",
    "citation_quality",
    "noise_robustness",
    "information_integration",
    "numeric_factuality",
    "overall",
)


@dataclass(frozen=True)
class LLMJudgeConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_records: int | None = None
    max_doc_chars: int = 1200
    sleep_seconds: float = 0.0


class JudgeClient(Protocol):
    def judge(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return one parsed JSON judgment."""


def analyze_with_llm_judge(
    run_path: str | Path,
    query_path: str | Path,
    documents_path: str | Path,
    config: LLMJudgeConfig,
    doc_text_fields: list[str] | None = None,
) -> dict[str, Any]:
    run_records = _read_jsonl(run_path)
    queries = load_queries(query_path)
    query_map = {query.query_id: query for query in queries}
    if config.max_records is not None:
        run_records = run_records[: config.max_records]

    judge = _make_judge_client(config)
    wanted_doc_ids = _collect_reference_ids(run_records)
    documents = load_documents(documents_path, allowed_doc_ids=wanted_doc_ids, text_fields=doc_text_fields)
    doc_map = {document.doc_id: document for document in documents}

    judgments: list[dict[str, Any]] = []
    for index, record in enumerate(run_records, start=1):
        metadata = record.get("metadata", {})
        query_id = str(metadata.get("narrative_id", ""))
        query = query_map.get(query_id)
        query_text = query.text if query else str(metadata.get("narrative", ""))
        payload = _judge_payload(
            record=record,
            query_id=query_id,
            query_text=query_text,
            doc_map=doc_map,
            max_doc_chars=config.max_doc_chars,
        )
        judgment = judge.judge(payload)
        judgment["query_id"] = query_id
        judgment["record_index"] = index
        judgments.append(_normalize_judgment(judgment))
        if config.sleep_seconds > 0:
            time.sleep(config.sleep_seconds)

    return {
        "run_path": str(run_path),
        "query_path": str(query_path),
        "documents_path": str(documents_path),
        "metric_type": "llm_as_judge",
        "provider": config.provider,
        "model": config.model,
        "num_records_judged": len(judgments),
        "score_scale": "1=poor, 2=weak, 3=adequate, 4=good, 5=excellent",
        "score_means": _score_means(judgments),
        "score_distributions": _score_distributions(judgments),
        "judgments": judgments,
    }


def write_llm_judge_report(report: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _make_judge_client(config: LLMJudgeConfig) -> JudgeClient:
    if config.provider != "openai":
        raise ValueError(
            "Only provider='openai' is implemented in this repo right now. "
            "Claude and Gemini can be added behind the same JudgeClient interface."
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; cannot run OpenAI LLM judge.")
    return OpenAIJudgeClient(model=config.model, temperature=config.temperature)


class OpenAIJudgeClient:
    def __init__(self, model: str, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    def judge(self, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY")

        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _judge_system_prompt()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM judge returned an empty response.")
        return json.loads(content)


def _judge_payload(
    record: dict[str, Any],
    query_id: str,
    query_text: str,
    doc_map: dict[str, Document],
    max_doc_chars: int,
) -> dict[str, Any]:
    references = [str(item) for item in record.get("references", [])]
    evidence = []
    for index, doc_id in enumerate(references):
        doc = doc_map.get(doc_id)
        evidence.append(
            {
                "reference_index": index,
                "doc_id": doc_id,
                "title": doc.title if doc else "",
                "text_excerpt": _truncate(_document_text(doc), max_doc_chars),
            }
        )
    return {
        "query_id": query_id,
        "query": query_text,
        "references": evidence,
        "answer": record.get("answer", []),
        "instructions": {
            "use_only_supplied_evidence": True,
            "score_scale": "1=poor, 2=weak, 3=adequate, 4=good, 5=excellent",
            "citation_indices_refer_to_references": True,
        },
    }


def _judge_system_prompt() -> str:
    return """You are a strict RAG evaluation judge for scientific QA.

Evaluate the answer only against the supplied query and evidence. Do not use outside knowledge.
Return a single JSON object with this exact shape:
{
  "scores": {
    "context_relevance": 1-5,
    "answer_relevance": 1-5,
    "faithfulness": 1-5,
    "completeness": 1-5,
    "citation_quality": 1-5,
    "noise_robustness": 1-5,
    "information_integration": 1-5,
    "numeric_factuality": 1-5,
    "overall": 1-5
  },
  "qualitative": {
    "strengths": ["short bullet"],
    "weaknesses": ["short bullet"],
    "failure_modes": ["short label"],
    "recommended_fix": "short recommendation"
  }
}

Rubric:
- context_relevance: selected references are relevant to the query.
- answer_relevance: answer directly addresses the query.
- faithfulness: answer claims are supported by cited evidence.
- completeness: answer covers the important aspects available in evidence.
- citation_quality: citations are present, valid, and attached to supporting claims.
- noise_robustness: answer avoids using irrelevant/noisy references.
- information_integration: answer combines multiple references when needed; if not needed, score based on not overusing evidence.
- numeric_factuality: numeric/date/statistical claims match cited evidence; if no numeric claims, score 5.
- overall: holistic usefulness for the query.
Be conservative. Penalize query restatement, title-only answers, off-topic cited evidence, unsupported claims, and malformed extraction artifacts."""


def _normalize_judgment(judgment: dict[str, Any]) -> dict[str, Any]:
    raw_scores = judgment.get("scores", {})
    scores = {field: _normalize_score(raw_scores.get(field)) for field in SCORE_FIELDS}
    qualitative = judgment.get("qualitative", {})
    return {
        "query_id": str(judgment.get("query_id", "")),
        "record_index": int(judgment.get("record_index", 0) or 0),
        "scores": scores,
        "qualitative": {
            "strengths": _string_list(qualitative.get("strengths")),
            "weaknesses": _string_list(qualitative.get("weaknesses")),
            "failure_modes": _string_list(qualitative.get("failure_modes")),
            "recommended_fix": normalize_text(str(qualitative.get("recommended_fix", ""))),
        },
    }


def _score_means(judgments: list[dict[str, Any]]) -> dict[str, float]:
    return {
        field: _ratio(sum(item["scores"][field] for item in judgments), len(judgments))
        for field in SCORE_FIELDS
    }


def _score_distributions(judgments: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    distributions: dict[str, dict[str, int]] = {}
    for field in SCORE_FIELDS:
        bucket = {str(score): 0 for score in range(1, 6)}
        for item in judgments:
            bucket[str(item["scores"][field])] += 1
        distributions[field] = bucket
    return distributions


def _normalize_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 1
    return max(1, min(5, score))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [normalize_text(str(item)) for item in value if normalize_text(str(item))]
    if value:
        return [normalize_text(str(value))]
    return []


def _collect_reference_ids(run_records: list[dict[str, Any]]) -> set[str]:
    doc_ids: set[str] = set()
    for record in run_records:
        doc_ids.update(str(item) for item in record.get("references", []))
    return doc_ids


def _document_text(document: Document | None) -> str:
    if document is None:
        return ""
    return normalize_text(f"{document.title}\n{document.text}")


def _truncate(text: str, max_chars: int) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)
