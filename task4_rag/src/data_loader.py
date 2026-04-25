"""Flexible loaders for LongEval-RAG query, candidate, and document files."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


QUERY_ID_FIELDS = ("narrative_id", "query_id", "qid", "id")
QUERY_TEXT_FIELDS = ("narrative", "query", "text", "question")
DOC_ID_FIELDS = ("doc_id", "docid", "id", "paper_id")
DOC_TEXT_FIELDS = ("fullText", "full_text", "text", "body", "contents", "abstract")
CANDIDATE_LIST_FIELDS = ("candidate_doc_ids", "candidate_docs", "candidates", "doc_ids")
TIMESTAMP_FIELDS = (
    "publishedDate",
    "publication_date",
    "published",
    "timestamp",
    "date",
    "createdDate",
    "year",
)


@dataclass(frozen=True)
class Query:
    """One Task 4 narrative/query."""

    query_id: str
    text: str
    candidate_doc_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Document:
    """One scientific document from the candidate pool."""

    doc_id: str
    text: str
    title: str = ""
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskInstance:
    """A query paired with its available candidate documents."""

    query: Query
    documents: list[Document]
    missing_candidate_ids: list[str] = field(default_factory=list)


def load_task(
    queries_path: str | Path,
    documents_path: str | Path,
    candidates_path: str | Path | None = None,
    candidate_limit: int | None = None,
    max_queries: int | None = None,
    document_text_fields: list[str] | None = None,
) -> list[TaskInstance]:
    """Load all query instances for a run.

    The official data format may evolve, so the loader accepts JSONL, JSON arrays,
    CSV, and TSV with common field aliases. Candidate IDs can live directly in the
    query rows or in a separate file keyed by query/narrative ID.
    """

    queries = load_queries(queries_path, limit=max_queries)
    candidate_map = load_candidate_map(candidates_path) if candidates_path else {}
    all_candidate_ids: set[str] = set()
    for query in queries:
        candidate_ids = query.candidate_doc_ids or candidate_map.get(query.query_id, [])
        if candidate_limit is not None:
            candidate_ids = candidate_ids[:candidate_limit]
        all_candidate_ids.update(candidate_ids)

    documents = load_documents(
        documents_path,
        allowed_doc_ids=all_candidate_ids or None,
        text_fields=document_text_fields,
    )
    doc_map = {doc.doc_id: doc for doc in documents}

    instances: list[TaskInstance] = []
    for query in queries:
        candidate_ids = query.candidate_doc_ids or candidate_map.get(query.query_id, [])
        if candidate_limit is not None:
            candidate_ids = candidate_ids[:candidate_limit]
        docs: list[Document] = []
        missing: list[str] = []
        for doc_id in candidate_ids:
            doc = doc_map.get(doc_id)
            if doc is None:
                missing.append(doc_id)
            else:
                docs.append(doc)
        instances.append(TaskInstance(query=query, documents=docs, missing_candidate_ids=missing))
    return instances


def load_queries(path: str | Path, limit: int | None = None) -> list[Query]:
    """Load query records from JSONL, JSON, CSV, or TSV."""

    rows = _load_records(path)
    queries: list[Query] = []
    for row in rows:
        query_id = _pick(row, QUERY_ID_FIELDS, required=True)
        text = _pick(row, QUERY_TEXT_FIELDS, required=True)
        candidate_doc_ids = _parse_candidate_ids(row)
        metadata = {k: v for k, v in row.items() if k not in set(QUERY_ID_FIELDS + QUERY_TEXT_FIELDS)}
        queries.append(
            Query(
                query_id=str(query_id),
                text=str(text).strip(),
                candidate_doc_ids=candidate_doc_ids,
                metadata=metadata,
            )
        )
        if limit is not None and len(queries) >= limit:
            break
    return queries


def load_documents(
    path: str | Path,
    allowed_doc_ids: set[str] | None = None,
    text_fields: list[str] | None = None,
) -> list[Document]:
    """Load document records from a file or a directory of supported files."""

    path = Path(path)
    text_fields = text_fields or list(DOC_TEXT_FIELDS)
    wanted = {str(doc_id) for doc_id in allowed_doc_ids} if allowed_doc_ids else None
    documents: list[Document] = []
    found_ids: set[str] = set()
    for row in _iter_records(path):
        doc_id = str(_pick(row, DOC_ID_FIELDS, required=True))
        if wanted is not None and doc_id not in wanted:
            continue
        text = _compose_document_text(row, text_fields=text_fields)
        if text is None:
            continue
        timestamp = _pick(row, TIMESTAMP_FIELDS, required=False)
        title = str(row.get("title", "") or "").strip()
        metadata = dict(row)
        documents.append(
            Document(
                doc_id=doc_id,
                text=text,
                title=title,
                timestamp=str(timestamp) if timestamp not in (None, "") else None,
                metadata=metadata,
            )
        )
        found_ids.add(doc_id)
        if wanted is not None and found_ids >= wanted:
            break
    return documents


def load_candidate_map(path: str | Path) -> dict[str, list[str]]:
    """Load query-to-candidate document IDs from a flexible table."""

    rows = _load_records(path)
    candidate_map: dict[str, list[str]] = {}
    for row in rows:
        query_id = _pick(row, QUERY_ID_FIELDS, required=True)
        ids = _parse_candidate_ids(row)
        if not ids:
            doc_id = _pick(row, DOC_ID_FIELDS, required=False)
            if doc_id is not None:
                ids = [str(doc_id)]
        candidate_map.setdefault(str(query_id), []).extend(ids)
    return {qid: _dedupe_preserve_order(doc_ids) for qid, doc_ids in candidate_map.items()}


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    return list(_iter_records(path))


def _iter_records(path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in {".jsonl", ".json", ".csv", ".tsv"}:
                yield from _iter_records(child)
        return

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _read_jsonl(path)
        return
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("queries", "documents", "records", "data"):
                if isinstance(data.get(key), list):
                    for item in data[key]:
                        yield dict(item)
                    return
            yield data
            return
        for item in data:
            yield dict(item)
        return
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter=delimiter):
                yield dict(row)
        return
    raise ValueError(f"Unsupported file format for {path}")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object on {path}:{line_no}")
            yield item


def _compose_document_text(row: dict[str, Any], text_fields: list[str]) -> str | None:
    title = str(row.get("title", "") or "").strip()
    for field_name in text_fields:
        value = row.get(field_name)
        if value not in (None, ""):
            text = str(value).strip()
            if text:
                return text
    return title or None


def _parse_candidate_ids(row: dict[str, Any]) -> list[str]:
    value = _pick(row, CANDIDATE_LIST_FIELDS, required=False)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
        for sep in ("\t", "|", ";", ","):
            if sep in raw:
                return [part.strip() for part in raw.split(sep) if part.strip()]
        if " " in raw:
            return [part.strip() for part in raw.split() if part.strip()]
        return [raw]
    return [str(value)]


def _pick(row: dict[str, Any], fields: tuple[str, ...], required: bool) -> Any:
    for field_name in fields:
        if field_name in row and row[field_name] not in (None, ""):
            return row[field_name]
    if required:
        raise ValueError(f"Missing required field; expected one of {fields} in {row}")
    return None


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output
