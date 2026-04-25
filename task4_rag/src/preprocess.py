"""Preprocessing and passage splitting for scientific RAG evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .data_loader import Document


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass(frozen=True)
class Passage:
    """A retrievable evidence unit."""

    passage_id: str
    doc_id: str
    text: str
    title: str = ""
    timestamp: str | None = None
    metadata: dict | None = None


def normalize_text(text: str) -> str:
    """Collapse whitespace while preserving readable text."""

    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> list[str]:
    """Tokenize text for reproducible lexical retrieval."""

    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with a lightweight deterministic heuristic."""

    text = normalize_text(text)
    if not text:
        return []
    sentences = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    return sentences or [text]


def split_document_into_passages(
    document: Document,
    max_words: int = 140,
    stride_sentences: int = 1,
) -> list[Passage]:
    """Create overlapping sentence-window passages from one document."""

    sentences = split_sentences(document.text)
    passages: list[Passage] = []
    current: list[str] = []
    current_words = 0
    passage_index = 0

    for sentence in sentences:
        word_count = len(tokenize(sentence))
        if current and current_words + word_count > max_words:
            text = normalize_text(" ".join(current))
            passages.append(_make_passage(document, passage_index, text))
            passage_index += 1
            keep = current[-stride_sentences:] if stride_sentences > 0 else []
            current = keep[:]
            current_words = sum(len(tokenize(item)) for item in current)
        current.append(sentence)
        current_words += word_count

    if current:
        text = normalize_text(" ".join(current))
        passages.append(_make_passage(document, passage_index, text))

    return passages


def split_documents_into_passages(
    documents: list[Document],
    max_words: int = 140,
    stride_sentences: int = 1,
) -> list[Passage]:
    """Split all candidate documents into passages."""

    passages: list[Passage] = []
    for document in documents:
        passages.extend(
            split_document_into_passages(
                document,
                max_words=max_words,
                stride_sentences=stride_sentences,
            )
        )
    return passages


def first_sentence(text: str, max_words: int = 45) -> str:
    """Return a compact sentence-like claim for extractive fallback generation."""

    sentences = split_sentences(text)
    candidate = sentences[0] if sentences else normalize_text(text)
    words = candidate.split()
    if len(words) > max_words:
        candidate = " ".join(words[:max_words]).rstrip(",;:")
        candidate = f"{candidate}."
    if candidate and candidate[-1] not in ".!?":
        candidate = f"{candidate}."
    return candidate


def _make_passage(document: Document, index: int, text: str) -> Passage:
    return Passage(
        passage_id=f"{document.doc_id}::p{index}",
        doc_id=document.doc_id,
        text=text,
        title=document.title,
        timestamp=document.timestamp,
        metadata=document.metadata,
    )
