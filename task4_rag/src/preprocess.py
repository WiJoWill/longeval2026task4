"""Preprocessing and passage splitting for scientific RAG evidence."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from functools import lru_cache
from math import sqrt
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
    chunk_mode: str = "rule",
    semantic_chunk_model: str | None = None,
    semantic_merge_threshold: float = 0.72,
    topic_shift_model: str | None = None,
    topic_shift_boundary_threshold: float = 0.18,
    min_sentences_per_chunk: int = 1,
    max_sentences_per_chunk: int | None = None,
) -> list[Passage]:
    """Create retrievable passages from one document."""

    if chunk_mode == "rule":
        return _split_document_rule(
            document,
            max_words=max_words,
            stride_sentences=stride_sentences,
        )
    if chunk_mode == "semantic":
        if not semantic_chunk_model:
            _warn_chunk_fallback("semantic", "semantic_chunk_model is not configured")
            return _split_document_rule(document, max_words=max_words, stride_sentences=stride_sentences)
        try:
            return _split_document_semantic(
                document,
                model_name=semantic_chunk_model,
                max_words=max_words,
                merge_threshold=semantic_merge_threshold,
                min_sentences=min_sentences_per_chunk,
                max_sentences=max_sentences_per_chunk,
            )
        except Exception as exc:
            _warn_chunk_fallback("semantic", str(exc))
            return _split_document_rule(document, max_words=max_words, stride_sentences=stride_sentences)
    if chunk_mode == "topic_shift":
        if not topic_shift_model:
            _warn_chunk_fallback("topic_shift", "topic_shift_model is not configured")
            return _split_document_rule(document, max_words=max_words, stride_sentences=stride_sentences)
        try:
            return _split_document_topic_shift(
                document,
                model_name=topic_shift_model,
                max_words=max_words,
                boundary_threshold=topic_shift_boundary_threshold,
                min_sentences=min_sentences_per_chunk,
                max_sentences=max_sentences_per_chunk,
            )
        except Exception as exc:
            _warn_chunk_fallback("topic_shift", str(exc))
            return _split_document_rule(document, max_words=max_words, stride_sentences=stride_sentences)
    raise ValueError(f"Unknown chunk_mode: {chunk_mode}")


def _split_document_rule(
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
    chunk_mode: str = "rule",
    semantic_chunk_model: str | None = None,
    semantic_merge_threshold: float = 0.72,
    topic_shift_model: str | None = None,
    topic_shift_boundary_threshold: float = 0.18,
    min_sentences_per_chunk: int = 1,
    max_sentences_per_chunk: int | None = None,
) -> list[Passage]:
    """Split all candidate documents into passages."""

    passages: list[Passage] = []
    for document in documents:
        passages.extend(
            split_document_into_passages(
                document,
                max_words=max_words,
                stride_sentences=stride_sentences,
                chunk_mode=chunk_mode,
                semantic_chunk_model=semantic_chunk_model,
                semantic_merge_threshold=semantic_merge_threshold,
                topic_shift_model=topic_shift_model,
                topic_shift_boundary_threshold=topic_shift_boundary_threshold,
                min_sentences_per_chunk=min_sentences_per_chunk,
                max_sentences_per_chunk=max_sentences_per_chunk,
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


def _split_document_semantic(
    document: Document,
    model_name: str,
    max_words: int,
    merge_threshold: float,
    min_sentences: int,
    max_sentences: int | None,
) -> list[Passage]:
    sentences = split_sentences(document.text)
    if not sentences:
        return []
    embeddings = _encode_sentences(sentences, model_name)
    chunks: list[list[str]] = []
    current_sentences: list[str] = []
    current_embeddings: list[list[float]] = []

    for sentence, embedding in zip(sentences, embeddings):
        should_split = False
        if current_sentences:
            next_word_count = _sentence_words(current_sentences) + len(tokenize(sentence))
            next_sentence_count = len(current_sentences) + 1
            similarity = _cosine(_average_embedding(current_embeddings), embedding)
            should_split = (
                (len(current_sentences) >= min_sentences and similarity < merge_threshold)
                or next_word_count > max_words
                or (max_sentences is not None and next_sentence_count > max_sentences)
            )
        if should_split:
            chunks.append(current_sentences)
            current_sentences = []
            current_embeddings = []
        current_sentences.append(sentence)
        current_embeddings.append(embedding)

    if current_sentences:
        chunks.append(current_sentences)
    return _chunks_to_passages(document, chunks)


def _split_document_topic_shift(
    document: Document,
    model_name: str,
    max_words: int,
    boundary_threshold: float,
    min_sentences: int,
    max_sentences: int | None,
) -> list[Passage]:
    sentences = split_sentences(document.text)
    if not sentences:
        return []
    embeddings = _encode_sentences(sentences, model_name)
    chunks: list[list[str]] = []
    current: list[str] = []

    for index, sentence in enumerate(sentences):
        should_split = False
        if current:
            next_word_count = _sentence_words(current) + len(tokenize(sentence))
            next_sentence_count = len(current) + 1
            drift = 1.0 - _cosine(embeddings[index - 1], embeddings[index])
            should_split = (
                (len(current) >= min_sentences and drift > boundary_threshold)
                or next_word_count > max_words
                or (max_sentences is not None and next_sentence_count > max_sentences)
            )
        if should_split:
            chunks.append(current)
            current = []
        current.append(sentence)

    if current:
        chunks.append(current)
    return _chunks_to_passages(document, chunks)


def _chunks_to_passages(document: Document, chunks: Sequence[Sequence[str]]) -> list[Passage]:
    passages: list[Passage] = []
    for index, sentences in enumerate(chunks):
        text = normalize_text(" ".join(sentences))
        if text:
            passages.append(_make_passage(document, index, text))
    return passages


def _encode_sentences(sentences: Sequence[str], model_name: str) -> list[list[float]]:
    model = _sentence_transformer(model_name)
    raw_embeddings = model.encode(list(sentences), normalize_embeddings=True)
    return [_as_vector(embedding) for embedding in raw_embeddings]


@lru_cache(maxsize=4)
def _sentence_transformer(model_name: str) -> object:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is not installed") from exc

    return SentenceTransformer(model_name)


def _as_vector(value: object) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()  # type: ignore[assignment]
    return [float(item) for item in value]  # type: ignore[union-attr]


def _average_embedding(embeddings: Sequence[Sequence[float]]) -> list[float]:
    if not embeddings:
        return []
    width = len(embeddings[0])
    return [sum(embedding[index] for embedding in embeddings) / len(embeddings) for index in range(width)]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _sentence_words(sentences: Sequence[str]) -> int:
    return sum(len(tokenize(sentence)) for sentence in sentences)


def _warn_chunk_fallback(chunk_mode: str, reason: str) -> None:
    print(f"Warning: falling back to rule chunking for {chunk_mode}: {reason}", file=sys.stderr)
