"""Deterministic multi-query expansion for candidate-constrained retrieval."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .preprocess import tokenize


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "later",
    "of",
    "on",
    "or",
    "should",
    "still",
    "that",
    "the",
    "their",
    "this",
    "through",
    "to",
    "using",
    "what",
    "when",
    "which",
    "while",
    "with",
}

TEMPORAL_HINTS = {
    "earlier",
    "later",
    "before",
    "after",
    "strengthen",
    "improve",
    "evolve",
    "evolution",
    "subsequent",
    "previous",
    "follow-up",
    "followup",
    "validation",
    "compare",
    "comparison",
}

METHOD_HINTS = {"method", "approach", "model", "framework", "technique", "algorithm"}


@dataclass(frozen=True)
class QueryVariant:
    """One retrieval query variant for multi-query fusion."""

    name: str
    text: str


def build_query_variants(
    query_text: str,
    max_variants: int = 5,
    prf_passages: Sequence[str] | None = None,
) -> list[QueryVariant]:
    """Build deterministic lexical query variants without outside knowledge."""

    variants: list[QueryVariant] = [QueryVariant(name="original", text=query_text.strip())]
    seen = {query_text.strip().lower()}

    keywords = extract_keywords(query_text)
    if keywords:
        keyword_text = " ".join(keywords)
        _append_variant(variants, seen, "keywords", keyword_text)

    if has_temporal_intent(query_text):
        temporal_keywords = keywords[:]
        temporal_keywords.extend(["earlier", "later", "validation", "extension", "comparison"])
        _append_variant(variants, seen, "temporal", " ".join(_dedupe(temporal_keywords)))

    if any(token in METHOD_HINTS for token in tokenize(query_text)):
        method_keywords = keywords[:]
        method_keywords.extend(["method", "approach", "framework"])
        _append_variant(variants, seen, "method", " ".join(_dedupe(method_keywords)))

    if prf_passages:
        expansion_terms = pseudo_relevance_terms(query_text, prf_passages)
        if expansion_terms:
            combined = " ".join(_dedupe(keywords + expansion_terms))
            _append_variant(variants, seen, "prf", combined)

    return variants[:max_variants]


def has_temporal_intent(query_text: str) -> bool:
    tokens = set(tokenize(query_text))
    return bool(tokens & TEMPORAL_HINTS)


def extract_keywords(query_text: str, limit: int = 12) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokenize(query_text):
        if token in STOPWORDS or len(token) < 3:
            continue
        if token not in seen:
            seen.add(token)
            keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def pseudo_relevance_terms(query_text: str, passages: Sequence[str], top_n: int = 6) -> list[str]:
    """Extract expansion terms from top passages for a lightweight PRF variant."""

    query_terms = set(extract_keywords(query_text, limit=20))
    counts: Counter[str] = Counter()
    for passage in passages:
        for token in tokenize(passage):
            if token in STOPWORDS or len(token) < 4 or token in query_terms:
                continue
            counts[token] += 1
    ranked = [term for term, _ in counts.most_common(top_n * 2)]
    return ranked[:top_n]


def _append_variant(
    variants: list[QueryVariant],
    seen: set[str],
    name: str,
    text: str,
) -> None:
    normalized = " ".join(text.split()).strip().lower()
    if normalized and normalized not in seen:
        seen.add(normalized)
        variants.append(QueryVariant(name=name, text=text.strip()))


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output
