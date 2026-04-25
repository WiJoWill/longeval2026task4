"""Heuristic and optional model-based reranking for evidence passages."""

from __future__ import annotations

from typing import Sequence

from .preprocess import Passage, tokenize
from .query_expansion import extract_keywords, has_temporal_intent


def rerank_passages(
    query: str,
    passages: Sequence[Passage],
    fused_scores: Sequence[float],
    title_boost: float = 0.15,
    temporal_boost: float = 0.1,
    citation_graph_boost: float = 0.05,
    citation_priors: dict[str, float] | None = None,
    cross_encoder_scores: Sequence[float] | None = None,
) -> list[float]:
    """Combine fused retrieval scores with lightweight evidence-quality features."""

    query_terms = set(extract_keywords(query, limit=20))
    temporal_query = has_temporal_intent(query)
    reranked: list[float] = []

    for index, (passage, fused_score) in enumerate(zip(passages, fused_scores)):
        text_tokens = tokenize(passage.text)
        title_tokens = tokenize(passage.title)
        text_vocab = set(text_tokens)
        title_vocab = set(title_tokens)
        coverage = len(query_terms & text_vocab) / max(len(query_terms), 1)
        title_match = len(query_terms & title_vocab) / max(len(query_terms), 1)
        phrase_bonus = _query_phrase_bonus(query, passage.text)
        temporal_feature = 1.0 if temporal_query and passage.timestamp else 0.0
        citation_feature = (citation_priors or {}).get(passage.doc_id, 0.0)
        score = fused_score
        score += 0.35 * coverage
        score += title_boost * title_match
        score += 0.15 * phrase_bonus
        score += temporal_boost * temporal_feature
        score += citation_graph_boost * citation_feature
        if cross_encoder_scores is not None and index < len(cross_encoder_scores):
            score += 0.25 * cross_encoder_scores[index]
        reranked.append(score)
    return reranked


def optional_cross_encoder_scores(
    query: str,
    passages: Sequence[Passage],
    model_name: str | None,
) -> list[float] | None:
    """Run an optional cross-encoder reranker if configured and available."""

    if not model_name:
        return None
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except ImportError:
        return None

    model = CrossEncoder(model_name)
    pairs = [(query, f"{passage.title}\n\n{passage.text}".strip()) for passage in passages]
    scores = model.predict(pairs)
    return [float(score) for score in scores]


def _query_phrase_bonus(query: str, passage_text: str) -> float:
    query_tokens = extract_keywords(query, limit=10)
    if len(query_tokens) < 2:
        return 0.0
    joined_passage = " ".join(tokenize(passage_text))
    hits = 0
    possible = 0
    for left, right in zip(query_tokens, query_tokens[1:]):
        possible += 1
        if f"{left} {right}" in joined_passage:
            hits += 1
    return hits / possible if possible else 0.0
