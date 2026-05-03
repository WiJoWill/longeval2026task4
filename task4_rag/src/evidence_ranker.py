"""Evidence ranking for LongEval-RAG candidate documents."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .preprocess import Passage, tokenize
from .query_expansion import build_query_variants, has_temporal_intent
from .reranker import optional_cross_encoder_scores, rerank_passages
from .temporal import choose_earliest_and_latest


@dataclass(frozen=True)
class ScoredPassage:
    """A passage with retrieval and reranking scores."""

    passage: Passage
    score: float
    lexical_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


class BM25Ranker:
    """Small Okapi BM25 implementation for deterministic lexical retrieval."""

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.texts = list(texts)
        self.k1 = k1
        self.b = b
        self.tokenized = [tokenize(text) for text in self.texts]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_frequencies = [Counter(tokens) for tokens in self.tokenized]
        self.document_frequencies: dict[str, int] = defaultdict(int)
        for tokens in self.tokenized:
            for token in set(tokens):
                self.document_frequencies[token] += 1

    def score(self, query: str) -> list[float]:
        query_tokens = tokenize(query)
        n_docs = len(self.texts)
        scores: list[float] = []
        for doc_index, term_frequency in enumerate(self.term_frequencies):
            doc_length = self.doc_lengths[doc_index]
            score = 0.0
            for token in query_tokens:
                frequency = term_frequency.get(token, 0)
                if frequency == 0:
                    continue
                document_frequency = self.document_frequencies.get(token, 0)
                idf = math.log(1 + (n_docs - document_frequency + 0.5) / (document_frequency + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * doc_length / (self.avgdl or 1))
                score += idf * frequency * (self.k1 + 1) / denominator
            scores.append(score)
        return scores


class EvidenceRanker:
    """Rank and select a compact, diverse, temporal-aware set of evidence passages."""

    def __init__(
        self,
        method: str = "rrf",
        dense_model: str | None = None,
        hybrid_alpha: float = 0.25,
        per_doc_limit: int | None = None,
        rrf_k: int = 60,
        enable_query_expansion: bool = True,
        enable_prf: bool = True,
        prf_top_passages: int = 4,
        preselect_k: int = 24,
        rerank_top_n: int = 24,
        enable_rerank: bool = True,
        title_boost: float = 0.15,
        temporal_boost: float = 0.1,
        citation_graph_path: str | None = None,
        citation_graph_boost: float = 0.05,
        reranker_model: str | None = None,
        query_variant_limit: int = 5,
    ) -> None:
        self.method = method
        self.dense_model = dense_model
        self.hybrid_alpha = hybrid_alpha
        self.per_doc_limit = per_doc_limit
        self.rrf_k = rrf_k
        self.enable_query_expansion = enable_query_expansion
        self.enable_prf = enable_prf
        self.prf_top_passages = prf_top_passages
        self.preselect_k = preselect_k
        self.rerank_top_n = rerank_top_n
        self.enable_rerank = enable_rerank
        self.title_boost = title_boost
        self.temporal_boost = temporal_boost
        self.citation_graph_boost = citation_graph_boost
        self.reranker_model = reranker_model
        self.query_variant_limit = query_variant_limit
        self.citation_priors = _load_citation_priors(citation_graph_path) if citation_graph_path else {}

    def rank(self, query: str, passages: Sequence[Passage]) -> list[ScoredPassage]:
        passages = list(passages)
        if not passages:
            return []

        base_texts = [passage.text for passage in passages]
        title_texts = [f"{passage.title} {passage.text}".strip() for passage in passages]
        body_ranker = BM25Ranker(base_texts)
        title_ranker = BM25Ranker(title_texts)

        original_scores = self._combine_lexical_scores(
            body_ranker.score(query),
            title_ranker.score(query),
        )
        lexical_order = _sorted_indices(original_scores)
        ranked_lists = [lexical_order]

        variants = build_query_variants(query, max_variants=self.query_variant_limit)
        if not self.enable_query_expansion:
            variants = variants[:1]

        prf_texts = [passages[index].text for index in ranked_lists[0][: self.prf_top_passages]]
        if self.enable_prf and prf_texts:
            variants = build_query_variants(
                query,
                max_variants=self.query_variant_limit,
                prf_passages=prf_texts,
            )

        lexical_scores_by_index = original_scores[:]
        for variant in variants[1:]:
            variant_scores = self._combine_lexical_scores(
                body_ranker.score(variant.text),
                title_ranker.score(variant.text),
            )
            ranked_lists.append(_sorted_indices(variant_scores))
            lexical_scores_by_index = [
                max(current, candidate) for current, candidate in zip(lexical_scores_by_index, variant_scores)
            ]

        dense_scores = self._dense_scores(query, passages)
        if any(score != 0.0 for score in dense_scores):
            ranked_lists.append(_sorted_indices(dense_scores))

        if self.method == "bm25":
            combined_scores = original_scores
            fused_order = lexical_order[: min(self.preselect_k, self.rerank_top_n)]
        else:
            combined_scores = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)
            fused_order = _sorted_indices(combined_scores)[: min(self.preselect_k, self.rerank_top_n)]
        preselected_passages = [passages[index] for index in fused_order]
        preselected_scores = [combined_scores[index] for index in fused_order]
        if self.enable_rerank:
            cross_scores = optional_cross_encoder_scores(query, preselected_passages, self.reranker_model)
            reranked_scores = rerank_passages(
                query=query,
                passages=preselected_passages,
                fused_scores=preselected_scores,
                title_boost=self.title_boost,
                temporal_boost=self.temporal_boost,
                citation_graph_boost=self.citation_graph_boost,
                citation_priors=self.citation_priors,
                cross_encoder_scores=cross_scores,
            )
        else:
            reranked_scores = preselected_scores

        results: list[ScoredPassage] = []
        for local_index, global_index in enumerate(fused_order):
            results.append(
                ScoredPassage(
                    passage=passages[global_index],
                    score=reranked_scores[local_index],
                    lexical_score=lexical_scores_by_index[global_index],
                    dense_score=dense_scores[global_index],
                    rrf_score=combined_scores[global_index],
                    rerank_score=reranked_scores[local_index],
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)

    def select(self, query: str, passages: Sequence[Passage], top_k: int = 8) -> list[ScoredPassage]:
        """Select top evidence with diversity and temporal coverage."""

        ranked = self.rank(query, passages)
        selected: list[ScoredPassage] = []
        counts_by_doc: dict[str, int] = defaultdict(int)
        seen_text: set[str] = set()
        forced_doc_ids: list[str] = []

        if has_temporal_intent(query):
            earliest_doc, latest_doc = choose_earliest_and_latest(
                (item.passage.doc_id, item.passage.timestamp, item.score) for item in ranked[: self.preselect_k]
            )
            forced_doc_ids = [doc_id for doc_id in (earliest_doc, latest_doc) if doc_id]

        for forced_doc_id in forced_doc_ids:
            for item in ranked:
                if item.passage.doc_id != forced_doc_id:
                    continue
                normalized_text = " ".join(tokenize(item.passage.text)[:40])
                if normalized_text in seen_text:
                    break
                selected.append(item)
                counts_by_doc[forced_doc_id] += 1
                seen_text.add(normalized_text)
                break

        for item in ranked:
            doc_id = item.passage.doc_id
            normalized_text = " ".join(tokenize(item.passage.text)[:40])
            if (self.per_doc_limit is not None and counts_by_doc[doc_id] >= self.per_doc_limit) or normalized_text in seen_text:
                continue
            selected.append(item)
            counts_by_doc[doc_id] += 1
            seen_text.add(normalized_text)
            if len(selected) >= top_k:
                break
        return selected[:top_k]

    def _combine_lexical_scores(self, body_scores: Sequence[float], title_scores: Sequence[float]) -> list[float]:
        body_norm = _minmax(body_scores)
        title_norm = _minmax(title_scores)
        return [
            (1 - self.title_boost) * body_score + self.title_boost * title_score
            for body_score, title_score in zip(body_norm, title_norm)
        ]

    def _dense_scores(self, query: str, passages: Sequence[Passage]) -> list[float]:
        """Optional sentence-transformer scoring, falling back to lexical-only if unavailable."""

        if not self.dense_model:
            return [0.0] * len(passages)
        try:
            from sentence_transformers import SentenceTransformer, util  # type: ignore
        except ImportError:
            return [0.0] * len(passages)

        model = SentenceTransformer(self.dense_model)
        query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        passage_embeddings = model.encode(
            [f"{passage.title}\n\n{passage.text}".strip() for passage in passages],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        similarities = util.cos_sim(query_embedding, passage_embeddings)[0]
        return _minmax([float(score) for score in similarities])


def build_reference_list(scored_passages: Sequence[ScoredPassage]) -> list[str]:
    """Return cited document IDs in first-evidence order."""

    references: list[str] = []
    seen: set[str] = set()
    for item in scored_passages:
        doc_id = item.passage.doc_id
        if doc_id not in seen:
            seen.add(doc_id)
            references.append(doc_id)
    return references


def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[int]], k: int = 60) -> list[float]:
    """Fuse multiple ranked lists by reciprocal rank fusion."""

    max_index = 0
    for ranked in ranked_lists:
        if ranked:
            max_index = max(max_index, max(ranked))
    scores = [0.0] * (max_index + 1)
    for ranked in ranked_lists:
        for rank, index in enumerate(ranked, start=1):
            scores[index] += 1.0 / (k + rank)
    return scores


def _sorted_indices(values: Sequence[float]) -> list[int]:
    return sorted(range(len(values)), key=lambda index: values[index], reverse=True)


def _load_citation_priors(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    counts: Counter[str] = Counter()
    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for field_name in ("citing_doc_id", "cited_doc_id"):
                doc_id = (row.get(field_name) or "").strip()
                if doc_id:
                    counts[doc_id] += 1
    if not counts:
        return {}
    max_count = max(counts.values())
    return {doc_id: value / max_count for doc_id, value in counts.items()}


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [1.0 if max_value > 0 else 0.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]
