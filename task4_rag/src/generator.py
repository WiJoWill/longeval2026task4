"""Answer generation strategies for LongEval-RAG JSONL output."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .data_loader import Query
from .evidence_ranker import ScoredPassage, build_reference_list
from .preprocess import Passage, first_sentence, normalize_text, split_sentences, tokenize
from .query_expansion import extract_keywords, has_temporal_intent
from .temporal import parse_year
from .validator import validate_record


BOILERPLATE_PREFIXES = (
    "this paper",
    "in this paper",
    "this article",
    "in this article",
    "this study",
    "in this study",
    "we present",
    "we propose",
    "we introduce",
    "we describe",
)

LOW_INFORMATION_PHRASES = (
    "has attracted much attention",
    "plays an important role",
    "is an important",
    "is a challenging task",
    "in recent years",
    "with the development of",
)

OCR_ARTIFACT_RE = re.compile(r"(.)\1{4,}|[_=|]{3,}|[^\w\s.,;:()\[\]/%+\-'<>=&]", flags=re.IGNORECASE)
BROKEN_TEXT_RE = re.compile(r"[a-z]\.[A-Z]|\b\w+-\s+\w+\b")
LOOSE_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
TEMPORAL_SENTENCE_TERMS = {"earlier", "later", "before", "after", "previous", "subsequent", "follow-up", "followup"}
TERM_NORMALIZATIONS = {"validation": "validate", "validated": "validate", "validating": "validate", "selecting": "select"}


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for answer generation."""

    team_id: str = "our_team"
    run_id: str = "caes_rag_rrf_v1"
    run_type: str = "automatic"
    mode: str = "hybrid_evidence_rag_v1"
    max_answer_sentences: int = 5
    provider: str = "none"
    model: str = ""
    temperature: float = 0.0
    prompts_dir: str | Path | None = None
    temporal_templates: bool = True
    sentence_rerank_model: str = ""
    sentence_rerank_top_n: int = 24


class AnswerGenerator:
    """Generate Task 4 answer records from selected evidence."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self._sentence_reranker: Any | None = None
        self._sentence_reranker_failed = False

    def generate_from_evidence(self, query: Query, evidence: Sequence[ScoredPassage]) -> dict[str, Any]:
        """Generate a full TREC RAG record from ranked evidence passages."""

        references = build_reference_list(evidence)
        passages = [item.passage for item in evidence]
        if self.config.mode == "extractive_evidence_only":
            answer = self._extractive_answer(query.text, references, passages, temporal_templates=False)
        elif self.config.provider == "openai" and self.config.model:
            answer = self._generate_with_openai(query, references, passages)
        else:
            answer = self._extractive_answer(
                query.text,
                references,
                passages,
                temporal_templates=self.config.temporal_templates,
            )

        record = self._record(query=query, references=references, answer=answer)
        validate_record(record)
        return record

    def generate_concat_baseline(self, query: Query, passages: Sequence[Passage], top_k: int = 6) -> dict[str, Any]:
        """Build a simple non-LLM baseline from the first candidate passages."""

        selected = list(passages)[:top_k]
        references: list[str] = []
        for passage in selected:
            if passage.doc_id not in references:
                references.append(passage.doc_id)
        answer = self._extractive_answer(query.text, references, selected, temporal_templates=False)
        record = self._record(query=query, references=references, answer=answer)
        validate_record(record)
        return record

    def generate_llm_all_docs(self, query: Query, passages: Sequence[Passage]) -> dict[str, Any]:
        """Generate with all candidate passages, or fall back to extractive output."""

        references: list[str] = []
        for passage in passages:
            if passage.doc_id not in references:
                references.append(passage.doc_id)
        if self.config.provider == "openai" and self.config.model:
            answer = self._generate_with_openai(query, references, list(passages))
        else:
            answer = self._extractive_answer(query.text, references, list(passages), temporal_templates=False)
        record = self._record(query=query, references=references, answer=answer)
        validate_record(record)
        return record

    def _extractive_answer(
        self,
        query_text: str,
        references: list[str],
        passages: Sequence[Passage],
        temporal_templates: bool = True,
    ) -> list[dict[str, Any]]:
        sentence_candidates = self._sentence_candidates(query_text, references, passages)
        answer: list[dict[str, Any]] = []
        used_texts: set[str] = set()
        used_citation_sets: set[tuple[int, ...]] = set()

        if temporal_templates and has_temporal_intent(query_text):
            temporal_sentences = self._temporal_answer_candidates(references, sentence_candidates)
            for item in temporal_sentences:
                key = item["text"].lower()
                if key not in used_texts:
                    answer.append(item)
                    used_texts.add(key)
                    used_citation_sets.add(tuple(item["citations"]))
                if len(answer) >= self.config.max_answer_sentences:
                    return answer

        for candidate in self._ordered_answer_candidates(sentence_candidates):
            self._append_candidate_answer(
                answer=answer,
                candidate=candidate,
                used_texts=used_texts,
                used_citation_sets=used_citation_sets,
                avoid_reused_citations=temporal_templates and has_temporal_intent(query_text),
            )
            if len(answer) >= self.config.max_answer_sentences:
                break

        if not answer:
            answer.extend(self._fallback_cited_answer(references, passages))
        return answer

    def _sentence_candidates(
        self,
        query_text: str,
        references: list[str],
        passages: Sequence[Passage],
    ) -> list[dict[str, Any]]:
        query_terms = set(extract_keywords(query_text, limit=20))
        temporal_query = has_temporal_intent(query_text)
        scored: list[dict[str, Any]] = []
        seen: set[tuple[str, tuple[int, ...]]] = set()

        for passage_rank, passage in enumerate(passages):
            if passage.doc_id not in references:
                continue
            citation_index = references.index(passage.doc_id)
            title_terms = set(extract_keywords(passage.title, limit=12))
            for sentence_rank, sentence in enumerate(_candidate_sentences(passage.text)[:5]):
                text = normalize_text(sentence)
                token_count = len(tokenize(text))
                if token_count < 6:
                    continue
                if token_count > 60:
                    text = first_sentence(text, max_words=45)
                    token_count = len(tokenize(text))
                if not self._is_useful_sentence(text, query_terms, title_terms, token_count, temporal_query):
                    continue
                signature = (text.lower(), (citation_index,))
                if signature in seen:
                    continue
                seen.add(signature)
                heuristic_score = self._sentence_score(text, query_terms, title_terms, sentence_rank)
                heuristic_score += max(0.0, 0.15 - 0.03 * passage_rank)
                heuristic_score += max(0.0, 0.08 - 0.03 * sentence_rank)
                scored.append(
                    {
                        "text": text,
                        "citations": [citation_index],
                        "score": heuristic_score,
                        "heuristic_score": heuristic_score,
                        "doc_id": passage.doc_id,
                        "timestamp": passage.timestamp,
                        "sentence_rank": sentence_rank,
                    }
                )
        scored = sorted(scored, key=lambda item: item["score"], reverse=True)
        return self._rerank_sentence_candidates(query_text, scored)

    def _rerank_sentence_candidates(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates or not self.config.sentence_rerank_model or self._sentence_reranker_failed:
            return candidates

        top_n = self.config.sentence_rerank_top_n
        candidates_to_score = candidates if top_n <= 0 else candidates[:top_n]
        try:
            scores = self._score_with_sentence_reranker(
                query_text=query_text,
                sentence_texts=[candidate["text"] for candidate in candidates_to_score],
            )
        except Exception as exc:
            self._sentence_reranker_failed = True
            print(f"Warning: sentence reranker unavailable; using heuristic sentence ranking: {exc}", file=sys.stderr)
            return candidates
        if len(scores) != len(candidates_to_score):
            self._sentence_reranker_failed = True
            print(
                "Warning: sentence reranker returned an unexpected number of scores; "
                "using heuristic sentence ranking.",
                file=sys.stderr,
            )
            return candidates

        reranked: list[dict[str, Any]] = []
        for candidate, score in zip(candidates_to_score, scores):
            updated = dict(candidate)
            updated["cross_encoder_score"] = float(score)
            updated["score"] = float(score)
            reranked.append(updated)
        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked + candidates[len(candidates_to_score) :]

    def _score_with_sentence_reranker(self, query_text: str, sentence_texts: Sequence[str]) -> list[float]:
        if self._sentence_reranker is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is not installed") from exc
            self._sentence_reranker = CrossEncoder(self.config.sentence_rerank_model)

        pairs = [(query_text, sentence_text) for sentence_text in sentence_texts]
        raw_scores = self._sentence_reranker.predict(pairs)
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        if isinstance(raw_scores, (int, float)):
            raw_scores = [raw_scores]
        return [float(score) for score in raw_scores]

    def _temporal_answer_candidates(
        self,
        references: list[str],
        sentence_candidates: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        best_by_doc: dict[str, dict[str, Any]] = {}
        for candidate in sentence_candidates:
            existing = best_by_doc.get(candidate["doc_id"])
            if existing is None or candidate["score"] > existing["score"]:
                best_by_doc[candidate["doc_id"]] = candidate

        dated = []
        for doc_id, candidate in best_by_doc.items():
            year = parse_year(candidate.get("timestamp"))
            if year is None:
                continue
            dated.append((year, doc_id, candidate))
        if len(dated) < 2:
            return []

        dated.sort(key=lambda item: (item[0], -item[2]["score"], item[1]))
        early_year, early_doc_id, early = dated[0]
        late_year, late_doc_id, late = dated[-1]
        if early_doc_id == late_doc_id:
            return []

        return [
            {
                "text": _temporalize_sentence(f"Earlier evidence ({early_year}) reports that", early["text"]),
                "citations": early["citations"],
            },
            {
                "text": _temporalize_sentence(f"Later evidence ({late_year}) reports that", late["text"]),
                "citations": late["citations"],
            },
        ]

    def _ordered_answer_candidates(self, sentence_candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prefer the best sentence from each document, then fill from remaining evidence."""

        best_by_doc: dict[str, dict[str, Any]] = {}
        for candidate in sentence_candidates:
            existing = best_by_doc.get(candidate["doc_id"])
            if existing is None or candidate["score"] > existing["score"]:
                best_by_doc[candidate["doc_id"]] = candidate

        primary = sorted(best_by_doc.values(), key=lambda item: item["score"], reverse=True)
        primary_ids = {id(item) for item in primary}
        secondary = [item for item in sentence_candidates if id(item) not in primary_ids]
        return primary + secondary

    def _append_candidate_answer(
        self,
        answer: list[dict[str, Any]],
        candidate: dict[str, Any],
        used_texts: set[str],
        used_citation_sets: set[tuple[int, ...]],
        avoid_reused_citations: bool,
    ) -> None:
        text = candidate["text"]
        citation_tuple = tuple(candidate["citations"])
        if text.lower() in used_texts:
            return
        if citation_tuple in used_citation_sets and avoid_reused_citations:
            return
        answer.append({"text": text, "citations": candidate["citations"]})
        used_texts.add(text.lower())
        used_citation_sets.add(citation_tuple)

    def _is_useful_sentence(
        self,
        text: str,
        query_terms: set[str],
        title_terms: set[str],
        token_count: int,
        temporal_query: bool,
    ) -> bool:
        if token_count < 6 or token_count > 70:
            return False
        stripped = text.strip()
        if not stripped or stripped[-1] not in ".!?":
            return False
        if OCR_ARTIFACT_RE.search(stripped) or BROKEN_TEXT_RE.search(stripped):
            return False

        lowered = stripped.lower()
        if any(lowered.startswith(prefix) for prefix in BOILERPLATE_PREFIXES):
            return False
        if any(phrase in lowered for phrase in LOW_INFORMATION_PHRASES):
            return False

        text_terms = set(tokenize(stripped))
        query_overlap = _term_overlap_count(query_terms, text_terms)
        title_overlap = len(title_terms & text_terms)
        temporal_overlap = bool(temporal_query and (TEMPORAL_SENTENCE_TERMS & text_terms))
        if query_terms and query_overlap == 0 and not temporal_overlap:
            return False
        return True

    def _sentence_score(self, text: str, query_terms: set[str], title_terms: set[str], sentence_rank: int) -> float:
        text_terms = set(tokenize(text))
        query_overlap = _term_overlap_count(query_terms, text_terms) / max(len(query_terms), 1)
        title_overlap = len(title_terms & text_terms) / max(len(title_terms), 1) if title_terms else 0.0
        specificity = min(len(text_terms) / 28.0, 1.0)
        early_sentence_penalty = 0.05 if sentence_rank == 0 and _starts_with_boilerplate_style(text) else 0.0
        return (0.72 * query_overlap) + (0.18 * title_overlap) + (0.10 * specificity) - early_sentence_penalty

    def _fallback_cited_answer(self, references: list[str], passages: Sequence[Passage]) -> list[dict[str, Any]]:
        for passage in passages:
            if passage.doc_id not in references:
                continue
            citation_index = references.index(passage.doc_id)
            for sentence in _candidate_sentences(passage.text):
                text = first_sentence(normalize_text(sentence), max_words=36)
                token_count = len(tokenize(text))
                if 6 <= token_count <= 70 and not OCR_ARTIFACT_RE.search(text):
                    return [{"text": text, "citations": [citation_index]}]
        return [{"text": "No supported answer could be generated from the provided candidate documents.", "citations": []}]

    def _generate_with_openai(
        self,
        query: Query,
        references: list[str],
        passages: Sequence[Passage],
    ) -> list[dict[str, Any]]:
        system_prompt, user_template = self._load_prompts()
        user_prompt = user_template.format(
            narrative_id=query.query_id,
            narrative=query.text,
            max_answer_sentences=self.config.max_answer_sentences,
            evidence_json=json.dumps(_evidence_payload(references, passages), ensure_ascii=False, indent=2),
        )
        try:
            raw_text = self._call_openai(system_prompt=system_prompt, user_prompt=user_prompt)
            answer = _parse_answer_json(raw_text)
        except Exception:
            answer = self._extractive_answer(query.text, references, passages, temporal_templates=self.config.temporal_templates)
        return _repair_answer(answer, references, self.config.max_answer_sentences)

    def _load_prompts(self) -> tuple[str, str]:
        prompts_dir = Path(self.config.prompts_dir or Path(__file__).resolve().parents[1] / "prompts")
        system_prompt = (prompts_dir / "rag_system_prompt.txt").read_text(encoding="utf-8")
        user_template = (prompts_dir / "rag_user_prompt_template.txt").read_text(encoding="utf-8")
        return system_prompt, user_template

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI response was empty")
        return content

    def _record(self, query: Query, references: list[str], answer: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "metadata": {
                "team_id": self.config.team_id,
                "run_id": self.config.run_id,
                "type": self.config.run_type,
                "narrative": query.text,
                "narrative_id": query.query_id,
            },
            "references": references,
            "answer": answer,
        }


def _evidence_payload(references: list[str], passages: Sequence[Passage]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for passage in passages:
        if passage.doc_id not in references:
            continue
        payload.append(
            {
                "reference_index": references.index(passage.doc_id),
                "doc_id": passage.doc_id,
                "timestamp": passage.timestamp,
                "title": passage.title,
                "text": normalize_text(passage.text),
            }
        )
    return payload


def _candidate_sentences(text: str) -> list[str]:
    """Split passage text a little more aggressively for noisy full text snippets."""

    sentences: list[str] = []
    for coarse_sentence in split_sentences(text):
        sentences.extend(part.strip() for part in LOOSE_SENTENCE_BOUNDARY_RE.split(coarse_sentence) if part.strip())
    return sentences or split_sentences(text)


def _parse_answer_json(raw_text: str) -> list[dict[str, Any]]:
    raw_text = raw_text.strip()
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if isinstance(parsed, dict) and isinstance(parsed.get("answer"), list):
        return parsed["answer"]
    if isinstance(parsed, list):
        return parsed
    raise ValueError("LLM output did not contain an answer list")


def _repair_answer(answer: list[dict[str, Any]], references: list[str], max_sentences: int) -> list[dict[str, Any]]:
    repaired: list[dict[str, Any]] = []
    for item in answer:
        text = normalize_text(str(item.get("text", "")))
        if not text:
            continue
        citations = item.get("citations", [])
        if not isinstance(citations, list):
            citations = []
        clean_citations: list[int] = []
        for citation in citations:
            if isinstance(citation, int) and 0 <= citation < len(references) and citation not in clean_citations:
                clean_citations.append(citation)
        if clean_citations or not repaired:
            repaired.append({"text": text, "citations": clean_citations})
        if len(repaired) >= max_sentences:
            break
    if not repaired:
        repaired.append({"text": "No supported answer could be generated from the provided candidate documents.", "citations": []})
    return repaired


def _temporalize_sentence(prefix: str, text: str) -> str:
    claim = normalize_text(text).rstrip(".!?")
    if claim:
        claim = claim[0].lower() + claim[1:] if len(claim) > 1 else claim.lower()
    sentence = f"{prefix} {claim}.".strip()
    return sentence


def _starts_with_boilerplate_style(text: str) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(prefix) for prefix in BOILERPLATE_PREFIXES)


def _term_overlap_count(query_terms: set[str], text_terms: set[str]) -> int:
    if not query_terms or not text_terms:
        return 0
    text_stems = {_light_stem(term) for term in text_terms}
    return sum(1 for term in query_terms if term in text_terms or _light_stem(term) in text_stems)


def _light_stem(term: str) -> str:
    if term in TERM_NORMALIZATIONS:
        return TERM_NORMALIZATIONS[term]
    for suffix in ("ations", "ation", "ments", "ment", "ing", "ies", "ied", "ed", "es", "s"):
        if len(term) > len(suffix) + 3 and term.endswith(suffix):
            if suffix == "ies":
                return f"{term[:-3]}y"
            if suffix == "ied":
                return f"{term[:-3]}y"
            return term[: -len(suffix)]
    return term
