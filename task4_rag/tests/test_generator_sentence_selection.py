import json

from task4_rag.src.data_loader import Query
from task4_rag.src.generator import AnswerGenerator, GenerationConfig
from task4_rag.src.preprocess import Passage


def test_sentence_filter_drops_boilerplate_and_artifacts():
    generator = AnswerGenerator(GenerationConfig(max_answer_sentences=3))
    passages = [
        Passage(
            passage_id="d1::p0",
            doc_id="d1",
            title="wireless throughput selection",
            text=(
                "In this paper, we propose a wireless system. "
                "The device avoids futile access attempts by predicting blocked links before throughput selection. "
                "aaaaa ____ malformed OCR fragment."
            ),
        )
    ]

    answer = generator._extractive_answer(
        query_text="How can a device avoid futile access attempts while selecting connectivity for throughput?",
        references=["d1"],
        passages=passages,
        temporal_templates=False,
    )

    assert answer == [
        {
            "text": "The device avoids futile access attempts by predicting blocked links before throughput selection.",
            "citations": [0],
        }
    ]


def test_answer_selector_prefers_one_strong_sentence_per_document_before_filling():
    generator = AnswerGenerator(GenerationConfig(max_answer_sentences=3))
    candidates = [
        {"text": "Doc one strongest throughput claim.", "citations": [0], "score": 0.9, "doc_id": "d1"},
        {"text": "Doc one second throughput claim.", "citations": [0], "score": 0.8, "doc_id": "d1"},
        {"text": "Doc two connectivity claim.", "citations": [1], "score": 0.7, "doc_id": "d2"},
    ]

    ordered = generator._ordered_answer_candidates(candidates)

    assert [item["text"] for item in ordered] == [
        "Doc one strongest throughput claim.",
        "Doc two connectivity claim.",
        "Doc one second throughput claim.",
    ]


def test_sentence_reranker_reorders_filtered_candidates(monkeypatch):
    generator = AnswerGenerator(
        GenerationConfig(
            max_answer_sentences=3,
            sentence_rerank_model="mock-cross-encoder",
            sentence_rerank_top_n=3,
        )
    )
    passages = [
        Passage(
            passage_id="d1::p0",
            doc_id="d1",
            title="wireless throughput selection",
            text=(
                "Wireless throughput selection uses a weak baseline for connectivity decisions. "
                "Wireless throughput selection predicts blocked links for connectivity decisions."
            ),
        )
    ]

    def fake_scores(query_text, sentence_texts):
        assert "avoid futile access" in query_text
        assert len(sentence_texts) == 2
        return [0.1, 0.9]

    monkeypatch.setattr(generator, "_score_with_sentence_reranker", fake_scores)

    candidates = generator._sentence_candidates(
        query_text="How can wireless throughput selection avoid futile access in connectivity decisions?",
        references=["d1"],
        passages=passages,
    )

    assert candidates[0]["text"] == "Wireless throughput selection predicts blocked links for connectivity decisions."
    assert candidates[0]["cross_encoder_score"] == 0.9


def test_multi_doc_synthesis_emits_multi_citation_answer():
    generator = AnswerGenerator(GenerationConfig(max_answer_sentences=3, multi_doc_synthesis=True))
    passages = [
        Passage(
            passage_id="d1::p0",
            doc_id="d1",
            title="wireless throughput selection",
            text="Wireless throughput selection predicts blocked links before futile access attempts.",
        ),
        Passage(
            passage_id="d2::p0",
            doc_id="d2",
            title="connectivity access decisions",
            text="Connectivity decisions avoid futile access by choosing stronger throughput links.",
        ),
    ]

    answer = generator._extractive_answer(
        query_text="How can wireless throughput selection avoid futile access in connectivity decisions?",
        references=["d1", "d2"],
        passages=passages,
        temporal_templates=False,
    )

    assert len(answer[0]["citations"]) == 2
    assert set(answer[0]["citations"]) == {0, 1}
    assert "Across the cited evidence" in answer[0]["text"]


def test_multi_doc_synthesis_can_be_disabled():
    generator = AnswerGenerator(GenerationConfig(max_answer_sentences=3, multi_doc_synthesis=False))
    passages = [
        Passage(
            passage_id="d1::p0",
            doc_id="d1",
            title="wireless throughput selection",
            text="Wireless throughput selection predicts blocked links before futile access attempts.",
        ),
        Passage(
            passage_id="d2::p0",
            doc_id="d2",
            title="connectivity access decisions",
            text="Connectivity decisions avoid futile access by choosing stronger throughput links.",
        ),
    ]

    answer = generator._extractive_answer(
        query_text="How can wireless throughput selection avoid futile access in connectivity decisions?",
        references=["d1", "d2"],
        passages=passages,
        temporal_templates=False,
    )

    assert all(len(item["citations"]) == 1 for item in answer)


def test_answer_candidate_selector_uses_score_threshold_and_limit():
    generator = AnswerGenerator(
        GenerationConfig(
            answer_candidate_score_threshold=0.55,
            answer_candidate_score_margin=0.0,
            max_selected_answer_candidates=3,
        )
    )
    candidates = [
        {"text": "A", "score": 0.9},
        {"text": "B", "score": 0.6},
        {"text": "C", "score": 0.5},
        {"text": "D", "score": 0.8},
    ]

    selected = generator._select_answer_sentence_candidates(candidates)

    assert [item["text"] for item in selected] == ["A", "B"]


def test_answer_candidate_selector_uses_top_score_margin():
    generator = AnswerGenerator(
        GenerationConfig(
            answer_candidate_score_threshold=0.0,
            answer_candidate_score_margin=0.2,
            max_selected_answer_candidates=50,
        )
    )
    candidates = [
        {"text": "A", "score": 0.9},
        {"text": "B", "score": 0.75},
        {"text": "C", "score": 0.65},
    ]

    selected = generator._select_answer_sentence_candidates(candidates)

    assert [item["text"] for item in selected] == ["A", "B"]


def test_answer_candidate_selector_keeps_top_one_when_filter_is_too_strict():
    generator = AnswerGenerator(
        GenerationConfig(
            answer_candidate_score_threshold=2.0,
            answer_candidate_score_margin=0.0,
            max_selected_answer_candidates=50,
        )
    )
    candidates = [
        {"text": "A", "score": 0.9},
        {"text": "B", "score": 0.8},
    ]

    selected = generator._select_answer_sentence_candidates(candidates)

    assert [item["text"] for item in selected] == ["A"]


def test_openai_generation_uses_sentence_evidence_and_preserves_multi_citations(monkeypatch):
    generator = AnswerGenerator(
        GenerationConfig(
            provider="openai",
            model="mock-openai",
            max_answer_sentences=3,
            multi_doc_synthesis=True,
            answer_candidate_score_margin=0.0,
        )
    )
    passages = [
        Passage(
            passage_id="d1::p0",
            doc_id="d1",
            title="wireless throughput selection",
            text=(
                "Wireless throughput selection predicts blocked links before futile access attempts. "
                "This whole passage sentence should not be sent unless it passes filtering."
            ),
        ),
        Passage(
            passage_id="d2::p0",
            doc_id="d2",
            title="connectivity access decisions",
            text="Connectivity decisions avoid futile access by choosing stronger throughput links.",
        ),
    ]
    captured = {}

    def fake_call(system_prompt, user_prompt):
        captured["user_prompt"] = user_prompt
        return json.dumps(
            {
                "answer": [
                    {
                        "text": "Blocked-link prediction and stronger throughput links can jointly avoid futile access attempts.",
                        "evidence_ids": [0, 1],
                    }
                ]
            }
        )

    monkeypatch.setattr(generator, "_call_openai", fake_call)

    answer = generator._generate_with_openai(
        Query(query_id="q1", text="How can wireless throughput selection avoid futile access?"),
        references=["d1", "d2"],
        passages=passages,
    )

    assert answer == [
        {
            "text": "Blocked-link prediction and stronger throughput links can jointly avoid futile access attempts.",
            "citations": [0, 1],
        }
    ]
    assert '"evidence_id"' in captured["user_prompt"]
    assert '"evidence_id": 0' in captured["user_prompt"]
    assert '"evidence_id": 1' in captured["user_prompt"]
    assert "whole passage sentence should not be sent" not in captured["user_prompt"]
