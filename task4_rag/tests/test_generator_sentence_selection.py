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
