from task4_rag.src.baseline_repair import repair_answer_items


def test_repair_answer_items_drops_filler_and_keeps_cited_items():
    repaired = repair_answer_items(
        [
            {"text": "I will next answer the question.", "citations": []},
            {"text": "Therefore, we have to consider the following aspects.", "citations": []},
            {"text": "A relevant cited claim.", "citations": [1]},
            {"text": "Another relevant cited claim.", "citations": [0, 2]},
        ],
        num_references=3,
        max_answer_sentences=5,
    )
    assert repaired == [
        {"text": "A relevant cited claim.", "citations": [1]},
        {"text": "Another relevant cited claim.", "citations": [0, 2]},
    ]
