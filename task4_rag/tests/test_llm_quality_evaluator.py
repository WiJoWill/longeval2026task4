from task4_rag.src.llm_quality_evaluator import _normalize_judgment, _score_means


def test_normalize_llm_judgment_clamps_scores_and_keeps_notes():
    judgment = _normalize_judgment(
        {
            "query_id": "q1",
            "record_index": 2,
            "scores": {
                "context_relevance": 5,
                "answer_relevance": "4",
                "faithfulness": 7,
                "overall": 0,
            },
            "qualitative": {
                "strengths": ["Grounded"],
                "weaknesses": "Incomplete",
                "failure_modes": ["partial_answer"],
                "recommended_fix": "Use stronger evidence.",
            },
        }
    )

    assert judgment["scores"]["context_relevance"] == 5
    assert judgment["scores"]["answer_relevance"] == 4
    assert judgment["scores"]["faithfulness"] == 5
    assert judgment["scores"]["overall"] == 1
    assert judgment["qualitative"]["weaknesses"] == ["Incomplete"]


def test_score_means_averages_all_dimensions():
    judgments = [
        _normalize_judgment({"scores": {"overall": 5, "faithfulness": 4}}),
        _normalize_judgment({"scores": {"overall": 3, "faithfulness": 2}}),
    ]

    means = _score_means(judgments)

    assert means["overall"] == 4
    assert means["faithfulness"] == 3
