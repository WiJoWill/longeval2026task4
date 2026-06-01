from task4_rag.src.evaluator import analyze_run


def test_evaluator_flags_empty_citations_and_filler():
    summary = analyze_run(
        "task4_rag/tests/fixtures/generated_responses_sample.jsonl",
        "task4_rag/tests/fixtures/real_query_docids.jsonl",
    )
    report = summary.report
    assert report["records_in_run"] == 1
    assert report["answer_items_without_citations"] > 0
    assert report["filler_answer_items"] > 0
    assert report["reference_exact_match_rate"] == 1.0
