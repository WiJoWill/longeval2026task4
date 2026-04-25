from task4_rag.src.evaluator import analyze_run


def test_evaluator_flags_empty_citations_and_filler():
    summary = analyze_run(
        "data/generated-responses.jsonl",
        "data/task4_longeval_rag-query_docids.jsonl",
    )
    report = summary.report
    assert report["records_in_run"] == 47
    assert report["answer_items_without_citations"] > 0
    assert report["filler_answer_items"] > 0
    assert report["reference_exact_match_rate"] == 1.0
