import json

from task4_rag.src.rag_quality_evaluator import analyze_rag_quality


def test_rag_quality_scores_context_faithfulness_and_rgb_proxies(tmp_path):
    run_path = tmp_path / "run.jsonl"
    run_record = {
        "metadata": {
            "team_id": "our_team",
            "run_id": "quality_test",
            "type": "automatic",
            "narrative": "How did the later study strengthen the earlier validation?",
            "narrative_id": "q-real-1",
        },
        "references": ["101", "202"],
        "answer": [
            {
                "text": "The later study describes additional validation analysis.",
                "citations": [1],
            }
        ],
    }
    run_path.write_text(json.dumps(run_record) + "\n", encoding="utf-8")

    report = analyze_rag_quality(
        run_path=run_path,
        query_path="task4_rag/tests/fixtures/real_query_docids.jsonl",
        documents_path="task4_rag/tests/fixtures/real_documents.jsonl",
        doc_text_fields=["fullText", "abstract", "title"],
    )

    assert report["quality_scores"]["context_relevance"] > 0
    assert report["quality_scores"]["answer_faithfulness_proxy"] > 0
    assert report["quality_scores"]["answer_relevance_proxy"] > 0
    assert report["rgb_like_abilities"]["information_integration_coverage"] == 1.0
    assert report["per_record_sample"][0]["num_cited_references"] == 1
