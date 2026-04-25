from pathlib import Path

from task4_rag.src.data_loader import load_documents, load_task


FIXTURES = Path("task4_rag/tests/fixtures")


def test_loader_supports_fulltext_priority_and_candidate_filtering():
    docs = load_documents(
        FIXTURES / "real_documents.jsonl",
        allowed_doc_ids={"101"},
        text_fields=["fullText", "abstract", "title"],
    )
    assert len(docs) == 1
    assert docs[0].doc_id == "101"
    assert docs[0].text.startswith("Full text evidence")
    assert docs[0].timestamp == "2024-05-01"


def test_load_task_reads_query_docids_style_input():
    instances = load_task(
        queries_path=FIXTURES / "real_query_docids.jsonl",
        documents_path=FIXTURES / "real_documents.jsonl",
        candidates_path=None,
        document_text_fields=["fullText", "abstract", "title"],
    )
    assert len(instances) == 1
    assert instances[0].query.query_id == "q-real-1"
    assert [doc.doc_id for doc in instances[0].documents] == ["101", "202"]
