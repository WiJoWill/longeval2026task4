from task4_rag.src import preprocess
from task4_rag.src.data_loader import Document


def test_rule_chunk_mode_matches_existing_splitter():
    document = Document(
        doc_id="d1",
        text=(
            "The first sentence describes wireless throughput selection. "
            "The second sentence adds connectivity evidence for access decisions."
        ),
    )

    rule = preprocess.split_document_into_passages(document, max_words=8, chunk_mode="rule")
    default = preprocess.split_document_into_passages(document, max_words=8)

    assert [passage.text for passage in rule] == [passage.text for passage in default]


def test_semantic_chunking_merges_similar_sentences_and_splits_drift(monkeypatch):
    def fake_encode(sentences, model_name):
        assert model_name == "mock-embedder"
        return [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]

    monkeypatch.setattr(preprocess, "_encode_sentences", fake_encode)
    document = Document(
        doc_id="d1",
        text=(
            "Wireless access selection improves throughput for blocked links. "
            "Connectivity selection also improves throughput under interference. "
            "Protein folding experiments evaluate structural biology results."
        ),
    )

    passages = preprocess.split_document_into_passages(
        document,
        chunk_mode="semantic",
        semantic_chunk_model="mock-embedder",
        semantic_merge_threshold=0.8,
        max_words=100,
    )

    assert [passage.text for passage in passages] == [
        (
            "Wireless access selection improves throughput for blocked links. "
            "Connectivity selection also improves throughput under interference."
        ),
        "Protein folding experiments evaluate structural biology results.",
    ]


def test_topic_shift_chunking_splits_on_embedding_drift(monkeypatch):
    def fake_encode(sentences, model_name):
        assert model_name == "mock-embedder"
        return [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0]]

    monkeypatch.setattr(preprocess, "_encode_sentences", fake_encode)
    document = Document(
        doc_id="d1",
        text=(
            "Wireless access selection improves throughput for blocked links. "
            "Connectivity selection also improves throughput under interference. "
            "Protein folding experiments evaluate structural biology results."
        ),
    )

    passages = preprocess.split_document_into_passages(
        document,
        chunk_mode="topic_shift",
        topic_shift_model="mock-embedder",
        topic_shift_boundary_threshold=0.18,
        max_words=100,
    )

    assert len(passages) == 2
    assert passages[0].text.endswith("under interference.")
    assert passages[1].text == "Protein folding experiments evaluate structural biology results."


def test_semantic_chunking_without_model_falls_back_to_rule():
    document = Document(
        doc_id="d1",
        text=(
            "The first sentence describes wireless throughput selection. "
            "The second sentence adds connectivity evidence for access decisions."
        ),
    )

    fallback = preprocess.split_document_into_passages(document, chunk_mode="semantic", max_words=8)
    rule = preprocess.split_document_into_passages(document, chunk_mode="rule", max_words=8)

    assert [passage.text for passage in fallback] == [passage.text for passage in rule]
