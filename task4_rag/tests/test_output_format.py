import json

from task4_rag.src.validator import validate_record


def test_valid_trec_rag_record_shape():
    record = {
        "metadata": {
            "team_id": "our_team",
            "run_id": "caes_rag_rrf_v1",
            "type": "automatic",
            "narrative": "What does the evidence say?",
            "narrative_id": "q1",
        },
        "references": ["doc_a", "doc_b"],
        "answer": [
            {"text": "The evidence reports a finding.", "citations": [0]},
            {"text": "A later document reports a related update.", "citations": [1]},
        ],
    }

    validate_record(record)
    assert json.loads(json.dumps(record))["metadata"]["narrative_id"] == "q1"
