import pytest

from task4_rag.src.validator import ValidationError, validate_record


def _record(citations):
    return {
        "metadata": {
            "team_id": "our_team",
            "run_id": "test_run",
            "type": "automatic",
            "narrative": "A query",
            "narrative_id": "q1",
        },
        "references": ["doc_a"],
        "answer": [{"text": "A supported claim.", "citations": citations}],
    }


def test_citation_index_must_point_into_references():
    with pytest.raises(ValidationError):
        validate_record(_record([1]))


def test_citation_indices_are_integers():
    with pytest.raises(ValidationError):
        validate_record(_record(["0"]))


def test_empty_citations_are_allowed_for_no_supported_answer():
    validate_record(_record([]))
