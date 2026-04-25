from task4_rag.src.generator import AnswerGenerator, GenerationConfig
from task4_rag.src.preprocess import Passage


def test_temporal_templates_produce_time_anchored_sentences():
    generator = AnswerGenerator(GenerationConfig())
    references = ["d1", "d2"]
    passages = [
        Passage(passage_id="d1::p0", doc_id="d1", text="The earlier system estimates a 10 kg payload through simulation.", timestamp="2021-10-11"),
        Passage(passage_id="d2::p0", doc_id="d2", text="The later study adds stress and thrust analysis to validate the same payload.", timestamp="2022-09-08"),
    ]
    answer = generator._extractive_answer(
        query_text="How did the later conceptual drone design strengthen validation of the agricultural transport drone proposed earlier?",
        references=references,
        passages=passages,
        temporal_templates=True,
    )
    assert answer[0]["text"].startswith("Earlier evidence (2021)")
    assert answer[1]["text"].startswith("Later evidence (2022)")
