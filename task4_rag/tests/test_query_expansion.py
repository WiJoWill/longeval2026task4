from task4_rag.src.query_expansion import build_query_variants, has_temporal_intent


def test_build_query_variants_adds_keyword_and_temporal_views():
    variants = build_query_variants(
        "How did the later conceptual drone design strengthen validation of the agricultural transport drone proposed earlier?"
    )
    variant_names = {variant.name for variant in variants}
    assert "original" in variant_names
    assert "keywords" in variant_names
    assert "temporal" in variant_names
    assert has_temporal_intent(variants[0].text)
