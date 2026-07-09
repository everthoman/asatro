"""Catalog sanity: the full reaction/vocabulary set (ported from ts-gnina, plus
asatro-specific leaving_smarts for the growth path) loads cleanly."""
from asatro.chemistry.catalog import REACTIONS, REACTION_BY_ID, START_REACTIONS, VOCAB


def test_catalog_size():
    assert len(REACTIONS) == 54
    assert len(START_REACTIONS) == 35
    assert len([r for r in REACTIONS if r["role"] == "extend"]) == 19


def test_no_duplicate_reaction_ids():
    ids = [r["id"] for r in REACTIONS]
    assert len(ids) == len(set(ids))


def test_every_reaction_component_has_a_known_accepts_class():
    for r in REACTIONS:
        for c in r["components"]:
            assert c["accepts"], f"{r['id']}: component '{c['label']}' has no accepts class"
            for cls in c["accepts"]:
                assert cls in VOCAB.query, f"{r['id']}: unknown class '{cls}'"


def test_extend_reactions_take_intermediate_start_reactions_dont():
    for r in REACTIONS:
        assert r["takes_intermediate"] == (r["role"] == "extend"), r["id"]


def test_vocab_size():
    assert len(VOCAB.names) == 28
