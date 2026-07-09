"""Catalog sanity: the full reaction/vocabulary set (ported from ts-gnina, plus
asatro-specific leaving_smarts for the growth path) loads cleanly."""
from rdkit import Chem
from rdkit.Chem import AllChem

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


def _fires_keeping_both_tags(rid, smi_a, smi_b, tag_a, tag_b):
    """RunReactants the given reagents through reaction ``rid`` and confirm
    the product contains both reagents' distinguishing tags -- catches a
    reaction template silently discarding one reagent's substituent (an
    unmapped "rest of the molecule" atom that's never referenced on the
    product side, so RDKit's reaction engine drops it and everything past
    it)."""
    rxn = AllChem.ReactionFromSmarts(REACTION_BY_ID[rid]["smarts"])
    a, b = Chem.MolFromSmiles(smi_a), Chem.MolFromSmiles(smi_b)
    prods = rxn.RunReactants((a, b))
    assert prods, f"{rid}: reaction did not fire on {smi_a!r} + {smi_b!r}"
    p = prods[0][0]
    Chem.SanitizeMol(p)
    smi = Chem.MolToSmiles(p)
    assert tag_a in smi, f"{rid}: reagent A's tag ({tag_a}) missing from product {smi!r}"
    assert tag_b in smi, f"{rid}: reagent B's tag ({tag_b}) missing from product {smi!r}"


def test_sonogashira_keeps_the_alkyne_reagents_own_substituent():
    # Regression: the alkyne pattern's R-group atom used to be unmapped and
    # silently dropped, so every sonogashira product was just a bare
    # aryl-C#CH with the real alkyne reagent discarded entirely.
    _fires_keeping_both_tags("sonogashira", "Brc1ccc(F)cc1", "C#Cc1ccc(Cl)cc1", "F", "Cl")


def test_urea_keeps_the_isocyanates_own_substituent():
    # Regression: same bug class -- the isocyanate's R-group atom was
    # unmapped and dropped, so every urea product lost the isocyanate
    # reagent entirely (just NH2-C(=O)-NH-R from the amine side).
    _fires_keeping_both_tags("urea", "NCc1ccc(F)cc1", "O=C=NCc1ccc(Cl)cc1", "F", "Cl")


def test_c_c_decarboxylation_keeps_the_ketoesters_own_substituent():
    # Regression: the aryl halide used to bond to the *ester* carbonyl
    # (wrong regiochemistry) while the real ketone carbon and its R-group
    # were discarded as if they were the leaving group.
    _fires_keeping_both_tags(
        "c_c_decarboxylation", "Brc1ccc(F)cc1", "O=C(Cc1ccc(Cl)cc1)C(=O)OCC", "F", "Cl")
