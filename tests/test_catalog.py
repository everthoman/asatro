"""Catalog sanity: the full reaction/vocabulary set (the Hartenfeller et al.
reaction SMIRKS set, ported verbatim except one confirmed-buggy SMARTS
scoping fix -- see test_decarboxylative_coupling_matches_reference_educt
below) loads cleanly. Every reaction is role="start"/takes_intermediate=False
-- the old start/extend catalog duplication is gone now that any reaction can
serve as an extend step by picking which of its own slots binds the running
intermediate (see asatro.chemistry.catalog.resolve_step)."""
from rdkit import Chem
from rdkit.Chem import AllChem

from asatro.chemistry.catalog import REACTIONS, REACTION_BY_ID, START_REACTIONS, VOCAB


def test_catalog_size():
    assert len(REACTIONS) == 58
    assert len(START_REACTIONS) == 58  # every reaction is role="start"


def test_no_duplicate_reaction_ids():
    ids = [r["id"] for r in REACTIONS]
    assert len(ids) == len(set(ids))


def test_every_reaction_component_has_a_known_accepts_class():
    for r in REACTIONS:
        for c in r["components"]:
            assert c["accepts"], f"{r['id']}: component '{c['label']}' has no accepts class"
            for cls in c["accepts"]:
                assert cls in VOCAB.query, f"{r['id']}: unknown class '{cls}'"


def test_no_reaction_takes_an_intermediate():
    # No hand-authored "extend" duplicates in this catalog -- see module docstring.
    for r in REACTIONS:
        assert r["role"] == "start" and r["takes_intermediate"] is False, r["id"]


def test_vocab_size():
    assert len(VOCAB.names) == 53


def _product_smiles(rid, smi_a, smi_b):
    rxn = AllChem.ReactionFromSmarts(REACTION_BY_ID[rid]["smarts"])
    a, b = Chem.MolFromSmiles(smi_a), Chem.MolFromSmiles(smi_b)
    prods = rxn.RunReactants((a, b))
    assert prods, f"{rid}: reaction did not fire on {smi_a!r} + {smi_b!r}"
    p = prods[0][0]
    Chem.SanitizeMol(p)
    return Chem.MolToSmiles(p)


def test_decarboxylative_coupling_matches_reference_educt():
    # Regression: Hartenfeller's own SMIRKS for this reaction has a SMARTS
    # scoping bug -- "!OH1" binds "!" only to "O" (per SMARTS grammar), so it
    # means "not-oxygen, exactly 1 H" rather than the clearly-intended "not a
    # hydroxyl". That excludes every real ortho-EWG (nitro, ketone, ...),
    # including Hartenfeller's own reference reagent, from matching at all.
    # Fixed by properly scoping the negation ("!$([OH1])"); this is the one
    # place this port deviates from Hartenfeller's SMIRKS verbatim.
    out = _product_smiles(
        "decarboxylative_coupling", "c1c(C(=O)O)c([N+](=O)[O-])ccc1", "c1ccccc1Br")
    assert out == Chem.CanonSmiles("O=[N+]([O-])c1ccccc1-c1ccccc1")
