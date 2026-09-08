"""Tier-1 handle detection + auto-core derivation."""
from rdkit import Chem
from rdkit.Chem import AllChem

from asatro.chemistry.handles import (
    analyze_fragment,
    bond_order_complaint,
    carve_substructure_3d,
    derive_core,
    detect_fg_classes,
)


def _compat(smiles):
    a = analyze_fragment(smiles)
    return {rid: info for rid, info in a["reactions"].items() if info["compatible"]}


def test_detect_classes():
    assert detect_fg_classes("OC(=O)c1ccncc1") == ["carboxylic_acid"]
    # ortho-aminobenzoic acid also matches the more specific anthranilic_acid class
    assert set(detect_fg_classes("OC(=O)c1ccccc1N")) == {
        "carboxylic_acid", "primary_amine", "anthranilic_acid"}
    assert detect_fg_classes("c1ccccc1") == []  # no handle


def test_acid_fragment_compatible_with_schotten_baumann_amide_slot0():
    info = analyze_fragment("OC(=O)c1ccncc1")["reactions"]["schotten_baumann_amide"]
    assert info["compatible"]
    slots = {s["index"]: s for s in info["slots"]}
    assert 0 in slots and slots[0]["fg_class"] == "carboxylic_acid"
    assert slots[0]["core_smarts"] == "O=Cc1ccncc1"  # hydroxyl drops, carbonyl kept


def test_aryl_halide_fragment_suzuki_keeps_ring():
    # NH-aromatic ring system: the whole ring system is conserved, halide leaves.
    s = analyze_fragment("Brc1ccc2[nH]ccc2c1")
    assert s["reactions"]["suzuki"]["compatible"]
    assert s["reactions"]["schotten_baumann_amide"]["compatible"] is False
    core = derive_core("Brc1ccc2[nH]ccc2c1", "aryl_halide")
    assert core == "c1ccc2[nH]ccc2c1"


def test_boronic_core_keeps_largest_remnant():
    # Dropping B(OH)2 leaves stray O fragments; the phenol ring must win.
    assert derive_core("Oc1ccc(B(O)O)cc1", "boronic") == "Oc1ccccc1"


def test_amine_conserves_whole_fragment():
    # The amine N survives -> no leaving group -> whole fragment is the core.
    assert derive_core("NCc1ccc(O)cc1", "primary_amine") == "NCc1ccc(O)cc1"


def test_ketone_reductive_amination():
    info = analyze_fragment("CC(=O)c1ccncc1")["reactions"]["reductive_amination"]
    assert info["compatible"]
    assert any(s["fg_class"] == "ketone" for s in info["slots"])


def test_no_handle_no_reactions():
    assert _compat("c1ccccc1") == {}


def test_protonated_and_charged_handles_detected():
    # Bound poses from prep are charged at physiological pH; detection must
    # neutralize first (protonated amine NH3+, deprotonated acid COO-).
    # also matches the more specific phenethylamine class (Pictet-Spengler)
    assert set(detect_fg_classes("[NH3+]CCc1ccccc1")) == {"primary_amine", "phenethylamine"}
    assert detect_fg_classes("[O-]C(=O)c1ccncc1") == ["carboxylic_acid"]
    a = analyze_fragment("[NH3+]CCc1ccccc1")
    assert a["fragment_smiles"] == Chem.CanonSmiles("NCCc1ccccc1")   # neutral
    assert a["reactions"]["schotten_baumann_amide"]["compatible"]
    # acid carboxylate still derives the carbonyl-kept core
    assert derive_core("[O-]C(=O)c1ccncc1", "carboxylic_acid") == "O=Cc1ccncc1"


def test_carve_substructure_3d_keeps_coordinates_and_connectivity():
    m = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NCc1ccncc1"))  # an amide product
    AllChem.EmbedMolecule(m, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(m)
    m = Chem.RemoveHs(m)
    core_q = Chem.MolFromSmiles("NCc1ccncc1")  # the amine's own conserved core
    match = m.GetSubstructMatch(core_q)
    assert match

    carved = carve_substructure_3d(m, match)
    assert carved.GetNumConformers() == 1
    assert Chem.MolToSmiles(carved) == Chem.CanonSmiles("NCc1ccncc1")
    # coordinates are the real docked/embedded ones, not a fresh embed
    conf, orig_conf = carved.GetConformer(), m.GetConformer()
    for new_idx, old_idx in enumerate(match):
        new_pt, old_pt = conf.GetAtomPosition(new_idx), orig_conf.GetAtomPosition(old_idx)
        assert (new_pt.x, new_pt.y, new_pt.z) == (old_pt.x, old_pt.y, old_pt.z)


def test_carve_substructure_3d_rejects_ring_cutting_match():
    m = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    m = Chem.RemoveHs(m)
    # A match that includes only part of the aromatic ring can't sanitize.
    import pytest
    with pytest.raises(ValueError, match="not a valid fragment"):
        carve_substructure_3d(m, (0, 1, 2))


# --- bond orders vs. geometry ---------------------------------------------
# A PDB-derived ligand carries guessed bond orders; when the guess is wrong the
# molecule contradicts its own coordinates (see bond_order_complaint).

def _posed(smiles):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=0xB0)
    AllChem.MMFFOptimizeMolecule(m)
    return Chem.RemoveHs(m)


def test_self_consistent_fragment_draws_no_complaint():
    assert bond_order_complaint(_posed("O=C(O)c1ccnc(=O)[nH]1")) is None


def test_puckered_ring_with_sp3_atoms_is_fine():
    """The check keys on flat rings only — a real sp3 ring must pass."""
    assert bond_order_complaint(_posed("OC(=O)C1CCNCC1")) is None


def test_planar_ring_read_as_sp3_is_caught():
    """The real failure: the planar pyrimidinone pose with the 4H tautomer's
    bond orders on it — what OpenBabel returns for a ligand extracted from a
    PDB without CONECT records."""
    posed = _posed("O=C(O)c1ccnc(=O)[nH]1")
    wrong = AllChem.AssignBondOrdersFromTemplate(
        Chem.MolFromSmiles("O=C1N=CCC(C(=O)O)=N1"), posed)   # same pose, 4H bond orders
    msg = bond_order_complaint(wrong)
    assert msg and "planar" in msg and "sp3" in msg
