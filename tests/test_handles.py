"""Tier-1 handle detection + auto-core derivation."""
from asatro.chemistry.handles import analyze_fragment, derive_core, detect_fg_classes


def _compat(smiles):
    a = analyze_fragment(smiles)
    return {rid: info for rid, info in a["reactions"].items() if info["compatible"]}


def test_detect_classes():
    assert detect_fg_classes("OC(=O)c1ccncc1") == ["carboxylic_acid"]
    assert set(detect_fg_classes("OC(=O)c1ccccc1N")) == {"carboxylic_acid", "primary_amine"}
    assert detect_fg_classes("c1ccccc1") == []  # no handle


def test_acid_fragment_compatible_with_amide_slot1():
    info = analyze_fragment("OC(=O)c1ccncc1")["reactions"]["amide"]
    assert info["compatible"]
    slots = {s["index"]: s for s in info["slots"]}
    assert 1 in slots and slots[1]["fg_class"] == "carboxylic_acid"
    assert slots[1]["core_smarts"] == "O=Cc1ccncc1"  # hydroxyl drops, carbonyl kept


def test_aryl_halide_fragment_suzuki_keeps_ring():
    # NH-aromatic ring system: the whole ring system is conserved, halide leaves.
    s = analyze_fragment("Brc1ccc2[nH]ccc2c1")
    assert s["reactions"]["suzuki"]["compatible"]
    assert s["reactions"]["amide"]["compatible"] is False
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
