"""Carving a growth-ready fragment out of one reagent's contribution to a
docked hit (asatro/seed.py) -- the "seed growth from a combi result" flow.
"""
from rdkit import Chem
from rdkit.Chem import AllChem

from asatro.chemistry.handles import analyze_fragment
from asatro.seed import carve_fragment, component_route_meta


def _docked(smiles, seed=7):
    """Stand-in for a real docked pose: embed + minimize, matching how the
    rest of the test suite fakes a "docked" 3D structure."""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return Chem.RemoveHs(m)


def test_component_route_meta_single_step():
    meta = component_route_meta(["amide"])
    assert meta == [
        {"reaction_id": "amide", "label": "Primary amine", "accepts": ["primary_amine"]},
        {"reaction_id": "amide", "label": "Carboxylic acid", "accepts": ["carboxylic_acid"]},
    ]


def test_component_route_meta_multi_step_matches_route_order():
    # suzuki (2 components) then suzuki_ext_halide (1 component) -- same
    # flattening order build_combi_route uses for reagent files.
    meta = component_route_meta(["suzuki", "suzuki_ext_halide"])
    assert [m["reaction_id"] for m in meta] == ["suzuki", "suzuki", "suzuki_ext_halide"]
    assert [m["label"] for m in meta] == ["Aryl halide", "Boronic acid", "Boronic acid"]


def test_component_route_meta_unknown_reaction():
    import pytest
    with pytest.raises(KeyError):
        component_route_meta(["not_a_reaction"])


def test_carve_fragment_recovers_the_amine_with_its_docked_coordinates():
    pose = _docked("CC(=O)NCc1ccncc1")  # amide product
    meta = component_route_meta(["amide"])
    carved = carve_fragment(pose, "NCc1ccncc1", meta[0]["accepts"])
    assert Chem.MolToSmiles(carved) == Chem.CanonSmiles("NCc1ccncc1")
    assert carved.GetNumConformers() == 1


def test_carve_fragment_recovers_the_acid():
    pose = _docked("CC(=O)NCc1ccncc1")
    meta = component_route_meta(["amide"])
    carved = carve_fragment(pose, "CC(=O)O", meta[1]["accepts"])
    # hydroxyl drops, carbonyl kept -- same conserved-core rule growth uses
    assert Chem.MolToSmiles(carved) == Chem.CanonSmiles("CC=O")


def test_carve_fragment_output_is_growth_ready():
    """The carved fragment round-trips through analyze_fragment the same way
    a real bound fragment would -- it's still amine-compatible, so a follow-up
    growth run can pick amide coupling as step 1 again."""
    pose = _docked("CC(=O)NCc1ccncc1")
    meta = component_route_meta(["amide"])
    carved = carve_fragment(pose, "NCc1ccncc1", meta[0]["accepts"])
    a = analyze_fragment(Chem.MolToSmiles(carved))
    assert a["reactions"]["amide"]["compatible"]
    assert any(s["fg_class"] == "primary_amine" for s in a["reactions"]["amide"]["slots"])


def test_carve_fragment_unresolvable_class_raises():
    import pytest
    pose = _docked("CC(=O)NCc1ccncc1")
    with pytest.raises(ValueError, match="doesn't match any of its component's accepted classes"):
        carve_fragment(pose, "NCc1ccncc1", ["carboxylic_acid"])  # wrong class on purpose
