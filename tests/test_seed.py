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
    meta = component_route_meta(["schotten_baumann_amide"])
    assert meta == [
        {"reaction_id": "schotten_baumann_amide", "label": "Carboxylic acid",
         "accepts": ["carboxylic_acid"]},
        {"reaction_id": "schotten_baumann_amide", "label": "Amine",
         "accepts": ["primary_amine", "secondary_amine"]},
    ]


def test_component_route_meta_multi_step_matches_route_order():
    # suzuki (2 components) then schotten_baumann_amide reused generically
    # (slot=0, its own acid slot bound to the intermediate) -- same
    # flattening order build_combi_route uses for reagent files.
    meta = component_route_meta(
        ["suzuki", {"reaction_id": "schotten_baumann_amide", "slot": 0}])
    assert [m["reaction_id"] for m in meta] == ["suzuki", "suzuki", "schotten_baumann_amide"]
    assert [m["label"] for m in meta] == ["Boronic acid", "Aryl halide", "Amine"]


def test_component_route_meta_unknown_reaction():
    import pytest
    with pytest.raises(KeyError):
        component_route_meta(["not_a_reaction"])


def test_component_route_meta_generalized_slot_skips_intermediate_component():
    # suzuki reused for step 2 with slot=1 -- the boronic-acid slot binds the
    # intermediate, so only the aryl-halide component (slot 0) should appear
    # for that step; must stay in lockstep with what build_combi_route
    # flattens for the identical steps.
    from asatro.combi import build_combi_route

    steps = ["suzuki", {"reaction_id": "suzuki", "slot": 1}]
    meta = component_route_meta(steps)
    assert [m["reaction_id"] for m in meta] == ["suzuki", "suzuki", "suzuki"]
    assert [m["label"] for m in meta] == ["Boronic acid", "Aryl halide", "Boronic acid"]

    _files, route, _summary = build_combi_route(
        steps, [["boronic1.smi", "halide1.smi"], ["boronic2.smi"]], "/tmp")
    assert len(meta) == sum(n for _s, n, _slot in route)


def test_carve_fragment_recovers_the_amine_with_its_docked_coordinates():
    pose = _docked("CC(=O)NCc1ccncc1")  # amide product
    meta = component_route_meta(["schotten_baumann_amide"])
    carved = carve_fragment(pose, "NCc1ccncc1", meta[1]["accepts"])
    assert Chem.MolToSmiles(carved) == Chem.CanonSmiles("NCc1ccncc1")
    assert carved.GetNumConformers() == 1


def test_carve_fragment_recovers_the_acid():
    pose = _docked("CC(=O)NCc1ccncc1")
    meta = component_route_meta(["schotten_baumann_amide"])
    carved = carve_fragment(pose, "CC(=O)O", meta[0]["accepts"])
    # hydroxyl drops, carbonyl kept -- same conserved-core rule growth uses
    assert Chem.MolToSmiles(carved) == Chem.CanonSmiles("CC=O")


def test_carve_fragment_output_is_growth_ready():
    """The carved fragment round-trips through analyze_fragment the same way
    a real bound fragment would -- it's still amine-compatible, so a follow-up
    growth run can pick amide coupling as step 1 again."""
    pose = _docked("CC(=O)NCc1ccncc1")
    meta = component_route_meta(["schotten_baumann_amide"])
    carved = carve_fragment(pose, "NCc1ccncc1", meta[1]["accepts"])
    a = analyze_fragment(Chem.MolToSmiles(carved))
    assert a["reactions"]["schotten_baumann_amide"]["compatible"]
    assert any(s["fg_class"] == "primary_amine"
              for s in a["reactions"]["schotten_baumann_amide"]["slots"])


def test_carve_fragment_unresolvable_class_raises():
    import pytest
    pose = _docked("CC(=O)NCc1ccncc1")
    with pytest.raises(ValueError, match="doesn't match any of its component's accepted classes"):
        carve_fragment(pose, "NCc1ccncc1", ["carboxylic_acid"])  # wrong class on purpose
