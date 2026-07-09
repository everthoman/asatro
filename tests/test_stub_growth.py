"""Stub-growth refinement: real substituents must physically fit the pocket."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from asatro.chemistry.accessibility import assess_fragment, growth_vectors
from asatro.chemistry.handles import neutralize
from asatro.chemistry.stub_growth import (
    StubParams, assess_with_stubs, build_grown, refine_vector,
)


def _embed(smiles):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=0xA5A)
    AllChem.MMFFOptimizeMolecule(m)
    return Chem.RemoveHs(m)


def _slab(center, normal, radius=5.0, spacing=1.0):
    normal = normal / np.linalg.norm(normal)
    a = np.cross(normal, [1, 0, 0])
    if np.linalg.norm(a) < 1e-3:
        a = np.cross(normal, [0, 1, 0])
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    g = np.arange(-radius, radius + 1e-9, spacing)
    return np.array([center + u * a + v * b for u in g for v in g])


def test_build_grown_removes_handle_and_adds_stub():
    mol = _embed("Brc1ccccc1")
    ev = growth_vectors(mol, "aryl_halide")[0]
    grown, core_pairs, stub_idxs = build_grown(mol, ev.attach_idx, ev.leaving, "C")
    # Br (1 atom) removed, methyl (1 C) added -> same heavy-atom count, no Br.
    assert grown.GetNumAtoms() == mol.GetNumAtoms()
    assert all(a.GetSymbol() != "Br" for a in grown.GetAtoms())
    assert len(stub_idxs) == 1


def test_stub_fits_in_open_pocket():
    mol = _embed("Brc1ccccc1")
    ev = growth_vectors(mol, "aryl_halide")[0]
    res = refine_vector(mol, ev, np.empty((0, 3)))
    assert res["accessible"]
    assert "methyl" in res["fits"] and "phenyl" in res["fits"]


def test_wall_blocks_all_stubs():
    mol = _embed("Brc1ccccc1")
    ev = growth_vectors(mol, "aryl_halide")[0]
    wall = _slab(ev.attach_pos + ev.direction * 1.6, ev.direction, radius=6.0, spacing=0.8)
    res = refine_vector(mol, ev, wall, StubParams(seeds=10))
    assert not res["accessible"] and res["fits"] == []


def test_assess_with_stubs_blocked_vs_open():
    mol = _embed("Brc1ccccc1")
    ev = growth_vectors(mol, "aryl_halide")[0]
    wall = _slab(ev.attach_pos + ev.direction * 1.6, ev.direction, radius=6.0, spacing=0.8)
    blocked = assess_with_stubs(mol, wall, sp=StubParams(seeds=10))
    assert blocked["refined"] is True
    assert blocked["reactions"]["suzuki"]["accessible"] is False
    assert "suzuki" not in blocked["accessible_reactions"]

    # Open pocket: survives both passes and the stub block reports what fits.
    openp = assess_with_stubs(mol, np.empty((0, 3)))
    assert "suzuki" in openp["accessible_reactions"]
    stub = openp["reactions"]["suzuki"]["slots"][0]["stub"]
    assert stub["accessible"] and stub["vectors"][0]["largest_fit"] is not None


def test_build_grown_handles_neutralized_charged_amine():
    # Bound poses come in protonated ([NH3+]); neutralize() restores the
    # neutral amine but *locks* the N's H count (RDKit Uncharger sets
    # noImplicit=True) instead of leaving it on the auto-adjusting implicit-
    # valence path. Growing a stub onto that N (amines have no leaving group
    # to make room) used to overflow its valence and silently fail to
    # sanitize -- build_grown must free a slot itself instead.
    mol = neutralize(_embed("[NH3+]Cc1ccccc1"))
    n = mol.GetAtomWithIdx(0)
    assert n.GetSymbol() == "N" and n.GetNoImplicit()  # locked, as expected

    ev = growth_vectors(mol, "primary_amine")[0]
    built = build_grown(mol, ev.attach_idx, ev.leaving, "C")
    assert built is not None
    grown, _core_pairs, stub_idxs = built
    assert len(stub_idxs) == 1


def test_refine_vector_fits_a_neutralized_charged_amine_in_an_open_pocket():
    mol = neutralize(_embed("[NH3+]Cc1ccccc1"))
    ev = growth_vectors(mol, "primary_amine")[0]
    res = refine_vector(mol, ev, np.empty((0, 3)))
    assert res["accessible"]
    assert res["fits"]


def test_geometric_prune_skips_stub_work():
    # When the geometric probe already kills the whole reaction, the stub pass
    # skips it entirely (no wasted embeddings) -> no 'stub' block is attached.
    mol = _embed("Brc1ccccc1")
    ev = growth_vectors(mol, "aryl_halide")[0]
    wall = _slab(ev.attach_pos + ev.direction * 1.6, ev.direction, radius=6.0, spacing=0.8)
    a = assess_with_stubs(mol, wall, sp=StubParams(seeds=4))
    assert a["reactions"]["suzuki"]["accessible"] is False
    assert "stub" not in a["reactions"]["suzuki"]["slots"][0]
