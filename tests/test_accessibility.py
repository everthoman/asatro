"""Accessibility pre-pass: growth-vector geometry + cone probing + pruning."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from asatro.chemistry.accessibility import (
    ProbeParams, assess_fragment, growth_vectors, load_receptor_atoms, probe_vector,
)


def _embed(smiles):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=0xA5A)
    AllChem.MMFFOptimizeMolecule(m)
    return Chem.RemoveHs(m)


def _slab(center, normal, radius=4.0, spacing=1.2):
    """A dense plane of atoms (a wall) centered at ``center`` with the given
    normal — used to block a growth direction in tests."""
    normal = normal / np.linalg.norm(normal)
    a = np.cross(normal, [1, 0, 0])
    if np.linalg.norm(a) < 1e-3:
        a = np.cross(normal, [0, 1, 0])
    a = a / np.linalg.norm(a)
    b = np.cross(normal, a)
    pts = []
    g = np.arange(-radius, radius + 1e-9, spacing)
    for u in g:
        for v in g:
            pts.append(center + u * a + v * b)
    return np.array(pts)


def test_growth_vector_points_along_c_halide():
    mol = _embed("Brc1ccccc1")
    vecs = growth_vectors(mol, "aryl_halide")
    assert len(vecs) == 1
    ev = vecs[0]
    conf = mol.GetConformer()
    # The Br atom lies along the exit direction from the attachment carbon.
    br = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Br")
    to_br = np.array(conf.GetAtomPosition(br)) - ev.attach_pos
    to_br /= np.linalg.norm(to_br)
    assert float(np.dot(to_br, ev.direction)) > 0.95


def test_open_vector_reaches_max():
    mol = _embed("Brc1ccccc1")
    ev = growth_vectors(mol, "aryl_halide")[0]
    res = probe_vector(ev, np.empty((0, 3)))
    assert res["accessible"] and res["max_free"] == ProbeParams().max_reach


def test_wall_blocks_vector():
    mol = _embed("Brc1ccccc1")
    ev = growth_vectors(mol, "aryl_halide")[0]
    # Put a wall ~2 Å out, square across the growth direction -> nothing can grow.
    wall = _slab(ev.attach_pos + ev.direction * 2.0, ev.direction, radius=5.0, spacing=1.0)
    res = probe_vector(ev, wall)
    assert not res["accessible"]
    assert res["max_free"] < ProbeParams().min_free


def test_assess_prunes_blocked_reaction():
    mol = _embed("Brc1ccccc1")  # aryl halide -> suzuki only
    ev = growth_vectors(mol, "aryl_halide")[0]
    wall = _slab(ev.attach_pos + ev.direction * 2.0, ev.direction, radius=6.0, spacing=1.0)
    blocked = assess_fragment(mol, wall)
    assert blocked["reactions"]["suzuki"]["compatible"]
    assert blocked["reactions"]["suzuki"]["accessible"] is False
    assert "suzuki" not in blocked["accessible_reactions"]

    # Same fragment, empty pocket -> suzuki stays accessible.
    openp = assess_fragment(mol, np.empty((0, 3)))
    assert "suzuki" in openp["accessible_reactions"]


def test_receptor_parser_skips_water_and_h():
    pdb = "\n".join([
        "ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00  0.00           C",
        "ATOM      2  HB1 ALA A   1      11.000  10.000  10.000  1.00  0.00           H",
        "HETATM    3  O   HOH A   2      20.000  20.000  20.000  1.00  0.00           O",
    ])
    at = load_receptor_atoms(pdb)
    assert at.shape == (1, 3)  # only the carbon survives
    assert np.allclose(at[0], [10, 10, 10])
