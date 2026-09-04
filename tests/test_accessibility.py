"""Accessibility pre-pass: growth-vector geometry + cone probing + pruning."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from asatro.chemistry.accessibility import (
    ProbeParams, assess_fragment, growth_vectors, load_receptor_atoms, probe_vector,
)


def _carboxyl_atoms(mol):
    """(carbonyl C, =O, -OH) of the acid in ``mol``."""
    return mol.GetSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2H1]"))


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


# --- handle rotamers -------------------------------------------------------
# The bound pose shows one torsion state of a handle; growth is not bound to it.

def test_carboxyl_flip_opens_a_vector_blocked_as_posed():
    """A wall against the -OH must not prune the acid: the carboxyl turns 180°
    about the ring bond, putting the new bond where the C=O sits (~121° away,
    far outside the probe cone), which is open."""
    mol = _embed("Cc1ccc(cc1)C(=O)O")
    conf = mol.GetConformer()
    c, _o_carbonyl, o_h = _carboxyl_atoms(mol)
    pos = lambda i: np.array(conf.GetAtomPosition(i))
    to_oh = pos(o_h) - pos(c); to_oh /= np.linalg.norm(to_oh)
    wall = _slab(pos(o_h) + to_oh * 2.8, to_oh, radius=9.0, spacing=1.2)

    ev = growth_vectors(mol, "carboxylic_acid")[0]
    assert [r.angle for r in ev.rotamers] == [0.0, 180.0]
    assert ev.free_atoms == (_o_carbonyl,)      # the C=O moves with the flip
    res = probe_vector(ev, wall)
    assert res["accessible"] and res["rotamer"] == 180.0
    assert "schotten_baumann_amide" in assess_fragment(mol, wall)["accessible_reactions"]


def test_enclosed_carboxyl_is_still_pruned():
    """Rotamers open real space, not any space: with both C-O directions walled
    in, the handle must still be pruned."""
    mol = _embed("Cc1ccc(cc1)C(=O)O")
    conf = mol.GetConformer()
    c = _carboxyl_atoms(mol)[0]
    centre = np.array(conf.GetAtomPosition(c))
    shell = np.array([centre + np.array([x, y, z])
                      for x in np.arange(-8, 8.1, 1.0)
                      for y in np.arange(-8, 8.1, 1.0)
                      for z in np.arange(-8, 8.1, 1.0)
                      if 4.0 <= np.linalg.norm([x, y, z]) <= 5.0])
    assert assess_fragment(mol, shell)["accessible_reactions"] == []


def test_a_rotamer_that_buries_the_handle_is_rejected():
    """The turn has to be physically available: a wall where the C=O would land
    blocks that rotamer, leaving only the as-posed direction."""
    mol = _embed("Cc1ccc(cc1)C(=O)O")
    conf = mol.GetConformer()
    _c, o_carbonyl, _o_h = _carboxyl_atoms(mol)
    ev = growth_vectors(mol, "carboxylic_acid")[0]
    flipped_o = ev.rotamers[1].free_xyz[0]
    # a wall right on top of the flipped C=O position
    wall = _slab(flipped_o, flipped_o - ev.attach_pos, radius=4.0, spacing=1.0)
    res = probe_vector(ev, wall)
    assert res["rotamer"] == 0.0


def test_ring_bound_handles_have_no_phantom_rotamers():
    """An aryl halide's direction is fixed by the ring — the bond into the core
    is aromatic, so there is nothing to turn about."""
    mol = _embed("Brc1ccccc1")
    ev = growth_vectors(mol, "aryl_halide")[0]
    assert [r.angle for r in ev.rotamers] == [0.0] and ev.free_atoms == ()
