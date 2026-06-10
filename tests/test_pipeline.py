"""Accessibility pre-pass -> growth connection.

Uses a fake growth runner so the full pipeline (prune -> plan targets -> resolve
reactants -> launch) is exercised without invoking gnina.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from asatro.chemistry.handles import derive_core
from asatro.growth import GrowthTarget, grow_accessible, plan_targets


def _bound_sdf(tmp_path, smiles, name="frag.sdf"):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(m)
    m = Chem.RemoveHs(m)
    p = tmp_path / name
    Chem.MolToMolFile(m, str(p))
    return str(p), m


def _slab_pdb(center, normal, radius=6.0, spacing=0.8):
    normal = normal / np.linalg.norm(normal)
    a = np.cross(normal, [1, 0, 0])
    if np.linalg.norm(a) < 1e-3:
        a = np.cross(normal, [0, 1, 0])
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    g = np.arange(-radius, radius + 1e-9, spacing)
    lines = []
    i = 0
    for u in g:
        for v in g:
            p = center + u * a + v * b
            i += 1
            lines.append(f"ATOM  {i:5d}  C   WAL A   1    "
                         f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00           C")
    return "\n".join(lines)


def _recording_runner():
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        return [[ -7.5, "GROWN", "fake" ]]  # pretend a dock happened

    return runner, calls


def _boronic(tmp_path):
    p = tmp_path / "boronic.smi"
    p.write_text("OB(O)c1ccccc1 phB\n")
    return str(p)


def test_plan_targets_from_assessment():
    # Aminobenzoic acid: amide is compatible at BOTH slots (amine and acid).
    from asatro.chemistry.accessibility import assess_fragment
    m = Chem.AddHs(Chem.MolFromSmiles("OC(=O)c1ccccc1N"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    m = Chem.RemoveHs(m)
    targets = plan_targets(assess_fragment(m, np.empty((0, 3))))  # open pocket
    amide = sorted(t.fragment_slot for t in targets if t.reaction_id == "amide")
    assert amide == [0, 1]


def test_open_pocket_grows_surviving_reaction(tmp_path):
    sdf, _ = _bound_sdf(tmp_path, "Brc1ccccc1")   # aryl halide -> suzuki only
    runner, calls = _recording_runner()
    out = grow_accessible(
        fragment_sdf=sdf, receptor_pdb="",        # empty receptor = wide open
        reactant_resolver=lambda rid, ci, acc: _boronic(tmp_path),
        work_dir=str(tmp_path / "runs"), runner=runner)
    assert [t["reaction_id"] for t in out["targets"]] == ["suzuki"]
    assert len(calls) == 1
    call = calls[0]
    assert call["reaction_id"] == "suzuki" and call["fragment_slot"] == 0
    assert call["core_smarts"] == derive_core("Brc1ccccc1", "aryl_halide")
    assert set(call["reactant_files"].keys()) == {1}   # the boronic slot


def test_walled_pocket_prunes_before_growth(tmp_path):
    sdf, mol = _bound_sdf(tmp_path, "Brc1ccccc1")
    from asatro.chemistry.accessibility import growth_vectors
    ev = growth_vectors(mol, "aryl_halide")[0]
    wall = _slab_pdb(ev.attach_pos + ev.direction * 1.6, ev.direction)
    runner, calls = _recording_runner()
    out = grow_accessible(
        fragment_sdf=sdf, receptor_pdb=wall,
        reactant_resolver=lambda rid, ci, acc: _boronic(tmp_path),
        work_dir=str(tmp_path / "runs"), runner=runner)
    assert out["targets"] == []          # suzuki pruned by the wall
    assert calls == []                   # nothing grown


def test_missing_reactant_skips_target(tmp_path):
    sdf, _ = _bound_sdf(tmp_path, "Brc1ccccc1")
    runner, calls = _recording_runner()
    out = grow_accessible(
        fragment_sdf=sdf, receptor_pdb="",
        reactant_resolver=lambda rid, ci, acc: None,   # no library available
        work_dir=str(tmp_path / "runs"), runner=runner)
    assert len(out["targets"]) == 1 and calls == []    # planned but not run
    assert out["runs"][0]["skipped"].startswith("no reactant library")
