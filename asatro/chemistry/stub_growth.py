"""Stub-growth refinement of the accessibility pre-pass.

The geometric cone probe (accessibility.py) asks "is the *ray* open?". This pass
asks the stronger question "does a real *substituent* fit?" — it grows a few
minimal stubs (–Me, –Ph, morpholine) onto the conserved core in 3D, pinned to the
fragment's bound pose, and keeps a growth vector only if some stub places without
clashing into the receptor. It catches the cases the ray misses (a thin channel a
ray slips through but a flat phenyl cannot), at the cost of a handful of
constrained embeddings per vector.

No docking and no GNINA — just constrained embedding + a heavy-atom clash check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from asatro.chemistry.accessibility import ExitVector, growth_vectors
from asatro.chemistry.handles import analyze_fragment

# Stubs, ordered small -> large. Each attaches through atom 0 of its SMILES.
# "fits" if *any* of these places cleanly; the largest that fits is a richness cue.
DEFAULT_STUBS: Tuple[Tuple[str, str], ...] = (
    ("methyl", "C"),
    ("phenyl", "c1ccccc1"),
    ("morpholine", "N1CCOCC1"),
)


@dataclass
class StubParams:
    seeds: int = 8               # constrained-embed attempts per stub (orientation sampling)
    clash_radius: float = 2.4    # stub heavy atom within this of a receptor atom = clash (Å)
    stubs: Tuple[Tuple[str, str], ...] = DEFAULT_STUBS


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation R and translation t that best superpose P onto Q (row-vector
    points: ``P @ R.T + t ≈ Q``)."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Q.mean(0) - R @ P.mean(0)


def build_grown(fragment: Chem.Mol, attach_idx: int, leaving: tuple, stub_smiles: str):
    """Conserved core (fragment minus ``leaving``) + ``stub`` bonded at the
    attachment atom. Returns ``(mol, core_pairs, stub_idxs)`` where ``core_pairs``
    maps grown-atom index -> original fragment index (so we can pin them to the
    bound pose), or ``None`` if the product won't sanitize."""
    frag = Chem.Mol(fragment)
    for a in frag.GetAtoms():
        a.SetIntProp("_o", a.GetIdx())
    stub = Chem.MolFromSmiles(stub_smiles)
    if stub is None:
        return None
    for a in stub.GetAtoms():
        a.SetBoolProp("_stub", True)
    stub.GetAtomWithIdx(0).SetBoolProp("_sattach", True)

    combo = Chem.RWMol(Chem.CombineMols(frag, stub))
    idx = lambda pred: [a.GetIdx() for a in combo.GetAtoms() if pred(a)]
    attach_c = idx(lambda a: a.HasProp("_o") and a.GetIntProp("_o") == attach_idx)[0]
    sattach_c = idx(lambda a: a.HasProp("_sattach"))[0]
    combo.AddBond(attach_c, sattach_c, Chem.BondType.SINGLE)
    for j in sorted(idx(lambda a: a.HasProp("_o") and a.GetIntProp("_o") in set(leaving)),
                    reverse=True):
        combo.RemoveAtom(j)

    mol = combo.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    core_pairs = [(a.GetIdx(), a.GetIntProp("_o")) for a in mol.GetAtoms() if a.HasProp("_o")]
    stub_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.HasProp("_stub")]
    return mol, core_pairs, stub_idxs


def place_stub(fragment: Chem.Mol, grown: Chem.Mol, seed: int) -> Optional[np.ndarray]:
    """Constrained-embed ``grown`` with the core pinned to the fragment's bound
    coordinates, then rigid-align the core onto the exact bound pose. Returns the
    stub heavy-atom coordinates in the receptor frame, or ``None`` on failure.

    Hydrogens are added before embedding (better geometry); the core/stub atom
    indices are recovered from the ``_o``/``_stub`` tags, which survive AddHs."""
    conf = fragment.GetConformer()
    m = Chem.AddHs(Chem.Mol(grown))
    core_pairs = [(a.GetIdx(), a.GetIntProp("_o")) for a in m.GetAtoms() if a.HasProp("_o")]
    stub_idxs = [a.GetIdx() for a in m.GetAtoms() if a.HasProp("_stub")]
    coord_map = {gi: conf.GetAtomPosition(oi) for gi, oi in core_pairs}
    cid = AllChem.EmbedMolecule(m, coordMap=coord_map, randomSeed=seed)
    if cid < 0:
        cid = AllChem.EmbedMolecule(m, coordMap=coord_map, randomSeed=seed,
                                    useRandomCoords=True)
        if cid < 0:
            return None
    c = m.GetConformer()
    P = np.array([list(c.GetAtomPosition(gi)) for gi, _ in core_pairs])
    Q = np.array([list(conf.GetAtomPosition(oi)) for _, oi in core_pairs])
    R, t = _kabsch(P, Q)
    stub = np.array([list(c.GetAtomPosition(si)) for si in stub_idxs])
    if stub.size == 0:
        return stub
    return stub @ R.T + t


def _clash(stub_xyz: np.ndarray, receptor: np.ndarray, radius: float) -> bool:
    if stub_xyz.size == 0 or receptor.size == 0:
        return False
    near = receptor[np.linalg.norm(receptor - stub_xyz.mean(0), axis=1) <= 15.0]
    if near.size == 0:
        return False
    r2 = radius * radius
    for p in stub_xyz:
        if float(np.min(np.sum((near - p) ** 2, axis=1))) < r2:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-vector and fragment-level refinement
# ---------------------------------------------------------------------------
def refine_vector(fragment: Chem.Mol, ev: ExitVector, receptor: np.ndarray,
                  p: StubParams = StubParams()) -> dict:
    """Which stubs physically fit on this growth vector. Accessible if any does."""
    fits: List[str] = []
    for name, smi in p.stubs:
        built = build_grown(fragment, ev.attach_idx, ev.leaving, smi)
        if built is None:
            continue
        grown, _core_pairs, _stub_idxs = built
        for s in range(p.seeds):
            xyz = place_stub(fragment, grown, 0xC0FFEE + s)
            if xyz is None:
                continue
            if not _clash(xyz, receptor, p.clash_radius):
                fits.append(name)
                break
    return {
        "attach_idx": ev.attach_idx,
        "fits": fits,
        "largest_fit": fits[-1] if fits else None,
        "accessible": bool(fits),
    }


def refine_assessment(mol: Chem.Mol, receptor: np.ndarray, assessment: dict,
                      p: StubParams = StubParams()) -> dict:
    """Refine the geometric pre-pass result: for reactions/slots that survived the
    cone probe, grow stubs and tighten ``accessible`` to the stub verdict. Slots
    the geometry already pruned are left pruned (skipped, not re-opened)."""
    for rid, info in assessment["reactions"].items():
        if not info.get("accessible"):
            continue
        rxn_ok = False
        for slot in info["slots"]:
            if not slot.get("accessible"):
                slot["stub"] = {"accessible": False, "skipped": "geometric-pruned"}
                continue
            vecs = growth_vectors(mol, slot["fg_class"])
            results = [refine_vector(mol, ev, receptor, p) for ev in vecs]
            fits = (not results) or any(r["accessible"] for r in results)
            slot["stub"] = {"vectors": results, "accessible": bool(fits)}
            slot["accessible"] = bool(fits)
            rxn_ok = rxn_ok or bool(fits)
        info["accessible"] = rxn_ok
    assessment["accessible_reactions"] = sorted(
        rid for rid, info in assessment["reactions"].items() if info.get("accessible"))
    assessment["refined"] = True
    return assessment


def assess_with_stubs(mol: Chem.Mol, receptor: np.ndarray,
                      gp=None, sp: StubParams = StubParams()) -> dict:
    """Geometric pre-pass followed by stub-growth refinement of the survivors."""
    from asatro.chemistry.accessibility import ProbeParams, assess_fragment
    from asatro.chemistry.handles import neutralize
    mol = neutralize(mol)   # match assess_fragment so stub indices line up
    analysis = assess_fragment(mol, receptor, gp or ProbeParams())
    return refine_assessment(mol, receptor, analysis, sp)
