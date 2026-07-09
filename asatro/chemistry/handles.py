"""Tier-1 fragment analysis: which reactions a bound fragment can grow by, and the
conserved core each one preserves.

The bound fragment carries reactive *handles* (functional groups). A start
reaction is *compatible* when one of its components accepts a functional-group
class the fragment bears; that component is the slot the fragment occupies, the
others vary. The *conserved core* is the fragment minus the atoms that leave when
the handle reacts (per-class ``leaving_smarts``) — used downstream to anchor the
constrained placement, and surfaced so the user can see/override it.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

from rdkit import Chem

from rdkit.Chem.MolStandardize import rdMolStandardize

from asatro.chemistry.catalog import START_REACTIONS, VOCAB

MolOrSmiles = Union[Chem.Mol, str]

_UNCHARGER = rdMolStandardize.Uncharger()


def to_mol(frag: MolOrSmiles) -> Chem.Mol:
    """Accept an RDKit mol or a SMILES string; raise on an unparseable SMILES."""
    if isinstance(frag, Chem.Mol):
        return frag
    mol = Chem.MolFromSmiles(frag)
    if mol is None:
        raise ValueError(f"could not parse fragment SMILES: {frag!r}")
    return mol


def neutralize(mol: Chem.Mol) -> Chem.Mol:
    """Neutralize a fragment's charged groups before handle detection.

    Bound poses from ligand prep are protonated at physiological pH — a primary
    amine is ``[NH3+]`` (an NX4), a carboxylic acid is the carboxylate ``COO-`` —
    neither of which matches the neutral vocabulary SMARTS (nor the reaction
    templates). Uncharger restores the neutral forms. It only adjusts formal
    charges and hydrogen counts, so 3D conformers and heavy-atom indices are
    preserved (provided hydrogens are implicit, i.e. read with ``removeHs=True``)."""
    try:
        return _UNCHARGER.uncharge(mol)
    except Exception:
        return mol


def detect_fg_classes(frag: MolOrSmiles) -> List[str]:
    """The functional-group classes the fragment bears (by vocabulary match)."""
    mol = neutralize(to_mol(frag))
    return [name for name, q in VOCAB.query.items() if mol.HasSubstructMatch(q)]


def derive_core(frag: MolOrSmiles, fg_class: str) -> str:
    """Conserved-core SMILES = fragment minus the reacting handle of ``fg_class``.

    Drops every atom matched by the class's ``leaving_smarts`` (the handle that
    does not survive into the product) and keeps the largest connected remnant.
    When the class declares no leaving group (e.g. an amine, whose N stays) the
    whole fragment is conserved. The handle atoms are matched only within the
    fragment's actual occurrence of the class, so an unrelated group elsewhere on
    the scaffold is left intact.
    """
    mol = neutralize(to_mol(frag))
    lq = VOCAB.leaving.get(fg_class)
    if lq is None:
        return Chem.MolToSmiles(mol)

    # Restrict leaving-atom matches to the reacting functional group's occurrence,
    # so e.g. a free alcohol elsewhere is not mistaken for an acid hydroxyl.
    fg_q = VOCAB.query.get(fg_class)
    fg_atoms: set = set()
    if fg_q is not None:
        for m in mol.GetSubstructMatches(fg_q):
            fg_atoms.update(m)

    drop: set = set()
    for m in mol.GetSubstructMatches(lq):
        if not fg_atoms or any(a in fg_atoms for a in m):
            drop.update(m)
    if not drop:
        return Chem.MolToSmiles(mol)

    rw = Chem.RWMol(mol)
    for idx in sorted(drop, reverse=True):
        rw.RemoveAtom(idx)
    remnant = rw.GetMol()
    frags = Chem.GetMolFrags(remnant, asMols=True, sanitizeFrags=False)
    if not frags:
        return Chem.MolToSmiles(mol)
    core = max(frags, key=lambda x: x.GetNumAtoms())
    return Chem.MolToSmiles(core)


def carve_substructure_3d(mol: Chem.Mol, match: Sequence[int]) -> Chem.Mol:
    """Carve the atoms in ``match`` out of ``mol`` *with their conformer
    coordinates*, keeping only the bonds between them, and sanitize the result.

    Used to build a template mol from a real 3D structure: the conserved core
    of a bound fragment (``AnchoredFragmentEvaluator._load_core``), or a
    growth-ready fragment carved out of one reagent's contribution to a
    finished combi/growth hit's docked pose (``asatro/seed.py``). ``mol`` must
    have a conformer; ``match`` is a substructure-match atom-index tuple
    (e.g. from ``GetSubstructMatch``) into it.
    """
    core = Chem.RWMol()
    conf = mol.GetConformer()
    old2new: Dict[int, int] = {}
    new_conf_pts = []
    for old_idx in match:
        a = mol.GetAtomWithIdx(old_idx)
        # Copy the whole atom (not just the atomic number) so explicit Hs,
        # formal charge and the aromatic flag survive -- without the pyrrole
        # N-H, an NH-aromatic core (indole/pyrrole/imidazole...) can't be
        # kekulized and SanitizeMol below blows up with "Can't kekulize mol".
        old2new[old_idx] = core.AddAtom(a)
        new_conf_pts.append(conf.GetAtomPosition(old_idx))
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in old2new and j in old2new:
            core.AddBond(old2new[i], old2new[j], b.GetBondType())
    core = core.GetMol()
    new_conf = Chem.Conformer(core.GetNumAtoms())
    for new_idx, pt in enumerate(new_conf_pts):
        new_conf.SetAtomPosition(new_idx, pt)
    core.AddConformer(new_conf, assignId=True)
    try:
        Chem.SanitizeMol(core)
    except Exception as e:
        # The carved sub-graph won't sanitize -- almost always because the
        # match cut through an aromatic ring (an in-ring atom was excluded),
        # leaving a partial ring that can't be kekulized.
        raise ValueError(
            f"the carved substructure is not a valid fragment on its own "
            f"({e}). This usually means the match excluded an in-ring atom "
            f"-- keep whole aromatic rings intact.") from e
    return core


def analyze_fragment(frag: MolOrSmiles) -> dict:
    """Full Tier-1 analysis for a bound fragment.

    Returns ``{fragment_smiles, fg_classes, reactions}`` where ``reactions`` maps
    each *start* reaction id to ``{compatible, slots}`` and each slot is
    ``{index, fg_class, core_smarts}`` (the component the fragment fills and the
    core it conserves there). Extend reactions are out of scope here — the
    fragment only seeds the first step.
    """
    mol = neutralize(to_mol(frag))
    classes = set(detect_fg_classes(mol))
    reactions: Dict[str, dict] = {}
    for r in START_REACTIONS:
        slots: List[dict] = []
        for i, comp in enumerate(r["components"]):
            hit = classes.intersection(comp.get("accepts", []))
            if hit:
                fg = sorted(hit)[0]
                slots.append({
                    "index": i,
                    "fg_class": fg,
                    "core_smarts": derive_core(mol, fg),
                })
        reactions[r["id"]] = {"compatible": bool(slots), "slots": slots}
    return {
        "fragment_smiles": Chem.MolToSmiles(mol),
        "fg_classes": sorted(classes),
        "reactions": reactions,
    }


def _cli() -> None:
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description="Asatro Tier-1 fragment analysis")
    ap.add_argument("smiles", help="fragment SMILES")
    args = ap.parse_args()
    print(_json.dumps(analyze_fragment(args.smiles), indent=2))


if __name__ == "__main__":
    _cli()
