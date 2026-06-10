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

from typing import Dict, List, Optional, Union

from rdkit import Chem

from asatro.chemistry.catalog import START_REACTIONS, VOCAB

MolOrSmiles = Union[Chem.Mol, str]


def to_mol(frag: MolOrSmiles) -> Chem.Mol:
    """Accept an RDKit mol or a SMILES string; raise on an unparseable SMILES."""
    if isinstance(frag, Chem.Mol):
        return frag
    mol = Chem.MolFromSmiles(frag)
    if mol is None:
        raise ValueError(f"could not parse fragment SMILES: {frag!r}")
    return mol


def detect_fg_classes(frag: MolOrSmiles) -> List[str]:
    """The functional-group classes the fragment bears (by vocabulary match)."""
    mol = to_mol(frag)
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
    mol = to_mol(frag)
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


def analyze_fragment(frag: MolOrSmiles) -> dict:
    """Full Tier-1 analysis for a bound fragment.

    Returns ``{fragment_smiles, fg_classes, reactions}`` where ``reactions`` maps
    each *start* reaction id to ``{compatible, slots}`` and each slot is
    ``{index, fg_class, core_smarts}`` (the component the fragment fills and the
    core it conserves there). Extend reactions are out of scope here — the
    fragment only seeds the first step.
    """
    mol = to_mol(frag)
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
