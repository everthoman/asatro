"""Seed a growth-ready fragment from one reagent's contribution to a finished
combi or growth job's docked hit.

When no bound fragment exists, a reasonable starting point is: run an
unanchored combi search first, take the best-scoring hit's docked pose, and
carve out just the piece contributed by one reagent (e.g. "the amine" side of
an amide coupling) -- now with a real, gnina-placed 3D pose -- to use as the
bound fragment for a follow-up growth run (anchored placement, RMSD-guarded,
and able to chain further steps the original route couldn't reach).

This reuses the exact conserved-core logic the growth path already applies to
a real bound fragment (``derive_core``): call it on the *isolated* reagent's
own SMILES to get its conserved core, then substructure-match that core
against the docked *product's* pose to locate -- and carve out, with real
coordinates -- just that reagent's atoms.
"""
from __future__ import annotations

from typing import Dict, List

from rdkit import Chem

from asatro.chemistry.catalog import REACTION_BY_ID
from asatro.chemistry.handles import carve_substructure_3d, derive_core, detect_fg_classes


def component_route_meta(steps: List[str]) -> List[Dict]:
    """Flatten a route's reaction ids into one ``{reaction_id, label, accepts}``
    per component, in the same order ``build_combi_route``/``build_growth_route``
    already flatten reagent files -- i.e. matching the index into a scored
    product's ``components`` list (see ``asatro.engine.gnina_evaluator.GninaEvaluator.components_scored``).
    """
    meta: List[Dict] = []
    for rid in steps:
        rxn = REACTION_BY_ID.get(rid)
        if rxn is None:
            raise KeyError(f"unknown reaction: {rid}")
        for comp in rxn["components"]:
            meta.append({"reaction_id": rid, "label": comp["label"],
                        "accepts": comp.get("accepts", [])})
    return meta


def carve_fragment(pose_mol: Chem.Mol, reagent_smiles: str, accepts: List[str]) -> Chem.Mol:
    """Carve the sub-pose contributed by one reagent out of a docked product's
    pose. ``accepts`` is that reagent's component's accepted FG classes (from
    ``component_route_meta``); the actual matching class is resolved the same
    way ``analyze_fragment`` picks one for a multi-class-accepting slot.
    """
    classes = sorted(set(detect_fg_classes(reagent_smiles)) & set(accepts))
    if not classes:
        raise ValueError(
            f"'{reagent_smiles}' doesn't match any of its component's accepted "
            f"classes {accepts} -- can't derive a conserved core to carve.")
    fg_class = classes[0]
    core_smiles = derive_core(reagent_smiles, fg_class)
    core_q = Chem.MolFromSmiles(core_smiles)
    if core_q is None:
        raise ValueError(f"derived core '{core_smiles}' does not parse")
    match = pose_mol.GetSubstructMatch(core_q)
    if not match:
        raise ValueError(
            f"the conserved core '{core_smiles}' (from reagent '{reagent_smiles}', "
            f"class '{fg_class}') was not found in the docked product's pose -- "
            f"the hit may not actually include this reagent.")
    return carve_substructure_3d(pose_mol, match)
