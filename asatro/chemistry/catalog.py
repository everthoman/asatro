"""Load Asatro's chemistry vocabulary: the functional-group classes (with their
match + leaving-group SMARTS) and the reaction catalog.

This is the single source of truth the handle-detection layer reads from. Lifted
(standalone) from the TS+GNINA app's vocabulary; kept deliberately small.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from rdkit import Chem
from rdkit.Chem import AllChem

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class Vocab:
    """The functional-group vocabulary from ``functional_groups.json``.

    Per class it holds a compiled match query (``query``), the leaving-group
    query used to carve the conserved core (``leaving``, may be ``None`` when
    nothing leaves), the family, and whether the class is a refinement of a
    parent (e.g. activated aryl halide ⊂ aryl halide).
    """

    def __init__(self, path: Path):
        doc = json.loads(Path(path).read_text())
        self.version = doc.get("version")
        self.groups: Dict[str, dict] = doc["groups"]
        self.query: Dict[str, Chem.Mol] = {}
        self.leaving: Dict[str, Optional[Chem.Mol]] = {}
        self.family: Dict[str, str] = {}
        self.refinement: set = set()
        for name, g in self.groups.items():
            q = Chem.MolFromSmarts(g["smarts"])
            if q is None:
                raise ValueError(f"Bad SMARTS for functional group {name!r}: {g['smarts']}")
            self.query[name] = q
            ls = g.get("leaving_smarts")
            lq = Chem.MolFromSmarts(ls) if ls else None
            if ls and lq is None:
                raise ValueError(f"Bad leaving_smarts for {name!r}: {ls}")
            self.leaving[name] = lq
            self.family[name] = g.get("family", name)
            if g.get("refinement"):
                self.refinement.add(name)

    @property
    def names(self) -> List[str]:
        return list(self.query)

    def label(self, name: str) -> str:
        return self.groups.get(name, {}).get("label", name)


def _load_reactions(path: Path) -> List[dict]:
    return json.loads(Path(path).read_text())["reactions"]


# Module-level singletons (the catalog is static).
VOCAB = Vocab(DATA_DIR / "functional_groups.json")
REACTIONS: List[dict] = _load_reactions(DATA_DIR / "reactions.json")
START_REACTIONS: List[dict] = [r for r in REACTIONS if r.get("role") == "start"]
REACTION_BY_ID: Dict[str, dict] = {r["id"]: r for r in REACTIONS}


@functools.lru_cache(maxsize=None)
def _compiled_reaction(reaction_id: str) -> Tuple[AllChem.ChemicalReaction, Tuple[Chem.Mol, ...]]:
    """One compiled reaction and its reactant templates, cached *together*.

    ``GetReactantTemplate`` hands back a molecule the reaction still owns, not a
    copy: keep only the templates and the reaction is freed under them, so the
    next substructure match reads freed memory and segfaults the interpreter
    (reproduced -- it is not a Python-level error you can catch). Caching the
    pair keeps the owner alive for as long as the templates are reachable."""
    rxn = AllChem.ReactionFromSmarts(REACTION_BY_ID[reaction_id]["smarts"])
    return rxn, tuple(rxn.GetReactantTemplate(i) for i in range(rxn.GetNumReactantTemplates()))


def reactant_templates(reaction_id: str) -> Tuple[Chem.Mol, ...]:
    """One reaction's compiled reagent patterns, indexed like its ``components``.

    The catalog's component list and the reaction SMARTS' reactant templates
    are in the same order for every reaction in the catalog, so
    ``reactant_templates(rid)[i]`` is what component ``i`` has to match for
    ``RunReactants`` to produce anything. Compiled once per reaction: handle
    analysis asks this for all 58 start reactions on every fragment.

    A component's FG class says the handle is the right *kind*; this says
    whether a particular molecule is one the reaction will actually consume --
    ``primary_amine`` covers both anilines and alkylamines, but a reductive
    amination only fires on the latter."""
    return _compiled_reaction(reaction_id)[1]


StepSpec = Union[str, Dict]  # bare reaction id, or {"reaction_id": str, "slot": Optional[int]}


def resolve_step(step: StepSpec, index: int) -> dict:
    """Normalize + validate one route-step spec against the catalog.

    ``step`` is either a bare reaction id (legacy shape: for a 1-component
    "extend" reaction the implicit intermediate slot is always 0; for step 0
    no slot concept applies at all) or ``{"reaction_id": str, "slot":
    Optional[int]}`` (needed to pick which of a reaction's *own* components
    binds the running intermediate, so any reaction can serve as an extend
    step -- not just the 19 hand-authored ``role="extend"`` rows -- "the
    reaction SMARTS itself is the real gate", not a role label).

    ``index == 0`` must resolve to a ``role == "start"`` reaction; ``slot`` is
    ignored here (growth's ``fragment_slot`` is a separate, step-0-only
    concept handled by its own caller). ``index > 0`` accepts any reaction:
    a 1-component reaction needs no ``slot`` (defaults to 0, matching every
    existing "extend" row's implicit intermediate-first authoring); a
    reaction with 2+ components requires an explicit, in-range ``slot``.

    Returns ``{"reaction_id", "rxn", "intermediate_slot", "fresh_indices"}``
    -- ``intermediate_slot`` is ``None`` at index 0; ``fresh_indices`` is the
    ordered list of component indices *not* bound to the intermediate (every
    index, at index 0).
    """
    if isinstance(step, dict):
        rid = step.get("reaction_id")
        slot = step.get("slot")
    else:
        rid = step
        slot = None
    rxn = REACTION_BY_ID.get(rid)
    if rxn is None:
        raise KeyError(f"unknown reaction: {rid}")
    comps = rxn["components"]

    if index == 0:
        if rxn.get("role") != "start":
            raise ValueError(
                f"step 1 must be a 'start' reaction, got '{rid}' ({rxn.get('role')})")
        return {"reaction_id": rid, "rxn": rxn, "intermediate_slot": None,
                "fresh_indices": list(range(len(comps)))}

    if len(comps) == 1:
        # Legacy 1-component "extend" shape: the intermediate was never a
        # declared component to begin with (every such SMARTS was hand-
        # authored with the intermediate-matching pattern first, implicit),
        # so the sole declared component is entirely the fresh reagent --
        # nothing to skip. intermediate_slot=0 here describes RunReactants'
        # reactant *array* position (2 slots: intermediate, fresh), not an
        # index into `comps` (which only ever names the fresh one).
        return {"reaction_id": rid, "rxn": rxn, "intermediate_slot": 0,
                "fresh_indices": [0]}

    if slot is None:
        raise ValueError(
            f"step {index + 1} ('{rid}') has {len(comps)} components -- "
            f"give 'slot' to say which one binds the running intermediate")
    if not 0 <= slot < len(comps):
        raise ValueError(
            f"slot {slot} out of range for '{rid}' ({len(comps)} components)")
    fresh_indices = [ci for ci in range(len(comps)) if ci != slot]
    return {"reaction_id": rid, "rxn": rxn, "intermediate_slot": slot,
            "fresh_indices": fresh_indices}
