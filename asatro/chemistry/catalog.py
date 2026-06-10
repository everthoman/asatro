"""Load Asatro's chemistry vocabulary: the functional-group classes (with their
match + leaving-group SMARTS) and the reaction catalog.

This is the single source of truth the handle-detection layer reads from. Lifted
(standalone) from the TS+GNINA app's vocabulary; kept deliberately small.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from rdkit import Chem

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
