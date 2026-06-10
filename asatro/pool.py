"""Master reagent pool: one building-block file, tagged by functional-group class
and pruned per reaction component on demand.

Instead of curating a separate library per slot, you give Asatro a single
``.smi`` pool. Each building block is desalted + neutralized and tagged with every
vocabulary class it bears (an amino acid lands in both ``primary_amine`` and
``carboxylic_acid``). A reaction component is then served the union of blocks
matching the classes it ``accepts`` — so choosing a reaction *is* the pruning.

Same idea as the TS+GNINA app's synthon pool, kept standalone here.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from asatro.chemistry.catalog import VOCAB
from asatro.chemistry.handles import neutralize

_LARGEST = rdMolStandardize.LargestFragmentChooser()

Reagent = Tuple[str, str]  # (canonical neutral SMILES, name)


def _clean(smiles: str) -> Optional[str]:
    """Desalt (largest fragment) + neutralize -> canonical SMILES, or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _LARGEST.choose(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(neutralize(mol))


def read_pool(text_or_path: str) -> List[Reagent]:
    """Parse a ``.smi`` pool (``SMILES name`` per line; name optional). Accepts a
    path or the raw text."""
    if "\n" not in text_or_path and Path(text_or_path).is_file():
        text_or_path = Path(text_or_path).read_text()
    out: List[Reagent] = []
    for i, line in enumerate(text_or_path.splitlines()):
        parts = line.split()
        if not parts:
            continue
        smiles = parts[0]
        name = parts[1] if len(parts) > 1 else f"BB{i+1}"
        out.append((smiles, name))
    return out


class Pool:
    """A tagged building-block pool: ``by_class[class] -> [(smiles, name), …]``."""

    def __init__(self, reagents: List[Reagent]):
        self.by_class: Dict[str, List[Reagent]] = {name: [] for name in VOCAB.names}
        self.n_total = 0
        self.n_tagged = 0
        seen_per_class: Dict[str, set] = {name: set() for name in VOCAB.names}
        for raw_smiles, name in reagents:
            clean = _clean(raw_smiles)
            if clean is None:
                continue
            self.n_total += 1
            mol = Chem.MolFromSmiles(clean)
            hit = False
            for cls, q in VOCAB.query.items():
                if mol.HasSubstructMatch(q) and clean not in seen_per_class[cls]:
                    self.by_class[cls].append((clean, name))
                    seen_per_class[cls].add(clean)
                    hit = True
            self.n_tagged += 1 if hit else 0

    @classmethod
    def from_file(cls, text_or_path: str) -> "Pool":
        return cls(read_pool(text_or_path))

    def counts(self) -> Dict[str, int]:
        """How many blocks fall in each class — the 'annotated' view of a pool."""
        return {c: len(v) for c, v in self.by_class.items() if v}

    def prune(self, accepts: List[str]) -> List[Reagent]:
        """Blocks matching ANY accepted class (deduped by SMILES, order kept)."""
        out: List[Reagent] = []
        seen = set()
        for cls in accepts:
            for smiles, name in self.by_class.get(cls, []):
                if smiles not in seen:
                    out.append((smiles, name))
                    seen.add(smiles)
        return out

    def write_component(self, accepts: List[str], path: str) -> Optional[str]:
        """Write the pruned ``.smi`` for a component; None if nothing matches."""
        rows = self.prune(accepts)
        if not rows:
            return None
        Path(path).write_text("".join(f"{s} {n}\n" for s, n in rows))
        return path


def pool_resolver(pool: Pool, work_dir: str):
    """A ReactantResolver backed by the pool: prune by the component's accepts,
    write the pruned ``.smi`` into ``work_dir``, and hand back its path."""
    base = Path(work_dir)
    base.mkdir(parents=True, exist_ok=True)

    def resolve(reaction_id: str, component_index: int, accepts: List[str]) -> Optional[str]:
        path = base / f"pool_{reaction_id}_c{component_index}.smi"
        return pool.write_component(accepts, str(path))

    return resolve
