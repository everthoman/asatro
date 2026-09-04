#!/usr/bin/env python
"""Convert a supplier/inventory SDF into an Asatro master pool (``.smi``).

    python tools/sdf_to_pool.py KLARA_Sep_25.sdf asatro/data/klara_sep_25.smi --id-field KLARA_ID

Each record is desalted (largest fragment) and neutralized -- the same
normalization ``asatro.pool.Pool`` applies on load -- then written as
``SMILES name``. Records are dropped when they carry no carbon (inorganic
stock: salts, acids, metals), when the SDF field naming the block is missing,
when RDKit can't round-trip the structure through SMILES, or when the SMILES
duplicates one already kept.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from a checkout

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from asatro.chemistry.handles import neutralize


def convert(src: str, dst: str, id_field: str) -> Counter:
    largest = rdMolStandardize.LargestFragmentChooser()
    stats: Counter = Counter()
    seen: set = set()
    rows = []
    for mol in Chem.ForwardSDMolSupplier(src, sanitize=True, removeHs=True):
        stats["records"] += 1
        if mol is None:
            stats["unparsable"] += 1
            continue
        try:
            mol = largest.choose(mol)
        except Exception:  # noqa: BLE001 -- keep the record as-is if desalting fails
            pass
        if not any(a.GetSymbol() == "C" for a in mol.GetAtoms()):
            stats["no_carbon"] += 1
            continue
        try:
            smiles = Chem.MolToSmiles(neutralize(mol))
        except Exception:  # noqa: BLE001
            smiles = ""
        # Round-trip: a few SDF records sanitize but yield SMILES RDKit won't
        # re-read (hypervalent B/Si, ferrocene-style bonding). The pool loader
        # would silently drop them, so drop them here where it's visible.
        if not smiles or Chem.MolFromSmiles(smiles) is None:
            stats["unparsable"] += 1
            continue
        if not mol.HasProp(id_field):
            stats["no_id"] += 1
            continue
        # Inventory ids print with thousands separators ("9,332"); the pool
        # format is whitespace-delimited, so strip them.
        name = mol.GetProp(id_field).replace(",", "").strip().replace(" ", "_")
        if not name:
            stats["no_id"] += 1
            continue
        if smiles in seen:
            stats["duplicate"] += 1
            continue
        seen.add(smiles)
        rows.append((smiles, name))
        stats["kept"] += 1
    with open(dst, "w") as fh:
        fh.writelines(f"{s} {n}\n" for s, n in rows)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sdf")
    ap.add_argument("smi")
    ap.add_argument("--id-field", default="KLARA_ID",
                    help="SDF property to use as the block name (default: KLARA_ID)")
    args = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    stats = convert(args.sdf, args.smi, args.id_field)
    print(f"{args.sdf} -> {args.smi}")
    for k in ("records", "kept", "no_carbon", "no_id", "unparsable", "duplicate"):
        print(f"  {k:11s} {stats[k]}")


if __name__ == "__main__":
    main()
