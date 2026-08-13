"""Inline SVG structure rendering for the web UI's results gallery."""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D


def mol_svg(smiles: str, width: int = 200, height: int = 160) -> str:
    """Render a SMILES to an inline SVG (XML declaration stripped), or "" if the
    SMILES doesn't parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().padding = 0.08
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    i = svg.find("<svg")
    return svg[i:] if i != -1 else svg


def mol_props(smiles: str) -> dict:
    """Display descriptors for a gallery structure -- ``mw`` (g/mol) and cLogP
    -- using the same RDKit descriptors as the pre-dock MW/logP filters
    (``MolFilters``). Returns ``None`` values if the SMILES doesn't parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"mw": None, "logp": None}
    return {"mw": round(Descriptors.MolWt(mol), 1),
            "logp": round(Crippen.MolLogP(mol), 2)}
