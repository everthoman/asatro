"""Inline SVG structure rendering for the web UI's results gallery."""
from __future__ import annotations

from rdkit import Chem
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
