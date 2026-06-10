"""Asatro web app — placeholder skeleton.

A runnable FastAPI shell so the project serves and deploys from day one. The
fragment-growing engine (handle detection, accessibility pruning, Thompson-Sampled
growth) is not implemented yet — see DESIGN.md.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from rdkit import Chem

from asatro import __version__
from asatro.chemistry.accessibility import assess_fragment, load_receptor_atoms
from asatro.chemistry.handles import analyze_fragment

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_HTML = (BASE_DIR / "templates" / "index.html").read_text()
PORT = int(os.environ.get("ASATRO_PORT", "5023"))

app = FastAPI(title="Asatro", version=__version__)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.replace("__VERSION__", __version__))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "app": "asatro", "version": __version__}


@app.get("/analyze")
async def analyze(smiles: str) -> dict:
    """Tier-1: given a fragment SMILES, report its functional-group handles, the
    compatible start reactions (and which slot the fragment fills), and the
    conserved core auto-derived for each."""
    try:
        return analyze_fragment(smiles)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/prune")
async def prune(fragment: UploadFile = File(...), receptor: UploadFile = File(...)) -> dict:
    """Accessibility pre-pass: given the bound fragment (SDF, in its pose) and the
    receptor (PDB), return the Tier-1 analysis augmented with per-vector probe
    results and an ``accessible`` flag, plus the list of reactions that survive
    pruning (their growth vectors have room in the pocket)."""
    mol = Chem.MolFromMolBlock((await fragment.read()).decode("utf-8", "replace"), removeHs=True)
    if mol is None:
        raise HTTPException(400, "could not read fragment SDF")
    if mol.GetNumConformers() == 0:
        raise HTTPException(400, "fragment SDF has no 3D conformer (need the bound pose)")
    receptor_atoms = load_receptor_atoms((await receptor.read()).decode("utf-8", "replace"))
    return assess_fragment(mol, receptor_atoms)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("asatro.app:app", host="0.0.0.0", port=PORT, reload=True)
