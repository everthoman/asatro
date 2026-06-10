"""Asatro web app — placeholder skeleton.

A runnable FastAPI shell so the project serves and deploys from day one. The
fragment-growing engine (handle detection, accessibility pruning, Thompson-Sampled
growth) is not implemented yet — see DESIGN.md.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from asatro import __version__
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("asatro.app:app", host="0.0.0.0", port=PORT, reload=True)
