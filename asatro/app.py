"""Asatro web app — placeholder skeleton.

A runnable FastAPI shell so the project serves and deploys from day one. The
fragment-growing engine (handle detection, accessibility pruning, Thompson-Sampled
growth) is not implemented yet — see DESIGN.md.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from asatro import __version__

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("asatro.app:app", host="0.0.0.0", port=PORT, reload=True)
