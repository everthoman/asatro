"""Asatro web app.

FastAPI surface for fragment growing: handle analysis (``/analyze``), the
accessibility pre-pass (``/prune``), and accessibility-gated growth runs as
background jobs (``/grow`` + ``/jobs`` + log streaming).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from rdkit import Chem

from asatro import __version__
from asatro.chemistry.accessibility import assess_fragment, load_receptor_atoms
from asatro.chemistry.handles import analyze_fragment
from asatro.chemistry.catalog import REACTIONS, VOCAB
from asatro.chemistry.stub_growth import assess_with_stubs
from asatro.jobs import JOBS, list_jobs, start_growth_job

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_HTML = (BASE_DIR / "templates" / "index.html").read_text()
PORT = int(os.environ.get("ASATRO_PORT", "5015"))

# Static catalog the UI needs to render reaction names + slot labels.
CATALOG = {
    "reactions": [
        {"id": r["id"], "name": r["name"], "role": r.get("role"),
         "components": [{"label": c["label"], "accepts": c.get("accepts", [])}
                        for c in r["components"]]}
        for r in REACTIONS
    ],
    "groups": {k: g.get("label", k) for k, g in VOCAB.groups.items()},
}

app = FastAPI(title="Asatro", version=__version__)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (INDEX_HTML
            .replace("__VERSION__", __version__)
            .replace("__CATALOG_JSON__", json.dumps(CATALOG)))
    return HTMLResponse(html)


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
async def prune(fragment: UploadFile = File(...), receptor: UploadFile = File(...),
                refine: bool = Form(False)) -> dict:
    """Accessibility pre-pass: given the bound fragment (SDF, in its pose) and the
    receptor (PDB), return the Tier-1 analysis augmented with per-vector probe
    results and an ``accessible`` flag, plus the list of reactions that survive
    pruning (their growth vectors have room in the pocket).

    ``refine=true`` runs the slower stub-growth refinement on the geometric
    survivors — actually growing –Me/–Ph/morpholine onto each vector and keeping
    only those where a real substituent fits."""
    mol = Chem.MolFromMolBlock((await fragment.read()).decode("utf-8", "replace"), removeHs=True)
    if mol is None:
        raise HTTPException(400, "could not read fragment SDF")
    if mol.GetNumConformers() == 0:
        raise HTTPException(400, "fragment SDF has no 3D conformer (need the bound pose)")
    receptor_atoms = load_receptor_atoms((await receptor.read()).decode("utf-8", "replace"))
    if refine:
        return assess_with_stubs(mol, receptor_atoms)
    return assess_fragment(mol, receptor_atoms)


# ---------------------------------------------------------------------------
# Growth jobs
# ---------------------------------------------------------------------------
@app.post("/pool-preview")
async def pool_preview(pool: UploadFile = File(...)) -> dict:
    """Annotate a master reagent pool: how many building blocks fall in each
    functional-group class (and how many carry no handle). This is the pruning a
    reaction's slots would draw on."""
    from asatro.pool import Pool
    p = Pool.from_file((await pool.read()).decode("utf-8", "replace"))
    return {"n_total": p.n_total, "n_tagged": p.n_tagged,
            "n_untagged": p.n_total - p.n_tagged, "counts": p.counts()}


@app.post("/grow")
async def grow(fragment: UploadFile = File(...), receptor: UploadFile = File(...),
               reactants: List[UploadFile] = File(default=[]),
               pool: UploadFile = File(default=None),
               config: str = Form("{}"), session_name: str = Form("")) -> dict:
    """Start an accessibility-gated growth run as a background job.

    Uploads: the bound fragment (SDF, in pose), the receptor (PDB), and the
    building blocks for the non-fragment slots — EITHER a single tagged master
    ``pool`` (.smi, pruned per reaction component by FG class) OR one ``reactants``
    library per slot (each file's name stem = its FG class, e.g. ``boronic.smi``).
    ``config`` is JSON (refine, num_warmup, num_cycles, num_to_select, seed,
    score_field, cnn_scoring). Returns the job id."""
    try:
        cfg = json.loads(config or "{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"bad config JSON: {e}")

    work = Path(os.environ.get("ASATRO_JOBS_DIR", str(Path(__file__).resolve().parent.parent / "jobs")))
    stage = work / "_uploads" / f"{int(time.time()*1000)}"
    stage.mkdir(parents=True, exist_ok=True)
    frag_path = stage / "fragment.sdf"
    rec_path = stage / "receptor.pdb"
    frag_path.write_bytes(await fragment.read())
    rec_path.write_bytes(await receptor.read())

    pool_path = None
    if pool is not None and pool.filename:
        pool_path = str(stage / "pool.smi")
        Path(pool_path).write_bytes(await pool.read())

    reactant_by_class = {}
    for rf in reactants:
        cls = Path(rf.filename or "").stem
        if not cls:
            continue
        p = stage / f"reactant_{cls}.smi"
        p.write_bytes(await rf.read())
        reactant_by_class[cls] = str(p)

    if not pool_path and not reactant_by_class:
        raise HTTPException(400, "provide a master pool (.smi) or per-class reactant libraries")

    job = start_growth_job(fragment_path=str(frag_path), receptor_path=str(rec_path),
                           reactant_by_class=reactant_by_class, pool_path=pool_path,
                           cfg=cfg, session_name=session_name)
    return {"job_id": job.id, "status": job.status}


@app.get("/jobs")
async def jobs() -> dict:
    return {"jobs": list_jobs()}


@app.get("/jobs/{job_id}")
async def job_detail(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is not None:
        return {**job.meta(), "result": job.result, "n_log": len(job.lines)}
    # Past run: read persisted metadata/results from disk.
    base = Path(os.environ.get("ASATRO_JOBS_DIR", str(Path(__file__).resolve().parent.parent / "jobs")))
    d = base / job_id
    if d.is_dir() and (d / "job.json").is_file():
        meta = json.loads((d / "job.json").read_text())
        res = json.loads((d / "results.json").read_text()) if (d / "results.json").is_file() else None
        return {**meta, "result": res}
    raise HTTPException(404, "unknown job")


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    job.cancel_event.set()
    job.log("Cancellation requested")
    return {"status": "cancelling"}


@app.get("/jobs/{job_id}/stream")
async def stream(job_id: str) -> StreamingResponse:
    """Server-sent events: live console lines, then an ``end`` event with the
    final status."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")

    async def gen():
        import asyncio
        sent = 0
        while True:
            while sent < len(job.lines):
                yield f"data: {job.lines[sent]}\n\n"
                sent += 1
            if job.status in ("done", "error", "cancelled"):
                yield f"event: end\ndata: {job.status}\n\n"
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("asatro.app:app", host="0.0.0.0", port=PORT, reload=True)
