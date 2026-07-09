"""Asatro web app.

FastAPI surface for fragment growing: handle analysis (``/analyze``), the
accessibility pre-pass (``/prune``), and fragment-anchored growth runs -- one
user-chosen, possibly multi-step route, validated against the pre-pass -- as
background jobs (``/grow`` + ``/jobs`` + log streaming) -- plus the plain,
unanchored ts-gnina combinatorial search (``/combi``), sharing the same job
layer and endpoints.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from rdkit import Chem

from asatro import __version__
from asatro.chemistry.accessibility import assess_fragment, load_receptor_atoms
from asatro.chemistry.handles import analyze_fragment
from asatro.chemistry.catalog import REACTION_BY_ID, REACTIONS, VOCAB, resolve_step
from asatro.chemistry.stub_growth import assess_with_stubs
from asatro.jobs import JOBS, jobs_dir, list_jobs, start_combi_job, start_growth_job
from asatro.seed import carve_fragment, component_route_meta
from asatro.svg import mol_svg

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_HTML = (BASE_DIR / "templates" / "index.html").read_text()
REACTION_TABLE_HTML = (BASE_DIR / "reaction-catalog.html").read_text()
PORT = int(os.environ.get("ASATRO_PORT", "5015"))

# Bundled master pool (Enamine Rush-Delivery EU, reused from ts-gnina) — the
# default reactant source when a run doesn't supply its own pool or per-class
# libraries.
DEFAULT_POOL_PATH = str(BASE_DIR / "asatro" / "data" / "enamine_rush_EU.smi")

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


@app.get("/reactions", response_class=HTMLResponse)
async def reactions_page() -> HTMLResponse:
    """Standalone reference: every reaction in the catalog, searchable, with its
    full SMARTS and accepted reagent classes."""
    return HTMLResponse(REACTION_TABLE_HTML)


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
async def pool_preview(pool: UploadFile = File(default=None)) -> dict:
    """Annotate a master reagent pool: how many building blocks fall in each
    functional-group class (and how many carry no handle). This is the pruning a
    reaction's slots would draw on. With no upload, annotates the bundled default
    pool."""
    from asatro.pool import Pool
    if pool is not None and pool.filename:
        p = Pool.from_file((await pool.read()).decode("utf-8", "replace"))
    else:
        p = Pool.from_file(DEFAULT_POOL_PATH)
    return {"n_total": p.n_total, "n_tagged": p.n_tagged,
            "n_untagged": p.n_total - p.n_tagged, "counts": p.counts()}


@app.post("/grow")
async def grow(fragment: UploadFile = File(...), receptor: UploadFile = File(...),
               reactants: List[UploadFile] = File(default=[]),
               pool: UploadFile = File(default=None),
               config: str = Form("{}"), session_name: str = Form("")) -> dict:
    """Start a fragment-anchored growth run (one user-chosen, possibly
    multi-step route) as a background job.

    Uploads: the bound fragment (SDF, in pose), the receptor (PDB), and the
    building blocks for every non-fragment slot across the whole route — EITHER
    a single tagged master ``pool`` (.smi, pruned per reaction component by FG
    class) OR one ``reactants`` library per slot (each file's name stem = its
    FG class, e.g. ``boronic.smi``). The same pool/class-tagged files serve
    every step, since a component is resolved by the FG class(es) it accepts,
    not by position. Neither upload given falls back to the bundled default
    pool (Enamine Rush-Delivery EU).

    ``config`` is JSON: ``steps`` (list of reaction ids — the first must be an
    accessible "start" reaction for this fragment, later ones "extend"),
    ``fragment_slot`` (int, which component of ``steps[0]`` the fragment
    fills), plus the same run knobs as ``/combi`` (refine, num_warmup,
    num_cycles, num_to_select, seed, score_field, cnn_scoring, search_method
    [``"ts"``|``"rws"``], min_cpds_per_core, stop, max_core_rmsd — the core-
    drift placement guard, in Å). Returns the job id."""
    try:
        cfg = json.loads(config or "{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"bad config JSON: {e}")

    steps = cfg.get("steps") or []
    if not steps:
        raise HTTPException(400, "config.steps: at least one reaction id required")
    if cfg.get("fragment_slot") is None:
        raise HTTPException(400, "config.fragment_slot: which component of step 1 the fragment fills")
    fragment_slot = int(cfg["fragment_slot"])
    for i, s in enumerate(steps):
        try:
            resolve_step(s, i)
        except KeyError as e:
            raise HTTPException(400, str(e.args[0]) if e.args else str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))

    stage = jobs_dir() / "_uploads" / f"{int(time.time()*1000)}"
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
        pool_path = DEFAULT_POOL_PATH  # bundled Enamine Rush-Delivery EU pool

    job = start_growth_job(fragment_path=str(frag_path), receptor_path=str(rec_path),
                           steps=steps, fragment_slot=fragment_slot,
                           reactant_by_class=reactant_by_class, pool_path=pool_path,
                           cfg=cfg, session_name=session_name)
    return {"job_id": job.id, "status": job.status}


@app.post("/combi")
async def combi(receptor: UploadFile = File(...),
                reference: UploadFile = File(default=None),
                reactants: List[UploadFile] = File(default=[]),
                config: str = Form("{}"), session_name: str = Form("")) -> dict:
    """Start an unanchored (plain ts-gnina) combinatorial search as a background job.

    No bound fragment: every slot of a (possibly multi-step) reaction route is
    Thompson-sampled from a real reagent library and freely docked. Uploads: the
    receptor (PDB), an optional ``reference`` ligand (SDF, for GNINA's autobox)
    -- if omitted ``config.center``/``config.size`` must give an explicit pocket
    -- and one reagent library (.smi) per route component, uploaded flat in
    route order (step 0's components first in order, then step 1's, ...; counts
    per step come from ``config.steps`` via the reaction catalog).

    ``config`` is JSON: ``steps`` (list of reaction ids -- the first must be a
    ``"start"`` reaction, later ones ``"extend"``), ``center``/``size`` ([x,y,z]
    each, pocket mode), plus the same run knobs as ``/grow`` (num_warmup,
    num_cycles, num_to_select, seed, score_field, cnn_scoring, search_method
    [``"ts"``|``"rws"``], min_cpds_per_core, stop). Returns the job id."""
    try:
        cfg = json.loads(config or "{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"bad config JSON: {e}")

    steps = cfg.get("steps") or []
    if not steps:
        raise HTTPException(400, "config.steps: at least one reaction id required")
    counts = []
    for i, s in enumerate(steps):
        try:
            info = resolve_step(s, i)
        except KeyError as e:
            raise HTTPException(400, str(e.args[0]) if e.args else str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        counts.append(len(info["fresh_indices"]))
    if len(reactants) != sum(counts):
        raise HTTPException(
            400, f"steps {steps} need {sum(counts)} reagent file(s) (route order), "
            f"got {len(reactants)}")

    center = tuple(cfg["center"]) if cfg.get("center") else None
    size = tuple(cfg["size"]) if cfg.get("size") else None
    reference_given = reference is not None and reference.filename
    if not reference_given and center is None:
        raise HTTPException(400, "give a reference ligand upload or config.center [x,y,z]")

    stage = jobs_dir() / "_uploads" / f"{int(time.time()*1000)}"
    stage.mkdir(parents=True, exist_ok=True)
    rec_path = stage / "receptor.pdb"
    rec_path.write_bytes(await receptor.read())

    reference_path = None
    if reference_given:
        reference_path = str(stage / "reference.sdf")
        Path(reference_path).write_bytes(await reference.read())

    reagent_files: List[List[str]] = []
    idx = 0
    for si, n in enumerate(counts):
        step_paths = []
        for ci in range(n):
            rf = reactants[idx]
            p = stage / f"step{si}_comp{ci}_{Path(rf.filename or 'reagent.smi').name}"
            p.write_bytes(await rf.read())
            step_paths.append(str(p))
            idx += 1
        reagent_files.append(step_paths)

    job = start_combi_job(receptor_path=str(rec_path), steps=steps, reagent_files=reagent_files,
                          reference_path=reference_path, center=center, size=size,
                          cfg=cfg, session_name=session_name)
    return {"job_id": job.id, "status": job.status}


def _top_items(rows) -> list:
    """Render ``(score, smiles, name)`` rows (best-first) into gallery items."""
    items = []
    for rank, (score, smiles, name) in enumerate(rows, start=1):
        items.append({"rank": rank, "score": round(float(score), 3),
                      "smiles": str(smiles), "name": str(name),
                      "svg": mol_svg(str(smiles))})
    return items


@app.get("/jobs/{job_id}/top")
async def job_top(job_id: str, n: int = 12) -> dict:
    """Live leaderboard (structure gallery) for the growth target currently
    docking. Only meaningful while a job is running — its evaluator holds every
    score gathered so far. Finished jobs carry their per-target results
    (already with structure SVGs) in ``GET /jobs/{id}``."""
    n = max(1, min(int(n), 60))
    job = JOBS.get(job_id)
    if job is not None and job.status == "running" and job.evaluator is not None:
        rows = job.evaluator.top_scored(n)
        total = job.evaluator.stats()["unique_scored"]
        return {"ready": bool(rows), "live": True, "target": job.current_target,
                "items": _top_items(rows), "total": total}
    return {"ready": False, "live": False, "items": []}


@app.get("/jobs/{job_id}/convergence")
async def job_convergence(job_id: str) -> dict:
    """Best-score-so-far vs docks for the growth target currently docking."""
    job = JOBS.get(job_id)
    if job is not None and job.status == "running" and job.evaluator is not None:
        ev = job.evaluator
        pts = ev.convergence()
        st = ev.stats()
        return {
            "ready": bool(pts), "live": True, "target": job.current_target,
            "score_field": ev.score_field, "higher_better": bool(ev.higher_is_better),
            "docked": st["docked"], "best": st["best_score"],
            "points": [{"dock": d, "best": b} for d, b in pts],
        }
    return {"ready": False, "live": False, "points": []}


@app.get("/jobs")
async def jobs() -> dict:
    return {"jobs": list_jobs()}


@app.get("/jobs/{job_id}")
async def job_detail(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is not None:
        return {**job.meta(), "result": job.result, "n_log": len(job.lines)}
    # Past run: read persisted metadata/results from disk.
    d = jobs_dir() / job_id
    if d.is_dir() and (d / "job.json").is_file():
        meta = json.loads((d / "job.json").read_text())
        res = json.loads((d / "results.json").read_text()) if (d / "results.json").is_file() else None
        return {**meta, "result": res}
    raise HTTPException(404, "unknown job")


@app.get("/jobs/{job_id}/poses/{filename}")
async def job_poses(job_id: str, filename: str) -> FileResponse:
    """Download the docked poses (SDF) for one growth target of a job."""
    if not re.fullmatch(r"poses_\d+\.sdf", filename):
        raise HTTPException(400, "invalid filename")
    p = jobs_dir() / job_id / filename
    if not p.is_file():
        raise HTTPException(404, "poses not found")
    return FileResponse(str(p), media_type="chemical/x-mdl-sdfile", filename=filename)


@app.post("/jobs/{job_id}/seed")
async def seed_fragment(job_id: str, rank: int = Form(...),
                        component_index: int = Form(...)) -> Response:
    """Carve a growth-ready fragment out of one reagent's contribution to a
    finished job's ``rank``-th docked hit (1-based, matching the results
    panel's display order) -- e.g. seed a growth run from the amine of a
    combi job's best amide-coupling hit. Returns the carved fragment as a
    downloadable SDF, with real 3D coordinates taken straight from that hit's
    docked pose. Reuse the *same* receptor for the follow-up growth run so
    the coordinate frame lines up."""
    job = JOBS.get(job_id)
    if job is not None:
        result = job.result
    else:
        d = jobs_dir() / job_id
        if not (d.is_dir() and (d / "results.json").is_file()):
            raise HTTPException(404, "unknown job (or it hasn't produced results yet)")
        result = json.loads((d / "results.json").read_text())
    if not result or not result.get("runs"):
        raise HTTPException(400, "job has no results yet")

    steps = result.get("steps")
    if not steps:
        raise HTTPException(400, "job has no route info to seed from")
    try:
        meta = component_route_meta(steps)
    except KeyError as e:
        raise HTTPException(400, f"unknown reaction in job route: {e}")

    top = result["runs"][0].get("top") or []
    if not (1 <= rank <= len(top)):
        raise HTTPException(400, f"rank {rank} out of range (job has {len(top)} top hit(s))")
    components = top[rank - 1].get("components") or []
    if not (0 <= component_index < len(components)) or component_index >= len(meta):
        raise HTTPException(
            400, f"component_index {component_index} out of range "
            f"({len(components)} component(s) for this hit)")
    reagent = components[component_index]
    accepts = meta[component_index]["accepts"]

    poses_path = jobs_dir() / job_id / "poses_0.sdf"
    if not poses_path.is_file():
        raise HTTPException(400, "no docked poses available to seed from")
    pose_mol = None
    for m in Chem.SDMolSupplier(str(poses_path), sanitize=True, removeHs=False):
        if m is not None and m.HasProp("DockingRank") and int(m.GetProp("DockingRank")) == rank:
            pose_mol = m
            break
    if pose_mol is None:
        raise HTTPException(404, f"no pose found for rank {rank}")

    try:
        carved = carve_fragment(pose_mol, reagent["smiles"], accepts)
    except ValueError as e:
        raise HTTPException(400, str(e))

    sdf = Chem.MolToMolBlock(carved) + "$$$$\n"
    filename = f"fragment_{job_id}_rank{rank}_comp{component_index}.sdf"
    return Response(content=sdf, media_type="chemical/x-mdl-sdfile",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


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
