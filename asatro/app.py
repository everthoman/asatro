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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from rdkit import Chem
from starlette.concurrency import run_in_threadpool

from asatro import __version__
from asatro.chemistry.accessibility import assess_fragment, load_receptor_atoms
from asatro.chemistry.handles import analyze_fragment, bond_order_complaint
from asatro.chemistry.catalog import REACTION_BY_ID, REACTIONS, VOCAB, resolve_step
from asatro.chemistry.stub_growth import assess_with_stubs
from asatro.jobs import (JOBS, delete_job, jobs_dir, list_jobs, reap_orphaned_jobs,
                         staged_uploads, start_combi_job, start_growth_job, sweep_staged_uploads)
from asatro.seed import carve_fragment, component_route_meta
from asatro.svg import mol_props, mol_svg, palette_css, retheme_svg

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_HTML = (BASE_DIR / "templates" / "index.html").read_text()
REACTION_TABLE_HTML = (BASE_DIR / "reaction-catalog.html").read_text()
PORT = int(os.environ.get("ASATRO_PORT", "5015"))

# Bundled master pools — the reactant source when a run doesn't supply its own
# pool or per-class libraries. A request names one with the ``pool_id`` form
# field; anything unknown (or empty) falls back to the first, which stays the
# app-wide default.
POOL_DIR = BASE_DIR / "asatro" / "data"
BUNDLED_POOLS = [
    {"id": "enamine_rush_EU", "label": "Enamine Rush-Delivery EU",
     "file": "enamine_rush_EU.smi"},
    {"id": "klara_sep_25", "label": "KLARA in-house stock (Sep 2025)",
     "file": "klara_sep_25.smi"},
]
DEFAULT_POOL_PATH = str(POOL_DIR / BUNDLED_POOLS[0]["file"])


def bundled_pool_path(pool_id: str = "") -> str:
    """Path of the bundled pool named by ``pool_id``, else the default pool."""
    for p in BUNDLED_POOLS:
        if p["id"] == pool_id:
            return str(POOL_DIR / p["file"])
    return DEFAULT_POOL_PATH


async def stage_pool(pool: Optional[UploadFile], pool_id: str, stage: Path) -> str:
    """The .smi pool a request should use: an upload staged into ``stage``, else
    the bundled pool ``pool_id`` names."""
    if pool is not None and pool.filename:
        path = stage / "pool.smi"
        path.write_bytes(await pool.read())
        return str(path)
    return bundled_pool_path(pool_id)


def _pool_size(path: Path) -> int:
    """Building blocks in a bundled pool (non-blank lines), for the UI label."""
    try:
        with open(path, "rb") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0

# Static catalog the UI needs to render reaction names + slot labels.
CATALOG = {
    "reactions": [
        {"id": r["id"], "name": r["name"], "role": r.get("role"),
         "components": [{"label": c["label"], "accepts": c.get("accepts", [])}
                        for c in r["components"]]}
        for r in REACTIONS
    ],
    "groups": {k: g.get("label", k) for k, g in VOCAB.groups.items()},
    "pools": [{"id": p["id"], "label": p["label"],
               "n": _pool_size(POOL_DIR / p["file"])} for p in BUNDLED_POOLS],
}

@asynccontextmanager
async def _lifespan(app: FastAPI):
    reap_orphaned_jobs()
    yield


app = FastAPI(title="Asatro", version=__version__, lifespan=_lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (INDEX_HTML
            .replace("__VERSION__", __version__)
            .replace("__MOL_PALETTE__", palette_css())
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
    # CPU-bound RDKit work (the refine path in particular actually grows
    # stub substituents onto each vector) -- run off the event loop thread so
    # it doesn't stall every other request (SSE streams, job polling) for its
    # duration.
    if refine:
        result = await run_in_threadpool(assess_with_stubs, mol, receptor_atoms)
    else:
        result = await run_in_threadpool(assess_fragment, mol, receptor_atoms)
    # Surface a fragment whose bond orders don't match its own geometry here,
    # where the user is still looking at the fragment, rather than after a run
    # has grown the wrong tautomer (see bond_order_complaint).
    result["bond_order_warning"] = bond_order_complaint(mol)
    return result


# ---------------------------------------------------------------------------
# Growth jobs
# ---------------------------------------------------------------------------
@app.post("/pool-preview")
async def pool_preview(pool: UploadFile = File(default=None),
                       pool_id: str = Form("")) -> dict:
    """Annotate a master reagent pool: how many building blocks fall in each
    functional-group class (and how many carry no handle). This is the pruning a
    reaction's slots would draw on. With no upload, annotates the bundled pool
    ``pool_id`` names (the default pool when unset)."""
    from asatro.pool import Pool
    if pool is not None and pool.filename:
        p = Pool.from_file((await pool.read()).decode("utf-8", "replace"))
    else:
        p = Pool.from_file(bundled_pool_path(pool_id))
    return {"n_total": p.n_total, "n_tagged": p.n_tagged,
            "n_untagged": p.n_total - p.n_tagged, "counts": p.counts()}


@app.post("/suggest-params")
async def suggest_params(fragment: UploadFile = File(...),
                         config: str = Form("{}"),
                         pool: UploadFile = File(default=None),
                         pool_id: str = Form(""),
                         reactants: List[UploadFile] = File(default=[])) -> dict:
    """Pre-launch dry run for the growth form: resolve the chosen route's pools
    and prune unreachable reagents (**no docking, no receptor needed**), then
    return the post-prune variable-slot sizes and a suggested TS budget so the
    "Annotate pool" step can fill ``num_warmup``/``num_cycles``. Read-only;
    reports failures as ``{"ok": false, "error": ...}`` so the UI can still
    show the plain pool annotation."""
    import shutil

    from asatro.growth import suggest_growth_params
    from asatro.jobs import make_class_resolver
    from asatro.pool import Pool, pool_resolver

    try:
        cfg = json.loads(config or "{}")
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"bad config JSON: {e}"}
    steps = cfg.get("steps") or []
    if not steps or cfg.get("fragment_slot") is None:
        return {"ok": False, "error": "route not fully specified yet"}

    stage = jobs_dir() / "_suggest" / f"{int(time.time() * 1000)}"
    stage.mkdir(parents=True, exist_ok=True)
    try:
        frag_path = stage / "fragment.sdf"
        frag_path.write_bytes(await fragment.read())
        if pool is not None and pool.filename:
            pool_path = await stage_pool(pool, pool_id, stage)
            resolver = pool_resolver(Pool.from_file(pool_path), str(stage / "pool"))
        else:
            reactant_by_class = {}
            for rf in reactants:
                cls = Path(rf.filename or "").stem
                if not cls:
                    continue
                p = stage / f"reactant_{cls}.smi"
                p.write_bytes(await rf.read())
                reactant_by_class[cls] = str(p)
            if reactant_by_class:
                resolver = make_class_resolver(reactant_by_class)
            else:
                resolver = pool_resolver(Pool.from_file(bundled_pool_path(pool_id)),
                                         str(stage / "pool"))
        result = await run_in_threadpool(
            suggest_growth_params,
            fragment_sdf=str(frag_path), steps=steps,
            fragment_slot=int(cfg["fragment_slot"]), resolver=resolver, work_dir=str(stage))
        return {"ok": True, **result}
    except Exception as e:  # noqa: BLE001 -- report, don't 500 the annotate flow
        return {"ok": False, "error": str(e)}
    finally:
        shutil.rmtree(stage, ignore_errors=True)


@app.post("/suggest-combi-params")
async def suggest_combi_params_endpoint(config: str = Form("{}"),
                         pool: UploadFile = File(default=None),
                         pool_id: str = Form(""),
                         reactants: List[UploadFile] = File(default=[])) -> dict:
    """Pre-launch dry run for the combi form: resolve the chosen route's pools
    and prune unreachable reagents (**no docking, no receptor needed**), then
    return the post-prune variable-slot sizes and a suggested TS budget so the
    "Annotate pool" step can fill ``num_warmup``/``num_cycles``. Read-only;
    reports failures as ``{"ok": false, "error": ...}`` so the UI can still
    show the plain pool annotation. Mirrors ``/suggest-params`` (growth),
    except reagent resolution follows ``/combi``'s own convention: per-slot
    ``reactants`` is a flat list in route order, not filename-stem-keyed."""
    import shutil

    from asatro.combi import resolve_combi_reactant_files, suggest_combi_params
    from asatro.pool import Pool, pool_resolver

    try:
        cfg = json.loads(config or "{}")
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"bad config JSON: {e}"}
    steps = cfg.get("steps") or []
    if not steps:
        return {"ok": False, "error": "route not fully specified yet"}

    counts = []
    for i, s in enumerate(steps):
        try:
            info = resolve_step(s, i)
        except (KeyError, ValueError) as e:
            return {"ok": False, "error": str(e.args[0]) if e.args else str(e)}
        counts.append(len(info["fresh_indices"]))

    stage = jobs_dir() / "_suggest" / f"{int(time.time() * 1000)}"
    stage.mkdir(parents=True, exist_ok=True)
    try:
        if reactants:
            if len(reactants) != sum(counts):
                return {"ok": False, "error": f"steps {steps} need {sum(counts)} "
                        f"reagent file(s) (route order), got {len(reactants)}"}
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
        else:
            resolver = pool_resolver(
                Pool.from_file(await stage_pool(pool, pool_id, stage)), str(stage / "pool"))
            reagent_files = resolve_combi_reactant_files(steps, resolver)
        result = await run_in_threadpool(
            suggest_combi_params, steps=steps, reagent_files=reagent_files, work_dir=str(stage))
        return {"ok": True, **result}
    except Exception as e:  # noqa: BLE001 -- report, don't 500 the annotate flow
        return {"ok": False, "error": str(e)}
    finally:
        shutil.rmtree(stage, ignore_errors=True)


@app.post("/grow")
async def grow(fragment: UploadFile = File(...), receptor: UploadFile = File(...),
               reactants: List[UploadFile] = File(default=[]),
               pool: UploadFile = File(default=None), pool_id: str = Form(""),
               config: str = Form("{}"), session_name: str = Form("")) -> dict:
    """Start a fragment-anchored growth run (one user-chosen, possibly
    multi-step route) as a background job.

    Uploads: the bound fragment (SDF, in pose), the receptor (PDB), and the
    building blocks for every non-fragment slot across the whole route — EITHER
    a single tagged master ``pool`` (.smi, pruned per reaction component by FG
    class) OR one ``reactants`` library per slot (each file's name stem = its
    FG class, e.g. ``boronic.smi``). The same pool/class-tagged files serve
    every step, since a component is resolved by the FG class(es) it accepts,
    not by position. Neither upload given falls back to the bundled pool
    ``pool_id`` names (the default, Enamine Rush-Delivery EU, when unset).

    ``config`` is JSON: ``steps`` (list of reaction ids — the first must be an
    accessible "start" reaction for this fragment, later ones "extend"),
    ``fragment_slot`` (int, which component of ``steps[0]`` the fragment
    fills), plus the same run knobs as ``/combi`` (refine, num_warmup,
    num_cycles, num_to_select, seed, score_field, cnn_scoring, search_method
    [``"ts"``|``"rws"``], min_cpds_per_core, stop, max_core_rmsd — the core-
    drift placement guard, in Å — ``concurrency``, ``cpu``). Returns the job id."""
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

    # A fragment whose bond orders contradict its geometry is the wrong molecule
    # to grow from, and the whole run inherits the error -- refuse it here unless
    # the user has looked and decided otherwise.
    frag_mol = Chem.MolFromMolFile(str(frag_path), removeHs=True)
    if frag_mol is not None and not cfg.get("ignore_bond_order_warning"):
        complaint = bond_order_complaint(frag_mol)
        if complaint:
            raise HTTPException(400, complaint + " (set ignore_bond_order_warning "
                                "in the run config to grow from it anyway)")

    pool_path = None
    if pool is not None and pool.filename:
        pool_path = await stage_pool(pool, pool_id, stage)

    reactant_by_class = {}
    for rf in reactants:
        cls = Path(rf.filename or "").stem
        if not cls:
            continue
        p = stage / f"reactant_{cls}.smi"
        p.write_bytes(await rf.read())
        reactant_by_class[cls] = str(p)

    if not pool_path and not reactant_by_class:
        pool_path = bundled_pool_path(pool_id)  # bundled pool (default: Enamine Rush EU)

    try:
        job = start_growth_job(fragment_path=str(frag_path), receptor_path=str(rec_path),
                               steps=steps, fragment_slot=fragment_slot,
                               reactant_by_class=reactant_by_class, pool_path=pool_path,
                               cfg=cfg, session_name=session_name)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"job_id": job.id, "status": job.status}


@app.post("/combi")
async def combi(receptor: UploadFile = File(...),
                reference: UploadFile = File(default=None),
                reactants: List[UploadFile] = File(default=[]),
                pool: UploadFile = File(default=None), pool_id: str = Form(""),
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
    [``"ts"``|``"rws"``], min_cpds_per_core, stop, ``concurrency`` — number of
    products to build+dock in parallel (default 1), ``cpu`` — cores per dock
    (default: ``DOCK_CPU`` split evenly across ``concurrency`` when unset)).
    Returns the job id."""
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
    if reactants and len(reactants) != sum(counts):
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

    if reactants:
        # Per-slot uploaded libraries (route order).
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
    else:
        # Master-pool mode: resolve each component from the tagged pool by its
        # accepted FG class(es), same as growth. Uploaded pool, else the bundled
        # one ``pool_id`` names (default: Enamine Rush-Delivery EU).
        from asatro.combi import resolve_combi_reactant_files
        from asatro.pool import Pool, pool_resolver
        resolver = pool_resolver(
            Pool.from_file(await stage_pool(pool, pool_id, stage)), str(stage / "pool"))
        try:
            reagent_files = resolve_combi_reactant_files(steps, resolver)
        except ValueError as e:
            raise HTTPException(400, str(e))

    try:
        job = start_combi_job(receptor_path=str(rec_path), steps=steps, reagent_files=reagent_files,
                              reference_path=reference_path, center=center, size=size,
                              cfg=cfg, session_name=session_name)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"job_id": job.id, "status": job.status}


def _top_items(rows) -> list:
    """Render ``(score, smiles, name)`` rows (best-first) into gallery items."""
    items = []
    for rank, (score, smiles, name) in enumerate(rows, start=1):
        items.append({"rank": rank, "score": round(float(score), 3),
                      "smiles": str(smiles), "name": str(name),
                      "svg": mol_svg(str(smiles)), **mol_props(str(smiles))})
    return items


def _enrich_top_props(result: Optional[dict]) -> Optional[dict]:
    """Bring a persisted run's gallery data up to date on read: fill in
    ``mw``/``logp`` for top-hits that predate those fields, and re-theme any
    structure drawn before the depictions became theme-aware (otherwise a run
    finished under the old renderer shows white boxes on a dark card forever --
    its SVGs are frozen in results.json). Both are computed from what's already
    stored, and both are idempotent."""
    if not result:
        return result
    for run in result.get("runs", []):
        for it in run.get("top", []) or []:
            if it.get("mw") is None and it.get("logp") is None:
                it.update(mol_props(str(it.get("smiles", ""))))
            if it.get("svg"):
                it["svg"] = retheme_svg(it["svg"])
        for slot in run.get("reagents", []) or []:
            for r in slot.get("reagents", []) or []:
                if r.get("svg"):
                    r["svg"] = retheme_svg(r["svg"])
    return result


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


@app.get("/jobs/{job_id}/reagents")
async def job_reagents(job_id: str) -> dict:
    """Live per-reagent ranking ("Top building blocks") for the job currently
    docking -- same shape as the persisted ``GninaEvaluator.reagent_rankings()``
    on a finished job's ``results.json`` (see ``_summarize_combi``), but read
    live off the in-progress evaluator's score cache, same pattern as
    ``/jobs/{id}/top``."""
    job = JOBS.get(job_id)
    if job is not None and job.status == "running" and job.evaluator is not None:
        rankings = (job.evaluator.reagent_rankings()
                   if hasattr(job.evaluator, "reagent_rankings") else [])
        for slot in rankings:
            for r in slot["reagents"]:
                r["svg"] = mol_svg(r["smiles"])
        return {"ready": bool(rankings), "live": True, "target": job.current_target,
                "reagents": rankings}
    return {"ready": False, "live": False, "reagents": []}


@app.get("/jobs")
async def jobs() -> dict:
    return {"jobs": await run_in_threadpool(list_jobs)}


def _delete_one(job_id: str) -> dict:
    """Delete one run, mapping the job layer's refusals onto HTTP codes."""
    try:
        return delete_job(job_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))     # running: cancel it first
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.delete("/jobs/{job_id}")
async def delete_one_job(job_id: str) -> dict:
    """Delete a finished run and everything under its job directory."""
    return await run_in_threadpool(_delete_one, job_id)


@app.post("/jobs/delete")
async def delete_many_jobs(ids: List[str] = Body(..., embed=True)) -> dict:
    """Delete several runs by explicit id.

    The caller always names every run to delete -- there is deliberately no
    "delete everything" verb, so a job launched between the client listing the
    jobs and confirming the delete can never be swept up in it. Failures are
    reported per id rather than aborting the batch: one running job in the
    selection shouldn't stop the rest being cleaned up."""
    def run() -> dict:
        deleted, failed, freed = [], {}, 0
        for job_id in ids:
            try:
                freed += delete_job(job_id)["freed"]
                deleted.append(job_id)
            except (ValueError, RuntimeError, FileNotFoundError) as e:
                failed[job_id] = str(e)
        return {"deleted": deleted, "failed": failed, "freed": freed}
    return await run_in_threadpool(run)


@app.get("/uploads")
async def uploads() -> dict:
    """Size of the staged-upload area (fragments/receptors/pools kept from
    every launch), so the UI can say what a sweep would reclaim."""
    return await run_in_threadpool(staged_uploads)


@app.post("/uploads/sweep")
async def uploads_sweep(max_age_hours: float = Body(24.0, embed=True)) -> dict:
    """Delete staged uploads older than ``max_age_hours`` (see
    ``sweep_staged_uploads`` for what it refuses to touch)."""
    return await run_in_threadpool(sweep_staged_uploads, max_age_hours)


def _job_path(job_id: str, *parts: str) -> Path:
    """Resolve a path under jobs_dir() rooted at ``job_id``, rejecting any
    job_id that would escape it (e.g. "..") -- job_id comes straight from a
    request path parameter and is otherwise used unsanitized to build
    filesystem paths."""
    base = jobs_dir().resolve()
    p = (base / job_id).joinpath(*parts).resolve()
    if p != base and base not in p.parents:
        raise HTTPException(400, "invalid job id")
    return p


@app.get("/jobs/{job_id}")
async def job_detail(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is not None:
        return {**job.meta(), "result": _enrich_top_props(job.result), "n_log": len(job.lines)}
    # Past run: read persisted metadata/results from disk.
    d = _job_path(job_id)
    if d.is_dir() and (d / "job.json").is_file():
        meta = json.loads((d / "job.json").read_text())
        res = json.loads((d / "results.json").read_text()) if (d / "results.json").is_file() else None
        return {**meta, "result": _enrich_top_props(res)}
    raise HTTPException(404, "unknown job")


def _iter_sdf_records(path: Path):
    """Yield the records of an SDF as raw text, each including its ``$$$$``
    terminator.

    Text rather than RDKit: a finished job's pose file holds every docked pose
    (tens of thousands on a big combi run), and both slicing a download and
    pulling one pose out by rank are pure record-selection -- parsing every
    molecule to do it would cost minutes and risk round-tripping the
    coordinates/props the caller asked for verbatim."""
    buf: List[str] = []
    with open(path) as fh:
        for line in fh:
            buf.append(line)
            if line.rstrip("\r\n") == "$$$$":
                yield "".join(buf)
                buf = []
    if any(chunk.strip() for chunk in buf):   # malformed tail: no final $$$$
        yield "".join(buf)


def _count_sdf_records(path: Path) -> int:
    with open(path) as fh:
        return sum(1 for line in fh if line.rstrip("\r\n") == "$$$$")


def _record_rank(record: str) -> Optional[int]:
    """The ``DockingRank`` stamped on one SDF record, if it has one."""
    lines = record.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(">") and "<DockingRank>" in line and i + 1 < len(lines):
            try:
                return int(lines[i + 1].strip())
            except ValueError:
                return None
    return None


def _pose_record(poses_path: Path, rank: int) -> Optional[str]:
    """The docked pose of the given 1-based gallery rank, as SDF text."""
    for idx, record in enumerate(_iter_sdf_records(poses_path), start=1):
        stamped = _record_rank(record)
        if stamped == rank or (stamped is None and idx == rank):
            return record
    return None


@app.get("/jobs/{job_id}/poses/{filename}")
async def job_poses(job_id: str, filename: str,
                    n: Optional[int] = Query(None, ge=1,
                                             description="download only the top n poses"),
                    pct: Optional[float] = Query(None, gt=0, le=100,
                                                 description="download only the top pct% of poses")):
    """Download the docked poses (SDF) of a job -- all of them by default, or
    the best ``n`` / best ``pct`` percent.

    Poses are written best-scored first (``DockingRank`` 1 = best), so every
    slice is just a prefix of the file: no re-scoring, and the ranks in a
    partial download still match the results gallery."""
    if not re.fullmatch(r"poses_\d+\.sdf", filename):
        raise HTTPException(400, "invalid filename")
    if n is not None and pct is not None:
        raise HTTPException(400, "pass either n or pct, not both")
    p = _job_path(job_id, filename)
    if not p.is_file():
        raise HTTPException(404, "poses not found")
    # Saved under the run's own name (the job id *is* the slugified session
    # name), not the on-disk "poses_0.sdf" -- a few downloads from different
    # runs otherwise land in ~/Downloads as poses_0(1).sdf, poses_0(2).sdf and
    # so on, with nothing to say which run each came from. The target index is
    # kept only when it isn't the usual 0, so two files can't collide.
    idx = filename[len("poses_"):-len(".sdf")]
    stem = f"{job_id}_poses" if idx == "0" else f"{job_id}_poses_{idx}"
    if n is None and pct is None:
        return FileResponse(str(p), media_type="chemical/x-mdl-sdfile",
                            filename=f"{stem}.sdf")

    total = await run_in_threadpool(_count_sdf_records, p)
    if pct is not None:
        keep = -(-total * pct // 100)          # ceil: a non-zero % never rounds down to nothing
        keep = max(1, int(keep))
        label = f"top{pct:g}pct".replace(".", "_")
    else:
        keep = int(n)
        label = f"top{keep}"
    keep = min(keep, total)

    def gen():
        for i, record in enumerate(_iter_sdf_records(p)):
            if i >= keep:
                return
            yield record

    out_name = f"{stem}_{label}.sdf"
    return StreamingResponse(
        gen(), media_type="chemical/x-mdl-sdfile",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"',
                 "X-Poses-Total": str(total), "X-Poses-Returned": str(keep)})


@app.get("/jobs/{job_id}/pose/{rank}")
async def job_pose(job_id: str, rank: int) -> Response:
    """Download a single docked pose (SDF) by its 1-based gallery rank -- the
    ``DockingRank`` written into ``poses_0.sdf`` (best-scored first), which
    lines up with the results gallery's ``#rank``."""
    poses_path = _job_path(job_id, "poses_0.sdf")
    if not poses_path.is_file():
        raise HTTPException(404, "no docked poses for this job")
    sdf = await run_in_threadpool(_pose_record, poses_path, rank)
    if sdf is None:
        raise HTTPException(404, f"no pose with rank {rank}")
    return Response(content=sdf, media_type="chemical/x-mdl-sdfile",
                    headers={"Content-Disposition": f'attachment; filename="{job_id}_pose_{rank}.sdf"'})


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
        d = _job_path(job_id)
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

    poses_path = _job_path(job_id, "poses_0.sdf")
    if not poses_path.is_file():
        raise HTTPException(400, "no docked poses available to seed from")
    record = _pose_record(poses_path, rank)
    pose_mol = Chem.MolFromMolBlock(record, removeHs=False) if record else None
    if pose_mol is None:
        raise HTTPException(404, f"no pose found for rank {rank}")

    try:
        carved = carve_fragment(pose_mol, reagent["smiles"], accepts)
    except ValueError as e:
        raise HTTPException(400, str(e))

    sdf = Chem.MolToMolBlock(carved) + "$$$$\n"
    # Run name first, like the pose downloads, so everything saved out of one
    # session sorts together in the download folder.
    filename = f"{job_id}_fragment_rank{rank}_comp{component_index}.sdf"
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
