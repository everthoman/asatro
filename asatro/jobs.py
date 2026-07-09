"""In-process job layer for Asatro growth and combi runs.

Both run as a background thread and share the same ``GrowthJob`` bookkeeping
(status, live log lines, evaluator handle for live polling, persisted results):
a growth run does the accessibility pre-pass, then grows the surviving
reaction/slots (one run per target); a combi run is always exactly one
explicit, unanchored multi-step route (no pre-pass, no skips). Console lines
stream live; metadata + a results summary persist under the job dir so
finished runs stay viewable after a restart.

Mirrors the TS app's Job pattern, slimmed for growth (no CNN re-dock sub-run).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from asatro.combi import run_combi
from asatro.engine.gnina_evaluator import MolFilters
from asatro.growth import grow_accessible, run_growth
from asatro.svg import mol_svg

BASE_DIR = Path(__file__).resolve().parent.parent

TOP_N = 25  # hits kept per target in the persisted summary


def jobs_dir() -> Path:
    """Resolved fresh on every call (not cached at import time) so that setting
    ``ASATRO_JOBS_DIR`` -- including via ``monkeypatch.setenv`` in tests --
    actually takes effect. A module-level constant here previously froze the
    path at import time, so every test that set the env var to an isolated
    ``tmp_path`` silently kept writing into the real package ``jobs/``
    directory instead."""
    d = Path(os.environ.get("ASATRO_JOBS_DIR", str(BASE_DIR / "jobs")))
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class GrowthJob:
    id: str
    dir: Path
    status: str = "queued"            # queued | running | done | error | cancelled
    lines: List[str] = field(default_factory=list)
    error: Optional[str] = None
    thread: Optional[threading.Thread] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    result: Optional[dict] = None
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    _log_lock: threading.Lock = field(default_factory=threading.Lock)
    # Set while a target is actively docking so the UI can poll live progress
    # (structure gallery + convergence chart) before the job's final summary
    # exists. Cleared once the job finishes.
    evaluator: Optional[object] = None
    current_target: Optional[str] = None

    @property
    def log_path(self) -> Path:
        return self.dir / "run.log"

    def _append(self, line: str) -> None:
        with self._log_lock:
            self.lines.append(line)
            try:
                with open(self.log_path, "a") as fh:
                    fh.write(line + "\n")
            except Exception:
                pass

    def log(self, msg: str) -> None:
        self._append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def meta(self) -> dict:
        return {
            "id": self.id, "status": self.status, "error": self.error,
            "started": self.started, "finished": self.finished,
            "n_targets": len((self.result or {}).get("runs", [])),
        }


JOBS: Dict[str, GrowthJob] = {}


def _slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._-")[:64]


def _range_or_none(val) -> Optional[tuple]:
    """``[lo, hi]`` (either may be blank/None) -> ``(lo, hi)`` floats, or None if
    both ends are unset."""
    if not val:
        return None
    lo, hi = val
    lo = None if lo in (None, "") else float(lo)
    hi = None if hi in (None, "") else float(hi)
    if lo is None and hi is None:
        return None
    return (lo, hi)


def make_filters(cfg: dict) -> MolFilters:
    """Pre-docking PAINS/REOS/MW/logP filters from a job's ``filters`` config."""
    fcfg = cfg.get("filters") or {}
    return MolFilters(
        use_pains=bool(fcfg.get("pains", False)),
        use_reos=bool(fcfg.get("reos", False)),
        mw_range=_range_or_none(fcfg.get("mw")),
        logp_range=_range_or_none(fcfg.get("logp")),
    )


def make_class_resolver(reactant_by_class: Dict[str, str]):
    """A ReactantResolver that maps a component's accepted classes to the first
    uploaded reactant file tagged with one of those classes."""
    def resolve(reaction_id: str, component_index: int, accepts: List[str]):
        for cls in accepts:
            if cls in reactant_by_class:
                return reactant_by_class[cls]
        return None
    return resolve


def _summarize(out: dict, higher_is_better: Optional[bool], job_dir: Optional[Path] = None) -> dict:
    """JSON-safe summary of a grow_accessible result: per target, the top hits.

    ``sampler.search()`` only returns products docked during the *search*
    phase — warm-up docks (one per reagent, always real docking work) are
    never re-visited by search and so never appear in its return value. The
    evaluator's score cache (``top_scored`` / ``stats``) covers warm-up *and*
    search, so it — not the raw search rows — is the source of truth whenever
    a real evaluator is available. Without one (the fake runners used in
    tests) we fall back to ranking the rows we were handed directly."""
    runs = []
    for i, r in enumerate(out["runs"]):
        entry = {"target": r["target"]}
        if "skipped" in r:
            entry["skipped"] = r["skipped"]
            runs.append(entry)
            continue
        res = r["result"]
        rows, ev = res if isinstance(res, tuple) else (res, None)
        if ev is not None:
            ranked = ev.top_scored(TOP_N)
            n_docked = ev.stats()["unique_scored"]
        else:
            hib = bool(higher_is_better)
            ranked = sorted(rows, key=lambda x: x[0], reverse=hib)[:TOP_N]
            n_docked = len(rows)
        entry["n_docked"] = n_docked
        entry["top"] = [{"score": s, "smiles": sm, "name": nm, "svg": mol_svg(sm)}
                        for s, sm, nm in ranked]
        if ev is not None:
            pts = ev.convergence()
            st = ev.stats()
            entry["convergence"] = {
                "points": [{"dock": d, "best": b} for d, b in pts],
                "score_field": ev.score_field, "higher_better": bool(ev.higher_is_better),
                "docked": st["docked"], "best": st["best_score"],
            }
            if st.get("rejections"):
                entry["rejections"] = st["rejections"]
            if job_dir is not None:
                fname = f"poses_{i}.sdf"
                if ev.write_top_poses(str(job_dir / fname), n=TOP_N) > 0:
                    entry["poses"] = fname
        runs.append(entry)
    return {
        "fg_classes": out["assessment"]["fg_classes"],
        "accessible_reactions": out["assessment"]["accessible_reactions"],
        "runs": runs,
    }


def _summarize_combi(rows: list, evaluator, higher_is_better: Optional[bool],
                     job_dir: Optional[Path] = None) -> dict:
    """JSON-safe summary of a run_combi result, in the same ``{runs: [...]}``
    shape ``_summarize`` produces for a growth job -- a combi job is just
    always exactly one run (no accessibility pre-pass, no per-target skips),
    so it carries a single-element ``runs`` list."""
    if evaluator is not None:
        ranked = evaluator.top_scored(TOP_N)
        n_docked = evaluator.stats()["unique_scored"]
    else:
        hib = bool(higher_is_better)
        ranked = sorted(rows, key=lambda x: x[0], reverse=hib)[:TOP_N]
        n_docked = len(rows)
    entry = {"n_docked": n_docked,
             "top": [{"score": s, "smiles": sm, "name": nm, "svg": mol_svg(sm)}
                    for s, sm, nm in ranked]}
    if evaluator is not None:
        pts = evaluator.convergence()
        st = evaluator.stats()
        entry["convergence"] = {
            "points": [{"dock": d, "best": b} for d, b in pts],
            "score_field": evaluator.score_field, "higher_better": bool(evaluator.higher_is_better),
            "docked": st["docked"], "best": st["best_score"],
        }
        if st.get("rejections"):
            entry["rejections"] = st["rejections"]
        if job_dir is not None and evaluator.write_top_poses(str(job_dir / "poses_0.sdf"), n=TOP_N) > 0:
            entry["poses"] = "poses_0.sdf"
    return {"runs": [entry]}


def _run(job: GrowthJob, fragment_path: str, receptor_path: str,
         reactant_by_class: Dict[str, str], pool_path: Optional[str],
         cfg: dict, runner: Callable) -> None:
    job.status = "running"
    _persist_meta(job)
    job.log(f"Growth job {job.id} started")
    try:
        def _runner(**kw):
            def _on_evaluator(ev):
                job.evaluator = ev
                job.current_target = f"{kw.get('reaction_id')} (slot {kw.get('fragment_slot')})"
            return runner(progress_callback=job.log, cancel_event=job.cancel_event,
                          on_evaluator=_on_evaluator, **kw)

        if pool_path:
            from asatro.pool import Pool, pool_resolver
            pool = Pool.from_file(pool_path)
            job.log(f"Master pool: {pool.n_tagged}/{pool.n_total} blocks tagged "
                    f"— {pool.counts()}")
            resolver = pool_resolver(pool, str(job.dir / "pool"))
        else:
            resolver = make_class_resolver(reactant_by_class)

        mol_filters = make_filters(cfg)
        if mol_filters.active:
            job.log(f"Filters: PAINS {len(mol_filters.pains_patterns)} pattern(s), "
                    f"REOS {len(mol_filters.reos_rules)} rule(s), MW {mol_filters.mw_range}, "
                    f"logP {mol_filters.logp_range}")

        search_method = "rws" if str(cfg.get("search_method", "ts")).lower() == "rws" else "ts"
        job.log("Selection: Roulette Wheel Sampling + thermal cycling (Zhao 2025)"
                if search_method == "rws" else "Selection: standard Thompson Sampling (argmax)")

        out = grow_accessible(
            fragment_sdf=fragment_path, receptor_pdb=receptor_path,
            reactant_resolver=resolver,
            work_dir=str(job.dir / "runs"), runner=_runner, log=job.log,
            refine=bool(cfg.get("refine", False)),
            num_warmup=int(cfg.get("num_warmup", 3)),
            num_cycles=int(cfg.get("num_cycles", 25)),
            num_to_select=cfg.get("num_to_select"),
            seed=cfg.get("seed"),
            score_field=cfg.get("score_field", "minimizedAffinity"),
            cnn_scoring=cfg.get("cnn_scoring", "none"),
            filters=mol_filters if mol_filters.active else None,
            search_method=search_method,
            min_cpds_per_core=int(cfg.get("min_cpds_per_core", 50)),
            stop=int(cfg.get("stop", 6000)),
        )
        job.result = _summarize(out, higher_is_better=False, job_dir=job.dir)
        (job.dir / "results.json").write_text(json.dumps(job.result, indent=2))
        job.status = "cancelled" if job.cancel_event.is_set() else "done"
        job.log(f"Job {job.status} — {len(job.result['runs'])} target(s)")
    except Exception as e:  # noqa: BLE001 — surface any failure to the UI
        job.status = "error"
        job.error = str(e)
        job.log(f"ERROR: {e}")
    finally:
        job.evaluator = None
        job.current_target = None
        job.finished = time.time()
        _persist_meta(job)


def _run_combi(job: GrowthJob, receptor_path: str, steps: List[str],
              reagent_files: List[List[str]], reference_path: Optional[str],
              center: Optional[tuple], size: Optional[tuple],
              cfg: dict, runner: Callable) -> None:
    """Unanchored (plain ts-gnina) combinatorial search: one explicit
    multi-step route, no accessibility pre-pass and so no per-target loop --
    contrast with ``_run``'s multiple growth targets."""
    job.status = "running"
    _persist_meta(job)
    job.log(f"Combi job {job.id} started — route {' -> '.join(steps)}")
    try:
        def _on_evaluator(ev):
            job.evaluator = ev
            job.current_target = " -> ".join(steps)

        mol_filters = make_filters(cfg)
        if mol_filters.active:
            job.log(f"Filters: PAINS {len(mol_filters.pains_patterns)} pattern(s), "
                    f"REOS {len(mol_filters.reos_rules)} rule(s), MW {mol_filters.mw_range}, "
                    f"logP {mol_filters.logp_range}")

        search_method = "rws" if str(cfg.get("search_method", "ts")).lower() == "rws" else "ts"
        job.log("Selection: Roulette Wheel Sampling + thermal cycling (Zhao 2025)"
                if search_method == "rws" else "Selection: standard Thompson Sampling (argmax)")

        rows, evaluator = runner(
            receptor_path=receptor_path, steps=steps, reagent_files=reagent_files,
            work_dir=str(job.dir / "run"), reference_path=reference_path,
            center=center, size=size,
            num_warmup=int(cfg.get("num_warmup", 3)),
            num_cycles=int(cfg.get("num_cycles", 25)),
            num_to_select=cfg.get("num_to_select"),
            seed=cfg.get("seed"),
            score_field=cfg.get("score_field", "minimizedAffinity"),
            cnn_scoring=cfg.get("cnn_scoring", "none"),
            filters=mol_filters if mol_filters.active else None,
            search_method=search_method,
            min_cpds_per_core=int(cfg.get("min_cpds_per_core", 50)),
            stop=int(cfg.get("stop", 6000)),
            progress_callback=job.log, cancel_event=job.cancel_event,
            on_evaluator=_on_evaluator,
        )
        job.result = _summarize_combi(rows, evaluator, higher_is_better=False, job_dir=job.dir)
        (job.dir / "results.json").write_text(json.dumps(job.result, indent=2))
        job.status = "cancelled" if job.cancel_event.is_set() else "done"
        job.log(f"Job {job.status} — {job.result['runs'][0]['n_docked']} docked")
    except Exception as e:  # noqa: BLE001 — surface any failure to the UI
        job.status = "error"
        job.error = str(e)
        job.log(f"ERROR: {e}")
    finally:
        job.evaluator = None
        job.current_target = None
        job.finished = time.time()
        _persist_meta(job)


def _persist_meta(job: GrowthJob) -> None:
    try:
        (job.dir / "job.json").write_text(json.dumps(job.meta(), indent=2, default=str))
    except Exception:
        pass


def start_growth_job(*, fragment_path: str, receptor_path: str,
                     reactant_by_class: Optional[Dict[str, str]] = None,
                     pool_path: Optional[str] = None, cfg: Optional[dict] = None,
                     session_name: str = "", runner: Optional[Callable] = None) -> GrowthJob:
    """Create a job dir, register the job, and run it in a background thread.

    Reactants come from either per-class files (``reactant_by_class``) or a single
    tagged master pool (``pool_path``), pruned per reaction component. ``runner``
    defaults to :func:`asatro.growth.run_growth` (resolved at call time so it stays
    patchable) and is injectable so the job layer can be driven without docking."""
    runner = runner or run_growth
    job_id = _slugify(session_name) or uuid.uuid4().hex[:12]
    job_dir = jobs_dir() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job = GrowthJob(id=job_id, dir=job_dir)
    JOBS[job_id] = job
    job.thread = threading.Thread(
        target=_run,
        args=(job, fragment_path, receptor_path, reactant_by_class or {}, pool_path,
              cfg or {}, runner),
        daemon=True,
    )
    job.thread.start()
    return job


def start_combi_job(*, receptor_path: str, steps: List[str], reagent_files: List[List[str]],
                    reference_path: Optional[str] = None, center: Optional[tuple] = None,
                    size: Optional[tuple] = None, cfg: Optional[dict] = None,
                    session_name: str = "", runner: Optional[Callable] = None) -> GrowthJob:
    """Create a job dir, register the job, and run an unanchored combi search in
    a background thread. ``runner`` defaults to :func:`asatro.combi.run_combi`
    (resolved at call time so it stays patchable), same convention as
    :func:`start_growth_job`."""
    runner = runner or run_combi
    job_id = _slugify(session_name) or uuid.uuid4().hex[:12]
    job_dir = jobs_dir() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job = GrowthJob(id=job_id, dir=job_dir)
    JOBS[job_id] = job
    job.thread = threading.Thread(
        target=_run_combi,
        args=(job, receptor_path, steps, reagent_files, reference_path, center, size,
              cfg or {}, runner),
        daemon=True,
    )
    job.thread.start()
    return job


def list_jobs() -> List[dict]:
    """Live jobs first, then any persisted-only past runs on disk (newest first)."""
    items: List[dict] = []
    seen = set()
    for d in sorted(jobs_dir().iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        live = JOBS.get(d.name)
        if live is not None:
            items.append(live.meta())
        else:
            f = d / "job.json"
            if f.is_file():
                try:
                    items.append(json.loads(f.read_text()))
                except Exception:
                    pass
        seen.add(d.name)
    return items
