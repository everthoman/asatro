"""In-process job layer for Asatro growth runs.

A growth run is a background thread: run the accessibility pre-pass, then grow the
surviving reaction/slots. Console lines stream live; metadata + a results summary
persist under the job dir so finished runs stay viewable after a restart.

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

from asatro.growth import grow_accessible, run_growth

BASE_DIR = Path(__file__).resolve().parent.parent
JOBS_DIR = Path(os.environ.get("ASATRO_JOBS_DIR", str(BASE_DIR / "jobs")))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 25  # hits kept per target in the persisted summary


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


def make_class_resolver(reactant_by_class: Dict[str, str]):
    """A ReactantResolver that maps a component's accepted classes to the first
    uploaded reactant file tagged with one of those classes."""
    def resolve(reaction_id: str, component_index: int, accepts: List[str]):
        for cls in accepts:
            if cls in reactant_by_class:
                return reactant_by_class[cls]
        return None
    return resolve


def _summarize(out: dict, higher_is_better: Optional[bool]) -> dict:
    """JSON-safe summary of a grow_accessible result: per target, the top hits."""
    runs = []
    for r in out["runs"]:
        entry = {"target": r["target"]}
        if "skipped" in r:
            entry["skipped"] = r["skipped"]
            runs.append(entry)
            continue
        res = r["result"]
        rows, ev = res if isinstance(res, tuple) else (res, None)
        hib = ev.higher_is_better if ev is not None else bool(higher_is_better)
        ranked = sorted(rows, key=lambda x: x[0], reverse=hib)[:TOP_N]
        entry["n_docked"] = len(rows)
        entry["top"] = [{"score": s, "smiles": sm, "name": nm} for s, sm, nm in ranked]
        runs.append(entry)
    return {
        "fg_classes": out["assessment"]["fg_classes"],
        "accessible_reactions": out["assessment"]["accessible_reactions"],
        "runs": runs,
    }


def _run(job: GrowthJob, fragment_path: str, receptor_path: str,
         reactant_by_class: Dict[str, str], pool_path: Optional[str],
         cfg: dict, runner: Callable) -> None:
    job.status = "running"
    _persist_meta(job)
    job.log(f"Growth job {job.id} started")
    try:
        def _runner(**kw):
            return runner(progress_callback=job.log, cancel_event=job.cancel_event, **kw)

        if pool_path:
            from asatro.pool import Pool, pool_resolver
            pool = Pool.from_file(pool_path)
            job.log(f"Master pool: {pool.n_tagged}/{pool.n_total} blocks tagged "
                    f"— {pool.counts()}")
            resolver = pool_resolver(pool, str(job.dir / "pool"))
        else:
            resolver = make_class_resolver(reactant_by_class)

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
        )
        job.result = _summarize(out, higher_is_better=False)
        (job.dir / "results.json").write_text(json.dumps(job.result, indent=2))
        job.status = "cancelled" if job.cancel_event.is_set() else "done"
        job.log(f"Job {job.status} — {len(job.result['runs'])} target(s)")
    except Exception as e:  # noqa: BLE001 — surface any failure to the UI
        job.status = "error"
        job.error = str(e)
        job.log(f"ERROR: {e}")
    finally:
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
    job_dir = JOBS_DIR / job_id
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


def list_jobs() -> List[dict]:
    """Live jobs first, then any persisted-only past runs on disk (newest first)."""
    items: List[dict] = []
    seen = set()
    for d in sorted(JOBS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
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
