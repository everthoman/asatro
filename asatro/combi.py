"""TS-driven combinatorial search — the plain ts-gnina path, unanchored.

No bound fragment: every slot of a (possibly multi-step) reaction route is
Thompson-sampled from a real reagent library, each product is freely embedded
and docked with GNINA (autoboxed on a reference ligand or an explicit pocket),
and the search ranks by score alone. This is ts-gnina's own search, reusing the
same lifted engine (``RouteSampler``, ``GninaEvaluator``) that ``growth.py``
builds the fragment-anchored path on top of.

Route building deliberately stays a pure function, same as
``growth.build_growth_route``: named-set/pool resolution to concrete ``.smi``
paths is a job-layer concern (mirrors ts-gnina's own ``_resolve_set`` /
``_build_route_pool``), not something this module does.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from asatro.chemistry.catalog import REACTION_BY_ID
from asatro.engine.gnina_evaluator import GninaEvaluator, MolFilters
from asatro.engine.route_sampler import RouteSampler


def build_combi_route(steps: List[str], reagent_files: List[List[str]], work_dir: Path
                      ) -> Tuple[List[str], List[Tuple[str, int]], List[str]]:
    """Flat reagent-file list (route order) + the multi-step route + a
    human-readable summary, mirroring ts-gnina's own ``_build_route``.

    ``steps`` is a list of reaction ids: the first must be a ``"start"``
    reaction, every later one an ``"extend"`` reaction (consumes the running
    intermediate plus its own new reagent(s)). ``reagent_files[i]`` is the
    ordered list of already-resolved ``.smi`` paths for step ``i``'s
    components — one per entry in that reaction's ``components``.
    """
    if not steps:
        raise ValueError("no reaction steps given")
    if len(reagent_files) != len(steps):
        raise ValueError(
            f"reagent_files has {len(reagent_files)} step(s), steps has {len(steps)}")
    files: List[str] = []
    route: List[Tuple[str, int]] = []
    summary: List[str] = []
    for i, rid in enumerate(steps):
        rxn = REACTION_BY_ID.get(rid)
        if rxn is None:
            raise KeyError(f"unknown reaction: {rid}")
        role = rxn.get("role")
        if i == 0 and role != "start":
            raise ValueError(f"step 1 must be a 'start' reaction, got '{rid}' ({role})")
        if i > 0 and role != "extend":
            raise ValueError(f"step {i + 1} must be an 'extend' reaction, got '{rid}' ({role})")
        comps = rxn["components"]
        step_files = reagent_files[i]
        if len(step_files) != len(comps):
            raise ValueError(
                f"'{rid}' needs {len(comps)} reagent file(s) for step {i + 1}, "
                f"got {len(step_files)}")
        files.extend(step_files)
        route.append((rxn["smarts"], len(comps)))
        labels = [f"{c['label']} [{Path(f).name}]" for c, f in zip(comps, step_files)]
        summary.append(f"Step {i + 1}: {rxn['name']} [{', '.join(labels)}]")
    return files, route, summary


def make_evaluator(*, receptor_path: str, work_dir: str,
                   reference_path: Optional[str] = None,
                   center: Optional[Tuple[float, float, float]] = None,
                   size: Optional[Tuple[float, float, float]] = None,
                   score_field: str = "minimizedAffinity", cnn_scoring: str = "none",
                   filters: Optional[MolFilters] = None, **extra) -> GninaEvaluator:
    """Build the plain (unanchored) evaluator: free ETKDG embed + full re-dock,
    autoboxed on either a reference ligand or an explicit pocket center/size.
    No fragment, no core pinning — this is exactly ts-gnina's own evaluator."""
    if not reference_path and center is None:
        raise ValueError("combi.make_evaluator: give reference_path or center")
    d = dict(receptor_path=receptor_path, work_dir=work_dir, score_field=score_field,
             cnn_scoring=cnn_scoring)
    if reference_path:
        d["reference_path"] = reference_path
    else:
        d["center"] = center
        if size is not None:
            d["size"] = size
    if filters is not None:
        d["filters"] = filters
    d.update(extra)
    return GninaEvaluator(d)


def run_combi(*, receptor_path: str, steps: List[str], reagent_files: List[List[str]],
             work_dir: str, reference_path: Optional[str] = None,
             center: Optional[Tuple[float, float, float]] = None,
             size: Optional[Tuple[float, float, float]] = None,
             num_warmup: int = 3, num_cycles: int = 25,
             num_to_select: Optional[int] = None, seed: Optional[int] = None,
             mode: str = "minimize", concurrency: int = 1, hide_progress: bool = True,
             search_method: str = "ts", min_cpds_per_core: int = 50, stop: int = 6000,
             on_evaluator: Optional[Callable[[object], None]] = None,
             **gnina_opts):
    """Run the full combinatorial search. Returns ``(results, evaluator)``,
    the same shape as ``growth.run_growth`` so a caller can share summarization
    code between the two paths.

    Unlike ``growth.run_growth`` there is no "single variable slot" shortcut:
    a combi route has no fragment fixing a slot, so every component genuinely
    varies and the search always runs — matching ts-gnina's own dispatch.
    ``on_evaluator``, if given, is called with the evaluator as soon as it's
    built (before the search runs), same convention as ``run_growth``."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    files, route, summary = build_combi_route(steps, reagent_files, work)

    evaluator = make_evaluator(receptor_path=receptor_path, reference_path=reference_path,
                               center=center, size=size, work_dir=str(work / "dock"),
                               **gnina_opts)
    if on_evaluator is not None:
        on_evaluator(evaluator)
    if evaluator.progress_callback is not None:
        for line in summary:
            evaluator.progress_callback(line)

    sampler = RouteSampler(mode=mode)
    if seed is not None:
        sampler.set_seed(seed)
    sampler.set_concurrency(concurrency)
    sampler.set_hide_progress(hide_progress)
    sampler.read_reagents(reagent_file_list=files, num_to_select=num_to_select)
    sampler.set_route(route)
    sampler.set_evaluator(evaluator)

    if search_method == "rws":
        warmup_results = sampler.warm_up_rws(num_warmup_trials=num_warmup)
        if not warmup_results:
            # Same bail-out as run_growth: nothing scored means the per-reagent
            # posteriors search_rws needs were never seeded.
            results = []
        else:
            search_results = sampler.search_rws(
                num_targets=num_cycles, min_cpds_per_core=min_cpds_per_core, stop=stop)
            results = warmup_results + search_results
    else:
        sampler.warm_up(num_warmup_trials=num_warmup)
        results = sampler.search(num_cycles=num_cycles)
    return results, evaluator
