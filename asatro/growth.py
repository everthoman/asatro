"""TS-driven fragment growth.

Fix the bound fragment in one slot of a (possibly multi-step) route, Thompson-
sample the reactant library in every other slot, and for each sampled final
product constrained-place it onto the bound pose and score it with GNINA — via
the lifted ``AnchoredFragmentEvaluator`` and ``RouteSampler``.

The fragment is "fixed" the same way the TS app did it: written as a one-line
reagent file, so the sampler always picks it for that component while the real
search happens over the other component(s). The conserved core (from the handle
analysis) anchors the placement; pick it with ``derive_core`` so the reacting
handle is excluded. A route is a user-chosen chain: step 0 is a "start"
reaction with the fragment fixed into one slot; any further steps are "extend"
reactions consuming the running intermediate plus their own new reagent(s) —
same route shape as ``combi.build_combi_route``, just with the fragment
occupying one of step 0's slots instead of every slot being a real library.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from rdkit import Chem

from asatro.chemistry.catalog import StepSpec, resolve_step
from asatro.chemistry.handles import neutralize
from asatro.chemistry.reachability import prune_unreachable_reagents
from asatro.engine.anchored_fragment_evaluator import AnchoredFragmentEvaluator
from asatro.engine.gnina_evaluator import MolFilters
from asatro.engine.route_sampler import RouteSampler
from asatro.engine.ts_autoparams import resolve_ts_budget, suggest_ts_params


def fragment_smiles_from_sdf(fragment_sdf: str) -> str:
    mol = Chem.MolFromMolFile(fragment_sdf, removeHs=True)
    if mol is None:
        raise ValueError(f"could not read fragment SDF: {fragment_sdf}")
    return Chem.MolToSmiles(neutralize(mol))  # neutral so reaction templates match


def write_fragment_smi(smiles: str, work_dir: Path, name: str = "FRAG") -> str:
    """The fixed fragment as a one-line reagent file (single-entry component)."""
    p = Path(work_dir) / "fragment.smi"
    p.write_text(f"{smiles.strip()}\t{name}\n")
    return str(p)


def build_growth_route(steps: List[StepSpec], fragment_smiles: str, fragment_slot: int,
                       reactant_files: List[Dict[int, str]], work_dir: Path
                       ) -> Tuple[List[str], List[Tuple[str, int, Optional[int]]], List[str]]:
    """Reagent-file list (route order) + the multi-step route + a human-readable
    summary, mirroring ``combi.build_combi_route`` — except step 0's fragment
    slot is filled by the bound fragment (a one-entry reagent file) instead of a
    real library, so it never varies.

    ``steps`` is a list of reaction ids (or ``{"reaction_id", "slot"}`` dicts
    for steps 1+ that reuse a multi-component reaction generically -- see
    ``asatro.chemistry.catalog.resolve_step``): the first must be a
    ``"start"`` reaction, every later one consumes the running intermediate
    plus its own new reagent(s). ``reactant_files[i]`` maps component index ->
    resolved ``.smi`` path for step ``i``'s non-fragment/non-intermediate
    components (step 0 excludes ``fragment_slot``; later steps exclude
    whichever slot binds the intermediate)."""
    if not steps:
        raise ValueError("no reaction steps given")
    if len(reactant_files) != len(steps):
        raise ValueError(
            f"reactant_files has {len(reactant_files)} step(s), steps has {len(steps)}")
    files: List[str] = []
    route: List[Tuple[str, int, Optional[int]]] = []
    summary: List[str] = []
    for i, step in enumerate(steps):
        info = resolve_step(step, i)
        rid, rxn = info["reaction_id"], info["rxn"]
        comps = rxn["components"]
        if i == 0 and not 0 <= fragment_slot < len(comps):
            raise ValueError(f"fragment_slot {fragment_slot + 1} out of range for '{rid}' "
                             f"({len(comps)} components)")
        step_files: List[str] = []
        labels: List[str] = []
        for ci in info["fresh_indices"]:
            comp = comps[ci]
            if i == 0 and ci == fragment_slot:
                f = write_fragment_smi(fragment_smiles, work_dir)
                labels.append(f"{comp['label']} = bound fragment")
            else:
                f = reactant_files[i].get(ci)
                if not f:
                    raise ValueError(
                        f"no reactant file for component {ci} of '{rid}' (step {i + 1})")
                labels.append(f"{comp['label']} [{Path(f).name}]")
            step_files.append(f)
        files.extend(step_files)
        route.append((rxn["smarts"], len(info["fresh_indices"]), info["intermediate_slot"]))
        summary.append(f"Step {i + 1}: {rxn['name']} [{', '.join(labels)}]")
    return files, route, summary


def resolve_reactant_files(steps: List[StepSpec], fragment_slot: int, resolver
                           ) -> List[Dict[int, str]]:
    """Resolve each step's fresh reagent components to ``.smi`` paths via
    ``resolver`` (a pool- or class-backed ReactantResolver), skipping step 0's
    fragment slot. Shared by the live run (``jobs._run``) and the pre-launch
    parameter suggestion (:func:`suggest_growth_params`) so both compute over
    the exact same resolved pools."""
    reactant_files: List[Dict[int, str]] = []
    for i, step in enumerate(steps):
        step_info = resolve_step(step, i)
        rid, rxn = step_info["reaction_id"], step_info["rxn"]
        step_map: Dict[int, str] = {}
        for ci in step_info["fresh_indices"]:
            if i == 0 and ci == fragment_slot:
                continue
            comp = rxn["components"][ci]
            path = resolver(rid, ci, comp.get("accepts", []))
            if not path:
                raise ValueError(
                    f"no reactant library for component {ci} of '{rid}' (step {i + 1})")
            step_map[ci] = path
        reactant_files.append(step_map)
    return reactant_files


def suggest_growth_params(*, fragment_sdf: str, steps: List[StepSpec],
                          fragment_slot: int, resolver, work_dir: str) -> dict:
    """Dry-run the pool resolution + reachability prune (**no docking**) and
    report the post-prune variable-slot sizes plus a suggested TS budget, so the
    UI can show/fill ``num_warmup``/``num_cycles`` before launch. Read-only;
    raises like a real run would on a bad route or an empty (unreachable) pool.

    Returns ``{"mode": "search", "num_warmup", "num_cycles", "variable_slots",
    "est_docks"}`` for a real TS search, or ``{"mode": "exhaustive",
    "candidates", "variable_slots"}`` when at most one slot varies (the run
    would dock the whole small library and skip warm-up/search)."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    frag_smiles = fragment_smiles_from_sdf(fragment_sdf)
    reactant_files = resolve_reactant_files(steps, fragment_slot, resolver)
    files, route, _summary = build_growth_route(
        steps, frag_smiles, fragment_slot, reactant_files, work)
    files = prune_unreachable_reagents(route, files, work_dir=str(work / "reachability"))
    sizes = [sum(1 for _ in open(f)) for f in files]
    variable = [s for s in sizes if s > 1]
    if len(variable) <= 1:
        return {"mode": "exhaustive", "candidates": int(math.prod(sizes) if sizes else 0),
                "variable_slots": variable}
    num_warmup, num_cycles = suggest_ts_params(variable)
    est_docks = num_warmup * sum(sizes) + num_cycles
    return {"mode": "search", "num_warmup": num_warmup, "num_cycles": num_cycles,
            "variable_slots": variable, "est_docks": est_docks}


def make_evaluator(*, fragment_sdf: str, receptor_path: str, core_smarts: Optional[str],
                   work_dir: str, score_field: str = "minimizedAffinity",
                   cnn_scoring: str = "none", max_core_rmsd: float = 1.5,
                   local_only: bool = True, filters: Optional[MolFilters] = None,
                   **extra) -> AnchoredFragmentEvaluator:
    """Build the anchored evaluator. The fragment SDF doubles as the autobox
    reference, so no separate binding site is needed."""
    d = dict(receptor_path=receptor_path, fragment_sdf=fragment_sdf,
             core_smarts=core_smarts, work_dir=work_dir, score_field=score_field,
             cnn_scoring=cnn_scoring, max_core_rmsd=max_core_rmsd, local_only=local_only)
    if filters is not None:
        d["filters"] = filters
    d.update(extra)
    return AnchoredFragmentEvaluator(d)


def run_growth(*, fragment_sdf: str, receptor_path: str, steps: List[StepSpec],
               fragment_slot: int, core_smarts: Optional[str],
               reactant_files: List[Dict[int, str]], work_dir: str,
               num_warmup: Optional[int] = None, num_cycles: Optional[int] = None,
               num_to_select: Optional[int] = None, seed: Optional[int] = None,
               mode: str = "minimize", concurrency: int = 1,
               hide_progress: bool = True,
               search_method: str = "ts", min_cpds_per_core: int = 50, stop: int = 6000,
               max_core_rmsd: float = 1.5, prune_unreachable: bool = True,
               on_evaluator: Optional[Callable[[object], None]] = None,
               **gnina_opts):
    """Run the full growth search over a user-chosen (possibly multi-step)
    route. Returns ``(results, evaluator)`` where results is the list of
    ``[score, smiles, name]`` rows the sampler collected.

    ``max_core_rmsd`` (A) is the placement guard: ``AnchoredFragmentEvaluator``
    rejects any docked pose whose conserved-core atoms drift more than this
    from the bound reference -- the fragment's known binding mode has to
    survive the grow, however many steps it took, or the product doesn't count.

    If at most one reagent component actually varies across the whole route
    (true of a single-step start reaction: the fragment fixes one slot, one
    reagent library fills the other), ``search_method``/``num_warmup``/
    ``num_cycles`` are moot -- the whole library is small enough to just dock
    exhaustively, and there's no unseen combination left for a bandit search to
    find once warm-up alone would have touched every reagent. Otherwise (2+
    variable slots -- the common case once an extend step is chained on),
    ``search_method`` picks the sampler: ``"ts"`` (default) is argmax Thompson
    Sampling; ``"rws"`` is Roulette Wheel Selection with thermal cycling (Zhao
    et al. 2025), which trades some greediness for better coverage of
    3+-component routes on ultralarge libraries. ``num_cycles`` is the shared
    search budget (search iterations for TS, unique products to dock for RWS)
    so a TS vs RWS run on the same budget is comparable; ``min_cpds_per_core``/
    ``stop`` only apply to RWS.

    ``on_evaluator``, if given, is called with the evaluator as soon as it's built
    (before the search runs) so a caller can stash a live reference — e.g. for a
    web UI to poll ``top_scored()``/``convergence()`` while docking is still in
    progress."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    files, route, summary = build_growth_route(
        steps, fragment_smiles_from_sdf(fragment_sdf), fragment_slot,
        reactant_files, work)

    evaluator = make_evaluator(fragment_sdf=fragment_sdf, receptor_path=receptor_path,
                               core_smarts=core_smarts, work_dir=str(work / "dock"),
                               max_core_rmsd=max_core_rmsd, **gnina_opts)
    if on_evaluator is not None:
        on_evaluator(evaluator)
    if evaluator.progress_callback is not None:
        for line in summary:
            evaluator.progress_callback(line)
    if prune_unreachable:
        # Drop reagents that can't complete a downstream step *before* the
        # search wastes warm-up on them (see reachability.py). Exact prune.
        files = prune_unreachable_reagents(
            route, files, work_dir=str(work / "reachability"),
            log=evaluator.progress_callback)
    sampler = RouteSampler(mode=mode)
    if seed is not None:
        sampler.set_seed(seed)
    sampler.set_concurrency(concurrency)
    sampler.set_hide_progress(hide_progress)
    sampler.read_reagents(reagent_file_list=files, num_to_select=num_to_select)
    sampler.set_route(route)
    sampler.set_evaluator(evaluator)
    n_variable = sum(1 for rl in sampler.reagent_lists if len(rl) > 1)
    if n_variable <= 1:
        # At most one reagent slot actually varies (the fragment fills the
        # other(s) with a single fixed choice) -- warm-up would have to touch
        # every reagent to seed its prior anyway, and a subsequent adaptive
        # search has no unseen combination left to find. Skip the TS/RWS
        # machinery entirely and just dock the whole (small) library once.
        if evaluator.progress_callback is not None:
            evaluator.progress_callback(
                f"Single variable slot ({sampler.num_prods} candidates) — "
                f"exhaustive dock, no warm-up/search split")
        results = sampler.dock_all()
        return results, evaluator
    # Two or more variable slots: fill in an auto TS budget from the (pruned)
    # pool sizes for any of num_warmup/num_cycles the caller left unset.
    num_warmup, num_cycles = resolve_ts_budget(
        num_warmup, num_cycles, sampler.reagent_lists, log=evaluator.progress_callback)
    if search_method == "rws":
        warmup_results = sampler.warm_up_rws(num_warmup_trials=num_warmup)
        if not warmup_results:
            # search_rws needs the per-reagent posteriors warm_up_rws seeds;
            # nothing scored means those were never initialized, and searching
            # further would just repeat the same all-nan warm-up. Bail out
            # cleanly instead of the AttributeError search_rws would raise.
            results = []
        else:
            search_results = sampler.search_rws(
                num_targets=num_cycles, min_cpds_per_core=min_cpds_per_core, stop=stop)
            results = warmup_results + search_results
    else:
        warmup_results = sampler.warm_up(num_warmup_trials=num_warmup)
        if not warmup_results:
            # search() draws from per-reagent priors warm_up() seeds; nothing
            # scored means those were never initialized (every reagent is still
            # in its uninitialized "warmup" phase), so searching further would
            # sample meaningless all-zero priors. Bail out cleanly, mirroring
            # the RWS branch's guard above.
            results = []
        else:
            results = sampler.search(num_cycles=num_cycles)
    return results, evaluator
