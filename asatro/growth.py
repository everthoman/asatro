"""TS-driven fragment growth.

Fix the bound fragment in one slot of a start reaction, Thompson-sample the
reactant library in the other slot(s), and for each sampled product
constrained-place it onto the bound pose and score it with GNINA — via the
lifted ``AnchoredFragmentEvaluator`` and ``RouteSampler``.

The fragment is "fixed" the same way the TS app did it: written as a one-line
reagent file, so the sampler always picks it for that component while the real
search happens over the other component(s). The conserved core (from the handle
analysis) anchors the placement; pick it with ``derive_core`` so the reacting
handle is excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from rdkit import Chem

from asatro.chemistry.accessibility import ProbeParams, assess_fragment, load_receptor_atoms
from asatro.chemistry.catalog import REACTION_BY_ID
from asatro.chemistry.handles import neutralize
from asatro.chemistry.stub_growth import StubParams, assess_with_stubs
from asatro.engine.anchored_fragment_evaluator import AnchoredFragmentEvaluator
from asatro.engine.gnina_evaluator import MolFilters
from asatro.engine.route_sampler import RouteSampler

# A reactant resolver maps (reaction_id, component_index, accepts_classes) to a
# .smi path for that non-fragment component, or None if it can't supply one.
ReactantResolver = Callable[[str, int, List[str]], Optional[str]]


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


def build_growth_route(reaction_id: str, fragment_smiles: str, fragment_slot: int,
                       reactant_files: Dict[int, str], work_dir: Path
                       ) -> Tuple[List[str], List[Tuple[str, int]]]:
    """Reagent-file list (component order) + the single-step route.

    The fragment fills ``fragment_slot``; every other component must have a path
    in ``reactant_files`` (keyed by component index). All components are sampled
    in step 0 — the fragment's is just a one-entry list, so it never varies."""
    rxn = REACTION_BY_ID.get(reaction_id)
    if rxn is None:
        raise KeyError(f"unknown reaction: {reaction_id}")
    if rxn.get("role") != "start":
        raise ValueError(f"growth seeds on a 'start' reaction, got '{reaction_id}'")
    ncomp = len(rxn["components"])
    if not 0 <= fragment_slot < ncomp:
        raise ValueError(f"fragment_slot {fragment_slot} out of range for {reaction_id} "
                         f"({ncomp} components)")
    files: List[str] = []
    for i in range(ncomp):
        if i == fragment_slot:
            files.append(write_fragment_smi(fragment_smiles, work_dir))
        else:
            f = reactant_files.get(i)
            if not f:
                raise ValueError(f"no reactant file for component {i} of '{reaction_id}'")
            files.append(f)
    return files, [(rxn["smarts"], ncomp)]


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


def run_growth(*, fragment_sdf: str, receptor_path: str, reaction_id: str,
               fragment_slot: int, core_smarts: Optional[str],
               reactant_files: Dict[int, str], work_dir: str,
               num_warmup: int = 3, num_cycles: int = 25,
               num_to_select: Optional[int] = None, seed: Optional[int] = None,
               mode: str = "minimize", concurrency: int = 1,
               hide_progress: bool = True,
               search_method: str = "ts", min_cpds_per_core: int = 50, stop: int = 6000,
               on_evaluator: Optional[Callable[[object], None]] = None,
               **gnina_opts):
    """Run the full growth search. Returns ``(results, evaluator)`` where results
    is the list of ``[score, smiles, name]`` rows the sampler collected.

    If at most one reagent component actually varies (true of every current
    start-reaction route: the fragment fixes one slot, one reagent library
    fills the other), ``search_method``/``num_warmup``/``num_cycles`` are
    moot -- the whole library is small enough to just dock exhaustively, and
    there's no unseen combination left for a bandit search to find once
    warm-up alone would have touched every reagent. Otherwise (2+ variable
    slots), ``search_method`` picks the sampler: ``"ts"`` (default) is argmax
    Thompson Sampling; ``"rws"`` is Roulette Wheel Selection with thermal
    cycling (Zhao et al. 2025), which trades some greediness for better
    coverage of 3+-component routes on ultralarge libraries. ``num_cycles`` is
    the shared search budget (search iterations for TS, unique products to
    dock for RWS) so a TS vs RWS run on the same budget is comparable;
    ``min_cpds_per_core``/``stop`` only apply to RWS.

    ``on_evaluator``, if given, is called with the evaluator as soon as it's built
    (before the search runs) so a caller can stash a live reference — e.g. for a
    web UI to poll ``top_scored()``/``convergence()`` while docking is still in
    progress."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    files, route = build_growth_route(
        reaction_id, fragment_smiles_from_sdf(fragment_sdf), fragment_slot,
        reactant_files, work)

    evaluator = make_evaluator(fragment_sdf=fragment_sdf, receptor_path=receptor_path,
                               core_smarts=core_smarts, work_dir=str(work / "dock"),
                               **gnina_opts)
    if on_evaluator is not None:
        on_evaluator(evaluator)
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
    elif search_method == "rws":
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
        sampler.warm_up(num_warmup_trials=num_warmup)
        results = sampler.search(num_cycles=num_cycles)
    return results, evaluator


# ---------------------------------------------------------------------------
# Accessibility-gated growth: only grow vectors that survived the pre-pass.
# ---------------------------------------------------------------------------
@dataclass
class GrowthTarget:
    reaction_id: str
    fragment_slot: int
    fg_class: str
    core_smarts: str


def plan_targets(assessment: dict) -> List[GrowthTarget]:
    """Turn an accessibility assessment into the list of growth targets to run:
    one per accessible slot of each accessible reaction (a reaction with two
    accessible slots — e.g. an amino-acid fragment in amide coupling — yields two
    targets, the fragment growing as either partner). The auto-derived conserved
    core is carried through as the placement anchor."""
    targets: List[GrowthTarget] = []
    for rid, info in assessment["reactions"].items():
        if not info.get("accessible"):
            continue
        for slot in info["slots"]:
            if not slot.get("accessible", True):
                continue
            targets.append(GrowthTarget(rid, slot["index"], slot["fg_class"],
                                        slot["core_smarts"]))
    return targets


def grow_accessible(*, fragment_sdf: str, receptor_pdb: str,
                    reactant_resolver: ReactantResolver, work_dir: str,
                    refine: bool = False, probe_params: Optional[ProbeParams] = None,
                    stub_params: Optional[StubParams] = None,
                    runner: Callable = run_growth,
                    log: Optional[Callable[[str], None]] = None,
                    **growth_opts) -> dict:
    """Run the accessibility pre-pass, then grow only the surviving reaction/slots.

    For each accessible target, the non-fragment components are resolved to
    reactant files via ``reactant_resolver``; targets missing a reactant are
    recorded as skipped (not grown). ``runner`` defaults to :func:`run_growth`
    and is injectable so the pipeline can be exercised without docking.

    Returns ``{assessment, targets, runs}`` — ``runs`` carries each target's
    resolved reactant files and the runner's result (or a skip reason)."""
    _log = log or (lambda _m: None)
    mol = Chem.MolFromMolFile(fragment_sdf, removeHs=True)
    if mol is None:
        raise ValueError(f"could not read fragment SDF: {fragment_sdf}")
    if mol.GetNumConformers() == 0:
        raise ValueError("fragment SDF has no 3D conformer (need the bound pose)")
    receptor = load_receptor_atoms(receptor_pdb)
    _log(f"Accessibility pre-pass ({'geometric+stub' if refine else 'geometric'}) "
         f"on {receptor.shape[0]} receptor atoms…")

    if refine:
        assessment = assess_with_stubs(mol, receptor, probe_params or ProbeParams(),
                                       stub_params or StubParams())
    else:
        assessment = assess_fragment(mol, receptor, probe_params or ProbeParams())

    targets = plan_targets(assessment)
    _log(f"Handles {assessment['fg_classes']}; accessible reactions "
         f"{assessment['accessible_reactions']} → {len(targets)} growth target(s)")
    runs: List[dict] = []
    for t in targets:
        rxn = REACTION_BY_ID[t.reaction_id]
        reactant_files: Dict[int, str] = {}
        missing = None
        for ci, comp in enumerate(rxn["components"]):
            if ci == t.fragment_slot:
                continue
            path = reactant_resolver(t.reaction_id, ci, comp.get("accepts", []))
            if not path:
                missing = ci
                break
            reactant_files[ci] = path
        entry = {"target": t.__dict__, "reactant_files": reactant_files}
        if missing is not None:
            entry["skipped"] = f"no reactant library for component {missing}"
            _log(f"Skip {t.reaction_id} slot {t.fragment_slot}: {entry['skipped']}")
            runs.append(entry)
            continue
        _log(f"Growing {t.reaction_id} (slot {t.fragment_slot}, core {t.core_smarts})…")
        target_dir = Path(work_dir) / f"{t.reaction_id}_slot{t.fragment_slot}"
        entry["result"] = runner(
            fragment_sdf=fragment_sdf, receptor_path=receptor_pdb,
            reaction_id=t.reaction_id, fragment_slot=t.fragment_slot,
            core_smarts=t.core_smarts, reactant_files=reactant_files,
            work_dir=str(target_dir), **growth_opts)
        runs.append(entry)
    return {"assessment": assessment, "targets": [t.__dict__ for t in targets], "runs": runs}
