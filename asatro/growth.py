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

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from asatro.chemistry.catalog import REACTION_BY_ID
from asatro.engine.anchored_fragment_evaluator import AnchoredFragmentEvaluator
from asatro.engine.gnina_evaluator import MolFilters
from asatro.engine.route_sampler import RouteSampler


def fragment_smiles_from_sdf(fragment_sdf: str) -> str:
    mol = Chem.MolFromMolFile(fragment_sdf)
    if mol is None:
        raise ValueError(f"could not read fragment SDF: {fragment_sdf}")
    return Chem.MolToSmiles(mol)


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
               hide_progress: bool = True, **gnina_opts):
    """Run the full growth search. Returns ``(results, evaluator)`` where results
    is the list of ``[score, smiles, name]`` rows the sampler collected."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    files, route = build_growth_route(
        reaction_id, fragment_smiles_from_sdf(fragment_sdf), fragment_slot,
        reactant_files, work)

    evaluator = make_evaluator(fragment_sdf=fragment_sdf, receptor_path=receptor_path,
                               core_smarts=core_smarts, work_dir=str(work / "dock"),
                               **gnina_opts)
    sampler = RouteSampler(mode=mode)
    if seed is not None:
        sampler.set_seed(seed)
    sampler.set_concurrency(concurrency)
    sampler.set_hide_progress(hide_progress)
    sampler.read_reagents(reagent_file_list=files, num_to_select=num_to_select)
    sampler.set_route(route)
    sampler.set_evaluator(evaluator)
    sampler.warm_up(num_warmup_trials=num_warmup)
    results = sampler.search(num_cycles=num_cycles)
    return results, evaluator
