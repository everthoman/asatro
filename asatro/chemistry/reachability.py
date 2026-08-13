"""Downstream-reachability pre-filtering of route reagents.

A multi-step growth/combi route only yields a buildable product when the
running intermediate can actually react at every subsequent step. asatro's
reagent resolver prunes each component only by its own functional-group class,
with no awareness of what later steps need -- so e.g. a ``schotten_baumann_amide
-> suzuki`` growth enumerates all ~10.5k carboxylic acids even though only the
~1.4k that also carry an aryl-halide handle can ever complete the Suzuki step.
The dead reagents are pure waste: they get mass-retired in warm-up (the O(n^2)
retirement that once OOM'd the box) and make a run look like it barely docked
anything.

This pass removes, before the search starts, any fresh reagent whose step
product cannot serve as the next step's intermediate-slot reactant. It is an
**exact** prune with no false negatives: RDKit ``RunReactants`` fires iff each
reactant matches its template, so a reagent whose product fails the next step's
intermediate template could never have completed the route.

Scope (v1): only prunes a step that has exactly one *varying* fresh pool (a
reagent file with >1 entry) given the fixed inputs so far -- which covers the
whole common growth family (a 2-component start reaction with the fragment
fixing one slot, plus single-reagent extend steps). Steps with two varying
pools, and downstream steps once the reachable-intermediate set grows past a
cap, are left untouched (warm-up still handles them). Everything here only ever
*removes* provably-dead reagents, so leaving a step unpruned is always safe.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from asatro.engine.ts_utils import create_reagents

Route = List[Tuple[str, int, Optional[int]]]

# Max distinct intermediates carried forward between steps (bounds memory/time).
_REACHABLE_CAP = 5000
# A step k>0 is only pruned when the set of reachable input intermediates is at
# most this large, so the per-reagent cross-check stays cheap. Step 0 has a
# single (or no) fixed input, so it is always pruned regardless.
_CROSS_CAP = 500


class UnreachableRouteError(ValueError):
    """Raised when reachability pruning empties a reagent pool -- no reagent in
    it can react at the downstream step, so the route as wired can never build
    a product. Surfaced to the user instead of silently docking nothing."""


def _run_step(rxn, intermediate_slot: Optional[int], intermediate, fresh_mols):
    """Fire one route step; return the sanitized product mol or ``None`` if it
    doesn't fire / can't sanitize.

    Mirrors ``RouteSampler._build_product``'s reactant placement
    (``route_sampler.py``): with no intermediate (first step) the fresh
    reagents fill the reactant array in component order; otherwise the
    intermediate binds ``intermediate_slot`` and the fresh reagents fill the
    remaining positions in order."""
    if intermediate is None:
        reactants = list(fresh_mols)
    else:
        slot = intermediate_slot if intermediate_slot is not None else 0
        reactants = [None] * (len(fresh_mols) + 1)
        reactants[slot] = intermediate
        it = iter(fresh_mols)
        for p in range(len(reactants)):
            if p != slot:
                reactants[p] = next(it)
    try:
        products = rxn.RunReactants(reactants)
        if not products:
            return None
        prod = products[0][0]
        Chem.SanitizeMol(prod)
        return prod
    except Exception:  # noqa: BLE001 -- any RDKit failure = this step didn't build
        return None


def prune_unreachable_reagents(
    route: Route, files: List[str], *, work_dir: str,
    log: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Return a copy of ``files`` with each prunable non-final step's varying
    reagent pool narrowed to reagents that can complete the next step.

    ``route`` and ``files`` are exactly what ``build_growth_route`` /
    ``build_combi_route`` return: ``route`` is ``[(smarts, n_fresh,
    intermediate_slot), ...]`` and ``files`` is the flat, route-ordered list of
    fresh reagent ``.smi`` paths (``sum(n_fresh) == len(files)``). Pruned pools
    are written as new ``.smi`` files under ``work_dir``.

    Raises :class:`UnreachableRouteError` if a prune empties a pool.
    """
    files = list(files)
    n = len(route)
    if n < 2:
        return files  # single-step route: nothing downstream to satisfy

    compiled = [AllChem.ReactionFromSmarts(smarts) for smarts, _, _ in route]
    # Flat files -> the index range belonging to each step's fresh components.
    step_ranges: List[List[int]] = []
    cursor = 0
    for _smarts, n_fresh, _slot in route:
        step_ranges.append(list(range(cursor, cursor + n_fresh)))
        cursor += n_fresh

    def _emit(msg: str) -> None:
        if log is not None:
            log(msg)

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    reachable = None  # intermediate mols entering the current step; None at step 0
    for k in range(n):
        _smarts_k, _n_fresh_k, slot_k = route[k]
        rxn_k = compiled[k]
        idxs = step_ranges[k]
        per_file = [create_reagents(files[i]) for i in idxs]
        varying = [p for p, rl in enumerate(per_file) if len(rl) > 1]
        is_last = (k == n - 1)

        if k == 0:
            inputs = [None]
            can_cross = True
        else:
            can_cross = reachable is not None and len(reachable) <= _CROSS_CAP
            inputs = reachable if reachable is not None else []

        if is_last or len(varying) != 1 or not can_cross:
            # Not prunable (last step, ambiguous, or reachable set too large/lost).
            # Drop the reachable set so no *downstream* step is pruned unsafely.
            reachable = None
            continue

        vpos = varying[0]
        varying_file_idx = idxs[vpos]
        next_slot = route[k + 1][2]
        next_template = compiled[k + 1].GetReactantTemplate(
            next_slot if next_slot is not None else 0)

        survivors = []
        products_after = []
        seen: set = set()
        for r in per_file[vpos]:
            if r.mol is None:
                continue
            fresh_mols = [r.mol if p == vpos else per_file[p][0].mol
                          for p in range(len(per_file))]
            matched = []
            for interm in inputs:
                prod = _run_step(rxn_k, slot_k, interm, fresh_mols)
                if prod is not None and prod.HasSubstructMatch(next_template):
                    matched.append(prod)
            if matched:
                survivors.append(r)
                for prod in matched:
                    smi = Chem.MolToSmiles(prod)
                    if smi not in seen:
                        seen.add(smi)
                        if len(products_after) < _REACHABLE_CAP:
                            products_after.append(prod)

        pool_name = Path(files[varying_file_idx]).name
        if not survivors:
            raise UnreachableRouteError(
                f"no reagent in {pool_name} can react at the downstream step "
                f"{k + 2} -- check the route wiring / handle")

        out = work / f"reachable_step{k + 1}_{Path(files[varying_file_idx]).stem}.smi"
        with open(out, "w") as fh:
            for r in survivors:
                fh.write(f"{r.smiles} {r.reagent_name}\n")
        files[varying_file_idx] = str(out)
        _emit(f"Reachability: step {k + 1} pool {pool_name} "
              f"{len(per_file[vpos])} -> {len(survivors)} reagents that can complete "
              f"step {k + 2}")

        # Carry surviving products forward as the next step's reachable inputs.
        # Exact when a single fixed input produced one product per survivor
        # (always true at step 0); dropped to None if we hit the dedup cap.
        reachable = products_after if 0 < len(products_after) < _REACHABLE_CAP else None

    return files
