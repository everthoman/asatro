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

Handles every step of a linear route (not just the first) and steps with more
than one varying fresh pool (e.g. an unanchored combi start reaction with two
libraries): each varying pool is pruned independently, keeping a reagent iff
*some* combination of the other pools' reagents and reachable input
intermediates lets the step fire into something the next step can consume. The
only concession to cost is bounding: a pool is pruned exactly when the space it
must be checked against (reachable inputs x the other varying pools) is within
``_PARTNER_CAP``, and the whole pass is capped at ``_FIRING_BUDGET`` reaction
firings; beyond either bound a pool/step is simply left unpruned (warm-up still
handles it) -- never a false negative, only occasionally a missed opportunity
on a very large deep step. All three caps are env-overridable.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from asatro.engine.ts_utils import create_reagents

Route = List[Tuple[str, int, Optional[int]]]

# Max distinct intermediates carried forward to the next step (bounds memory);
# once exceeded, downstream steps are left unpruned rather than checked.
_REACHABLE_CAP = int(os.environ.get("ASATRO_REACHABLE_CAP", 20000))
# A pool is only pruned when the partner space it must be checked against
# (reachable inputs x the other varying pools of the same step) is at most this.
_PARTNER_CAP = int(os.environ.get("ASATRO_REACHABLE_PARTNER_CAP", 5000))
# Hard ceiling on total RunReactants calls across the whole pass, so a large
# deep route can't turn the pre-pass into a multi-minute stall. ~0.3ms/firing
# here, so the default is a few minutes of worst case.
_FIRING_BUDGET = int(os.environ.get("ASATRO_REACHABLE_FIRING_BUDGET", 1_500_000))


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


def _fresh_mols(current, chosen):
    """Assemble one step's fresh reagent mols in component (file) order:
    ``chosen`` maps a varying position to the picked reagent; every other
    position takes its single fixed reagent."""
    return [(chosen[p] if p in chosen else current[p][0]).mol
            for p in range(len(current))]


def prune_unreachable_reagents(
    route: Route, files: List[str], *, work_dir: str,
    log: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Return a copy of ``files`` with each non-final step's fresh reagent
    pool(s) narrowed to reagents that can complete the next step.

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
    fired = [0]

    reachable = None  # intermediate mols entering the current step; None at step 0
    for k in range(n):
        _smarts_k, _n_fresh_k, slot_k = route[k]
        rxn_k = compiled[k]
        idxs = step_ranges[k]
        # Mutable per-position reagent lists (varying pools get pruned in place).
        current = [create_reagents(files[i]) for i in idxs]
        varying = [p for p, rl in enumerate(current) if len(rl) > 1]
        is_last = (k == n - 1)

        # The last step has nothing downstream to satisfy; and a step k>0 can't
        # be reasoned about if we lost track of its input intermediates.
        if is_last or (k > 0 and reachable is None):
            if k > 0 and reachable is None and not is_last:
                _emit(f"Reachability: step {k + 1} left unpruned "
                      f"(upstream intermediate set unavailable/too large)")
            reachable = None
            continue

        inputs = reachable if k > 0 else [None]
        next_slot = route[k + 1][2]
        next_template = compiled[k + 1].GetReactantTemplate(
            next_slot if next_slot is not None else 0)

        # -- Prune each varying pool against (inputs x the other varying pools) --
        for vpos in varying:
            others = [p for p in varying if p != vpos]
            partner_count = len(inputs)
            for p in others:
                partner_count *= len(current[p])
            if partner_count > _PARTNER_CAP or fired[0] >= _FIRING_BUDGET:
                _emit(f"Reachability: step {k + 1} pool {Path(files[idxs[vpos]]).name} "
                      f"left unpruned (partner space {partner_count} exceeds cap "
                      f"or firing budget reached)")
                continue

            keep = []
            for r in current[vpos]:
                if r.mol is None:
                    continue
                ok = False
                for interm in inputs:
                    for combo in itertools.product(*[current[p] for p in others]):
                        if fired[0] >= _FIRING_BUDGET:
                            ok = True  # can't confirm dead within budget -> keep (safe)
                            break
                        chosen = {vpos: r}
                        for pos, reagent in zip(others, combo):
                            chosen[pos] = reagent
                        fired[0] += 1
                        prod = _run_step(rxn_k, slot_k, interm, _fresh_mols(current, chosen))
                        if prod is not None and prod.HasSubstructMatch(next_template):
                            ok = True
                            break
                    if ok:
                        break
                if ok:
                    keep.append(r)

            orig = len(current[vpos])
            if not keep:
                raise UnreachableRouteError(
                    f"no reagent in {Path(files[idxs[vpos]]).name} can react at the "
                    f"downstream step {k + 2} -- check the route wiring / handle")
            if len(keep) < orig:
                current[vpos] = keep
                out = work / f"reachable_step{k + 1}_{Path(files[idxs[vpos]]).stem}.smi"
                with open(out, "w") as fh:
                    for r in keep:
                        fh.write(f"{r.smiles} {r.reagent_name}\n")
                files[idxs[vpos]] = str(out)
                _emit(f"Reachability: step {k + 1} pool {Path(out).name} "
                      f"{orig} -> {len(keep)} reagents that can complete step {k + 2}")

        # -- Carry the surviving intermediates forward for the next step. --
        reachable = _reachable_products(rxn_k, slot_k, inputs, current, varying, fired)

    return files


def _reachable_products(rxn, slot, inputs, current, varying, fired):
    """Distinct step products over (``inputs`` x the varying pools' survivors),
    deduped by canonical SMILES. Returns ``None`` (meaning "unknown -- don't
    prune downstream") if the enumeration would exceed the reachable/firing
    caps, so a later step is never pruned against a partial input set."""
    space = len(inputs)
    for p in varying:
        space *= len(current[p])
    if space > _REACHABLE_CAP:
        return None
    out = []
    seen: set = set()
    for interm in inputs:
        for combo in itertools.product(*[current[p] for p in varying]):
            if fired[0] >= _FIRING_BUDGET:
                return None
            chosen = {pos: reagent for pos, reagent in zip(varying, combo)}
            fired[0] += 1
            prod = _run_step(rxn, slot, interm, _fresh_mols(current, chosen))
            if prod is None:
                continue
            smi = Chem.MolToSmiles(prod)
            if smi not in seen:
                seen.add(smi)
                out.append(prod)
                if len(out) >= _REACHABLE_CAP:
                    return None
    return out or None
