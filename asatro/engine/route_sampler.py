"""
Multi-step (linear route) Thompson Sampling.

``RouteSampler`` generalises :class:`thompson_sampling.ThompsonSampler` from a
single reaction to an ordered sequence of reactions that build one final
product:

    step 0:  R0(reagent_a, reagent_b, ...)        -> intermediate_0
    step 1:  R1(intermediate_0, reagent_c)        -> intermediate_1
    step k:  Rk(intermediate_{k-1}, reagent_...)  -> intermediate_k
    ...
    final product = intermediate_last   (this is what gets scored)

Only the *final* product is passed to the evaluator, matching the requested
behaviour ("reactions applied sequentially and TS only applied to final
products").

The reagent components that Thompson Sampling samples over are the flat list of
"new reagent" inputs across all steps, in route order. The running intermediate
is threaded automatically and is never a sampled component. Everything else
(warm-up, search, the disallow tracker, the reagent priors) is inherited from
``ThompsonSampler`` unchanged. With a single step this reduces exactly to the
original single-reaction behaviour.
"""

from typing import List, Optional, Tuple, Union

from rdkit import Chem
from rdkit.Chem import AllChem

from asatro.engine.thompson_sampling import ThompsonSampler
from asatro.engine.ts_autoparams import resolve_rws_budget, resolve_ts_budget


class RouteSampler(ThompsonSampler):
    def __init__(self, mode="maximize", log_filename=None):
        super().__init__(mode=mode, log_filename=log_filename)
        # List of (compiled_reaction, num_new_reagents, intermediate_slot) in
        # route order. intermediate_slot is None for the first step (no
        # intermediate yet); for later steps it's which position in the full
        # reactant list the running intermediate occupies -- 0 for every
        # hand-authored "extend" reaction (they're all written with the
        # intermediate-matching pattern first), any valid position for a
        # reaction reused generically at one of its own slots.
        self.route_steps: List[Tuple[AllChem.ChemicalReaction, int, Optional[int]]] = []

    def set_route(self, steps: List[Union[Tuple[str, int], Tuple[str, int, Optional[int]]]]) -> None:
        """
        Define the reaction sequence.

        :param steps: list of ``(reaction_smarts, num_new_reagents)`` or
            ``(reaction_smarts, num_new_reagents, intermediate_slot)`` tuples
            in route order. ``num_new_reagents`` is the number of sampled
            reagent components the step consumes *in addition* to the running
            intermediate. The first step takes no intermediate, so the sum of
            ``num_new_reagents`` across all steps must equal the number of
            reagent components (``len(self.reagent_lists)``). The 2-tuple
            form (legacy) implies ``intermediate_slot=None`` -- position 0
            once an intermediate exists.
        """
        self.route_steps = [
            (AllChem.ReactionFromSmarts(step[0]), int(step[1]),
             (int(step[2]) if len(step) > 2 and step[2] is not None else None))
            for step in steps
        ]

    def set_reaction(self, rxn_smarts):
        """Convenience: a single-step route equivalent to the base class."""
        self.set_route([(rxn_smarts, len(self.reagent_lists) or 1)])

    def _expected_reagent_count(self) -> int:
        return sum(n for _rxn, n, *_ in self.route_steps)

    def _build_product(self, choice_list: List[int]):
        """
        Build the final product by running the reaction sequence.

        Overrides the single-reaction base method so that all of the base
        sampler's machinery (sequential ``evaluate`` and parallel
        ``evaluate_batch``, warm-up and search) drives the multi-step route
        unchanged. Pure / no shared state, so it is safe to call from worker
        threads.

        :param choice_list: list of reagent indices, one per reagent component,
            ordered to match the flat ``reagent_lists`` (route order).
        :return: ``(product_mol_or_None, smiles, product_name, selected_reagents)``.
        """
        selected_reagents = [
            self.reagent_lists[idx][choice] for idx, choice in enumerate(choice_list)
        ]
        product_name = self._product_name(selected_reagents)
        try:
            cursor = 0
            intermediate = None
            for rxn, n_new, intermediate_slot in self.route_steps:
                if intermediate is None:
                    reactants = [selected_reagents[cursor + k].mol for k in range(n_new)]
                    cursor += n_new
                else:
                    slot = intermediate_slot if intermediate_slot is not None else 0
                    reactants = [None] * (n_new + 1)
                    reactants[slot] = intermediate
                    for p in range(n_new + 1):
                        if p == slot:
                            continue
                        reactants[p] = selected_reagents[cursor].mol
                        cursor += 1
                products = rxn.RunReactants(reactants)
                if not products:
                    return None, "FAIL", product_name, selected_reagents
                intermediate = products[0][0]  # Tuple[Tuple[Mol]]
                Chem.SanitizeMol(intermediate)
            product_smiles = Chem.MolToSmiles(intermediate)
        except Exception:
            # Any RDKit failure in the route -> treat as a failed product (NaN).
            return None, "FAIL", product_name, selected_reagents
        return intermediate, product_smiles, product_name, selected_reagents


def run_ts_or_rws_search(sampler: ThompsonSampler, evaluator, search_method: str,
                         num_warmup: Optional[int], num_cycles: Optional[int],
                         min_cpds_per_core: Optional[int] = None,
                         stop: Optional[int] = None) -> list:
    """Auto-tune the TS/RWS budget from ``sampler``'s (pruned) pool sizes for
    any of ``num_warmup``/``num_cycles``/``min_cpds_per_core``/``stop`` left
    unset, then run the warm-up + search dispatch for ``search_method``
    (``"ts"`` or ``"rws"``). Shared by ``growth.run_growth`` and
    ``combi.run_combi``, which otherwise duplicated this dispatch verbatim.

    Returns the finite ``[score, smiles, name]`` result rows, or ``[]`` if
    nothing scored during warm-up (mirrors both callers' original bail-out)."""
    num_warmup, num_cycles = resolve_ts_budget(
        num_warmup, num_cycles, sampler.reagent_lists, log=evaluator.progress_callback)
    if search_method == "rws":
        min_cpds_per_core, stop = resolve_rws_budget(
            min_cpds_per_core, stop, num_cycles, log=evaluator.progress_callback)
        warmup_results = sampler.warm_up_rws(num_warmup_trials=num_warmup)
        if not warmup_results:
            # search_rws needs the per-reagent posteriors warm_up_rws seeds;
            # nothing scored means those were never initialized, and searching
            # further would just repeat the same all-nan warm-up. Bail out
            # cleanly instead of the AttributeError search_rws would raise.
            return []
        search_results = sampler.search_rws(
            num_targets=num_cycles, min_cpds_per_core=min_cpds_per_core, stop=stop)
        return warmup_results + search_results
    warmup_results = sampler.warm_up(num_warmup_trials=num_warmup)
    if not warmup_results:
        # search() draws from per-reagent priors warm_up() seeds; nothing
        # scored means those were never initialized (every reagent is still
        # in its uninitialized "warmup" phase), so searching further would
        # sample meaningless all-zero priors. Bail out cleanly, mirroring
        # the RWS branch's guard above.
        return []
    return sampler.search(num_cycles=num_cycles)
