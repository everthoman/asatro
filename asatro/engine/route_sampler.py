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
        product_name = "_".join(r.reagent_name for r in selected_reagents)
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
