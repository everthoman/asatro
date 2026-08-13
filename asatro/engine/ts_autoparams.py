"""Heuristic Thompson-Sampling budget suggestions from post-prune pool sizes.

Once reachability pruning (``asatro/chemistry/reachability.py``) leaves each
variable slot dense (~100% build rate), sensible search-budget defaults follow
directly from the slot sizes, so a user need not guess ``num_warmup`` /
``num_cycles``. These are heuristics, not proven optima -- TS tuning is
empirical -- and are only auto-applied when the user did not set a value.
"""
from typing import List, Sequence, Tuple

# num_cycles ~ CYCLES_PER_REAGENT x the largest variable pool, so each strong
# reagent gets several TS exploit draws in the search phase; clamped to a sane
# range so a tiny library isn't over-docked nor a huge one run unbounded.
CYCLES_PER_REAGENT = 2
MIN_CYCLES = 500
MAX_CYCLES = 10000
# Pruning keeps pools dense, so the sparse-pool warm-up inflation (which existed
# only to stop good reagents being retired for unlucky random partners) is
# unnecessary -- the literature-standard 3 is right again.
DEFAULT_WARMUP = 3


def suggest_ts_params(variable_pool_sizes: Sequence[int]) -> Tuple[int, int]:
    """Suggested ``(num_warmup, num_cycles)`` from the variable slot sizes
    (the reagent pools with more than one entry, after pruning)."""
    sizes: List[int] = [s for s in variable_pool_sizes if s > 1]
    if not sizes:
        return DEFAULT_WARMUP, MIN_CYCLES
    largest = max(sizes)
    num_cycles = max(MIN_CYCLES, min(MAX_CYCLES, CYCLES_PER_REAGENT * largest))
    return DEFAULT_WARMUP, num_cycles


def resolve_ts_budget(num_warmup, num_cycles, reagent_lists, log=None):
    """Fill in ``num_warmup``/``num_cycles`` from :func:`suggest_ts_params` when
    either is ``None`` (i.e. the user didn't set it), leaving explicit values
    untouched, and log the resulting budget + rough dock estimate.

    Returns the resolved ``(num_warmup, num_cycles)`` as ints.
    """
    variable_sizes = [len(rl) for rl in reagent_lists if len(rl) > 1]
    sw, sc = suggest_ts_params(variable_sizes)
    auto_w = num_warmup is None
    auto_c = num_cycles is None
    num_warmup = sw if auto_w else int(num_warmup)
    num_cycles = sc if auto_c else int(num_cycles)
    if log is not None:
        # warm-up docks every reagent (all slots) num_warmup times.
        warmup_docks = num_warmup * sum(len(rl) for rl in reagent_lists)
        log(f"TS budget: num_warmup={num_warmup}{' (auto)' if auto_w else ''}, "
            f"num_cycles={num_cycles}{' (auto)' if auto_c else ''} — variable slots "
            f"{variable_sizes}; ~{(warmup_docks + num_cycles) / 1000:.1f}k docks "
            f"incl. warm-up, before filters")
    return num_warmup, num_cycles
