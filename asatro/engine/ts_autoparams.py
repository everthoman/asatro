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

# RWS-only knobs (min_cpds_per_core, stop) have no pool-size-derived literature
# default the way num_warmup/num_cycles do, so these are scaled off the already
# -suggested num_cycles instead: min_cpds_per_core so the search does several
# batches (the thermal-cycling temperature adaptation needs more than one to
# act on) rather than one giant blob; stop (a consecutive-resample early-stop,
# in the same per-draw units as num_cycles) proportionally to num_cycles so a
# bigger search gets proportionally more patience before bailing as "library
# effectively exhausted" than a small one. Deliberately never goes below the
# old fixed defaults' neighborhood (MIN_BATCH/MIN_STOP) so this can't make a
# search bail out *earlier* than the previous hardcoded 50/6000 did.
RWS_TARGET_BATCHES = 20
MIN_BATCH = 10
MAX_BATCH = 200
STOP_MULTIPLIER = 1.0
MIN_STOP = 2000
MAX_STOP = 20000


def suggest_ts_params(variable_pool_sizes: Sequence[int]) -> Tuple[int, int]:
    """Suggested ``(num_warmup, num_cycles)`` from the variable slot sizes
    (the reagent pools with more than one entry, after pruning)."""
    sizes: List[int] = [s for s in variable_pool_sizes if s > 1]
    if not sizes:
        return DEFAULT_WARMUP, MIN_CYCLES
    largest = max(sizes)
    num_cycles = max(MIN_CYCLES, min(MAX_CYCLES, CYCLES_PER_REAGENT * largest))
    return DEFAULT_WARMUP, num_cycles


def suggest_rws_params(num_cycles: int) -> Tuple[int, int]:
    """Suggested ``(min_cpds_per_core, stop)`` for an RWS search, scaled off
    the already-suggested ``num_cycles`` (see :func:`suggest_ts_params`) --
    see the module-level comment above for the reasoning."""
    min_cpds_per_core = max(MIN_BATCH, min(MAX_BATCH, round(num_cycles / RWS_TARGET_BATCHES)))
    stop = max(MIN_STOP, min(MAX_STOP, round(STOP_MULTIPLIER * num_cycles)))
    return min_cpds_per_core, stop


def resolve_rws_budget(min_cpds_per_core, stop, num_cycles, log=None):
    """Fill in ``min_cpds_per_core``/``stop`` from :func:`suggest_rws_params`
    when either is ``None`` (i.e. the user didn't set it), leaving explicit
    values untouched. Mirrors :func:`resolve_ts_budget`; only meaningful for
    an RWS search, so callers should only invoke this when
    ``search_method == "rws"``.
    """
    sm, ss = suggest_rws_params(num_cycles)
    auto_m = min_cpds_per_core is None
    auto_s = stop is None
    min_cpds_per_core = sm if auto_m else int(min_cpds_per_core)
    stop = ss if auto_s else int(stop)
    if log is not None:
        log(f"RWS budget: min_cpds_per_core={min_cpds_per_core}{' (auto)' if auto_m else ''}, "
            f"stop={stop}{' (auto)' if auto_s else ''}")
    return min_cpds_per_core, stop


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
            f"{variable_sizes}; ~{(warmup_docks + num_cycles) / 1000:.1f}k products "
            f"evaluated (incl. warm-up, before filters/dedup)")
    return num_warmup, num_cycles
