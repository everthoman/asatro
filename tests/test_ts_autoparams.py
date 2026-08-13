"""Auto-suggested Thompson-Sampling budget (asatro/engine/ts_autoparams.py)."""
from asatro.engine.ts_autoparams import (
    DEFAULT_WARMUP,
    MAX_CYCLES,
    MIN_CYCLES,
    resolve_ts_budget,
    suggest_ts_params,
)


class _RL(list):
    """A reagent list whose length is all resolve_ts_budget/suggest care about."""


def _pools(*sizes):
    return [_RL(range(s)) for s in sizes]


def test_suggest_scales_with_largest_pool():
    w, c = suggest_ts_params([1437, 1788])
    assert w == DEFAULT_WARMUP
    assert c == max(MIN_CYCLES, min(MAX_CYCLES, 2 * 1788))  # 3576


def test_suggest_clamps_small_and_large():
    assert suggest_ts_params([5])[1] == MIN_CYCLES          # tiny -> floor
    assert suggest_ts_params([100000])[1] == MAX_CYCLES     # huge -> ceiling


def test_suggest_ignores_fixed_slots():
    # A 1-entry slot (e.g. the bound fragment) is not a variable pool.
    w, c = suggest_ts_params([1, 1437, 1])
    assert (w, c) == suggest_ts_params([1437])


def test_resolve_fills_only_the_unset_values():
    reagent_lists = _pools(1437, 1, 1788)  # includes a fixed fragment slot
    # both unset -> both auto
    w, c = resolve_ts_budget(None, None, reagent_lists)
    assert w == DEFAULT_WARMUP
    assert c == 2 * 1788
    # explicit values are left untouched
    assert resolve_ts_budget(7, 50, reagent_lists) == (7, 50)
    # mixed: warm-up explicit, cycles auto
    w, c = resolve_ts_budget(5, None, reagent_lists)
    assert w == 5 and c == 2 * 1788


def test_resolve_logs_budget_and_marks_auto():
    logs = []
    resolve_ts_budget(None, 123, _pools(1437, 1788), log=logs.append)
    assert len(logs) == 1
    msg = logs[0]
    assert "num_warmup=3 (auto)" in msg
    assert "num_cycles=123" in msg and "num_cycles=123 (auto)" not in msg
    assert "[1437, 1788]" in msg
