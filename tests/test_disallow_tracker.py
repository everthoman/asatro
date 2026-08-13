"""DisallowTracker retirement semantics + the memory property behind them.

Regression coverage for a confirmed production OOM: retiring a synthon used to
enumerate every pairing of that synthon with the other cycles' synthons into
``_disallow_mask`` (O(product of other cycles' sizes) dict entries per
retirement). On a real two-large-slot growth route that retired ~10k of ~10.5k
synthons in one slot, each against ~11.8k in the other, this reached ~10^8
entries / tens of GB. Retirement is now O(1) per synthon via a per-cycle set.
"""
import numpy as np

from asatro.engine.disallow_tracker import DisallowTracker


def _mask_index_of_to_fill(n, cycle):
    sel = [DisallowTracker.Empty] * n
    sel[cycle] = DisallowTracker.To_Fill
    return sel


def test_retired_synthon_is_disallowed_for_its_cycle():
    dt = DisallowTracker([4, 5])
    dt.retire_one_synthon(0, 2)
    disallowed = dt.get_disallowed_selection_mask(_mask_index_of_to_fill(2, 0))
    assert 2 in disallowed
    # ...but not for a different synthon in the same cycle
    assert 1 not in disallowed


def test_retirement_does_not_touch_other_cycles():
    dt = DisallowTracker([4, 5])
    dt.retire_one_synthon(0, 2)
    # Retiring synthon 2 of cycle 0 says nothing about which synthons of cycle 1
    # are allowed -- cycle 1 is still fully open.
    disallowed_c1 = dt.get_disallowed_selection_mask(_mask_index_of_to_fill(2, 1))
    assert disallowed_c1 == set()


def test_retirement_is_cheap_and_does_not_explode_the_disallow_mask():
    # The heart of the OOM regression: retiring many synthons in one large slot
    # must NOT scale the stored _disallow_mask by the size of the other slot.
    dt = DisallowTracker([2000, 3000])
    for j in range(1900):  # retire almost all of cycle 0
        dt.retire_one_synthon(0, j)
    # Old behaviour: ~1900 * 3000 ≈ 5.7M entries. New: retirements live in the
    # per-cycle _retired set, and nothing has been *sampled* yet, so the
    # combination mask stays empty.
    assert len(dt._disallow_mask) == 0
    assert len(dt._retired[0]) == 1900


def test_search_style_selection_never_picks_a_retired_synthon():
    dt = DisallowTracker([6, 4])
    for j in (0, 2, 4):
        dt.retire_one_synthon(0, j)
    rng = np.random.default_rng(0)
    for _ in range(200):
        sel = _mask_index_of_to_fill(2, 0)
        disallowed = dt.get_disallowed_selection_mask(sel)
        scores = rng.uniform(size=6)
        if disallowed:
            scores[list(disallowed)] = np.nan
        pick = int(np.nanargmax(scores))
        assert pick in (1, 3, 5), f"picked retired synthon {pick}"


def test_sample_honours_retirement():
    # DisallowTracker.sample() (unused in asatro's own search path, but part of
    # the class's contract) must also avoid retired synthons.
    dt = DisallowTracker([3, 3])
    dt.retire_one_synthon(0, 0)
    dt.retire_one_synthon(0, 1)
    for _ in range(20):
        # cycle 0 can now only ever be synthon 2
        pre = len(dt._disallow_mask)
        sel = dt.sample()
        assert sel[0] == 2
        # sanity: sampling still records the *combination* it drew
        assert len(dt._disallow_mask) >= pre
        if dt._n_sampled >= 3:  # only 3 combos exist (2 fixed by retirement)
            break
