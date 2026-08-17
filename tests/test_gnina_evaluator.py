"""Protecting-group stripping (deprotect_smiles) and its wiring into
GninaEvaluator.evaluate_detailed. Ported from ts-gnina, which found real
Boc/Cbz/Fmoc-protected building blocks in commercial pools reacting at one
handle while leaving the other's protecting group on the docked/reported
product.
"""
import subprocess
import sys
import threading
import time

from rdkit import Chem

from asatro.combi import make_evaluator
from asatro.engine.gnina_evaluator import DockingCancelled, deprotect_smiles


def _canon(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def test_deprotect_boc_amine():
    assert deprotect_smiles("CC(C)(C)OC(=O)N1CCNCC1") == _canon("C1CNCCN1")


def test_deprotect_cbz_amine():
    assert deprotect_smiles("O=C(OCc1ccccc1)NCc1ccccc1") == _canon("NCc1ccccc1")


def test_deprotect_fmoc_amine():
    assert deprotect_smiles("O=C(OCC1c2ccccc2-c2ccccc21)NCC(=O)O") == _canon("NCC(=O)O")


def test_deprotect_tbu_ester():
    assert deprotect_smiles("CC(=O)OC(C)(C)C") == _canon("CC(=O)O")


def test_deprotect_bn_ester():
    assert deprotect_smiles("CC(=O)OCc1ccccc1") == _canon("CC(=O)O")


def test_deprotect_bpin_boronate():
    assert deprotect_smiles("B1(c2ccccc2)OC(C)(C)C(C)(C)O1") == _canon("OB(O)c1ccccc1")
    assert deprotect_smiles("B1(c2ccncc2)OC(C)(C)C(C)(C)O1") == _canon("OB(O)c1ccncc1")


def test_deprotect_boc_and_tbu_ester_together():
    # Boc-protected amino acid tert-butyl ester -> free amino acid
    assert deprotect_smiles("CC(NC(=O)OC(C)(C)C)C(=O)OC(C)(C)C") == _canon("CC(N)C(=O)O")


def test_ester_rules_do_not_touch_carbamates():
    # The tBu/Bn "ester" rules require a carbon neighbour on the carbonyl, so
    # they must not fire on a carbamate (N-C(=O)-O-) that the Boc/Cbz rules
    # already handle -- regression guard for that ordering.
    assert deprotect_smiles("CC(C)(C)OC(=O)NC") == _canon("CN")
    assert deprotect_smiles("O=C(OCc1ccccc1)NC") == _canon("CN")


def test_deprotect_leaves_unprotected_molecules_unchanged():
    assert deprotect_smiles("c1ccccc1") == _canon("c1ccccc1")


def test_deprotect_bad_smiles_passthrough():
    assert deprotect_smiles("not a smiles") == "not a smiles"


def test_evaluator_caches_by_deprotected_smiles(tmp_path):
    """A Boc-protected product's score/reason/name caches are keyed by the
    deprotected SMILES, and the (mocked) dock is invoked with the free form
    -- not the as-built reagent-combo SMILES."""
    rec = tmp_path / "receptor.pdb"
    rec.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n")
    ev = make_evaluator(receptor_path=str(rec), center=(0.0, 0.0, 0.0),
                        work_dir=str(tmp_path / "dock"))

    docked_with = []

    def fake_dock(smiles):
        docked_with.append(smiles)
        return -5.0
    ev._dock = fake_dock

    protected = "CC(C)(C)OC(=O)NCc1ccccc1"  # Boc-protected benzylamine
    free = _canon("NCc1ccccc1")
    mol = Chem.MolFromSmiles(protected)
    mol.SetProp("_Name", "reagentA_reagentB")

    score, reason = ev.evaluate_detailed(mol)
    assert score == -5.0 and reason is None
    assert docked_with == [free]
    assert ev._score_cache == {free: -5.0}
    assert ev._name_cache == {free: "reagentA_reagentB"}
    assert protected not in ev._score_cache

    # Re-evaluating the same (still-protected) product hits the cache and
    # does not dock again.
    mol2 = Chem.MolFromSmiles(protected)
    score2, reason2 = ev.evaluate_detailed(mol2)
    assert score2 == -5.0 and reason2 is None


def _make_ev(tmp_path, **extra):
    rec = tmp_path / "receptor.pdb"
    rec.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n")
    return make_evaluator(receptor_path=str(rec), center=(0.0, 0.0, 0.0),
                          work_dir=str(tmp_path / "dock"), **extra)


def test_cuda_visible_devices_hidden_when_cnn_scoring_off(tmp_path):
    # cnn_scoring="none" -> pure CPU Vina, no GPU should ever be exposed,
    # regardless of gpu_id/gpu_ids being set.
    ev = _make_ev(tmp_path, cnn_scoring="none", gpu_ids=[2, 5])
    assert ev._next_cuda_visible_devices() == ""


def test_cuda_visible_devices_single_gpu_fallback(tmp_path):
    # No gpu_ids given -> falls back to the single gpu_id (default 0).
    ev = _make_ev(tmp_path, cnn_scoring="rescore", gpu_id=3)
    assert ev._next_cuda_visible_devices() == "3"
    assert ev._next_cuda_visible_devices() == "3"


def test_cuda_visible_devices_round_robins_across_gpu_ids(tmp_path):
    # Concurrent CNN-scoring docks must not all pile onto the same GPU --
    # this is what triggered a real hang under GPU contention (concurrency=4,
    # cnn_scoring="rescore", all pinned to gpu_id=0 by default).
    ev = _make_ev(tmp_path, cnn_scoring="rescore", gpu_ids=[2, 5])
    seen = [ev._next_cuda_visible_devices() for _ in range(5)]
    assert seen == ["2", "5", "2", "5", "2"]


def test_run_pollable_completes_normally(tmp_path):
    ev = _make_ev(tmp_path, cnn_scoring="none")
    proc = ev._run_pollable(["echo", "hi"], env={}, timeout=5)
    assert proc.returncode == 0
    assert "hi" in proc.stdout


def test_run_pollable_honors_cancel_event_mid_flight(tmp_path):
    # Regression test for the real production hang: a batch of concurrent
    # docks used to be uninterruptible once launched -- cancel_event was only
    # checked before a dock started, so a slow/hung dock blocked the whole
    # job for up to `timeout` (600s default) no matter what. _run_pollable
    # must notice cancel_event *while a dock is running* and kill it well
    # before either the process finishes on its own or the timeout fires.
    ev = _make_ev(tmp_path, cnn_scoring="none", timeout=30)
    ev.cancel_event = threading.Event()

    def cancel_soon():
        time.sleep(0.3)
        ev.cancel_event.set()
    threading.Thread(target=cancel_soon, daemon=True).start()

    start = time.monotonic()
    try:
        ev._run_pollable(["sleep", "10"], env={}, timeout=30)
        assert False, "expected DockingCancelled"
    except DockingCancelled:
        pass
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"cancellation took {elapsed:.2f}s -- should be ~0.3-0.5s, not blocked until sleep/timeout"


def test_run_pollable_bounds_stdout_from_a_high_volume_child(tmp_path):
    # Regression test for a real production OOM (2026-08-12): Popen(stdout=PIPE)
    # + repeated communicate(timeout=poll) retries used to accumulate *all* of a
    # slow child's output in this process's own memory across every retry,
    # unboundedly, for as long as it ran (up to `timeout`, 600s default) --
    # confirmed live to hit ~2.9GB parent RSS in 8s against a synthetic runaway
    # child, and this is what actually OOM-killed asatro-webapp in production
    # (anon-rss 58.9GB) on a job stuck for 16 minutes. stdout/stderr are now
    # captured to on-disk tempfiles with only the tail kept, so the returned
    # CompletedProcess can never reflect more than a bounded slice regardless
    # of how much the child actually produced.
    ev = _make_ev(tmp_path, cnn_scoring="none")
    cmd = [sys.executable, "-c", "print('y' * 300000)"]
    proc = ev._run_pollable(cmd, env={}, timeout=5)
    assert proc.returncode == 0
    assert len(proc.stdout) <= 65536 + 16  # _tail_text's max_bytes + a little slack


def test_run_pollable_raises_timeout_expired_without_waiting_full_sleep(tmp_path):
    ev = _make_ev(tmp_path, cnn_scoring="none")
    start = time.monotonic()
    try:
        ev._run_pollable(["sleep", "10"], env={}, timeout=1)
        assert False, "expected TimeoutExpired"
    except subprocess.TimeoutExpired:
        pass
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"took {elapsed:.2f}s -- should stop at ~1s (the configured timeout), not the full 10s sleep"


def test_concurrent_duplicate_products_dock_once(tmp_path):
    # Single-flight: at concurrency>1 many samples collapse to the same product
    # (protected variants deprotect to identical structures). Without an
    # in-flight guard, every worker re-docks it -- inflating dock_count well
    # above the unique-product count. Concurrent identical products must dock
    # exactly once, and all callers get that one result.
    ev = _make_ev(tmp_path, cnn_scoring="none")
    ev.filters = None
    calls = []
    lock = threading.Lock()

    def fake_dock(smiles):
        with lock:
            calls.append(smiles)
        time.sleep(0.25)   # hold the in-flight slot so the others pile up behind it
        return -7.0
    ev._dock = fake_dock

    results = []

    def run():
        m = Chem.MolFromSmiles("c1ccccc1")
        results.append(ev.evaluate_detailed(m))
    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1                    # docked once, not 8x
    assert results == [(-7.0, None)] * 8       # every caller got the shared result


def test_reagent_rankings_aggregates_by_slot(tmp_path):
    ev = _make_ev(tmp_path, cnn_scoring="none")

    def prod(smi, score, acid, boronic):
        ev._score_cache[smi] = score
        ev._components_cache[smi] = [
            {"smiles": f"acid{acid}", "name": f"A{acid}"},
            {"smiles": "FRAG", "name": "FRAG"},          # fixed fragment slot
            {"smiles": f"bor{boronic}", "name": f"B{boronic}"},
        ]
    prod("p1", -8.0, 1, 1)
    prod("p2", -7.5, 1, 2)
    prod("p3", -5.0, 2, 1)

    r = ev.reagent_rankings()
    # only the variable slots are ranked; the single-reagent fragment (raw
    # component index 1) is omitted but still leads the display order (it's
    # the growth seed), so acid/boronic renumber to slots 1/2, not 0/2
    assert {s["slot"] for s in r} == {1, 2}
    acids = next(s for s in r if s["slot"] == 1)["reagents"]
    # minimize: A1 (mean -7.75, over 2 products) ranks above A2 (mean -5.0)
    assert acids[0]["name"] == "A1"
    assert acids[0]["count"] == 2
    assert acids[0]["best"] == -8.0
    assert acids[-1]["name"] == "A2"


def test_reagent_rankings_keeps_one_hit_wonders_for_best_sort(tmp_path):
    """A reagent that lands a single outstanding hit but also a much weaker
    one elsewhere can have a mean outside the top-``top`` cutoff -- it must
    still appear (with its real ``best``) so a best-sorted view can surface
    it, rather than silently dropping out of the ranking entirely."""
    ev = _make_ev(tmp_path, cnn_scoring="none")

    def prod(smi, score, acid):
        ev._score_cache[smi] = score
        ev._components_cache[smi] = [
            {"smiles": f"acid{acid}", "name": f"A{acid}"},
            {"smiles": "FRAG", "name": "FRAG"},
            {"smiles": "bor1", "name": "B1"},
        ]

    # A_lucky: one great hit (-9.5) but also one bad one (-5.5) -> mean -7.5,
    # outside the top-3-by-mean cutoff set by the three consistent A2..A4.
    prod("p_lucky_best", -9.5, "_lucky")
    prod("p_lucky_worst", -5.5, "_lucky")
    for i in range(2, 5):
        prod(f"p{i}a", -8.0, i)
        prod(f"p{i}b", -8.0, i)

    r = ev.reagent_rankings(top=3)
    # both the fragment (ci=1) and the single boronic value used throughout
    # (ci=2) are fixed here and lead the display order, so the acid slot
    # (the only variable one, raw ci=0) renumbers to display slot 2
    acids = next(s for s in r if s["slot"] == 2)["reagents"]
    names = {row["name"]: row for row in acids}
    assert "A_lucky" in names          # kept via the best-list, not dropped
    assert names["A_lucky"]["best"] == -9.5
    assert names["A_lucky"]["mean"] == -7.5
