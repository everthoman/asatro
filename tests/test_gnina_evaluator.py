"""Protecting-group stripping (deprotect_smiles) and its wiring into
GninaEvaluator.evaluate_detailed. Ported from ts-gnina, which found real
Boc/Cbz/Fmoc-protected building blocks in commercial pools reacting at one
handle while leaving the other's protecting group on the docked/reported
product.
"""
from rdkit import Chem

from asatro.combi import make_evaluator
from asatro.engine.gnina_evaluator import deprotect_smiles


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
    assert docked_with == [free]  # unchanged -- no second dock
