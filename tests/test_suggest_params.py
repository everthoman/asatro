"""Pre-launch TS-budget suggestion: growth.suggest_growth_params + /suggest-params.

Dry-runs the pool resolution + reachability prune (no docking) so the form can
fill num_warmup/num_cycles from the *buildable* pools before launch.
"""
import json

from rdkit import Chem
from rdkit.Chem import AllChem

from asatro.growth import suggest_growth_params
from asatro.jobs import make_class_resolver


def _amine_fragment_sdf(tmp_path):
    m = Chem.AddHs(Chem.MolFromSmiles("NCc1ccccc1"))  # benzylamine
    AllChem.EmbedMolecule(m, randomSeed=7)
    m = Chem.RemoveHs(m)
    p = tmp_path / "frag.sdf"
    Chem.MolToMolFile(m, str(p))
    return str(p)


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("".join(f"{smi} {nm}\n" for smi, nm in rows))
    return str(p)


def _acids(tmp_path):
    # 3 with an aryl halide (survive the Suzuki reachability prune), 2 without
    return _write(tmp_path, "carboxylic_acid.smi", [
        ("OC(=O)c1ccc(Br)cc1", "a1"), ("OC(=O)c1ccccc1Cl", "a2"),
        ("OC(=O)c1cc(Br)ccc1", "a3"), ("CC(=O)O", "a4"), ("OCC(=O)O", "a5")])


def _boronics(tmp_path):
    return _write(tmp_path, "boronic.smi", [
        ("OB(O)c1ccccc1", "b1"), ("OB(O)c1ccncc1", "b2"),
        ("OB(O)c1ccc(C)cc1", "b3"), ("OB(O)c1cccnc1", "b4")])


# reversed amide -> suzuki: fragment fills the amine slot (1); Suzuki binds the
# aryl-halide slot (1), so the fresh reagent is a boronic pool.
STEPS = ["schotten_baumann_amide", {"reaction_id": "suzuki", "slot": 1}]


def test_suggest_growth_params_prunes_then_suggests(tmp_path):
    resolver = make_class_resolver(
        {"carboxylic_acid": _acids(tmp_path), "boronic": _boronics(tmp_path)})
    r = suggest_growth_params(
        fragment_sdf=_amine_fragment_sdf(tmp_path), steps=STEPS, fragment_slot=1,
        resolver=resolver, work_dir=str(tmp_path / "wd"))
    assert r["mode"] == "search"
    assert r["variable_slots"] == [3, 4]        # acids pruned 5 -> 3; boronics 4
    assert r["num_warmup"] == 3
    assert r["num_cycles"] == 500               # clamped floor for tiny pools
    assert r["est_docks"] == 3 * (3 + 1 + 4) + 500


def test_suggest_params_endpoint_fills_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    from starlette.testclient import TestClient

    from asatro.app import app
    frag = open(_amine_fragment_sdf(tmp_path), "rb").read()
    with TestClient(app) as client:
        r = client.post("/suggest-params", files=[
            ("fragment", ("frag.sdf", frag, "chemical/x-mdl-sdfile")),
            ("reactants", ("carboxylic_acid.smi", open(_acids(tmp_path), "rb").read(), "text/plain")),
            ("reactants", ("boronic.smi", open(_boronics(tmp_path), "rb").read(), "text/plain")),
        ], data={"config": json.dumps({"steps": STEPS, "fragment_slot": 1})})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["mode"] == "search"
    assert body["variable_slots"] == [3, 4]
    assert body["num_warmup"] == 3 and body["num_cycles"] == 500


def test_suggest_params_endpoint_reports_errors_gracefully(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    from starlette.testclient import TestClient

    from asatro.app import app
    frag = open(_amine_fragment_sdf(tmp_path), "rb").read()
    with TestClient(app) as client:
        # no steps -> not a 500, a clean {"ok": false}
        r = client.post("/suggest-params",
                        files=[("fragment", ("frag.sdf", frag, "chemical/x-mdl-sdfile"))],
                        data={"config": json.dumps({})})
    assert r.status_code == 200
    assert r.json()["ok"] is False
