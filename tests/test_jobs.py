"""Growth job layer + endpoints, driven with a fake docking runner (no gnina)."""
import json
import threading
import time

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import asatro.jobs as jobs
from asatro.jobs import JOBS, start_combi_job, start_growth_job


def _bound_sdf(tmp_path, smiles="Brc1ccccc1"):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(m)
    m = Chem.RemoveHs(m)
    p = tmp_path / "frag.sdf"
    Chem.MolToMolFile(m, str(p))
    return str(p)


def _boronic(tmp_path):
    p = tmp_path / "boronic.smi"
    p.write_text("OB(O)c1ccccc1 phB\n")
    return str(p)


def _fake_runner(**kwargs):
    # Pretend a dock happened: return ([score, smiles, name] rows, evaluator=None).
    return ([[-7.5, "GROWN_SMILES", "frag_phB"], [-6.1, "OTHER", "frag_x"]], None)


class _FakeEvaluator:
    """Minimal stand-in for a real GninaEvaluator: tracks every dock (warm-up
    included) in its own cache, independent of what ``search()`` returns."""
    higher_is_better = False
    score_field = "minimizedAffinity"

    def __init__(self, rows, components=None):
        self._rows = rows  # [(score, smiles, name), ...]
        self._components = components or {}  # smiles -> [{"smiles","name"}, ...]

    def top_scored(self, n=12):
        return sorted(self._rows, key=lambda r: r[0])[:n]

    def stats(self):
        best = min((r[0] for r in self._rows), default=None)
        return {"unique_scored": len(self._rows), "docked": len(self._rows), "best_score": best}

    def convergence(self):
        pts, best = [], None
        for i, (score, _smi, _name) in enumerate(self._rows, start=1):
            if best is None or score < best:
                best = score
                pts.append((i, best))
        return pts

    def components_scored(self):
        return dict(self._components)

    def write_top_poses(self, path, n=100):
        w = Chem.SDWriter(path)
        written = 0
        for rank, (score, smi, name) in enumerate(sorted(self._rows, key=lambda r: r[0])[:n], start=1):
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            AllChem.EmbedMolecule(m, randomSeed=1)
            m = Chem.RemoveHs(m)
            m.SetProp("_Name", name)
            m.SetProp("DockingRank", str(rank))
            w.write(m)
            written += 1
        w.close()
        return written


def _await(job, timeout=10):
    t0 = time.time()
    while job.status in ("queued", "running") and time.time() - t0 < timeout:
        time.sleep(0.02)
    return job


def test_growth_job_runs_and_summarizes(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    sdf = _bound_sdf(tmp_path)
    job = start_growth_job(
        fragment_path=sdf, receptor_path="",        # open pocket
        steps=["suzuki"], fragment_slot=1,
        reactant_by_class={"boronic": _boronic(tmp_path)},
        cfg={"num_cycles": 1, "num_warmup": 1}, runner=_fake_runner)
    _await(job)
    assert job.status == "done"
    # bromobenzene's aryl-halide handle is accepted by several start reactions
    # now (buchwald, sonogashira, ullmann, ...) -- just check suzuki survived
    # the pre-pass, not that it's the only accessible reaction.
    assert "suzuki" in job.result["accessible_reactions"]
    assert job.result["steps"] == [
        {"reaction_id": "suzuki", "slot": None, "name": "Suzuki"}]
    run = job.result["runs"][0]
    assert run["n_docked"] == 2
    # ranked best-first (minimize: lowest score first)
    assert run["top"][0]["score"] == -7.5
    # persisted to disk
    assert (job.dir / "results.json").is_file()
    assert any("route suzuki" in ln for ln in job.lines)


def test_growth_job_summarizes_from_evaluator_not_search_rows(tmp_path, monkeypatch):
    """Regression: with a real evaluator, warm-up docks (one per reagent --
    always real docking work) must survive into the summary even when
    search() itself returns nothing new. This is the normal case whenever the
    reagent library is small enough that warm-up alone exhausts it -- a real
    ``run_growth`` call there returns ([], evaluator), and the old code that
    only looked at the (empty) rows silently dropped every scored product."""
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    sdf = _bound_sdf(tmp_path)
    ev = _FakeEvaluator([(-9.5, "c1ccccc1", "a"), (-7.0, "CCO", "b")])

    def runner(**k):
        return ([], ev)

    job = start_growth_job(
        fragment_path=sdf, receptor_path="",
        steps=["suzuki"], fragment_slot=1,
        reactant_by_class={"boronic": _boronic(tmp_path)},
        cfg={"num_cycles": 1, "num_warmup": 1}, runner=runner)
    _await(job)
    assert job.status == "done"
    run = job.result["runs"][0]
    assert run["n_docked"] == 2
    assert run["top"][0]["score"] == -9.5 and run["top"][0]["smiles"] == "c1ccccc1"
    assert run["poses"] == "poses_0.sdf"
    assert (job.dir / "poses_0.sdf").is_file()


def test_growth_job_errors_when_chosen_slot_is_pruned(tmp_path, monkeypatch):
    """The user picks step 1 from what /prune showed as accessible, but a job
    re-runs the pre-pass itself (the source of truth at run time) and must
    refuse -- as a job error, not a silent skip -- if that slot turns out
    pruned (e.g. stale UI state, or refine=true tightening the geometric
    pass's verdict)."""
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    sdf = _bound_sdf(tmp_path)
    # Build a wall PDB across the C-Br exit so suzuki's slot is pruned.
    from asatro.chemistry.accessibility import growth_vectors
    import numpy as np
    mol = Chem.MolFromMolFile(sdf, removeHs=True)
    ev = growth_vectors(mol, "aryl_halide")[0]
    n = ev.direction / np.linalg.norm(ev.direction)
    a = np.cross(n, [1, 0, 0]); a /= np.linalg.norm(a); b = np.cross(n, a)
    c = ev.attach_pos + n * 1.6
    lines, i = [], 0
    for u in np.arange(-6, 6.01, 0.8):
        for v in np.arange(-6, 6.01, 0.8):
            p = c + u * a + v * b; i += 1
            lines.append(f"ATOM  {i:5d}  C   WAL A   1    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00           C")
    wall = tmp_path / "wall.pdb"; wall.write_text("\n".join(lines))

    called = []
    job = start_growth_job(
        fragment_path=sdf, receptor_path=str(wall),
        steps=["suzuki"], fragment_slot=1,
        reactant_by_class={"boronic": _boronic(tmp_path)},
        cfg={}, runner=lambda **k: called.append(k) or ([], None))
    _await(job)
    assert job.status == "error"
    assert "pruned" in job.error
    assert called == []  # nothing grown


def test_growth_job_error_is_captured(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    bad = tmp_path / "bad.sdf"; bad.write_text("not an sdf")
    job = start_growth_job(
        fragment_path=str(bad), receptor_path="",
        steps=["suzuki"], fragment_slot=1,
        reactant_by_class={"boronic": _boronic(tmp_path)}, runner=_fake_runner)
    _await(job)
    assert job.status == "error" and job.error


def test_growth_job_passes_filters_to_runner(tmp_path, monkeypatch):
    """The ``filters`` block of a job's config builds a MolFilters and reaches
    the runner (and from there the evaluator) -- PAINS/REOS/MW/logP apply to
    every enumerated product before docking."""
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    sdf = _bound_sdf(tmp_path)
    captured = []

    def runner(**k):
        captured.append(k.get("filters"))
        return _fake_runner(**k)

    job = start_growth_job(
        fragment_path=sdf, receptor_path="",
        steps=["suzuki"], fragment_slot=1,
        reactant_by_class={"boronic": _boronic(tmp_path)},
        cfg={"num_cycles": 1, "num_warmup": 1,
             "filters": {"mw": [100, 400], "logp": [None, 5]}},
        runner=runner)
    _await(job)
    assert job.status == "done"
    assert captured and captured[0] is not None
    f = captured[0]
    assert f.mw_range == (100.0, 400.0)
    assert f.logp_range == (None, 5.0)
    assert f.pains_patterns == [] and f.reos_rules == []  # not requested


def test_grow_endpoint_and_jobs_listing(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    # Make the endpoint's background job use the fake runner instead of gnina.
    monkeypatch.setattr(jobs, "run_growth", _fake_runner)
    from starlette.testclient import TestClient
    from asatro.app import app

    sdf_bytes = open(_bound_sdf(tmp_path), "rb").read()
    with TestClient(app) as client:
        r = client.post(
            "/grow",
            files={
                "fragment": ("frag.sdf", sdf_bytes, "chemical/x-mdl-sdfile"),
                "receptor": ("receptor.pdb", b"", "chemical/x-pdb"),
                "reactants": ("boronic.smi", b"OB(O)c1ccccc1 phB\n", "text/plain"),
            },
            data={"config": json.dumps({
                "steps": ["suzuki"], "fragment_slot": 1,
                "num_cycles": 1, "num_warmup": 1})},
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]

        for _ in range(200):
            d = client.get(f"/jobs/{job_id}").json()
            if d["status"] in ("done", "error", "cancelled"):
                break
            time.sleep(0.02)
        assert d["status"] == "done", d
        assert "suzuki" in d["result"]["accessible_reactions"]
        assert any(j["id"] == job_id for j in client.get("/jobs").json()["jobs"])


def test_grow_endpoint_rejects_missing_steps(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    from starlette.testclient import TestClient
    from asatro.app import app

    sdf_bytes = open(_bound_sdf(tmp_path), "rb").read()
    with TestClient(app) as client:
        r = client.post(
            "/grow",
            files={
                "fragment": ("frag.sdf", sdf_bytes, "chemical/x-mdl-sdfile"),
                "receptor": ("receptor.pdb", b"", "chemical/x-pdb"),
            },
            data={"config": json.dumps({})},
        )
        assert r.status_code == 400
        assert "steps" in r.text


def test_growth_job_with_master_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    sdf = _bound_sdf(tmp_path)                     # bromobenzene -> suzuki (boronic)
    pool = tmp_path / "pool.smi"
    pool.write_text("OB(O)c1ccccc1 phB\nOB(O)c1ccc(C)cc1 tolB\nCCC(=O)O acid\n")

    calls = []
    def runner(**k):
        calls.append(k)
        return ([[-7.0, "X", "x"]], None)

    job = start_growth_job(
        fragment_path=sdf, receptor_path="", steps=["suzuki"], fragment_slot=1,
        pool_path=str(pool), cfg={"num_cycles": 1}, runner=runner)
    _await(job)
    assert job.status == "done"
    assert len(calls) == 1
    # the pool was pruned to the boronic component (2 boronics, not the acid)
    # -- reactant_files is one dict per step; step 0's boronic slot is index 0
    boronic_smi = calls[0]["reactant_files"][0][0]
    names = [l.split()[1] for l in open(boronic_smi).read().splitlines() if l.strip()]
    assert sorted(names) == ["phB", "tolB"]


def _fake_combi_runner(**kwargs):
    return ([[-7.5, "COMBI_SMILES", "p1"], [-6.1, "OTHER", "p2"]], None)


def test_combi_job_runs_and_summarizes(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    rec = tmp_path / "receptor.pdb"; rec.write_text("")
    halide = tmp_path / "halide.smi"; halide.write_text("Brc1ccccc1 phBr\n")
    boronic = tmp_path / "boronic.smi"; boronic.write_text("OB(O)c1ccccc1 phB\n")
    job = start_combi_job(
        receptor_path=str(rec), steps=["suzuki"],
        reagent_files=[[str(halide), str(boronic)]],
        center=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0),
        cfg={"num_cycles": 1, "num_warmup": 1}, runner=_fake_combi_runner)
    _await(job)
    assert job.status == "done"
    run = job.result["runs"][0]
    assert run["n_docked"] == 2
    # ranked best-first (minimize: lowest score first)
    assert run["top"][0]["score"] == -7.5
    assert (job.dir / "results.json").is_file()
    assert any("Combi job" in ln for ln in job.lines)


def test_combi_job_persists_steps_and_components(tmp_path, monkeypatch):
    """The route (steps) and per-hit reagent provenance (components) that
    /jobs/{id}/seed needs are both in the persisted result."""
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    rec = tmp_path / "receptor.pdb"; rec.write_text("")
    boronic = tmp_path / "boronic.smi"; boronic.write_text("OB(O)c1ccccc1 phB\n")
    halide = tmp_path / "halide.smi"; halide.write_text("Brc1ccccc1 phBr\n")
    ev = _FakeEvaluator(
        [(-7.5, "c1ccc(-c2ccccc2)cc1", "phB_phBr")],
        components={"c1ccc(-c2ccccc2)cc1": [
            {"smiles": "OB(O)c1ccccc1", "name": "phB"},
            {"smiles": "Brc1ccccc1", "name": "phBr"},
        ]},
    )

    def runner(**k):
        return ([], ev)

    job = start_combi_job(
        receptor_path=str(rec), steps=["suzuki"],
        reagent_files=[[str(boronic), str(halide)]],
        center=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0),
        cfg={"num_cycles": 1}, runner=runner)
    _await(job)
    assert job.status == "done"
    assert job.result["steps"] == [
        {"reaction_id": "suzuki", "slot": None, "name": "Suzuki"}]
    top0 = job.result["runs"][0]["top"][0]
    assert top0["components"] == [
        {"smiles": "OB(O)c1ccccc1", "name": "phB"},
        {"smiles": "Brc1ccccc1", "name": "phBr"},
    ]


def test_combi_job_error_is_captured(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))

    def bad_runner(**kwargs):
        raise ValueError("boom")

    job = start_combi_job(
        receptor_path="", steps=["suzuki"], reagent_files=[["a.smi", "b.smi"]],
        center=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0), runner=bad_runner)
    _await(job)
    assert job.status == "error" and job.error == "boom"


def test_combi_endpoint_and_jobs_listing(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    # Make the endpoint's background job use the fake runner instead of gnina.
    monkeypatch.setattr(jobs, "run_combi", _fake_combi_runner)
    from starlette.testclient import TestClient
    from asatro.app import app

    with TestClient(app) as client:
        r = client.post(
            "/combi",
            files=[
                ("receptor", ("receptor.pdb", b"", "chemical/x-pdb")),
                ("reactants", ("halide.smi", b"Brc1ccccc1 phBr\n", "text/plain")),
                ("reactants", ("boronic.smi", b"OB(O)c1ccccc1 phB\n", "text/plain")),
            ],
            data={"config": json.dumps({
                "steps": ["suzuki"], "center": [0.0, 0.0, 0.0], "size": [20.0, 20.0, 20.0],
                "num_cycles": 1, "num_warmup": 1})},
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]

        for _ in range(200):
            d = client.get(f"/jobs/{job_id}").json()
            if d["status"] in ("done", "error", "cancelled"):
                break
            time.sleep(0.02)
        assert d["status"] == "done", d
        assert d["result"]["runs"][0]["n_docked"] == 2
        assert any(j["id"] == job_id for j in client.get("/jobs").json()["jobs"])


def test_seed_endpoint_carves_a_component_from_a_finished_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    from rdkit import Chem
    product = Chem.CanonSmiles("CC(=O)NCc1ccncc1")  # amide of an amine + acetic acid
    ev = _FakeEvaluator(
        [(-7.5, product, "acid1_amine1")],
        components={product: [
            {"smiles": "CC(=O)O", "name": "acid1"},
            {"smiles": "NCc1ccncc1", "name": "amine1"},
        ]},
    )

    def runner(**k):
        return ([], ev)

    monkeypatch.setattr(jobs, "run_combi", runner)
    from starlette.testclient import TestClient
    from asatro.app import app

    with TestClient(app) as client:
        r = client.post(
            "/combi",
            files=[
                ("receptor", ("receptor.pdb", b"", "chemical/x-pdb")),
                ("reactants", ("acid.smi", b"CC(=O)O acid1\n", "text/plain")),
                ("reactants", ("amine.smi", b"NCc1ccncc1 amine1\n", "text/plain")),
            ],
            data={"config": json.dumps({
                "steps": ["schotten_baumann_amide"],
                "center": [0.0, 0.0, 0.0], "size": [20.0, 20.0, 20.0]})},
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
        for _ in range(200):
            d = client.get(f"/jobs/{job_id}").json()
            if d["status"] in ("done", "error", "cancelled"):
                break
            time.sleep(0.02)
        assert d["status"] == "done", d

        # component 0 = the acid (hydroxyl drops, carbonyl kept)
        r = client.post(f"/jobs/{job_id}/seed", data={"rank": 1, "component_index": 0})
        assert r.status_code == 200, r.text
        carved = Chem.MolFromMolBlock(r.text)
        assert carved is not None and carved.GetNumConformers() == 1
        assert Chem.MolToSmiles(carved) == Chem.CanonSmiles("CC=O")

        # component 1 = the amine
        r = client.post(f"/jobs/{job_id}/seed", data={"rank": 1, "component_index": 1})
        assert r.status_code == 200, r.text
        carved = Chem.MolFromMolBlock(r.text)
        assert Chem.MolToSmiles(carved) == Chem.CanonSmiles("NCc1ccncc1")

        # out-of-range rank / component_index -> 400
        assert client.post(f"/jobs/{job_id}/seed", data={"rank": 99, "component_index": 0}).status_code == 400
        assert client.post(f"/jobs/{job_id}/seed", data={"rank": 1, "component_index": 99}).status_code == 400


def test_seed_endpoint_carves_from_a_generalized_slot1_extend_step(tmp_path, monkeypatch):
    """Step 2 reuses "schotten_baumann_amide" generically with slot=0
    (intermediate binds the acid pattern -- this reaction's own component
    order is [acid, amine]; a diacid step-1 reagent leaves a free -COOH for
    it to react through -- see tests/test_combi.py's
    test_route_sampler_binds_intermediate_to_a_non_first_slot for the same
    chemistry proven directly against RDKit). Confirms job.result["steps"]
    persists "slot" correctly and /jobs/{id}/seed resolves the *fresh*
    (slot-1) reagent of that reused step -- the one place a fresh_indices
    mismatch between build_combi_route and component_route_meta would
    silently corrupt seed provenance."""
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    from rdkit import Chem
    product = Chem.CanonSmiles("CCNC(=O)c1ccc(C(=O)NCCC)cc1")
    ev = _FakeEvaluator(
        [(-7.5, product, "diacid_amine1_amine2")],
        components={product: [
            {"smiles": "OC(=O)c1ccc(C(=O)O)cc1", "name": "diacid"},
            {"smiles": "CCN", "name": "amine1"},
            {"smiles": "CCCN", "name": "amine2"},
        ]},
    )

    def runner(**k):
        return ([], ev)

    job = start_combi_job(
        receptor_path="",
        steps=["schotten_baumann_amide",
               {"reaction_id": "schotten_baumann_amide", "slot": 0}],
        reagent_files=[["diacid.smi", "amine1.smi"], ["amine2.smi"]],
        center=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0),
        cfg={"num_cycles": 1}, runner=runner)
    _await(job)
    assert job.status == "done"
    assert job.result["steps"] == [
        {"reaction_id": "schotten_baumann_amide", "slot": None, "name": "Schotten-Baumann_amide"},
        {"reaction_id": "schotten_baumann_amide", "slot": 0, "name": "Schotten-Baumann_amide"},
    ]

    from starlette.testclient import TestClient
    from asatro.app import app

    with TestClient(app) as client:
        # component_index 2 = step 2's sole fresh component (slot 1, the
        # amine) -- slot 0 (the acid) is excluded since it binds the
        # intermediate, not a real reagent.
        r = client.post(f"/jobs/{job.id}/seed", data={"rank": 1, "component_index": 2})
        assert r.status_code == 200, r.text
        carved = Chem.MolFromMolBlock(r.text)
        assert carved is not None and carved.GetNumConformers() == 1
        assert Chem.MolToSmiles(carved) == Chem.CanonSmiles("CCCN")


def test_seed_endpoint_unknown_job(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    from starlette.testclient import TestClient
    from asatro.app import app
    with TestClient(app) as client:
        r = client.post("/jobs/does-not-exist/seed", data={"rank": 1, "component_index": 0})
        assert r.status_code == 404


def test_combi_endpoint_rejects_missing_steps(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    from starlette.testclient import TestClient
    from asatro.app import app
    with TestClient(app) as client:
        r = client.post(
            "/combi",
            files=[("receptor", ("receptor.pdb", b"", "chemical/x-pdb"))],
            data={"config": json.dumps({})})
        assert r.status_code == 400
        assert "steps" in r.text


def test_combi_endpoint_rejects_reagent_count_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    from starlette.testclient import TestClient
    from asatro.app import app
    with TestClient(app) as client:
        r = client.post(
            "/combi",
            files=[
                ("receptor", ("receptor.pdb", b"", "chemical/x-pdb")),
                ("reactants", ("halide.smi", b"Brc1ccccc1 phBr\n", "text/plain")),
            ],
            data={"config": json.dumps({
                "steps": ["suzuki"], "center": [0.0, 0.0, 0.0], "size": [20.0, 20.0, 20.0]})})
        assert r.status_code == 400
        assert "need 2 reagent" in r.text


def test_combi_endpoint_requires_binding_site(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    from starlette.testclient import TestClient
    from asatro.app import app
    with TestClient(app) as client:
        r = client.post(
            "/combi",
            files=[
                ("receptor", ("receptor.pdb", b"", "chemical/x-pdb")),
                ("reactants", ("halide.smi", b"Brc1ccccc1 phBr\n", "text/plain")),
                ("reactants", ("boronic.smi", b"OB(O)c1ccccc1 phB\n", "text/plain")),
            ],
            data={"config": json.dumps({"steps": ["suzuki"]})})
        assert r.status_code == 400
        assert "reference ligand" in r.text


def _job(tmp_path, job_id="j1"):
    d = tmp_path / job_id
    d.mkdir(parents=True, exist_ok=True)
    return jobs.GrowthJob(id=job_id, dir=d)


def test_dock_resources_defaults_to_serial_no_gpu(tmp_path):
    concurrency, cpu, gpu_ids = jobs._dock_resources({}, _job(tmp_path))
    assert concurrency == 1 and cpu is None and gpu_ids is None


def test_dock_resources_splits_cpu_across_concurrency(tmp_path):
    concurrency, cpu, gpu_ids = jobs._dock_resources({"concurrency": 4}, _job(tmp_path))
    assert concurrency == 4
    assert cpu == max(1, jobs.DOCK_CPU // 4)


def test_dock_resources_explicit_cpu_not_overridden(tmp_path):
    concurrency, cpu, gpu_ids = jobs._dock_resources(
        {"concurrency": 4, "cpu": 6}, _job(tmp_path))
    assert cpu == 6


def test_dock_resources_no_gpu_ids_when_cnn_scoring_off(tmp_path, monkeypatch):
    # cnn_scoring="none" is pure CPU Vina -- must never auto-populate GPU ids,
    # even with concurrency>1 and multiple real GPUs on the box.
    monkeypatch.setattr(jobs, "_detect_gpu_ids", lambda: [0, 1])
    _, _, gpu_ids = jobs._dock_resources(
        {"concurrency": 4, "cnn_scoring": "none"}, _job(tmp_path))
    assert gpu_ids is None


def test_dock_resources_no_gpu_ids_when_concurrency_1(tmp_path, monkeypatch):
    # No contention risk at concurrency=1 -- don't bother.
    monkeypatch.setattr(jobs, "_detect_gpu_ids", lambda: [0, 1])
    _, _, gpu_ids = jobs._dock_resources(
        {"concurrency": 1, "cnn_scoring": "rescore"}, _job(tmp_path))
    assert gpu_ids is None


def test_dock_resources_auto_detects_gpu_ids_for_concurrent_cnn_scoring(tmp_path, monkeypatch):
    # This is the real bug fix: concurrency>1 + CNN rescoring used to pin
    # every concurrent dock to the same GPU (gpu_id defaults to 0), which
    # caused a genuine production hang under contention. Multiple real GPUs
    # should now be spread across concurrent docks automatically.
    monkeypatch.setattr(jobs, "_detect_gpu_ids", lambda: [0, 1])
    _, _, gpu_ids = jobs._dock_resources(
        {"concurrency": 4, "cnn_scoring": "rescore"}, _job(tmp_path))
    assert gpu_ids == [0, 1]


def test_dock_resources_explicit_gpu_ids_not_overridden(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_detect_gpu_ids", lambda: [0, 1])
    _, _, gpu_ids = jobs._dock_resources(
        {"concurrency": 4, "cnn_scoring": "rescore", "gpu_ids": [2]}, _job(tmp_path))
    assert gpu_ids == [2]


def test_dock_resources_single_gpu_detected_stays_none(tmp_path, monkeypatch):
    # Only one real GPU on the box -- nothing to round-robin, so leave it to
    # the evaluator's plain gpu_id default rather than a pointless [0] list.
    monkeypatch.setattr(jobs, "_detect_gpu_ids", lambda: [0])
    _, _, gpu_ids = jobs._dock_resources(
        {"concurrency": 4, "cnn_scoring": "rescore"}, _job(tmp_path))
    assert gpu_ids is None


def test_reap_orphaned_jobs_marks_stale_running_as_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    d = tmp_path / "jobs" / "stuck_job"
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({
        "id": "stuck_job", "status": "running", "error": None,
        "started": 123.0, "finished": None, "n_targets": 0}))
    (d / "run.log").write_text("[00:00:00] started\n")

    fixed = jobs.reap_orphaned_jobs()

    assert fixed == ["stuck_job"]
    meta = json.loads((d / "job.json").read_text())
    assert meta["status"] == "error"
    assert "orphaned" in meta["error"]
    assert meta["finished"] is not None
    assert "orphaned" in (d / "run.log").read_text()


def test_reap_orphaned_jobs_leaves_finished_jobs_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    d = tmp_path / "jobs" / "done_job"
    d.mkdir(parents=True)
    original = {"id": "done_job", "status": "done", "error": None,
               "started": 1.0, "finished": 2.0, "n_targets": 1}
    (d / "job.json").write_text(json.dumps(original))

    fixed = jobs.reap_orphaned_jobs()

    assert fixed == []
    assert json.loads((d / "job.json").read_text()) == original


def test_watchdog_stays_quiet_when_rss_is_healthy(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_WATCHDOG_SOFT_MB", 4096.0)
    monkeypatch.setattr(jobs, "_WATCHDOG_POLL_S", 0.0)
    monkeypatch.setattr(jobs, "_process_rss_mb", lambda: 200.0)

    job = jobs.GrowthJob(id="wd-healthy", dir=tmp_path)
    stop_event = threading.Event()
    poll_count = {"n": 0}
    real_wait = stop_event.wait

    def counting_wait(timeout):
        poll_count["n"] += 1
        if poll_count["n"] >= 5:
            stop_event.set()
        return real_wait(timeout)
    monkeypatch.setattr(stop_event, "wait", counting_wait)

    jobs._watchdog(job, stop_event)  # returns normally once stop_event is set

    assert not job.cancel_event.is_set()
    assert job.lines == []


def test_watchdog_cancels_job_past_soft_limit(tmp_path, monkeypatch):
    # Regression test: a real, still-unexplained leak OOM-killed asatro-webapp
    # twice on 2026-08-12 on a shared ~130-user box. This bounds the blast
    # radius regardless of root cause -- past a soft RSS threshold, cancel the
    # job the normal way (same cancel_event the UI's Cancel button uses).
    monkeypatch.setattr(jobs, "_WATCHDOG_SOFT_MB", 100.0)
    monkeypatch.setattr(jobs, "_WATCHDOG_HARD_MB", 10_000.0)  # unreachable here
    monkeypatch.setattr(jobs, "_WATCHDOG_GRACE_S", 10_000.0)  # unreachable here
    monkeypatch.setattr(jobs, "_WATCHDOG_POLL_S", 0.0)

    rss_values = iter([50.0, 150.0, 150.0])
    monkeypatch.setattr(jobs, "_process_rss_mb", lambda: next(rss_values, 150.0))

    job = jobs.GrowthJob(id="wd-soft", dir=tmp_path)
    stop_event = threading.Event()
    real_wait = stop_event.wait

    def stop_after_third_poll(timeout):
        if stop_after_third_poll.n >= 3:
            stop_event.set()
        stop_after_third_poll.n += 1
        return real_wait(timeout)
    stop_after_third_poll.n = 0
    monkeypatch.setattr(stop_event, "wait", stop_after_third_poll)

    jobs._watchdog(job, stop_event)

    assert job.cancel_event.is_set()
    assert any("soft limit" in ln for ln in job.lines)


def test_watchdog_force_exits_when_cancellation_does_not_reclaim_memory(tmp_path, monkeypatch):
    # The live 2026-08-12 recurrence: cancellation was requested but RSS kept
    # climbing for minutes afterward -- something was leaking in a path that
    # never checks cancel_event. Simulate that: RSS crosses the soft limit,
    # cancellation fires, but RSS keeps climbing straight past the hard
    # ceiling. The watchdog must force-exit rather than trust cancellation.
    monkeypatch.setattr(jobs, "_WATCHDOG_SOFT_MB", 100.0)
    monkeypatch.setattr(jobs, "_WATCHDOG_HARD_MB", 200.0)
    monkeypatch.setattr(jobs, "_WATCHDOG_GRACE_S", 10_000.0)  # hard ceiling trips first
    monkeypatch.setattr(jobs, "_WATCHDOG_POLL_S", 0.0)

    rss_values = iter([50.0, 150.0, 250.0])
    monkeypatch.setattr(jobs, "_process_rss_mb", lambda: next(rss_values, 250.0))

    exited = {}

    def fake_exit(code):
        exited["code"] = code
        raise SystemExit(code)  # stand in for a real, unrecoverable process exit
    monkeypatch.setattr(jobs.os, "_exit", fake_exit)

    job = jobs.GrowthJob(id="wd-hard", dir=tmp_path)
    stop_event = threading.Event()

    with pytest.raises(SystemExit):
        jobs._watchdog(job, stop_event)

    assert job.cancel_event.is_set()
    assert exited["code"] == 1
    assert any("soft limit" in ln for ln in job.lines)
    assert any("forcing process exit" in ln for ln in job.lines)


def test_watchdog_force_exits_after_grace_period_even_below_hard_ceiling(tmp_path, monkeypatch):
    # RSS plateaus just above the soft limit -- never reaching the hard
    # ceiling -- but cancellation still isn't bringing it down. The grace-
    # period check must catch this "stuck, not climbing" case too, not just
    # a fast runaway.
    monkeypatch.setattr(jobs, "_WATCHDOG_SOFT_MB", 100.0)
    monkeypatch.setattr(jobs, "_WATCHDOG_HARD_MB", 10_000.0)  # unreachable here
    monkeypatch.setattr(jobs, "_WATCHDOG_GRACE_S", 0.0)  # trips immediately after soft
    monkeypatch.setattr(jobs, "_WATCHDOG_POLL_S", 0.0)
    monkeypatch.setattr(jobs, "_process_rss_mb", lambda: 150.0)

    exited = {}

    def fake_exit(code):
        exited["code"] = code
        raise SystemExit(code)
    monkeypatch.setattr(jobs.os, "_exit", fake_exit)

    job = jobs.GrowthJob(id="wd-grace", dir=tmp_path)
    stop_event = threading.Event()

    with pytest.raises(SystemExit):
        jobs._watchdog(job, stop_event)

    assert job.cancel_event.is_set()
    assert exited["code"] == 1


def test_growth_job_runs_normally_with_watchdog_wired_in(tmp_path, monkeypatch):
    # The watchdog wraps every real job thread now (start_growth_job/
    # start_combi_job) -- a healthy, fast job must still complete normally.
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    sdf = _bound_sdf(tmp_path)
    job = start_growth_job(
        fragment_path=sdf, receptor_path="",
        steps=["suzuki"], fragment_slot=1,
        reactant_by_class={"boronic": _boronic(tmp_path)},
        cfg={"num_cycles": 1, "num_warmup": 1}, runner=_fake_runner)
    _await(job)
    assert job.status == "done"
    assert not job.cancel_event.is_set()


def test_pool_preview_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    from starlette.testclient import TestClient
    from asatro.app import app
    with TestClient(app) as client:
        r = client.post("/pool-preview", files={
            "pool": ("pool.smi", b"NCc1ccccc1 a\nOB(O)c1ccccc1 b\nc1ccccc1 none\n", "text/plain")})
        assert r.status_code == 200
        j = r.json()
        assert j["n_total"] == 3 and j["n_untagged"] == 1
        assert j["counts"].get("primary_amine") == 1 and j["counts"].get("boronic") == 1
