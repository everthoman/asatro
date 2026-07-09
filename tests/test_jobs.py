"""Growth job layer + endpoints, driven with a fake docking runner (no gnina)."""
import json
import time

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

    def __init__(self, rows):
        self._rows = rows  # [(score, smiles, name), ...]

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

    def write_top_poses(self, path, n=100):
        w = Chem.SDWriter(path)
        written = 0
        for score, smi, name in sorted(self._rows, key=lambda r: r[0])[:n]:
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            AllChem.EmbedMolecule(m, randomSeed=1)
            m = Chem.RemoveHs(m)
            m.SetProp("_Name", name)
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
        steps=["suzuki"], fragment_slot=0,
        reactant_by_class={"boronic": _boronic(tmp_path)},
        cfg={"num_cycles": 1, "num_warmup": 1}, runner=_fake_runner)
    _await(job)
    assert job.status == "done"
    assert job.result["accessible_reactions"] == ["suzuki"]
    assert job.result["steps"] == [{"reaction_id": "suzuki", "name": "Suzuki coupling (aryl halide + boronic acid)"}]
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
        steps=["suzuki"], fragment_slot=0,
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
        steps=["suzuki"], fragment_slot=0,
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
        steps=["suzuki"], fragment_slot=0,
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
        steps=["suzuki"], fragment_slot=0,
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
                "steps": ["suzuki"], "fragment_slot": 0,
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
        assert d["result"]["accessible_reactions"] == ["suzuki"]
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
        fragment_path=sdf, receptor_path="", steps=["suzuki"], fragment_slot=0,
        pool_path=str(pool), cfg={"num_cycles": 1}, runner=runner)
    _await(job)
    assert job.status == "done"
    assert len(calls) == 1
    # the pool was pruned to the boronic component (2 boronics, not the acid)
    # -- reactant_files is one dict per step; step 0's boronic slot is index 1
    boronic_smi = calls[0]["reactant_files"][0][1]
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
