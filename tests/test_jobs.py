"""Growth job layer + endpoints, driven with a fake docking runner (no gnina)."""
import json
import time

from rdkit import Chem
from rdkit.Chem import AllChem

import asatro.jobs as jobs
from asatro.jobs import JOBS, start_growth_job


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
        reactant_by_class={"boronic": _boronic(tmp_path)},
        cfg={"num_cycles": 1, "num_warmup": 1}, runner=_fake_runner)
    _await(job)
    assert job.status == "done"
    assert job.result["accessible_reactions"] == ["suzuki"]
    run = job.result["runs"][0]
    assert run["target"]["reaction_id"] == "suzuki"
    assert run["n_docked"] == 2
    # ranked best-first (minimize: lowest score first)
    assert run["top"][0]["score"] == -7.5
    # persisted to disk
    assert (job.dir / "results.json").is_file()
    assert any("Growing suzuki" in ln for ln in job.lines)


def test_growth_job_skips_when_pruned(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    sdf = _bound_sdf(tmp_path)
    # Build a wall PDB across the C-Br exit so suzuki is pruned.
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
        reactant_by_class={"boronic": _boronic(tmp_path)},
        cfg={}, runner=lambda **k: called.append(k) or ([], None))
    _await(job)
    assert job.status == "done"
    assert job.result["accessible_reactions"] == []
    assert called == []  # nothing grown


def test_growth_job_error_is_captured(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    bad = tmp_path / "bad.sdf"; bad.write_text("not an sdf")
    job = start_growth_job(
        fragment_path=str(bad), receptor_path="",
        reactant_by_class={"boronic": _boronic(tmp_path)}, runner=_fake_runner)
    _await(job)
    assert job.status == "error" and job.error


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
            data={"config": json.dumps({"num_cycles": 1, "num_warmup": 1})},
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
