"""Single docked-pose download by gallery rank (GET /jobs/{id}/pose/{rank})."""
from rdkit import Chem
from rdkit.Chem import AllChem
from starlette.testclient import TestClient

from asatro.app import app
from asatro.jobs import jobs_dir


def _make_poses(jobdir):
    jobdir.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(jobdir / "poses_0.sdf"))
    for rank, smi in [(1, "c1ccccc1"), (2, "CCO")]:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=1)
        m = Chem.RemoveHs(m)
        m.SetProp("DockingRank", str(rank))
        m.SetProp("SMILES", smi)
        w.write(m)
    w.close()


def test_download_single_pose_by_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    _make_poses(jobs_dir() / "job1")
    with TestClient(app) as client:
        r = client.get("/jobs/job1/pose/2")
        assert r.status_code == 200
        assert r.text.count("$$$$") == 1                 # exactly one SDF record
        assert "DockingRank" in r.text                   # props preserved
        assert "attachment" in r.headers.get("content-disposition", "")
        assert "job1_pose_2.sdf" in r.headers.get("content-disposition", "")
        assert client.get("/jobs/job1/pose/99").status_code == 404  # no such rank


def test_download_pose_without_poses_file_404(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    (jobs_dir() / "empty").mkdir(parents=True)
    with TestClient(app) as client:
        assert client.get("/jobs/empty/pose/1").status_code == 404
