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


def _make_pose_set(jobdir, n):
    """``n`` ranked poses, as a finished job writes them: best-scored first,
    DockingRank 1..n."""
    jobdir.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(jobdir / "poses_0.sdf"))
    for rank in range(1, n + 1):
        m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(m, randomSeed=rank)
        m = Chem.RemoveHs(m)
        m.SetProp("DockingRank", str(rank))
        w.write(m)
    w.close()


def _ranks(sdf_text):
    return [int(line.strip())
            for i, line in enumerate(sdf_text.splitlines())
            if i and "<DockingRank>" in sdf_text.splitlines()[i - 1]]


def test_download_poses_all_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    _make_pose_set(jobs_dir() / "job1", 200)
    with TestClient(app) as client:
        r = client.get("/jobs/job1/poses/poses_0.sdf")
        assert r.status_code == 200
        assert r.text.count("$$$$") == 200


def test_download_poses_top_n(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    _make_pose_set(jobs_dir() / "job1", 200)
    with TestClient(app) as client:
        r = client.get("/jobs/job1/poses/poses_0.sdf", params={"n": 12})
        assert r.status_code == 200
        assert r.text.count("$$$$") == 12
        assert _ranks(r.text) == list(range(1, 13))      # the *best* 12, in order
        assert "poses_0_top12.sdf" in r.headers["content-disposition"]
        # asking for more than there are is not an error -- you get all of them
        assert client.get("/jobs/job1/poses/poses_0.sdf",
                          params={"n": 5000}).text.count("$$$$") == 200


def test_download_poses_top_percent(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    _make_pose_set(jobs_dir() / "job1", 200)
    with TestClient(app) as client:
        r = client.get("/jobs/job1/poses/poses_0.sdf", params={"pct": 10})
        assert r.status_code == 200
        assert r.text.count("$$$$") == 20
        assert _ranks(r.text) == list(range(1, 21))
        assert "poses_0_top10pct.sdf" in r.headers["content-disposition"]
        # a percentage too small to reach one whole pose still yields the best one
        assert client.get("/jobs/job1/poses/poses_0.sdf",
                          params={"pct": 0.1}).text.count("$$$$") == 1


def test_download_poses_rejects_bad_slice_args(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    _make_pose_set(jobs_dir() / "job1", 5)
    with TestClient(app) as client:
        assert client.get("/jobs/job1/poses/poses_0.sdf",
                          params={"n": 1, "pct": 10}).status_code == 400
        assert client.get("/jobs/job1/poses/poses_0.sdf", params={"n": 0}).status_code == 422
        assert client.get("/jobs/job1/poses/poses_0.sdf", params={"pct": 101}).status_code == 422
