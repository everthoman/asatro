"""Single docked-pose download by gallery rank (GET /jobs/{id}/pose/{rank})."""
from rdkit import Chem
from rdkit.Chem import AllChem
from starlette.testclient import TestClient

from asatro.app import app
from asatro.jobs import jobs_dir


def _make_poses(jobdir, named=True):
    jobdir.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(jobdir / "poses_0.sdf"))
    for rank, smi in [(1, "c1ccccc1"), (2, "CCO")]:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=1)
        m = Chem.RemoveHs(m)
        if named:
            # As the evaluator titles a pose: fragment name + reagent id.
            m.SetProp("_Name", f"TH17145_{rank}0000")
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
        # Saved under the pose's own product name, not its rank.
        assert "TH17145_20000.sdf" in r.headers.get("content-disposition", "")
        assert client.get("/jobs/job1/pose/99").status_code == 404  # no such rank


def test_untitled_pose_falls_back_to_the_rank_name(tmp_path, monkeypatch):
    """A pose with no title (nothing named it) still downloads under a name
    that says which run and rank it came from."""
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    _make_poses(jobs_dir() / "job1", named=False)
    with TestClient(app) as client:
        r = client.get("/jobs/job1/pose/2")
        assert r.status_code == 200
        assert "job1_pose_2.sdf" in r.headers.get("content-disposition", "")


def test_pose_title_is_never_taken_raw_as_a_filename(tmp_path, monkeypatch):
    """An unnamed molecule falls back to its SMILES as the title, which is full
    of characters (and, in the worst case, path separators) a filename cannot
    carry -- so the title is slugified, never used as-is."""
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    jobdir = jobs_dir() / "job1"
    jobdir.mkdir(parents=True)
    m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    m = Chem.RemoveHs(m)
    m.SetProp("_Name", "../../O=C(N)c1ccccc1")
    m.SetProp("DockingRank", "1")
    w = Chem.SDWriter(str(jobdir / "poses_0.sdf"))
    w.write(m)
    w.close()
    with TestClient(app) as client:
        cd = client.get("/jobs/job1/pose/1").headers.get("content-disposition", "")
        assert "/" not in cd and ".." not in cd
        assert "O_C_N_c1ccccc1.sdf" in cd


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
        # saved under the run's name, not the on-disk poses_0.sdf
        assert "job1_poses.sdf" in r.headers["content-disposition"]


def test_download_poses_top_n(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    _make_pose_set(jobs_dir() / "job1", 200)
    with TestClient(app) as client:
        r = client.get("/jobs/job1/poses/poses_0.sdf", params={"n": 12})
        assert r.status_code == 200
        assert r.text.count("$$$$") == 12
        assert _ranks(r.text) == list(range(1, 13))      # the *best* 12, in order
        assert "job1_poses_top12.sdf" in r.headers["content-disposition"]
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
        assert "job1_poses_top10pct.sdf" in r.headers["content-disposition"]
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


# --- mid-run downloads -----------------------------------------------------
# A run's poses only reach poses_0.sdf when it finishes. Until then the live
# gallery can show a promising hit, so it must also be able to hand it over --
# out of the evaluator's own pose cache, keyed by SMILES rather than by rank.

def _live_job(job_id="live"):
    """A running job whose evaluator holds two docked poses, best first."""
    from asatro.combi import make_evaluator
    from asatro.jobs import GrowthJob, JOBS

    d = jobs_dir() / job_id
    d.mkdir(parents=True, exist_ok=True)
    rec = d / "receptor.pdb"
    rec.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n")
    ev = make_evaluator(receptor_path=str(rec), center=(0.0, 0.0, 0.0),
                        work_dir=str(d / "dock"))
    for score, smi, name in [(-9.1, "c1ccccc1", "TH17145_10000"),
                             (-7.4, "CCO", "TH17145_20000")]:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=1)
        m = Chem.RemoveHs(m)
        m.SetProp("_Name", name)
        m.SetProp("SMILES", smi)
        ev._score_cache[smi] = score
        ev._pose_cache[smi] = (score, m)
        ev._name_cache[smi] = name
        ev._components_cache[smi] = [{"smiles": smi, "name": name}]
    job = GrowthJob(id=job_id, dir=d, status="running")
    job.evaluator = ev
    JOBS[job_id] = job
    return job, ev


def test_live_pose_downloads_while_the_run_is_still_going(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    _live_job()
    with TestClient(app) as client:
        r = client.get("/jobs/live/live-pose", params={"smiles": "c1ccccc1"})
        assert r.status_code == 200
        assert r.text.count("$$$$") == 1
        assert "TH17145_10000" in r.text
        # Stamped like the final file: rank now, and the reagents behind it.
        assert "DockingRank" in r.text and "Reagent_1_Name" in r.text
        # Saved under the product's own name, as the finished-run download is.
        assert "TH17145_10000.sdf" in r.headers.get("content-disposition", "")


def test_live_pose_is_addressed_by_smiles_not_rank(tmp_path, monkeypatch):
    """The leaderboard reorders with every dock that lands, so a rank clicked
    off the gallery can name a different molecule by the time the request
    arrives. The SMILES on the card cannot -- asking for the runner-up gets the
    runner-up, and the rank it is stamped with follows the score."""
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    _live_job()
    with TestClient(app) as client:
        r = client.get("/jobs/live/live-pose", params={"smiles": "CCO"})
        assert r.status_code == 200
        assert "TH17145_20000" in r.text
        rank = r.text.split("<DockingRank>")[1].splitlines()[1].strip()
        assert rank == "2"


def test_live_pose_download_does_not_disturb_the_cached_pose(tmp_path, monkeypatch):
    """Downloading mid-run works on a copy: the pose still in the cache is
    untouched, so the final write_top_poses stamps the real ranks onto poses
    that carry nothing from a passing download."""
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    _job, ev = _live_job()
    with TestClient(app) as client:
        assert client.get("/jobs/live/live-pose", params={"smiles": "CCO"}).status_code == 200
    assert not ev._pose_cache["CCO"][1].HasProp("DockingRank")


def test_live_pose_404s_for_an_unscored_product_and_a_finished_job(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    job, _ev = _live_job()
    with TestClient(app) as client:
        assert client.get("/jobs/live/live-pose",
                          params={"smiles": "CCCCN"}).status_code == 404
        job.status = "done"
        # Finished: the poses file is the way in, and it holds the full set.
        assert client.get("/jobs/live/live-pose",
                          params={"smiles": "c1ccccc1"}).status_code == 404


def test_live_gallery_marks_which_hits_have_a_pose_to_download(tmp_path, monkeypatch):
    """The card only offers a download when there is one: a product that scored
    but has no cached pose is flagged, rather than given a dead link."""
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    _job, ev = _live_job()
    ev._score_cache["CCCCN"] = -8.0          # scored, but no pose came back
    ev._name_cache["CCCCN"] = "TH17145_30000"
    with TestClient(app) as client:
        items = client.get("/jobs/live/top").json()["items"]
    by_smiles = {it["smiles"]: it for it in items}
    assert by_smiles["c1ccccc1"]["pose"] is True
    assert by_smiles["CCCCN"]["pose"] is False
