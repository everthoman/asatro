"""Downloading a run's console log (GET /jobs/{id}/log)."""
import json

import pytest
from starlette.testclient import TestClient

from asatro.app import app
from asatro.jobs import GrowthJob, JOBS, jobs_dir


@pytest.fixture(autouse=True)
def _isolated_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    JOBS.clear()
    yield
    JOBS.clear()


def _finished_job(job_id, lines):
    d = jobs_dir() / job_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.json").write_text(json.dumps(
        {"id": job_id, "status": "done", "started": 1.0, "finished": 2.0, "n_targets": 1}))
    (d / "run.log").write_text("".join(l + "\n" for l in lines))
    return d


def test_download_run_log(tmp_path):
    _finished_job("job1", ["[10:00:00] Growth job job1 started",
                           "[10:04:12] docked 40 | best CNN_VS=3.678",
                           "[10:09:30] Job done — 517 docked"])
    with TestClient(app) as client:
        r = client.get("/jobs/job1/log")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        assert 'filename="job1.log"' in r.headers.get("content-disposition", "")
        # The whole log, verbatim -- not a tail, not the in-memory buffer.
        assert r.text.splitlines()[0].endswith("started")
        assert r.text.splitlines()[-1].endswith("517 docked")
        assert len(r.text.splitlines()) == 3


def test_missing_log_is_404(tmp_path):
    (jobs_dir() / "nolog").mkdir(parents=True)
    with TestClient(app) as client:
        assert client.get("/jobs/nolog/log").status_code == 404
        assert client.get("/jobs/never-ran/log").status_code == 404


def test_log_of_a_running_job_serves_what_has_been_logged_so_far(tmp_path):
    """The link is live during a run: lines are appended to disk as they're
    emitted, so a mid-run download is a prefix, not an error."""
    d = jobs_dir() / "live"
    d.mkdir(parents=True)
    job = GrowthJob(id="live", dir=d, status="running")
    JOBS["live"] = job
    job.log("Growth job live started")
    with TestClient(app) as client:
        first = client.get("/jobs/live/log")
        assert first.status_code == 200
        assert "started" in first.text
        job.log("docked 40")
        second = client.get("/jobs/live/log")
        assert "docked 40" in second.text
        assert second.text.startswith(first.text)   # append-only


def test_log_path_cannot_escape_the_jobs_dir(tmp_path):
    with TestClient(app) as client:
        assert client.get("/jobs/..%2f..%2fetc/log").status_code in (400, 404)
