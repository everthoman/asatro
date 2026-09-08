"""Deleting old runs (one, several, all) and reclaiming staged uploads."""
import json
import os
import time

import pytest
from starlette.testclient import TestClient

from asatro.app import app
from asatro.jobs import (GrowthJob, JOBS, delete_job, jobs_dir, list_jobs,
                         staged_uploads, sweep_staged_uploads)


@pytest.fixture(autouse=True)
def _isolated_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("ASATRO_JOBS_DIR", str(tmp_path / "jobs"))
    JOBS.clear()
    yield
    JOBS.clear()


def _finished_job(job_id, payload_kb=1, status="done"):
    d = jobs_dir() / job_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.json").write_text(json.dumps(
        {"id": job_id, "status": status, "started": 1.0, "finished": 2.0, "n_targets": 1}))
    (d / "results.json").write_text("x" * (payload_kb * 1024))
    return d


def _running_job(job_id):
    d = jobs_dir() / job_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.json").write_text(json.dumps({"id": job_id, "status": "running"}))
    job = GrowthJob(id=job_id, dir=d, status="running")
    JOBS[job_id] = job
    return job


# --- listing ---------------------------------------------------------------

def test_listing_reports_size_and_skips_the_upload_staging_area():
    _finished_job("a", payload_kb=4)
    (jobs_dir() / "_uploads" / "1783501598040").mkdir(parents=True)
    (jobs_dir() / "_suggest").mkdir(parents=True)
    rows = list_jobs()
    assert [r["id"] for r in rows] == ["a"]
    assert rows[0]["size"] >= 4 * 1024        # what deleting it would reclaim


# --- deleting one ----------------------------------------------------------

def test_delete_one_removes_the_directory_and_reports_what_it_freed():
    d = _finished_job("old_run", payload_kb=8)
    with TestClient(app) as client:
        r = client.delete("/jobs/old_run")
        assert r.status_code == 200
        assert r.json()["freed"] >= 8 * 1024
    assert not d.exists()


def test_delete_refuses_a_running_job():
    """Its thread would go on writing into a deleted directory."""
    _running_job("live_run")
    with TestClient(app) as client:
        r = client.delete("/jobs/live_run")
        assert r.status_code == 409
        assert "cancel it" in r.json()["detail"]
    assert (jobs_dir() / "live_run").is_dir()


def test_delete_unknown_job_404s():
    with TestClient(app) as client:
        assert client.delete("/jobs/nope").status_code == 404


def test_delete_rejects_ids_that_escape_the_jobs_dir():
    outsider = jobs_dir().parent / "keep_me"
    outsider.mkdir(parents=True, exist_ok=True)
    for bad in ("../keep_me", "_uploads", "_suggest", "."):
        with pytest.raises(ValueError):
            delete_job(bad)
    assert outsider.is_dir()


def test_deleting_drops_the_in_memory_record_too():
    d = _finished_job("done_run")
    JOBS["done_run"] = GrowthJob(id="done_run", dir=d, status="done")
    delete_job("done_run")
    assert "done_run" not in JOBS


# --- deleting several / all ------------------------------------------------

def test_delete_many_reports_per_id_and_keeps_going():
    """One running job in the selection must not block the rest -- and must
    itself survive."""
    _finished_job("a"), _finished_job("b")
    _running_job("live_run")
    with TestClient(app) as client:
        r = client.post("/jobs/delete", json={"ids": ["a", "live_run", "b", "ghost"]})
        assert r.status_code == 200
        body = r.json()
    assert sorted(body["deleted"]) == ["a", "b"]
    assert set(body["failed"]) == {"live_run", "ghost"}
    assert body["freed"] > 0
    assert (jobs_dir() / "live_run").is_dir()
    assert not (jobs_dir() / "a").exists()


def test_delete_all_is_just_every_listed_id():
    for name in ("a", "b", "c"):
        _finished_job(name)
    with TestClient(app) as client:
        ids = [j["id"] for j in client.get("/jobs").json()["jobs"]]
        assert client.post("/jobs/delete", json={"ids": ids}).json()["deleted"].sort() == ids.sort()
        assert client.get("/jobs").json()["jobs"] == []


# --- staged uploads --------------------------------------------------------

def _staged(name, age_hours):
    d = jobs_dir() / "_uploads" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "receptor.pdb").write_text("y" * 2048)
    when = time.time() - age_hours * 3600
    os.utime(d, (when, when))
    return d


def test_sweep_deletes_old_staged_uploads_and_keeps_recent_ones():
    old, fresh = _staged("old", 48), _staged("fresh", 1)
    assert staged_uploads()["n"] == 2
    out = sweep_staged_uploads(max_age_hours=24)
    assert out["deleted"] == 1 and out["kept"] == 1 and out["freed"] >= 2048
    assert not old.exists() and fresh.is_dir()


def test_sweep_spares_uploads_a_still_running_job_may_be_reading():
    """A long run's staging dir ages past the cutoff mid-run; it is still in
    use, so the oldest live job's start time floors the cutoff."""
    stale_but_in_use = _staged("launch_of_the_long_run", 47)   # staged at launch
    job = _running_job("long_run")
    job.started = time.time() - 47 * 3600
    out = sweep_staged_uploads(max_age_hours=24)
    assert out["deleted"] == 0 and out["kept"] == 1
    assert stale_but_in_use.is_dir()


def test_sweep_endpoint(tmp_path):
    _staged("old", 48)
    with TestClient(app) as client:
        assert client.get("/uploads").json()["n"] == 1
        assert client.post("/uploads/sweep", json={"max_age_hours": 24}).json()["deleted"] == 1
        assert client.get("/uploads").json()["n"] == 0
