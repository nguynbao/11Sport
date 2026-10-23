from __future__ import annotations

from fastapi.testclient import TestClient

from app.constants import JobStatus
from app.main import create_app
from app.worker import claim_next, process_one, recover_stuck_jobs


def client():
    return TestClient(create_app())


def upload(c, video, **form):
    data = {"target_langs": "en,vi", "voice_gender": "female", "subtitle_mode": "hard",
            "auto_approve": "false", "orig_volume": "0.15"}
    data.update(form)
    with open(video, "rb") as f:
        return c.post("/api/jobs", data=data, files={"file": ("Video có dấu.mp4", f, "video/mp4")})


def test_health_and_config(env):
    with client() as c:
        assert c.get("/api/health").json()["provider_mode"] == "fake"
        cfg = c.get("/api/config").json()
        assert [x["code"] for x in cfg["target_languages"]] == ["en", "vi"]


def test_upload_validation(env, tmp_path):
    bad = tmp_path / "a.txt"
    bad.write_text("x")
    with client() as c:
        with open(bad, "rb") as f:
            r = c.post("/api/jobs", data={"target_langs": "en"}, files={"file": ("a.txt", f, "text/plain")})
        assert r.status_code == 400
        assert "Định dạng" in r.json()["detail"]


def test_bad_lang(env, sample_video):
    with client() as c:
        assert upload(c, sample_video, target_langs="fr").status_code == 400
        assert upload(c, sample_video, target_langs="en,en").status_code == 400


def test_full_flow_via_api_and_worker(env, sample_video):
    with client() as c:
        r = upload(c, sample_video)
        assert r.status_code == 200, r.text
        jobs = r.json()
        assert len(jobs) == 2 and jobs[0]["group_id"] == jobs[1]["group_id"]
        jid = jobs[0]["id"]

        # worker phase A for both jobs
        assert process_one() and process_one()
        assert claim_next() is None
        j = c.get(f"/api/jobs/{jid}").json()
        assert j["status"] == JobStatus.REVIEW and j["segment_count"] > 0

        segs = c.get(f"/api/jobs/{jid}/segments").json()
        r = c.put(f"/api/jobs/{jid}/segments", json={"items": [{"idx": 0, "translated_text": "Hello team"}]})
        assert r.status_code == 200 and r.json()[0]["edited"] is True
        assert c.post(f"/api/jobs/{jid}/approve").status_code == 200
        assert c.post(f"/api/jobs/{jid}/approve").status_code == 409

        assert process_one()  # phase B
        j = c.get(f"/api/jobs/{jid}").json()
        assert j["status"] == JobStatus.DONE, j["error"]
        assert j["has_output"]

        r = c.get(f"/api/jobs/{jid}/files/output")
        assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
        assert "_en.mp4" in r.headers["content-disposition"]
        assert c.get(f"/api/jobs/{jid}/files/srt").status_code == 200

        # editing not allowed when DONE; reopen then edit
        assert c.put(f"/api/jobs/{jid}/segments", json={"items": [{"idx": 0, "translated_text": "x"}]}
                     ).status_code == 409
        assert c.post(f"/api/jobs/{jid}/reopen").json()["status"] == JobStatus.REVIEW

        listing = c.get("/api/jobs").json()
        assert listing["total"] == 2
        assert len(segs) == j["segment_count"]

        assert c.delete(f"/api/jobs/{jid}").status_code == 200
        assert c.get(f"/api/jobs/{jid}").status_code == 404


def test_recover_stuck(env, sample_video):
    from app.db import SessionLocal
    from app.models import Job

    with client() as c:
        jid = upload(c, sample_video, target_langs="en").json()[0]["id"]
    with SessionLocal() as db:
        db.get(Job, jid).status = JobStatus.TRANSCRIBING
        db.commit()
    recover_stuck_jobs()
    with SessionLocal() as db:
        assert db.get(Job, jid).status == JobStatus.QUEUED


def test_glossary_crud(env):
    with client() as c:
        r = c.post("/api/glossary", json={"source_term": "11Sport", "en": "11Sport", "vi": "11Sport"})
        assert r.status_code == 200
        tid = r.json()["id"]
        assert c.post("/api/glossary", json={"source_term": "11Sport"}).status_code == 409
        assert c.put(f"/api/glossary/{tid}", json={"source_term": "11Sport", "vi": "Mười Một Sport"}).json()["vi"] == "Mười Một Sport"
        assert len(c.get("/api/glossary").json()) == 1
        assert c.delete(f"/api/glossary/{tid}").status_code == 200
