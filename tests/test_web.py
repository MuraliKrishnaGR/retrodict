"""The web page's API: upload a photo, follow the live events, read the finished run, never pay twice."""
import io
import json

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tests.test_step2 import StoryModels  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app import config, web
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(web, "UPLOADS", tmp_path / "uploads")
    models = StoryModels()
    monkeypatch.setattr(web, "get_models", lambda: models)
    web.JOBS.clear()
    with TestClient(web.app) as c:
        c.models = models
        yield c


def _jpeg(color=(180, 170, 160)):
    buf = io.BytesIO()
    Image.new("RGB", (100, 80), color).save(buf, format="JPEG")
    return buf.getvalue()


def _events(client, job):
    with client.stream("GET", f"/api/jobs/{job}/events") as r:
        return [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]


def test_upload_streams_steps_and_moves_then_the_run_can_be_read(client):
    r = client.post("/api/run", files={"photo": ("tea.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    job, name = r.json()["job"], r.json()["name"]
    kinds = [e["kind"] for e in _events(client, job)]
    assert kinds[0] == "step_started" and "move" in kinds and kinds[-1] == "done"
    assert [e for e in kinds if e == "step_done"]                      # every step reported as it finished
    run = client.get(f"/api/runs/{name}").json()
    assert run["result"]["summary"] and "input.jpg" in run["files"] and "detail_o1.jpg" in run["files"]
    assert [x["name"] for x in client.get("/api/runs").json()] == [name]


def test_the_same_photo_is_never_paid_for_twice(client):
    first = client.post("/api/run", files={"photo": ("a.jpg", _jpeg(), "image/jpeg")}).json()
    _events(client, first["job"])
    calls = len(client.models.tags)
    again = client.post("/api/run", files={"photo": ("b.jpg", _jpeg(), "image/jpeg")}).json()
    assert again["name"] == first["name"]
    assert [e["kind"] for e in _events(client, again["job"])] == ["cached", "done"]
    assert len(client.models.tags) == calls                             # no model call at all


def test_only_photos_are_accepted_and_paths_cannot_escape(client):
    assert client.post("/api/run", files={"photo": ("x.txt", b"hello", "text/plain")}).status_code == 415
    assert client.get("/api/runs/..%2F..%2Fsecret").status_code == 404
    assert client.get("/api/jobs/nope/events").status_code == 404
