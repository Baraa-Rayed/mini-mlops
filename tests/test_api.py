"""Tests for the HTTP API.

Uses FastAPI's TestClient, which drives the real app in-process — no server,
no ports, but the same routing, validation and status codes a real client
gets. Triggering hits the real scheduler and spawns real subprocesses.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="the API is an optional extra")

from fastapi.testclient import TestClient  # noqa: E402

from mini.api import create_app  # noqa: E402
from mini.state import SUCCESS  # noqa: E402


@pytest.fixture
def client(tmp_path):
    app = create_app(home=tmp_path / "home")
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_the_store_it_is_using(client, tmp_path):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["dags"] >= 2
    assert str(tmp_path) in body["home"]


def test_list_dags_finds_the_examples(client):
    ids = {d["dag_id"] for d in client.get("/dags").json()}
    assert {"iris", "flaky"} <= ids


def test_dag_detail_exposes_the_graph(client):
    body = client.get("/dags/iris").json()
    assert body["graph"]["select_best"] == ["train_logreg", "train_tree"]
    assert body["order"].index("prep") < body["order"].index("register")


def test_unknown_dag_is_404_not_500(client):
    assert client.get("/dags/nope").status_code == 404


def test_trigger_returns_202_with_a_run_id(client):
    response = client.post("/dags/flaky/runs", json={"wait": False})
    assert response.status_code == 202
    assert response.json()["run_id"].startswith("flaky__")
    assert response.json()["trigger"] == "api"


def test_trigger_with_wait_runs_to_completion(client):
    """A short DAG is worth waiting on; the default is not to, because a real
    pipeline outlives a sensible HTTP timeout."""
    body = client.post("/dags/flaky/runs", json={"wait": True}).json()
    assert body["state"] in {"success", "failed"}
    assert body["finished_at"] is not None
    states = {t["task_id"]: t["state"] for t in body["tasks"]}
    # flaky fails on purpose, and the failure must poison only its descendants.
    assert states["always_fails"] == "failed"
    assert states["never_runs"] == "upstream_failed"
    assert states["publish"] == SUCCESS


def test_run_detail_and_listing_agree(client):
    run_id = client.post("/dags/flaky/runs", json={"wait": True}).json()["run_id"]
    detail = client.get(f"/runs/{run_id}").json()
    listed = {r["run_id"]: r for r in client.get("/runs").json()}
    assert detail["run_id"] in listed
    assert listed[detail["run_id"]]["state"] == detail["state"]


def test_unknown_run_is_404(client):
    assert client.get("/runs/nope").status_code == 404


def test_task_logs_are_retrievable(client):
    run_id = client.post("/dags/flaky/runs", json={"wait": True}).json()["run_id"]
    body = client.get(f"/runs/{run_id}/tasks/ingest/logs").json()
    assert body["logs"]
    assert "mini.runner" in body["logs"][0]["text"]


def test_logs_for_a_task_that_never_ran_are_404(client):
    run_id = client.post("/dags/flaky/runs", json={"wait": True}).json()["run_id"]
    assert client.get(f"/runs/{run_id}/tasks/never_runs/logs").status_code == 404


def test_predict_requires_a_registered_model(client, monkeypatch, tmp_path):
    """A first-run state, so 404 rather than a 500 stack trace."""
    import mini.inference as inference

    monkeypatch.setattr(inference, "latest_version", lambda name=None: None)
    inference.clear_cache()
    response = client.post("/predict", json={"rows": [[5.1, 3.5, 1.4, 0.2]]})
    assert response.status_code == 404
    assert "run the iris pipeline" in response.json()["detail"]


def test_predict_rejects_a_wrong_width_row(client, monkeypatch):
    """422 with a readable message, not a schema-enforcement stack trace."""
    import mini.inference as inference

    class FakeModel:
        def predict(self, frame):
            raise AssertionError("must not reach the model")

    fake = inference.RegisteredModel(
        name="iris-classifier", version="1", uri="models:/iris-classifier/1",
        kind="LogisticRegression", accuracy="1.0",
        features=["a", "b", "c", "d"], classes=["x", "y", "z"],
        source_run="r", experiment="iris", model=FakeModel(),
    )
    monkeypatch.setattr(inference, "load", lambda *a, **k: fake)
    monkeypatch.setattr("mini.api.load", lambda *a, **k: fake)
    response = client.post("/predict", json={"rows": [[1.0, 2.0, 3.0]]})
    assert response.status_code == 422
    assert "expected 4 values" in response.json()["detail"]


def test_openapi_schema_is_served(client):
    """The point of the API: /docs renders this, so the system is explorable
    without anyone remembering curl invocations."""
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "/dags/{dag_id}/runs" in schema["paths"]
    assert client.get("/docs").status_code == 200
