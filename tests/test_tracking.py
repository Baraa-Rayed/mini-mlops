"""Tests for the MLflow integration.

Deliberately thin. MLflow is a dependency, not our code — what is worth
testing is the seam: that we point it at *our* store rather than whatever
`./mlruns` happens to be nearby, and that a task's run is findable afterwards
by the orchestrator run that produced it.
"""

from __future__ import annotations

import pytest

from mini.runner import Context
from mini.tracking import artifact_root, connect, task_run, tracking_uri

mlflow = pytest.importorskip("mlflow", reason="mlflow is an optional pipeline dependency")


@pytest.fixture
def store_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MINI_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ARTIFACT_ROOT", raising=False)
    # `mini.state.HOME` is read at import time, so the env var alone is not
    # enough once the module is already loaded.
    import mini.state

    monkeypatch.setattr(mini.state, "HOME", tmp_path / "home")
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    return tmp_path / "home"


def ctx_for(task_id: str, run_id: str = "iris__test__abc123") -> Context:
    return Context(dag_id="iris", run_id=run_id, task_id=task_id, run_dir=None)


def test_tracking_uri_is_sqlite_not_a_file_store(store_home):
    """The Model Registry cannot exist on a file:// store, so this must be a DB."""
    uri = tracking_uri()
    assert uri.startswith("sqlite:///")
    assert str(store_home) in uri


def test_tracking_uri_respects_an_explicit_override(store_home, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    assert tracking_uri() == "http://localhost:5000"


def test_artifacts_land_under_mini_home(store_home):
    assert str(store_home) in artifact_root()


def test_connect_creates_the_experiment_with_an_explicit_artifact_location(store_home):
    """Left to itself MLflow drops artifacts in ./mlruns relative to whatever
    directory the task started in — which differs per executor."""
    connect("iris")
    experiment = mlflow.get_experiment_by_name("iris")
    assert experiment is not None
    assert str(store_home) in experiment.artifact_location


def test_task_run_tags_link_back_to_the_orchestrator_run(store_home):
    """The join key between the two systems: our SQLite store says whether a
    task succeeded, MLflow says what it scored, and `mini.run_id` is what lets
    you line the two up."""
    with task_run(ctx_for("train_logreg")) as (mlf, run):
        mlf.log_metric("train_accuracy", 0.97)

    finished = mlflow.get_run(run.info.run_id)
    assert finished.data.tags["mini.run_id"] == "iris__test__abc123"
    assert finished.data.tags["mini.task_id"] == "train_logreg"
    assert finished.data.tags["mini.dag_id"] == "iris"
    assert finished.data.metrics["train_accuracy"] == pytest.approx(0.97)


def test_one_dag_run_is_recoverable_from_its_tag(store_home):
    """Tasks are separate processes, so there is no parent run to nest under.
    Tags do the grouping — this is the query the MLflow UI filter box runs."""
    for task_id in ("prep", "train_logreg", "select_best"):
        with task_run(ctx_for(task_id)) as (mlf, _):
            mlf.log_metric("noop", 1)

    experiment = mlflow.get_experiment_by_name("iris")
    found = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.`mini.run_id` = 'iris__test__abc123'",
    )
    assert sorted(found["tags.mini.task_id"]) == ["prep", "select_best", "train_logreg"]


def test_prediction_trace_records_inference(store_home):
    """Training is recorded as runs, inference as traces. Without this, a
    served model answers requests that leave no mark anywhere — which is how
    drift goes unnoticed."""
    from mlflow.tracking import MlflowClient

    from mini.tracking import prediction_trace

    with prediction_trace("iris", "models:/iris-classifier/1") as span:
        span.set_inputs({"rows": [[5.1, 3.5, 1.4, 0.2]]})
        span.set_outputs({"predictions": [0], "labels": ["setosa"]})

    mlflow.flush_trace_async_logging()
    experiment = mlflow.get_experiment_by_name("iris")
    traces = MlflowClient().search_traces(locations=[experiment.experiment_id])
    assert len(traces) == 1
    assert "setosa" in str(traces[0].data.spans[0].outputs)
