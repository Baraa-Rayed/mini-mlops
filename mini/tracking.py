"""MLflow integration — the experiment ledger and the real model registry.

Maps to: MLflow Tracking + MLflow Model Registry.

This module is where the project stops hand-rolling and starts using the
actual tool. The `model_registry/` folder the iris pipeline used to write was
fifteen lines of `shutil.copy2` — enough to demonstrate the *idea* of
promotion, and nowhere near enough to answer the questions a registry exists
to answer: which run produced this, what did it score, what else was tried,
and which version is serving.

Two deliberate constraints:

**Every import is inside a function.** The orchestrator core depends on
nothing outside the standard library, and that stays true. Tracking is
something a *pipeline* opts into; the scheduler must never need it.

**The backend is SQLite, not a file store.** MLflow's file store cannot host a
Model Registry at all — `register_model` against a `file://` URI fails. The
registry needs a database, so we point tracking at `mlflow.db` beside the
orchestrator's own state and keep artifacts on disk next to it.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

# One registered model per pipeline is the common shape; the iris DAG uses
# this name and `mlflow models serve` can address it by name and version.
DEFAULT_MODEL_NAME = "iris-classifier"


def home() -> Path:
    from .state import HOME

    return Path(HOME)


def tracking_uri() -> str:
    """Where runs are recorded. SQLite because the registry demands a DB."""
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{home() / 'mlflow.db'}"


def artifact_root() -> str:
    return os.environ.get("MLFLOW_ARTIFACT_ROOT") or str(home() / "mlartifacts")


def connect(experiment: str):
    """Point MLflow at our store and make sure the experiment exists.

    The experiment is created with an explicit `artifact_location`; left to
    itself MLflow drops artifacts in `./mlruns` relative to whatever directory
    the task happened to start in, which for us is a different place per
    executor.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    root = Path(artifact_root())
    root.mkdir(parents=True, exist_ok=True)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=root.as_uri())
    mlflow.set_experiment(experiment)
    return mlflow


@contextmanager
def task_run(ctx, experiment: str | None = None):
    """One MLflow run per task, tagged so a whole DAG run can be reassembled.

    Each task is its own process, so there is no shared parent run object to
    nest under. Tags do the grouping instead: filter the MLflow UI by
    `mini.run_id` and you get exactly the tasks of one pipeline execution.
    """
    mlflow = connect(experiment or ctx.dag_id)
    with mlflow.start_run(
        run_name=f"{ctx.task_id}",
        tags={
            "mini.dag_id": ctx.dag_id,
            "mini.run_id": ctx.run_id,
            "mini.task_id": ctx.task_id,
            "mlflow.note.content": f"{ctx.dag_id}/{ctx.task_id} from mini-mlops run {ctx.run_id}",
        },
    ) as run:
        yield mlflow, run


def promote(model_uri: str, name: str = DEFAULT_MODEL_NAME, tags: dict | None = None):
    """Register a logged model as a new version of `name`.

    Returns the ModelVersion. Versioning is MLflow's job, not ours — the
    hand-rolled registry keyed versions by our run_id, which made "what is
    version 3?" unanswerable without reading JSON files.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    version = mlflow.register_model(model_uri, name, tags=tags or {})
    return version


def latest_version(name: str = DEFAULT_MODEL_NAME):
    """The newest registered version, or None if nothing is registered."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri())
    versions = MlflowClient().search_model_versions(f"name='{name}'")
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))
