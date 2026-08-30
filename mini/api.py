"""The HTTP API — the orchestrator and the model, reachable from anywhere.

Maps to: Airflow's REST API (`/api/v1/dags/{id}/dagRuns`), plus a scoring
endpoint of our own.

Why this exists when MLflow already serves the model: `mlflow models serve`
answers `/invocations` and nothing else. It cannot trigger a pipeline, report
whether one succeeded, show you a task's logs, or decode a class index into a
label — and it is MLflow's own app, so there is nowhere to hook our tracing
in. This wraps the whole system instead of just the model.

Everything is browsable at /docs, which matters more than it sounds: a
system you can only exercise by remembering curl invocations is a system
nobody exercises.

    python -m mini api --host 0.0.0.0
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .dagbag import DEFAULT_DAGS_DIR, DagBag
from .executors import LocalExecutor
from .inference import NoModelRegistered, clear_cache, load, predict
from .scheduler import Scheduler
from .state import RUNNING, SUCCESS, Store


# --- request/response shapes ------------------------------------------------
# Declared rather than returning bare dicts: these are what /docs renders as
# the schema, and an API you cannot read the shape of is one you cannot use.
class TaskState(BaseModel):
    task_id: str
    state: str
    try_number: int
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None


class RunSummary(BaseModel):
    run_id: str
    dag_id: str
    state: str
    trigger: str
    created_at: float
    finished_at: float | None = None


class RunDetail(RunSummary):
    run_dir: str
    tasks: list[TaskState]
    results: dict[str, Any] = Field(default_factory=dict)


class DagSummary(BaseModel):
    dag_id: str
    description: str
    schedule: str | None
    tasks: list[str]


class DagDetail(DagSummary):
    graph: dict[str, list[str]] = Field(description="task_id -> its upstream task_ids")
    order: list[str] = Field(description="a valid topological execution order")


class TriggerRequest(BaseModel):
    wait: bool = Field(default=False, description="block until the run finishes")


class PredictRequest(BaseModel):
    rows: list[list[float]] = Field(
        description="one row of feature values per prediction",
        json_schema_extra={"example": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]},
    )
    model_name: str | None = None
    version: str | None = None
    trace: bool = Field(default=True, description="record the call in MLflow's Traces tab")


class PredictionOut(BaseModel):
    row: list[float]
    prediction: int
    label: str | None
    confidence: float | None


class PredictResponse(BaseModel):
    model_name: str
    version: str
    predictions: list[PredictionOut]


class ModelInfo(BaseModel):
    name: str
    version: str
    uri: str
    kind: str
    accuracy: str | None
    features: list[str]
    classes: list[str]
    source_run: str | None


def create_app(home: Path | str | None = None, dags_dir: Path | str = DEFAULT_DAGS_DIR,
               parallelism: int = 4):
    """Build the FastAPI app.

    A factory rather than a module-level `app` so tests can point it at a
    temporary store instead of the developer's real one.
    """
    from fastapi import Body, FastAPI, HTTPException

    store = Store(home) if home else Store()
    bag = DagBag(dags_dir)

    app = FastAPI(
        title="mini-mlops",
        version="0.1.0",
        description=(
            "Trigger pipelines, inspect runs, and call the registered model.\n\n"
            "The orchestrator's state lives in SQLite; experiment history and the "
            "model registry live in MLflow. Every prediction here is recorded as an "
            "MLflow trace, so the dashboard shows how the model is *used* as well as "
            "how it was built."
        ),
    )

    def _run_detail(run_id: str) -> RunDetail:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(404, f"no run {run_id!r}")
        tasks = [
            TaskState(task_id=tid, state=row["state"], try_number=row["try_number"],
                      started_at=row["started_at"], finished_at=row["finished_at"],
                      error=row["error"])
            for tid, row in sorted(store.task_states_full(run_id).items())
        ]
        return RunDetail(run_id=run["run_id"], dag_id=run["dag_id"], state=run["state"],
                         trigger=run["trigger"], created_at=run["created_at"],
                         finished_at=run["finished_at"], run_dir=run["run_dir"],
                         tasks=tasks, results=store.results(run_id))

    # --- health ---------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index():
        """Send the root to the docs.

        Browsing to the bare host is the first thing anyone does, and a 404
        there reads as "the server is broken" rather than "the interesting
        page is one path over".
        """
        from fastapi.responses import RedirectResponse

        return RedirectResponse("/docs")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        """Browsers ask for this unprompted; answering 204 keeps the log clean
        instead of filling it with 404s nobody caused."""
        from fastapi import Response

        return Response(status_code=204)

    @app.get("/health", tags=["system"])
    def health() -> dict:
        return {"status": "ok", "dags": len(bag), "home": str(store.home)}

    # --- pipelines ------------------------------------------------------
    @app.get("/dags", response_model=list[DagSummary], tags=["pipelines"])
    def list_dags():
        bag.collect()  # rescan, so an edited pipeline shows up without a restart
        return [DagSummary(dag_id=d.dag_id, description=d.description,
                           schedule=d.schedule, tasks=sorted(d.tasks))
                for d in bag]

    @app.get("/dags/{dag_id}", response_model=DagDetail, tags=["pipelines"])
    def get_dag(dag_id: str):
        try:
            dag = bag.get(dag_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from None
        return DagDetail(
            dag_id=dag.dag_id, description=dag.description, schedule=dag.schedule,
            tasks=sorted(dag.tasks),
            graph={tid: sorted(t.upstream) for tid, t in sorted(dag.tasks.items())},
            order=dag.topological_order(),
        )

    @app.post("/dags/{dag_id}/runs", response_model=RunDetail, status_code=202, tags=["pipelines"])
    def trigger_dag(dag_id: str, request: TriggerRequest = Body(default=TriggerRequest())):
        """Start a run. Returns immediately with the run_id unless `wait` is set.

        A pipeline takes tens of seconds, which is far longer than an HTTP
        request should hold a connection open, so the default is to hand back
        a run_id and let the caller poll `GET /runs/{run_id}`. That is the
        same shape Airflow's API uses, and for the same reason.
        """
        try:
            dag = bag.get(dag_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from None

        scheduler = Scheduler(store, executor=LocalExecutor(), parallelism=parallelism)
        run = store.create_run(dag.dag_id, trigger="api")
        for task_id in dag.tasks:
            store.init_task(run["run_id"], task_id)

        def drive():
            scheduler._run_loop(dag, run["run_id"], Path(run["run_dir"]))

        if request.wait:
            drive()
        else:
            # A daemon thread: if the server is shut down mid-run the process
            # must still exit. The run is left in `running` and `mini reap`
            # closes it out — the same story as a killed scheduler.
            threading.Thread(target=drive, daemon=True).start()
            for _ in range(50):  # let it register as started before replying
                if store.get_run(run["run_id"])["state"] == RUNNING:
                    break
                time.sleep(0.01)
        return _run_detail(run["run_id"])

    # --- runs -----------------------------------------------------------
    @app.get("/runs", response_model=list[RunSummary], tags=["runs"])
    def list_runs(dag_id: str | None = None, limit: int = 20):
        return [RunSummary(run_id=r["run_id"], dag_id=r["dag_id"], state=r["state"],
                           trigger=r["trigger"], created_at=r["created_at"],
                           finished_at=r["finished_at"])
                for r in store.list_runs(dag_id, limit=limit)]

    @app.get("/runs/{run_id}", response_model=RunDetail, tags=["runs"])
    def get_run(run_id: str):
        return _run_detail(run_id)

    @app.get("/runs/{run_id}/tasks/{task_id}/logs", tags=["runs"])
    def get_logs(run_id: str, task_id: str, all_tries: bool = False):
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(404, f"no run {run_id!r}")
        logs = sorted((Path(run["run_dir"]) / "logs").glob(f"{task_id}.*.log"))
        if not logs:
            raise HTTPException(404, f"no logs for task {task_id!r} in {run_id}")
        chosen = logs if all_tries else logs[-1:]
        return {"run_id": run_id, "task_id": task_id,
                "logs": [{"attempt": p.name, "text": p.read_text()} for p in chosen]}

    # --- the model ------------------------------------------------------
    @app.get("/model", response_model=ModelInfo, tags=["model"])
    def model_info(name: str | None = None, version: str | None = None):
        try:
            registered = load(name, version)
        except NoModelRegistered as exc:
            raise HTTPException(404, str(exc)) from None
        return ModelInfo(name=registered.name, version=registered.version, uri=registered.uri,
                         kind=registered.kind, accuracy=registered.accuracy,
                         features=registered.features, classes=registered.classes,
                         source_run=registered.source_run)

    @app.post("/model/reload", tags=["model"])
    def reload_model():
        """Drop the cached model, so a freshly promoted version is picked up
        without restarting the server."""
        clear_cache()
        return {"status": "cache cleared"}

    @app.post("/predict", response_model=PredictResponse, tags=["model"])
    def predict_rows(request: PredictRequest):
        """Score rows with the registered model.

        Unlike MLflow's `/invocations`, this decodes the class index into a
        label and reports confidence — a caller should not have to know that
        `2` means virginica.
        """
        try:
            registered = load(request.model_name, request.version)
        except NoModelRegistered as exc:
            raise HTTPException(404, str(exc)) from None
        try:
            results = predict(registered, request.rows, trace=request.trace)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return PredictResponse(
            model_name=registered.name,
            version=registered.version,
            predictions=[PredictionOut(row=r.row, prediction=r.prediction,
                                       label=r.label, confidence=r.confidence)
                         for r in results],
        )

    return app
