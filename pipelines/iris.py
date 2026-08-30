"""A real ML pipeline, tracked in MLflow: prep -> train two -> pick one -> register.

    prep ──┬── train_logreg ──┬── select_best ── register
           └── train_tree   ──┘

Three things this example exists to demonstrate:

1. **Tasks pass references, not data.** Every function returns a small JSON
   dict of ids, paths and metrics. The feature data moves through the run
   directory on disk and the artifact store; only the *name* of it moves
   through the orchestrator. Break this rule and your metadata database
   becomes your data lake.

2. **Fan-out is free.** `train_logreg` and `train_tree` have no edge between
   them, so the scheduler runs them at the same time without being told to.
   Parallelism is a property of the graph, not a flag.

3. **The orchestrator schedules; MLflow remembers.** Two different jobs, and
   conflating them is the classic mistake. Our SQLite store answers "did this
   task succeed, and should the next one start?" It has no opinion about
   accuracy. MLflow answers "what did we try, what did it score, and which
   version is live?" Every task opens an MLflow run tagged with the
   orchestrator's run id, so the two views join on one key.

Open the ledger with:  mini ui
"""

from __future__ import annotations

import json

from mini import DAG, Context, Task
from mini.tracking import DEFAULT_MODEL_NAME, connect, promote, task_run


def prep(ctx: Context) -> dict:
    """Load the dataset and split it. Writes CSVs; returns their paths."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris(as_frame=True)
    frame = data.frame
    test_size = ctx.params.get("test_size", 0.25)
    train, test = train_test_split(
        frame, test_size=test_size, random_state=0, stratify=frame["target"]
    )
    train_path, test_path = ctx.artifact("train.csv"), ctx.artifact("test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    with task_run(ctx) as (mlflow, _):
        mlflow.log_params({"test_size": test_size, "stratified": True, "random_state": 0})
        mlflow.log_metrics({"train_rows": len(train), "test_rows": len(test)})
        # The split itself is an artifact worth keeping: without it, a metric
        # from six months ago cannot be reproduced or even argued with.
        mlflow.log_artifact(str(train_path), artifact_path="dataset")
        mlflow.log_artifact(str(test_path), artifact_path="dataset")

    return {
        "train": str(train_path),
        "test": str(test_path),
        "rows": len(train),
        # Carried forward so inference never has to guess the column order or
        # decode a bare class index. A model without its schema is a liability.
        "feature_names": list(data.feature_names),
        "class_names": list(data.target_names),
    }


def _fit(ctx: Context, model, params: dict) -> dict:
    """Train one candidate and log it as an MLflow model.

    Returns the MLflow run id, which is what makes the winner registrable
    later: `runs:/<id>/model` is the canonical way to name a trained artifact,
    and it carries the run's params and metrics along with it.
    """
    import pandas as pd

    train = pd.read_csv(ctx.upstream["prep"]["train"])
    X, y = train.drop(columns=["target"]), train["target"]
    model.fit(X, y)

    with task_run(ctx) as (mlflow, run):
        mlflow.log_params(params)
        mlflow.log_metric("train_accuracy", float(model.score(X, y)))
        mlflow.sklearn.log_model(
            model,
            name="model",
            # The signature records expected columns and dtypes, so a caller
            # sending the wrong shape gets an error instead of a wrong answer.
            input_example=X.head(3),
        )
        return {
            "kind": type(model).__name__,
            "mlflow_run_id": run.info.run_id,
            # `runs:/<id>/model`, not the `models:/m-<id>` that log_model
            # returns in MLflow 3. That newer LoggedModel id does not resolve
            # from another process against this store, and every one of our
            # tasks *is* another process — the downstream task would fail with
            # "Logged model not found" on a model that had just been written.
            "model_uri": f"runs:/{run.info.run_id}/model",
            "train_accuracy": float(model.score(X, y)),
        }


def train_logreg(ctx: Context) -> dict:
    from sklearn.linear_model import LogisticRegression

    max_iter = ctx.params.get("max_iter", 500)
    return _fit(ctx, LogisticRegression(max_iter=max_iter), {"max_iter": max_iter})


def train_tree(ctx: Context) -> dict:
    from sklearn.tree import DecisionTreeClassifier

    max_depth = ctx.params.get("max_depth", 3)
    return _fit(ctx, DecisionTreeClassifier(max_depth=max_depth, random_state=0),
                {"max_depth": max_depth, "random_state": 0})


def select_best(ctx: Context) -> dict:
    """Score every candidate on the held-out split and keep the winner.

    Note how this reads `ctx.upstream` for *both* trainers without knowing how
    many there were — the graph decided that, not this code.
    """
    import pandas as pd

    # Point MLflow at our store *before* resolving any `runs:/` URI. Loading a
    # model is a tracking-store lookup, not a file read: without this the
    # default ./mlruns is consulted, and a run written seconds earlier by a
    # sibling task comes back as "Run not found".
    _mlflow = connect(ctx.dag_id)

    test = pd.read_csv(ctx.upstream["prep"]["test"])
    X, y = test.drop(columns=["target"]), test["target"]

    scored = []
    for task_id, up in sorted(ctx.upstream.items()):
        if not task_id.startswith("train_"):
            continue
        model = _mlflow.sklearn.load_model(up["model_uri"])
        accuracy = float(model.score(X, y))
        scored.append({
            "task": task_id,
            "kind": up["kind"],
            "mlflow_run_id": up["mlflow_run_id"],
            "model_uri": up["model_uri"],
            "accuracy": accuracy,
        })

    scored.sort(key=lambda row: row["accuracy"], reverse=True)
    best = scored[0]
    ctx.artifact("metrics.json").write_text(json.dumps(scored, indent=2))

    with task_run(ctx) as (mlflow, _):
        # Log every candidate's held-out score against the *comparison* run, so
        # the bake-off is a single row you can look at rather than something to
        # reconstruct by opening each trainer.
        for row in scored:
            mlflow.log_metric(f"test_accuracy.{row['task']}", row["accuracy"])
        mlflow.log_metric("best_accuracy", best["accuracy"])
        mlflow.set_tag("winner", best["task"])
        mlflow.log_artifact(str(ctx.artifact("metrics.json")))

    print(f"candidates: {[(s['task'], round(s['accuracy'], 4)) for s in scored]}")
    print(f"winner: {best['task']} @ {best['accuracy']:.4f}")
    return {"best": best, "candidates": scored}


def register(ctx: Context) -> dict:
    """The quality gate, and the promotion.

    Raising here is the *point*: a model that misses the bar must fail the
    run, not get promoted with a warning nobody reads. Note the ordering —
    the gate is checked before `promote`, so a rejected model never becomes a
    registry version at all.
    """
    best = ctx.upstream["select_best"]["best"]
    prep_out = ctx.upstream["prep"]
    threshold = ctx.params.get("min_accuracy", 0.90)
    name = ctx.params.get("model_name", DEFAULT_MODEL_NAME)

    if best["accuracy"] < threshold:
        raise ValueError(
            f"accuracy {best['accuracy']:.4f} below threshold {threshold} — refusing to register"
        )

    version = promote(best["model_uri"], name=name, tags={
        "mini.run_id": ctx.run_id,
        "mini.dag_id": ctx.dag_id,
        "accuracy": f"{best['accuracy']:.4f}",
        "kind": best["kind"],
        # The label vocabulary travels with the version. sklearn predicts a
        # bare class index; without this, serving it would mean hard-coding
        # "0 means setosa" somewhere far away from the model that decided it.
        "features": json.dumps(prep_out["feature_names"]),
        "classes": json.dumps(prep_out["class_names"]),
    })

    with task_run(ctx) as (mlflow, _):
        mlflow.log_metric("registered_accuracy", best["accuracy"])
        mlflow.set_tags({"registered_model": name, "registered_version": version.version})

    print(f"registered {best['kind']} as {name} v{version.version} (accuracy {best['accuracy']:.4f})")
    return {
        "model_name": name,
        "version": int(version.version),
        "model_uri": f"models:/{name}/{version.version}",
        "accuracy": best["accuracy"],
        "features": prep_out["feature_names"],
        "classes": prep_out["class_names"],
    }


with DAG("iris", schedule="@hourly", description="train + gate an iris classifier") as dag:
    t_prep = Task("prep", prep, params={"test_size": 0.25})
    t_logreg = Task("train_logreg", train_logreg, retries=1)
    t_tree = Task("train_tree", train_tree, retries=1)
    t_select = Task("select_best", select_best)
    t_register = Task("register", register, params={"min_accuracy": 0.90})

    t_prep >> [t_logreg, t_tree] >> t_select >> t_register
