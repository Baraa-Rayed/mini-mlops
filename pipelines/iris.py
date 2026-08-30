"""A real (small) ML pipeline: prep -> train two models -> pick one -> register.

    prep ──┬── train_logreg ──┬── select_best ── register
           └── train_tree   ──┘

Two things this example exists to demonstrate:

1. **Tasks pass references, not data.** Every function returns a small JSON
   dict of paths and metrics. The 1MB of feature data moves through the run
   directory on disk; only the *name* of it moves through the orchestrator.
   Break this rule and your metadata database becomes your data lake.

2. **Fan-out is free.** `train_logreg` and `train_tree` have no edge between
   them, so the scheduler runs them at the same time without being told to.
   Parallelism is a property of the graph, not a flag.
"""

from __future__ import annotations

import json
import os

from mini import DAG, Context, Task

# The registry outlives any single run, so it must not live in the run
# directory. Overridable because the docker executor mounts the code read-only.
REGISTRY = os.environ.get("MINI_REGISTRY", "model_registry")


def prep(ctx: Context) -> dict:
    """Load the dataset and split it. Writes CSVs; returns their paths."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris(as_frame=True)
    frame = data.frame
    train, test = train_test_split(
        frame, test_size=ctx.params.get("test_size", 0.25), random_state=0, stratify=frame["target"]
    )
    train_path, test_path = ctx.artifact("train.csv"), ctx.artifact("test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return {"train": str(train_path), "test": str(test_path), "rows": len(train), "features": len(data.feature_names)}


def _fit(ctx: Context, model):
    import joblib
    import pandas as pd

    train = pd.read_csv(ctx.upstream["prep"]["train"])
    X, y = train.drop(columns=["target"]), train["target"]
    model.fit(X, y)
    path = ctx.artifact(f"{ctx.task_id}.joblib")
    joblib.dump(model, path)
    return {"model": str(path), "kind": type(model).__name__, "train_score": float(model.score(X, y))}


def train_logreg(ctx: Context) -> dict:
    from sklearn.linear_model import LogisticRegression

    return _fit(ctx, LogisticRegression(max_iter=ctx.params.get("max_iter", 500)))


def train_tree(ctx: Context) -> dict:
    from sklearn.tree import DecisionTreeClassifier

    return _fit(ctx, DecisionTreeClassifier(max_depth=ctx.params.get("max_depth", 3), random_state=0))


def select_best(ctx: Context) -> dict:
    """Score every candidate on the held-out split and keep the winner.

    Note how this task reads `ctx.upstream` for *both* trainers without
    knowing how many there were — the graph decided that, not this code.
    """
    import joblib
    import pandas as pd

    test = pd.read_csv(ctx.upstream["prep"]["test"])
    X, y = test.drop(columns=["target"]), test["target"]

    scored = []
    for task_id, up in sorted(ctx.upstream.items()):
        if not task_id.startswith("train_"):
            continue
        accuracy = float(joblib.load(up["model"]).score(X, y))
        scored.append({"task": task_id, "kind": up["kind"], "model": up["model"], "accuracy": accuracy})

    scored.sort(key=lambda row: row["accuracy"], reverse=True)
    ctx.artifact("metrics.json").write_text(json.dumps(scored, indent=2))
    best = scored[0]
    print(f"candidates: {[(s['task'], round(s['accuracy'], 4)) for s in scored]}")
    print(f"winner: {best['task']} @ {best['accuracy']:.4f}")
    return {"best": best, "candidates": scored}


def register(ctx: Context) -> dict:
    """The quality gate. Raising here is the *point*: a model that misses the
    bar must fail the run, not get promoted with a warning nobody reads."""
    import shutil
    from pathlib import Path

    best = ctx.upstream["select_best"]["best"]
    threshold = ctx.params.get("min_accuracy", 0.90)
    if best["accuracy"] < threshold:
        raise ValueError(f"accuracy {best['accuracy']:.4f} below threshold {threshold} — refusing to register")

    registry = Path(REGISTRY)
    registry.mkdir(parents=True, exist_ok=True)
    version = ctx.run_id
    target = registry / f"{version}.joblib"
    shutil.copy2(best["model"], target)
    (registry / "latest.json").write_text(
        json.dumps({"version": version, "path": str(target), **best}, indent=2)
    )
    print(f"registered {best['kind']} as {version}")
    return {"version": version, "path": str(target), "accuracy": best["accuracy"]}


with DAG("iris", schedule="@hourly", description="train + gate an iris classifier") as dag:
    t_prep = Task("prep", prep, params={"test_size": 0.25})
    t_logreg = Task("train_logreg", train_logreg, retries=1)
    t_tree = Task("train_tree", train_tree, retries=1)
    t_select = Task("select_best", select_best)
    t_register = Task("register", register, params={"min_accuracy": 0.90})

    t_prep >> [t_logreg, t_tree] >> t_select >> t_register
