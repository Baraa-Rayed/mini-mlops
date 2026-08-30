"""A deliberately unreliable DAG — the one you run to watch things break.

    ingest ── flaky_fetch ──┬── transform ── publish
                            └── always_fails ── never_runs

Run it and read the state table:

    flaky_fetch    success            try=3   ← retried until it worked
    always_fails   failed             try=2   ← retried, then gave up
    never_runs     upstream_failed    try=0   ← never even started

That third line is the one that matters. A task downstream of a failure is not
pending and not failed — it is *unreachable*, and saying so is what lets the
scheduler declare the run over instead of waiting forever for a task whose
preconditions can never be met.
"""

from __future__ import annotations

import random
import time

from mini import DAG, Context, Task


def ingest(ctx: Context) -> dict:
    ctx.artifact("raw.txt").write_text("42\n")
    return {"path": str(ctx.artifact("raw.txt")), "records": 1}


def flaky_fetch(ctx: Context) -> dict:
    """Fails ~60% of the time. Because each retry is a fresh *process*, this
    is a fair simulation of the real thing: a transient network error, not an
    exception you could have caught in-line."""
    if random.random() < ctx.params.get("failure_rate", 0.6):
        raise ConnectionError("upstream API returned 503 (simulated)")
    time.sleep(0.2)
    return {"fetched": True}


def transform(ctx: Context) -> dict:
    value = int(open(ctx.upstream["ingest"]["path"]).read().strip())
    return {"value": value * 2}


def publish(ctx: Context) -> dict:
    print(f"publishing {ctx.upstream['transform']['value']}")
    return {"published": True}


def always_fails(ctx: Context) -> dict:
    raise RuntimeError("this task exists to fail — nothing is wrong")


def never_runs(ctx: Context) -> dict:
    raise AssertionError("unreachable: always_fails must poison this task")


with DAG("flaky", schedule=None, description="retries and upstream_failed, on purpose") as dag:
    t_ingest = Task("ingest", ingest)
    t_fetch = Task("flaky_fetch", flaky_fetch, retries=4, retry_delay=0.2, params={"failure_rate": 0.6})
    t_transform = Task("transform", transform)
    t_publish = Task("publish", publish)
    t_fails = Task("always_fails", always_fails, retries=1, retry_delay=0.1)
    t_never = Task("never_runs", never_runs)

    t_ingest >> t_fetch >> [t_transform, t_fails]
    t_transform >> t_publish
    t_fails >> t_never
