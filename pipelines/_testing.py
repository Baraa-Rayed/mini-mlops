"""Task functions used by the test suite.

Underscore-prefixed so the DagBag skips it — but still a normal importable
module, which is what matters: a task's `ref` has to resolve in a *fresh*
process, so test tasks cannot be closures defined inside a test function.
"""

from __future__ import annotations

from mini import Context


def ok(ctx: Context) -> dict:
    return {"task": ctx.task_id, "value": ctx.params.get("value", 1)}


def boom(ctx: Context) -> dict:
    raise RuntimeError(f"{ctx.task_id} failed on purpose")


def flaky(ctx: Context) -> dict:
    """Fails until its attempt counter reaches `succeed_on`.

    The counter lives on disk because each attempt is a separate process —
    an in-memory counter would reset every time and never succeed.
    """
    counter = ctx.artifact(f"{ctx.task_id}.attempts")
    attempts = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(attempts))
    if attempts < ctx.params.get("succeed_on", 2):
        raise ConnectionError(f"attempt {attempts} fails")
    return {"attempts": attempts}


def echo_upstream(ctx: Context) -> dict:
    return {"saw": sorted(ctx.upstream)}


def hard_exit(ctx: Context) -> dict:
    """Dies without unwinding, so the runner never gets to write a result.

    This is the case a plain `try/except` cannot cover — the OOM killer, a
    segfaulting native library, `kubectl delete pod`. The executor has to
    notice the missing result file and report a failure anyway.
    """
    import os

    os._exit(9)
