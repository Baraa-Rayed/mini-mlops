"""The DAG model — the core abstraction every orchestrator is built on.

Maps to: Airflow's `airflow/models/dag.py` + `baseoperator.py`.

A DAG is just (1) a set of named tasks and (2) a set of edges between them.
Everything else an orchestrator does — scheduling, retries, distribution —
is bookkeeping on top of this graph.
"""

from __future__ import annotations

from typing import Callable, Iterable

# The DAG currently open in a `with DAG(...)` block. This is how Airflow lets
# you write tasks without passing `dag=` to every one of them.
_ACTIVE_DAG: "DAG | None" = None


class Task:
    """One unit of work.

    `fn` must be an importable module-level function taking a single
    `Context` argument. It has to be importable by *name* because a worker in
    another process / container / pod cannot receive a Python closure — it can
    only receive the string "pipelines.iris:prep" and import it itself.
    """

    def __init__(
        self,
        task_id: str,
        fn: Callable,
        retries: int = 0,
        retry_delay: float = 1.0,
        params: dict | None = None,
    ):
        self.task_id = task_id
        self.fn = fn
        self.retries = retries
        self.retry_delay = retry_delay
        self.params = params or {}
        self.upstream: set[str] = set()
        self.downstream: set[str] = set()
        self.dag: DAG | None = None

        if _ACTIVE_DAG is not None:
            _ACTIVE_DAG.add_task(self)

    @property
    def ref(self) -> str:
        """'module:function' — the only thing a remote worker needs to run this."""
        return f"{self.fn.__module__}:{self.fn.__qualname__}"

    # --- dependency syntax: a >> b >> [c, d] -------------------------------
    def __rshift__(self, other):
        for task in _as_tasks(other):
            self.downstream.add(task.task_id)
            task.upstream.add(self.task_id)
        return other

    def __lshift__(self, other):
        for task in _as_tasks(other):
            task >> self
        return other

    # A plain list has no `>>`, so Python falls back to these on the Task side.
    # They are what make fan-in read as nicely as fan-out: `[a, b] >> c`.
    def __rrshift__(self, other):
        for task in _as_tasks(other):
            task >> self
        return self

    def __rlshift__(self, other):
        for task in _as_tasks(other):
            self >> task
        return self

    def __repr__(self) -> str:
        return f"<Task {self.task_id}>"


def _as_tasks(obj) -> list[Task]:
    return list(obj) if isinstance(obj, (list, tuple, set)) else [obj]


class DAG:
    """A named, scheduled graph of tasks.

    `schedule` is an interval, not a cron expression — deliberately. Cron
    parsing is incidental complexity; the interesting part is the loop that
    asks "is this DAG due?" (see scheduler.py).
        "@hourly" | "@daily" | "30s" | "5m" | None (manual trigger only)
    """

    def __init__(self, dag_id: str, schedule: str | None = None, description: str = ""):
        self.dag_id = dag_id
        self.schedule = schedule
        self.description = description
        self.tasks: dict[str, Task] = {}

    def add_task(self, task: Task) -> None:
        if task.task_id in self.tasks:
            raise ValueError(f"duplicate task_id {task.task_id!r} in DAG {self.dag_id!r}")
        task.dag = self
        self.tasks[task.task_id] = task

    def __enter__(self) -> "DAG":
        global _ACTIVE_DAG
        self._previous, _ACTIVE_DAG = _ACTIVE_DAG, self
        return self

    def __exit__(self, *exc) -> None:
        global _ACTIVE_DAG
        _ACTIVE_DAG = self._previous
        self.validate()

    def validate(self) -> None:
        """Reject edges to unknown tasks and cycles. A cycle means the graph
        can never finish — better to fail at parse time than at 3am."""
        for task in self.tasks.values():
            for dep in task.upstream | task.downstream:
                if dep not in self.tasks:
                    raise ValueError(f"{self.dag_id}: task {task.task_id!r} references unknown {dep!r}")
        self.topological_order()  # raises on a cycle

    def topological_order(self) -> list[str]:
        """Kahn's algorithm. Also our cycle detector: if we cannot drain the
        graph, whatever is left is part of a cycle."""
        indegree = {tid: len(t.upstream) for tid, t in self.tasks.items()}
        queue = sorted(tid for tid, n in indegree.items() if n == 0)
        order: list[str] = []
        while queue:
            tid = queue.pop(0)
            order.append(tid)
            for child in sorted(self.tasks[tid].downstream):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(self.tasks):
            stuck = sorted(set(self.tasks) - set(order))
            raise ValueError(f"{self.dag_id}: cycle detected among {stuck}")
        return order

    def ready_tasks(self, states: dict[str, str]) -> list[Task]:
        """Tasks whose every upstream has succeeded — i.e. what can run *now*.

        This one function is the scheduler's whole decision procedure. Airflow
        calls the same idea "dependency checking"; Argo Workflows calls it
        "node readiness".
        """
        from .state import SUCCESS, PENDING

        return [
            t
            for tid, t in sorted(self.tasks.items())
            if states.get(tid, PENDING) == PENDING
            and all(states.get(u) == SUCCESS for u in t.upstream)
        ]

    def ancestors(self, task_id: str) -> set[str]:
        """Every task that must succeed before `task_id` can run.

        Direct parents are not enough: `select_best` sits below the trainers
        but legitimately wants `prep`'s output too. Handing a task its whole
        ancestor set — and *only* that — keeps it deterministic. Airflow's
        XCom lets you pull from any task in the run, siblings included, which
        is exactly how you get a pipeline whose result depends on which of two
        parallel branches happened to finish first.
        """
        seen: set[str] = set()
        queue = list(self.tasks[task_id].upstream)
        while queue:
            tid = queue.pop()
            if tid in seen:
                continue
            seen.add(tid)
            queue.extend(self.tasks[tid].upstream)
        return seen

    def roots(self) -> Iterable[Task]:
        return (t for t in self.tasks.values() if not t.upstream)

    def __repr__(self) -> str:
        return f"<DAG {self.dag_id} tasks={len(self.tasks)} schedule={self.schedule}>"
