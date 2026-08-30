"""The scheduler — the loop that turns a graph plus a database into a run.

Maps to: Airflow's `SchedulerJob` / `airflow/jobs/scheduler_job_runner.py`.

There are exactly two loops here, and it is worth keeping them separate in
your head:

    _run_loop()  "given a run in flight, what can I start right now?"
    serve()      "given the clock, which DAGs are due for a new run?"

The first is the interesting one. Its entire decision procedure is
`dag.ready_tasks(states)` — and `states` is read back from SQLite on every
pass, never cached. That is the discipline that makes the scheduler
restartable: it has no memory, so there is nothing to lose when it dies.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .dag import DAG, Task
from .dagbag import DagBag
from .executors import BaseExecutor, LocalExecutor
from .runner import Context
from .state import (
    FAILED,
    PENDING,
    RUNNING,
    SUCCESS,
    UPSTREAM_FAILED,
    Store,
)

_ALIASES = {"@hourly": 3600.0, "@daily": 86400.0, "@weekly": 604800.0, "@once": None}
_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_schedule(schedule: str | None) -> float | None:
    """'30s' | '5m' | '@daily' -> seconds. None means manual-trigger only.

    Deliberately not cron. Cron parsing is a solved, boring problem that would
    double this file's size and teach nothing about orchestration.
    """
    if schedule is None:
        return None
    if schedule in _ALIASES:
        return _ALIASES[schedule]
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", schedule.strip())
    if not match:
        raise ValueError(f"bad schedule {schedule!r}; use 30s / 5m / 2h / @hourly / @daily")
    return float(match.group(1)) * _UNITS[match.group(2)]


class Scheduler:
    def __init__(self, store: Store, executor: BaseExecutor | None = None, parallelism: int = 4):
        self.store = store
        self.executor = executor or LocalExecutor()
        self.parallelism = max(1, parallelism)

    # --- running one DAG ----------------------------------------------------
    def trigger(self, dag: DAG, trigger: str = "manual") -> str:
        """Create a run and drive it to a terminal state. Returns the run_id."""
        dag.validate()
        run = self.store.create_run(dag.dag_id, trigger=trigger)
        for task_id in dag.tasks:
            self.store.init_task(run["run_id"], task_id)
        return self._run_loop(dag, run["run_id"], Path(run["run_dir"]))

    def resume(self, dag: DAG, run_id: str) -> str:
        """Pick up a run abandoned by a dead scheduler.

        Airflow calls this "adopting orphaned task instances". Any task left
        RUNNING has no process behind it any more — the only honest thing to
        do is put it back to PENDING and let it start over. This is why tasks
        must be idempotent, and why that requirement is not a nicety.
        """
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(f"no such run {run_id!r}")
        for task_id, state in self.store.task_states(run_id).items():
            if state == RUNNING:
                self.store.set_task_state(run_id, task_id, PENDING, error=None)
        return self._run_loop(dag, run_id, Path(run["run_dir"]))

    def _run_loop(self, dag: DAG, run_id: str, run_dir: Path) -> str:
        in_flight: dict = {}
        with ThreadPoolExecutor(max_workers=self.parallelism) as pool:
            while True:
                states = self.store.task_states(run_id)  # re-read every pass: no cache, no drift
                for task in dag.ready_tasks(states):
                    if len(in_flight) >= self.parallelism:
                        break
                    self.store.set_task_state(run_id, task.task_id, RUNNING, started_at=time.time(), error=None)
                    in_flight[pool.submit(self._execute, dag, run_id, run_dir, task)] = task

                if not in_flight:
                    break  # nothing running and nothing runnable: the run is over
                done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
                for future in done:
                    task = in_flight.pop(future)
                    self._settle(dag, run_id, task, future.result())

        final = self.store.task_states(run_id)
        state = SUCCESS if all(s == SUCCESS for s in final.values()) else FAILED
        self.store.finish_run(run_id, state)
        return run_id

    def _execute(self, dag: DAG, run_id: str, run_dir: Path, task: Task) -> dict:
        """Run one task, retrying in place. Returns the last result envelope.

        Retrying inside the worker (rather than rescheduling the task, as
        Airflow does) costs us one thing: a scheduler restart mid-backoff
        restarts the attempt count. In exchange the retry policy stays five
        readable lines instead of a `next_retry_datetime` column.
        """
        results = self.store.results(run_id)
        upstream = {tid: results.get(tid) for tid in sorted(dag.ancestors(task.task_id))}
        ctx = Context(
            dag_id=dag.dag_id,
            run_id=run_id,
            task_id=task.task_id,
            run_dir=run_dir,
            params=task.params,
            upstream=upstream,
        )

        result: dict = {}
        for attempt in range(task.retries + 1):
            try_number = self.store.bump_try(run_id, task.task_id)
            log_path = run_dir / "logs" / f"{task.task_id}.{try_number}.log"
            result = self.executor.execute(task.ref, ctx, log_path)
            if result.get("ok"):
                return result
            if attempt < task.retries:
                time.sleep(task.retry_delay)
        return result

    def _settle(self, dag: DAG, run_id: str, task: Task, result: dict) -> None:
        if result.get("ok"):
            self.store.set_task_state(
                run_id, task.task_id, SUCCESS,
                finished_at=time.time(),
                result=json.dumps(result.get("result"), default=str),
                error=None,
            )
            return
        self.store.set_task_state(
            run_id, task.task_id, FAILED,
            finished_at=time.time(),
            error=result.get("error", "unknown error"),
        )
        self._fail_downstream(dag, run_id, task.task_id)

    def _fail_downstream(self, dag: DAG, run_id: str, task_id: str) -> None:
        """A failed task poisons everything reachable from it. Marking those
        UPSTREAM_FAILED rather than leaving them PENDING is what lets the run
        loop terminate instead of spinning on tasks that can never be ready."""
        states = self.store.task_states(run_id)
        queue, seen = list(dag.tasks[task_id].downstream), {task_id}
        while queue:
            tid = queue.pop()
            if tid in seen:
                continue
            seen.add(tid)
            if states.get(tid) == PENDING:
                self.store.set_task_state(run_id, tid, UPSTREAM_FAILED, finished_at=time.time(),
                                          error=f"upstream {task_id!r} failed")
            queue.extend(dag.tasks[tid].downstream)

    # --- the clock ----------------------------------------------------------
    def due(self, dag: DAG, now: float | None = None) -> bool:
        interval = parse_schedule(dag.schedule)
        last = self.store.last_run_at(dag.dag_id)
        if dag.schedule == "@once":
            return last is None
        if interval is None:
            return False  # unscheduled: manual trigger only
        return last is None or (now or time.time()) - last >= interval

    def tick(self, dagbag: DagBag) -> list[str]:
        """One pass over every DAG: trigger the due ones. Returns new run_ids.

        `mark_scheduled` is written *before* the run starts, on purpose. If we
        marked it after, a run that takes longer than its own interval would
        re-trigger itself the instant it finished, forever.
        """
        started = []
        for dag in dagbag:
            if not self.due(dag):
                continue
            now = time.time()
            self.store.mark_scheduled(dag.dag_id, now)
            started.append(self.trigger(dag, trigger="schedule"))
        return started

    def serve(self, dagbag: DagBag, interval: float = 5.0, max_ticks: int | None = None) -> None:
        """The daemon. Rescan the DAG folder each tick so edits land without a
        restart — the same reason Airflow re-parses its dags folder."""
        ticks = 0
        while max_ticks is None or ticks < max_ticks:
            dagbag.collect()
            for run_id in self.tick(dagbag):
                print(f"[scheduler] started {run_id}")
            ticks += 1
            if max_ticks is None or ticks < max_ticks:
                time.sleep(interval)
