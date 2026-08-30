# mini-mlops

A workflow orchestrator small enough to read in one sitting — ~800 lines of
executable Python (1,500 counting the comments that explain it), using nothing
outside the standard library. The example pipelines need scikit-learn; the
orchestrator itself needs `sqlite3`, `subprocess` and `argparse`.

It is not a toy re-implementation of Airflow. It is an attempt to isolate the
handful of ideas that *every* orchestrator (Airflow, Argo Workflows, Prefect,
Dagster, Argo CD) is built from, and to write each one down once, plainly,
with the trade-offs named in the comments.

```
prep ──┬── train_logreg ──┬── select_best ── register
       └── train_tree   ──┘
```

```console
$ python -m mini run iris

triggering iris on the local executor

run iris__20260830T151123__d54025  [success]  in 1.8s
  prep               success              try=1    0.5s
  register           success              try=1    0.0s
  select_best        success              try=1    0.6s
  train_logreg       success              try=1    0.6s
  train_tree         success              try=1    0.6s
```

## The five ideas

Read the modules in this order. Each is one idea, and each maps onto something
you already have a name for.

| Module | Idea | In Airflow |
|---|---|---|
| [`mini/dag.py`](mini/dag.py) | What work exists, and in what order | `models/dag.py` |
| [`mini/state.py`](mini/state.py) | What has happened, durably | the metadata database |
| [`mini/runner.py`](mini/runner.py) | How one task is executed | `airflow tasks run` |
| [`mini/executors.py`](mini/executors.py) | *Where* that runner is placed | `executors/` |
| [`mini/scheduler.py`](mini/scheduler.py) | The loop tying the four together | `SchedulerJob` |

Plus two things built on top: [`mini/dagbag.py`](mini/dagbag.py) (discovering
DAGs by importing them) and [`mini/gitops.py`](mini/gitops.py) (git as the
source of truth — the Argo CD idea).

## The one design decision that matters

`runner.py` executes a single task, and its interface is a **command line**:

```bash
python -m mini.runner --ref pipelines.iris:train --context ctx.json --out out.json
```

That is the whole portability story. Because the contract is a command rather
than a function call, the identical runner can be placed three ways:

```python
LocalExecutor   subprocess.run([...])        # a process here
DockerExecutor  docker run image [...]       # a container here
K8sExecutor     Pod{ command: [...] }        # a container somewhere else
```

Nothing above that line changes when you move to a cluster. This is also why
a task is referenced as the string `"pipelines.iris:train"` and not as a
Python object: a worker in another container cannot receive a closure, but it
can import a name.

## Quickstart

```bash
pip install -r requirements.txt

python -m mini dags                # what DAGs exist
python -m mini graph iris          # tasks in dependency order
python -m mini run iris            # trigger and wait
python -m mini runs                # history
python -m mini logs <run_id> prep  # a task's stdout/stderr
python -m mini scheduler           # the scheduling loop
```

State lives in `~/.mini-mlops` (override with `MINI_HOME`): a SQLite database
and one directory per run holding that run's artifacts and logs.

## What each idea buys you

**The scheduler holds no state in memory.** Every pass re-reads task states
from SQLite. That single discipline is what separates an orchestrator from a
shell script with a for-loop — kill it mid-run and `python -m mini resume
<run_id>` picks up exactly where it stopped, re-running only what was still
in flight.

**Parallelism is a property of the graph.** `train_logreg` and `train_tree`
have no edge between them, so they run at the same time. Nobody passes a
flag; `dag.ready_tasks()` simply returns both.

**Failure has three states, not two.** A task downstream of a failure is not
`pending` and not `failed` — it is `upstream_failed`, meaning *unreachable*.
Saying so is what lets the run loop terminate instead of waiting forever on
preconditions that can never be met. Watch it happen:

```console
$ python -m mini run flaky

  flaky_fetch        success            try=3   ← retried until it worked
  always_fails       failed             try=2   ← retried, then gave up
  never_runs         upstream_failed    try=0   ← never even started
  transform          success            try=1   ← unrelated branch, unaffected
```

**A task that dies without unwinding still reports.** The runner writes its
result to a file; if that file is missing, the executor synthesises a failure.
This is the OOM-killer case, and it is why the scheduler never hangs.

## GitOps

`mini/gitops.py` reduces Argo CD to its core loop:

```python
desired = git rev-parse HEAD           # what the repo says
actual  = store.applied_revision(app)  # what we last ran
if desired != actual: sync
```

```bash
python -m mini gitops --repo git@github.com:you/pipelines.git --app prod sync
python -m mini gitops --repo ... --app prod serve   # poll forever
```

A revision becomes "applied" only after its DAGs succeed, so a broken commit
stays out-of-sync and is retried rather than being marked done and forgotten.

One subtle bug worth knowing about, fixed here and easy to hit anywhere:
CPython validates a cached `.pyc` against its source's mtime **and size**, at
one-second resolution. Two commits a second apart whose file lengths happen to
match will reuse the old bytecode — so the deploy reports success while
running the *previous* commit. Checking out a new revision is exactly when
that coincidence is likely, so `checkout()` purges `__pycache__` and workers
run with `PYTHONDONTWRITEBYTECODE=1`.

## Docker

```bash
docker build -t mini-mlops:latest .
python -m mini --executor docker run iris
```

The interesting problem is not Docker, it is *paths*: the scheduler and the
container disagree about what a directory is called. `DockerExecutor` mounts
the code at `/app` and the run directory at `/run`, then rewrites the context
to match. Kubernetes just spells that seam `volumeMounts`.

## Tests

```bash
python -m pytest tests/ -q     # 41 tests
```

The run tests spawn real subprocesses through the real executor. That is slow
on purpose — mocking the executor away would delete the only part of the
system capable of surprising us.

## What is deliberately missing

Cron parsing (intervals only — `30s`, `5m`, `@daily`), a web UI, backfills,
SLAs, pools, sensors, and a Kubernetes executor. Each is real work in a real
orchestrator and none of them change the five ideas above.
