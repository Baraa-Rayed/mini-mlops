"""The command line — the only part of this project a user actually touches.

Maps to: the `airflow` CLI (`dags list`, `dags trigger`, `tasks logs`,
`scheduler`).

There is no logic in this file. Every command is three lines: build a DagBag,
build a Scheduler, call one method. If you find yourself wanting to add a
decision here, it probably belongs in scheduler.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .dagbag import DEFAULT_DAGS_DIR, DagBag
from .executors import get_executor
from .scheduler import Scheduler, parse_schedule
from .state import FAILED, SUCCESS, Store

# ANSI colour, but only when a human is watching.
_COLOURS = {SUCCESS: "\033[32m", FAILED: "\033[31m", "upstream_failed": "\033[33m",
            "running": "\033[36m", "pending": "\033[90m"}


def paint(state: str) -> str:
    if not sys.stdout.isatty():
        return state
    return f"{_COLOURS.get(state, '')}{state}\033[0m"


def pad(state: str, width: int) -> str:
    """Colour a state, then pad it to a *visible* width.

    `f"{paint(s):<10}"` counts the ANSI escape bytes as characters, so on a
    terminal the padding is consumed by codes nobody can see and every
    following column shifts left. Pad against the bare text instead.
    """
    return paint(state) + " " * max(0, width - len(state))


def ago(ts: float | None) -> str:
    if not ts:
        return "-"
    delta = time.time() - ts
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if delta >= size:
            return f"{delta / size:.0f}{unit} ago"
    return f"{delta:.0f}s ago"


def _bag(args) -> DagBag:
    bag = DagBag(args.dags)
    for path, error in bag.errors.items():
        print(f"\033[31mfailed to import {path}\033[0m\n{error}", file=sys.stderr)
    return bag


def _scheduler(args) -> Scheduler:
    executor = get_executor(args.executor, **({"image": args.image} if args.executor == "docker" else {}))
    return Scheduler(Store(args.home), executor=executor, parallelism=args.parallelism)


# --- commands ---------------------------------------------------------------
def cmd_dags(args) -> int:
    bag = _bag(args)
    if not len(bag):
        print(f"no DAGs under {bag.dags_dir}")
        return 1
    print(f"{'DAG':<24} {'SCHEDULE':<10} {'TASKS':<6} DESCRIPTION")
    for dag in bag:
        print(f"{dag.dag_id:<24} {str(dag.schedule or '-'):<10} {len(dag.tasks):<6} {dag.description}")
    return 0


def cmd_graph(args) -> int:
    dag = _bag(args).get(args.dag_id)
    print(f"{dag.dag_id}  (schedule={dag.schedule or 'manual'})\n")
    for task_id in dag.topological_order():
        task = dag.tasks[task_id]
        deps = ", ".join(sorted(task.upstream)) or "-"
        print(f"  {task_id:<18} <- {deps:<28} {task.ref}")
    return 0


def cmd_run(args) -> int:
    dag = _bag(args).get(args.dag_id)
    scheduler = _scheduler(args)
    print(f"triggering {dag.dag_id} on the {args.executor} executor")
    run_id = scheduler.trigger(dag, trigger="manual")
    return _report(scheduler.store, run_id)


def cmd_resume(args) -> int:
    scheduler = _scheduler(args)
    run = scheduler.store.get_run(args.run_id)
    if run is None:
        print(f"no such run {args.run_id}", file=sys.stderr)
        return 1
    dag = _bag(args).get(run["dag_id"])
    run_id = scheduler.resume(dag, args.run_id)
    return _report(scheduler.store, run_id)


def _report(store: Store, run_id: str) -> int:
    run = store.get_run(run_id)
    took = f"{(run['finished_at'] or time.time()) - run['created_at']:.1f}s"
    print(f"\nrun {run_id}  [{paint(run['state'])}]  in {took}")
    for task_id, row in sorted(store.task_states_full(run_id).items()):
        took = f"{row['finished_at'] - row['started_at']:.1f}s" if row["started_at"] and row["finished_at"] else "-"
        line = f"  {task_id:<18} {pad(row['state'], 17)} try={row['try_number']} {took:>7}"
        print(line + (f"\n      {row['error']}" if row["error"] else ""))
    print(f"\nartifacts: {run['run_dir']}")
    return 0 if run["state"] == SUCCESS else 1


def cmd_runs(args) -> int:
    store = Store(args.home)
    rows = store.list_runs(args.dag_id, limit=args.limit)
    if not rows:
        print("no runs yet")
        return 0
    print(f"{'RUN':<44} {'DAG':<18} {'STATE':<16} {'TRIGGER':<9} CREATED")
    for row in rows:
        print(f"{row['run_id']:<44} {row['dag_id']:<18} {pad(row['state'], 16)} "
              f"{row['trigger']:<9} {ago(row['created_at'])}")
    return 0


def cmd_show(args) -> int:
    store = Store(args.home)
    if store.get_run(args.run_id) is None:
        print(f"no such run {args.run_id}", file=sys.stderr)
        return 1
    return _report(store, args.run_id)


def cmd_logs(args) -> int:
    run = Store(args.home).get_run(args.run_id)
    if run is None:
        print(f"no such run {args.run_id}", file=sys.stderr)
        return 1
    logs = sorted((Path(run["run_dir"]) / "logs").glob(f"{args.task_id}.*.log"))
    if not logs:
        print(f"no logs for task {args.task_id!r} in {args.run_id}", file=sys.stderr)
        return 1
    for path in logs if args.all_tries else logs[-1:]:
        print(f"===== {path.name} =====")
        print(path.read_text())
    return 0


def cmd_reap(args) -> int:
    """Mark runs abandoned by a dead scheduler.

    `resume` adopts orphaned *tasks* inside a run, but nothing closed out the
    run itself: kill a scheduler mid-flight and its row sits in `running`
    forever, so `mini runs` shows work that no process is doing.

    We have no heartbeat, so age is the proxy — a run older than the threshold
    and still `running` is presumed dead. That is a real limitation, not a
    detail: a genuinely long run gets reaped if you set the threshold too low,
    which is why nothing calls this automatically.
    """
    from .state import RUNNING

    store = Store(args.home)
    now = time.time()
    stale = [r for r in store.list_runs(limit=1000)
             if r["state"] == RUNNING and now - r["created_at"] > args.older_than]
    if not stale:
        print(f"no runs stuck in 'running' for more than {args.older_than:g}s")
        return 0

    for run in stale:
        tasks = store.task_states_full(run["run_id"])
        orphans = [t for t, row in tasks.items() if row["state"] == RUNNING]
        print(f"{'would reap' if args.dry_run else 'reaped'} {run['run_id']} "
              f"({ago(run['created_at'])}, {len(orphans)} task(s) mid-flight)")
        if args.dry_run:
            continue
        for task_id in orphans:
            store.set_task_state(run["run_id"], task_id, FAILED, finished_at=now,
                                 error="abandoned: the scheduler running this task died")
        store.finish_run(run["run_id"], FAILED)

    if not args.dry_run:
        print(f"\n{len(stale)} run(s) closed out. `mini resume <run_id>` re-drives one from where it stopped.")
    return 0


def local_hostnames() -> list[str]:
    """Every name and address this machine answers to.

    `socket.gethostname()` is not enough. A VPN such as Tailscale adds an
    interface in 100.64.0.0/10 (RFC 6598 shared address space) which is
    neither the machine's hostname nor inside the RFC 1918 ranges MLflow
    trusts by default — so browsing over the VPN is refused while the LAN
    address works. Enumerating the interfaces is what catches it.
    """
    import socket
    import subprocess

    names: list[str] = []
    try:
        hostname = socket.gethostname()
        names += [hostname, socket.getfqdn()]
        names += socket.gethostbyname_ex(hostname)[2]
    except OSError:
        pass
    try:
        proc = subprocess.run(["ip", "-o", "-4", "addr", "show"],
                              capture_output=True, text=True, timeout=5)
        for line in proc.stdout.splitlines():
            fields = line.split()
            if "inet" in fields:
                names.append(fields[fields.index("inet") + 1].split("/")[0])
    except (OSError, subprocess.SubprocessError):
        pass  # not Linux, or no iproute2 — --allow-host still works
    return list(dict.fromkeys(n for n in names if n))


def security_env(host: str, port: int, allow_host: list[str] | None,
                 base: dict) -> tuple[dict, list[str]]:
    """Relax MLflow's two security middlewares enough to serve a remote browser.

    Returns the environment to launch with, and the names it now trusts (empty
    when binding to loopback, where MLflow's defaults already work).

    Serving the UI anywhere but loopback trips *both* guards, and fixing one
    leaves a UI that loads and then fails on every write:

        HostValidationMiddleware  rejects an unrecognised `Host` header
        CORSBlockingMiddleware    rejects POST/PUT/DELETE on /api/ and
                                  /ajax-api/ unless `Origin` is localhost

    A browser sends `Origin` even for same-origin POSTs, so any remote address
    counts as cross-origin no matter that it is the same server.

    Both variables *replace* MLflow's defaults rather than extending them, so
    the defaults are restated here — otherwise trusting a VPN address would
    silently break loopback.
    """
    env = dict(base)
    names = list(allow_host or [])
    if host in ("127.0.0.1", "localhost", "::1"):
        return env, []
    names += local_hostnames()
    names = list(dict.fromkeys(n for n in names if n))

    if not env.get("MLFLOW_SERVER_ALLOWED_HOSTS"):
        defaults = ["localhost", "127.0.0.1", "[::1]", "0.0.0.0",
                    "localhost:*", "127.0.0.1:*", "[[]::1]:*", "0.0.0.0:*",
                    "192.168.*", "10.*", *[f"172.{n}.*" for n in range(16, 32)]]
        allowed = [f"{n}:*" for n in names] + names + defaults
        env["MLFLOW_SERVER_ALLOWED_HOSTS"] = ",".join(dict.fromkeys(allowed))

    if not env.get("MLFLOW_SERVER_CORS_ALLOWED_ORIGINS"):
        origins: list[str] = []
        for name in names:
            origins += [f"http://{name}:{port}", f"https://{name}:{port}",
                        f"http://{name}:*", f"https://{name}:*"]
        origins += ["http://localhost:*", "http://127.0.0.1:*"]
        env["MLFLOW_SERVER_CORS_ALLOWED_ORIGINS"] = ",".join(dict.fromkeys(origins))

    return env, names


def cmd_ui(args) -> int:
    """Open the MLflow dashboard against *our* store.

    Bare `mlflow ui` reads ./mlruns in whatever directory you happen to be in,
    which is empty and looks like nothing ever ran. Pointing it at the same
    backend the pipelines write to is the whole value of this command.
    """
    import subprocess

    from .tracking import artifact_root, tracking_uri

    # MLflow guards the server with TWO independent middlewares, and serving
    # the UI on anything but loopback trips both:
    #
    #   HostValidationMiddleware  rejects an unrecognised `Host` header
    #                             ("possible DNS rebinding attack detected")
    #   CORSBlockingMiddleware    rejects POST/PUT/DELETE on /api/ and
    #                             /ajax-api/ when `Origin` is not localhost
    #
    # Fixing only the first gets you a UI that loads and then fails on every
    # write — the browser sends an Origin header even for same-origin POSTs,
    # so a remote address is treated as cross-origin regardless.
    #
    # Both env vars *replace* MLflow's defaults rather than extending them,
    # so the defaults are restated here or loopback access breaks.
    #
    # Note /health and /version are exempt from host validation, so a health
    # probe succeeds against a server no browser can use.
    env, names = security_env(args.host, args.port, args.allow_host, os.environ.copy())
    remote = bool(names)

    # `sys.executable -m mlflow`, not the bare `mlflow` console script: the
    # script only exists on PATH if this environment's bin directory happens
    # to be active, which is not true when the orchestrator is invoked by
    # absolute interpreter path, from a venv, or by the scheduler. Going
    # through the running interpreter guarantees the same environment that
    # wrote the runs is the one that reads them.
    cmd = [sys.executable, "-m", "mlflow", "ui",
           "--backend-store-uri", tracking_uri(),
           "--default-artifact-root", artifact_root(),
           "--host", args.host, "--port", str(args.port)]
    print(f"tracking  {tracking_uri()}")
    print(f"artifacts {artifact_root()}")
    if remote:
        print("reachable " + f"http://127.0.0.1:{args.port}")
        for host in names:
            print(f"          http://{host}:{args.port}")
    else:
        print(f"opening   http://{args.host}:{args.port}")
    print("          ctrl-c to stop\n")
    try:
        return subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        return 0


def cmd_serve(args) -> int:
    """Serve the registered model over HTTP, so it can be called by anything.

    `mini predict` is fine for a human at a terminal, but a model reachable
    only from one shell on one machine is not deployed. MLflow ships a scoring
    server; this wires it to our registry and fixes the two things that make
    the raw command fail here.
    """
    import subprocess

    from .tracking import DEFAULT_MODEL_NAME, latest_version, tracking_uri

    name = args.model or DEFAULT_MODEL_NAME
    version = args.version
    if version is None:
        found = latest_version(name)
        if found is None:
            print(f"no registered model {name!r} — run `mini run iris` first", file=sys.stderr)
            return 1
        version = found.version

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = tracking_uri()
    # The scoring server shells out to `bash -c 'exec uvicorn ...'`, and a bare
    # `uvicorn` resolves against PATH — which may belong to a different
    # environment that cannot import mlflow, failing with a bare
    # ModuleNotFoundError. Put this interpreter's bin directory first.
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])

    uri = f"models:/{name}/{version}"
    print(f"serving   {uri}")
    print(f"tracking  {tracking_uri()}")
    print(f"endpoint  http://{args.host}:{args.port}/invocations")
    print(f"health    http://{args.host}:{args.port}/ping")
    print("          ctrl-c to stop\n")
    cmd = [sys.executable, "-m", "mlflow", "models", "serve", "-m", uri,
           "--host", args.host, "--port", str(args.port),
           # `local` reuses this environment. The default rebuilds the model's
           # recorded environment from scratch, which is correct for a real
           # deployment and needlessly slow for poking at it locally.
           "--env-manager", args.env_manager]
    try:
        return subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        return 0


def cmd_predict(args) -> int:
    """Score a row with the model MLflow currently has registered.

    The counterpart to `register`: training is only half a pipeline, and a
    model nobody can call is indistinguishable from one that was never built.
    It resolves `models:/<name>/<version>` from the registry rather than a
    path you supply, so what answers here is by construction what the quality
    gate approved and promoted.
    """
    from .tracking import DEFAULT_MODEL_NAME, latest_version, tracking_uri

    name = args.model or DEFAULT_MODEL_NAME
    try:
        version = latest_version(name)
    except Exception as exc:  # noqa: BLE001 — a missing store is a normal first-run state
        print(f"cannot reach the MLflow registry at {tracking_uri()}: {exc}", file=sys.stderr)
        return 1
    if version is None:
        print(f"no registered model {name!r} — run `mini run iris` first", file=sys.stderr)
        return 1

    import mlflow
    import pandas as pd

    mlflow.set_tracking_uri(tracking_uri())
    tags = version.tags or {}
    # Schema travels with the model: the signature carries column names, and
    # the class labels ride along as a tag so a bare index never reaches a user.
    model = mlflow.sklearn.load_model(f"models:/{name}/{version.version}")
    signature_inputs = getattr(getattr(model, "feature_names_in_", None), "tolist", lambda: None)()
    features = signature_inputs or json.loads(tags.get("features", "[]"))
    classes = json.loads(tags.get("classes", "[]"))

    if args.show:
        print(f"model     {name} v{version.version}  ({tags.get('kind', 'unknown')})")
        print(f"accuracy  {tags.get('accuracy', '?')}")
        print(f"source    run {tags.get('mini.run_id', '?')}")
        print(f"features  {', '.join(features)}")
        print(f"classes   {', '.join(classes)}")
        print(f"uri       models:/{name}/{version.version}")
        return 0

    rows = []
    for raw in args.values:
        parts = [p for p in raw.replace(",", " ").split() if p]
        if len(parts) != len(features):
            print(f"expected {len(features)} values ({', '.join(features)}), got {len(parts)}: {raw!r}",
                  file=sys.stderr)
            return 1
        rows.append([float(p) for p in parts])
    # A DataFrame with the training column names, not a bare array: sklearn
    # matches features by position but warns when the names go missing, and
    # a silent column-order mismatch is a wrong answer rather than an error.
    frame = pd.DataFrame(rows, columns=features)
    predictions = model.predict(frame)

    probabilities = model.predict_proba(frame) if hasattr(model, "predict_proba") else None

    if not args.no_trace:
        # Record the call so it shows up in the dashboard's Traces tab. Without
        # this, inference is invisible: the tracking store knows how the model
        # was built and nothing about how it is used.
        from .tracking import prediction_trace

        experiment = tags.get("mini.dag_id") or "inference"
        try:
            with prediction_trace(experiment, f"models:/{name}/{version.version}") as span:
                span.set_inputs({"rows": rows, "features": features})
                span.set_outputs({
                    "predictions": [int(p) for p in predictions],
                    "labels": [classes[int(p)] for p in predictions],
                })
        except Exception as exc:  # noqa: BLE001 — never fail a prediction over telemetry
            print(f"(trace not recorded: {exc})", file=sys.stderr)

    for i, (row, prediction) in enumerate(zip(rows, predictions)):
        label = classes[int(prediction)]
        line = f"{row} -> {label}"
        if probabilities is not None:
            best = probabilities[i][int(prediction)]
            line += f"  (confidence {best:.3f})"
        print(line)
    return 0


def _gitops(args):
    from .gitops import GitOps

    executor = get_executor(args.executor, **({"image": args.image} if args.executor == "docker" else {}))
    return GitOps(Store(args.home), repo=args.repo, app=args.app, branch=args.branch,
                  path=args.path, executor=executor, parallelism=args.parallelism)


def cmd_gitops_status(args) -> int:
    state = _gitops(args).status()
    print(f"app       {state['app']}")
    print(f"repo      {state['repo']} @ {state['branch']}")
    print(f"desired   {state['desired']}")
    print(f"applied   {state['applied'] or '(never synced)'}")
    print(f"status    {'in sync' if state['in_sync'] else 'OUT OF SYNC'}")
    return 0 if state["in_sync"] else 1


def cmd_gitops_sync(args) -> int:
    result = _gitops(args).sync(force=args.force, only=args.dag or None)
    if result["action"] == "none":
        print(f"already in sync at {result['desired'][:8]} — nothing to do")
        return 0
    for path, error in result.get("errors", {}).items():
        print(f"\033[31m{path}\033[0m\n{error}", file=sys.stderr)
    print(f"{result['action']} {result['desired'][:8]}: {len(result['runs'])} run(s)")
    for run_id in result["runs"]:
        print(f"  {run_id}")
    return 0 if result["action"] == "synced" else 1


def cmd_gitops_serve(args) -> int:
    print(f"[gitops] reconciling {args.app} from {args.repo}@{args.branch} every {args.interval}s")
    try:
        _gitops(args).serve(interval=args.interval, max_ticks=args.max_ticks)
    except KeyboardInterrupt:
        print("\n[gitops] stopped")
    return 0


def cmd_scheduler(args) -> int:
    bag = _bag(args)
    scheduler = _scheduler(args)
    for dag in bag:
        interval = parse_schedule(dag.schedule)
        print(f"  {dag.dag_id:<24} every {interval}s" if interval else f"  {dag.dag_id:<24} manual only")
    print(f"[scheduler] watching {bag.dags_dir}, tick={args.interval}s — ctrl-c to stop")
    try:
        scheduler.serve(bag, interval=args.interval, max_ticks=args.max_ticks)
    except KeyboardInterrupt:
        print("\n[scheduler] stopped")
    return 0


# --- wiring -----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mini", description="a very small MLOps orchestrator")
    ap.add_argument("--home", default=None, help="state directory (default $MINI_HOME or ~/.mini-mlops)")
    ap.add_argument("--dags", default=DEFAULT_DAGS_DIR, help="folder to scan for DAGs")
    ap.add_argument("--executor", default="local", choices=["local", "docker"])
    ap.add_argument("--image", default="mini-mlops:latest", help="image for the docker executor")
    ap.add_argument("--parallelism", type=int, default=4, help="max tasks running at once")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("dags", help="list discovered DAGs").set_defaults(func=cmd_dags)

    p = sub.add_parser("graph", help="show a DAG's tasks in dependency order")
    p.add_argument("dag_id")
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("run", help="trigger a DAG now and wait for it")
    p.add_argument("dag_id")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("resume", help="re-drive a run abandoned by a dead scheduler")
    p.add_argument("run_id")
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("runs", help="run history")
    p.add_argument("dag_id", nargs="?")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("show", help="task states for one run")
    p.add_argument("run_id")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("logs", help="stdout/stderr of one task")
    p.add_argument("run_id")
    p.add_argument("task_id")
    p.add_argument("--all-tries", action="store_true", help="show every attempt, not just the last")
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser("reap", help="close out runs abandoned by a dead scheduler")
    p.add_argument("--older-than", type=float, default=900.0,
                   help="only runs stuck in 'running' longer than this many seconds (default 900)")
    p.add_argument("--dry-run", action="store_true", help="list what would be reaped, change nothing")
    p.set_defaults(func=cmd_reap)

    p = sub.add_parser("ui", help="open the MLflow dashboard on this project's store")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to expose on the network")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--allow-host", action="append",
                   help="extra Host header to accept, e.g. a DNS name or reverse-proxy "
                        "domain (repeatable); MLflow 403s anything it does not recognise")
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("serve", help="serve the registered model over HTTP")
    p.add_argument("--model", default=None, help="registered model name")
    p.add_argument("--version", default=None, help="version to serve (default: newest)")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to expose on the network")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--env-manager", default="local", choices=["local", "virtualenv", "uv"],
                   help="'local' reuses this environment; the others rebuild the model's own")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("predict", help="score a row with the registered model")
    p.add_argument("values", nargs="*", help='one row per argument, e.g. "5.1,3.5,1.4,0.2"')
    p.add_argument("--model", default=None, help="registered model name")
    p.add_argument("--show", action="store_true", help="describe the registered model and exit")
    p.add_argument("--no-trace", action="store_true",
                   help="do not record this call in the MLflow Traces tab")
    p.set_defaults(func=cmd_predict)

    g = sub.add_parser("gitops", help="reconcile pipelines from a git repo")
    g.add_argument("--repo", required=True, help="git URL or local path")
    g.add_argument("--app", default="default", help="name for this reconciliation target")
    g.add_argument("--branch", default="main")
    g.add_argument("--path", default="pipelines", help="DAG folder within the repo")
    gsub = g.add_subparsers(dest="gitops_command", required=True)

    gsub.add_parser("status", help="compare git HEAD to the applied revision").set_defaults(func=cmd_gitops_status)

    gp = gsub.add_parser("sync", help="reconcile once")
    gp.add_argument("--force", action="store_true", help="sync even if already in sync")
    gp.add_argument("--dag", action="append", help="only this DAG (repeatable)")
    gp.set_defaults(func=cmd_gitops_sync)

    gp = gsub.add_parser("serve", help="poll git and reconcile forever")
    gp.add_argument("--interval", type=float, default=30.0)
    gp.add_argument("--max-ticks", type=int, default=None)
    gp.set_defaults(func=cmd_gitops_serve)

    p = sub.add_parser("scheduler", help="run the scheduling loop")
    p.add_argument("--interval", type=float, default=5.0, help="seconds between ticks")
    p.add_argument("--max-ticks", type=int, default=None, help="stop after N ticks (for tests/demos)")
    p.set_defaults(func=cmd_scheduler)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.home is None:
        from .state import HOME
        args.home = HOME
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
