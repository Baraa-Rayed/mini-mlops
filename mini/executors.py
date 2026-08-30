"""Executors — *where* the runner process gets placed.

Maps to: Airflow's `airflow/executors/` (LocalExecutor, CeleryExecutor,
KubernetesExecutor).

Everything in this file is a variation on a single theme. `runner.py` defined
the contract as a command line:

    python -m mini.runner --ref REF --context ctx.json --out out.json

An executor's only job is to decide what wraps that command:

    LocalExecutor   nothing            -> subprocess on this machine
    DockerExecutor  `docker run ...`   -> a container on this machine
    K8sExecutor     a Pod spec         -> a container on some machine (stage 4)

Note what is *absent*: none of them know what a DAG is, what a retry is, or
what ran before. They take one task and one context and report what happened.
That narrowness is why swapping them changes nothing else.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .runner import Context

# Where this checkout lives — the code a task needs to import.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ExecutorError(RuntimeError):
    pass


class BaseExecutor:
    """Shared plumbing: write the context, invoke *something*, read the result.

    Subclasses implement `build_command()` only.
    """

    name = "base"

    def __init__(self, import_root: Path | str | None = None):
        # Which checkout the task's code is imported from. Normally this one;
        # the GitOps sync points it at a git working copy instead.
        self.import_root = Path(import_root).resolve() if import_root else PROJECT_ROOT

    def build_command(self, ref: str, ctx_path: Path, out_path: Path, ctx: Context) -> list[str]:
        raise NotImplementedError

    def prepare_context(self, ctx: Context) -> Context:
        """Hook for executors that see different paths than we do (Docker)."""
        return ctx

    def execute(self, ref: str, ctx: Context, log_path: Path) -> dict:
        """Run one task to completion. Returns the runner's result envelope:
        {"ok": True, "result": ...} or {"ok": False, "error": ..., ...}.

        Never raises on *task* failure — a failing task is normal operation,
        not an exception. It raises only if the executor itself is broken.
        """
        work = Path(ctx.run_dir) / ".mini"
        work.mkdir(parents=True, exist_ok=True)
        ctx_path = work / f"{ctx.task_id}.ctx.json"
        out_path = work / f"{ctx.task_id}.out.json"
        out_path.unlink(missing_ok=True)

        ctx_path.write_text(json.dumps(self.prepare_context(ctx).to_dict(), indent=2))
        cmd = self.build_command(ref, ctx_path, out_path, ctx)

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            log.write(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n\n")
            log.flush()
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=self.import_root, env=self.env())

        if out_path.exists():
            return json.loads(out_path.read_text())
        # No result file: the process died before the runner could write one
        # (OOM kill, image pull failure, bad `--ref`). Synthesise a failure so
        # the scheduler sees a normal failed task rather than a hang.
        tail = log_path.read_text()[-2000:] if log_path.exists() else ""
        return {
            "ok": False,
            "error": f"{self.name} executor: process exited {proc.returncode} without writing a result",
            "traceback": tail,
        }

    def env(self) -> dict:
        """Task code lives in this checkout, so the child must be able to
        import it. Airflow solves this with a shared filesystem or a baked
        image; locally, PYTHONPATH is the honest equivalent."""
        env = os.environ.copy()
        # `mini` itself always comes from this checkout; only the *task* code
        # follows import_root. Keeping both on the path means a synced repo
        # can contain only pipelines, not a copy of the orchestrator.
        roots = [str(self.import_root), str(PROJECT_ROOT)]
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(filter(None, roots + [env.get("PYTHONPATH", "")])))
        # Never let a task import stale bytecode. CPython validates a .pyc by
        # the source's mtime *and size*, both stored at one-second resolution.
        # A git checkout that changes a file without changing its length,
        # within the same second as the last run, passes that check — and the
        # worker silently executes the previous commit. Writing no .pyc at all
        # costs a few milliseconds of parse time and removes the failure mode.
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env


class LocalExecutor(BaseExecutor):
    """Stage 1: a subprocess on this machine.

    A subprocess, not a function call, and that is the whole point. It gives
    us a real process boundary for free: a task that segfaults or calls
    `sys.exit()` kills itself, not the scheduler.
    """

    name = "local"

    def __init__(self, python: str | None = None, import_root: Path | str | None = None):
        super().__init__(import_root)
        self.python = python or sys.executable

    def build_command(self, ref, ctx_path, out_path, ctx) -> list[str]:
        return [self.python, "-m", "mini.runner", "--ref", ref,
                "--context", str(ctx_path), "--out", str(out_path)]


class DockerExecutor(BaseExecutor):
    """Stage 2: a container on this machine.

    The interesting problem here is not Docker, it is *paths*. The runner
    writes its result to a path the scheduler then reads, but the two now
    disagree about what that path is called. So we mount two directories at
    fixed locations and rewrite the context to use the container's names:

        <project root> -> /app     (the code)
        <run dir>      -> /run     (the artifacts + the result file)

    Every distributed executor has this same seam. Kubernetes just spells it
    `volumeMounts` instead.
    """

    name = "docker"

    CODE_MOUNT = "/app"
    RUN_MOUNT = "/run"
    SRC_MOUNT = "/mini-src"

    def __init__(self, image: str = "mini-mlops:latest", docker: str = "docker",
                 extra_args: list[str] | None = None, import_root: Path | str | None = None):
        super().__init__(import_root)
        self.image = image
        self.docker = docker
        self.extra_args = extra_args or []

    def prepare_context(self, ctx: Context) -> Context:
        inner = Context.from_dict(ctx.to_dict())
        inner.run_dir = Path(self.RUN_MOUNT)
        return inner

    def build_command(self, ref, ctx_path, out_path, ctx) -> list[str]:
        run_dir = Path(ctx.run_dir).resolve()
        # ctx_path/out_path live under run_dir, so they are already mounted;
        # translate them to their in-container names.
        rel_ctx = Path(ctx_path).resolve().relative_to(run_dir)
        rel_out = Path(out_path).resolve().relative_to(run_dir)

        # Task code at /app. When it comes from a git checkout, the
        # orchestrator itself still has to be importable, so mount it too.
        mounts = ["-v", f"{self.import_root}:{self.CODE_MOUNT}:ro"]
        pythonpath = [self.CODE_MOUNT]
        if self.import_root != PROJECT_ROOT:
            mounts += ["-v", f"{PROJECT_ROOT}:{self.SRC_MOUNT}:ro"]
            pythonpath.append(self.SRC_MOUNT)

        return [
            self.docker, "run", "--rm",
            *mounts,
            "-v", f"{run_dir}:{self.RUN_MOUNT}",
            "-w", self.CODE_MOUNT,
            "-e", f"PYTHONPATH={os.pathsep.join(pythonpath)}",
            *self.extra_args,
            self.image,
            "python", "-m", "mini.runner",
            "--ref", ref,
            "--context", f"{self.RUN_MOUNT}/{rel_ctx}",
            "--out", f"{self.RUN_MOUNT}/{rel_out}",
        ]


EXECUTORS = {"local": LocalExecutor, "docker": DockerExecutor}


def get_executor(name: str, **kwargs) -> BaseExecutor:
    try:
        return EXECUTORS[name](**kwargs)
    except KeyError:
        raise ExecutorError(f"unknown executor {name!r}; have {sorted(EXECUTORS)}") from None
