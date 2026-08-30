"""GitOps — git as the source of truth for what should be running.

Maps to: Argo CD's application controller; Flux's source + kustomize
controllers.

The idea is small enough to state in one sentence: **the desired state lives
in a git commit, the actual state lives in our database, and a loop closes the
gap.** Everything Argo CD does is elaboration on that.

    desired = git rev-parse HEAD           (what the repo says)
    actual  = store.applied_revision(app)  (what we last ran)
    if desired != actual: sync

Two properties fall out of this for free, and they are the reason anyone
bothers:

* **The deploy is auditable.** "Why did this model retrain at 3am?" has a
  commit hash as its answer, not a person's memory of clicking a button.
* **Rollback is `git revert`.** There is no second system holding deploy
  state that could disagree with the repo.

What we deliberately do *not* copy from Argo CD: pruning, health checks,
sync waves, and drift detection on live objects. Those matter when the thing
you are reconciling is a running cluster. Here the reconciled object is "which
pipeline code ran", which is much simpler.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .dagbag import DagBag
from .executors import BaseExecutor, LocalExecutor
from .scheduler import Scheduler
from .state import Store


class GitError(RuntimeError):
    pass


def git(*args: str, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
        env={"GIT_TERMINAL_PROMPT": "0", "PATH": __import__("os").environ.get("PATH", "")},
    )
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout.strip()


class GitOps:
    """One reconciled application: a repo, a branch, and a path within it.

    `app` names the reconciliation target, not the repo — the same repo synced
    on two branches is two applications with two independent applied
    revisions, which is exactly how you run a staging and a prod pipeline off
    one codebase.
    """

    def __init__(
        self,
        store: Store,
        repo: str,
        app: str = "default",
        branch: str = "main",
        path: str = "pipelines",
        workdir: Path | str | None = None,
        executor: BaseExecutor | None = None,
        parallelism: int = 4,
    ):
        self.store = store
        self.repo = repo
        self.app = app
        self.branch = branch
        self.path = path
        self.workdir = Path(workdir) if workdir else Path(store.home) / "gitops" / app
        self._executor = executor
        self.parallelism = parallelism

    # --- talking to git -----------------------------------------------------
    def checkout(self) -> Path:
        """Clone on first sync, fetch after. Returns the working copy.

        A hard reset rather than a merge, on purpose: the working copy is a
        cache of a commit, never somewhere anyone edits. If it has diverged,
        the repo is right and we are wrong.
        """
        if not (self.workdir / ".git").exists():
            if self.workdir.exists():
                shutil.rmtree(self.workdir)
            self.workdir.parent.mkdir(parents=True, exist_ok=True)
            git("clone", "--branch", self.branch, self.repo, str(self.workdir))
        else:
            git("fetch", "origin", self.branch, cwd=self.workdir)
            git("reset", "--hard", f"origin/{self.branch}", cwd=self.workdir)
        self._purge_bytecode()
        return self.workdir

    def _purge_bytecode(self) -> None:
        """Delete every __pycache__ in the working copy after moving commits.

        CPython decides a .pyc is current by comparing the source's mtime and
        size — at one-second resolution. Two commits a second apart whose file
        lengths happen to match will reuse the old bytecode, and the deploy
        runs the previous revision while reporting success. Checking out a new
        commit is exactly when that coincidence is likely, so the cache goes.
        """
        for cache in self.workdir.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)

    def desired_revision(self) -> str:
        return git("rev-parse", "HEAD", cwd=self.checkout())

    def applied_revision(self) -> str | None:
        return self.store.applied_revision(self.app)

    # --- the reconcile loop -------------------------------------------------
    def status(self) -> dict:
        desired = self.desired_revision()
        applied = self.applied_revision()
        return {
            "app": self.app,
            "repo": self.repo,
            "branch": self.branch,
            "desired": desired,
            "applied": applied,
            "in_sync": desired == applied,
        }

    def dagbag(self) -> DagBag:
        """Load DAGs *from the checkout*, not from this repo.

        `import_root` is the working copy, so a task's ref resolves against
        the synced commit's code — the point of the whole exercise.
        """
        return DagBag(self.workdir / self.path, import_root=self.workdir)

    def scheduler(self) -> Scheduler:
        executor = self._executor or LocalExecutor(import_root=self.workdir)
        return Scheduler(self.store, executor=executor, parallelism=self.parallelism)

    def sync(self, force: bool = False, only: list[str] | None = None) -> dict:
        """Reconcile once. Runs the repo's DAGs if the revision moved.

        Recording the revision *after* a successful sync (and not at all after
        a failed one) is what makes this converge: a broken commit stays
        out-of-sync and gets retried, instead of being marked done and
        forgotten.
        """
        state = self.status()
        if state["in_sync"] and not force:
            return {**state, "action": "none", "runs": []}

        bag = self.dagbag()
        if bag.errors:
            # Refuse to deploy a revision we cannot even parse. Argo CD calls
            # this a failed sync; the previous revision stays applied.
            return {**state, "action": "failed", "runs": [], "errors": bag.errors}

        scheduler = self.scheduler()
        targets = [d for d in bag if only is None or d.dag_id in only]
        runs = [scheduler.trigger(dag, trigger="gitops") for dag in targets]

        ok = all(self.store.get_run(r)["state"] == "success" for r in runs)
        if ok:
            self.store.set_applied_revision(self.app, state["desired"])
        return {
            **state,
            "action": "synced" if ok else "failed",
            "runs": runs,
            "applied": state["desired"] if ok else state["applied"],
            "in_sync": ok,
        }

    def serve(self, interval: float = 30.0, max_ticks: int | None = None) -> None:
        """Poll git forever. Argo CD defaults to three minutes; the mechanism
        is identical, and a webhook is only an optimisation on top of it."""
        ticks = 0
        while max_ticks is None or ticks < max_ticks:
            try:
                result = self.sync()
                if result["action"] != "none":
                    print(f"[gitops] {self.app}: {result['action']} {result['desired'][:8]} "
                          f"({len(result['runs'])} run(s))")
            except GitError as exc:
                # A network blip must not kill the controller.
                print(f"[gitops] {self.app}: {exc}")
            ticks += 1
            if max_ticks is None or ticks < max_ticks:
                time.sleep(interval)
