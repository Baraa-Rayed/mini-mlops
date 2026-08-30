"""The task runner — the process that actually executes one task.

Maps to: Airflow's `airflow tasks run` command; Argo Workflows' init/wait
sidecar contract.

THE KEY IDEA OF THIS PROJECT LIVES HERE.

The runner is a plain CLI that takes "which function, with what inputs" and
writes "what happened" to a file. It knows nothing about schedulers, queues,
Docker or Kubernetes:

    python -m mini.runner --ref pipelines.iris:train --context ctx.json --out out.json

Because the contract is a *command line*, the exact same runner can be placed
in three different ways, and that is the entire local -> Docker -> Kubernetes
journey:

    LocalExecutor  : subprocess.run([...])                  (stage 1)
    DockerExecutor : docker run image  [...]                (stage 2)
    K8sExecutor    : Pod{ command: [...] }                  (stage 4)

Nothing above this line changes when you move to a cluster. That is what
"portable orchestration" actually means.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Context:
    """What a task is handed. Airflow's is a fat dict; ours is four fields."""

    dag_id: str
    run_id: str
    task_id: str
    run_dir: Path
    params: dict = field(default_factory=dict)
    upstream: dict = field(default_factory=dict)  # {ancestor task_id: its return value}

    def artifact(self, name: str) -> Path:
        """Path for an output file. Every run gets its own directory, so runs
        can never overwrite each other's artifacts."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir / name

    @classmethod
    def from_dict(cls, d: dict) -> "Context":
        return cls(
            dag_id=d["dag_id"],
            run_id=d["run_id"],
            task_id=d["task_id"],
            run_dir=Path(d["run_dir"]),
            params=d.get("params", {}),
            upstream=d.get("upstream", {}),
        )

    def to_dict(self) -> dict:
        return {
            "dag_id": self.dag_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "run_dir": str(self.run_dir),
            "params": self.params,
            "upstream": self.upstream,
        }


def load(ref: str):
    """'pipelines.iris:train' -> the actual function object."""
    module_name, _, func_name = ref.partition(":")
    module = importlib.import_module(module_name)
    fn = module
    for part in func_name.split("."):
        fn = getattr(fn, part)
    return fn


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mini.runner")
    ap.add_argument("--ref", required=True, help="module:function to execute")
    ap.add_argument("--context", required=True, help="path to context JSON")
    ap.add_argument("--out", required=True, help="path to write result JSON")
    args = ap.parse_args(argv)

    ctx = Context.from_dict(json.loads(Path(args.context).read_text()))
    out = Path(args.out)

    try:
        result = load(args.ref)(ctx)
        # The return value is persisted and handed to downstream tasks, so it
        # must be JSON-serialisable. Return references (paths, metrics), never
        # a 4GB DataFrame — the same rule Airflow's XCom enforces.
        out.write_text(json.dumps({"ok": True, "result": result}, default=str))
        return 0
    except Exception as exc:  # noqa: BLE001 - the runner is the failure boundary
        traceback.print_exc()
        out.write_text(
            json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
