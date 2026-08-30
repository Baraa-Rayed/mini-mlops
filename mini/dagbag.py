"""DAG discovery — turning a folder of Python files into a registry of DAGs.

Maps to: Airflow's `airflow/models/dagbag.py`.

The trick every orchestrator uses is the same: DAG files are not configuration,
they are *programs*. We import them and then look at what module-level `DAG`
objects they left behind. That is why you can write `for i in range(10):` in a
DAG file and get ten tasks — the file is executed, not parsed.

The cost of that trick is real, and worth naming: importing a DAG file runs
arbitrary code, in this process, every time we scan. Airflow spent years
walling this off into a separate parsing process. We keep it simple and just
refuse to let one bad file take down the scan.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import traceback
from pathlib import Path

from .dag import DAG

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DAGS_DIR = PROJECT_ROOT / "pipelines"


class DagBag:
    """All DAGs found under a directory, keyed by dag_id.

    `import_root` is the directory that must be on `sys.path` for those files
    to import — normally the project root, but the GitOps sync points it at a
    git checkout instead. It matters because a task's `ref` is built from
    `fn.__module__`: get the root wrong and you produce refs like
    `iris:train` that no worker process can resolve.
    """

    def __init__(self, dags_dir: Path | str = DEFAULT_DAGS_DIR, import_root: Path | str | None = None):
        self.dags_dir = Path(dags_dir).resolve()
        if import_root is not None:
            self.import_root = Path(import_root).resolve()
        elif self.dags_dir == PROJECT_ROOT or PROJECT_ROOT in self.dags_dir.parents:
            self.import_root = PROJECT_ROOT
        else:
            # A folder outside this checkout: its parent is the best guess at
            # the package root, so `pipelines/iris.py` still imports as
            # `pipelines.iris` rather than a bare `iris`.
            self.import_root = self.dags_dir.parent
        self.dags: dict[str, DAG] = {}
        self.errors: dict[str, str] = {}
        self.collect()

    def collect(self) -> None:
        self.dags.clear()
        self.errors.clear()
        # Position 0, not append: if two checkouts both define `pipelines`,
        # the one we were asked for has to win.
        root = str(self.import_root)
        if sys.path and sys.path[0] != root:
            sys.path.insert(0, root)

        for path in sorted(self.dags_dir.rglob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                module = self._import(path)
            except Exception:  # noqa: BLE001 — one broken file must not hide the rest
                self.errors[self._label(path)] = traceback.format_exc()
                continue
            for obj in vars(module).values():
                if isinstance(obj, DAG):
                    if obj.dag_id in self.dags and self.dags[obj.dag_id] is not obj:
                        self.errors[self._label(path)] = f"duplicate dag_id {obj.dag_id!r}"
                        continue
                    self.dags[obj.dag_id] = obj

    def _label(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.import_root))
        except ValueError:
            return str(path)

    def _module_name(self, path: Path) -> str:
        return ".".join(path.relative_to(self.import_root).with_suffix("").parts)

    def _import(self, path: Path):
        """Import by dotted module name so that a task's `ref` — which is
        built from `fn.__module__` — is importable by a *worker* too. Loading
        the file anonymously would produce refs no other process could resolve.
        """
        module_name = self._module_name(path)
        self._evict_foreign(module_name)
        cached = sys.modules.get(module_name)
        if cached is not None:
            return importlib.reload(cached)
        return importlib.import_module(module_name)

    def _evict_foreign(self, module_name: str) -> None:
        """Drop cached modules that resolve to a *different* checkout.

        Without this, syncing a new git revision would silently re-serve the
        old code: `importlib.reload` reloads from the module's original
        `__file__`, and a cached parent package makes Python skip `sys.path`
        for its children entirely. Stale imports are the classic way a GitOps
        deploy reports success while running the previous commit.
        """
        parts = module_name.split(".")
        for depth in range(1, len(parts) + 1):
            name = ".".join(parts[:depth])
            module = sys.modules.get(name)
            if module is None:
                continue
            origin = getattr(module, "__file__", None) or ""
            paths = list(getattr(module, "__path__", []))  # packages
            locations = [Path(p).resolve() for p in ([origin] if origin else []) + paths]
            if locations and not any(
                loc == self.import_root or self.import_root in loc.parents for loc in locations
            ):
                del sys.modules[name]

    def get(self, dag_id: str) -> DAG:
        try:
            return self.dags[dag_id]
        except KeyError:
            known = ", ".join(sorted(self.dags)) or "(none)"
            raise KeyError(f"no DAG {dag_id!r} in {self.dags_dir}; found: {known}") from None

    def __iter__(self):
        return iter(sorted(self.dags.values(), key=lambda d: d.dag_id))

    def __len__(self) -> int:
        return len(self.dags)
