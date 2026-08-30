"""mini-mlops — an orchestrator small enough to read in one sitting.

The whole system is five ideas, one per module:

    dag.py        what work exists, and in what order      (Airflow: models/dag.py)
    state.py      what has happened, durably               (Airflow: metadata DB)
    runner.py     how one task is executed                 (Airflow: `tasks run`)
    executors.py  *where* that runner is placed            (Airflow: executors/)
    scheduler.py  the loop that ties the four together     (Airflow: scheduler_job)

Read them in that order.
"""

__all__ = ["DAG", "Task", "Context"]
__version__ = "0.1.0"


def __getattr__(name):
    """Lazy re-export (PEP 562).

    Importing `.runner` eagerly here would put `mini.runner` in `sys.modules`
    before `python -m mini.runner` got to execute it as `__main__`, giving two
    copies of the module and a RuntimeWarning in every task log. Since the
    runner *is* our entry point, the package must not pull it in on import.
    """
    if name in ("DAG", "Task"):
        from . import dag

        return getattr(dag, name)
    if name == "Context":
        from .runner import Context

        return Context
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
