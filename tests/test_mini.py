"""Tests for the orchestrator itself — not for any pipeline it runs.

They fall into two halves:

  * graph tests, which are pure and instant;
  * run tests, which spawn real subprocesses through the real executor.

The second half is slow on purpose. Mocking the executor away would delete
the only part of the system that can actually surprise us.
"""

from __future__ import annotations

import pytest

from mini.dag import DAG, Task
from mini.dagbag import DagBag
from mini.executors import LocalExecutor
from mini.runner import Context
from mini.scheduler import Scheduler, parse_schedule
from mini.state import FAILED, PENDING, RUNNING, SUCCESS, UPSTREAM_FAILED, Store
from pipelines import _testing as fixtures


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "home")


@pytest.fixture
def scheduler(store):
    return Scheduler(store, executor=LocalExecutor(), parallelism=4)


def linear_dag(dag_id="linear"):
    with DAG(dag_id) as dag:
        a = Task("a", fixtures.ok)
        b = Task("b", fixtures.ok)
        a >> b
    return dag


# --- the graph --------------------------------------------------------------
def test_topological_order_respects_edges():
    with DAG("diamond") as dag:
        a, b, c, d = (Task(t, fixtures.ok) for t in "abcd")
        a >> [b, c] >> d
    order = dag.topological_order()
    assert order.index("a") < order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")


def test_fan_in_syntax_builds_edges_both_ways():
    """`[a, b] >> c` relies on Task.__rrshift__ — a list has no `>>`."""
    with DAG("fan") as dag:
        a, b, c = (Task(t, fixtures.ok) for t in "abc")
        [a, b] >> c
    assert dag.tasks["c"].upstream == {"a", "b"}
    assert dag.tasks["a"].downstream == {"c"}


def test_cycle_is_rejected_at_definition_time():
    with pytest.raises(ValueError, match="cycle detected"):
        with DAG("cyclic") as dag:
            a, b = Task("a", fixtures.ok), Task("b", fixtures.ok)
            a >> b >> a


def test_duplicate_task_id_is_rejected():
    with pytest.raises(ValueError, match="duplicate task_id"):
        with DAG("dupes"):
            Task("a", fixtures.ok)
            Task("a", fixtures.ok)


def test_ready_tasks_gates_on_upstream_success():
    dag = linear_dag()
    assert [t.task_id for t in dag.ready_tasks({})] == ["a"]
    assert dag.ready_tasks({"a": RUNNING}) == []
    assert [t.task_id for t in dag.ready_tasks({"a": SUCCESS})] == ["b"]
    assert dag.ready_tasks({"a": FAILED}) == []


def test_ancestors_are_transitive():
    with DAG("deep") as dag:
        a, b, c = (Task(t, fixtures.ok) for t in "abc")
        a >> b >> c
    assert dag.ancestors("c") == {"a", "b"}
    assert dag.ancestors("a") == set()


def test_ref_is_importable_by_name():
    assert Task("a", fixtures.ok).ref == "pipelines._testing:ok"


@pytest.mark.parametrize(
    "text,expected",
    [("30s", 30.0), ("5m", 300.0), ("2h", 7200.0), ("1d", 86400.0), ("@hourly", 3600.0), (None, None)],
)
def test_parse_schedule(text, expected):
    assert parse_schedule(text) == expected


def test_parse_schedule_rejects_cron():
    with pytest.raises(ValueError):
        parse_schedule("*/5 * * * *")


# --- running ----------------------------------------------------------------
def test_successful_run_records_results(scheduler, store):
    run_id = scheduler.trigger(linear_dag())
    assert store.get_run(run_id)["state"] == SUCCESS
    assert store.task_states(run_id) == {"a": SUCCESS, "b": SUCCESS}
    assert store.results(run_id)["b"]["task"] == "b"


def test_task_receives_ancestor_results(scheduler, store):
    with DAG("ancestry") as dag:
        a = Task("a", fixtures.ok)
        b = Task("b", fixtures.ok)
        c = Task("c", fixtures.echo_upstream)
        a >> b >> c
    run_id = scheduler.trigger(dag)
    # 'a' is a grandparent, not a parent, and must still be visible.
    assert store.results(run_id)["c"]["saw"] == ["a", "b"]


def test_retries_until_success(scheduler, store):
    with DAG("retrying") as dag:
        Task("flaky", fixtures.flaky, retries=3, retry_delay=0.01, params={"succeed_on": 3})
    run_id = scheduler.trigger(dag)
    assert store.get_run(run_id)["state"] == SUCCESS
    assert store.task_states_full(run_id)["flaky"]["try_number"] == 3


def test_retries_are_bounded(scheduler, store):
    with DAG("giving_up") as dag:
        Task("flaky", fixtures.flaky, retries=1, retry_delay=0.01, params={"succeed_on": 99})
    run_id = scheduler.trigger(dag)
    assert store.get_run(run_id)["state"] == FAILED
    assert store.task_states_full(run_id)["flaky"]["try_number"] == 2  # 1 attempt + 1 retry


def test_failure_poisons_descendants_but_not_siblings(scheduler, store):
    with DAG("poison") as dag:
        root = Task("root", fixtures.ok)
        bad = Task("bad", fixtures.boom)
        child = Task("child", fixtures.ok)
        grandchild = Task("grandchild", fixtures.ok)
        sibling = Task("sibling", fixtures.ok)
        root >> [bad, sibling]
        bad >> child >> grandchild
    run_id = scheduler.trigger(dag)
    states = store.task_states(run_id)
    assert states == {
        "root": SUCCESS,
        "bad": FAILED,
        "child": UPSTREAM_FAILED,
        "grandchild": UPSTREAM_FAILED,  # poison is transitive
        "sibling": SUCCESS,             # an unrelated branch still completes
    }
    assert store.get_run(run_id)["state"] == FAILED


def test_unimportable_ref_becomes_a_failed_task(tmp_path):
    """A typo'd ref is a task failure, not a scheduler crash."""
    ctx = Context(dag_id="d", run_id="r", task_id="t", run_dir=tmp_path / "run")
    result = LocalExecutor().execute("no_such_module:fn", ctx, tmp_path / "t.log")
    assert result["ok"] is False
    assert "ModuleNotFoundError" in result["error"]


def test_killed_process_still_reports_a_failure(scheduler, store):
    """No result file, no unwinding, no traceback — and still a clean FAILED.

    Without this the scheduler would wait on a task that can never report.
    """
    with DAG("killed") as dag:
        Task("dies", fixtures.hard_exit)
    run_id = scheduler.trigger(dag)
    assert store.get_run(run_id)["state"] == FAILED
    assert "without writing a result" in store.task_states_full(run_id)["dies"]["error"]


def test_run_artifacts_are_isolated_per_run(scheduler, store):
    dag = linear_dag()
    first, second = scheduler.trigger(dag), scheduler.trigger(dag)
    assert store.get_run(first)["run_dir"] != store.get_run(second)["run_dir"]


# --- durability -------------------------------------------------------------
def test_resume_adopts_orphaned_running_tasks(scheduler, store):
    """Simulate a scheduler killed mid-run: 'b' is stuck RUNNING with no
    process behind it. Resume must reset and re-run it."""
    dag = linear_dag()
    run = store.create_run(dag.dag_id)
    for task_id in dag.tasks:
        store.init_task(run["run_id"], task_id)
    store.set_task_state(run["run_id"], "a", SUCCESS, result='{"task": "a", "value": 1}')
    store.set_task_state(run["run_id"], "b", RUNNING)

    scheduler.resume(dag, run["run_id"])
    assert store.task_states(run["run_id"]) == {"a": SUCCESS, "b": SUCCESS}
    assert store.get_run(run["run_id"])["state"] == SUCCESS


def test_completed_tasks_are_not_rerun_on_resume(scheduler, store):
    """The point of durable state: work already done stays done."""
    with DAG("resume_once") as dag:
        Task("flaky", fixtures.flaky, params={"succeed_on": 1})
    run = store.create_run(dag.dag_id)
    store.init_task(run["run_id"], "flaky")
    store.set_task_state(run["run_id"], "flaky", SUCCESS, try_number=1, result='{"attempts": 1}')

    scheduler.resume(dag, run["run_id"])
    assert store.task_states_full(run["run_id"])["flaky"]["try_number"] == 1  # not incremented


def test_state_survives_a_new_store_object(tmp_path):
    """The scheduler keeps nothing in memory, so a fresh Store sees everything."""
    home = tmp_path / "home"
    run_id = Scheduler(Store(home), executor=LocalExecutor()).trigger(linear_dag())
    assert Store(home).task_states(run_id) == {"a": SUCCESS, "b": SUCCESS}


# --- discovery and scheduling ----------------------------------------------
def test_dagbag_finds_the_example_pipelines():
    bag = DagBag()
    assert {"iris", "flaky"} <= set(bag.dags)
    assert not bag.errors
    assert bag.get("iris").tasks["register"].ref == "pipelines.iris:register"


def test_dagbag_skips_underscore_files():
    assert "_testing" not in {d.dag_id for d in DagBag()}


def test_due_respects_the_interval(scheduler, store):
    with DAG("every_hour", schedule="@hourly") as dag:
        Task("a", fixtures.ok)
    assert scheduler.due(dag) is True          # never run
    store.mark_scheduled(dag.dag_id, __import__("time").time())
    assert scheduler.due(dag) is False         # just ran
    store.mark_scheduled(dag.dag_id, __import__("time").time() - 3601)
    assert scheduler.due(dag) is True          # interval elapsed


def test_unscheduled_dags_never_fire(scheduler):
    assert scheduler.due(linear_dag()) is False


def test_once_fires_exactly_once(scheduler, store):
    with DAG("boot", schedule="@once") as dag:
        Task("a", fixtures.ok)
    assert scheduler.due(dag) is True
    store.mark_scheduled(dag.dag_id, 0.0)
    assert scheduler.due(dag) is False
