"""GitOps tests, against a real git repository in a temp directory.

Mocking git here would be a mistake: the failure modes we care about — a
stale checkout, a cached module from the previous revision — only appear when
there really are two commits on disk.
"""

from __future__ import annotations

import textwrap

import pytest

from mini.executors import LocalExecutor
from mini.gitops import GitOps, git
from mini.state import SUCCESS, Store

PIPELINE = '''
from mini import DAG, Task, Context


def emit(ctx: Context) -> dict:
    return {{"marker": "{marker}"}}


with DAG("synced", description="from git") as dag:
    Task("emit", emit)
'''


@pytest.fixture
def origin(tmp_path):
    """A git repo containing a `pipelines/` folder with one DAG."""
    repo = tmp_path / "origin"
    (repo / "pipelines").mkdir(parents=True)
    git("init", "-q", "-b", "main", str(repo))
    git("config", "user.email", "test@example.com", cwd=repo)
    git("config", "user.name", "Test", cwd=repo)
    (repo / "pipelines" / "__init__.py").write_text("")
    write_pipeline(repo, "v1")
    commit(repo, "first")
    return repo


def write_pipeline(repo, marker):
    (repo / "pipelines" / "synced.py").write_text(textwrap.dedent(PIPELINE.format(marker=marker)))


def commit(repo, message):
    git("add", "-A", cwd=repo)
    git("commit", "-qm", message, cwd=repo)
    return git("rev-parse", "HEAD", cwd=repo)


@pytest.fixture
def app(tmp_path, origin):
    store = Store(tmp_path / "home")
    return GitOps(store, repo=str(origin), app="test", branch="main",
                  workdir=tmp_path / "checkout", executor=None)


def test_status_reports_out_of_sync_before_first_sync(app):
    state = app.status()
    assert state["applied"] is None
    assert state["in_sync"] is False
    assert len(state["desired"]) == 40


def test_sync_runs_the_dag_and_records_the_revision(app):
    result = app.sync()
    assert result["action"] == "synced"
    assert len(result["runs"]) == 1
    assert app.store.get_run(result["runs"][0])["state"] == SUCCESS
    assert app.store.get_run(result["runs"][0])["trigger"] == "gitops"
    assert app.applied_revision() == result["desired"]
    assert app.status()["in_sync"] is True


def test_second_sync_is_a_no_op(app):
    app.sync()
    result = app.sync()
    assert result["action"] == "none"
    assert result["runs"] == []


def test_force_syncs_an_unchanged_revision(app):
    app.sync()
    assert app.sync(force=True)["action"] == "synced"


def test_a_new_commit_runs_the_new_code(app, origin):
    """The test that justifies the module-eviction logic in DagBag: after a
    commit, the sync must run the *new* function, not a cached import."""
    first = app.sync()
    assert app.store.results(first["runs"][0])["emit"]["marker"] == "v1"

    write_pipeline(origin, "v2")
    commit(origin, "second")

    second = app.sync()
    assert second["action"] == "synced"
    assert second["desired"] != first["desired"]
    assert app.store.results(second["runs"][0])["emit"]["marker"] == "v2"


def test_broken_revision_does_not_become_applied(app, origin):
    app.sync()
    good = app.applied_revision()

    (origin / "pipelines" / "broken.py").write_text("this is not python(")
    commit(origin, "break it")

    result = app.sync()
    assert result["action"] == "failed"
    assert result["errors"]
    # The last good revision stays applied, so the next sync retries.
    assert app.applied_revision() == good
    assert app.status()["in_sync"] is False


def test_failing_dag_does_not_become_applied(app, origin):
    app.sync()
    good = app.applied_revision()

    (origin / "pipelines" / "synced.py").write_text(textwrap.dedent('''
        from mini import DAG, Task, Context


        def emit(ctx: Context) -> dict:
            raise RuntimeError("bad deploy")


        with DAG("synced") as dag:
            Task("emit", emit)
    '''))
    commit(origin, "ship a bug")

    result = app.sync()
    assert result["action"] == "failed"
    assert app.applied_revision() == good


def test_only_filters_which_dags_run(app, origin):
    (origin / "pipelines" / "other.py").write_text(textwrap.dedent('''
        from mini import DAG, Task, Context


        def noop(ctx: Context) -> dict:
            return {}


        with DAG("other") as dag:
            Task("noop", noop)
    '''))
    commit(origin, "add a second dag")

    result = app.sync(only=["other"])
    assert len(result["runs"]) == 1
    assert app.store.get_run(result["runs"][0])["dag_id"] == "other"


def test_local_edits_to_the_checkout_are_discarded(app):
    """The working copy is a cache of a commit, not a place to edit."""
    app.sync()
    stray = app.workdir / "pipelines" / "synced.py"
    stray.write_text("# vandalised\n")
    app.checkout()
    assert "vandalised" not in stray.read_text()


def test_changing_the_repo_url_retargets_an_existing_checkout(tmp_path, origin):
    """The workdir is keyed by app, not URL, so an existing checkout outlives
    a change of --repo. It must follow the new remote, not keep fetching the
    old one while reporting the new."""
    store = Store(tmp_path / "home")
    workdir = tmp_path / "shared"

    first = GitOps(store, repo=str(origin), app="a", branch="main", workdir=workdir)
    first.checkout()
    assert git("remote", "get-url", "origin", cwd=workdir) == str(origin)

    other = tmp_path / "other"
    git("clone", "-q", str(origin), str(other))
    second = GitOps(store, repo=str(other), app="a", branch="main", workdir=workdir)
    second.checkout()
    assert git("remote", "get-url", "origin", cwd=workdir) == str(other)


def test_serve_reports_a_heartbeat_when_in_sync(app, capsys):
    """A controller that prints only on change looks hung."""
    app.serve(interval=0, max_ticks=1)   # first tick syncs
    capsys.readouterr()
    app.serve(interval=30, max_ticks=1)  # second tick has nothing to do
    out = capsys.readouterr().out
    assert "in sync" in out
    assert "next check in 30s" in out


def test_serve_reconciles_then_stops(app, origin):
    app.serve(interval=0, max_ticks=1)
    assert app.status()["in_sync"] is True
    write_pipeline(origin, "v3")
    commit(origin, "third")
    app.serve(interval=0, max_ticks=1)
    assert app.status()["in_sync"] is True


def test_executor_import_root_points_at_the_checkout(app):
    app.checkout()
    assert app.scheduler().executor.import_root == app.workdir.resolve()


def test_supplied_executor_is_retargeted_at_the_checkout(tmp_path, origin):
    """The CLI always passes an executor, so a default-only retarget would
    leave real runs importing from the wrong checkout."""
    store = Store(tmp_path / "home2")
    supplied = LocalExecutor()  # built with import_root = this project
    app = GitOps(store, repo=str(origin), app="explicit", branch="main",
                 workdir=tmp_path / "checkout2", executor=supplied)
    result = app.sync()
    assert result["action"] == "synced"
    assert store.results(result["runs"][0])["emit"]["marker"] == "v1"


def test_switching_back_to_the_local_pipelines_still_works(tmp_path, origin):
    """Regression: evicting a stale parent package left its children cached.

    `reload()` of a submodule needs its parent in sys.modules, so dropping only
    `pipelines` made `pipelines.iris` permanently unloadable — every later scan
    failed with "parent 'pipelines' not in sys.modules" and the DagBag reported
    *no* DAGs rather than stale ones. Alternating between a checkout and the
    local folder is exactly what a sync followed by a local run does.
    """
    from mini.dagbag import DagBag

    assert {"iris", "flaky"} <= set(DagBag().dags)          # local
    GitOps(Store(tmp_path / "h"), repo=str(origin), app="flip",
           workdir=tmp_path / "co").sync()                   # checkout
    local = DagBag()                                         # local again
    assert not local.errors, local.errors
    assert {"iris", "flaky"} <= set(local.dags)
