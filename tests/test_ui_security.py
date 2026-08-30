"""Tests for the MLflow UI's two security guards.

This is here because the same class of bug shipped twice: serving the UI to a
remote browser trips *two* independent middlewares, and fixing only the first
produces a dashboard that loads and then 403s on every write — which looks
like a working server until you click something.

The rules under test come from `mlflow/server/security_utils.py`:
host validation uses fnmatch against MLFLOW_SERVER_ALLOWED_HOSTS, and CORS
blocking rejects state-changing methods whose Origin is neither localhost nor
in MLFLOW_SERVER_CORS_ALLOWED_ORIGINS.
"""

from __future__ import annotations

import fnmatch

import pytest

from mini.cli import local_hostnames, security_env


def allows_host(env: dict, host: str) -> bool:
    patterns = [p.strip() for p in env["MLFLOW_SERVER_ALLOWED_HOSTS"].split(",")]
    return any(fnmatch.fnmatch(host, p) if "*" in p else host == p for p in patterns)


def allows_origin(env: dict, origin: str) -> bool:
    patterns = [p.strip() for p in env["MLFLOW_SERVER_CORS_ALLOWED_ORIGINS"].split(",")]
    return any(fnmatch.fnmatch(origin, p) if "*" in p else origin == p for p in patterns)


def test_loopback_binding_changes_nothing(monkeypatch):
    """MLflow's defaults already cover localhost; overriding them would only
    risk breaking what works."""
    monkeypatch.delenv("MLFLOW_SERVER_ALLOWED_HOSTS", raising=False)
    env, names = security_env("127.0.0.1", 5000, None, {})
    assert names == []
    assert "MLFLOW_SERVER_ALLOWED_HOSTS" not in env
    assert "MLFLOW_SERVER_CORS_ALLOWED_ORIGINS" not in env


def test_binding_publicly_sets_both_guards():
    """Setting only the host allowlist is the bug that shipped: the UI loads
    and every POST is refused."""
    env, names = security_env("0.0.0.0", 5000, None, {})
    assert names
    assert "MLFLOW_SERVER_ALLOWED_HOSTS" in env
    assert "MLFLOW_SERVER_CORS_ALLOWED_ORIGINS" in env


def test_loopback_still_allowed_after_relaxing():
    """Both env vars *replace* MLflow's defaults, so the defaults must be
    restated — otherwise trusting a VPN address breaks localhost."""
    env, _ = security_env("0.0.0.0", 5000, None, {})
    assert allows_host(env, "localhost:5000")
    assert allows_host(env, "127.0.0.1:5000")
    assert allows_host(env, "0.0.0.0:5000")


def test_a_vpn_address_is_trusted_on_both_guards():
    """Tailscale hands out 100.64.0.0/10 (RFC 6598), which is neither the
    hostname nor inside the RFC 1918 ranges MLflow trusts by default."""
    env, _ = security_env("0.0.0.0", 5000, ["100.117.17.38"], {})
    assert allows_host(env, "100.117.17.38:5000")
    assert allows_origin(env, "http://100.117.17.38:5000")


def test_an_explicit_extra_host_covers_a_proxy_domain():
    env, _ = security_env("0.0.0.0", 8080, ["mlflow.example.com"], {})
    assert allows_host(env, "mlflow.example.com:8080")
    assert allows_origin(env, "http://mlflow.example.com:8080")
    assert allows_origin(env, "https://mlflow.example.com:8080")


def test_unknown_origins_are_still_refused():
    """The relaxation is bounded. An arbitrary site must not be able to drive
    this server from a victim's browser — that is the attack the guard exists
    for, and 'expose it on 0.0.0.0' is not a reason to disable it."""
    env, _ = security_env("0.0.0.0", 5000, None, {})
    assert not allows_origin(env, "http://evil.example.com")
    assert not allows_host(env, "evil.example.com:5000")


def test_an_operator_override_is_never_clobbered():
    base = {"MLFLOW_SERVER_ALLOWED_HOSTS": "only.this.host",
            "MLFLOW_SERVER_CORS_ALLOWED_ORIGINS": "http://only.this.host"}
    env, _ = security_env("0.0.0.0", 5000, ["other.host"], base)
    assert env["MLFLOW_SERVER_ALLOWED_HOSTS"] == "only.this.host"
    assert env["MLFLOW_SERVER_CORS_ALLOWED_ORIGINS"] == "http://only.this.host"


def test_local_hostnames_finds_this_machine():
    names = local_hostnames()
    assert names
    assert any(n == "127.0.0.1" or n.startswith("127.") for n in names)


@pytest.mark.parametrize("host", ["localhost", "::1"])
def test_other_loopback_spellings_also_skip_the_override(host):
    _, names = security_env(host, 5000, None, {})
    assert names == []
