"""Loading and calling the registered model — shared by the CLI and the API.

This exists so that `mini predict` and `POST /predict` cannot drift apart.
Two code paths that both "call the model" are two chances to decode a class
index differently, or for one to forget to record a trace, and then the
dashboard disagrees with the terminal about what the system did.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .tracking import DEFAULT_MODEL_NAME, latest_version, prediction_trace, tracking_uri


class NoModelRegistered(RuntimeError):
    """Raised when nothing has been promoted yet — a normal first-run state,
    not a failure, so callers can turn it into a 404 rather than a 500."""


@dataclass
class RegisteredModel:
    name: str
    version: str
    uri: str
    kind: str
    accuracy: str | None
    features: list[str]
    classes: list[str]
    source_run: str | None
    experiment: str
    model: Any = field(repr=False, default=None)


_CACHE: dict[tuple[str, str | None], RegisteredModel] = {}

# Swagger's "Try it out" pre-fills every optional string field with the
# literal "string". Treating that as "unset" is the difference between a
# newcomer's first click working and it returning an error about a model
# version named "string".
_PLACEHOLDERS = {"", "string", "null", "none"}


def _normalise_name(name: str | None) -> str | None:
    if name is None or str(name).strip().lower() in _PLACEHOLDERS:
        return None
    return str(name).strip()


def _normalise_version(version: str | None) -> str | None:
    if version is None or str(version).strip().lower() in _PLACEHOLDERS:
        return None
    return str(version).strip()


def load(name: str | None = None, version: str | None = None, cache: bool = True) -> RegisteredModel:
    """Resolve and load the registered model.

    Cached because deserialising is far slower than predicting: an API that
    reloaded per request would spend milliseconds on arithmetic and seconds on
    disk. Pass `cache=False` after promoting a new version.
    """
    name = _normalise_name(name) or DEFAULT_MODEL_NAME
    version = _normalise_version(version)
    key = (name, version)
    if cache and key in _CACHE:
        return _CACHE[key]

    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    if version is None:
        found = latest_version(name)
        if found is None:
            raise NoModelRegistered(
                f"no registered model {name!r} at {tracking_uri()} — run the iris pipeline first"
            )
        version, tags = found.version, dict(found.tags or {})
    else:
        from mlflow.tracking import MlflowClient

        try:
            found = MlflowClient().get_model_version(name, version)
        except Exception as exc:  # noqa: BLE001 — MlflowException and friends
            # Asking for a version that does not exist is a caller mistake, not
            # a server fault. Without this it escapes as a 500 with a stack
            # trace, which tells the caller nothing they can act on.
            raise NoModelRegistered(f"no version {version!r} of model {name!r}: {exc}") from None
        tags = dict(found.tags or {})

    uri = f"models:/{name}/{version}"
    model = mlflow.sklearn.load_model(uri)
    # Prefer what sklearn itself recorded over what we tagged: the estimator
    # is the authority on its own column order.
    columns = getattr(model, "feature_names_in_", None)
    features = list(columns) if columns is not None else json.loads(tags.get("features", "[]"))

    resolved = RegisteredModel(
        name=name,
        version=str(version),
        uri=uri,
        kind=tags.get("kind", type(model).__name__),
        accuracy=tags.get("accuracy"),
        features=features,
        classes=json.loads(tags.get("classes", "[]")),
        source_run=tags.get("mini.run_id"),
        experiment=tags.get("mini.dag_id") or "inference",
        model=model,
    )
    if cache:
        _CACHE[key] = resolved
    return resolved


def clear_cache() -> None:
    _CACHE.clear()


@dataclass
class Prediction:
    row: list[float]
    prediction: int
    label: str | None
    confidence: float | None


def predict(registered: RegisteredModel, rows: list[list[float]], trace: bool = True) -> list[Prediction]:
    """Score rows, decoding class indices and recording a trace.

    Raises ValueError on a row of the wrong width — caught before it reaches
    the model so the caller gets "expected 4 values, got 3" rather than a
    schema-enforcement stack trace.
    """
    import pandas as pd

    width = len(registered.features)
    for row in rows:
        if len(row) != width:
            raise ValueError(
                f"expected {width} values ({', '.join(registered.features)}), got {len(row)}: {row}"
            )

    # A DataFrame with the training column names, not a bare array: sklearn
    # matches by position but a silent column-order mismatch is a wrong answer
    # rather than an error.
    frame = pd.DataFrame(rows, columns=registered.features)
    raw = registered.model.predict(frame)
    probabilities = (
        registered.model.predict_proba(frame) if hasattr(registered.model, "predict_proba") else None
    )

    results = []
    for i, (row, value) in enumerate(zip(rows, raw)):
        index = int(value)
        results.append(Prediction(
            row=list(row),
            prediction=index,
            label=registered.classes[index] if index < len(registered.classes) else None,
            confidence=float(probabilities[i][index]) if probabilities is not None else None,
        ))

    if trace:
        _record(registered, rows, results)
    return results


def _record(registered: RegisteredModel, rows: list[list[float]], results: list[Prediction]) -> None:
    """Telemetry must never cost an answer: a broken tracking store is not a
    reason to fail a prediction the model already made."""
    try:
        with prediction_trace(registered.experiment, registered.uri) as span:
            span.set_inputs({"rows": rows, "features": registered.features})
            span.set_outputs({
                "predictions": [r.prediction for r in results],
                "labels": [r.label for r in results],
            })
    except Exception:  # noqa: BLE001
        pass
