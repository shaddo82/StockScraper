"""Promote the best MLflow model version to the champion alias."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Optional

from app import config

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover - optional dependency
    mlflow = None
    MlflowClient = None


@dataclass
class PromotionResult:
    promoted: bool
    model_name: str
    alias: str
    metric_name: str
    best_version: Optional[str]
    best_score: Optional[float]
    previous_version: Optional[str]
    previous_score: Optional[float]


def _get_version_metric(client: MlflowClient, version, metric_name: str) -> Optional[float]:
    run = client.get_run(version.run_id)
    metric = run.data.metrics.get(metric_name)
    return float(metric) if metric is not None else None


def _list_model_versions(client: MlflowClient, model_name: str):
    return client.search_model_versions(f"name = '{model_name}'")


def promote_best_model(
    model_name: str = config.MODEL_REGISTRY_NAME,
    alias: str = "champion",
    metric_name: str = "f1",
    min_improvement: float = 0.0,
) -> PromotionResult:
    if mlflow is None or MlflowClient is None:
        raise RuntimeError("mlflow is not installed")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    versions = list(_list_model_versions(client, model_name))

    best_version = None
    best_score = None
    for version in versions:
        score = _get_version_metric(client, version, metric_name)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_version = version
            best_score = score

    previous_version = None
    previous_score = None
    try:
        previous_version = client.get_model_version_by_alias(model_name, alias)
        previous_score = _get_version_metric(client, previous_version, metric_name)
    except Exception:
        previous_version = None

    if best_version is None or best_score is None:
        return PromotionResult(
            promoted=False,
            model_name=model_name,
            alias=alias,
            metric_name=metric_name,
            best_version=None,
            best_score=None,
            previous_version=previous_version.version if previous_version else None,
            previous_score=previous_score,
        )

    should_promote = (
        previous_score is None
        or best_score >= previous_score + min_improvement
        or previous_version is None
    )

    if should_promote:
        if previous_version is not None and previous_version.version != best_version.version:
            client.set_registered_model_alias(
                model_name,
                "previous",
                previous_version.version,
            )
        client.set_registered_model_alias(model_name, alias, best_version.version)

    return PromotionResult(
        promoted=should_promote,
        model_name=model_name,
        alias=alias,
        metric_name=metric_name,
        best_version=best_version.version,
        best_score=best_score,
        previous_version=previous_version.version if previous_version else None,
        previous_score=previous_score,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote best MLflow model version")
    parser.add_argument("--model-name", default=config.MODEL_REGISTRY_NAME)
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--min-improvement", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = promote_best_model(
        model_name=args.model_name,
        alias=args.alias,
        metric_name=args.metric,
        min_improvement=args.min_improvement,
    )
    print(asdict(result))


if __name__ == "__main__":
    main()
