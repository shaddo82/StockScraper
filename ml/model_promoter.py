"""Promote newer MLflow model versions to challenger/champion aliases."""
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
    champion_alias: str
    challenger_alias: str
    metric_name: str
    champion_version: Optional[str]
    champion_score: Optional[float]
    challenger_version: Optional[str]
    challenger_score: Optional[float]


def _get_version_metric(client: MlflowClient, version, metric_name: str) -> Optional[float]:
    run = client.get_run(version.run_id)
    metric = run.data.metrics.get(metric_name)
    return float(metric) if metric is not None else None


def _list_model_versions(client: MlflowClient, model_name: str):
    return client.search_model_versions(f"name = '{model_name}'")


def _version_number(version) -> int:
    try:
        return int(version.version)
    except Exception:
        return -1


def _select_best_version(client: MlflowClient, versions, metric_name: str):
    best_version = None
    best_score = None
    for version in versions:
        score = _get_version_metric(client, version, metric_name)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_version = version
            best_score = score
    return best_version, best_score


def promote_best_model(
    model_name: str = config.MODEL_REGISTRY_NAME,
    champion_alias: str = "champion",
    challenger_alias: str = "challenger",
    metric_name: str = "f1",
    min_improvement: float = 0.0,
    promote_champion: bool = False,
) -> PromotionResult:
    if mlflow is None or MlflowClient is None:
        raise RuntimeError("mlflow is not installed")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    versions = sorted(
        list(_list_model_versions(client, model_name)),
        key=_version_number,
    )

    champion_version = None
    champion_score = None
    try:
        champion_version = client.get_model_version_by_alias(model_name, champion_alias)
        champion_score = _get_version_metric(client, champion_version, metric_name)
    except Exception:
        champion_version = None

    if champion_version is None:
        if not versions:
            return PromotionResult(
                promoted=False,
                model_name=model_name,
                champion_alias=champion_alias,
                challenger_alias=challenger_alias,
                metric_name=metric_name,
                champion_version=None,
                champion_score=None,
                challenger_version=None,
                challenger_score=None,
            )

        latest_version = versions[-1]
        latest_score = _get_version_metric(client, latest_version, metric_name)
        client.set_registered_model_alias(
            model_name,
            champion_alias,
            latest_version.version,
        )
        return PromotionResult(
            promoted=True,
            model_name=model_name,
            champion_alias=champion_alias,
            challenger_alias=challenger_alias,
            metric_name=metric_name,
            champion_version=latest_version.version,
            champion_score=latest_score,
            challenger_version=None,
            challenger_score=None,
        )

    champion_version_number = _version_number(champion_version)
    challenger_candidates = [
        version for version in versions if _version_number(version) > champion_version_number
    ]
    challenger_version, challenger_score = _select_best_version(
        client,
        challenger_candidates,
        metric_name,
    )

    if challenger_version is None or challenger_score is None:
        return PromotionResult(
            promoted=False,
            model_name=model_name,
            champion_alias=champion_alias,
            challenger_alias=challenger_alias,
            metric_name=metric_name,
            champion_version=champion_version.version,
            champion_score=champion_score,
            challenger_version=None,
            challenger_score=None,
        )

    client.set_registered_model_alias(
        model_name,
        challenger_alias,
        challenger_version.version,
    )

    should_promote = False
    if promote_champion and (
        champion_score is None
        or challenger_score >= champion_score + min_improvement
    ):
        client.set_registered_model_alias(
            model_name,
            champion_alias,
            challenger_version.version,
        )
        should_promote = True

    return PromotionResult(
        promoted=should_promote,
        model_name=model_name,
        champion_alias=champion_alias,
        challenger_alias=challenger_alias,
        metric_name=metric_name,
        champion_version=champion_version.version,
        champion_score=champion_score,
        challenger_version=challenger_version.version,
        challenger_score=challenger_score,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote best MLflow model version")
    parser.add_argument("--model-name", default=config.MODEL_REGISTRY_NAME)
    parser.add_argument("--alias", default="champion", help="Champion alias name")
    parser.add_argument(
        "--challenger-alias",
        default="challenger",
        help="Alias used for the newest evaluated candidate",
    )
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument(
        "--promote-champion",
        action="store_true",
        help="Also move champion when the challenger is better",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = promote_best_model(
        model_name=args.model_name,
        champion_alias=args.alias,
        challenger_alias=args.challenger_alias,
        metric_name=args.metric,
        min_improvement=args.min_improvement,
        promote_champion=args.promote_champion,
    )
    print(asdict(result))


if __name__ == "__main__":
    main()
