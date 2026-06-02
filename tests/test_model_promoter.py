from __future__ import annotations

from dataclasses import dataclass

from ml import model_promoter


@dataclass
class FakeVersion:
    version: str
    run_id: str


class FakeRun:
    def __init__(self, metrics):
        self.data = type("RunData", (), {"metrics": metrics})


class FakeClient:
    def __init__(self):
        self.versions = [
            FakeVersion(version="1", run_id="run-low"),
            FakeVersion(version="2", run_id="run-high"),
            FakeVersion(version="3", run_id="run-latest"),
        ]
        self.runs = {
            "run-low": FakeRun({"f1": 0.52}),
            "run-high": FakeRun({"f1": 0.66}),
            "run-latest": FakeRun({"f1": 0.61}),
        }
        self.aliases = {"champion": self.versions[1]}
        self.set_alias_calls = []

    def search_model_versions(self, query):
        return self.versions

    def get_run(self, run_id):
        return self.runs[run_id]

    def get_model_version_by_alias(self, model_name, alias):
        return self.aliases[alias]

    def set_registered_model_alias(self, model_name, alias, version):
        self.set_alias_calls.append((model_name, alias, version))


def test_promote_best_model_sets_champion_alias(monkeypatch):
    fake_client = FakeClient()
    monkeypatch.setattr(model_promoter.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(model_promoter, "MlflowClient", lambda: fake_client)

    result = model_promoter.promote_best_model(
        model_name="stock-direction-model",
        champion_alias="champion",
        challenger_alias="challenger",
        metric_name="f1",
    )

    assert result.promoted is False
    assert result.champion_version == "2"
    assert result.challenger_version == "3"
    assert ("stock-direction-model", "challenger", "3") in fake_client.set_alias_calls
    assert ("stock-direction-model", "champion", "3") not in fake_client.set_alias_calls


def test_promote_best_model_ignores_versions_before_champion(monkeypatch):
    fake_client = FakeClient()
    fake_client.aliases = {"champion": fake_client.versions[2]}
    fake_client.runs["run-latest"] = FakeRun({"f1": 0.71})
    monkeypatch.setattr(model_promoter.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(model_promoter, "MlflowClient", lambda: fake_client)

    result = model_promoter.promote_best_model(
        model_name="stock-direction-model",
        champion_alias="champion",
        challenger_alias="challenger",
        metric_name="f1",
    )

    assert result.promoted is False
    assert result.champion_version == "3"
    assert result.challenger_version is None
    assert fake_client.set_alias_calls == []
