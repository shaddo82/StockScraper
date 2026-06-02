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
        ]
        self.runs = {
            "run-low": FakeRun({"f1": 0.52}),
            "run-high": FakeRun({"f1": 0.66}),
        }
        self.aliases = {"champion": self.versions[0]}
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
        alias="champion",
        metric_name="f1",
    )

    assert result.promoted is True
    assert result.best_version == "2"
    assert result.previous_version == "1"
    assert ("stock-direction-model", "previous", "1") in fake_client.set_alias_calls
    assert ("stock-direction-model", "champion", "2") in fake_client.set_alias_calls
