"""Tests for registry_download module."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import mlops_g116.registry_download as registry_download


class DummyArtifact:
    """Minimal W&B artifact stub."""

    def __init__(self, download_path: Path) -> None:
        self._download_path = download_path

    def download(self, root: str) -> str:
        """Return the artifact download path."""
        del root
        return str(self._download_path)


class DummyApi:
    """Minimal W&B API stub."""

    def __init__(self, artifact: DummyArtifact) -> None:
        self._artifact = artifact
        self.requested: list[str] = []

    def artifact(self, name: str) -> DummyArtifact:
        """Return the dummy artifact and capture the name."""
        self.requested.append(name)
        return self._artifact


def test_download_and_load_copies_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure download_and_load copies model.pth to the repo models directory."""
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir(parents=True)
    model_path = artifact_dir / "model.pth"
    model_path.write_text("weights", encoding="utf-8")

    dummy_api = DummyApi(DummyArtifact(artifact_dir))
    monkeypatch.setattr(registry_download.wandb, "Api", lambda: dummy_api)
    monkeypatch.setattr(registry_download, "REPO_ROOT", tmp_path)

    output_path = registry_download.download_and_load(
        entity="entity",
        registry="registry",
        collection="collection",
        alias="latest",
        artifact_dir=tmp_path / "downloads",
    )

    assert output_path == tmp_path / "models" / "model.pth"
    assert output_path.exists(), "Expected model.pth to be copied to repo models dir"
    assert dummy_api.requested == ["entity/registry/collection:latest"]


def test_download_and_load_raises_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure download_and_load raises when model.pth is missing."""
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir(parents=True)
    dummy_api = DummyApi(DummyArtifact(artifact_dir))
    monkeypatch.setattr(registry_download.wandb, "Api", lambda: dummy_api)
    monkeypatch.setattr(registry_download, "REPO_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="model.pth"):
        registry_download.download_and_load(
            entity="entity",
            registry="registry",
            collection="collection",
            alias="latest",
            artifact_dir=tmp_path / "downloads",
        )


def test_main_uses_env_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure main reads environment defaults and calls download_and_load."""
    called: dict[str, object] = {}

    def _fake_download_and_load(
        entity: str,
        registry: str,
        collection: str,
        alias: str,
        artifact_dir: Path,
    ) -> Path:
        called.update(
            {
                "entity": entity,
                "registry": registry,
                "collection": collection,
                "alias": alias,
                "artifact_dir": artifact_dir,
            }
        )
        return tmp_path / "models" / "model.pth"

    monkeypatch.setattr(registry_download, "download_and_load", _fake_download_and_load)
    monkeypatch.setattr(registry_download, "load_dotenv", lambda: None)
    monkeypatch.setenv("WANDB_ENTITY", "my-entity")
    monkeypatch.setenv("WANDB_REGISTRY_ENTITY", "")
    monkeypatch.setenv("WANDB_REGISTRY", "my-registry")
    monkeypatch.setenv("WANDB_COLLECTION_MAIN", "my-collection")
    monkeypatch.setenv("WANDB_ALIAS", "prod")
    monkeypatch.setenv("WANDB_ARTIFACT_DIR", str(tmp_path / "artifacts"))

    registry_download.main()

    assert called["entity"] == "my-entity"
    assert called["registry"] == "my-registry"
    assert called["collection"] == "my-collection"
    assert called["alias"] == "prod"
    assert called["artifact_dir"] == tmp_path / "artifacts"


def test_main_requires_entity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure main raises when no entity is configured."""
    monkeypatch.setattr(registry_download, "load_dotenv", lambda: None)
    monkeypatch.delenv("WANDB_REGISTRY_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)

    with pytest.raises(ValueError, match="WANDB_ENTITY"):
        registry_download.main()
