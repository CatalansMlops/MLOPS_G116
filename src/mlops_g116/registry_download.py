"""Download model artifacts from the W&B registry."""

import os
import shutil
from pathlib import Path

import wandb

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parents[2]


def download_and_load(
    entity: str,
    registry: str,
    collection: str,
    alias: str,
    artifact_dir: Path,
) -> Path:
    """Download a registry artifact and save the checkpoint locally.

    Args:
        entity: W&B entity name.
        registry: W&B registry name.
        collection: W&B collection name within the registry.
        alias: Artifact alias to resolve.
        artifact_dir: Directory to store the downloaded artifact.

    Returns:
        Path to the saved model checkpoint in the local models directory.
    """
    api = wandb.Api()
    artifact_name = f"{entity}/{registry}/{collection}:{alias}"
    artifact = api.artifact(artifact_name)
    local_dir = Path(artifact.download(root=str(artifact_dir)))
    model_path = local_dir / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Expected model.pth inside artifact, not found at {model_path}")
    output_dir = REPO_ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.pth"
    shutil.copy2(model_path, output_path)
    return output_path


def main() -> None:
    """Download a model artifact from W&B registry using environment defaults."""
    if load_dotenv is not None:
        load_dotenv()
    entity = os.getenv("WANDB_REGISTRY_ENTITY") or os.getenv("WANDB_ENTITY")
    if not entity:
        raise ValueError("Set WANDB_ENTITY to your W&B entity before running.")
    registry = os.getenv("WANDB_REGISTRY", "wandb-registry-mlops_g116")
    collection = os.getenv("WANDB_COLLECTION_MAIN", "mlops_g116-main-models")
    alias = os.getenv("WANDB_ALIAS", "latest")
    artifact_dir = Path(os.getenv("WANDB_ARTIFACT_DIR", "artifacts/registry_model"))
    output_path = download_and_load(entity, registry, collection, alias, artifact_dir)
    print(f"Saved model checkpoint to: {output_path}")


if __name__ == "__main__":
    main()
