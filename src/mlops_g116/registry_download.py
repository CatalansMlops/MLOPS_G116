import os
from pathlib import Path

import torch
import wandb

from mlops_g116.model import TumorDetectionModel

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


def download_and_load(
    entity: str,
    registry: str,
    collection: str,
    alias: str,
    artifact_dir: Path,
) -> Path:
    """Download a registry artifact and load its weights into the model."""
    api = wandb.Api()
    artifact_name = f"{entity}/{registry}/{collection}:{alias}"
    artifact = api.artifact(artifact_name)
    local_dir = Path(artifact.download(root=str(artifact_dir)))
    model = TumorDetectionModel()
    state_dict = torch.load(local_dir / "model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    return local_dir


def main() -> None:
    """Download a model artifact from W&B registry using environment defaults."""
    if load_dotenv is not None:
        load_dotenv()
    entity = os.getenv("WANDB_REGISTRY_ENTITY") or os.getenv("WANDB_ENTITY")
    if not entity:
        raise ValueError("Set WANDB_ENTITY to your W&B entity before running.")
    registry = os.getenv("WANDB_REGISTRY", "wandb-registry-mlops_g116")
    collection = os.getenv("WANDB_COLLECTION", "mlops_g116")
    alias = os.getenv("WANDB_ALIAS", "latest")
    artifact_dir = Path(os.getenv("WANDB_ARTIFACT_DIR", "artifacts/registry_model"))
    local_dir = download_and_load(entity, registry, collection, alias, artifact_dir)
    print(f"Loaded model from: {local_dir}")


if __name__ == "__main__":
    main()  
