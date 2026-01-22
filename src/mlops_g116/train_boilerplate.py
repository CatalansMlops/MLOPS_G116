import os
import shutil
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import wandb
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

from mlops_g116.data import load_data

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> dict[str, float]:
    """Compute evaluation metrics for a model over a dataloader.

    Args:
        model: Trained model.
        dataloader: DataLoader providing evaluation batches.

    Returns:
        Dictionary with accuracy, precision, recall, and f1 scores.
    """
    model.eval()
    preds = []
    targets = []
    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(model.device)
            logits = model(images)
            preds.append(logits.detach().cpu())
            targets.append(labels.detach().cpu())
    preds_tensor = torch.cat(preds, 0)
    targets_tensor = torch.cat(targets, 0)
    pred_labels = preds_tensor.argmax(dim=1)
    accuracy = accuracy_score(targets_tensor.numpy(), pred_labels.numpy())
    precision = precision_score(targets_tensor.numpy(), pred_labels.numpy(), average="weighted", zero_division=0)
    recall = recall_score(targets_tensor.numpy(), pred_labels.numpy(), average="weighted", zero_division=0)
    f1 = f1_score(targets_tensor.numpy(), pred_labels.numpy(), average="weighted", zero_division=0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@hydra.main(config_path=str(CONFIG_DIR), config_name="config.yaml", version_base=None)
def train(config: DictConfig) -> None:
    """Train a LightningModule and save the model artifacts."""
    hparams = config.hyperparameters
    dotenv_available = load_dotenv is not None
    if dotenv_available:
        load_dotenv()
    pl.seed_everything(hparams.seed, workers=True)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(output_dir / "train.log", level="INFO")
    logger.info("Training with Lightning boilerplate")
    logger.info(f"{hparams.lr=}, {hparams.batch_size=}, {hparams.epochs=}")
    logger.info(f"Model config: {OmegaConf.to_container(config.model, resolve=True)}")
    if not dotenv_available:
        logger.warning("python-dotenv is not installed; .env files will not be loaded.")

    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_project = os.getenv("WANDB_PROJECT", "mlops_g116")
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_mode = os.getenv("WANDB_MODE")
    wandb_logger = None
    wandb_run = None
    try:
        wandb_logger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            save_dir=str(wandb_dir),
            job_type="train_boilerplate",
        )
        if wandb_mode:
            wandb_logger.experiment.settings.mode = wandb_mode
        wandb_logger.log_hyperparams(
            {
                "lr": hparams.lr,
                "batch_size": hparams.batch_size,
                "epochs": hparams.epochs,
                "seed": hparams.seed,
                "model": OmegaConf.to_container(config.model, resolve=True),
            }
        )
        wandb_run = wandb_logger.experiment
    except Exception as exc:
        logger.warning(f"W&B init failed; continuing without logging: {exc}")
    wandb_enabled = wandb_run is not None

    model_config = OmegaConf.to_container(config.model, resolve=True)
    target = model_config.get("_target_") if isinstance(model_config, dict) else None
    if isinstance(target, str) and ".model." in target:
        model_config["_target_"] = target.replace(".model.", ".model_boilerplate.")
    if isinstance(model_config, dict):
        model_config.setdefault("lr", hparams.lr)
    model = instantiate(model_config)

    train_set, test_set = load_data()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=hparams.batch_size)

    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger if wandb_logger is not None else False,
        enable_checkpointing=False,
        default_root_dir=str(output_dir),
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pth"
    torch.save(model.state_dict(), model_path)

    local_model_dir = REPO_ROOT / "models"
    local_model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, local_model_dir / "model.pth")

    train_metrics = _evaluate_model(model, train_loader)
    val_metrics = _evaluate_model(model, val_loader)
    logger.info(f"Train metrics: {train_metrics}")
    logger.info(f"Validation metrics: {val_metrics}")
    if wandb_enabled:
        wandb_run.log(
            {
                "train/accuracy_final": train_metrics["accuracy"],
                "train/precision_final": train_metrics["precision"],
                "train/recall_final": train_metrics["recall"],
                "train/f1_final": train_metrics["f1"],
                "val/accuracy_final": val_metrics["accuracy"],
                "val/precision_final": val_metrics["precision"],
                "val/recall_final": val_metrics["recall"],
                "val/f1_final": val_metrics["f1"],
            }
        )
        artifact_name = os.getenv("WANDB_ARTIFACT_NAME", "mlops_g116_models")
        artifact = wandb.Artifact(
            name=f"{artifact_name}-{wandb_run.id}",
            type="models",
            description="Lightning boilerplate model",
            metadata={
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
        )
        artifact.add_file(str(model_path))
        wandb_run.log_artifact(artifact)
        registry_name = os.getenv("WANDB_REGISTRY", "wandb-registry-mlops_g116")
        collection_name = os.getenv("WANDB_COLLECTION_TRAIN", "mlops_g116-train-local")
        registry_entity = os.getenv("WANDB_REGISTRY_ENTITY") or wandb_entity
        if registry_entity:
            target_path = f"{registry_entity}/{registry_name}/{collection_name}"
        else:
            target_path = f"{registry_name}/{collection_name}"
        wandb_run.link_artifact(artifact, target_path=target_path, aliases=["latest"])
        wandb.finish()


if __name__ == "__main__":
    train()
