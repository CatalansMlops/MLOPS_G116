import cProfile
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
import torch
import wandb
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

from mlops_g116.data import load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _pick_available_port(preferred_port: int, max_tries: int = 10) -> int:
    """Return an available port, preferring the requested one."""
    for port in range(preferred_port, preferred_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred_port


def _is_port_open(port: int) -> bool:
    """Check whether a local TCP port is accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _launch_tensorboard(trace_dir: Path, preferred_port: int, open_browser: bool) -> None:
    """Start TensorBoard for the profiler traces if available."""
    tensorboard_cmd = shutil.which("tensorboard")
    if tensorboard_cmd is None:
        logger.warning("tensorboard is not installed; skipping automatic launch.")
        return
    port = _pick_available_port(preferred_port, max_tries=25)
    cmd = [
        tensorboard_cmd,
        "--logdir",
        str(trace_dir),
        "--load_fast=false",
        "--port",
        str(port),
    ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://localhost:{port}/#pytorch_profiler"
    for _ in range(25):
        if _is_port_open(port):
            logger.info(f"TensorBoard: {url}")
            logger.info("If you open a forwarded port, include the /#pytorch_profiler path.")
            if open_browser:
                webbrowser.open(url, new=2)
            return
        time.sleep(0.2)
    logger.warning(f"TensorBoard did not start. Try: tensorboard --logdir {trace_dir} --port {port}")


def _launch_snakeviz(profile_path: Path, preferred_port: int) -> None:
    """Start Snakeviz for the run profile if available."""
    try:
        import snakeviz  # noqa: F401
    except ModuleNotFoundError:
        logger.warning("snakeviz is not installed; skipping profiler visualization.")
        return
    port = _pick_available_port(preferred_port, max_tries=25)
    cmd = [sys.executable, "-m", "snakeviz", "-p", str(port), str(profile_path)]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info(f"Snakeviz: http://127.0.0.1:{port}/snakeviz/")


@hydra.main(config_path=str(CONFIG_DIR), config_name="evaluate.yaml", version_base=None)
def evaluate(config: DictConfig) -> None:
    """Evaluate a trained model.

    Args:
        config: Hydra configuration for evaluation.
    """
    dotenv_available = load_dotenv is not None
    if dotenv_available:
        load_dotenv()
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    reports_dir = output_dir / "reports"
    evaluation_dir = reports_dir / "evaluation"
    figures_dir = reports_dir / "figures"
    profiler_dir = output_dir / "profiler"
    torch_trace_dir = profiler_dir / "evaluate"
    wandb_dir = output_dir / "wandb"
    repo_figures_dir = REPO_ROOT / "reports" / "figures"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    profiler_dir.mkdir(parents=True, exist_ok=True)
    torch_trace_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    repo_figures_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(output_dir / "evaluate.log", level="INFO")
    logger.info("Evaluation run")
    if not dotenv_available:
        logger.warning("python-dotenv is not installed; .env files will not be loaded.")

    evaluation_config = config.evaluations if "evaluations" in config else config.evaluation
    batch_size = evaluation_config.batch_size
    checkpoint_path = Path(evaluation_config.checkpoint_path)
    logger.info(f"Model config: {OmegaConf.to_container(config.model, resolve=True)}")
    logger.info(f"Evaluation config: {OmegaConf.to_container(evaluation_config, resolve=True)}")

    wandb_project = os.getenv("WANDB_PROJECT", "mlops_g116")
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_mode = os.getenv("WANDB_MODE")
    wandb_kwargs = {
        "project": wandb_project,
        "job_type": "eval_local",
        "config": {
            "evaluation": OmegaConf.to_container(evaluation_config, resolve=True),
            "model": OmegaConf.to_container(config.model, resolve=True),
        },
        "dir": str(wandb_dir),
        "settings": wandb.Settings(console="off"),
    }
    if wandb_entity:
        wandb_kwargs["entity"] = wandb_entity
    if wandb_mode:
        wandb_kwargs["mode"] = wandb_mode
    wandb_run = None
    try:
        wandb_run = wandb.init(**wandb_kwargs)
    except Exception as exc:
        logger.warning(f"W&B init failed; continuing without logging: {exc}")
    wandb_enabled = wandb_run is not None

    profile_path = profiler_dir / "evaluate.prof"
    profiler = cProfile.Profile()
    profiler.enable()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    model = instantiate(config.model).to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    _, test_set = load_data()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    model.eval()
    preds = []
    targets = []
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(torch_trace_dir)),
    ) as prof:
        with torch.inference_mode():
            for img, target in test_dataloader:
                img, target = img.to(DEVICE), target.to(DEVICE)
                y_pred = model(img)
                preds.append(y_pred.detach().cpu())
                targets.append(target.detach().cpu())
                prof.step()

    preds_tensor = torch.cat(preds, 0)
    targets_tensor = torch.cat(targets, 0)
    pred_labels = preds_tensor.argmax(dim=1)
    accuracy = accuracy_score(targets_tensor.numpy(), pred_labels.numpy())
    precision = precision_score(targets_tensor.numpy(), pred_labels.numpy(), average="weighted", zero_division=0)
    recall = recall_score(targets_tensor.numpy(), pred_labels.numpy(), average="weighted", zero_division=0)
    f1 = f1_score(targets_tensor.numpy(), pred_labels.numpy(), average="weighted", zero_division=0)
    metrics = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/f1": f1,
    }
    if wandb_enabled:
        wandb.log(metrics)
    metrics_path = evaluation_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    logger.info(f"Evaluation metrics: {metrics}")

    cm_display = ConfusionMatrixDisplay.from_predictions(
        targets_tensor.numpy(),
        pred_labels.numpy(),
        normalize=None,
    )
    cm_path = figures_dir / "evaluation_confusion_matrix.png"
    cm_display.figure_.savefig(cm_path)
    plt.close(cm_display.figure_)
    shutil.copyfile(cm_path, repo_figures_dir / cm_path.name)

    if wandb_enabled:
        wandb.log({"eval/confusion_matrix": wandb.Image(str(cm_path))})

    artifact_base = os.getenv("WANDB_EVAL_ARTIFACT_NAME", "mlops_g116_evaluations")
    if wandb_enabled:
        artifact_name = f"{artifact_base}-{wandb_run.id}"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="evaluations",
            description="Evaluation metrics for a trained model",
            metadata={
                "checkpoint_path": str(checkpoint_path),
                "metrics": metrics,
                "model": OmegaConf.to_container(config.model, resolve=True),
            },
        )
        artifact.add_file(str(metrics_path))
        artifact.add_file(str(cm_path))
        wandb_run.log_artifact(artifact)
        registry_name = os.getenv("WANDB_REGISTRY", "wandb-registry-mlops_g116")
        collection_name = os.getenv("WANDB_COLLECTION_EVAL", "mlops_g116-eval-local")
        registry_entity = os.getenv("WANDB_REGISTRY_ENTITY") or wandb_entity
        if registry_entity:
            target_path = f"{registry_entity}/{registry_name}/{collection_name}"
        else:
            target_path = f"{registry_name}/{collection_name}"
        wandb_run.link_artifact(artifact, target_path=target_path)

    profiler.disable()
    profiler.dump_stats(profile_path)
    run_snakeviz = os.getenv("RUN_SNAKEVIZ", "1") == "1"
    run_tensorboard = os.getenv("RUN_TENSORBOARD", "1") == "1"
    open_tensorboard = os.getenv("OPEN_TENSORBOARD", "1") == "1"
    if run_snakeviz:
        _launch_snakeviz(profile_path, preferred_port=8080)
    if run_tensorboard:
        preferred_port = int(os.getenv("TENSORBOARD_PORT", "6006"))
        _launch_tensorboard(torch_trace_dir, preferred_port, open_tensorboard)
    if wandb_enabled:
        wandb.finish()

def main() -> None:
    """Run the Hydra evaluation entrypoint."""
    evaluate()

if __name__ == "__main__":
    evaluate()
