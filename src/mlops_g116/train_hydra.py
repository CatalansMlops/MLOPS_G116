import cProfile
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import hydra
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score
import torch
import wandb
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from mlops_g116.data import load_data
from mlops_g116.model import TumorDetectionModel

# Select the best available device:
# - CUDA if an NVIDIA GPU is available
# - MPS for Apple Silicon
# - CPU otherwise
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _pick_available_port(preferred_port: int, max_tries: int = 10) -> int:
    """Return an available port, preferring the requested one."""
    for port in range(preferred_port, preferred_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred_port

def _is_port_open(port: int) -> bool:
    """Check whether a local TCP port is accepting connections.

    Args:
        port: TCP port number to probe.

    Returns:
        True if the port accepts connections, otherwise False.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0

def _launch_tensorboard(output_dir: Path, preferred_port: int, open_browser: bool) -> None:
    """Start TensorBoard for the run directory if available.

    Args:
        output_dir: Directory containing the profiler traces.
        preferred_port: Preferred port to bind for TensorBoard.
        open_browser: Whether to open the TensorBoard URL in a browser tab.
    """
    tensorboard_cmd = shutil.which("tensorboard")
    if tensorboard_cmd is None:
        logger.warning("tensorboard is not installed; skipping automatic launch.")
        return
    port = _pick_available_port(preferred_port, max_tries=25)
    cmd = [
        tensorboard_cmd,
        "--logdir",
        str(output_dir),
        "--load_fast=false",
        "--port",
        str(port),
    ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://localhost:{port}/#pytorch_profiler"
    for _ in range(25):
        if _is_port_open(port):
            logger.info(f"TensorBoard: {url}")
            if open_browser:
                webbrowser.open(url, new=2)
            return
        time.sleep(0.2)
    logger.warning(f"TensorBoard did not start. Try: tensorboard --logdir {output_dir} --port {port}")

@hydra.main(config_path=str(CONFIG_DIR), config_name="config.yaml", version_base=None)
def train(config) -> None:
    '''
    Train a neural network on the MNIST dataset and save the trained model
    together with training statistics.

            Parameters:
                    lr (float): Learning rate used by the optimizer.
                    batch_size (int): Number of samples per training batch.
                    epochs (int): Number of full passes over the training dataset.

            Returns:
                    None: The function saves the trained model weights to disk
                          and stores training loss and accuracy plots.
    '''
    hparams = config.hyperparameters
    dotenv_available = load_dotenv is not None
    if dotenv_available:
        load_dotenv()
    torch.manual_seed(hparams.seed)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(output_dir / "train.log", level="INFO")
    logger.info("Training day and night")
    logger.info(f"{hparams.lr=}, {hparams.batch_size=}, {hparams.epochs=}")
    if not dotenv_available:
        logger.warning("python-dotenv is not installed; .env files will not be loaded.")
    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_project = os.getenv("WANDB_PROJECT", "mlops_g116")
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_mode = os.getenv("WANDB_MODE")
    wandb_kwargs = {
        "project": wandb_project,
        "job_type": "train",
        "config": {
            "lr": hparams.lr,
            "batch_size": hparams.batch_size,
            "epochs": hparams.epochs,
            "seed": hparams.seed,
        },
        "dir": str(wandb_dir),
    }
    if wandb_entity:
        wandb_kwargs["entity"] = wandb_entity
    if wandb_mode:
        wandb_kwargs["mode"] = wandb_mode
    wandb_run = wandb.init(**wandb_kwargs)
    if "lr" in wandb.config:
        hparams.lr = float(wandb.config.lr)
    if "batch_size" in wandb.config:
        hparams.batch_size = int(wandb.config.batch_size)
    if "epochs" in wandb.config:
        hparams.epochs = int(wandb.config.epochs)
    logger.info(f"Effective hyperparameters: {hparams.lr=}, {hparams.batch_size=}, {hparams.epochs=}")
    model_dir = output_dir / "models"
    figure_dir = output_dir / "reports" / "figures"
    trace_dir = output_dir / "profiler"
    model_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "profile.prof"
    profiler = cProfile.Profile()
    profiler.enable()
    logger.info(f"Run outputs saved under: {output_dir}")

    # Initialize model and move it to the selected device (GPU/CPU)
    model = TumorDetectionModel().to(DEVICE)

    # Load corrupted MNIST dataset
    # train_set is used for training, test_set is ignored here
    train_set, _ = load_data()
    train_labels = train_set.tensors[1]
    unique_labels, label_counts = torch.unique(train_labels, return_counts=True)
    label_summary = list(zip(unique_labels.tolist(), label_counts.tolist()))
    logger.info(f"Train labels: {label_summary}")
    if hasattr(model, "fc1"):
        logger.info(f"Model output classes: {model.fc1.out_features}")
    # Wrap dataset into a DataLoader to iterate in mini-batches
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=hparams.batch_size
    )

    # Standard loss function for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()

    # Adam optimizer updates model parameters using gradients
    optimizer = instantiate(config.optimizer, params=model.parameters())

    # Store statistics for visualization later
    statistics = {"train_loss": [], "train_accuracy": []}

    final_preds = None
    final_targets = None

    for epoch in range(hparams.epochs):
        preds = []
        targets = []
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        model.train()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
        ) as prof:
            for i, (img, target) in enumerate(train_dataloader):
                img, target = img.to(DEVICE), target.to(DEVICE)

                optimizer.zero_grad()

                y_pred = model(img)

                with torch.profiler.record_function("model_loss"):
                    loss = loss_fn(y_pred, target)

                with torch.profiler.record_function("backward"):
                    loss.backward()
                    optimizer.step()

                statistics["train_loss"].append(loss.item())

                accuracy = (
                    (y_pred.argmax(dim=1) == target)
                    .float()
                    .mean()
                    .item()
                )
                epoch_loss += loss.item()
                epoch_acc += accuracy
                num_batches += 1
                statistics["train_accuracy"].append(accuracy)
                wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

                preds.append(y_pred.detach().cpu())
                targets.append(target.detach().cpu())

                if i % 100 == 0:
                    logger.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
                    images = []
                    for j, im in enumerate(img[:5]):
                        im_vis = im.detach().cpu()
                        im_min = im_vis.min()
                        im_max = im_vis.max()
                        if (im_max - im_min) > 0:
                            im_vis = (im_vis - im_min) / (im_max - im_min)
                        im_vis = (im_vis * 255).clamp(0, 255)
                        images.append(wandb.Image(im_vis, caption=f"Input {j}"))
                    wandb.log({"images": images})

                    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                    wandb.log({"gradients": wandb.Histogram(grads)})

                prof.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        wandb.log({"epoch_loss": avg_loss, "epoch_acc": avg_acc, "epoch": epoch})
        preds_tensor = torch.cat(preds, 0)
        targets_tensor = torch.cat(targets, 0)
        num_classes = preds_tensor.shape[1]
        for class_id in range(num_classes):
            if (targets_tensor == class_id).any():
                one_hot = torch.zeros_like(targets_tensor)
                one_hot[targets_tensor == class_id] = 1
                _ = RocCurveDisplay.from_predictions(
                    one_hot.numpy(),
                    preds_tensor[:, class_id].numpy(),
                    name=f"ROC curve for {class_id}",
                    plot_chance_level=(class_id == 2),
                )

        fig = plt.gcf()
        wandb.log({"roc": wandb.Image(fig)})
        plt.close(fig)
        final_preds = preds_tensor
        final_targets = targets_tensor

    logger.info("Training complete")

    # Save trained model parameters to disk
    torch.save(model.state_dict(), model_dir / "model.pth")
    if final_preds is not None and final_targets is not None:
        final_pred_labels = final_preds.argmax(dim=1)
        final_accuracy = accuracy_score(final_targets.numpy(), final_pred_labels.numpy())
        final_precision = precision_score(
            final_targets.numpy(),
            final_pred_labels.numpy(),
            average="weighted",
            zero_division=0,
        )
        final_recall = recall_score(final_targets.numpy(), final_pred_labels.numpy(), average="weighted")
        final_f1 = f1_score(final_targets.numpy(), final_pred_labels.numpy(), average="weighted")
        wandb.log(
            {
                "final_accuracy": final_accuracy,
                "final_precision": final_precision,
                "final_recall": final_recall,
                "final_f1": final_f1,
            }
        )
        artifact_name = os.getenv("WANDB_ARTIFACT_NAME", "mlops_g116_models")
        artifact = wandb.Artifact(
            name=artifact_name,
            type="models",
            description="Model trained to classify brain tumor images",
            metadata={
                "accuracy": final_accuracy,
                "precision": final_precision,
                "recall": final_recall,
                "f1": final_f1,
            },
        )
        artifact.add_file(str(model_dir / "model.pth"))
        wandb_run.log_artifact(artifact)
        registry_name = os.getenv("WANDB_REGISTRY", "wandb-registry-mlops_g116")
        collection_name = os.getenv("WANDB_COLLECTION", "mlops_g116")
        registry_entity = os.getenv("WANDB_REGISTRY_ENTITY") or wandb_entity
        if registry_entity:
            target_path = f"{registry_entity}/{registry_name}/{collection_name}"
        else:
            target_path = f"{registry_name}/{collection_name}"
        wandb_run.link_artifact(artifact, target_path=target_path, aliases=["latest"])

    # Plot training loss and accuracy
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")

    # Save figure in reports folder
    fig.savefig(figure_dir / "training_statistics.png")
    profiler.disable()
    profiler.dump_stats(profile_path)
    run_snakeviz = os.getenv("RUN_SNAKEVIZ", "1") == "1"
    run_tensorboard = os.getenv("RUN_TENSORBOARD", "1") == "1"
    open_tensorboard = os.getenv("OPEN_TENSORBOARD", "1") == "1"
    if run_snakeviz:
        try:
            subprocess.Popen([sys.executable, "-m", "snakeviz", str(profile_path)])
        except FileNotFoundError:
            logger.warning("snakeviz is not installed; skipping profiler visualization.")
    if run_tensorboard:
        preferred_port = int(os.getenv("TENSORBOARD_PORT", "6006"))
        _launch_tensorboard(output_dir, preferred_port, open_tensorboard)
    wandb.finish()

def main() -> None:
    """Run the Hydra training entrypoint."""
    train()

if __name__ == "__main__":
    main()
