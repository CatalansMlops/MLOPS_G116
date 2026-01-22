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
from loguru import logger
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score
import torch
import wandb
from google.cloud import storage
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from mlops_g116.data import load_data

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
DEFAULT_OUTPUT_BUCKET = "mlops116"


def _resolve_output_prefix(output_dir: Path, output_root_env: str | None) -> str:
    """Resolve a GCS prefix for a run output directory.

    Args:
        output_dir: The run output directory on disk.
        output_root_env: Optional OUTPUT_ROOT environment variable value.

    Returns:
        A GCS prefix to use for uploads.
    """
    if output_root_env:
        output_root = Path(output_root_env).resolve()
        try:
            relative = output_dir.relative_to(output_root)
            return f"outputs/{relative.as_posix()}"
        except ValueError:
            pass
    cwd = Path.cwd()
    try:
        relative = output_dir.relative_to(cwd)
        return relative.as_posix()
    except ValueError:
        pass
    if "outputs" in output_dir.parts:
        idx = output_dir.parts.index("outputs")
        return "/".join(output_dir.parts[idx:])
    return f"outputs/{output_dir.name}"


def _upload_outputs_to_gcs(output_dir: Path, bucket_name: str, prefix: str) -> None:
    """Upload run outputs to a GCS bucket.

    Args:
        output_dir: Directory containing run outputs.
        bucket_name: Target GCS bucket name.
        prefix: Prefix within the bucket to store outputs.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    normalized_prefix = prefix.strip("/")
    base_prefix = f"{normalized_prefix}/" if normalized_prefix else ""
    for path in output_dir.rglob("*"):
        if path.is_file():
            rel_path = path.relative_to(output_dir).as_posix()
            blob = bucket.blob(f"{base_prefix}{rel_path}")
            blob.upload_from_filename(str(path))


def _cleanup_output_dir(output_dir: Path) -> None:
    """Remove run outputs and delete empty parent directories."""
    shutil.rmtree(output_dir, ignore_errors=True)
    parent = output_dir.parent
    while parent != parent.parent:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


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
            logger.info("If you open a forwarded port, include the /#pytorch_profiler path.")
            if open_browser:
                webbrowser.open(url, new=2)
            return
        time.sleep(0.2)
    logger.warning(f"TensorBoard did not start. Try: tensorboard --logdir {output_dir} --port {port}")


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


def _strip_classifier(model: torch.nn.Module) -> None:
    """Replace the classifier head with an identity for embedding extraction."""
    if hasattr(model, "fc1"):
        model.fc1 = torch.nn.Identity()
        return
    if hasattr(model, "backbone"):
        backbone = model.backbone
        if hasattr(backbone, "fc"):
            backbone.fc = torch.nn.Identity()
            return
        if hasattr(backbone, "classifier"):
            backbone.classifier = torch.nn.Identity()


@hydra.main(config_path=str(CONFIG_DIR), config_name="config.yaml", version_base=None)
def main(config: DictConfig) -> None:
    """Train and evaluate a model with shared configuration.

    Args:
        config: Hydra configuration with hyperparameters, model, and optimizer.
    """
    hparams = config.hyperparameters
    dotenv_available = load_dotenv is not None
    if dotenv_available:
        load_dotenv()
    torch.manual_seed(hparams.seed)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(output_dir / "run.log", level="INFO")
    logger.info("Training and evaluation run")
    logger.info(f"{hparams.lr=}, {hparams.batch_size=}, {hparams.epochs=}")
    logger.info(f"Model config: {OmegaConf.to_container(config.model, resolve=True)}")
    logger.info(f"Optimizer config: {OmegaConf.to_container(config.optimizer, resolve=True)}")
    if not dotenv_available:
        logger.warning("python-dotenv is not installed; .env files will not be loaded.")
    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_project = os.getenv("WANDB_PROJECT", "mlops_g116")
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_mode = os.getenv("WANDB_MODE")
    wandb_job_type = os.getenv("WANDB_JOB_TYPE", "train_eval_main")
    wandb_kwargs = {
        "project": wandb_project,
        "job_type": wandb_job_type,
        "config": {
            "lr": hparams.lr,
            "batch_size": hparams.batch_size,
            "epochs": hparams.epochs,
            "seed": hparams.seed,
            "model": OmegaConf.to_container(config.model, resolve=True),
            "optimizer": OmegaConf.to_container(config.optimizer, resolve=True),
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
    model_dir = output_dir / "models"
    figure_dir = output_dir / "reports" / "figures"
    evaluation_dir = output_dir / "reports" / "evaluation"
    trace_dir = output_dir / "profiler"
    trace_train_dir = trace_dir / "train"
    trace_eval_dir = trace_dir / "evaluate"
    trace_visualize_dir = trace_dir / "visualize"
    model_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    trace_train_dir.mkdir(parents=True, exist_ok=True)
    trace_eval_dir.mkdir(parents=True, exist_ok=True)
    trace_visualize_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "profile.prof"
    profiler = cProfile.Profile()
    profiler.enable()
    logger.info(f"Run outputs saved under: {output_dir}")

    # Initialize model and move it to the selected device (GPU/CPU)
    model = instantiate(config.model).to(DEVICE)

    train_set, test_set = load_data()
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
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_train_dir)),
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

                batch_accuracy = (
                    (y_pred.argmax(dim=1) == target)
                    .float()
                    .mean()
                    .item()
                )
                epoch_loss += loss.item()
                epoch_acc += batch_accuracy
                num_batches += 1
                statistics["train_accuracy"].append(batch_accuracy)
                if wandb_enabled:
                    wandb.log(
                        {
                            "train/loss_batch": loss.item(),
                            "train/accuracy_batch": batch_accuracy,
                            "train/epoch": epoch,
                        }
                    )

                preds.append(y_pred.detach().cpu())
                targets.append(target.detach().cpu())

                if i % 100 == 0:
                    logger.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
                    if wandb_enabled:
                        images = []
                        for j, im in enumerate(img[:5]):
                            im_vis = im.detach().cpu()
                            im_min = im_vis.min()
                            im_max = im_vis.max()
                            if (im_max - im_min) > 0:
                                im_vis = (im_vis - im_min) / (im_max - im_min)
                            im_vis = (im_vis * 255).clamp(0, 255)
                            images.append(wandb.Image(im_vis, caption=f"Input {j}"))
                        wandb.log({"train/images": images})

                        grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                        wandb.log({"train/gradients": wandb.Histogram(grads)})

                prof.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        if wandb_enabled:
            wandb.log({"train/loss_epoch": avg_loss, "train/accuracy_epoch": avg_acc, "train/epoch": epoch})
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
        if wandb_enabled:
            wandb.log({"train/roc": wandb.Image(fig)})
        plt.close(fig)
        final_preds = preds_tensor
        final_targets = targets_tensor

    logger.info("Training complete")

    # Save trained model parameters to disk
    model_path = model_dir / "model.pth"
    torch.save(model.state_dict(), model_path)
    train_accuracy = None
    train_precision = None
    train_recall = None
    train_f1 = None
    if final_preds is not None and final_targets is not None:
        final_pred_labels = final_preds.argmax(dim=1)
        train_accuracy = accuracy_score(final_targets.numpy(), final_pred_labels.numpy())
        train_precision = precision_score(
            final_targets.numpy(),
            final_pred_labels.numpy(),
            average="weighted",
            zero_division=0,
        )
        train_recall = recall_score(final_targets.numpy(), final_pred_labels.numpy(), average="weighted")
        train_f1 = f1_score(final_targets.numpy(), final_pred_labels.numpy(), average="weighted")
        if wandb_enabled:
            wandb.log(
                {
                    "train/accuracy_final": train_accuracy,
                    "train/precision_final": train_precision,
                    "train/recall_final": train_recall,
                    "train/f1_final": train_f1,
                }
            )

    model.eval()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=hparams.batch_size)
    eval_preds = []
    eval_targets = []
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_eval_dir)),
    ) as prof:
        with torch.inference_mode():
            for img, target in test_dataloader:
                img, target = img.to(DEVICE), target.to(DEVICE)
                y_pred = model(img)
                eval_preds.append(y_pred.detach().cpu())
                eval_targets.append(target.detach().cpu())
                prof.step()

    eval_preds_tensor = torch.cat(eval_preds, 0)
    eval_targets_tensor = torch.cat(eval_targets, 0)
    eval_labels = eval_preds_tensor.argmax(dim=1)
    test_accuracy = accuracy_score(eval_targets_tensor.numpy(), eval_labels.numpy())
    test_precision = precision_score(
        eval_targets_tensor.numpy(),
        eval_labels.numpy(),
        average="weighted",
        zero_division=0,
    )
    test_recall = recall_score(
        eval_targets_tensor.numpy(),
        eval_labels.numpy(),
        average="weighted",
        zero_division=0,
    )
    test_f1 = f1_score(
        eval_targets_tensor.numpy(),
        eval_labels.numpy(),
        average="weighted",
        zero_division=0,
    )
    evaluation_metrics = {
        "test/accuracy": test_accuracy,
        "test/precision": test_precision,
        "test/recall": test_recall,
        "test/f1": test_f1,
    }
    if wandb_enabled:
        wandb.log(evaluation_metrics)
    logger.info(f"Evaluation metrics: {evaluation_metrics}")

    metrics_path = evaluation_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(evaluation_metrics, handle, indent=2)

    if wandb_enabled:
        artifact_name = os.getenv("WANDB_ARTIFACT_NAME", "mlops_g116_models")
        model_metadata = {
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
        }
        if train_accuracy is not None:
            model_metadata.update(
                {
                    "train_accuracy": train_accuracy,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                    "train_f1": train_f1,
                }
            )
        model_artifact = wandb.Artifact(
            name=artifact_name,
            type="models",
            description="Model trained to classify brain tumor images",
            metadata=model_metadata,
        )
        model_artifact.add_file(str(model_path))
        wandb_run.log_artifact(model_artifact)
        registry_name = os.getenv("WANDB_REGISTRY", "wandb-registry-mlops_g116")
        collection_name_model = os.getenv("WANDB_COLLECTION_MAIN", "mlops_g116-main-models")
        collection_name_eval = os.getenv("WANDB_COLLECTION_MAIN_EVAL", "mlops_g116-main-evals")
        registry_entity = os.getenv("WANDB_REGISTRY_ENTITY") or wandb_entity
        if registry_entity:
            target_path_model = f"{registry_entity}/{registry_name}/{collection_name_model}"
            target_path_eval = f"{registry_entity}/{registry_name}/{collection_name_eval}"
        else:
            target_path_model = f"{registry_name}/{collection_name_model}"
            target_path_eval = f"{registry_name}/{collection_name_eval}"
        wandb_run.link_artifact(model_artifact, target_path=target_path_model, aliases=["latest"])

        evaluation_artifact_name = os.getenv("WANDB_EVAL_ARTIFACT_NAME", "mlops_g116_evaluations")
        evaluation_artifact = wandb.Artifact(
            name=f"{evaluation_artifact_name}-{wandb_run.id}",
            type="evaluations",
            description="Evaluation metrics for the trained model",
            metadata={
                "checkpoint_path": str(model_path),
                "metrics": evaluation_metrics,
            },
        )
        evaluation_artifact.add_file(str(metrics_path))
        wandb_run.log_artifact(evaluation_artifact)
        wandb_run.link_artifact(evaluation_artifact, target_path=target_path_eval, aliases=["latest"])

    _strip_classifier(model)
    visualize_dataloader = torch.utils.data.DataLoader(test_set, batch_size=hparams.batch_size)
    embeddings = []
    labels = []
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_visualize_dir)),
    ) as prof:
        with torch.inference_mode():
            for img, target in visualize_dataloader:
                img = img.to(DEVICE)
                output = model(img)
                embeddings.append(output.detach().cpu())
                labels.append(target.detach().cpu())
                prof.step()
    embeddings_tensor = torch.cat(embeddings, 0)
    labels_tensor = torch.cat(labels, 0)
    embeddings_np = embeddings_tensor.numpy()
    labels_np = labels_tensor.numpy()

    if embeddings_np.shape[1] > 500:
        pca = PCA(n_components=100)
        embeddings_np = pca.fit_transform(embeddings_np)
    tsne = TSNE(n_components=2)
    embeddings_np = tsne.fit_transform(embeddings_np)

    plt.figure(figsize=(10, 10))
    for class_id in sorted(set(labels_np.tolist())):
        mask = labels_np == class_id
        plt.scatter(embeddings_np[mask, 0], embeddings_np[mask, 1], label=str(class_id))
    plt.legend()
    visualization_path = figure_dir / "embedding_tsne.png"
    plt.savefig(visualization_path)
    if wandb_enabled:
        wandb.log({"visualize/tsne": wandb.Image(str(visualization_path))})
    plt.close()
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
        _launch_snakeviz(profile_path, preferred_port=8080)
    if run_tensorboard:
        preferred_port = int(os.getenv("TENSORBOARD_PORT", "6006"))
        _launch_tensorboard(trace_dir, preferred_port, open_tensorboard)
    output_bucket = os.getenv("OUTPUT_GCS_BUCKET", DEFAULT_OUTPUT_BUCKET)
    output_prefix = _resolve_output_prefix(output_dir, os.getenv("OUTPUT_ROOT"))
    try:
        _upload_outputs_to_gcs(output_dir, output_bucket, output_prefix)
        logger.info(f"Uploaded outputs to gs://{output_bucket}/{output_prefix}")
        if not (run_snakeviz or run_tensorboard):
            _cleanup_output_dir(output_dir)
    except Exception as exc:
        logger.warning(f"Failed to upload outputs to GCS: {exc}")
    if wandb_enabled:
        wandb.finish()

if __name__ == "__main__":
    main()
