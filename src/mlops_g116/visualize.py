import cProfile
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import typer
import wandb
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

from mlops_g116.data import load_data
from mlops_g116.model import DenseNet121, ResNet18, TumorDetectionModelSimple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


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


def _build_model(model_name: str) -> torch.nn.Module:
    """Create a model instance from a short name."""
    name = model_name.lower()
    if name == "simple":
        return TumorDetectionModelSimple()
    if name == "resnet18":
        return ResNet18()
    if name == "densenet121":
        return DenseNet121()
    raise ValueError(f"Unsupported model name: {model_name}")


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


def visualize(
    model_checkpoint: str = "models/model.pth",
    model_name: str = "simple",
    figure_name: str = "embeddings.png",
    batch_size: int = 32,
) -> None:
    """Visualize model embeddings."""
    dotenv_available = load_dotenv is not None
    if dotenv_available:
        load_dotenv()
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    visualize_dir = reports_dir / "visualize"
    profiler_dir = reports_dir / "profiler"
    torch_trace_dir = profiler_dir / "visualize"
    wandb_dir = reports_dir / "wandb"
    figures_dir.mkdir(parents=True, exist_ok=True)
    visualize_dir.mkdir(parents=True, exist_ok=True)
    profiler_dir.mkdir(parents=True, exist_ok=True)
    torch_trace_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(visualize_dir / "visualize.log", level="INFO")
    logger.info("Visualizing embeddings")
    if not dotenv_available:
        logger.warning("python-dotenv is not installed; .env files will not be loaded.")

    wandb_project = os.getenv("WANDB_PROJECT", "mlops_g116")
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_mode = os.getenv("WANDB_MODE")
    wandb_job_type = os.getenv("WANDB_JOB_TYPE", "visualize_local")
    wandb_kwargs = {
        "project": wandb_project,
        "job_type": wandb_job_type,
        "config": {
            "batch_size": batch_size,
            "checkpoint_path": model_checkpoint,
            "model_name": model_name,
        },
        "dir": str(wandb_dir),
        "settings": wandb.Settings(console="off"),
    }
    if wandb_entity:
        wandb_kwargs["entity"] = wandb_entity
    if wandb_mode:
        wandb_kwargs["mode"] = wandb_mode
    wandb_run = wandb.init(**wandb_kwargs)

    profile_path = profiler_dir / "visualize.prof"
    profiler = cProfile.Profile()
    profiler.enable()

    checkpoint_path = Path(model_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    model = _build_model(model_name).to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    _strip_classifier(model)

    _, test_set = load_data()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    model.eval()
    embeddings = []
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
                img = img.to(DEVICE)
                output = model(img)
                embeddings.append(output.detach().cpu())
                targets.append(target.detach().cpu())
                prof.step()

    embeddings_tensor = torch.cat(embeddings, 0)
    targets_tensor = torch.cat(targets, 0)
    embeddings_np = embeddings_tensor.numpy()
    targets_np = targets_tensor.numpy()

    if embeddings_np.shape[1] > 500:
        pca = PCA(n_components=100)
        embeddings_np = pca.fit_transform(embeddings_np)
    tsne = TSNE(n_components=2)
    embeddings_np = tsne.fit_transform(embeddings_np)

    plt.figure(figsize=(10, 10))
    for class_id in sorted(set(targets_np.tolist())):
        mask = targets_np == class_id
        plt.scatter(embeddings_np[mask, 0], embeddings_np[mask, 1], label=str(class_id))
    plt.legend()
    figure_path = figures_dir / figure_name
    plt.savefig(figure_path)
    wandb.log({"visualize/tsne": wandb.Image(str(figure_path))})
    plt.close()

    artifact_base = os.getenv("WANDB_VIS_ARTIFACT_NAME", "mlops_g116_visualizations")
    artifact_name = f"{artifact_base}-{wandb_run.id}"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="visualizations",
        description="Embedding visualization for a trained model",
        metadata={
            "checkpoint_path": str(checkpoint_path),
            "model_name": model_name,
            "figure_name": figure_name,
        },
    )
    artifact.add_file(str(figure_path))
    wandb_run.log_artifact(artifact)
    registry_name = os.getenv("WANDB_REGISTRY", "wandb-registry-mlops_g116")
    collection_name = os.getenv("WANDB_COLLECTION", "mlops_g116-local-visualize")
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
    wandb.finish()


def main() -> None:
    typer.run(visualize)


if __name__ == "__main__":
    main()
