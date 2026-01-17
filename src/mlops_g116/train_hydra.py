import cProfile
import subprocess
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch

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

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
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
    print("Training day and night")
    print(f"{hparams.lr=}, {hparams.batch_size=}, {hparams.epochs=}")
    torch.manual_seed(hparams.seed)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    model_dir = output_dir / "models"
    figure_dir = output_dir / "reports" / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "profile.prof"
    profiler = cProfile.Profile()
    profiler.enable()

    # Initialize model and move it to the selected device (GPU/CPU)
    model = TumorDetectionModel().to(DEVICE)

    # Load corrupted MNIST dataset
    # train_set is used for training, test_set is ignored here
    train_set, _ = load_data()
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

    for epoch in range(hparams.epochs):
        model.train()  # Enables training mode (important for layers like dropout)

        for i, (img, target) in enumerate(train_dataloader):
            # img shape: (B, 1, 28, 28)
            # target shape: (B,)
            img, target = img.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            # y_pred shape: (B, 10)
            y_pred = model(img)

            # Compute loss comparing predictions with ground truth labels
            loss = loss_fn(y_pred, target)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Store loss value
            statistics["train_loss"].append(loss.item())

            # Compute accuracy for this batch
            accuracy = (
                (y_pred.argmax(dim=1) == target)
                .float()
                .mean()
                .item()
            )
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")

    # Save trained model parameters to disk
    torch.save(model.state_dict(), model_dir / "model.pth")

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
    subprocess.run([sys.executable, "-m", "snakeviz", str(profile_path)], check=False)

def main() -> None:
    """Run the Hydra training entrypoint."""
    train()

if __name__ == "__main__":
    train()

## Seguir con profiling Pytorch