import typer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from mlops_g116.data import load_data
from mlops_g116.model import TumorDetectionModelSimple

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """
    Train a neural network using PyTorch Lightning with boilerplate flags.
    """

    # Load datasets
    train_set, val_set = load_data()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size)

    # Initialize model
    model = TumorDetectionModelSimple()

    # Create the W&B logger
    wandb_logger = WandbLogger(project="dtu_mlops")

    # Instantiate Lightning Trainer with boilerplate flags
    trainer = Trainer(
        logger=wandb_logger,
        default_root_dir="lightning_logs",  # Saves checkpoints & logs here
        max_epochs=epochs,                  # Limit epochs (default is 1000)
        limit_train_batches=0.2,            # Use only 20% of training data
        limit_val_batches=0.5,              # Optional: limit validation batches
        accelerator="auto",                  # Use GPU if available
        devices=1                            # Number of devices
    )

    # Train model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Training complete!")

def main() -> None:
    typer.run(train)

if __name__ == "__main__":
    main()
