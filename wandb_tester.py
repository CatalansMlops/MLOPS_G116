"""Minimal Weights & Biases logging test for container auth."""

import random

import wandb


def main() -> None:
    """Run a minimal W&B logging test."""
    wandb.init(project="wandb-test")
    for _ in range(100):
        wandb.log({"test_metric": random.random()})
    wandb.finish()


if __name__ == "__main__":
    main()
