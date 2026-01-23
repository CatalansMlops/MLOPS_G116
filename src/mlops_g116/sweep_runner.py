import subprocess
import sys

import wandb


def main() -> None:
    """Run a sweep-configured training job via Hydra overrides."""
    run = wandb.init()
    cfg = run.config
    args = [
        f"hyperparameters.lr={cfg.lr}",
        f"hyperparameters.batch_size={cfg.batch_size}",
        f"hyperparameters.epochs={cfg.epochs}",
        f"model={cfg.model}",
        f"optimizer={cfg.optimizer}",
    ]
    subprocess.check_call([sys.executable, "-m", "mlops_g116.main", *args])


if __name__ == "__main__":
    main()
