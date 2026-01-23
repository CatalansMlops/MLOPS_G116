import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops_g116"
PYTHON_VERSION = "3.12"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


@task
def python(ctx):
    """prueba"""
    ctx.run("which python" if os.name != "nt" else "where python")


@task
def git(ctx, message):
    ctx.run(f"git add .")
    ctx.run(f"git commit -m '{message}'")
    ctx.run(f"git push")


@task
def conda(ctx, name: str = "mlops_g116"):
    """Create and set up a conda environment from environment.yml file."""
    ctx.run(f"conda env create -f environment.yml", echo=True)
    ctx.run(f"conda activate {name}", echo=True)
    ctx.run(f"pip install -e .", echo=True)


# The two below are for dvc data version control (we will erase them later most probably)


@task
def dvc(ctx, folder="data", message="Add new data"):
    """Port data folder to dvc and push to remote storage."""
    ctx.run(f"dvc add {folder}")
    ctx.run(f"git add {folder}.dvc .gitignore")
    ctx.run(f"git commit -m '{message}'")
    ctx.run(f"git push")
    ctx.run(f"dvc push")


""" No se exactamente que hacen dos de abajo """


@task
def gcloud(ctx):
    """Create and set up a conda environment from environment.yml file."""
    ctx.run(f"sudo apt-get update", echo=True)
    ctx.run(f"sudo apt-get install -y ca-certificates", echo=True)
    ctx.run(f"sudo update-ca-certificates", echo=True)
    ctx.run(f"conda install -c conda-forge ca-certificates certifi openssl", echo=True)
    ctx.run(f"sudo apt-get install apt-transport-https ca-certificates gnupg curl", echo=True)
    ctx.run(
        "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | "
        "sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg",
        echo=True,
    )
    ctx.run(
        "echo "
        '"deb [signed-by=/usr/share/keyrings/cloud.google.gpg] '
        'https://packages.cloud.google.com/apt cloud-sdk main" '
        "| sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list",
        echo=True,
    )
    ctx.run(f"sudo apt-get update && sudo apt-get install google-cloud-cli", echo=True)
    ctx.run(f"gcloud auth login --no-launch-browser", echo=True)
    ctx.run(f"gcloud auth application-default login --no-launch-browser", echo=True)
    ctx.run(f"gcloud config set project mlops116", echo=True)
    ctx.run(f"pip install --upgrade google-api-python-client", echo=True)
    ctx.run(f"gcloud services enable apigateway.googleapis.com", echo=True)
    ctx.run(f"gcloud services enable servicemanagement.googleapis.com", echo=True)
    ctx.run(f"gcloud services enable servicecontrol.googleapis.com", echo=True)
    # To be able to pull containers from the cloud:
    # docker pull europe-west1-docker.pkg.dev/dtumlops-484509/mlops-container-images/cloud-container:latest
    ctx.run("gcloud auth configure-docker europe-west1-docker.pkg.dev", echo=True)


@task
def pull_data(ctx):
    ctx.run("dvc pull")


@task(pull_data)
def train(ctx):
    ctx.run("my_cli train")


@task
def coverage(ctx):
    ctx.run("coverage run --source=src -m pytest tests/")
    ctx.run("coverage report -m")
