# MLOPS_G116

# Project: NeuroClassify - MLOps Pipeline for Brain Tumor Detection

## 1. Dataset Selection
**Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)
* **Source:** Kaggle (Masoud Nickparvar)
* **Type:** Medical Imaging (2D MRI Slices)

## 2. Model Selection
**Model:** ResNet-18
* **Architecture:** Convolutional Neural Network (CNN) with residual connections.
* **Rationale:** We selected ResNet-18 because it provides a strong baseline for medical image classification while being computationally efficient. This allows for faster iteration cycles in our MLOps pipeline (training, testing, deployment) without requiring massive GPU resources.

## 3. Project Description

### a. Overall Goal of the Project
The goal is to engineer a robust, end-to-end MLOps pipeline that automates the classification of brain MRI scans into four diagnostic categories. The project focuses on the *operational* aspects of machine learning—such as automated data validation, model versioning, continuous integration/training (CI/CD), and model monitoring—rather than just achieving the highest possible accuracy.

### b. Data Description
We will be using the **Brain Tumor MRI Dataset** aggregated from multiple open-source medical repositories.
* **Total Samples:** Approximately **7,023 images**.
* **Modality:** 2D MRI scans (JPG format).
* **Classes:** 4 categories (Glioma, Meningioma, Pituitary, No Tumor).
* **Size:** ~150 MB.
* **Structure:** The data is organized into train/test folders by class label, simplifying the initial data loading pipeline.

### c. Expected Models
* **Baseline:** We will start with a **ResNet-18** backbone pre-trained on ImageNet, modifying the final fully connected layer to output 4 classes.
* **Future/Experimental:** If the MLOps pipeline is stable, we may experiment with **EfficientNet-B0** for better parameter efficiency or explore **MONAI**-specific implementations (e.g., DenseNet121) tailored for healthcare imaging.

## Project structure

The directory structure of the project looks like this:
```txt
├── .devcontainer/            # VS Code Dev Container config
│   ├── devcontainer.json
│   └── Dockerfile
├── .dvc/                     # DVC metadata
├── .github/                  # CI workflows and Dependabot
│   ├── agents/
│   ├── prompts/
│   ├── dependabot.yaml
│   └── workflows/
│       ├── linting.yaml
│       └── tests.yaml
├── configs/                  # Hydra configs (training/eval/visualize/sweeps)
│   ├── cloudbuild.yaml
│   ├── config.yaml
│   ├── dataset/
│   ├── model/
│   ├── optimizer/
│   ├── training/
│   ├── evaluation/
│   ├── visualization/
│   ├── vertex/
│   ├── sweep.yaml
│   ├── evaluate.yaml
│   └── visualize.yaml
├── data/                     # Data tracked with DVC
│   ├── raw/                  # Raw dataset
│   ├── processed/            # Processed tensors
│   ├── raw.dvc
│   └── processed.dvc
├── dockerfiles/              # Docker build targets
│   ├── api.dockerfile
│   ├── backend.dockerfile
│   ├── frontend.dockerfile
│   ├── main.dockerfile
│   ├── main.CPU.dockerfile
│   ├── main.local.dockerfile
│   ├── train.dockerfile
│   ├── train.local.dockerfile
│   ├── evaluate.dockerfile
│   ├── visualize.dockerfile
│   ├── wandb.dockerfile
│   └── *_entrypoint.sh
├── docs/                     # MkDocs documentation
│   ├── mkdocs.yaml
│   └── source/
│       ├── index.md
│       └── cloud_deployment.md
├── models/                   # Optimal trained model artifacts
│   └── model.pth
├── outputs/                  # Hydra run outputs (dated folders)
├── reports/                  # Exam report and generator, and final figures from evaluation and visualization
│   ├── README.md
│   ├── report.html
│   ├── report.py
│   └── figures/
├── src/                      # Python package
│   └── mlops_g116/
│       ├── __init__.py
│       ├── backend.py
│       ├── frontend.py
│       ├── data.py
│       ├── data_importfromcloud.py
│       ├── train.py
│       ├── train_boilerplate.py
│       ├── evaluate.py
│       ├── visualize.py
│       ├── main.py
│       ├── model.py
│       ├── model_boilerplate.py
│       ├── sweep_runner.py
│       └── registry_download.py
├── tests/                    # Unit/integration/performance tests
│   ├── unittests/
│   ├── integrationtests/
│   └── performancetests/
├── .dvcignore
├── .gcloudignore
├── .gitignore
├── .pre-commit-config.yaml   # Pre-commit hooks
├── .python-version
├── AGENTS.md
├── LICENSE
├── pyproject.toml            # Project metadata
├── requirements*.txt         # Dependency pins
├── tasks.py                  # Invoke tasks
└── README.md                 # Project overview
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
