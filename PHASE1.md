# PHASE 1: Project Design & Model Development

This document presents the planning, system design, data engineering, and model development progress achieved in Phase 1 of the Sudoku Solver MLOps project. Conducted under SE489 — Machine Learning Engineering for Production, this phase delivers a robust and reproducible foundation for scaling our ML solution using MLOps principles.

## 1. Project Proposal

### 1.1 Project Scope and Objectives

**Problem Statement:**
The project addresses the challenge of automatically solving Sudoku puzzles from real-world images. This involves detecting a grid from noisy or skewed inputs, extracting digits, classifying them correctly, and predicting the missing digits using a trained deep learning model. The goal is to provide an end-to-end automated pipeline that achieves high accuracy and reproducibility.

**Objectives:**

* Build a complete ML pipeline from image to solution using OpenCV, CNNs, and supervised learning.
* Attain strong performance benchmarks in both digit classification and puzzle-solving.
* Implement project structuring, containerization, and reproducibility via MLOps tools.
* Enable scalability and reusability through modular code and Dockerized workflows.

**Success Metrics:**

* Digit classifier accuracy ≥ 98% on MNIST
* Sudoku solver model accuracy ≥ 90% on full puzzle completion
* Complete training and inference reproducibility using Docker

**Project Description:**
This project transforms raw puzzle images into solved Sudoku boards using a two-stage machine learning pipeline. OpenCV is used to preprocess images, correct perspective distortion, and extract individual grid cells. These cells are then classified using a CNN trained on MNIST. The partial puzzle grid is completed by a second model trained on over 50,000 Sudoku boards. Our modular structure, based on Cookiecutter, follows MLOps best practices and ensures consistent execution. The training pipeline is fully automated, datasets are fetched via `gdown`, and model evaluation is logged and visualized. Phase 1 laid the foundation for this system through rigorous model validation and tooling integration.

### 1.2 Selection of Data

**Datasets Used:**

* MNIST: For training the digit classifier
* Kaggle Sudoku Dataset: CSV-formatted puzzles and solutions used for solver model

**Access and Justification:**

* MNIST accessed via `keras.datasets`, chosen for its quality and widespread use
* Sudoku dataset downloaded via `gdown` to ensure automation and reproducibility

**Preprocessing Summary:**

* MNIST: Normalized, reshaped to 28×28×1, and one-hot encoded
* Sudoku: Parsed from CSV to 9×9 grids, normalized, and labels reshaped to integer indices for classification

### 1.3 Model Considerations

**Digit Classifier:**

* Architecture: Two Conv2D layers → MaxPooling → Dropout → Flatten → Dense layers
* Optimized with Adam and categorical crossentropy
* Achieved test accuracy of 98.49%

**Sudoku Solver:**

* Architecture: Stacked Conv2D layers with BatchNormalization → Flatten → Dense(729) → Reshape to (81×9)
* Output: Multiclass softmax across 9 digits per cell
* Evaluation: 95% full-puzzle accuracy on 100 test puzzles

**Why These Models:**

* CNNs are computationally efficient and highly effective for visual recognition tasks
* The solver model’s output design mirrors the puzzle structure and supports parallel digit prediction

### 1.4 Open-Source Tools and Frameworks

* **OpenCV**: For image transformation, warping, thresholding, and contour detection
* **TensorFlow / Keras**: For building and training deep learning models
* **Docker**: For containerized training and prediction environments
* **gdown**: For streamlined dataset downloading from Google Drive
* **Cookiecutter**: For initializing a clean, MLOps-compliant project structure

## 2. Code Organization & Setup

### 2.1 Repository Structure

* Project structured using Cookiecutter for modularity and best practices
* GitHub repository includes logical separation into folders: `/data`, `/models`, `/scripts`, `/tests`, and `/reports`
* Includes `.gitignore`, `Makefile`, and `pyproject.toml` for standardization

### 2.2 Environment Configuration

* Python 3.11 used with virtual environment setup
* All dependencies listed in `requirements.txt`
* Dockerfiles created for training (`Dockerfile.train`) and inference (`Dockerfile.predict`)
* Colab integration validated for GPU training when needed

## 3. Version Control & Team Collaboration

### 3.1 Git Usage

* GitHub repository created under organization account with full history
* Feature branches created for model dev, preprocessing, and automation
* Pull requests merged after peer review and conflict resolution

### 3.2 Team Roles and Responsibilities

* **Nishant Kashyap**: Designed and trained CNN-based digit classifier and solver model
* **Nisarg Jatinkumar Patel**: Built OpenCV-based preprocessing pipeline for digit extraction
* **Narasimha Reddy Putta**: Developed dataset automation, Docker setup, and documentation

Weekly team syncs were held to review progress and assign modular tasks.

## 4. Data Handling

### 4.1 Data Preparation Process

* Data preparation logic implemented in `make_dataset.py`
* MNIST augmented with rotations and shifts using Keras `ImageDataGenerator`
* Sudoku puzzles reshaped, normalized, and converted into model-ready format

### 4.2 Data Documentation

* All data transformations clearly commented and documented
* Sample formats and expected outputs are explained in the README
* Pipeline overview included in the reports section for visualization

## 5. Model Training and Evaluation

### 5.1 Infrastructure Setup

* Training conducted inside Docker containers for full reproducibility
* GPU training via Google Colab and fallback tested locally on CPU
* Training logs, model checkpoints, and metrics stored in `/models`

### 5.2 Baseline Model Results

* **Digit Classifier**: Reached 98.49% accuracy on MNIST test set
* **Sudoku Solver**: Achieved 95% full-puzzle accuracy on a random sample of 100 puzzles
* Metrics visualized using plots and confusion matrices in `/reports`

## 6. Documentation & Reporting

### 6.1 README Overview

* README includes project objectives, architecture diagram, installation steps, and contribution breakdown
* Detailed setup instructions provided for both Docker and Python environments

### 6.2 Code Quality and Style

* Code documented with Google-style docstrings
* Static analysis tools like `ruff` and `mypy` integrated into the workflow
* Makefile includes standardized commands for running, cleaning, and testing the pipeline

---

**Phase 1 Outcome:**
The project has a stable foundation with reproducible pipelines, well-documented models, and automated data workflows. Our team’s focus on modularity, code clarity, and results-driven development positions us strongly for Phase 2, where we will implement CI/CD pipelines, experiment tracking, and deployment workflows.
