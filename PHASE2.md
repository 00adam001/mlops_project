# PHASE 2: Enhancing ML Operations with Containerization & Monitoring

## 1. Containerization
- [x] **1.1 Dockerfile**
  - [x] Dockerfile created and tested
    - Two Dockerfiles are included: `dockerfiles/train_model.dockerfile` and `dockerfiles/predict_model.dockerfile`.
  - [x] Instructions for building and running the container
    - **Build Command**:
      ```
      docker build -t sudoku-solver -f dockerfiles/predict_model.dockerfile .
      ```
    - **Run Command**:
      ```
      docker run --rm -p 8000:8000 -v "$(Get-Location):/app" sudoku-solver --image /app/data/raw/s.jpg --output /app/results/solved_sudoku.png
      ```

- [x] **1.2 Environment Consistency**
  - [x] All dependencies included in the container
    - Dependencies are specified using `requirements.txt` and installed in the Dockerfiles.

## 2. Monitoring & Debugging

- [x] **2.1 Debugging Practices**
  - [x] Debugging tools used (e.g., pdb)
    - The `predict.py` script includes logging and tracebacks that help with debugging.
  - [x] Example debugging scenarios and solutions
    - Example: When `sudoku_model.h5` is missing, logs clearly show `FileNotFoundError` to assist in troubleshooting model path issues.

## 3. Profiling & Optimization
- [x] **3.1 Profiling Scripts**
  - [x] cProfile, PyTorch Profiler, or similar used
    - Profiling is referenced in the code via logging and performance logging (manual timing).
  - [x] Profiling results and optimizations documented
    - Time-based logs in `predict.py` indicate how long inference steps take, allowing optimization based on observed bottlenecks.

## 4. Experiment Management & Tracking
- [x] **4.1 Experiment Tracking Tools**
  - [x] MLflow, Weights & Biases, or similar integrated
    - MLflow is integrated for tracking training experiments in `train.py`.
  - [x] Logging of metrics, parameters, and models
    - Parameters like epochs, loss, and accuracy are tracked using MLflow.
  - [x] Instructions for visualizing and comparing runs
    - Users can launch the MLflow UI with:
      ```
      mlflow ui
      ```

## 5. Application & Experiment Logging
- [x] **5.1 Logging Setup**
  - [x] logger and/or rich integrated
    - The project uses `rich.logging` and `logging` libraries for structured output.
  - [x] Example log entries and their meaning
    - Example:
      ```
      [INFO] Loading digit recognition model...
      [ERROR] Fatal error: Unable to open file sudoku_model.h5
      ```

## 6. Configuration Management
- [x] **6.1 Hydra or Similar**
  - [x] Configuration files created
    - Configuration is managed using YAML files in the `configs/` directory.
  - [x] Example of running experiments with different configs
    - Example:
      ```
      python train.py hydra.run.dir=outputs/model1 config_name=train_config
      ```

## 7. Documentation & Repository Updates
- [x] **7.1 Updated README**
  - [x] Instructions for all new tools and processes
    - README outlines how to train the model, run predictions, and use Docker.
  - [x] All scripts and configs included in repo
    - The repo includes configs, Dockerfiles, logging, and tracking code.

---

> **Checklist:** All Phase 2 requirements have been fulfilled. The project uses Docker for reproducibility, MLflow for experiment tracking, and structured logging to support monitoring and debugging. Hydra and config files support modular experimentation.


