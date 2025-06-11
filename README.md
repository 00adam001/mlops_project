
# sudoku-mlops-solver

MLOps pipeline setup for SE 489 — Spring 2025

---

## 1. Team Information

**Team Name:** Team 3

**Team Members:**

- Nishant Kashyap (nkashyap@depaull.edu)
- Nisarg Jatinkumar Patel (npate298@depaul.edu)
- Narasimha Reddy Putta (nputta@depaul.edu)

**Course & Section:** SE489: Machine Learning Engineering for Production (MLOps), Spring 2025

---

## 2. Project Overview

**Brief Summary:**  
This project implements a deep learning pipeline to automate Sudoku solving. Using a combination of CNN-based digit recognition and a custom prediction model, the system extracts Sudoku puzzles from images and returns the completed board. MLOps tools are applied to ensure a scalable, reproducible, and deployable solution.

**Problem Statement and Motivation:**  
Solving Sudoku puzzles by hand or via traditional OCR systems is error-prone when digit quality is low. Our objective is to build a production-grade ML pipeline that takes a raw puzzle image and outputs an accurate solution using automated preprocessing, recognition, and inference.

**Main Objectives:**

- Develop a robust digit recognition model (trained on MNIST)
- Build a Sudoku completion model trained on Kaggle puzzle datasets
- Integrate the system with OpenCV to handle real-world image noise
- Containerize the solution using Docker and structure it with Cookiecutter MLOps template
- Maintain reproducibility and extensibility using Git, Makefiles, and documentation standards

---

## 3. Project Architecture Diagram

```
              ┌────────────────────────────────────┐
              │         Raw Input Image            │
              └──────────────┬─────────────────────┘
                             ▼
              ┌────────────────────────────────────┐
              │      Preprocessing Module          │
              └──────────────┬─────────────────────┘
                             ▼
              ┌────────────────────────────────────┐
              │   Digit Recognition Module         │
              └──────────────┬─────────────────────┘
                             ▼
              ┌────────────────────────────────────┐
              │     Puzzle Solver Module           │
              └──────────────┬─────────────────────┘
                             ▼
              ┌────────────────────────────────────┐
              │     Visualization & Rendering      │
              └────────────────────────────────────┘
```

---

## 4. Phase Deliverables

- [x] **PHASE1.md** – Project Design & Model Development  
- [x] **PHASE2.md** – Containerization, CI/CD, Profiling, Monitoring, MLflow  
- [x] **PHASE3.md** – CML, Docker Artifact Push, Cloud Run Deployment, Final Reporting

---

## 5. Setup Instructions

### Environment Setup
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Training the Model
```bash
docker build -t sudoku-train -f Dockerfile.train .
docker run sudoku-train
```

### Predicting with Trained Model
```bash
docker build -t sudoku-predict -f Dockerfile.predict .
docker run sudoku-predict
```

### Build and Run Final Inference Pipeline
```bash
docker build -t sudoku-solver -f dockerfiles/predict_model.dockerfile .
docker run --rm -p 8000:8000 -v "$PWD":/app sudoku-solver   --image /app/data/raw/s.jpg   --output /app/results/solved_sudoku.png
```

---

## 6. Phase 2: CI/CD, Monitoring, Profiling, Experiment Tracking

- Implemented **CI/CD pipeline** using GitHub Actions
- Set up **pre-commit hooks** and Python linters (`ruff`)
- Added unit tests (`pytest`) and integrated into CI workflow
- **Monitored model metrics** via FastAPI Prometheus endpoint
- **Profiled** inference using `cProfile` + `snakeviz`
- **Tracked experiments** using MLflow
- Used Makefile for automation of testing, linting, profiling, and training
- Dockerized all steps with separate Dockerfiles for training and inference

---

## 7. Phase 3: CML & Cloud Deployment

- Implemented **Continuous Machine Learning** with `cml`
- Evaluated models and published GitHub PR comments with `metrics.json`
- Trained model locally and pushed Docker image to **GCP Artifact Registry**
- Deployed the model to **Google Cloud Run** using `gcloud run deploy`
- Skipped Cloud Functions; used **FastAPI** app containerized for REST API
- Verified Sudoku solving results through API with input/output visuals

### Deployment Commands Used

**Authenticate and Configure Docker with GCP:**
```bash
echo "${{ secrets.GCP_SA_KEY }}" > key.json
gcloud auth activate-service-account --key-file=key.json
gcloud auth configure-docker us-central1-docker.pkg.dev
```

**Build and Tag Image:**
```bash
docker build -f Dockerfile.predict -t image-caption-model .
docker tag image-caption-model us-central1-docker.pkg.dev/dvc-drive-459219/image-captioning-repo/image-caption-model
```

**Push Docker Image:**
```bash
docker push us-central1-docker.pkg.dev/dvc-drive-459219/image-captioning-repo/image-caption-model
```

**Deploy via Cloud Run:**
```bash
gcloud run deploy image-caption-model \
  --image=us-central1-docker.pkg.dev/dvc-drive-459219/image-captioning-repo/image-caption-model \
  --platform=managed \
  --region=us-central1 \
  --allow-unauthenticated
```

---

## 8. Contribution Summary

- **Nishant Kashyap:** Designed and trained CNN models for digit classification and Sudoku solving.
- **Nisarg Jatinkumar Patel:** Developed the image preprocessing pipeline using OpenCV and digit extraction logic, ML workflows, monitoring setup.
- **Narasimha Reddy Putta:** Led data acquisition and Docker containerization tasks.

---

## 9. References

- MNIST Dataset – Handwritten digit training data  
- Kaggle Sudoku Dataset  
- gdown – Google Drive download automation  
- OpenCV – Image manipulation and transformation  
- `skimage.segmentation.clear_border` – Border cleaning in image preprocessing  
- TensorFlow / Keras – Model development and training  
