# sudoku-mlops-solver

MLOps pipeline setup for SE 489 — Spring 2025

## 1. Team Information

* **Team Name:** Team 3
* **Team Members (Name & Email):**

    * Nishant Kashyap (nkashyap@depaull.edu)
    * Nisarg Jatinkumar Patel (npate298@depaul.edu)
    * Narasimha Reddy Putta (nputta@depaul.edu)
* **Course & Section:** SE489: Machine Learning Engineering for Production (MLOps), Spring 2025

## 2. Project Overview

* **Brief Summary:**
  This project implements a deep learning pipeline to automate Sudoku solving. Using a combination of CNN-based digit recognition and a custom prediction model, the system extracts Sudoku puzzles from images and returns the completed board. MLOps tools are applied to ensure a scalable, reproducible, and deployable solution.

* **Problem Statement and Motivation:**
  Solving Sudoku puzzles by hand or via traditional OCR systems is error-prone when digit quality is low. Our objective is to build a production-grade ML pipeline that takes a raw puzzle image and outputs an accurate solution using automated preprocessing, recognition, and inference.

* **Main Objectives:**

    * Develop a robust digit recognition model (trained on MNIST).
    * Build a Sudoku completion model trained on Kaggle puzzle datasets.
    * Integrate the system with OpenCV to handle real-world image noise.
    * Containerize the solution using Docker and structure it with Cookiecutter MLOps template.
    * Maintain reproducibility and extensibility using Git, Makefiles, and documentation standards.

## 3. Project Architecture Diagram

                      ┌────────────────────────────────────┐
                      │         Raw Input Image            │
                      │  - Captured or uploaded Sudoku     │
                      │  - Format: .jpg / .png             │
                      └──────────────┬─────────────────────┘
                                     │
                                     ▼
                      ┌────────────────────────────────────┐
                      │      Preprocessing Module          │
                      │  Tools: OpenCV                     │
                      │  - Convert to grayscale            │
                      │  - Gaussian blurring               │
                      │  - Adaptive thresholding           │
                      │  - Detect largest contour (grid)   │
                      │  - Warp perspective to 9x9 square  │
                      └──────────────┬─────────────────────┘
                                     │
                                     ▼
                      ┌────────────────────────────────────┐
                      │   Digit Recognition Module         │
                      │  Model: CNN (Trained on MNIST)     │
                      │  - Classifies digits 0–9           │
                      │  - Handles 28x28 resized patches   │
                      │  Output: Sparse 9x9 grid matrix    │
                      └──────────────┬─────────────────────┘
                                     │
                                     ▼
                      ┌────────────────────────────────────┐
                      │     Puzzle Solver Module           │
                      │  Model: Deep CNN (Kaggle Sudoku)   │
                      │  - Predicts missing values         │
                      │  - Output shape: (81, 9)           │
                      │  - Argmax + reshape to 9x9         │
                      └──────────────┬─────────────────────┘
                                     │
                                     ▼
                      ┌────────────────────────────────────┐
                      │     Visualization & Rendering      │
                      │  - Overlay solved digits on image  │
                      │  - Use OpenCV for rendering        │
                      │  - Output: Completed Sudoku image  │
                      └────────────────────────────────────┘

## 4. Phase Deliverables

* [PHASE1.md](./PHASE1.md): Project Design & Model Development

## 5. Setup Instructions

* **Environment Setup:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

* **Running the Code:**

  **Train the Model**

  ```bash
  docker build -t sudoku-train -f Dockerfile.train .
  docker run sudoku-train
  ```

  **Predict with Trained Model**

  ```bash
  docker build -t sudoku-predict -f Dockerfile.predict .
  docker run sudoku-predict
  ```

  **How to run The monitoring ,profiling and tracking**
  Building image "docker build -t sudoku-solver -f dockerfiles/predict_model.dockerfile ."
  
  Run this script " docker run --rm -p 8000:8000 -v "$PWD":/app sudoku-solver \
  --image /app/data/raw/s.jpg \                  
  --output /app/results/solved_sudoku.png"

  Monitoring :- "http://0.0.0.0:8000/metrics "  

  Profiling :- "snakeviz profile_output.prof "

  Tracking :- "mlflow ui" -- then navigate to "http://127.0.0.1:5000 "


  FOR MAC :- curl http://localhost:8000/metrics | grep sudoku_


## 6. Contribution Summary

* **Nishant Kashyap:** Designed and trained CNN models for digit classification and Sudoku solving.
* **Nisarg Jatinkumar Patel:** Developed the image preprocessing pipeline using OpenCV and digit extraction logic , ML workflows , Monitoring.
* **Narasimha Reddy Putta:** Led data acquisition, Docker containerization.

## 7. References

* [MNIST Dataset](https://keras.io/api/datasets/mnist/) – Handwritten digit training data
* [Kaggle Sudoku Dataset](https://www.kaggle.com/datasets/bryanpark/sudoku)
* `gdown` – Google Drive download automation
* `OpenCV` – Image manipulation and transformation
* `skimage.segmentation.clear_border` – Border cleaning in image preprocessing
* `TensorFlow` / `Keras` – Model development and training

---