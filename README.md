# sudoku-mlops-solver

MLOps pipeline setup for SE 489 — Spring 2025

## 1. Team Information

* **Team Name:** Team 3
* **Team Members (Name & Email):**

    * Nishant Kashyap
    * Nisarg Jatinkumar Patel
    * Narasimha Reddy Putta
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

* [x] Inserted architecture diagram as a visual PNG file (see below).

![Project Architecture Diagram](reports/figures/architecture_diagram.png)

> This diagram illustrates the complete end-to-end machine learning workflow:
>
> * **Input Layer:** Raw image of a Sudoku puzzle
> * **Preprocessing Module:** OpenCV-based image processing and grid detection
> * **Digit Recognition Module:** CNN-based digit classification (trained on MNIST)
> * **Puzzle Solver Module:** Inference model trained on Kaggle datasets
> * **Visualization & Output:** Displays the completed puzzle
>
> The diagram also reflects key MLOps components: dataset pipeline, containerized scripts, and modular code organization supporting continuous integration and deployment.

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

  **Dataset Download Script**

  ```python
  import gdown
  import os

  def download_from_gdrive(file_path='data/raw/sudoku.csv', file_id='12c_UTy7pXdzJkuL1HfVdaTP15Z8QZgA2'):
      if not os.path.exists(file_path):
          os.makedirs(os.path.dirname(file_path), exist_ok=True)
          gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
  ```

## 6. Contribution Summary

* **Nishant Kashyap:** Designed and trained CNN models for digit classification and Sudoku solving.
* **Nisarg Jatinkumar Patel:** Developed the image preprocessing pipeline using OpenCV and digit extraction logic.
* **Narasimha Reddy Putta:** Led data acquisition, Docker containerization, and MLOps workflow setup.

## 7. References

* [MNIST Dataset](https://keras.io/api/datasets/mnist/) – Handwritten digit training data
* [Kaggle Sudoku Dataset](https://www.kaggle.com/datasets/bryanpark/sudoku)
* `gdown` – Google Drive download automation
* `OpenCV` – Image manipulation and transformation
* `skimage.segmentation.clear_border` – Border cleaning in image preprocessing
* `TensorFlow` / `Keras` – Model development and training

---

**Tip:** Keep this README updated as your project evolves. Link to each phase deliverable and update the architecture diagram as your pipeline matures.
