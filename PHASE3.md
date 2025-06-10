
# PHASE 3: Continuous Machine Learning (CML) & Deployment

## 1. Continuous Integration & Testing

- [x] **1.1 Unit Testing with pytest**
  - Three test scripts created in `tests/test_sudoku_solver.py` for image preprocessing, dummy model prediction, and image loading.
  - Tests are CI-safe (do not require model files) and executed successfully locally and in GitHub Actions.
  - Documented with comments and structured test logic.

  **Run Locally:**
  ```bash
  pytest tests/
  ```

- [x] **1.2 GitHub Actions Workflows**
  - `CI.yml` configured to run on push and pull requests to `main`.
  - Runs linting with `ruff`, installs dependencies, and runs `pytest`.
  - Includes Python 3.11 setup and error-safe linting/test flows.

  **CI.yml snippet:**
  ```yaml
  jobs:
    ci:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: '3.11'
        - name: Install dependencies
          run: |
            pip install -r requirements.txt
            pip install pytest ruff
        - name: Lint with ruff
          run: ruff check .
        - name: Run tests
          run: pytest tests/
  ```

- [x] **1.3 Pre-commit Hooks**
  - `.pre-commit-config.yaml` present and used for linting and formatting checks.
  - Ruff checks and unused import detection configured and documented.

  **Run Pre-commit Checks:**
  ```bash
  pre-commit install
  pre-commit run --all-files
  ```

## 2. Continuous Docker Building & CML

- [x] **2.1 Docker Image Automation**
  - Dockerfiles (`predict_model.dockerfile`, `train_model.dockerfile`) created.
  - GitHub Actions workflow builds and pushes image to Artifact Registry.
  - Deployment step commented out with clear explanation after Cloud Run success.

  **Build Docker Image Locally:**
  ```bash
  docker build -f Dockerfile.predict -t image-caption-model .
  ```

  **Tag for Artifact Registry:**
  ```bash
  docker tag image-caption-model us-central1-docker.pkg.dev/dvc-drive-459219/image-captioning-repo/image-caption-model
  ```

  **Push Image:**
  ```bash
  docker push us-central1-docker.pkg.dev/dvc-drive-459219/image-captioning-repo/image-caption-model
  ```

  **Artifact Registry Auth via GitHub Actions:**
  ```bash
  echo "${{ secrets.GCP_SA_KEY }}" > key.json
  gcloud auth activate-service-account --key-file=key.json
  gcloud auth configure-docker us-central1-docker.pkg.dev
  ```

- [x] **2.2 Continuous Machine Learning (CML)**
  - `cml.yml` GitHub Actions workflow included for evaluation.
  - Metrics (`metrics.json`, `confusion_matrix.png`) generated locally and committed.
  - `cml-publish` used to publish evaluation results in GitHub PR.
  - GCP service account JSON stored in `GCP_SA_KEY` secret.
  - DVC authentication and pull configured in CI pipeline using that secret.

  **Evaluate Model and Generate Report:**
  ```bash
  python evaluate.py --save-metrics
  ```

  **CML GitHub Actions Workflow Snippet:**
  ```yaml
  jobs:
    cml:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: '3.11'
        - name: Install dependencies
          run: |
            pip install -r requirements.txt
            pip install cml
        - name: Evaluate
          run: python evaluate.py --save-metrics
        - name: Comment PR
          env:
            REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          run: |
            echo "## Evaluation Report" > report.md
            cat metrics.json >> report.md
            cml comment create report.md
  ```

## 3. Deployment on Google Cloud Platform (GCP)

- [x] **3.1 GCP Artifact Registry**
  - Docker image tagged and pushed using GitHub Actions.
  - Registry path: `us-central1-docker.pkg.dev/dvc-drive-459219/image-captioning-repo/image-caption-model`.

- [ ] **3.2 Custom Training Job on GCP**
  - Not implemented. All model training was performed locally.

- [x] **3.3 Deploying API with FastAPI & GCP Cloud Functions**
  - FastAPI app defined in `predict.py` using Uvicorn.
  - Not deployed via Cloud Functions — deployed using Cloud Run.

  **Run FastAPI Locally:**
  ```bash
  uvicorn predict:app --host 0.0.0.0 --port 8080
  ```

- [x] **3.4 Dockerize & Deploy Model with GCP Cloud Run**
  - Dockerfile serves model using FastAPI on port 8080.
  - `gcloud run deploy` used in GitHub Actions.
  - Deployment is commented out post-deployment to avoid CI re-deployment.

  ![Cloud Analytic showing deployment metrics](Images/Screenshot%202025-06-10%20153950.png)
  ![Cloud Analytic showing deployment metrics](Images/Screenshot%202025-06-10%20154043.png)

  **Deploy via Cloud Run:**
  ```bash
  gcloud run deploy image-caption-model \
    --image=us-central1-docker.pkg.dev/dvc-drive-459219/image-captioning-repo/image-caption-model \
    --platform=managed \
    --region=us-central1 \
    --allow-unauthenticated
  ```

  ![API Running](Images/Screenshot%202025-06-10%20154018.png)
  ![Cloud Log](Images/Screenshot%202025-06-10%20154110.png)

- [ ] **3.5 Interactive UI Deployment**
  - Not implemented.
  - No Streamlit or Gradio demo deployed.

## 4. Documentation & Repository Updates

- [x] **4.1 Comprehensive README**
  - Setup, usage, CI/CD, and deployment steps documented.
  - Includes Docker, FastAPI, DVC, and CML usage.
  - Screenshots and confusion matrix included.

- [x] **4.2 Resource Cleanup Reminder**
  - Comment added in GitHub Actions: Deployment is disabled post-Cloud Run setup.
  - Reminder to shut down instances and avoid extra billing.

  ![Working Model](Images/Screenshot%202025-06-10%20153807.png)

  **Left: Original Sudoku | Right: Solved output by model**

---

> **Checklist:** All CI/CD and CML integration is production-ready. Deployment is stable and well-documented. Evaluation artifacts are available and test coverage is ensured in CI.
