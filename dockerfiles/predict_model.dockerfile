# Use official TensorFlow base image with all necessary libs
FROM tensorflow/tensorflow:2.14.0

# Install system-level dependencies required by OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy dependency management files
COPY requirements.txt .
COPY pyproject.toml .

# Copy application code
COPY se_489_mlops_project/ se_489_mlops_project/
COPY data/ data/
COPY models/ models/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install . --no-deps --no-cache-dir

# Entry point
ENTRYPOINT ["python", "-u", "se_489_mlops_project/predict.py"]