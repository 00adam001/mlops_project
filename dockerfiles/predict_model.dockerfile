FROM python:3.11-slim

# Install system dependencies required by OpenCV, TensorFlow, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY se_489_mlops_project/ se_489_mlops_project/
COPY models/ models/
COPY data/ data/

# Expose Prometheus metrics port
EXPOSE 8000

# Debug: Show what's in models directory
RUN echo "📂 Models directory contents:" && ls -alh /app/models

# Change the entrypoint to run your predict.py script
CMD ["python", "se_489_mlops_project/predict.py"]