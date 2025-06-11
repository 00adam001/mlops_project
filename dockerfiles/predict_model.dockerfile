FROM --platform=linux/arm64 tensorflow/tensorflow:2.10.0

# Install system dependencies required by OpenCV, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory to the application's root within the project
WORKDIR /app

# Set pip timeout to a higher value
ENV PIP_DEFAULT_TIMEOUT=1000

# Copy requirements file first to leverage Docker caching from the project root
COPY requirements.txt ./


RUN pip install --no-cache-dir -r requirements.txt

# Copy project files from the project root to the new working directory
COPY se_489_mlops_project/ se_489_mlops_project/
COPY models/ models/
COPY data/ data/

# Expose the port for the Flask application
EXPOSE 8080

# Optional: Check directory contents for debugging
# RUN echo "📂 Models directory contents:" && ls -alh models

# Change the entrypoint to run your Flask app.py using Gunicorn from the current WORKDIR
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 se_489_mlops_project/app:app