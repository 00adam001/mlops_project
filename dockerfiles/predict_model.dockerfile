# Use Python base image
FROM python:3.11-slim

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy only requirements file first to leverage caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY se_489_mlops_project/ se_489_mlops_project/
COPY models/ models/
COPY data/ data/

# Debug: Confirm model directory (optional)
RUN echo "📂 Models:" && ls -alh /app/models

# Run prediction module (without API or UI)
ENTRYPOINT ["python", "-m", "se_489_mlops_project.predict"]