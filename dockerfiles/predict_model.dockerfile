FROM python:3.11-slim

# Install system dependencies
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

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY se_489_mlops_project/ se_489_mlops_project/
COPY models/ models/
COPY data/ data/

# Expose metrics port (Prometheus) and default Cloud Run port
EXPOSE 8000
EXPOSE 8080

# Debug: list model contents
RUN echo "📂 Models directory contents:" && ls -alh /app/models

# Start the FastAPI server using Uvicorn
CMD ["uvicorn", "se_489_mlops_project.predict:app", "--host", "0.0.0.0", "--port", "8080"]