# # Use official TensorFlow base image with all necessary libs
# FROM tensorflow/tensorflow:2.14.0

# # Install system-level dependencies required by OpenCV
# RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && apt-get clean && rm -rf /var/lib/apt/lists/*

# # Set the working directory
# WORKDIR /app

# # Copy dependency management files
# COPY requirements.txt .
# COPY pyproject.toml .

# # Copy application code
# COPY se_489_mlops_project/ se_489_mlops_project/
# COPY data/ data/
# COPY models/ models/

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install . --no-deps --no-cache-dir

# # Entry point
# ENTRYPOINT ["python", "-u", "se_489_mlops_project/predict.py"]

# ✅ Use prebuilt TensorFlow image to skip huge pip install
# FROM tensorflow/tensorflow:2.14.0

# # ✅ Install system dependencies for OpenCV
# RUN apt-get update && apt-get install -y \
#     libgl1 \
#     libglib2.0-0 \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*

# # ✅ Set working directory
# WORKDIR /app

# # ✅ Copy only requirements first to enable Docker caching
# COPY requirements.txt .

# # ✅ Install only extra packages not in base TensorFlow image
# RUN pip install --no-cache-dir -r requirements.txt

# # ✅ Copy your project code AFTER installing packages
# COPY se_489_mlops_project/ se_489_mlops_project/
# COPY models/ models/
# COPY data/ data/

# # ✅ Expose Prometheus metrics port
# EXPOSE 8000

# # ✅ Run your prediction script
# ENTRYPOINT ["python", "-m", "se_489_mlops_project.predict"]

# Base image with preinstalled scientific libraries
# FROM python:3.11-slim

# # Install system dependencies for OpenCV and other libs
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     gcc \
#     libgl1 \
#     libglib2.0-0 \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*

# # Set working directory
# WORKDIR /app

# # Copy only requirements first to leverage Docker layer caching
# COPY requirements.txt .

# # ✅ Install Python dependencies with caching enabled
# RUN pip install -r requirements.txt

# # ✅ Copy the project files (only after dependencies to optimize caching)
# COPY se_489_mlops_project/ se_489_mlops_project/
# COPY models/ models/
# COPY data/ data/

# EXPOSE 8000

# # ✅ Optional: confirm contents of model directory for debugging
# RUN echo "📂 Models directory contents:" && ls -alh /app/models

# # ✅ Define entrypoint to run the prediction module
# ENTRYPOINT ["python", "-m", "se_489_mlops_project.predict"]



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

# ✅ Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Copy project files after dependencies (for caching efficiency)
COPY se_489_mlops_project/ se_489_mlops_project/
COPY models/ models/
COPY data/ data/

# ✅ Expose Prometheus metrics port
EXPOSE 8000

# ✅ Debug: Show what's in models directory (optional but helpful)
RUN echo "📂 Models directory contents:" && ls -alh /app/models

# ✅ Set module entrypoint
ENTRYPOINT ["python", "-m", "se_489_mlops_project.predict"]
