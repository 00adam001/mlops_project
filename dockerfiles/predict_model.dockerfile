# ✅ Use prebuilt TensorFlow image to skip huge pip install
FROM tensorflow/tensorflow:2.14.0

# ✅ Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ✅ Set working directory
WORKDIR /app

# ✅ Copy only requirements first to enable Docker caching
COPY requirements.txt .

# ✅ Install only extra packages not in base TensorFlow image
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Copy your project code AFTER installing packages
COPY se_489_mlops_project/ se_489_mlops_project/
COPY models/ models/
COPY data/ data/

# ✅ Expose Prometheus metrics port or API port
EXPOSE 8000

# ✅ Run your prediction script (FastAPI or any other logic inside)
ENTRYPOINT ["python", "-m", "se_489_mlops_project.predict"]