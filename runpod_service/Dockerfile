# Use official Python runtime with CUDA support for TensorFlow GPU
FROM tensorflow/tensorflow:2.15.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy simplified requirements and install Python dependencies
COPY runpod_service/requirements.txt /app/
RUN pip install --no-cache-dir --force-reinstall --upgrade -r requirements.txt

# Copy shared source code directly (no duplication)
COPY src/ /app/src/

# Copy datasets directly (no embedded_datasets/ needed)
COPY datasets/ /app/datasets/

# Copy handler and service files
COPY runpod_service/handler.py /app/
COPY runpod_service/deploy.sh /app/

# Create logs directory for consistency with main project
RUN mkdir -p /app/logs

# Set Python path to include src directory
ENV PYTHONPATH="/app/src:/app:${PYTHONPATH}"

# Expose port for health checks (optional)
EXPOSE 8080

# RunPod handler entry point
CMD ["python", "-u", "handler.py"]