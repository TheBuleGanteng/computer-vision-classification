# Use Python 3.12 base image to match your venv
FROM python:3.12-slim

# Uncomment below for production/RunPod deployment with GPU support
# FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app

# Install system dependencies (simplified for Python base image)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libhdf5-dev \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# For CUDA base image, use these dependencies instead:
# RUN apt-get update && apt-get install -y \
#     python3 \
#     python3-pip \
#     python3-dev \
#     git \
#     wget \
#     curl \
#     build-essential \
#     libhdf5-dev \
#     pkg-config \
#     && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# For CUDA base image, use python3 instead:
# RUN pip3 install --no-cache-dir --upgrade pip
# RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs
RUN mkdir -p /app/optimization_results
RUN mkdir -p /app/datasets

# Set permissions
RUN chmod +x /app/src/api_server.py

# Expose the port that FastAPI will run on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application from src directory
CMD ["python", "src/api_server.py"]

# For CUDA base image, use python3 instead:
# CMD ["python3", "api_server.py"]