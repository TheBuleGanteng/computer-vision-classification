#!/bin/bash

# Docker build and push script for hyperparameter optimization server
# Author: thebuleganteng
# Repository: thebuleganteng/hyperparameter-optimizer

set -e  # Exit on any error

# Configuration
DOCKER_USERNAME="thebuleganteng"
REPO_NAME="hyperparameter-optimizer"
DOCKER_REPO="${DOCKER_USERNAME}/${REPO_NAME}"

# Get current timestamp for versioning
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "🚀 Starting Docker build and push process..."
echo "Repository: ${DOCKER_REPO}"
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Login to Docker Hub (will prompt for password if not logged in)
echo "🔐 Logging into Docker Hub..."
docker login

echo ""
echo "🏗️  Building images..."

# Build local development image (python:3.12-slim based)
echo "📦 Building local development image..."
docker build -t ${DOCKER_REPO}:dev-${TIMESTAMP} -f Dockerfile .
docker tag ${DOCKER_REPO}:dev-${TIMESTAMP} ${DOCKER_REPO}:dev-latest

# Build production image (CUDA based for RunPod)
echo "📦 Building production image..."
docker build -t ${DOCKER_REPO}:prod-${TIMESTAMP} -f Dockerfile.production .
docker tag ${DOCKER_REPO}:prod-${TIMESTAMP} ${DOCKER_REPO}:latest
docker tag ${DOCKER_REPO}:prod-${TIMESTAMP} ${DOCKER_REPO}:production

echo ""
echo "📤 Pushing images to Docker Hub..."

# Push timestamped versions
echo "Pushing timestamped development image..."
docker push ${DOCKER_REPO}:dev-${TIMESTAMP}

echo "Pushing timestamped production image..."
docker push ${DOCKER_REPO}:prod-${TIMESTAMP}

# Push latest tags
echo "Pushing latest tags..."
docker push ${DOCKER_REPO}:dev-latest
docker push ${DOCKER_REPO}:latest
docker push ${DOCKER_REPO}:production

echo ""
echo "✅ Build and push completed successfully!"
echo ""
echo "📋 Available images:"
echo "  - ${DOCKER_REPO}:latest (production/RunPod)"
echo "  - ${DOCKER_REPO}:production (production/RunPod)"
echo "  - ${DOCKER_REPO}:dev-latest (local development)"
echo "  - ${DOCKER_REPO}:prod-${TIMESTAMP} (timestamped production)"
echo "  - ${DOCKER_REPO}:dev-${TIMESTAMP} (timestamped development)"
echo ""
echo "🎯 For RunPod deployment, use: ${DOCKER_REPO}:latest"
echo ""
echo "🔧 Next steps:"
echo "  1. Create RunPod template with image: ${DOCKER_REPO}:latest"
echo "  2. Set container start command: python3 src/api_server.py"
echo "  3. Expose HTTP port: 8000"
echo "  4. Configure volume mounts for /app/optimization_results"