#!/bin/bash

# Docker build and push script for hyperparameter optimization server
# Author: thebuleganteng
# Repository: thebuleganteng/hyperparameter-optimizer
# Run using: ./docker_build_script.sh [option]
# Arguments:
#   true: builds both dev and production images (full build)
#   false: builds only dev image
#   quick: builds only production code update (fast for testing)

set -e  # Exit on any error

# Configuration
DOCKER_USERNAME="thebuleganteng"
REPO_NAME="hyperparameter-optimizer"
DOCKER_REPO="${DOCKER_USERNAME}/${REPO_NAME}"

# Parse arguments
BUILD_OPTION=${1:-false}

# Validate argument
case "$BUILD_OPTION" in
    true|false|quick)
        ;; # Valid options, do nothing
    *)
        echo "❌ Error: Invalid argument. Use 'true', 'false', or 'quick'"
        echo "Usage: ./docker_build_script.sh [option]"
        echo "  true: builds both dev and production images (full build)"
        echo "  false: builds only dev image"
        echo "  quick: builds only production code update (fast for testing)"
        exit 1
        ;;
esac

# Get current timestamp for versioning
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "🚀 Starting Docker build and push process..."
echo "Repository: ${DOCKER_REPO}"
echo "Build option: ${BUILD_OPTION}"
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

# Handle quick build option
if [[ "$BUILD_OPTION" == "quick" ]]; then
    echo "⚡ Quick code update build (production only)..."
    echo "📦 Building production image with code updates..."
    
    # Build only the code-update stage (reuses cached base layers)
    docker build --target code-update -t ${DOCKER_REPO}:quick-${TIMESTAMP} -f Dockerfile.production .
    docker tag ${DOCKER_REPO}:quick-${TIMESTAMP} ${DOCKER_REPO}:latest
    docker tag ${DOCKER_REPO}:quick-${TIMESTAMP} ${DOCKER_REPO}:production
    
    echo "📤 Pushing quick update to Docker Hub..."
    docker push ${DOCKER_REPO}:quick-${TIMESTAMP}
    docker push ${DOCKER_REPO}:latest
    docker push ${DOCKER_REPO}:production
    
    echo ""
    echo "⚡ Quick build completed successfully!"
    echo ""
    echo "📋 Updated images:"
    echo "  - ${DOCKER_REPO}:latest (production/RunPod - quick update)"
    echo "  - ${DOCKER_REPO}:production (production/RunPod - quick update)"
    echo "  - ${DOCKER_REPO}:quick-${TIMESTAMP} (timestamped quick update)"
    echo ""
    echo "🎯 RunPod will use the updated :latest image with your code changes"
    echo "⏱️  This was much faster because it reused cached base layers!"
    exit 0
fi

# Regular build process (existing logic)
echo "🏗️  Building images..."

# Build local development image (python:3.12-slim based)
echo "📦 Building local development image..."
docker build -t ${DOCKER_REPO}:dev-${TIMESTAMP} -f Dockerfile .
docker tag ${DOCKER_REPO}:dev-${TIMESTAMP} ${DOCKER_REPO}:dev-latest

# Build production image only if BUILD_OPTION=true
if [[ "$BUILD_OPTION" == "true" ]]; then
    echo "📦 Building production image (full build)..."
    docker build --target full -t ${DOCKER_REPO}:prod-${TIMESTAMP} -f Dockerfile.production .
    docker tag ${DOCKER_REPO}:prod-${TIMESTAMP} ${DOCKER_REPO}:latest
    docker tag ${DOCKER_REPO}:prod-${TIMESTAMP} ${DOCKER_REPO}:production
fi

echo ""
echo "📤 Pushing images to Docker Hub..."

# Push timestamped versions
echo "Pushing timestamped development image..."
docker push ${DOCKER_REPO}:dev-${TIMESTAMP}

if [[ "$BUILD_OPTION" == "true" ]]; then
    echo "Pushing timestamped production image..."
    docker push ${DOCKER_REPO}:prod-${TIMESTAMP}
fi

# Push latest tags
echo "Pushing latest tags..."
docker push ${DOCKER_REPO}:dev-latest

if [[ "$BUILD_OPTION" == "true" ]]; then
    docker push ${DOCKER_REPO}:latest
    docker push ${DOCKER_REPO}:production
fi

echo ""
echo "✅ Build and push completed successfully!"
echo ""
echo "📋 Available images:"
if [[ "$BUILD_OPTION" == "true" ]]; then
    echo "  - ${DOCKER_REPO}:latest (production/RunPod)"
    echo "  - ${DOCKER_REPO}:production (production/RunPod)"
    echo "  - ${DOCKER_REPO}:dev-latest (local development)"
    echo "  - ${DOCKER_REPO}:prod-${TIMESTAMP} (timestamped production)"
    echo "  - ${DOCKER_REPO}:dev-${TIMESTAMP} (timestamped development)"
else
    echo "  - ${DOCKER_REPO}:dev-latest (local development)"
    echo "  - ${DOCKER_REPO}:dev-${TIMESTAMP} (timestamped development)"
fi
echo ""
if [[ "$BUILD_OPTION" == "true" ]]; then
    echo "🎯 For RunPod deployment, use: ${DOCKER_REPO}:latest"
    echo ""
    echo "🔧 Next steps:"
    echo "  1. Create RunPod template with image: ${DOCKER_REPO}:latest"
    echo "  2. Set container start command: python3 src/api_server.py"
    echo "  3. Expose HTTP port: 8000"
    echo "  4. Configure volume mounts for /app/optimization_results"
else
    echo "🔧 Development build complete."
    echo "💡 For quick production updates: ./docker_build_script.sh quick"
    echo "💡 For full production build: ./docker_build_script.sh true"
fi