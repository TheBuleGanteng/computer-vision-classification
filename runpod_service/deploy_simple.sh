#!/bin/bash

# Simplified RunPod Service Deployment Script
# Builds and pushes container to Docker Hub for RunPod deployment
# Tests are commented out for faster deployment

set -e  # Exit on any error

echo "🚀 Simplified RunPod Container Build & Push"
echo "============================================"

# Generate unique image tag
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
UNIQUE_TAG="v${TIMESTAMP}-${GIT_HASH}"

# Configuration with unique tag
IMAGE_NAME="cv-classification-optimizer"
TAG="${UNIQUE_TAG}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load environment variables from .env file
echo -e "${BLUE}🔍 Loading environment variables...${NC}"

# Detect if we're running from root or runpod_service directory
if [ -f ".env" ]; then
    ENV_FILE=".env"
    echo -e "${BLUE}📁 Running from project root directory${NC}"
elif [ -f "../.env" ]; then
    ENV_FILE="../.env"
    echo -e "${BLUE}📁 Running from runpod_service directory${NC}"
else
    echo -e "${RED}❌ .env file not found${NC}"
    echo -e "${YELLOW}💡 Please ensure .env file exists in project root with:${NC}"
    echo "DOCKERHUB_USERNAME=your_dockerhub_username"
    exit 1
fi

# Source the .env file
set -a  # Mark variables for export
source "$ENV_FILE"
set +a  # Stop marking variables for export

# Validate required environment variables
if [ -z "$DOCKERHUB_USERNAME" ]; then
    echo -e "${RED}❌ DOCKERHUB_USERNAME not found in .env file${NC}"
    exit 1
fi

echo -e "${BLUE}👤 Docker Hub Username: ${DOCKERHUB_USERNAME}${NC}"
echo -e "${BLUE}🏷️  Image Tag: ${UNIQUE_TAG}${NC}"

# Check Docker status
echo -e "${BLUE}🔍 Checking Docker status...${NC}"
if ! docker --version > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not installed or not running${NC}"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker daemon is not running${NC}"
    echo -e "${YELLOW}💡 Please start Docker and try again${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker is running${NC}"

# Change to runpod_service directory if not already there
if [ ! -f "Dockerfile" ]; then
    if [ -d "runpod_service" ]; then
        echo -e "${BLUE}📁 Changing to runpod_service directory${NC}"
        cd runpod_service
    else
        echo -e "${RED}❌ Cannot find Dockerfile or runpod_service directory${NC}"
        exit 1
    fi
fi

# Build Docker image (using parent directory as build context, like deploy.sh)
echo -e "${BLUE}🏗️  Building Docker image: ${FULL_IMAGE_NAME}...${NC}"
echo -e "${BLUE}🏷️  Using unique tag: ${UNIQUE_TAG}${NC}"

# Go up to parent directory and build with correct context
cd ..
if docker build -f runpod_service/Dockerfile -t "$FULL_IMAGE_NAME" .; then
    cd runpod_service  # Change back to runpod_service directory
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
    
    # Show image info
    IMAGE_SIZE=$(docker images --format "table {{.Size}}" "$FULL_IMAGE_NAME" | tail -n 1)
    echo -e "${BLUE}📏 Image size: ${IMAGE_SIZE}${NC}"
else
    echo -e "${RED}❌ Failed to build Docker image${NC}"
    exit 1
fi

# Tag for Docker Hub
REMOTE_IMAGE="${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}"
echo -e "${BLUE}🏷️  Tagging for Docker Hub: ${REMOTE_IMAGE}${NC}"
docker tag "$FULL_IMAGE_NAME" "$REMOTE_IMAGE"

# Push to Docker Hub
echo -e "${BLUE}⬆️  Pushing to Docker Hub...${NC}"
if docker push "$REMOTE_IMAGE"; then
    echo -e "${GREEN}✅ Image pushed to Docker Hub successfully${NC}"
    echo ""
    echo -e "${GREEN}🎉 DEPLOYMENT COMPLETE!${NC}"
    echo "========================="
    echo ""
    echo -e "${BLUE}📦 Container Image Name:${NC}"
    echo -e "${YELLOW}${REMOTE_IMAGE}${NC}"
    echo ""
    echo -e "${BLUE}📋 Next Steps:${NC}"
    echo "1. Go to RunPod Console: https://www.runpod.io/console/serverless"
    echo "2. Find your endpoint: ${ENDPOINT_ID_RUNPOD:-your-endpoint}"
    echo "3. Update the Docker image to:"
    echo -e "   ${YELLOW}${REMOTE_IMAGE}${NC}"
    echo "4. Test the updated endpoint"
    echo ""
else
    echo -e "${RED}❌ Failed to push image to Docker Hub${NC}"
    echo -e "${YELLOW}⚠️  Please check Docker Hub credentials and try again${NC}"
    exit 1
fi

# ============================================================================
# COMMENTED OUT: All testing sections removed for faster deployment
# ============================================================================
# 
# # Test the image locally first
# echo -e "${BLUE}🧪 Testing Docker image locally...${NC}"
# [... all testing code commented out ...]
#
# # End-to-end testing  
# echo -e "${BLUE}🧪 Running end-to-end test...${NC}"
# [... all E2E testing code commented out ...]
#
# ============================================================================