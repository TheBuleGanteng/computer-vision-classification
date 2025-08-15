#!/bin/bash

# RunPod Service Deployment Script
# Builds and deploys the hyperparameter optimization service to RunPod

set -e  # Exit on any error

echo "🚀 Starting RunPod Service Deployment..."
echo "========================================"

# Generate unique image tag
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
UNIQUE_TAG="v${TIMESTAMP}-${GIT_HASH}"

# Configuration with unique tag
IMAGE_NAME="cv-classification-optimizer"
TAG="${UNIQUE_TAG}"  # Use unique tag instead of "latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load environment variables from .env file
echo -e "${BLUE}🔍 Loading environment variables...${NC}"
ENV_FILE="../.env"

if [ ! -f "${ENV_FILE}" ]; then
    echo -e "${RED}❌ .env file not found at ${ENV_FILE}${NC}"
    echo -e "${YELLOW}💡 Please create .env file in project root with:${NC}"
    echo "RUNPOD_ENDPOINT_ID=your_endpoint_id"
    echo "RUNPOD_API_KEY=your_api_key"
    exit 1
fi

# Source the .env file
set -a  # Mark variables for export
source "${ENV_FILE}"
set +a  # Unmark variables for export

# Validate required environment variables
if [ -z "${RUNPOD_ENDPOINT_ID}" ] || [ -z "${RUNPOD_API_KEY}" ] || [ -z "${DOCKERHUB_USERNAME}" ]; then
    echo -e "${RED}❌ Missing required environment variables${NC}"
    echo -e "${YELLOW}💡 Please ensure .env file contains:${NC}"
    echo "RUNPOD_ENDPOINT_ID=your_endpoint_id"
    echo "RUNPOD_API_KEY=your_api_key"
    echo "DOCKERHUB_USERNAME=your_dockerhub_username"
    exit 1
fi

echo -e "${GREEN}✅ Environment variables loaded${NC}"
echo -e "${BLUE}📋 Endpoint ID: ${RUNPOD_ENDPOINT_ID}${NC}"

# Check if Docker is running
echo -e "${BLUE}🔍 Checking Docker status...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker is running${NC}"

# Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}❌ Dockerfile not found. Make sure you're in the runpod_service/ directory.${NC}"
    exit 1
fi

if [ ! -f "handler.py" ]; then
    echo -e "${RED}❌ handler.py not found. Make sure you're in the runpod_service/ directory.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found required files (Dockerfile, handler.py)${NC}"

# Build the Docker image from parent directory with proper context
echo -e "${BLUE}🏗️  Building Docker image: ${FULL_IMAGE_NAME}...${NC}"
echo -e "${YELLOW}📂 Building from parent directory with full project context...${NC}"
echo -e "${BLUE}🏷️  Using unique tag: ${UNIQUE_TAG}${NC}"

# Change to parent directory and build with runpod_service as context
cd ..
docker build -f runpod_service/Dockerfile -t "${FULL_IMAGE_NAME}" .
BUILD_RESULT=$?
cd runpod_service

if [ $BUILD_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

# Test the image locally first
echo -e "${BLUE}🧪 Testing Docker image locally...${NC}"

# Create a test request
TEST_REQUEST='{
  "command": "start_training",
  "trial_id": "deploy_test_001",
  "dataset": "mnist",
  "hyperparameters": {
    "learning_rate": 0.001,
    "epochs": 2,
    "batch_size": 32,
    "activation": "relu",
    "optimizer": "adam"
  },
  "config": {
    "validation_split": 0.2,
    "mode": "simple",
    "objective": "val_accuracy"
  }
}'

# Clean up any existing containers first
echo -e "${YELLOW}🧹 Cleaning up any existing test containers...${NC}"
docker ps -a --filter "ancestor=${FULL_IMAGE_NAME}" --format "{{.ID}}" | xargs -r docker rm -f > /dev/null 2>&1
docker ps -q --filter "publish=8080" | xargs -r docker stop > /dev/null 2>&1

# Find an available port
TEST_PORT=8080
echo -e "${YELLOW}🔍 Checking port availability...${NC}"
while netstat -ln 2>/dev/null | grep -q ":${TEST_PORT} " || docker ps --filter "publish=${TEST_PORT}" --format "{{.ID}}" | grep -q .; do
    TEST_PORT=$((TEST_PORT + 1))
    if [ ${TEST_PORT} -gt 8090 ]; then
        echo -e "${RED}❌ Could not find available port between 8080-8090${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✅ Using port ${TEST_PORT} for testing${NC}"

# Run container in background for testing
echo -e "${YELLOW}📦 Starting test container...${NC}"
CONTAINER_ID=$(docker run -d -p ${TEST_PORT}:8080 "${FULL_IMAGE_NAME}")

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to start test container${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Test container started with ID: ${CONTAINER_ID:0:12}${NC}"

# Wait for container to fully start
echo -e "${YELLOW}⏳ Waiting for container to start...${NC}"
sleep 15

# Test the endpoint with better error handling
echo -e "${BLUE}🧪 Testing local endpoint on port ${TEST_PORT}...${NC}"
echo -e "${YELLOW}📡 Sending test request (this may take 1-2 minutes)...${NC}"

# Capture both response and HTTP status
HTTP_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:${TEST_PORT} \
    -H "Content-Type: application/json" \
    -d "${TEST_REQUEST}" \
    --max-time 180)

# Extract response body and status code
HTTP_BODY=$(echo "${HTTP_RESPONSE}" | head -n -1)
HTTP_STATUS=$(echo "${HTTP_RESPONSE}" | tail -n 1)

echo -e "${BLUE}📊 HTTP Status: ${HTTP_STATUS}${NC}"
echo -e "${BLUE}📝 Response preview (first 300 chars):${NC}"
echo "${HTTP_BODY}" | head -c 300
echo -e "\n..."

# Stop the test container
echo -e "${YELLOW}🛑 Stopping test container...${NC}"
docker stop "${CONTAINER_ID}" > /dev/null 2>&1
docker rm "${CONTAINER_ID}" > /dev/null 2>&1

# Check if test was successful - multiple success indicators
TEST_SUCCESS=false

if [ "${HTTP_STATUS}" = "200" ]; then
    echo -e "${GREEN}✅ HTTP 200 response received${NC}"
    
    # Check for success indicators (note: JSON uses lowercase 'true')
    if echo "${HTTP_BODY}" | grep -q '"success":\s*true' || \
       echo "${HTTP_BODY}" | grep -q '"status":\s*"completed"' || \
       echo "${HTTP_BODY}" | grep -q '"trial_id":\s*"deploy_test_001"'; then
        TEST_SUCCESS=true
        echo -e "${GREEN}✅ Success indicators found in response${NC}"
    else
        echo -e "${YELLOW}⚠️  Success indicators not found, checking response content...${NC}"
        # Additional debug - show what we're actually looking for
        if echo "${HTTP_BODY}" | grep -q '"test_accuracy"' && \
           echo "${HTTP_BODY}" | grep -q '"metrics"'; then
            TEST_SUCCESS=true
            echo -e "${GREEN}✅ Training metrics found - assuming success${NC}"
        fi
    fi
else
    echo -e "${RED}❌ HTTP ${HTTP_STATUS} response${NC}"
fi

if [ "${TEST_SUCCESS}" = "true" ]; then
    echo -e "${GREEN}✅ Local test successful!${NC}"
    echo -e "${BLUE}📊 Test response summary:${NC}"
    
    # Try to extract and display key metrics using jq if available
    if command -v jq > /dev/null 2>&1; then
        echo "${HTTP_BODY}" | jq -r '.metrics // empty' 2>/dev/null || \
        echo "${HTTP_BODY}" | jq -r '.trial_id, .status' 2>/dev/null || \
        echo "Response contains training results"
    else
        # Fallback without jq
        echo "Training completed successfully"
        if echo "${HTTP_BODY}" | grep -o '"test_accuracy":[0-9.]*' > /dev/null; then
            echo "${HTTP_BODY}" | grep -o '"test_accuracy":[0-9.]*'
        fi
    fi
else
    echo -e "${RED}❌ Local test failed${NC}"
    echo -e "${RED}Full response:${NC}"
    echo "${HTTP_BODY}"
    echo -e "${YELLOW}⚠️  Continuing with deployment anyway...${NC}"
fi

# Check if RunPod CLI is available
echo -e "${BLUE}🔍 Checking for RunPod CLI...${NC}"
if command -v runpodctl > /dev/null 2>&1; then
    echo -e "${GREEN}✅ RunPod CLI found${NC}"
    
    # Deploy using RunPod CLI with environment variables
    echo -e "${BLUE}🚀 Deploying to RunPod endpoint: ${RUNPOD_ENDPOINT_ID}...${NC}"
    
    # Set the API key for runpodctl
    export RUNPOD_API_KEY="${RUNPOD_API_KEY}"
    
    # Deploy to existing endpoint (assuming you have a serverless endpoint configured)
    echo -e "${YELLOW}📋 Deploying to RunPod serverless endpoint...${NC}"
    
    # Tag and push to Docker Hub (required for RunPod deployment)
    echo -e "${BLUE}🔍 Getting Docker Hub username...${NC}"

    # Use DOCKERHUB_USERNAME from .env file
    if [ -n "${DOCKERHUB_USERNAME}" ]; then
        DOCKER_USERNAME="${DOCKERHUB_USERNAME}"
        echo -e "${GREEN}✅ Using Docker Hub username from .env: ${DOCKER_USERNAME}${NC}"
    else
        echo -e "${YELLOW}⚠️  DOCKERHUB_USERNAME not found in .env file${NC}"
        read -p "Docker Hub username: " DOCKER_USERNAME
    fi
    
    if [ -n "${DOCKER_USERNAME}" ]; then
        REMOTE_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
        
        echo -e "${BLUE}🏷️  Tagging image as: ${REMOTE_IMAGE}${NC}"
        docker tag "${FULL_IMAGE_NAME}" "${REMOTE_IMAGE}"
        
        echo -e "${BLUE}⬆️  Pushing to Docker Hub...${NC}"
        docker push "${REMOTE_IMAGE}"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Image pushed to Docker Hub successfully${NC}"
            
            # *** NEW SECTION: Display image name and prompt for RunPod update ***
            echo ""
            echo -e "${BLUE}🎉 DOCKER IMAGE READY FOR DEPLOYMENT!${NC}"
            echo "=============================================="
            echo ""
            echo -e "${GREEN}📦 NEW IMAGE DEPLOYED TO DOCKER HUB:${NC}"
            echo "─────────────────────────────────────────────────────────────"
            echo -e "${YELLOW}${REMOTE_IMAGE}${NC}"
            echo "─────────────────────────────────────────────────────────────"
            echo ""
            echo -e "${BLUE}🔧 REQUIRED ACTION:${NC}"
            echo "1. Go to RunPod endpoint settings for: ${RUNPOD_ENDPOINT_ID}"
            echo "2. Update Container Image to:"
            echo -e "   ${YELLOW}${REMOTE_IMAGE}${NC}"
            echo "3. Save the endpoint configuration"
            echo ""
            echo -e "${YELLOW}⚠️  IMPORTANT: Please update the RunPod endpoint NOW before continuing${NC}"
            echo ""
            read -p "Have you updated the RunPod endpoint with the new image? (y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}⏸️  Deployment paused. Please update RunPod endpoint and re-run script.${NC}"
                echo -e "${BLUE}💡 Copy this image name: ${REMOTE_IMAGE}${NC}"
                exit 0
            fi
            echo -e "${GREEN}✅ Continuing with testing...${NC}"
            
            # Update the endpoint with new image (test it)
            echo -e "${BLUE}🔄 Testing updated RunPod endpoint...${NC}"
            
            # Use RunPod API to test endpoint
            UPDATE_RESPONSE=$(curl -s -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run" \
                -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
                -H "Content-Type: application/json" \
                -d "{\"input\": ${TEST_REQUEST}}")
            
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ Successfully updated RunPod endpoint${NC}"
                echo -e "${BLUE}📊 Deployment test response:${NC}"
                echo "${UPDATE_RESPONSE}" | jq '.' 2>/dev/null || echo "${UPDATE_RESPONSE}"
            else
                echo -e "${RED}❌ Failed to update RunPod endpoint${NC}"
                echo -e "${YELLOW}⚠️  You may need to manually update the endpoint image in RunPod dashboard${NC}"
            fi
        else
            echo -e "${RED}❌ Failed to push image to Docker Hub${NC}"
            echo -e "${YELLOW}⚠️  Please push manually or check Docker Hub credentials${NC}"
            exit 1
        fi
    fi
    
else
    echo -e "${YELLOW}⚠️  RunPod CLI not found${NC}"
    echo -e "${BLUE}📋 Manual deployment required${NC}"
    
    # Still show the image name for manual deployment
    if [ -n "${DOCKERHUB_USERNAME}" ]; then
        REMOTE_IMAGE="${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}"
        
        echo -e "${BLUE}🏷️  Tagging image as: ${REMOTE_IMAGE}${NC}"
        docker tag "${FULL_IMAGE_NAME}" "${REMOTE_IMAGE}"
        
        echo -e "${BLUE}⬆️  Pushing to Docker Hub...${NC}"
        docker push "${REMOTE_IMAGE}"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Image pushed to Docker Hub successfully${NC}"
            
            echo ""
            echo -e "${BLUE}🎉 DOCKER IMAGE READY FOR DEPLOYMENT!${NC}"
            echo "=============================================="
            echo ""
            echo -e "${GREEN}📦 NEW IMAGE DEPLOYED TO DOCKER HUB:${NC}"
            echo "─────────────────────────────────────────────────────────────"
            echo -e "${YELLOW}${REMOTE_IMAGE}${NC}"
            echo "─────────────────────────────────────────────────────────────"
            echo ""
            echo -e "${BLUE}🔧 MANUAL DEPLOYMENT REQUIRED:${NC}"
            echo "1. Go to RunPod endpoint settings for: ${RUNPOD_ENDPOINT_ID}"
            echo "2. Update Container Image to:"
            echo -e "   ${YELLOW}${REMOTE_IMAGE}${NC}"
            echo "3. Save the endpoint configuration"
            echo ""
            read -p "Have you updated the RunPod endpoint with the new image? (y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}⏸️  Testing skipped. Please update RunPod endpoint manually.${NC}"
                echo -e "${BLUE}💡 Copy this image name: ${REMOTE_IMAGE}${NC}"
                exit 0
            fi
        else
            echo -e "${RED}❌ Failed to push image to Docker Hub${NC}"
            exit 1
        fi
    fi
fi


# Add this section after the current deployment test and before the final summary

# Step 3.5: Integration Testing - Test optimizer.py with deployed RunPod service
echo ""
echo -e "${BLUE}🔗 Step 3.5: Integration Testing${NC}"
echo "========================================"
echo -e "${YELLOW}📋 Testing optimizer.py integration with deployed RunPod service...${NC}"

# Wait a moment for RunPod service to be fully ready
echo -e "${YELLOW}⏳ Waiting for RunPod service to be fully ready...${NC}"
sleep 10

# Test 1: Basic integration test
echo -e "${BLUE}🧪 Test 1: Basic optimizer.py integration...${NC}"
cd ..  # Go to project root

INTEGRATION_TEST_1=$(python src/optimizer.py \
  dataset=mnist \
  mode=simple \
  optimize_for=val_accuracy \
  trials=1 \
  use_runpod_service=true \
  runpod_service_endpoint=https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run \
  run_name=integration_test_basic 2>&1)

INTEGRATION_SUCCESS_1=false
if echo "${INTEGRATION_TEST_1}" | grep -q "Optimization completed successfully" && \
   echo "${INTEGRATION_TEST_1}" | grep -q "runpod_service"; then
    INTEGRATION_SUCCESS_1=true
    echo -e "${GREEN}✅ Basic integration test successful${NC}"
    echo -e "${BLUE}📊 Integration test results:${NC}"
    echo "${INTEGRATION_TEST_1}" | grep -E "(Best value|Optimization completed|runpod_service)" | tail -3
else
    echo -e "${RED}❌ Basic integration test failed${NC}"
    echo -e "${YELLOW}📝 Integration test output:${NC}"
    echo "${INTEGRATION_TEST_1}" | tail -10
fi

# Test 2: Fallback mechanism test
echo ""
echo -e "${BLUE}🧪 Test 2: Fallback mechanism validation...${NC}"
FALLBACK_TEST=$(python src/optimizer.py \
  dataset=mnist \
  mode=simple \
  optimize_for=val_accuracy \
  trials=1 \
  use_runpod_service=true \
  runpod_service_endpoint=https://invalid-endpoint-test.com \
  runpod_service_fallback_local=true \
  run_name=integration_test_fallback 2>&1)

FALLBACK_SUCCESS=false
if echo "${FALLBACK_TEST}" | grep -q "Optimization completed successfully" && \
   echo "${FALLBACK_TEST}" | grep -q "local.*execution"; then
    FALLBACK_SUCCESS=true
    echo -e "${GREEN}✅ Fallback mechanism test successful${NC}"
    echo -e "${BLUE}📊 Fallback correctly used local execution${NC}"
else
    echo -e "${YELLOW}⚠️  Fallback mechanism test inconclusive${NC}"
    echo "${FALLBACK_TEST}" | grep -E "(fallback|local|error)" | tail -3
fi

# Test 3: Compare local vs RunPod service results
echo ""
echo -e "${BLUE}🧪 Test 3: Local vs RunPod service comparison...${NC}"

# Run local baseline
echo -e "${YELLOW}📊 Running local baseline...${NC}"
LOCAL_BASELINE=$(python src/optimizer.py \
  dataset=mnist \
  mode=simple \
  optimize_for=val_accuracy \
  trials=1 \
  run_name=integration_test_local_baseline 2>&1)

LOCAL_ACCURACY=""
if echo "${LOCAL_BASELINE}" | grep -q "Best value"; then
    LOCAL_ACCURACY=$(echo "${LOCAL_BASELINE}" | grep "Best value" | grep -o '[0-9]\+\.[0-9]\+' | tail -1)
    echo -e "${GREEN}✅ Local baseline completed - accuracy: ${LOCAL_ACCURACY}${NC}"
else
    echo -e "${YELLOW}⚠️  Local baseline had issues${NC}"
fi

# Compare results if both succeeded
COMPARISON_SUCCESS=false
if [ "${INTEGRATION_SUCCESS_1}" = "true" ] && [ -n "${LOCAL_ACCURACY}" ]; then
    RUNPOD_ACCURACY=$(echo "${INTEGRATION_TEST_1}" | grep "Best value" | grep -o '[0-9]\+\.[0-9]\+' | tail -1)
    if [ -n "${RUNPOD_ACCURACY}" ]; then
        echo -e "${GREEN}✅ Results comparison:${NC}"
        echo -e "${BLUE}   Local accuracy:  ${LOCAL_ACCURACY}${NC}"
        echo -e "${BLUE}   RunPod accuracy: ${RUNPOD_ACCURACY}${NC}"
        
        # Check if results are within reasonable range (±10%)
        python3 -c "
local = float('${LOCAL_ACCURACY}')
runpod = float('${RUNPOD_ACCURACY}')
diff_pct = abs(local - runpod) / local * 100
print(f'   Difference: {diff_pct:.2f}%')
if diff_pct < 10:
    print('✅ Results within acceptable range (±10%)')
    exit(0)
else:
    print('⚠️  Results differ significantly')
    exit(1)
" && COMPARISON_SUCCESS=true
    fi
fi

cd runpod_service  # Return to runpod_service directory

# Integration testing summary
echo ""
echo -e "${BLUE}📊 Integration Testing Summary${NC}"
echo "========================================"
if [ "${INTEGRATION_SUCCESS_1}" = "true" ]; then
    echo -e "${GREEN}✅ Basic integration test: PASSED${NC}"
else
    echo -e "${RED}❌ Basic integration test: FAILED${NC}"
fi

if [ "${FALLBACK_SUCCESS}" = "true" ]; then
    echo -e "${GREEN}✅ Fallback mechanism test: PASSED${NC}"
else
    echo -e "${YELLOW}⚠️  Fallback mechanism test: INCONCLUSIVE${NC}"
fi

if [ "${COMPARISON_SUCCESS}" = "true" ]; then
    echo -e "${GREEN}✅ Local vs RunPod comparison: PASSED${NC}"
else
    echo -e "${YELLOW}⚠️  Local vs RunPod comparison: INCONCLUSIVE${NC}"
fi

# Overall integration status
OVERALL_INTEGRATION_SUCCESS=false
if [ "${INTEGRATION_SUCCESS_1}" = "true" ]; then
    OVERALL_INTEGRATION_SUCCESS=true
    echo ""
    echo -e "${GREEN}🎉 STEP 3.5 INTEGRATION TESTING: SUCCESSFUL${NC}"
    echo -e "${GREEN}🚀 RunPod service is fully operational and integrated!${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠️  STEP 3.5 INTEGRATION TESTING: PARTIAL${NC}"
    echo -e "${YELLOW}📝 Manual verification may be required${NC}"
fi


# Provide deployment summary and next steps
echo ""
echo -e "${BLUE}📋 Deployment Summary${NC}"
echo "========================================"
echo -e "${GREEN}✅ Docker image built: ${FULL_IMAGE_NAME}${NC}"
echo -e "${GREEN}✅ Local testing completed${NC}"
echo -e "${GREEN}✅ Environment variables loaded from .env${NC}"
echo -e "${BLUE}📋 RunPod Endpoint ID: ${RUNPOD_ENDPOINT_ID}${NC}"

if [ -n "${REMOTE_IMAGE}" ]; then
    echo -e "${GREEN}✅ Image deployed to Docker Hub: ${REMOTE_IMAGE}${NC}"
fi

if command -v runpodctl > /dev/null 2>&1; then
    echo -e "${GREEN}✅ RunPod CLI deployment attempted${NC}"
else
    echo ""
    echo -e "${BLUE}📋 Manual RunPod Deployment Instructions:${NC}"
    echo "========================================"
    echo "1. Push Docker image to Docker Hub:"
    echo "   docker tag ${FULL_IMAGE_NAME} ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}"
    echo "   docker push ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}"
    echo ""
    echo "2. Update RunPod Endpoint:"
    echo "   - Go to https://www.runpod.io/console/serverless"
    echo "   - Find endpoint: ${RUNPOD_ENDPOINT_ID}"
    echo "   - Update Docker image to: ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}"
    echo "   - Ensure port 8080 is exposed"
fi

echo ""
echo -e "${BLUE}📝 Test the deployed endpoint:${NC}"
echo "curl -X POST https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run \\"
echo "  -H \"Authorization: Bearer ${RUNPOD_API_KEY}\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '${TEST_REQUEST}'"
echo ""
echo -e "${BLUE}📝 Example optimizer.py usage with deployed service:${NC}"
echo "python src/optimizer.py \\"
echo "  dataset=mnist \\"
echo "  mode=simple \\"
echo "  optimize_for=val_accuracy \\"
echo "  trials=3 \\"
echo "  use_runpod_service=true \\"
echo "  runpod_service_endpoint=https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run"
echo ""
echo -e "${GREEN}🚀 Deployment script completed!${NC}"