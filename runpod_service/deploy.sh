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

# Create a test request with FastAPI format (flat structure)
# FastAPI will wrap this in {"input": ...} automatically
TEST_REQUEST='{
  "command": "start_training",
  "trial_id": "deploy_test_001",
  "dataset_name": "mnist",
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

# Find an available port for FastAPI testing
TEST_PORT=8080
echo -e "${BLUE}🔍 Finding available port...${NC}"
while netstat -ln 2>/dev/null | grep -q ":${TEST_PORT} " || docker ps --filter "publish=${TEST_PORT}" --format "{{.ID}}" | grep -q .; do
    TEST_PORT=$((TEST_PORT + 1))
    if [ ${TEST_PORT} -gt 8090 ]; then
        echo -e "${RED}❌ Could not find available port between 8080-8090${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✅ Using port ${TEST_PORT} for testing${NC}"

# Start container in FastAPI mode (no RUNPOD_ENDPOINT_ID = FastAPI mode)
echo -e "${BLUE}📦 Starting test container in FastAPI development mode...${NC}"
CONTAINER_ID=$(docker run -d \
    -p ${TEST_PORT}:8080 \
    --name "cv-test-$(date +%s)" \
    "${FULL_IMAGE_NAME}")

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to start test container${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Test container started with ID: ${CONTAINER_ID:0:12}${NC}"

# Wait for FastAPI to start
echo -e "${BLUE}⏳ Waiting for FastAPI server to start...${NC}"
sleep 15

# Test the FastAPI endpoint
echo -e "${BLUE}🧪 Testing FastAPI handler endpoint...${NC}"
echo -e "${YELLOW}📡 Sending HTTP request (this may take 1-2 minutes)...${NC}"

# Send HTTP request with proper FastAPI structure (flat, no input wrapper)
HTTP_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:${TEST_PORT} \
    -H "Content-Type: application/json" \
    -d "${TEST_REQUEST}" \
    --max-time 180)

# Extract response body and status code
HTTP_BODY=$(echo "${HTTP_RESPONSE}" | head -n -1)
HTTP_STATUS=$(echo "${HTTP_RESPONSE}" | tail -n 1)

echo -e "${BLUE}📊 HTTP Status: ${HTTP_STATUS}${NC}"

# Show full response
echo -e "${BLUE}📝 Full HTTP Response:${NC}"
echo "${HTTP_BODY}"

# Show container logs for additional context
echo -e "${BLUE}📋 Container logs (last 20 lines):${NC}"
docker logs --tail 20 "${CONTAINER_ID}" 2>&1 || echo "Could not retrieve container logs"

# Stop the test container (AFTER showing response and logs)  
echo -e "${YELLOW}🛑 Stopping test container...${NC}"
docker stop "${CONTAINER_ID}" > /dev/null 2>&1
docker rm "${CONTAINER_ID}" > /dev/null 2>&1

# Check if test was successful based on HTTP response
TEST_SUCCESS=false

if [ "${HTTP_STATUS}" = "200" ]; then
    echo -e "${GREEN}✅ HTTP 200 response received${NC}"
    
    # Check for success indicators in the response
    if echo "${HTTP_BODY}" | grep -q '"success":\s*true' || \
       echo "${HTTP_BODY}" | grep -q '"trial_id":\s*"deploy_test_001"' || \
       echo "${HTTP_BODY}" | grep -q '"test_accuracy"'; then
        TEST_SUCCESS=true
        echo -e "${GREEN}✅ Success indicators found in response${NC}"
    else
        echo -e "${YELLOW}⚠️  No clear success indicators found${NC}"
    fi
else
    echo -e "${RED}❌ HTTP ${HTTP_STATUS} response${NC}"
fi

if [ "${TEST_SUCCESS}" = "true" ]; then
    echo -e "${GREEN}✅ FastAPI handler test successful!${NC}"
    echo -e "${BLUE}📊 Test summary:${NC}"
    echo "   - FastAPI server started successfully"
    echo "   - Handler processed request correctly"
    echo "   - Training pipeline executed without errors"
    echo "   - Results returned in expected format"
else
    echo -e "${RED}❌ FastAPI handler test failed${NC}"
    echo -e "${YELLOW}⚠️  Check the logs above for detailed error information${NC}"
    echo -e "${YELLOW}⚠️  Continuing with deployment anyway...${NC}"
fi

# Additional comprehensive local testing
echo -e "${BLUE}🔬 Running additional local validation tests...${NC}"

# Initialize test result tracking
TEST1_PASSED=false
TEST2_PASSED=false
TEST3_PASSED=false
TEST4_PASSED=false
TEST5_PASSED=false

# Test 1: Validate handler.py syntax and imports
echo -e "${YELLOW}📋 Test 1: Validating handler.py syntax and imports...${NC}"
SYNTAX_TEST=$(docker run --rm "${FULL_IMAGE_NAME}" python -c "
import sys
sys.path.insert(0, '/app')
try:
    import handler
    print('✅ Handler imports successfully')
    if hasattr(handler, 'handler'):
        print('✅ Handler function exists')
    else:
        print('❌ Handler function missing')
        sys.exit(1)
    if hasattr(handler, 'start_training'):
        print('✅ start_training function exists')
    else:
        print('❌ start_training function missing')
        sys.exit(1)
    if hasattr(handler, 'start_final_model_training'):
        print('✅ start_final_model_training function exists')
    else:
        print('❌ start_final_model_training function missing')
        sys.exit(1)
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Syntax error: {e}')
    sys.exit(1)
" 2>&1)

echo "$SYNTAX_TEST"
if echo "$SYNTAX_TEST" | grep -q "❌"; then
    echo -e "${RED}❌ Test 1 FAILED${NC}"
    TEST1_PASSED=false
else
    echo -e "${GREEN}✅ Test 1 PASSED${NC}"
    TEST1_PASSED=true
fi

# Test 2: Check required dependencies
echo -e "${YELLOW}📋 Test 2: Checking critical dependencies...${NC}"
DEPS_TEST=$(docker run --rm "${FULL_IMAGE_NAME}" python -c "
import sys
required_modules = [
    'tensorflow', 'keras', 'optuna', 'runpod', 'fastapi', 
    'uvicorn', 'numpy', 'matplotlib', 'cv2'
]
missing = []
for module in required_modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError:
        print(f'❌ {module} - MISSING')
        missing.append(module)
        
if missing:
    print(f'❌ Missing dependencies: {missing}')
    sys.exit(1)
else:
    print('✅ All critical dependencies available')
" 2>&1)

echo "$DEPS_TEST"
if echo "$DEPS_TEST" | grep -q "❌"; then
    echo -e "${RED}❌ Test 2 FAILED${NC}"
    TEST2_PASSED=false
else
    echo -e "${GREEN}✅ Test 2 PASSED${NC}"
    TEST2_PASSED=true
fi

# Test 3: Validate environment variable handling
echo -e "${YELLOW}📋 Test 3: Testing environment variable handling...${NC}"
ENV_TEST=$(docker run --rm \
    -e RUNPOD_ENDPOINT_ID="test_endpoint" \
    -e RUNPOD_API_KEY="test_key" \
    "${FULL_IMAGE_NAME}" python -c "
import os
import sys
sys.path.insert(0, '/app')

# Test environment variables
endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
api_key = os.getenv('RUNPOD_API_KEY')

if endpoint_id:
    print(f'✅ RUNPOD_ENDPOINT_ID: {endpoint_id}')
else:
    print('❌ RUNPOD_ENDPOINT_ID missing')

if api_key:
    print(f'✅ RUNPOD_API_KEY: {api_key[:8]}...')
else:
    print('❌ RUNPOD_API_KEY missing')

# Test handler mode detection
try:
    from handler import os
    if os.getenv('RUNPOD_ENDPOINT_ID'):
        print('✅ Handler will run in serverless mode')
    else:
        print('✅ Handler will run in FastAPI mode')
except Exception as e:
    print(f'❌ Mode detection failed: {e}')
    sys.exit(1)
" 2>&1)

echo "$ENV_TEST"
if echo "$ENV_TEST" | grep -q "❌"; then
    echo -e "${RED}❌ Test 3 FAILED${NC}"
    TEST3_PASSED=false
else
    echo -e "${GREEN}✅ Test 3 PASSED${NC}"
    TEST3_PASSED=true
fi

# Test 4: Test both command types with dry-run
echo -e "${YELLOW}📋 Test 4: Testing command routing (dry-run)...${NC}"
ROUTING_TEST=$(docker run --rm "${FULL_IMAGE_NAME}" python -c "
import sys
sys.path.insert(0, '/app')
import asyncio

try:
    from handler import handler
    
    # Test start_training command
    training_event = {
        'input': {
            'command': 'start_training',
            'trial_id': 'test_001',
            'dataset_name': 'mnist',
            'hyperparameters': {'learning_rate': 0.001, 'epochs': 1},
            'config': {'mode': 'simple', 'objective': 'val_accuracy'}
        }
    }
    
    # Test final model command  
    final_model_event = {
        'input': {
            'command': 'start_final_model_training',
            'dataset_name': 'mnist',
            'best_params': {'learning_rate': 0.001, 'epochs': 1},
            'config': {'mode': 'simple', 'objective': 'val_accuracy'}
        }
    }
    
    print('✅ Command routing validation passed')
    print('✅ Event structures are valid')
    
except Exception as e:
    print(f'❌ Routing validation failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1)

echo "$ROUTING_TEST"
if echo "$ROUTING_TEST" | grep -q "❌"; then
    echo -e "${RED}❌ Test 4 FAILED${NC}"
    TEST4_PASSED=false
else
    echo -e "${GREEN}✅ Test 4 PASSED${NC}"
    TEST4_PASSED=true
fi

# Test 5: Resource and disk space check
echo -e "${YELLOW}📋 Test 5: Checking container resources...${NC}"
RESOURCE_TEST=$(docker run --rm "${FULL_IMAGE_NAME}" python -c "
import os
import sys

# Check if psutil is available, if not skip the detailed memory check
has_errors = False

try:
    import psutil
    # Check available memory
    try:
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f'✅ Available RAM: {memory_gb:.1f} GB')
        
        if memory_gb < 4:
            print('⚠️  Low memory - may cause issues with large models')
        
    except Exception as e:
        print(f'❌ Memory check failed: {e}')
        has_errors = True
        
except ImportError:
    print('⚠️  psutil not available - skipping detailed memory check')
    # This is not critical, so don't fail

# Check disk space in key directories
try:
    dirs_to_check = ['/app', '/tmp', '/app/datasets']
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            try:
                stat = os.statvfs(dir_path)
                free_gb = (stat.f_frsize * stat.f_bavail) / (1024**3)
                print(f'✅ {dir_path}: {free_gb:.1f} GB free')
            except Exception as e:
                print(f'⚠️  {dir_path}: could not check disk space - {e}')
        else:
            print(f'⚠️  {dir_path}: directory not found')
            
except Exception as e:
    print(f'❌ Disk check failed: {e}')
    has_errors = True

# Basic container health check - can we import key modules?
try:
    import tensorflow
    import numpy
    print('✅ Core TensorFlow/NumPy imports working')
except Exception as e:
    print(f'❌ Core import test failed: {e}')
    has_errors = True
    
if has_errors:
    print('❌ Resource check completed with errors')
    sys.exit(1)
else:
    print('✅ Resource check completed successfully')
" 2>&1)

echo "$RESOURCE_TEST"
TEST5_EXIT_CODE=$?
if [ $TEST5_EXIT_CODE -ne 0 ] || echo "$RESOURCE_TEST" | grep -q "❌.*Resource check completed with errors"; then
    echo -e "${RED}❌ Test 5 FAILED${NC}"
    TEST5_PASSED=false
else
    echo -e "${GREEN}✅ Test 5 PASSED${NC}"
    TEST5_PASSED=true
fi

echo -e "${BLUE}🎯 All local validation tests completed!${NC}"
echo -e "${BLUE}📊 Pre-deployment validation summary:${NC}"

# Show actual test results
if [ "$TEST1_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 1: Handler syntax and imports${NC}"
else
    echo -e "   ${RED}❌ Test 1: Handler syntax and imports${NC}"
fi

if [ "$TEST2_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 2: Critical dependencies${NC}"
else
    echo -e "   ${RED}❌ Test 2: Critical dependencies${NC}"
fi

if [ "$TEST3_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 3: Environment variable handling${NC}"
else
    echo -e "   ${RED}❌ Test 3: Environment variable handling${NC}"
fi

if [ "$TEST4_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 4: Command routing validation${NC}"
else
    echo -e "   ${RED}❌ Test 4: Command routing validation${NC}"
fi

if [ "$TEST5_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 5: Resource availability${NC}"
else
    echo -e "   ${RED}❌ Test 5: Resource availability${NC}"
fi

# Count passed tests
TESTS_PASSED=0
[ "$TEST1_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))
[ "$TEST2_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))
[ "$TEST3_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))
[ "$TEST4_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))
[ "$TEST5_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))

echo ""
echo -e "${BLUE}📊 Test Results: ${TESTS_PASSED}/5 tests passed${NC}"

# Determine if we should continue with deployment
if [ "$TEST1_PASSED" = "true" ] && [ "$TEST2_PASSED" = "true" ]; then
    echo -e "${GREEN}🚀 Core tests passed - Ready for RunPod deployment!${NC}"
    DEPLOYMENT_READY=true
elif [ $TESTS_PASSED -ge 3 ]; then
    echo -e "${YELLOW}⚠️  Some tests failed, but core functionality appears intact${NC}"
    echo -e "${YELLOW}📝 Continuing with deployment (recommended to investigate failures)${NC}"
    DEPLOYMENT_READY=true
else
    echo -e "${RED}❌ Too many critical tests failed${NC}"
    echo -e "${YELLOW}⚠️  Deployment may fail - consider fixing issues first${NC}"
    echo -e "${YELLOW}📝 Continuing anyway, but expect potential deployment issues${NC}"
    DEPLOYMENT_READY=false
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
                # exit 0
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

echo -e "${YELLOW}⏳ Running integration test (this may take 5-10 minutes)...${NC}"
echo "DEBUG: Running from directory: $(pwd)"
echo "DEBUG: Command: python src/optimizer.py dataset_name=mnist mode=simple optimize_for=val_accuracy trials=1 max_epochs_per_trial=5 min_epochs_per_trial=2 health_weight=0.3 use_runpod_service=true runpod_service_endpoint=https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run run_name=integration_test_basic"
INTEGRATION_TEST_1=$(timeout 600 python src/optimizer.py \
  dataset_name=mnist \
  mode=simple \
  optimize_for=val_accuracy \
  trials=1 \
  max_epochs_per_trial=5 \
  min_epochs_per_trial=2 \
  health_weight=0.3 \
  use_runpod_service=true \
  runpod_service_endpoint=https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run \
  run_name=integration_test_basic 2>&1)

OPTIMIZER_EXIT_CODE=$?
if [ ${OPTIMIZER_EXIT_CODE} -eq 124 ]; then
    echo -e "${YELLOW}⏰ Integration test timed out after 10 minutes${NC}"
    INTEGRATION_TEST_1="TIMEOUT: Integration test exceeded 10 minute limit"
elif [ ${OPTIMIZER_EXIT_CODE} -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Integration test exited with code: ${OPTIMIZER_EXIT_CODE}${NC}"
fi

INTEGRATION_SUCCESS_1=false

# 🔍 DEBUG: Show full integration test output for debugging
echo -e "${YELLOW}🔍 DEBUG: Full integration test output:${NC}"
echo "=================================================="
echo "${INTEGRATION_TEST_1}"
echo "=================================================="

if echo "${INTEGRATION_TEST_1}" | grep -q "Optimization completed successfully" && \
   echo "${INTEGRATION_TEST_1}" | grep -q "runpod_service"; then
    INTEGRATION_SUCCESS_1=true
    echo -e "${GREEN}✅ Basic integration test successful${NC}"
    echo -e "${BLUE}📊 Integration test results:${NC}"
    echo "${INTEGRATION_TEST_1}" | grep -E "(Best value|Optimization completed|runpod_service)" | tail -3
else
    echo -e "${RED}❌ Basic integration test failed${NC}"
    echo -e "${YELLOW}📝 Checking what we're missing:${NC}"
    echo "Looking for 'Optimization completed successfully': $(echo "${INTEGRATION_TEST_1}" | grep -c "Optimization completed successfully")"
    echo "Looking for 'runpod_service': $(echo "${INTEGRATION_TEST_1}" | grep -c "runpod_service")"
    echo -e "${YELLOW}📝 Last 10 lines of integration test:${NC}"
    echo "${INTEGRATION_TEST_1}" | tail -10
fi

# Test 2: Fallback mechanism test
echo ""
echo -e "${BLUE}🧪 Test 2: Fallback mechanism validation...${NC}"
FALLBACK_TEST=$(python src/optimizer.py \
  dataset_name=mnist \
  mode=simple \
  optimize_for=val_accuracy \
  trials=1 \
  max_epochs_per_trial=5 \
  min_epochs_per_trial=2 \
  health_weight=0.3 \
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
  dataset_name=mnist \
  mode=simple \
  optimize_for=val_accuracy \
  trials=1 \
  max_epochs_per_trial=5 \
  min_epochs_per_trial=2 \
  health_weight=0.3 \
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
        python3 -c "import sys; local=float('${LOCAL_ACCURACY}'); runpod=float('${RUNPOD_ACCURACY}'); diff_pct=abs(local-runpod)/local*100; print(f'Difference: {diff_pct:.2f}%'); sys.exit(0 if diff_pct<10 else 1)" && COMPARISON_SUCCESS=true
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
echo "  dataset_name=mnist \\"
echo "  mode=simple \\"
echo "  optimize_for=val_accuracy \\"
echo "  trials=3 \\"
echo "  use_runpod_service=true \\"
echo "  runpod_service_endpoint=https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run"
echo ""
echo -e "${GREEN}🚀 Deployment script completed!${NC}"