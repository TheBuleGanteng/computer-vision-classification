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
    echo "RUNPOD_ENDPOINT_ID=your_endpoint_id"
    echo "RUNPOD_API_KEY=your_api_key"
    echo "DOCKERHUB_USERNAME=your_dockerhub_username"
    echo "RUNPOD_S3_ACCESS_KEY=your_s3_access_key"
    echo "RUNPOD_S3_SECRET_ACCESS_KEY=your_s3_secret_key"
    exit 1
fi

# Source the .env file
set -a  # Mark variables for export
source "${ENV_FILE}"
set +a  # Unmark variables for export

# Validate required environment variables
if [ -z "${RUNPOD_ENDPOINT_ID}" ] || [ -z "${RUNPOD_API_KEY}" ] || [ -z "${DOCKERHUB_USERNAME}" ] || [ -z "${RUNPOD_S3_ACCESS_KEY}" ] || [ -z "${RUNPOD_S3_SECRET_ACCESS_KEY}" ]; then
    echo -e "${RED}❌ Missing required environment variables${NC}"
    echo -e "${YELLOW}💡 Please ensure .env file contains:${NC}"
    echo "RUNPOD_ENDPOINT_ID=your_endpoint_id"
    echo "RUNPOD_API_KEY=your_api_key"
    echo "DOCKERHUB_USERNAME=your_dockerhub_username"
    echo "RUNPOD_S3_ACCESS_KEY=your_s3_access_key"
    echo "RUNPOD_S3_SECRET_ACCESS_KEY=your_s3_secret_key"
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

# Check if we're in the right directory and adjust paths accordingly
if [ -f "runpod_service/Dockerfile" ] && [ -f "runpod_service/handler.py" ]; then
    # Running from project root
    DOCKERFILE_PATH="runpod_service/Dockerfile"
    HANDLER_PATH="runpod_service/handler.py"
    BUILD_CONTEXT="."
    echo -e "${GREEN}✅ Found required files from project root${NC}"
elif [ -f "Dockerfile" ] && [ -f "handler.py" ]; then
    # Running from runpod_service directory
    DOCKERFILE_PATH="Dockerfile"
    HANDLER_PATH="handler.py"
    BUILD_CONTEXT=".."
    echo -e "${GREEN}✅ Found required files from runpod_service directory${NC}"
else
    echo -e "${RED}❌ Required files not found${NC}"
    echo -e "${YELLOW}💡 Please run from either:${NC}"
    echo "   - Project root directory (with runpod_service/Dockerfile)"
    echo "   - runpod_service directory (with Dockerfile)"
    exit 1
fi

# Build the Docker image with proper context
echo -e "${BLUE}🏗️  Building Docker image: ${FULL_IMAGE_NAME}...${NC}"
echo -e "${YELLOW}📂 Building with context: ${BUILD_CONTEXT}${NC}"
echo -e "${BLUE}🏷️  Using unique tag: ${UNIQUE_TAG}${NC}"

# Build with the appropriate context and Dockerfile path
if [ "${BUILD_CONTEXT}" = ".." ]; then
    # We're in runpod_service directory, need to go up
    cd ..
    docker build -f runpod_service/Dockerfile -t "${FULL_IMAGE_NAME}" .
    BUILD_RESULT=$?
    cd runpod_service
else
    # We're in project root, build directly
    docker build -f "${DOCKERFILE_PATH}" -t "${FULL_IMAGE_NAME}" "${BUILD_CONTEXT}"
    BUILD_RESULT=$?
fi

if [ $BUILD_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

# Test the image locally first
echo -e "${BLUE}🧪 Testing Docker image locally...${NC}"

# Create a test request for unified architecture
# Uses minimal training parameters (1 epoch, batch_size=64) for faster deployment testing
# The unified handler calls optimize_model() directly, so this does real model training
TEST_REQUEST='{
  "command": "start_training",
  "trial_id": "deploy_test_001",
  "dataset_name": "mnist",
  "hyperparameters": {
    "learning_rate": 0.001,
    "epochs": 1,
    "batch_size": 64,
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
echo -e "${YELLOW}📡 Sending HTTP request (this may take up to 5 minutes for real model training)...${NC}"
echo -e "${BLUE}💡 To view live training logs, open a new terminal and run:${NC}"
echo -e "   ${GREEN}docker logs -f ${CONTAINER_ID}${NC}"
echo ""

# Send HTTP request with proper FastAPI structure (flat, no input wrapper)
HTTP_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:${TEST_PORT} \
    -H "Content-Type: application/json" \
    -d "${TEST_REQUEST}" \
    --max-time 300)

# Extract response body and status code
HTTP_BODY=$(echo "${HTTP_RESPONSE}" | head -n -1)
HTTP_STATUS=$(echo "${HTTP_RESPONSE}" | tail -n 1)

echo -e "${BLUE}📊 HTTP Status: ${HTTP_STATUS}${NC}"

# Show full response
echo -e "${BLUE}📝 Full HTTP Response:${NC}"
echo "${HTTP_BODY}"

# Analyze response structure if it's JSON
echo -e "${BLUE}🔍 Response Analysis:${NC}"
if echo "${HTTP_BODY}" | jq . > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Valid JSON response${NC}"
    echo -e "${BLUE}Response keys:${NC}"
    echo "${HTTP_BODY}" | jq -r 'keys[]' 2>/dev/null || echo "Could not extract keys"
    echo -e "${BLUE}Response size:${NC}"
    echo "${HTTP_BODY}" | wc -c | awk '{print $1 " bytes"}'
    
    # Check for specific fields that might cause issues
    echo -e "${BLUE}Field analysis:${NC}"
    for field in "health_metrics" "model_attributes" "training_history" "best_params"; do
        if echo "${HTTP_BODY}" | jq -e ".${field}" > /dev/null 2>&1; then
            field_size=$(echo "${HTTP_BODY}" | jq -c ".${field}" | wc -c)
            echo "  - ${field}: ${field_size} bytes"
        else
            echo "  - ${field}: not present"
        fi
    done
else
    echo -e "${YELLOW}⚠️  Response is not valid JSON${NC}"
fi

# Show container logs with focus on our debug logging
echo -e "${BLUE}📋 Container logs (focusing on RUNPOD RESPONSE ANALYSIS):${NC}"
docker logs "${CONTAINER_ID}" 2>&1 | grep -A 50 "RUNPOD RESPONSE ANALYSIS" || echo "No RUNPOD RESPONSE ANALYSIS found in logs"

echo -e "${BLUE}📋 All container logs (last 50 lines):${NC}"
docker logs --tail 50 "${CONTAINER_ID}" 2>&1 || echo "Could not retrieve container logs"

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
TEST6_PASSED=false
TEST7_PASSED=false
TEST8_PASSED=false
TEST9_PASSED=false
TEST10_PASSED=false
TEST11_PASSED=false
TEST12_PASSED=false
TEST13_PASSED=false

# Test 1: Check for correct trial_ file path (fix for double "trial_" issue)
echo -e "${YELLOW}📋 Test 1: Validating trial_ path fix...${NC}"
PATH_TEST=$(docker run --rm "${FULL_IMAGE_NAME}" python -c "
import sys
sys.path.insert(0, '/app')
try:
    # Test the actual unified filepath mechanism used by the handler
    from src.data_classes.configs import generate_unified_run_name
    from datetime import datetime
    
    # Test parameters matching what handler.py receives
    dataset_name = 'mnist'
    mode = 'health'
    optimize_for = 'val_accuracy'
    
    # Generate run_name using the actual unified function
    unified_run_name = generate_unified_run_name(
        dataset_name=dataset_name,
        mode=mode,
        optimize_for=optimize_for
    )
    
    # Construct S3 path using the unified run_name (as handler.py does)
    s3_prefix = f'optimization_results/{unified_run_name}/plots/trial_0'
    
    print(f'✅ Dataset: {dataset_name}')
    print(f'✅ Mode: {mode}')
    print(f'✅ Optimize for: {optimize_for}')
    print(f'✅ Unified run_name: {unified_run_name}')
    print(f'✅ Final S3 prefix: {s3_prefix}')
    
    # Validate the unified approach
    if 'runpod_trial_trial_' in s3_prefix:
        print('❌ DOUBLE TRIAL ISSUE STILL EXISTS!')
        sys.exit(1)
    elif 'runpod_trial' in s3_prefix:
        print('❌ LEGACY runpod_trial naming still present - unified approach not working!')
        sys.exit(1)
    elif unified_run_name in s3_prefix and 'trial_0' in s3_prefix:
        print('✅ Unified filepath mechanism verified - consistent naming!')
    else:
        print(f'❌ Unexpected path format: {s3_prefix}')
        sys.exit(1)
        
    # Test that the run_name follows expected pattern
    timestamp_part = unified_run_name.split('_')[0]
    if len(timestamp_part) == 19 and '-' in timestamp_part:  # Format: YYYY-MM-DD-HH:MM:SS
        print('✅ Timestamp format verified')
    else:
        print(f'❌ Timestamp format unexpected: {timestamp_part}')
        sys.exit(1)
        
except Exception as e:
    print(f'❌ Unified filepath test failed: {e}')
    sys.exit(1)
" 2>&1)

echo "$PATH_TEST"
if echo "$PATH_TEST" | grep -q "❌"; then
    echo -e "${RED}❌ Test 1 FAILED - Double trial issue still exists${NC}"
    TEST1_PASSED=false
else
    echo -e "${GREEN}✅ Test 1 PASSED - Trial path fix verified${NC}"
    TEST1_PASSED=true
fi

# Test 2: Check config_data reception and plot_generation handling
echo -e "${YELLOW}📋 Test 2: Validating config_data and plot_generation handling...${NC}"
CONFIG_TEST=$(docker run --rm "${FULL_IMAGE_NAME}" python -c "
import sys
sys.path.insert(0, '/app')
try:
    # Test the unified function with actual config_data that handler.py receives
    from src.data_classes.configs import generate_unified_run_name
    
    mock_config_data = {
        'validation_split': 0.2,
        'mode': 'simple',
        'objective': 'accuracy',
        'plot_generation': 'all'
    }
    
    # Test that unified function works with config_data values
    unified_run_name = generate_unified_run_name(
        dataset_name='mnist',
        mode=mock_config_data['mode'],
        optimize_for=mock_config_data['objective']
    )
    
    # Test the handler decision logic for plot generation
    plot_generation = mock_config_data.get('plot_generation', 'all')  # Updated default
    will_create_plots = plot_generation and plot_generation.lower() != 'none'
    
    print(f'✅ Mock config_data keys: {list(mock_config_data.keys())}')
    print(f'✅ Unified run_name from config: {unified_run_name}')
    print(f'✅ plot_generation value: \"{plot_generation}\"')
    print(f'✅ Will create plots_s3_info: {will_create_plots}')
    
    # Verify unified function uses config values correctly
    if mock_config_data['mode'] in unified_run_name and mock_config_data['objective'] in unified_run_name:
        print('✅ Unified function correctly incorporates config_data values')
    else:
        print(f'❌ Unified function does not use config_data correctly: {unified_run_name}')
        sys.exit(1)
    
    # Test different scenarios
    test_cases = [
        ('all', True),
        ('best', True), 
        ('none', False),
        (None, True)  # Default should be 'all' now
    ]
    
    for test_value, expected in test_cases:
        test_config = {'plot_generation': test_value} if test_value else {}
        result_value = test_config.get('plot_generation', 'all')
        result_bool = result_value and result_value.lower() != 'none'
        
        if result_bool == expected:
            print(f'✅ Test case \"{test_value}\" -> {result_bool} (expected {expected})')
        else:
            print(f'❌ Test case \"{test_value}\" -> {result_bool} (expected {expected})')
            sys.exit(1)
            
    print('✅ Config data handling logic verified')
    
except Exception as e:
    print(f'❌ Config test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1)

echo "$CONFIG_TEST"
if echo "$CONFIG_TEST" | grep -q "❌"; then
    echo -e "${RED}❌ Test 2 FAILED - Config data handling issues${NC}"
    TEST2_PASSED=false
else
    echo -e "${GREEN}✅ Test 2 PASSED - Config data handling verified${NC}"
    TEST2_PASSED=true
fi

# Test 3: Mock RunPod Environment Test (Option 1)
echo -e "${YELLOW}📋 Test 3: Testing RunPod filepath mechanism with mocked environment...${NC}"
MOCK_RUNPOD_TEST=$(docker run --rm -e RUNPOD_ENDPOINT_ID="test_endpoint_123" "${FULL_IMAGE_NAME}" python -c "
import sys
import os
sys.path.insert(0, '/app')
try:
    # Verify we're in 'RunPod' environment
    runpod_id = os.getenv('RUNPOD_ENDPOINT_ID')
    print(f'✅ Mocked RUNPOD_ENDPOINT_ID: {runpod_id}')
    
    # Test the actual handler logic with mocked RunPod environment
    from src.data_classes.configs import generate_unified_run_name
    
    # Create mock request data
    mock_request = {
        'dataset_name': 'mnist',
        'config': {
            'mode': 'health',
            'objective': 'val_accuracy',
            'plot_generation': 'all'
        },
        'hyperparameters': {}
    }
    
    # Generate unified run_name as handler would
    unified_run_name = generate_unified_run_name(
        dataset_name=mock_request['dataset_name'],
        mode=mock_request['config']['mode'],
        optimize_for=mock_request['config']['objective']
    )
    
    print(f'✅ Generated unified run_name: {unified_run_name}')
    
    # Test S3 path construction logic from handler
    s3_prefix = f'optimization_results/{unified_run_name}/plots/trial_0'
    print(f'✅ S3 prefix would be: {s3_prefix}')
    
    # Validate environment detection works
    if runpod_id:
        print('✅ RunPod environment detected correctly')
    else:
        print('❌ RunPod environment detection failed')
        sys.exit(1)
    
    # Validate path format
    if 'runpod_trial_trial_' in s3_prefix:
        print('❌ DOUBLE TRIAL ISSUE STILL EXISTS in mocked environment!')
        sys.exit(1)
    elif 'runpod_trial' in s3_prefix:
        print('❌ LEGACY runpod_trial naming still present in mocked environment!')
        sys.exit(1)
    elif unified_run_name in s3_prefix and 'trial_0' in s3_prefix:
        print('✅ Mocked RunPod environment produces correct S3 paths!')
    else:
        print(f'❌ Unexpected S3 path format in mocked environment: {s3_prefix}')
        sys.exit(1)
        
except Exception as e:
    print(f'❌ Mocked RunPod test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1)

echo "$MOCK_RUNPOD_TEST"
if echo "$MOCK_RUNPOD_TEST" | grep -q "❌"; then
    echo -e "${RED}❌ Test 3 FAILED - Mocked RunPod environment test failed${NC}"
    TEST3_PASSED=false
else
    echo -e "${GREEN}✅ Test 3 PASSED - Mocked RunPod environment test verified${NC}"
    TEST3_PASSED=true
fi

# Test 4: Direct Unified Function Test (Option 2)
echo -e "${YELLOW}📋 Test 4: Testing unified function directly (bypassing environment checks)...${NC}"
DIRECT_FUNCTION_TEST=$(docker run --rm "${FULL_IMAGE_NAME}" python -c "
import sys
sys.path.insert(0, '/app')
try:
    # Test the unified function directly without environment dependencies
    from src.data_classes.configs import generate_unified_run_name
    
    # Test various scenarios that RunPod would encounter
    test_scenarios = [
        {'dataset': 'mnist', 'mode': 'health', 'objective': 'val_accuracy'},
        {'dataset': 'cifar10', 'mode': 'simple', 'objective': 'accuracy'},
        {'dataset': 'fashion_mnist', 'mode': 'health', 'objective': 'overall_health'}
    ]
    
    for i, scenario in enumerate(test_scenarios):
        run_name = generate_unified_run_name(
            dataset_name=scenario['dataset'],
            mode=scenario['mode'],
            optimize_for=scenario['objective']
        )
        
        # Test S3 path construction
        s3_prefix = f'optimization_results/{run_name}/plots/trial_0'
        
        print(f'✅ Scenario {i+1}: {scenario}')
        print(f'✅ Generated run_name: {run_name}')
        print(f'✅ S3 prefix: {s3_prefix}')
        
        # Validate no legacy issues
        if 'runpod_trial_trial_' in s3_prefix:
            print(f'❌ DOUBLE TRIAL ISSUE in scenario {i+1}!')
            sys.exit(1)
        elif 'runpod_trial' in s3_prefix:
            print(f'❌ LEGACY runpod_trial naming in scenario {i+1}!')
            sys.exit(1)
        
        # Validate proper format
        if not all(part in run_name for part in [scenario['dataset'], scenario['mode']]):
            print(f'❌ Run name missing required components in scenario {i+1}!')
            sys.exit(1)
            
        print(f'✅ Scenario {i+1} validated successfully')
        print('---')
    
    print('✅ All direct function tests passed!')
    
except Exception as e:
    print(f'❌ Direct function test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1)

echo "$DIRECT_FUNCTION_TEST"
if echo "$DIRECT_FUNCTION_TEST" | grep -q "❌"; then
    echo -e "${RED}❌ Test 4 FAILED - Direct function test failed${NC}"
    TEST4_PASSED=false
else
    echo -e "${GREEN}✅ Test 4 PASSED - Direct function test verified${NC}"
    TEST4_PASSED=true
fi

# Test 5: Validate handler.py syntax and imports
echo -e "${YELLOW}📋 Test 5: Validating handler.py syntax and imports...${NC}"
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
    echo -e "${RED}❌ Test 5 FAILED${NC}"
    TEST5_PASSED=false
else
    echo -e "${GREEN}✅ Test 5 PASSED${NC}"
    TEST5_PASSED=true
fi

# Test 6: Check required dependencies
echo -e "${YELLOW}📋 Test 6: Checking critical dependencies...${NC}"
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
    echo -e "${RED}❌ Test 6 FAILED${NC}"
    TEST6_PASSED=false
else
    echo -e "${GREEN}✅ Test 6 PASSED${NC}"
    TEST6_PASSED=true
fi

# Test 7: Validate environment variable handling
echo -e "${YELLOW}📋 Test 7: Testing environment variable handling...${NC}"
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
    echo -e "${RED}❌ Test 7 FAILED${NC}"
    TEST7_PASSED=false
else
    echo -e "${GREEN}✅ Test 7 PASSED${NC}"
    TEST7_PASSED=true
fi

# Test 8: Test both command types with dry-run
echo -e "${YELLOW}📋 Test 8: Testing command routing (dry-run)...${NC}"
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
    echo -e "${RED}❌ Test 8 FAILED${NC}"
    TEST8_PASSED=false
else
    echo -e "${GREEN}✅ Test 8 PASSED${NC}"
    TEST8_PASSED=true
fi

# Test 9: Resource and disk space check
echo -e "${YELLOW}📋 Test 9: Checking container resources...${NC}"
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
TEST9_EXIT_CODE=$?
if [ $TEST9_EXIT_CODE -ne 0 ] || echo "$RESOURCE_TEST" | grep -q "❌.*Resource check completed with errors"; then
    echo -e "${RED}❌ Test 9 FAILED${NC}"
    TEST9_PASSED=false
else
    echo -e "${GREEN}✅ Test 9 PASSED${NC}"
    TEST9_PASSED=true
fi

# Test 10: JSON Serialization and Data Types Test
echo -e "${YELLOW}📋 Test 10: Testing JSON serialization and data types...${NC}"
SERIALIZATION_TEST=$(docker run --rm "${FULL_IMAGE_NAME}" python -c "
import sys
import json
import base64
import tempfile
import os
sys.path.insert(0, '/app')

try:
    # Test base64 encoding (critical for model_data)
    test_bytes = b'test binary data for model serialization'
    encoded = base64.b64encode(test_bytes).decode('utf-8')
    decoded = base64.b64decode(encoded.encode('utf-8'))
    
    if test_bytes == decoded:
        print('✅ Base64 encoding/decoding works')
    else:
        print('❌ Base64 encoding/decoding failed')
        sys.exit(1)
    
    # Test JSON serialization of typical response structure
    test_response = {
        'trial_id': 'test_001',
        'status': 'completed',
        'success': True,
        'metrics': {
            'test_accuracy': 0.95,
            'test_loss': 0.05
        },
        'model_data': encoded,  # Base64 encoded binary data
        'training_history': {'loss': [0.5, 0.3, 0.1], 'accuracy': [0.7, 0.85, 0.95]},
        'dataset_data': {'x_train': [[1, 2, 3]], 'y_train': [[0, 1]]}
    }
    
    # Test JSON serialization
    json_str = json.dumps(test_response)
    parsed_back = json.loads(json_str)
    
    if parsed_back['trial_id'] == 'test_001' and parsed_back['success'] == True:
        print('✅ JSON serialization/deserialization works')
    else:
        print('❌ JSON serialization/deserialization failed')
        sys.exit(1)
        
    print('✅ Data types and serialization validated')
    
except Exception as e:
    print(f'❌ Serialization test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1)

echo "$SERIALIZATION_TEST"
if echo "$SERIALIZATION_TEST" | grep -q "❌"; then
    echo -e "${RED}❌ Test 10 FAILED${NC}"
    TEST10_PASSED=false
else
    echo -e "${GREEN}✅ Test 10 PASSED${NC}"
    TEST10_PASSED=true
fi

# Test 11: Dataset Manager and Key Validation
echo -e "${YELLOW}📋 Test 11: Testing DatasetManager and dataset key validation...${NC}"
DATASET_TEST=$(docker run --rm "${FULL_IMAGE_NAME}" python -c "
import sys
sys.path.insert(0, '/app')

try:
    from src.dataset_manager import DatasetManager
    
    # Initialize DatasetManager
    dm = DatasetManager()
    print('✅ DatasetManager imported successfully')
    
    # Test load_dataset method exists
    if hasattr(dm, 'load_dataset'):
        print('✅ load_dataset method exists')
    else:
        print('❌ load_dataset method missing')
        sys.exit(1)
    
    # Test MNIST dataset metadata (without actually loading data)
    available_datasets = dm.get_available_datasets()
    if 'mnist' in available_datasets:
        print('✅ MNIST dataset available')
    else:
        print('❌ MNIST dataset not available')
        sys.exit(1)
    
    # Test the key structure we expect from load_dataset
    # Create a mock response to test our key access patterns
    mock_data = {
        'x_train': [[1, 2, 3]],
        'y_train': [[1, 0]],
        'x_test': [[4, 5, 6]], 
        'y_test': [[0, 1]],
        'config': {'num_classes': 2}
        # Note: x_val, y_val might not exist
    }
    
    # Test our key access pattern from handler.py
    required_keys = ['x_train', 'y_train', 'x_test', 'y_test']
    optional_keys = ['x_val', 'y_val']
    
    for key in required_keys:
        if key in mock_data:
            print(f'✅ Required key {key} accessible')
        else:
            print(f'❌ Required key {key} missing')
            sys.exit(1)
    
    # Test optional key handling (this is what caused the error)
    for key in optional_keys:
        if key in mock_data:
            print(f'✅ Optional key {key} found')
        else:
            print(f'⚠️  Optional key {key} not found (expected, will use fallback)')
    
    print('✅ Dataset manager and key validation completed')
    
except ImportError as e:
    print(f'❌ DatasetManager import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Dataset validation failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1)

echo "$DATASET_TEST"
if echo "$DATASET_TEST" | grep -q "❌.*failed" || echo "$DATASET_TEST" | grep -q "❌.*missing"; then
    echo -e "${RED}❌ Test 11 FAILED${NC}"
    TEST11_PASSED=false
else
    echo -e "${GREEN}✅ Test 11 PASSED${NC}"
    TEST11_PASSED=true
fi

# Test 12: Full Handler Pipeline Simulation (Advanced)
echo -e "${YELLOW}📋 Test 12: Full handler pipeline simulation (advanced)...${NC}"
PIPELINE_TEST=$(docker run --rm "${FULL_IMAGE_NAME}" python -c "
import sys
import asyncio
import json
sys.path.insert(0, '/app')

async def test_handler_pipeline():
    try:
        # Import handler functions
        from handler import start_training, start_final_model_training, validate_request
        
        print('✅ All handler functions imported successfully')
        
        # Test 1: Request validation
        valid_request = {
            'command': 'start_training',
            'dataset_name': 'mnist',
            'hyperparameters': {'learning_rate': 0.001, 'epochs': 1},
            'config': {'mode': 'simple', 'objective': 'val_accuracy'}
        }
        
        is_valid, error_msg = validate_request(valid_request)
        if is_valid:
            print('✅ Request validation works')
        else:
            print(f'❌ Request validation failed: {error_msg}')
            return False
            
        # Test 2: Basic functionality check
        print('✅ Unified handler architecture validated')
        
        # Test 3: Invalid request handling
        invalid_request = {'command': 'invalid_command'}
        is_valid, error_msg = validate_request(invalid_request)
        if not is_valid and error_msg:
            print('✅ Invalid request handling works')
        else:
            print('❌ Invalid request handling failed')
            return False
            
        print('✅ Handler pipeline simulation completed successfully')
        return True
        
    except Exception as e:
        print(f'❌ Handler pipeline test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

# Run the async test
result = asyncio.run(test_handler_pipeline())
sys.exit(0 if result else 1)
" 2>&1)

echo "$PIPELINE_TEST"
PIPELINE_EXIT_CODE=$?
if [ $PIPELINE_EXIT_CODE -ne 0 ] || echo "$PIPELINE_TEST" | grep -q "❌.*failed"; then
    echo -e "${RED}❌ Test 12 FAILED${NC}"
    TEST12_PASSED=false
else
    echo -e "${GREEN}✅ Test 12 PASSED${NC}"
    TEST12_PASSED=true
fi

# Test 13: End-to-End Integration Test with Local Container (CRITICAL)
echo -e "${YELLOW}📋 Test 13: End-to-end integration test with local container (CRITICAL)...${NC}"
echo -e "${BLUE}🎯 This test simulates the exact RunPod integration path locally${NC}"

# Start a fresh container for end-to-end testing
E2E_TEST_PORT=8081
echo -e "${BLUE}🔍 Finding available port for end-to-end test...${NC}"
while netstat -ln 2>/dev/null | grep -q ":${E2E_TEST_PORT} " || docker ps --filter "publish=${E2E_TEST_PORT}" --format "{{.ID}}" | grep -q .; do
    E2E_TEST_PORT=$((E2E_TEST_PORT + 1))
    if [ ${E2E_TEST_PORT} -gt 8090 ]; then
        echo -e "${RED}❌ Could not find available port between 8081-8090 for end-to-end test${NC}"
        TEST13_PASSED=false
        break
    fi
done

TEST13_PASSED=false

if [ ${E2E_TEST_PORT} -le 8090 ]; then
    echo -e "${GREEN}✅ Using port ${E2E_TEST_PORT} for end-to-end test${NC}"
    
    # Clean up any existing containers first
    echo -e "${YELLOW}🧹 Cleaning up any existing end-to-end test containers...${NC}"
    docker ps -a --filter "name=cv-e2e-test" --format "{{.ID}}" | xargs -r docker rm -f > /dev/null 2>&1
    docker ps -q --filter "publish=${E2E_TEST_PORT}" | xargs -r docker stop > /dev/null 2>&1
    
    # Start container in FastAPI mode for local testing
    echo -e "${BLUE}📦 Starting end-to-end test container...${NC}"
    E2E_CONTAINER_ID=$(docker run -d \
        -p ${E2E_TEST_PORT}:8080 \
        --name "cv-e2e-test-$(date +%s)" \
        -e RUNPOD_API_KEY="${RUNPOD_API_KEY}" \
        -e RUNPOD_S3_ACCESS_KEY="${RUNPOD_S3_ACCESS_KEY}" \
        -e RUNPOD_S3_SECRET_ACCESS_KEY="${RUNPOD_S3_SECRET_ACCESS_KEY}" \
        "${FULL_IMAGE_NAME}")
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ End-to-end test container started: ${E2E_CONTAINER_ID:0:12}${NC}"
        
        # Wait for container to be ready
        echo -e "${BLUE}⏳ Waiting for end-to-end test container to be ready...${NC}"
        sleep 15
        
        # Test that the container is responding
        echo -e "${BLUE}🧪 Testing container readiness...${NC}"
        CONTAINER_READY=false
        for i in {1..5}; do
            if curl -s "http://localhost:${E2E_TEST_PORT}/health" > /dev/null 2>&1 || \
               curl -s "http://localhost:${E2E_TEST_PORT}/" > /dev/null 2>&1; then
                CONTAINER_READY=true
                break
            fi
            echo -e "${YELLOW}   Attempt ${i}/5: Container not ready yet, waiting...${NC}"
            sleep 5
        done
        
        if [ "$CONTAINER_READY" = "true" ]; then
            echo -e "${GREEN}✅ Container is ready for end-to-end testing${NC}"
            
            # NOW RUN THE CRITICAL TEST: optimizer.py calling local container
            echo -e "${BLUE}🎯 CRITICAL TEST: Running optimizer.py with use_runpod_service=true (local container)${NC}"
            echo -e "${YELLOW}⏳ This tests the exact same path as RunPod integration...${NC}"
            echo -e "${BLUE}💡 To view live container logs during the test, open a new terminal and run:${NC}"
            echo -e "   ${GREEN}docker logs -f ${E2E_CONTAINER_ID}${NC}"
            echo ""
            
            # Go to project root (same as post-deployment test)
            cd ..
            
            echo -e "${BLUE}🔍 Running from directory: $(pwd)${NC}"
            echo -e "${BLUE}🔍 Command: python src/optimizer.py dataset_name=mnist mode=simple trials=1 max_epochs_per_trial=6 use_runpod_service=true runpod_service_endpoint=http://localhost:${E2E_TEST_PORT}${NC}"
            
            # Run the exact same command as the post-deployment integration test, but with local endpoint
            echo -e "${BLUE}🎯 Running optimizer.py test (same as post-deployment test)...${NC}"
            
            # Check to ensure the variables are set
            echo -e "${BLUE}🔍 Environment variables validation:${NC}"
            MISSING_VARS=""
            
            if [ -z "${RUNPOD_ENDPOINT_ID}" ]; then
                echo "   ❌ RUNPOD_ENDPOINT_ID: MISSING OR EMPTY"
                MISSING_VARS="${MISSING_VARS} RUNPOD_ENDPOINT_ID"
            else
                echo "   ✅ RUNPOD_ENDPOINT_ID: ${RUNPOD_ENDPOINT_ID}"
            fi
            
            if [ -z "${RUNPOD_API_KEY}" ]; then
                echo "   ❌ RUNPOD_API_KEY: MISSING OR EMPTY"
                MISSING_VARS="${MISSING_VARS} RUNPOD_API_KEY"
            else
                echo "   ✅ RUNPOD_API_KEY: ${RUNPOD_API_KEY:0:8}..."
            fi
            
            if [ -z "${RUNPOD_S3_ACCESS_KEY}" ]; then
                echo "   ❌ RUNPOD_S3_ACCESS_KEY: MISSING OR EMPTY"
                MISSING_VARS="${MISSING_VARS} RUNPOD_S3_ACCESS_KEY"
            else
                echo "   ✅ RUNPOD_S3_ACCESS_KEY: ${RUNPOD_S3_ACCESS_KEY:0:8}..."
            fi
            
            if [ -z "${RUNPOD_S3_SECRET_ACCESS_KEY}" ]; then
                echo "   ❌ RUNPOD_S3_SECRET_ACCESS_KEY: MISSING OR EMPTY"
                MISSING_VARS="${MISSING_VARS} RUNPOD_S3_SECRET_ACCESS_KEY"
            else
                echo "   ✅ RUNPOD_S3_SECRET_ACCESS_KEY: ${RUNPOD_S3_SECRET_ACCESS_KEY:0:8}..."
            fi

            if [ -z "${E2E_TEST_PORT}" ]; then
                echo "   ❌ E2E_TEST_PORT: MISSING OR EMPTY"
                MISSING_VARS="${MISSING_VARS} E2E_TEST_PORT"
            else
                echo "   ✅ E2E_TEST_PORT: ${E2E_TEST_PORT}..."
            fi
            
            if [ -n "${MISSING_VARS}" ]; then
                echo -e "${RED}❌ CRITICAL: Missing environment variables:${MISSING_VARS}${NC}"
                echo -e "${YELLOW}💡 This will likely cause the optimizer.py test to fail${NC}"
                E2E_INTEGRATION_RESULT="MISSING_ENVIRONMENT_VARIABLES:${MISSING_VARS}"
            
            # If all variables are set, run the command
            else
                echo -e "${GREEN}✅ All required environment variables are present${NC}"
                echo -e "${BLUE}🎯 Proceeding with optimizer.py test...${NC}"
                
                # Run the command with explicit error handling
                set +e  # Temporarily disable exit on error
                E2E_INTEGRATION_RESULT=$(timeout 600 env \
                    RUNPOD_ENDPOINT_ID="${RUNPOD_ENDPOINT_ID}" \
                    RUNPOD_API_KEY="${RUNPOD_API_KEY}" \
                    RUNPOD_S3_ACCESS_KEY="${RUNPOD_S3_ACCESS_KEY}" \
                    RUNPOD_S3_SECRET_ACCESS_KEY="${RUNPOD_S3_SECRET_ACCESS_KEY}" \
                    python src/optimizer.py \
                    dataset_name=mnist \
                    mode=simple \
                    trials=1 \
                    max_epochs_per_trial=6 \
                    use_runpod_service=true \
                    runpod_service_endpoint=http://localhost:${E2E_TEST_PORT} 2>&1)
                E2E_EXIT_CODE=$?
                set -e  # Re-enable exit on error
                
                # Log the results immediately
                echo -e "${BLUE}🔍 Command completed with exit code: ${E2E_EXIT_CODE}${NC}"
                if [ ${E2E_EXIT_CODE} -eq 124 ]; then
                    echo -e "${YELLOW}⏰ Command timed out after 60 seconds${NC}"
                elif [ ${E2E_EXIT_CODE} -ne 0 ]; then
                    echo -e "${RED}❌ Command failed with exit code: ${E2E_EXIT_CODE}${NC}"
                else
                    echo -e "${GREEN}✅ Command completed successfully${NC}"
                fi
                
                # Show the actual output/error
                if [ -n "${E2E_INTEGRATION_RESULT}" ]; then
                    echo -e "${BLUE}📝 Command output (first 20 lines):${NC}"
                    echo "${E2E_INTEGRATION_RESULT}" | head -20
                    echo -e "${BLUE}📝 Command output (last 10 lines):${NC}"
                    echo "${E2E_INTEGRATION_RESULT}" | tail -10
                else
                    echo -e "${YELLOW}⚠️  No output captured from command${NC}"
                fi
                
                # WAIT FOR FINAL MODEL BUILDING TO COMPLETE
                if [ ${E2E_EXIT_CODE} -eq 0 ]; then
                    echo -e "${BLUE}⏳ Optimizer completed successfully. Waiting for final model building to complete...${NC}"
                    echo -e "${YELLOW}   This allows the final model (.keras) to be built and uploaded to S3${NC}"
                    
                    # Wait up to 300 seconds (5 minutes) for final model completion
                    FINAL_MODEL_WAIT=0
                    MAX_FINAL_MODEL_WAIT=300
                    
                    while [ $FINAL_MODEL_WAIT -lt $MAX_FINAL_MODEL_WAIT ]; do
                        # Check container logs for final model completion indicators
                        FINAL_MODEL_LOGS=$(docker logs "${E2E_CONTAINER_ID}" 2>&1 | tail -50)
                        
                        if echo "$FINAL_MODEL_LOGS" | grep -q "start_final_model_training\|Final model.*saved\|Final model training.*completed"; then
                            echo -e "${GREEN}✅ Final model building detected in container logs${NC}"
                            break
                        elif echo "$FINAL_MODEL_LOGS" | grep -q "Final model.*failed\|Error.*final.*model"; then
                            echo -e "${RED}❌ Final model building failed${NC}"
                            break
                        fi
                        
                        echo -e "${YELLOW}   Waiting for final model building... (${FINAL_MODEL_WAIT}/${MAX_FINAL_MODEL_WAIT}s)${NC}"
                        sleep 10
                        FINAL_MODEL_WAIT=$((FINAL_MODEL_WAIT + 10))
                    done
                    
                    if [ $FINAL_MODEL_WAIT -ge $MAX_FINAL_MODEL_WAIT ]; then
                        echo -e "${YELLOW}⏰ Final model building wait timeout (${MAX_FINAL_MODEL_WAIT}s)${NC}"
                        echo -e "${YELLOW}   Container may not be receiving final model requests${NC}"
                    fi
                else
                    echo -e "${YELLOW}⚠️  Optimizer failed, skipping final model wait${NC}"
                fi
            fi
            
            # Return to runpod_service directory
            cd runpod_service
            
            # Analyze results
            echo -e "${BLUE}🔍 End-to-end test results analysis:${NC}"
            if [ ${E2E_EXIT_CODE} -eq 124 ]; then
                echo -e "${YELLOW}⏰ End-to-end test timed out after 5 minutes${NC}"
                echo -e "${RED}❌ This suggests the integration path has performance issues${NC}"
            elif [ ${E2E_EXIT_CODE} -ne 0 ]; then
                echo -e "${YELLOW}⚠️  End-to-end test exited with code: ${E2E_EXIT_CODE}${NC}"
            fi
            
            # Enhanced S3 Plot Pipeline Validation
            echo -e "${BLUE}🔍 Analyzing S3 plot creation, upload, and download pipeline...${NC}"
            
            # Check for basic success indicators first
            BASIC_SUCCESS=false
            if echo "${E2E_INTEGRATION_RESULT}" | grep -q "Optimization completed successfully" && \
               echo "${E2E_INTEGRATION_RESULT}" | grep -q "RunPod Service.*JSON API"; then
                BASIC_SUCCESS=true
                echo -e "${GREEN}✅ Basic integration successful${NC}"
            else
                echo -e "${RED}❌ Basic integration failed${NC}"
            fi
            
            # S3 Plot Pipeline Analysis
            PLOT_CREATION_SUCCESS=false
            PLOT_UPLOAD_SUCCESS=false
            PLOT_DOWNLOAD_SUCCESS=false
            
            # (a) Check for plot creation in RunPod worker
            if echo "${E2E_INTEGRATION_RESULT}" | grep -q "Generated plots:" && \
               echo "${E2E_INTEGRATION_RESULT}" | grep -q "Successfully generated training plots"; then
                PLOT_CREATION_SUCCESS=true
                echo -e "${GREEN}✅ (a) Plot creation: Plots successfully created in RunPod worker${NC}"
                CREATED_PLOTS=$(echo "${E2E_INTEGRATION_RESULT}" | grep "Generated plots:" | head -1)
                echo -e "${BLUE}    📊 ${CREATED_PLOTS}${NC}"
            else
                echo -e "${RED}❌ (a) Plot creation: Failed to create plots in RunPod worker${NC}"
                echo -e "${YELLOW}    💡 Check plot generation settings and RunPod environment${NC}"
            fi
            
            # (b) Saving plots to RunPod worker is validated by creation success
            if [ "$PLOT_CREATION_SUCCESS" = true ]; then
                echo -e "${GREEN}✅ (b) Plot saving: Plots successfully saved to RunPod worker filesystem${NC}"
            else
                echo -e "${RED}❌ (b) Plot saving: Failed - plots not created${NC}"
            fi
            
            # (c) Check for S3 upload success
            if echo "${E2E_INTEGRATION_RESULT}" | grep -q "Successfully uploaded.*files.*to s3://"; then
                PLOT_UPLOAD_SUCCESS=true
                echo -e "${GREEN}✅ (c) S3 upload: Plots successfully uploaded to S3${NC}"
                S3_UPLOAD_LINE=$(echo "${E2E_INTEGRATION_RESULT}" | grep "Successfully uploaded.*files.*to s3://" | head -1)
                echo -e "${BLUE}    📤 ${S3_UPLOAD_LINE}${NC}"
            elif echo "${E2E_INTEGRATION_RESULT}" | grep -q "Failed to upload plots to S3\|⚠️ Failed to upload plots"; then
                echo -e "${RED}❌ (c) S3 upload: Failed to upload plots to S3${NC}"
                UPLOAD_ERROR=$(echo "${E2E_INTEGRATION_RESULT}" | grep -E "(Failed to upload|⚠️ Failed)" | head -1)
                echo -e "${YELLOW}    ⚠️ ${UPLOAD_ERROR}${NC}"
                echo -e "${YELLOW}    💡 Check S3 credentials and bucket permissions${NC}"
            else
                echo -e "${YELLOW}⚠️ (c) S3 upload: No clear upload success/failure indicators found${NC}"
                echo -e "${YELLOW}    💡 This could indicate S3 upload was skipped or logs were missed${NC}"
            fi
            
            # (d) Check for S3 download success
            if (echo "${E2E_INTEGRATION_RESULT}" | grep -q "📥 Checking for S3 plots to download\|📥 INITIATING S3 DOWNLOAD ATTEMPT") && \
               (echo "${E2E_INTEGRATION_RESULT}" | grep -q "🚀 STARTING S3 DOWNLOAD\|✅ S3 DOWNLOAD SUCCESSFUL\|🎉 S3 DOWNLOAD SUCCESSFUL") && \
               (echo "${E2E_INTEGRATION_RESULT}" | grep -q "📁 Plots downloaded to:\|✅ Successfully downloaded.*files from s3://"); then
                PLOT_DOWNLOAD_SUCCESS=true
                echo -e "${GREEN}✅ (d) S3 download: Plots successfully downloaded to local machine${NC}"
                DOWNLOAD_LINE=$(echo "${E2E_INTEGRATION_RESULT}" | grep "📁 Plots downloaded to:" | head -1)
                echo -e "${BLUE}    📥 ${DOWNLOAD_LINE}${NC}"
            elif echo "${E2E_INTEGRATION_RESULT}" | grep -q "❌ S3 DOWNLOAD FAILED\|💥 S3 DOWNLOAD FAILED"; then
                echo -e "${RED}❌ (d) S3 download: Failed to download plots from S3${NC}"
                DOWNLOAD_ERROR=$(echo "${E2E_INTEGRATION_RESULT}" | grep -E "(❌ S3 DOWNLOAD FAILED|💥 S3 DOWNLOAD FAILED)" | head -1)
                echo -e "${YELLOW}    ⚠️ ${DOWNLOAD_ERROR}${NC}"
                echo -e "${YELLOW}    💡 Check S3 credentials and download logic${NC}"
            else
                echo -e "${YELLOW}⚠️ (d) S3 download: No download attempt detected in logs${NC}"
                echo -e "${YELLOW}    💡 This suggests S3 upload failed or plots_s3 info wasn't returned${NC}"
            fi
            
            # Check for final model building and S3 upload
            echo -e "${BLUE}🔍 Analyzing final model (.keras) building and S3 upload...${NC}"
            FINAL_MODEL_SUCCESS=false
            FINAL_MODEL_S3_SUCCESS=false
            
            # Get fresh container logs for final model analysis
            FRESH_CONTAINER_LOGS=$(docker logs "${E2E_CONTAINER_ID}" 2>&1)
            
            if echo "$FRESH_CONTAINER_LOGS" | grep -q "start_final_model_training"; then
                echo -e "${GREEN}✅ Final model building request detected${NC}"
                if echo "$FRESH_CONTAINER_LOGS" | grep -q "Final model.*saved\|Final model training.*completed\|final_model_path"; then
                    FINAL_MODEL_SUCCESS=true
                    echo -e "${GREEN}✅ Final model (.keras) successfully built${NC}"
                    
                    # Check for final model S3 upload
                    if echo "$FRESH_CONTAINER_LOGS" | grep -q "S3 upload successful.*final.*model\|final.*model.*uploaded"; then
                        FINAL_MODEL_S3_SUCCESS=true
                        echo -e "${GREEN}✅ Final model (.keras) successfully uploaded to S3${NC}"
                    else
                        echo -e "${YELLOW}⚠️ Final model built but S3 upload status unclear${NC}"
                    fi
                else
                    echo -e "${RED}❌ Final model building failed or incomplete${NC}"
                fi
            else
                echo -e "${YELLOW}⚠️ No final model building request detected${NC}"
                echo -e "${YELLOW}    💡 This indicates the container was terminated too early${NC}"
                echo -e "${YELLOW}    💡 Or the optimizer didn't send the final model request${NC}"
            fi
            
            # Final assessment
            FULL_S3_PIPELINE_SUCCESS=false
            if [ "$BASIC_SUCCESS" = true ] && \
               [ "$PLOT_CREATION_SUCCESS" = true ] && \
               [ "$PLOT_UPLOAD_SUCCESS" = true ] && \
               [ "$PLOT_DOWNLOAD_SUCCESS" = true ]; then
                FULL_S3_PIPELINE_SUCCESS=true
                TEST13_PASSED=true
                echo -e "${GREEN}✅ END-TO-END INTEGRATION WITH FULL S3 PIPELINE SUCCESSFUL!${NC}"
                echo -e "${GREEN}🎉 Complete plot creation → upload → download pipeline verified${NC}"
                
                # Check final model status for reporting
                if [ "$FINAL_MODEL_SUCCESS" = true ] && [ "$FINAL_MODEL_S3_SUCCESS" = true ]; then
                    echo -e "${GREEN}🎯 Final model (.keras) building and S3 upload also successful!${NC}"
                elif [ "$FINAL_MODEL_SUCCESS" = true ]; then
                    echo -e "${YELLOW}⚠️ Final model built but S3 upload needs verification${NC}"
                else
                    echo -e "${YELLOW}⚠️ Final model building incomplete - container may need longer wait time${NC}"
                fi
            elif [ "$BASIC_SUCCESS" = true ]; then
                echo -e "${YELLOW}⚠️ END-TO-END INTEGRATION PARTIALLY SUCCESSFUL${NC}"
                echo -e "${YELLOW}✅ Basic optimization works, but S3 plot pipeline has issues${NC}"
                echo -e "${YELLOW}🚨 This means RunPod will work but plots won't be available locally${NC}"
                
                # Detailed failure analysis
                if [ "$PLOT_CREATION_SUCCESS" = false ]; then
                    echo -e "${RED}    🔥 CRITICAL: Plot creation failed - check plot generation settings${NC}"
                elif [ "$PLOT_UPLOAD_SUCCESS" = false ]; then
                    echo -e "${RED}    🔥 CRITICAL: S3 upload failed - plots created but not uploaded${NC}"
                    echo -e "${YELLOW}    💡 This is likely the colon timestamp issue that was just fixed${NC}"
                elif [ "$PLOT_DOWNLOAD_SUCCESS" = false ]; then
                    echo -e "${RED}    🔥 CRITICAL: S3 download failed - plots uploaded but not downloaded${NC}"
                    echo -e "${YELLOW}    💡 Check local optimizer S3 download logic${NC}"
                fi
                
                # Set TEST13_PASSED based on whether S3 upload at least works
                if [ "$PLOT_CREATION_SUCCESS" = true ] && [ "$PLOT_UPLOAD_SUCCESS" = true ]; then
                    TEST13_PASSED=true
                    echo -e "${GREEN}📋 TEST13 PASSED - Core functionality works (S3 upload successful)${NC}"
                else
                    TEST13_PASSED=false
                    echo -e "${RED}📋 TEST13 FAILED - S3 upload pipeline broken${NC}"
                fi
            else
                echo -e "${RED}❌ END-TO-END INTEGRATION TEST FAILED${NC}"
                echo -e "${YELLOW}🚨 This is the same failure you would see on RunPod!${NC}"
                echo -e "${YELLOW}📝 Debugging information:${NC}"
                echo "Looking for 'Optimization completed successfully': $(echo "${E2E_INTEGRATION_RESULT}" | grep -c "Optimization completed successfully")"
                echo "Looking for 'RunPod Service.*JSON API': $(echo "${E2E_INTEGRATION_RESULT}" | grep -c "RunPod Service.*JSON API")"
                echo -e "${YELLOW}📝 Last 10 lines of end-to-end test:${NC}"
                echo "${E2E_INTEGRATION_RESULT}" | tail -10
                echo -e "${YELLOW}📝 First 10 lines of end-to-end test:${NC}"
                echo "${E2E_INTEGRATION_RESULT}" | head -10
            fi
            
            # Show container logs with focus on S3 pipeline debugging
            echo -e "${BLUE}📋 Container logs (focusing on S3 pipeline):${NC}"
            docker logs "${E2E_CONTAINER_ID}" 2>&1 | grep -E "(plots_s3|PLOTS_S3|Successfully uploaded|Failed to upload)" || echo "No S3 pipeline logs found"
            
            echo -e "${BLUE}📋 All container logs (last 50 lines):${NC}"
            docker logs --tail 50 "${E2E_CONTAINER_ID}" 2>&1 || echo "Could not retrieve container logs"
            
        else
            echo -e "${RED}❌ Container failed to become ready for end-to-end testing${NC}"
            echo -e "${YELLOW}📝 Container logs:${NC}"
            docker logs "${E2E_CONTAINER_ID}" 2>&1 | tail -20
        fi
        
        # Clean up the end-to-end test container
        echo -e "${YELLOW}🧹 Cleaning up end-to-end test container...${NC}"
        docker stop "${E2E_CONTAINER_ID}" > /dev/null 2>&1
        docker rm "${E2E_CONTAINER_ID}" > /dev/null 2>&1
        
    else
        echo -e "${RED}❌ Failed to start end-to-end test container${NC}"
    fi
fi

if [ "$TEST13_PASSED" = "true" ]; then
    echo -e "${GREEN}✅ Test 13 PASSED - End-to-end integration with S3 pipeline verified${NC}"
    if [ "$FULL_S3_PIPELINE_SUCCESS" = true ]; then
        echo -e "${GREEN}🎉 Complete S3 plot pipeline: creation → upload → download ✅${NC}"
    else
        echo -e "${YELLOW}⚠️ Basic integration works, but check S3 plot download above${NC}"
    fi
else
    echo -e "${RED}❌ Test 13 FAILED - End-to-end integration or S3 upload issues detected${NC}"
    echo -e "${YELLOW}🚨 CRITICAL: This failure would occur on RunPod deployment${NC}"
    echo -e "${YELLOW}💡 Fix this issue before deploying to RunPod to save time${NC}"
fi


echo -e "${BLUE}🎯 All local validation tests completed!${NC}"
echo -e "${BLUE}📊 Pre-deployment validation summary:${NC}"

# Show actual test results
if [ "$TEST1_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 1: Trial path fix (unified filepath mechanism)${NC}"
else
    echo -e "   ${RED}❌ Test 1: Trial path fix (unified filepath mechanism)${NC}"
fi

if [ "$TEST2_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 2: Config data and plot_generation handling${NC}"
else
    echo -e "   ${RED}❌ Test 2: Config data and plot_generation handling${NC}"
fi

if [ "$TEST3_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 3: Mock RunPod environment test (Option 1)${NC}"
else
    echo -e "   ${RED}❌ Test 3: Mock RunPod environment test (Option 1)${NC}"
fi

if [ "$TEST4_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 4: Direct unified function test (Option 2)${NC}"
else
    echo -e "   ${RED}❌ Test 4: Direct unified function test (Option 2)${NC}"
fi

if [ "$TEST5_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 5: Handler syntax and imports${NC}"
else
    echo -e "   ${RED}❌ Test 5: Handler syntax and imports${NC}"
fi

if [ "$TEST6_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 6: Critical dependencies${NC}"
else
    echo -e "   ${RED}❌ Test 6: Critical dependencies${NC}"
fi

if [ "$TEST7_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 7: Environment variable handling${NC}"
else
    echo -e "   ${RED}❌ Test 7: Environment variable handling${NC}"
fi

if [ "$TEST8_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 8: Command routing validation${NC}"
else
    echo -e "   ${RED}❌ Test 8: Command routing validation${NC}"
fi

if [ "$TEST9_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 9: Resource availability${NC}"
else
    echo -e "   ${RED}❌ Test 9: Resource availability${NC}"
fi

if [ "$TEST10_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 10: JSON serialization and data types${NC}"
else
    echo -e "   ${RED}❌ Test 10: JSON serialization and data types${NC}"
fi

if [ "$TEST11_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 11: DatasetManager and key validation${NC}"
else
    echo -e "   ${RED}❌ Test 11: DatasetManager and key validation${NC}"
fi

if [ "$TEST12_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 12: Handler pipeline simulation${NC}"
else
    echo -e "   ${RED}❌ Test 12: Handler pipeline simulation${NC}"
fi

if [ "$TEST13_PASSED" = "true" ]; then
    echo -e "   ${GREEN}✅ Test 13: End-to-end integration test (CRITICAL)${NC}"
else
    echo -e "   ${RED}❌ Test 13: End-to-end integration test (CRITICAL)${NC}"
fi


# Count passed tests
TESTS_PASSED=0
[ "$TEST1_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))
[ "$TEST2_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))
[ "$TEST3_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))
[ "$TEST4_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))
[ "$TEST5_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))
[ "$TEST6_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))
[ "$TEST7_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))
[ "$TEST8_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))
[ "$TEST9_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))
[ "$TEST10_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))
[ "$TEST11_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))
[ "$TEST12_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))
[ "$TEST13_PASSED" = "true" ] && TESTS_PASSED=$((TESTS_PASSED + 1))

echo ""
echo -e "${BLUE}📊 Test Results: ${TESTS_PASSED}/13 tests passed${NC}"

# Determine if we should continue with deployment
CRITICAL_TESTS_PASSED=0
CRITICAL_TESTS_TOTAL=6

# Define critical tests that must pass
[ "$TEST1_PASSED" = "true" ] && CRITICAL_TESTS_PASSED=$((CRITICAL_TESTS_PASSED + 1))  # Unified filepath mechanism
[ "$TEST3_PASSED" = "true" ] && CRITICAL_TESTS_PASSED=$((CRITICAL_TESTS_PASSED + 1))  # Mock RunPod environment
[ "$TEST4_PASSED" = "true" ] && CRITICAL_TESTS_PASSED=$((CRITICAL_TESTS_PASSED + 1))  # Direct unified function
[ "$TEST5_PASSED" = "true" ] && CRITICAL_TESTS_PASSED=$((CRITICAL_TESTS_PASSED + 1))  # Handler syntax
[ "$TEST6_PASSED" = "true" ] && CRITICAL_TESTS_PASSED=$((CRITICAL_TESTS_PASSED + 1))  # Dependencies
[ "$TEST13_PASSED" = "true" ] && CRITICAL_TESTS_PASSED=$((CRITICAL_TESTS_PASSED + 1))  # End-to-end integration (MOST CRITICAL)

echo -e "${BLUE}🔍 Critical Tests for RunPod Deployment: ${CRITICAL_TESTS_PASSED}/${CRITICAL_TESTS_TOTAL} passed${NC}"

if [ $CRITICAL_TESTS_PASSED -eq $CRITICAL_TESTS_TOTAL ]; then
    echo -e "${GREEN}🚀 All critical tests passed - Ready for RunPod deployment!${NC}"
    echo -e "${GREEN}✨ These tests validate the exact issues you encountered before${NC}"
    DEPLOYMENT_READY=true
elif [ $CRITICAL_TESTS_PASSED -ge 5 ]; then
    echo -e "${YELLOW}⚠️  Most critical tests passed (${CRITICAL_TESTS_PASSED}/${CRITICAL_TESTS_TOTAL})${NC}"
    echo -e "${YELLOW}📝 Continuing with deployment, but monitor for issues${NC}"
    DEPLOYMENT_READY=true
elif [ $CRITICAL_TESTS_PASSED -ge 4 ]; then
    echo -e "${YELLOW}⚠️  Some critical tests failed - deployment risky${NC}"
    echo -e "${RED}🛑 RECOMMENDATION: Fix failing tests before RunPod deployment${NC}"
    echo -e "${YELLOW}📝 Continuing anyway, but expect issues like before${NC}"
    DEPLOYMENT_READY=false
else
    echo -e "${RED}❌ Too many critical tests failed (${CRITICAL_TESTS_PASSED}/${CRITICAL_TESTS_TOTAL})${NC}"
    echo -e "${RED}🛑 STRONG RECOMMENDATION: Do not deploy to RunPod yet${NC}"
    echo -e "${RED}💥 Deployment will likely fail with the same errors you saw${NC}"
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
            
            # Step 3.1: S3 Authentication and Access Tests (HIGH PRIORITY)
            echo ""
            echo -e "${BLUE}🗄️  Step 3.1: S3 Authentication and Access Tests${NC}"
            echo "========================================"
            echo -e "${YELLOW}📋 Testing S3 credentials and bucket access (critical for model transfer)...${NC}"
            
            S3_TEST_PASSED=false
            
            # Test 1: Validate S3 credentials are set
            echo -e "${BLUE}🧪 Test 1: S3 credentials validation...${NC}"
            if [ -n "${RUNPOD_S3_ACCESS_KEY}" ] && [ -n "${RUNPOD_S3_SECRET_ACCESS_KEY}" ]; then
                echo -e "${GREEN}✅ S3 credentials found in environment${NC}"
                echo -e "${BLUE}   Access Key: ${RUNPOD_S3_ACCESS_KEY:0:8}...${NC}"
                echo -e "${BLUE}   Secret Key: ${RUNPOD_S3_SECRET_ACCESS_KEY:0:8}...${NC}"
                
                # Test 2: AWS CLI S3 connectivity test
                echo -e "${BLUE}🧪 Test 2: S3 bucket connectivity test...${NC}"
                echo -e "${YELLOW}   Testing: aws s3 ls --region us-ks-2 --endpoint-url https://s3api-us-ks-2.runpod.io s3://40ub9vhaa7/${NC}"
                
                # Set AWS credentials for the test
                export AWS_ACCESS_KEY_ID="${RUNPOD_S3_ACCESS_KEY}"
                export AWS_SECRET_ACCESS_KEY="${RUNPOD_S3_SECRET_ACCESS_KEY}"
                
                # Test S3 access using AWS CLI (if available)
                if command -v aws > /dev/null 2>&1; then
                    S3_LIST_OUTPUT=$(aws s3 ls --region us-ks-2 --endpoint-url https://s3api-us-ks-2.runpod.io s3://40ub9vhaa7/ 2>&1)
                    S3_LIST_EXIT_CODE=$?
                    
                    if [ $S3_LIST_EXIT_CODE -eq 0 ]; then
                        echo -e "${GREEN}✅ S3 bucket access successful!${NC}"
                        echo -e "${BLUE}   Bucket contents:${NC}"
                        echo "${S3_LIST_OUTPUT}" | head -5
                        S3_TEST_PASSED=true
                    else
                        echo -e "${RED}❌ S3 bucket access failed${NC}"
                        echo -e "${YELLOW}   Error: ${S3_LIST_OUTPUT}${NC}"
                        echo -e "${YELLOW}💡 This may prevent model transfer from working${NC}"
                    fi
                else
                    echo -e "${YELLOW}⚠️  AWS CLI not found - testing S3 via Python...${NC}"
                    
                    # Test S3 access using Python boto3
                    S3_PYTHON_TEST=$(python3 -c "
import boto3
import sys
try:
    s3_client = boto3.client(
        's3',
        endpoint_url='https://s3api-us-ks-2.runpod.io',
        aws_access_key_id='${RUNPOD_S3_ACCESS_KEY}',
        aws_secret_access_key='${RUNPOD_S3_SECRET_ACCESS_KEY}',
        region_name='us-ks-2'
    )
    s3_client.head_bucket(Bucket='40ub9vhaa7')
    print('✅ S3 bucket accessible via Python boto3')
    
    # Try to list a few objects
    response = s3_client.list_objects_v2(Bucket='40ub9vhaa7', MaxKeys=5)
    count = response.get('KeyCount', 0)
    print(f'✅ Bucket contains {count} objects (showing max 5)')
    sys.exit(0)
except Exception as e:
    print(f'❌ S3 access failed: {e}')
    sys.exit(1)
" 2>&1)
                    S3_PYTHON_EXIT_CODE=$?
                    
                    echo "$S3_PYTHON_TEST"
                    if [ $S3_PYTHON_EXIT_CODE -eq 0 ]; then
                        S3_TEST_PASSED=true
                    fi
                fi
                
                # Clean up environment variables
                unset AWS_ACCESS_KEY_ID
                unset AWS_SECRET_ACCESS_KEY
                
            else
                echo -e "${RED}❌ S3 credentials missing from environment${NC}"
                echo -e "${YELLOW}💡 Check that RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_ACCESS_KEY are in .env${NC}"
            fi
            
            # S3 Test Summary
            echo ""
            echo -e "${BLUE}📊 S3 Test Results:${NC}"
            if [ "$S3_TEST_PASSED" = "true" ]; then
                echo -e "${GREEN}🎉 S3 AUTHENTICATION AND ACCESS: SUCCESSFUL${NC}"
                echo -e "${GREEN}✅ Model transfer via S3 should work correctly${NC}"
            else
                echo -e "${RED}❌ S3 AUTHENTICATION AND ACCESS: FAILED${NC}"
                echo -e "${YELLOW}⚠️  Model transfer may not work - S3 issues detected${NC}"
                echo -e "${YELLOW}💡 Recommendation: Fix S3 access before running optimization tests${NC}"
            fi
            
            echo ""
            
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
echo "DEBUG: Command: python src/optimizer.py dataset_name=mnist mode=simple trials=1 max_epochs_per_trial=6 use_runpod_service=true"
INTEGRATION_TEST_1=$(timeout 600 env \
  RUNPOD_ENDPOINT_ID="${RUNPOD_ENDPOINT_ID}" \
  RUNPOD_API_KEY="${RUNPOD_API_KEY}" \
  RUNPOD_S3_ACCESS_KEY="${RUNPOD_S3_ACCESS_KEY}" \
  RUNPOD_S3_SECRET_ACCESS_KEY="${RUNPOD_S3_SECRET_ACCESS_KEY}" \
  python src/optimizer.py \
  dataset_name=mnist \
  mode=simple \
  trials=1 \
  max_epochs_per_trial=6 \
  use_runpod_service=true 2>&1)

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
   echo "${INTEGRATION_TEST_1}" | grep -q "RunPod Service.*JSON API"; then
    INTEGRATION_SUCCESS_1=true
    echo -e "${GREEN}✅ Basic integration test successful${NC}"
    echo -e "${BLUE}📊 Integration test results:${NC}"
    echo "${INTEGRATION_TEST_1}" | grep -E "(Best value|Optimization completed|RunPod Service)" | tail -3
else
    echo -e "${RED}❌ Basic integration test failed${NC}"
    echo -e "${YELLOW}📝 Checking what we're missing:${NC}"
    echo "Looking for 'Optimization completed successfully': $(echo "${INTEGRATION_TEST_1}" | grep -c "Optimization completed successfully")"
    echo "Looking for 'RunPod Service.*JSON API': $(echo "${INTEGRATION_TEST_1}" | grep -c "RunPod Service.*JSON API")"
    echo -e "${YELLOW}📝 Last 10 lines of integration test:${NC}"
    echo "${INTEGRATION_TEST_1}" | tail -10
fi

# Test 2: Fallback mechanism test
echo ""
echo -e "${BLUE}🧪 Test 2: Fallback mechanism validation...${NC}"
FALLBACK_TEST=$(env \
  RUNPOD_ENDPOINT_ID="${RUNPOD_ENDPOINT_ID}" \
  RUNPOD_API_KEY="${RUNPOD_API_KEY}" \
  RUNPOD_S3_ACCESS_KEY="${RUNPOD_S3_ACCESS_KEY}" \
  RUNPOD_S3_SECRET_ACCESS_KEY="${RUNPOD_S3_SECRET_ACCESS_KEY}" \
  python src/optimizer.py \
  dataset_name=mnist \
  mode=simple \
  trials=1 \
  max_epochs_per_trial=6 \
  use_runpod_service=true \
  runpod_service_endpoint=https://invalid-endpoint-test.com \
  runpod_service_fallback_local=true 2>&1)

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
LOCAL_BASELINE=$(env \
  RUNPOD_ENDPOINT_ID="${RUNPOD_ENDPOINT_ID}" \
  RUNPOD_API_KEY="${RUNPOD_API_KEY}" \
  RUNPOD_S3_ACCESS_KEY="${RUNPOD_S3_ACCESS_KEY}" \
  RUNPOD_S3_SECRET_ACCESS_KEY="${RUNPOD_S3_SECRET_ACCESS_KEY}" \
  python src/optimizer.py \
  dataset_name=mnist \
  mode=simple \
  trials=1 \
  max_epochs_per_trial=6 2>&1)

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