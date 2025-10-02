#!/bin/bash
# Docker entrypoint script for backend - adapts to development or production mode

set -e

# Default to development if not specified
ENVIRONMENT=${ENVIRONMENT:-development}

echo "Starting backend in ${ENVIRONMENT} mode..."

if [ "$ENVIRONMENT" = "production" ]; then
    # Production: Use Gunicorn with single worker for now
    # NOTE: Multiple workers require shared state (Redis/database) for job tracking
    # TODO: Implement Redis-backed job storage to enable multiple workers
    WORKERS=${GUNICORN_WORKERS:-1}
    echo "Running with Gunicorn (${WORKERS} worker - single worker mode due to in-memory job storage)..."
    exec gunicorn api_server:app \
        --workers ${WORKERS} \
        --worker-class uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:8000 \
        --access-logfile - \
        --error-logfile - \
        --log-level info
else
    # Development: Use uvicorn with hot reload
    echo "Running with Uvicorn (development mode)..."
    exec uvicorn api_server:app \
        --host 0.0.0.0 \
        --port 8000 \
        --log-level info
fi
