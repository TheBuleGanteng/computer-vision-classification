#!/bin/sh
# Docker entrypoint script for frontend - adapts to development or production mode

set -e

# Default to development if not specified
ENVIRONMENT=${ENVIRONMENT:-development}

echo "Starting frontend in ${ENVIRONMENT} mode..."

if [ "$ENVIRONMENT" = "production" ]; then
    # Production: Build and run optimized production build
    echo "Building Next.js for production..."
    npm run build
    echo "Starting Next.js production server..."
    exec npm start
else
    # Development: Run development server with hot reload
    echo "Running Next.js development server..."
    exec npm run dev
fi
