#!/bin/bash

# Set error handling
set -e

# Default configuration
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"  # Using Qwen2.5 0.5B as Qwen3 0.5B might not be public/exact yet, adjust as needed
MODEL_PATH="/tmp/model_cache/qwen0.5b"
HTTP_PORT=8080

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Pulsing Inference System with Transformers Backend${NC}"
echo -e "${BLUE}Model: ${MODEL_NAME}${NC}"

# Create model directory if it doesn't exist
mkdir -p "$MODEL_PATH"

# Function to cleanup background processes on exit
cleanup() {
    echo -e "\n${BLUE}Shutting down services...${NC}"
    kill $(jobs -p) 2>/dev/null || true
    echo -e "${GREEN}Shutdown complete.${NC}"
}
trap cleanup EXIT

# 1. Start the Frontend (Router + HTTP Server)
# We use store_kv="file" and request_plane="http" as per default for lightweight setup
echo -e "${GREEN}Starting Frontend...${NC}"
python -m pulsing.cli frontend \
    --model-name "$MODEL_NAME" \
    --namespace dynamo \
    --http-port $HTTP_PORT \
    --store-kv file \
    --request-plane http \
    --router-mode kv &

FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait a bit for frontend to initialize (optional, but good practice)
sleep 2

# 2. Start the Transformers Backend Worker
echo -e "${GREEN}Starting Transformers Backend Worker...${NC}"
# Use 'mps' for Mac GPU (Apple Silicon)
python -m pulsing.cli transformers \
    --model "$MODEL_NAME" \
    --served-model-name "$MODEL_NAME" \
    --namespace dynamo \
    --device mps \
    --store-kv file \
    --request-plane http \
    --component backend \
    --endpoint generate &

WORKER_PID=$!
echo "Transformers Worker PID: $WORKER_PID"

echo -e "${GREEN}System is running!${NC}"
echo -e "You can send requests to: http://localhost:${HTTP_PORT}/v1/chat/completions"
echo -e "Logs are streaming below. Press Ctrl+C to stop."

# Wait for all background processes
wait

