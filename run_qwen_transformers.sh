#!/bin/bash

# Set error handling
set -e

# Default configuration
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
HTTP_PORT=8080

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Pulsing Inference System with Transformers Backend${NC}"
echo -e "${BLUE}Model: ${MODEL_NAME}${NC}"

# Function to cleanup background processes on exit
cleanup() {
    echo -e "\n${BLUE}Shutting down services...${NC}"
    kill $(jobs -p) 2>/dev/null || true
    echo -e "${GREEN}Shutdown complete.${NC}"
}
trap cleanup EXIT

# 1. Start the Frontend (Router + HTTP Server)
echo -e "${GREEN}Starting Frontend...${NC}"
pulsing frontend \
    --model_name "$MODEL_NAME" \
    --http_port $HTTP_PORT \
    --router_mode round-robin \
    -D runtime.store_kv=file \
    -D runtime.request_plane=http &

FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait for frontend to initialize
sleep 2

# 2. Start the Transformers Backend Worker
echo -e "${GREEN}Starting Transformers Backend Worker...${NC}"
pulsing transformers "$MODEL_NAME" \
    -D backend.transformers.device=mps \
    -D backend.transformers.served_model_name="$MODEL_NAME" \
    -D backend.transformers.component=backend \
    -D backend.transformers.endpoint=generate \
    -D runtime.store_kv=file \
    -D runtime.request_plane=http &

WORKER_PID=$!
echo "Transformers Worker PID: $WORKER_PID"

echo -e "${GREEN}System is running!${NC}"
echo -e "You can send requests to: http://localhost:${HTTP_PORT}/v1/chat/completions"
echo -e "Logs are streaming below. Press Ctrl+C to stop."

# Wait for all background processes
wait
