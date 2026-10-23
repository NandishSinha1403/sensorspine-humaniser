#!/bin/bash

# Humaniser Production-Ready Startup Script (v2.2)
# Optimized for robust process management and logging.

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}=======================================${NC}"
echo -e "${PURPLE}   Humaniser Application Suite         ${NC}"
echo -e "${PURPLE}=======================================${NC}"

# 1. Load Environment Variables Safely
if [ -f "humaniser/.env" ]; then
    echo -e "${BLUE}[CONFIG]${NC} Loading environment from humaniser/.env..."
    set -a
    source humaniser/.env
    set +a
else
    echo -e "${BLUE}[CONFIG]${NC} No .env file found, using defaults."
fi

# Set defaults if not provided
BACKEND_PORT=${BACKEND_PORT:-8000}
FRONTEND_PORT=${FRONTEND_PORT:-3000}

# Define Log Files
BACKEND_LOG="humaniser/backend.log"
FRONTEND_LOG="humaniser/frontend.log"

# Function to handle script termination
cleanup() {
    echo ""
    echo -e "${PURPLE}🛑 Shutting down all Humaniser services...${NC}"
    # Kill background jobs started by this script
    jobs -p | xargs kill -9 2>/dev/null
    # Precise cleanup by port as a fallback
    lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null
    lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# 2. Start Backend API
echo -e "${BLUE}[API]${NC} Starting Humaniser API on http://localhost:$BACKEND_PORT..."
echo -e "${BLUE}[API]${NC} Logs: $BACKEND_LOG"
(
    cd humaniser/backend
    # Ensure dependencies and data are ready
    python3 -m uvicorn app.main:app --host 0.0.0.0 --port $BACKEND_PORT > "../../$BACKEND_LOG" 2>&1
) &

# 3. Wait for Backend Health Check
echo -e "${GREEN}[INFO]${NC} Waiting for API to initialize..."
MAX_RETRIES=15
RETRY_COUNT=0
HEALTH_URL="http://localhost:$BACKEND_PORT/health"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s $HEALTH_URL | grep -q '"status":"ok"'; then
        echo -e "${GREEN}[INFO]${NC} API is healthy!"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT+1))
    echo -n "."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "\n${RED}❌ Error: API failed to start within 30 seconds.${NC}"
    echo -e "${RED}Last 15 lines of $BACKEND_LOG:${NC}"
    tail -n 15 "$BACKEND_LOG"
    cleanup
fi

# 4. Start Frontend
echo -e "${GREEN}[FRONTEND]${NC} Starting Dashboard on http://localhost:$FRONTEND_PORT..."
echo -e "${GREEN}[FRONTEND]${NC} Logs: $FRONTEND_LOG"
(
    cd humaniser/frontend
    npm run dev -- -p $FRONTEND_PORT > "../../$FRONTEND_LOG" 2>&1
) &

echo -e "${PURPLE}=======================================${NC}"
echo -e "${GREEN}✅ All services are now running!${NC}"
echo -e "   - Dashboard: http://localhost:$FRONTEND_PORT"
echo -e "   - API Health: $HEALTH_URL"
echo -e "${PURPLE}=======================================${NC}"
echo "Press Ctrl+C to stop all services."

# Keep script alive
wait
