#!/bin/bash

# Humaniser Production-Ready Startup Script

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}=======================================${NC}"
echo -e "${PURPLE}   Humaniser Application Suite         ${NC}"
echo -e "${PURPLE}=======================================${NC}"

# Load environment variables if .env exists
if [ -f "humaniser/.env" ]; then
    export $(cat humaniser/.env | xargs)
fi

# Set defaults if not provided
BACKEND_PORT=${BACKEND_PORT:-8000}
TRAINER_PORT=${TRAINER_PORT:-8001}
FRONTEND_PORT=${FRONTEND_PORT:-3000}

# Function to handle script termination
cleanup() {
    echo ""
    echo -e "${PURPLE}🛑 Shutting down all Humaniser services...${NC}"
    # Kill the processes by port to be precise
    lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null
    lsof -ti:$TRAINER_PORT | xargs kill -9 2>/dev/null
    lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Start Backend API
echo -e "${BLUE}[API]${NC} Starting Humaniser API on http://localhost:$BACKEND_PORT..."
(
    cd humaniser/backend
    python3 -m uvicorn app.main:app --host 0.0.0.0 --port $BACKEND_PORT > /dev/null 2>&1
) &

# Start Trainer Server
echo -e "${BLUE}[TRAINER]${NC} Starting Training Server on http://localhost:$TRAINER_PORT..."
(
    cd humaniser/backend
    python3 -m uvicorn app.trainer:app --host 0.0.0.0 --port $TRAINER_PORT > /dev/null 2>&1
) &

# Wait for backends to be ready
echo -e "${GREEN}[INFO]${NC} Waiting for services to initialize..."
sleep 5

# Start Frontend
echo -e "${GREEN}[FRONTEND]${NC} Starting Dashboard on http://localhost:$FRONTEND_PORT..."
(
    cd humaniser/frontend
    npm run dev -- -p $FRONTEND_PORT > /dev/null 2>&1
) &

echo -e "${PURPLE}=======================================${NC}"
echo -e "${GREEN}✅ All services are now running!${NC}"
echo -e "   - Dashboard: http://localhost:$FRONTEND_PORT"
echo -e "   - Trainer UI: http://localhost:$TRAINER_PORT"
echo -e "   - API Health: http://localhost:$BACKEND_PORT/health"
echo -e "${PURPLE}=======================================${NC}"
echo "Press Ctrl+C to stop all services."

# Keep script alive
wait
