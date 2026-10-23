#!/bin/bash

# Humaniser Shutdown Script (v2.2)

stop_service() {
    PORT=$1
    NAME=$2
    PID=$(lsof -ti:$PORT 2>/dev/null)
    
    if [ -n "$PID" ]; then
        kill -9 $PID 2>/dev/null
        echo "✓ $NAME (Port $PORT) stopped"
    else
        echo "— $NAME (Port $PORT) was not running"
    fi
}

echo "Stopping Humaniser services..."
stop_service 8000 "Backend"
stop_service 3000 "Frontend"
echo "All services handled."
