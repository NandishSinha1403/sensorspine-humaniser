#!/bin/bash

# Humaniser Shutdown Script

stop_service() {
    PORT=$1
    NAME=$2
    PID=$(lsof -ti:$PORT 2>/dev/null)
    
    if [ -n "$PID" ]; then
        kill $PID 2>/dev/null
        echo "✓ $NAME stopped"
    else
        echo "— $NAME was not running"
    fi
}

echo "Stopping Humaniser services..."
stop_service 8000 "Backend"
stop_service 8001 "Trainer"
stop_service 3000 "Frontend"
echo "All services handled."
