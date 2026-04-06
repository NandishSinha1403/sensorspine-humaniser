@echo off
setlocal enabledelayedexpansion

echo Stopping Humaniser services...

:: Stop Backend (Port 8000)
echo Finding Backend on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    echo Stopping Backend (PID: %%a)...
    taskkill /f /pid %%a >nul 2>&1
    set FOUND_BACKEND=1
)

:: Stop Frontend (Port 3000)
echo Finding Frontend on port 3000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3000 ^| findstr LISTENING') do (
    echo Stopping Frontend (PID: %%a)...
    taskkill /f /pid %%a >nul 2>&1
    set FOUND_FRONTEND=1
)

:: Stop Trainer if any (Port 8001 as per stop.sh)
echo Finding Trainer on port 8001...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8001 ^| findstr LISTENING') do (
    echo Stopping Trainer (PID: %%a)...
    taskkill /f /pid %%a >nul 2>&1
    set FOUND_TRAINER=1
)

if "%FOUND_BACKEND%"=="" echo - Backend was not running.
if "%FOUND_FRONTEND%"=="" echo - Frontend was not running.
if "%FOUND_TRAINER%"=="" echo - Trainer was not running.

echo All services handled.
pause
