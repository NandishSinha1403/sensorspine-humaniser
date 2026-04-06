@echo off
setlocal

echo =======================================
echo    Humaniser Application Suite         
echo =======================================

:: Load environment variables if .env exists
if exist "humaniser\.env" (
    for /f "tokens=*" %%a in (humaniser\.env) do set %%a
)

:: Set defaults if not provided
if "%BACKEND_PORT%"=="" set BACKEND_PORT=8000
if "%FRONTEND_PORT%"=="" set FRONTEND_PORT=3000

:: Start Backend API
echo [API] Starting Humaniser API on http://localhost:%BACKEND_PORT%...
start /b cmd /c "cd humaniser\backend && python -m uvicorn app.main:app --host 0.0.0.0 --port %BACKEND_PORT% > nul 2>&1"

:: Wait for backends to be ready
echo [INFO] Waiting for services to initialize...
timeout /t 3 /nobreak > nul

:: Start Frontend
echo [FRONTEND] Starting Dashboard on http://localhost:%FRONTEND_PORT%...
start /b cmd /c "cd humaniser\frontend && npm run dev -- -p %FRONTEND_PORT% > nul 2>&1"

echo =======================================
echo ✅ All services are now running!
echo    - Dashboard: http://localhost:%FRONTEND_PORT%
echo    - API Health: http://localhost:%BACKEND_PORT%/health
echo =======================================
echo Press Ctrl+C to stop this script.
echo NOTE: You may need to run stop.bat to fully clear processes.

:: Keep script alive
:loop
timeout /t 10 /nobreak > nul
goto loop
