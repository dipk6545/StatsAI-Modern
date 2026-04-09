@echo off
setlocal
color 0B

:: --- CONFIGURATION VARIABLES ---
set APP_TITLE=StatsAI Analyst Hub - Dual Engine
set APP_HEADER=StatsAI Analyst: Production Laboratory
set BACKEND_PORT=3001
set FRONTEND_PORT=8080
set BACKEND_SCRIPT=server/main.py
set FRONTEND_SCRIPT=nicegui_app/main.py
:: -------------------------------

title %APP_TITLE%

echo =========================================================
echo       %APP_HEADER%
echo =========================================================
echo.

:: 1. Cleanup
echo [1/4] Terminating stale Python processes...
taskkill /f /im python.exe /t >nul 2>&1

:: 2. Boot Backend Headless API
echo [2/4] Initializing Headless Cognitive API (Port %BACKEND_PORT%)...
start "StatsAI-Backend" /min cmd /k "python %BACKEND_SCRIPT%"

:: 3. Boot Frontend Client UI
echo [3/4] Initializing NiceGUI Client Interface (Port %FRONTEND_PORT%)...
start "StatsAI-Frontend" /min cmd /k "python %FRONTEND_SCRIPT%"

:: 4. Ready-state wait
echo [4/4] Waiting for engines to sync...
ping 127.0.0.1 -n 6 >nul

:: Launch Web Browser
start http://localhost:%FRONTEND_PORT%

echo.
echo =========================================================
echo [SUCCESS] Your StatsAI Dual-Engine Pipeline is LIVE!
echo [ BACKEND ] Running actively on Port %BACKEND_PORT%.
echo [ FRONTEND] Running actively on Port %FRONTEND_PORT%.
echo =========================================================
echo.

pause
