@echo off
setlocal
color 0B
title StatsAI Analyst Hub - Unified Master

echo ==================================================
echo       StatsAI Analyst: Unified Command Center
echo ==================================================
echo.

:: 1. Cleanup
echo [1/3] Terminating stale Python processes...
taskkill /f /im python.exe /t >nul 2>&1

:: 2. Boot Unified Server
echo [2/3] Initializing Master Hub (UI + Cognitive Routing)...
:: Launching from the server directory to ensure .env is correctly loaded
start "StatsAI-Master" /min cmd /c "cd server && python main.py"

:: 3. Ready-state wait
echo Waiting for server to initialize...
ping 127.0.0.1 -n 8 >nul

:: 4. Launch Workspace
echo [3/3] Opening your Workspace at http://localhost:3001
start http://localhost:3001

echo.
echo ==================================================
echo [SUCCESS] Your StatsAI environment is now LIVE!
echo [ HELP  ] Use http://localhost:3001 to access the UI.
echo ==================================================
echo.

pause
