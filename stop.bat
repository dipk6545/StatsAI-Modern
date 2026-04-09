@echo off
color 0C
echo ==================================================
echo       Stopping PDF Summarizer Background Services
echo ==================================================
echo.

call pm2 stop all
echo.
echo [STATUS] Applications have been stopped.
echo ==================================================
pause
