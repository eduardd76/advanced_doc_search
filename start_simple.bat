@echo off
echo ====================================================
echo Advanced Document Search System - Simple Startup
echo ====================================================

echo.
echo [1] Starting Backend Server...
echo.

cd /d "F:\Agentic_Apps\advanced-document-search\backend"
start cmd /k "python main.py"

timeout /t 5 /nobreak > nul

echo.
echo [2] Backend should be starting on http://localhost:8002
echo.
echo [3] To use the system:
echo     - Backend API: http://localhost:8002
echo     - API Documentation: http://localhost:8002/docs
echo.
echo [4] To install missing dependencies:
echo     pip install pandas numpy scikit-learn
echo.
echo [5] To start frontend (in a new terminal):
echo     cd frontend
echo     npm install (first time only)
echo     npm start
echo.
echo ====================================================
echo Press any key to exit...
pause > nul