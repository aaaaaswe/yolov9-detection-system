@echo off
echo ========================================
echo    YOLOv9 Web Application Launcher
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo [2/3] Checking Streamlit installation...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Streamlit not found. Installing...
    pip install streamlit
    if errorlevel 1 (
        echo Error: Failed to install Streamlit
        pause
        exit /b 1
    )
)

echo [3/3] Starting Streamlit application...
echo.
echo ========================================
echo    Application will open in your browser
echo    URL: http://localhost:8501
echo ========================================
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py

pause
