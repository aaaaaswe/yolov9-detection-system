@echo off
cd /d "%~dp0"

echo Starting YOLOv9 GUI...

REM Check if PyQt6 is installed
python -c "import PyQt6" 2>nul
if errorlevel 1 (
    echo Installing PyQt6...
    pip install PyQt6
)

REM Start GUI
python yolov9_gui.py
