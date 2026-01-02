#!/bin/bash

# Streamlit Web Application Launcher
# YOLOv9 Detection System

echo "========================================"
echo "   YOLOv9 Web Application Launcher"
echo "========================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check Python installation
echo "[1/4] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "   Python version: $(python3 --version)"
echo "   ✓ Python found"
echo ""

# Check Streamlit installation
echo "[2/4] Checking Streamlit installation..."
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "   Streamlit not found. Installing..."
    pip3 install streamlit
    if [ $? -ne 0 ]; then
        echo "   Error: Failed to install Streamlit"
        exit 1
    fi
fi

echo "   ✓ Streamlit found"
echo ""

# Install dependencies
echo "[3/4] Installing dependencies..."
if [ -f "requirements.txt" ]; then
    echo "   Installing web app dependencies..."
    pip3 install -q -r requirements.txt
    echo "   ✓ Web app dependencies installed"
fi

if [ -f "../requirements.txt" ]; then
    echo "   Installing main project dependencies..."
    pip3 install -q -r ../requirements.txt
    echo "   ✓ Main project dependencies installed"
fi

echo ""

# Create necessary directories
echo "   Creating necessary directories..."
mkdir -p uploads results temp logs
echo "   ✓ Directories created"
echo ""

# Start Streamlit application
echo "[4/4] Starting Streamlit application..."
echo ""
echo "========================================"
echo "   Application will open in your browser"
echo "   URL: http://localhost:8501"
echo "========================================"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
