#!/bin/bash

# YOLOv9 GUI 启动脚本

cd "$(dirname "$0")"

echo "启动 YOLOv9 图形化界面..."

# 检查是否安装了 PyQt6
if ! python3 -c "import PyQt6" 2>/dev/null; then
    echo "正在安装 PyQt6..."
    pip install PyQt6
fi

# 启动 GUI
python3 yolov9_gui.py
