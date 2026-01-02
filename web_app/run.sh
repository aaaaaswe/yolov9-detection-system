#!/bin/bash

# Streamlit Web应用启动脚本

echo "=========================================="
echo "YOLOv9 Web应用 - Streamlit"
echo "=========================================="
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python 3.8+"
    exit 1
fi

echo "1. 检查依赖包..."

# 安装依赖
if [ -f "requirements.txt" ]; then
    echo "安装Web应用依赖..."
    pip3 install -r requirements.txt
    
    # 检查父目录的依赖
    if [ -f "../requirements.txt" ]; then
        echo "安装主项目依赖..."
        pip3 install -r ../requirements.txt
    fi
else
    echo "错误: 未找到 requirements.txt"
    exit 1
fi

# 创建必要的目录
echo ""
echo "2. 创建必要的目录..."
mkdir -p uploads results temp

echo "   目录创建完成"

echo ""
echo "=========================================="
echo "启动Streamlit应用..."
echo "=========================================="
echo ""
echo "应用将在浏览器中打开: http://localhost:8501"
echo "按 Ctrl+C 停止应用"
echo ""

# 启动Streamlit应用
streamlit run app.py
