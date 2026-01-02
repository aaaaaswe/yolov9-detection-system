#!/bin/bash

# YOLOv9 快速启动脚本

echo "=========================================="
echo "YOLOv9 快速启动"
echo "=========================================="
echo ""

# 检查Python环境
echo "1. 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python 3.8+"
    exit 1
fi
echo "   Python版本: $(python3 --version)"

# 检查依赖
echo ""
echo "2. 检查依赖包..."
if [ ! -f "requirements.txt" ]; then
    echo "错误: 未找到 requirements.txt"
    exit 1
fi

echo "   安装依赖包..."
pip3 install -r requirements.txt

# 检查CUDA (可选)
echo ""
echo "3. 检查CUDA支持..."
if command -v nvidia-smi &> /dev/null; then
    echo "   已检测到NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/   /'
else
    echo "   未检测到GPU，将使用CPU训练"
fi

# 创建必要的目录
echo ""
echo "4. 创建必要的目录..."
mkdir -p data/custom_dataset/images/{train,val,test}
mkdir -p data/custom_dataset/labels/{train,val,test}
mkdir -p runs/train
mkdir -p runs/detect
mkdir -p weights
mkdir -p models
echo "   目录创建完成"

# 检查预训练模型
echo ""
echo "5. 检查预训练模型..."
if [ -f "yolov9s.pt" ] || [ -f "yolov9y.pt" ]; then
    echo "   已找到预训练模型"
else
    echo "   首次运行将自动下载预训练模型..."
fi

# 完成提示
echo ""
echo "=========================================="
echo "环境准备完成!"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  1. 交互模式:"
echo "     python3 yolov9_cli.py"
echo ""
echo "  2. 准备数据集:"
echo "     python3 prepare_dataset.py --mode create --dataset_path data/custom_dataset --classes person car"
echo ""
echo "  3. 训练模型:"
echo "     python3 train.py --data data/custom_dataset/data.yaml --model_size s --epochs 100"
echo ""
echo "  4. 摄像头检测:"
echo "     python3 detect.py --source 0 --weights yolov9s.pt"
echo ""
echo "  5. 视频检测:"
echo "     python3 detect.py --source video.mp4 --weights yolov9s.pt --output result.mp4"
echo ""
echo "查看更多示例: python3 example_usage.py"
echo ""
echo "=========================================="
