# YOLOv9 实时检测系统

基于YOLOv9的目标检测系统，支持自定义数据集训练、实时视频检测、批量图片处理等功能。

## 功能特性

- ✅ **完整的训练功能**: 支持从头训练和微调预训练模型
- ✅ **实时检测**: 支持摄像头实时检测
- ✅ **视频处理**: 支持视频文件检测和结果保存
- ✅ **批量处理**: 支持文件夹批量图片检测
- ✅ **数据集工具**: 完整的数据集准备和可视化工具
- ✅ **模型导出**: 支持导出为ONNX等格式
- ✅ **命令行工具**: 统一的命令行界面
- ✅ **交互模式**: 友好的交互式操作

## 项目结构

```
yolov9_detection/
├── data/                    # 数据集目录
│   └── custom_dataset/     # 自定义数据集
│       ├── images/          # 图像文件
│       │   ├── train/      # 训练集
│       │   ├── val/        # 验证集
│       │   └── test/       # 测试集
│       ├── labels/         # 标签文件
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── data.yaml       # 数据集配置
├── yolov9/                  # YOLOv9模型目录
├── models/                  # 保存的模型
├── weights/                 # 预训练权重
├── runs/                    # 训练和检测结果
│   ├── train/              # 训练结果
│   ├── detect/             # 检测结果
│   └── tensorboard/        # TensorBoard日志
├── config.yaml              # 配置文件
├── requirements.txt         # 依赖包
├── prepare_dataset.py       # 数据集准备脚本
├── train.py                 # 训练脚本
├── detect.py                # 检测脚本
├── yolov9_cli.py           # 统一命令行工具
└── README.md                # 本文档
```

## 安装

### 1. 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU加速，可选)

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "from ultralytics import YOLO; print('YOLO安装成功!')"
```

## 快速开始

### 方式1: 交互模式 (推荐初学者)

```bash
python yolov9_cli.py
```

然后按照提示选择操作：
1. 准备数据集
2. 训练模型
3. 检测目标
4. 导出模型

### 方式2: 命令行模式

#### 1. 准备数据集

创建YOLO格式的数据集结构：

```bash
python yolov9_cli.py prepare --mode create --dataset_path data/my_dataset --classes person car dog
```

划分数据集：

```bash
python prepare_dataset.py --mode split \
    --image_dir /path/to/images \
    --label_dir /path/to/labels \
    --dataset_path data/my_dataset
```

#### 2. 训练模型

使用默认配置训练：

```bash
python yolov9_cli.py train --data data/my_dataset/data.yaml --model_size y --epochs 100
```

指定参数训练：

```bash
python train.py --data data/my_dataset/data.yaml \
    --model_size y \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --optimizer auto \
    --device 0
```

恢复训练：

```bash
python train.py --mode resume --resume runs/train/exp/weights/last.pt
```

#### 3. 检测目标

摄像头实时检测：

```bash
python yolov9_cli.py detect --source 0 --weights yolov9y.pt
```

视频文件检测：

```bash
python detect.py --source video.mp4 --weights yolov9y.pt --output result.mp4 --save
```

图片检测：

```bash
python detect.py --source image.jpg --weights yolov9y.pt --output result.jpg
```

批量检测文件夹：

```bash
python detect.py --source /path/to/images --weights yolov9y.pt --output /path/to/results
```

#### 4. 导出模型

导出为ONNX格式：

```bash
python yolov9_cli.py export --weights runs/train/exp/weights/best.pt --format onnx
```

## 数据集准备

### YOLO格式数据集

数据集需要按照以下结构组织：

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    ├── val/
    └── test/
```

### 标注文件格式

每个图片对应的标签文件格式如下：

```
<class_id> <x_center> <y_center> <width> <height>
```

其中：
- `<class_id>`: 类别ID，从0开始
- `<x_center>`, `<y_center>`: 边界框中心坐标 (0-1，相对于图片宽高)
- `<width>`, `<height>`: 边界框宽高 (0-1，相对于图片宽高)

示例：

```
0 0.500000 0.500000 0.300000 0.400000
1 0.750000 0.250000 0.150000 0.200000
```

### data.yaml 配置文件

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 3  # 类别数量
names: ['person', 'car', 'dog']  # 类别名称
```

## 模型选择

YOLOv9 提供不同大小的模型：

| 模型 | 参数量 | 速度 | 精度 | 适用场景 |
|------|--------|------|------|----------|
| yolov9n | 2.7M | 最快 | 较低 | 边缘设备、实时应用 |
| yolov9s | 7.2M | 快 | 中等 | 平衡速度和精度 |
| yolov9m | 20.1M | 中等 | 较高 | 通用场景 |
| yolov9l | 55.4M | 慢 | 高 | 高精度要求 |
| yolov9x | 111.3M | 最慢 | 最高 | 竞赛、研究 |

## 训练技巧

### 1. 数据增强

适当的数据增强可以提高模型泛化能力：

```bash
python train.py --data data.yaml \
    --mosaic 1.0 \
    --mixup 0.1 \
    --hsv_h 0.015 \
    --hsv_s 0.7 \
    --hsv_v 0.4
```

### 2. 学习率调整

```bash
# 余弦退火学习率
python train.py --lr0 0.01 --lrf 0.01

# 使用学习率调度器
python train.py --optimizer SGD --lr0 0.01
```

### 3. 批次大小调整

根据GPU显存调整批次大小：

```bash
# 小批次
python train.py --batch 8

# 大批次 (需要更多显存)
python train.py --batch 32
```

### 4. 早停机制

防止过拟合：

```bash
python train.py --patience 50
```

## 检测参数调整

### 调整置信度阈值

```bash
# 较低阈值 (检测更多目标，但可能有误检)
python detect.py --weights best.pt --conf 0.15

# 较高阈值 (只检测高置信度目标)
python detect.py --weights best.pt --conf 0.50
```

### 调整IOU阈值

```bash
# 较低IOU (允许更多重叠框)
python detect.py --weights best.pt --iou 0.35

# 较高IOU (更严格的非极大值抑制)
python detect.py --weights best.pt --iou 0.60
```

## TensorBoard 可视化

启动TensorBoard查看训练过程：

```bash
tensorboard --logdir runs/tensorboard
```

然后在浏览器中打开: http://localhost:6006

## 常见问题

### 1. 内存不足

- 减小 `batch_size`
- 减小 `image_size`
- 使用更小的模型 (`--model_size n` 或 `s`)

### 2. 检测速度慢

- 使用更小的模型
- 减小 `image_size`
- 使用GPU (`--device 0`)

### 3. 精度不够

- 增加训练轮数
- 使用更大的模型
- 检查数据集质量和数量
- 尝试数据增强

### 4. CUDA错误

- 检查CUDA版本: `nvidia-smi`
- 检查PyTorch CUDA支持: `python -c "import torch; print(torch.cuda.is_available())"`
- 使用CPU: `--device cpu`

## 高级功能

### 1. 断点续训

训练中断后可以继续：

```bash
python train.py --resume runs/train/exp/weights/last.pt
```

### 2. 模型集成

使用多个模型进行预测投票：

```python
# 在代码中实现模型集成
model1 = YOLO('model1.pt')
model2 = YOLO('model2.pt')
results1 = model1(image)
results2 = model2(image)
# 对结果进行融合
```

### 3. 自定义数据增强

修改 `train.py` 中的数据增强参数：

```python
train_args = {
    'hsv_h': 0.015,      # HSV色调增强
    'hsv_s': 0.7,        # HSV饱和度增强
    'hsv_v': 0.4,        # HSV明度增强
    'degrees': 0.0,       # 旋转角度
    'translate': 0.1,    # 平移
    'scale': 0.5,        # 缩放
    'shear': 0.0,        # 剪切
    'perspective': 0.0,  # 透视变换
    'flipud': 0.0,       # 上下翻转
    'fliplr': 0.5,       # 左右翻转
}
```

## 性能基准

在COCO数据集上的性能参考：

| 模型 | mAP@50 | mAP@50-95 | 参数量 | FLOPs |
|------|--------|-----------|--------|-------|
| YOLOv9n | 62.2 | 46.8 | 2.7M | 5.2 |
| YOLOv9s | 66.6 | 50.7 | 7.2M | 16.5 |
| YOLOv9m | 70.8 | 54.6 | 20.1M | 39.1 |
| YOLOv9l | 73.6 | 57.0 | 55.4M | 80.9 |
| YOLOv9x | 75.3 | 58.7 | 111.3M | 165.8 |

## 许可证

本项目采用AGPL-3.0许可证。

## 常见问题 (FAQ)

### 1. Windows 上运行 Web 应用失败

#### 问题: `'streamlit' 不是内部或外部命令`

**原因**: Streamlit 未添加到系统 PATH

**解决方案**:

**方法 1 - 使用完整路径启动**:
```cmd
C:\Python\Python310\Scripts\streamlit.exe run web_app\app.py
```

**方法 2 - 添加到 PATH**:
1. 打开"环境变量"设置
2. 编辑"系统变量"中的 `Path`
3. 添加: `C:\Python\Python310\Scripts\`
4. 重新打开命令提示符

**方法 3 - 使用环境修复脚本**:
```cmd
fix_env.bat
```

#### 问题: `ModuleNotFoundError: No module named 'detect'`

**原因**: detect.py 模块导入失败

**解决方案**:

**方法 1 - 运行环境检查**:
```bash
python check_webapp.py
```

**方法 2 - 检查文件结构**:
确保项目结构如下：
```
yolov9-detection-system/
├── detect.py              # 必须存在
├── train.py
├── web_app/
│   └── app.py
└── ...
```

**方法 3 - 手动安装依赖**:
```bash
pip install ultralytics opencv-python numpy pillow streamlit
```

### 2. 模型加载失败

#### 问题: `ModuleNotFoundError: No module named 'ultralytics'`

**解决方案**:
```bash
pip install ultralytics
```

#### 问题: 模型下载慢

**解决方案**:
1. 手动下载模型: https://github.com/ultralytics/assets/releases
2. 放到项目根目录
3. 指定本地路径: `--weights yolov8s.pt`

### 3. CUDA 相关问题

#### 问题: CUDA out of memory

**解决方案**:
- 减小批次大小: `--batch 8`
- 使用更小的模型: `--model_size n`
- 清理 GPU 缓存
```python
import torch
torch.cuda.empty_cache()
```

#### 问题: CUDA not available

**检查 CUDA 是否安装**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果返回 `False`，需要：
1. 安装 CUDA Toolkit
2. 安装 PyTorch CUDA 版本
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Web 应用相关问题

#### 问题: 警告: `Thread 'MainThread': missing ScriptRunContext`

**原因**: 使用 `python app.py` 而不是 `streamlit run app.py`

**解决方案**:
```bash
# 错误 ❌
python web_app/app.py

# 正确 ✅
streamlit run web_app/app.py
```

#### 问题: 浏览器无法访问 `http://localhost:8501`

**检查清单**:
1. Streamlit 是否正在运行（查看终端）
2. 检查防火墙设置
3. 尝试使用 `http://127.0.0.1:8501`

#### 问题: 端口被占用

**解决方案** - 更换端口:
```bash
streamlit run web_app/app.py --server.port 8502
```

### 5. 训练问题

#### 问题: 训练中断

**解决方案** - 使用断点续训:
```bash
python train.py --mode resume --resume runs/train/exp/weights/last.pt
```

#### 问题: 训练速度慢

**解决方案**:
1. 使用 GPU: `--device 0`
2. 增加批次大小（如果显存足够）: `--batch 32`
3. 减小图像尺寸: `--imgsz 512`

### 6. GUI 相关问题

#### 问题: `ModuleNotFoundError: No module named 'PyQt6'`

**解决方案**:
```bash
pip install PyQt6
```

或运行:
```bash
start_gui.bat  # Windows
./start_gui.sh  # Linux/Mac
```

### 7. 依赖安装失败

#### 问题: pip install 速度慢

**解决方案** - 使用国内镜像:
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 问题: 版本冲突

**解决方案** - 使用虚拟环境:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 获取更多帮助

如果以上解决方案无法解决你的问题：

1. 查看详细日志
2. 检查 GitHub Issues: https://github.com/aaaaaswe/yolov9-detection-system/issues
3. 提交新的 Issue，并附上：
   - 操作系统和 Python 版本
   - 完整的错误信息
   - 复现步骤

## 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO实现
- [YOLOv9](https://github.com/WongKinYiu/yolov9) - YOLOv9原作者

## 联系方式

如有问题或建议，请提交Issue。

---

**Happy Detecting! 🚀**
