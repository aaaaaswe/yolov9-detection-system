# YOLOv9 Real-Time Detection System

A comprehensive object detection system based on YOLOv9, supporting custom dataset training, real-time video detection, and batch image processing.

## Features

- ✅ **Complete Training Pipeline**: Support for training from scratch and fine-tuning pretrained models
- ✅ **Real-Time Detection**: Support for real-time camera detection
- ✅ **Video Processing**: Support for video file detection and result saving
- ✅ **Batch Processing**: Support for batch image detection in folders
- ✅ **Dataset Tools**: Complete dataset preparation and visualization tools
- ✅ **Model Export**: Support for exporting to ONNX and other formats
- ✅ **CLI Tool**: Unified command-line interface
- ✅ **Interactive Mode**: User-friendly interactive operations

## Project Structure

```
yolov9_detection/
├── data/                    # Dataset directory
│   └── custom_dataset/     # Custom dataset
│       ├── images/          # Image files
│       │   ├── train/      # Training set
│       │   ├── val/        # Validation set
│       │   └── test/       # Test set
│       ├── labels/         # Label files
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── data.yaml       # Dataset configuration
├── yolov9/                  # YOLOv9 model directory
├── models/                  # Saved models
├── weights/                 # Pretrained weights
├── runs/                    # Training and detection results
│   ├── train/              # Training results
│   ├── detect/             # Detection results
│   └── tensorboard/        # TensorBoard logs
├── config.yaml              # Configuration file
├── requirements.txt         # Dependencies
├── prepare_dataset.py       # Dataset preparation script
├── train.py                 # Training script
├── detect.py                # Detection script
├── yolov9_cli.py           # Unified CLI tool
└── README.md                # This document
```

## Installation

### 1. Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration, optional)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "from ultralytics import YOLO; print('YOLO installed successfully!')"
```

## Quick Start

### Method 1: Interactive Mode (Recommended for Beginners)

```bash
python yolov9_cli.py
```

Then follow the prompts to select an operation:
1. Prepare dataset
2. Train model
3. Detect objects
4. Export model

### Method 2: Command Line Mode

#### 1. Prepare Dataset

Create YOLO format dataset structure:

```bash
python yolov9_cli.py prepare --mode create --dataset_path data/my_dataset --classes person car dog
```

Split dataset:

```bash
python prepare_dataset.py --mode split \
    --image_dir /path/to/images \
    --label_dir /path/to/labels \
    --dataset_path data/my_dataset
```

#### 2. Train Model

Train with default configuration:

```bash
python yolov9_cli.py train --data data/my_dataset/data.yaml --model_size y --epochs 100
```

Train with custom parameters:

```bash
python train.py --data data/my_dataset/data.yaml \
    --model_size y \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --optimizer auto \
    --device 0
```

Resume training:

```bash
python train.py --mode resume --resume runs/train/exp/weights/last.pt
```

#### 3. Detect Objects

Real-time camera detection:

```bash
python yolov9_cli.py detect --source 0 --weights yolov9y.pt
```

Video file detection:

```bash
python detect.py --source video.mp4 --weights yolov9y.pt --output result.mp4 --save
```

Image detection:

```bash
python detect.py --source image.jpg --weights yolov9y.pt --output result.jpg
```

Batch detect folder:

```bash
python detect.py --source /path/to/images --weights yolov9y.pt --output /path/to/results
```

#### 4. Export Model

Export to ONNX format:

```bash
python yolov9_cli.py export --weights runs/train/exp/weights/best.pt --format onnx
```

## Dataset Preparation

### YOLO Format Dataset

The dataset should be organized in the following structure:

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

### Label File Format

Each image should have a corresponding label file in the following format:

```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `<class_id>`: Class ID, starting from 0
- `<x_center>`, `<y_center>`: Bounding box center coordinates (0-1, relative to image width/height)
- `<width>`, `<height>`: Bounding box width and height (0-1, relative to image width/height)

Example:

```
0 0.500000 0.500000 0.300000 0.400000
1 0.750000 0.250000 0.150000 0.200000
```

## Web Application

A web-based interface is available for easier usage:

### Start Web App

```bash
cd web_app
streamlit run app.py
```

### Features

- **Image Detection**: Upload images for object detection
- **Video Detection**: Process video files and download results
- **Batch Processing**: Upload multiple images for batch detection
- **Model Selection**: Choose different model sizes (n/s/m/l/x)
- **Resource Download**: Download `.gitignore`, `requirements.txt`, and other project files

### Access Resources Page

Navigate to the "📦 Project Resources" section in the sidebar to view and download project configuration files.

## Configuration

Edit `config.yaml` to customize detection and training parameters:

```yaml
# Training parameters
model_size: y  # Model size: n/s/m/l/x
epochs: 100
batch_size: 16
image_size: 640

# Detection parameters
conf_thres: 0.25
iou_thres: 0.45
max_det: 300

# Dataset parameters
data_path: data/custom_dataset
```

## Model Performance

| Model Size | mAP50-95 | mAP50 | Speed | Parameters |
|------------|----------|-------|-------|------------|
| YOLOv9-n   | 37.8     | 53.6  | 1.2ms | 2.5M       |
| YOLOv9-s   | 46.1     | 63.3  | 1.8ms | 8.3M       |
| YOLOv9-m   | 52.4     | 69.1  | 3.2ms | 20.2M      |
| YOLOv9-l   | 55.3     | 72.1  | 5.8ms | 39.8M      |
| YOLOv9-x   | 56.9     | 73.5  | 9.5ms | 56.6M      |

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config.yaml
   - Use a smaller model size

2. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

3. **Dataset Loading Errors**
   - Verify dataset structure matches YOLO format
   - Check label file paths and formats

4. **Detection Slow Performance**
   - Use GPU for acceleration: `--device 0`
   - Try a smaller model size

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO implementation
- [PyTorch](https://pytorch.org/) - Deep learning framework

## Contact

For questions and support, please open an issue on GitHub.
