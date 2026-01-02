#!/usr/bin/env python3
"""
Web 应用环境检查和修复脚本
用于诊断和修复常见的导入问题
"""

import sys
import os
from pathlib import Path

print("=" * 80)
print("YOLOv9 Web 应用环境检查")
print("=" * 80)
print()

# 获取项目根目录
script_dir = Path(__file__).parent
web_app_dir = script_dir / "web_app"
project_root = script_dir

print(f"当前目录: {os.getcwd()}")
print(f"脚本目录: {script_dir}")
print(f"Web应用目录: {web_app_dir}")
print(f"项目根目录: {project_root}")
print()

# 检查 1: Python 版本
print("[检查 1/6] Python 版本")
print(f"Python 版本: {sys.version}")
if sys.version_info < (3, 8):
    print("❌ 错误: 需要 Python 3.8 或更高版本")
else:
    print("✓ Python 版本符合要求")
print()

# 检查 2: 必要的文件是否存在
print("[检查 2/6] 检查必要文件")
required_files = [
    "detect.py",
    "train.py",
    "web_app/app.py",
    "requirements.txt"
]

for file in required_files:
    file_path = project_root / file
    if file_path.exists():
        print(f"✓ {file}")
    else:
        print(f"❌ {file} - 文件不存在!")
print()

# 检查 3: 检测 detect.py 是否可导入
print("[检查 3/6] 检测 detect.py 模块")
sys.path.insert(0, str(project_root))
try:
    import detect
    print("✓ detect.py 模块导入成功")

    # 检查 YOLOv9Detector 类
    if hasattr(detect, 'YOLOv9Detector'):
        print("✓ YOLOv9Detector 类存在")
    else:
        print("❌ YOLOv9Detector 类不存在")

except ImportError as e:
    print(f"❌ 无法导入 detect.py: {e}")
except Exception as e:
    print(f"❌ 导入时发生错误: {e}")
print()

# 检查 4: 检测 Ultralytics
print("[检查 4/6] 检测 Ultralytics")
try:
    import ultralytics
    from ultralytics import YOLO
    print(f"✓ Ultralytics 已安装 (版本: {ultralytics.__version__})")

    # 测试加载模型
    print("  测试加载默认模型...")
    try:
        model = YOLO('yolov8n.pt')
        print("  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
except ImportError:
    print("❌ Ultralytics 未安装")
    print("  请运行: pip install ultralytics")
print()

# 检查 5: 检测 Streamlit
print("[检查 5/6] 检测 Streamlit")
try:
    import streamlit
    print(f"✓ Streamlit 已安装 (版本: {streamlit.__version__})")
except ImportError:
    print("❌ Streamlit 未安装")
    print("  请运行: pip install streamlit")
print()

# 检查 6: 检测其他依赖
print("[检查 6/6] 检测其他依赖")
dependencies = [
    ('cv2', 'opencv-python'),
    ('numpy', 'numpy'),
    ('PIL', 'Pillow'),
]

missing_deps = []
for module, package in dependencies:
    try:
        __import__(module)
        print(f"✓ {package}")
    except ImportError:
        print(f"❌ {package} - 未安装")
        missing_deps.append(package)

if missing_deps:
    print()
    print("安装缺失的依赖:")
    print(f"pip install {' '.join(missing_deps)}")
print()

# 检查 Web 应用启动
print("=" * 80)
print("启动建议")
print("=" * 80)
print()

if "detect" not in sys.modules:
    print("⚠️  警告: detect.py 无法导入")
    print()
    print("这可能是因为:")
    print("1. detect.py 文件不存在")
    print("2. detect.py 有语法错误")
    print("3. detect.py 依赖的模块未安装")
    print()
    print("Web 应用会自动使用简化的检测器，功能仍然可用。")
    print()
else:
    print("✓ 所有检查通过!")
    print()

print("启动 Web 应用:")
print("  Windows: cd web_app && streamlit run app.py")
print("  Linux/Mac: cd web_app && streamlit run app.py")
print("  或使用: web_app/run.bat (Windows) / web_app/run.sh (Linux/Mac)")
print()

print("=" * 80)
