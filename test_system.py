#!/usr/bin/env python3
"""
测试脚本
验证各个模块的功能
"""

import os
import sys
from pathlib import Path


def test_imports():
    """测试导入"""
    print("测试 1: 导入模块...")
    try:
        import torch
        print(f"  ✓ PyTorch版本: {torch.__version__}")
        print(f"  ✓ CUDA可用: {torch.cuda.is_available()}")
        
        import cv2
        print(f"  ✓ OpenCV版本: {cv2.__version__}")
        
        from ultralytics import YOLO
        print(f"  ✓ Ultralytics YOLO导入成功")
        
        print("  ✓ 所有依赖包导入成功!")
        return True
    except ImportError as e:
        print(f"  ✗ 导入失败: {e}")
        return False


def test_file_structure():
    """测试文件结构"""
    print("\n测试 2: 文件结构...")
    
    required_files = [
        'requirements.txt',
        'prepare_dataset.py',
        'train.py',
        'detect.py',
        'yolov9_cli.py',
        'example_usage.py',
        'config.yaml',
        'README.md',
    ]
    
    required_dirs = [
        'data',
        'yolov9',
        'models',
        'weights',
        'runs',
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} 不存在")
            all_exist = False
    
    for dir in required_dirs:
        if os.path.exists(dir):
            print(f"  ✓ {dir}/")
        else:
            print(f"  ✗ {dir}/ 不存在")
            all_exist = False
    
    if all_exist:
        print("  ✓ 文件结构完整!")
    else:
        print("  ✗ 部分文件或目录缺失")
    
    return all_exist


def test_dataset_preparation():
    """测试数据集准备"""
    print("\n测试 3: 数据集准备...")
    
    try:
        from prepare_dataset import create_yolo_dataset_structure
        
        # 创建测试数据集
        test_dataset = 'data/test_dataset'
        classes = ['test_class_1', 'test_class_2']
        
        create_yolo_dataset_structure(test_dataset, classes)
        
        # 验证文件是否创建
        if os.path.exists(f"{test_dataset}/data.yaml"):
            print("  ✓ 数据集结构创建成功")
            
            # 读取并验证配置
            import yaml
            with open(f"{test_dataset}/data.yaml", 'r') as f:
                config = yaml.safe_load(f)
            
            if config['nc'] == len(classes):
                print("  ✓ 配置文件正确")
            else:
                print("  ✗ 配置文件不正确")
                return False
            
            # 清理测试数据集
            import shutil
            shutil.rmtree(test_dataset)
            print("  ✓ 测试数据集已清理")
            
            return True
        else:
            print("  ✗ 数据集结构创建失败")
            return False
            
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False


def test_model_loading():
    """测试模型加载"""
    print("\n测试 4: 模型加载...")
    
    try:
        from ultralytics import YOLO
        
        # 测试加载官方模型
        print("  加载YOLOv9n模型...")
        model = YOLO('yolov9n.pt')
        
        print(f"  ✓ 模型加载成功")
        print(f"  ✓ 类别数: {len(model.names)}")
        print(f"  ✓ 类别: {list(model.names.values())[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        return False


def test_detection():
    """测试检测功能"""
    print("\n测试 5: 检测功能...")
    
    try:
        from detect import YOLOv9Detector
        import cv2
        import numpy as np
        
        # 创建测试图像
        print("  创建测试图像...")
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        cv2.imwrite('test_image.jpg', test_image)
        
        # 初始化检测器
        print("  初始化检测器...")
        detector = YOLOv9Detector(
            weights='yolov9n.pt',
            conf=0.5,
            device='cpu'  # 使用CPU测试
        )
        
        # 检测
        print("  执行检测...")
        results = detector.detect_image(
            image_path='test_image.jpg',
            output_path='test_result.jpg',
            show=False
        )
        
        if os.path.exists('test_result.jpg'):
            print("  ✓ 检测成功")
            print("  ✓ 结果图像已保存")
            
            # 清理测试文件
            os.remove('test_image.jpg')
            os.remove('test_result.jpg')
            print("  ✓ 测试文件已清理")
            
            return True
        else:
            print("  ✗ 检测失败")
            return False
            
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_init():
    """测试训练初始化"""
    print("\n测试 6: 训练初始化...")
    
    try:
        from train import YOLOv9Trainer
        
        # 创建测试数据集
        from prepare_dataset import create_yolo_dataset_structure
        test_dataset = 'data/train_test_dataset'
        create_yolo_dataset_structure(test_dataset, ['test'])
        
        # 初始化训练器
        print("  初始化训练器...")
        trainer = YOLOv9Trainer(
            model_size='n',
            data_yaml=f'{test_dataset}/data.yaml',
            device='cpu'
        )
        
        print("  ✓ 训练器初始化成功")
        
        # 加载模型
        print("  加载模型...")
        trainer.load_model('yolov9n.pt')
        print("  ✓ 模型加载成功")
        
        # 清理测试数据集
        import shutil
        shutil.rmtree(test_dataset)
        print("  ✓ 测试数据集已清理")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 80)
    print("YOLOv9 功能测试")
    print("=" * 80)
    
    results = []
    
    # 运行测试
    results.append(('导入模块', test_imports()))
    results.append(('文件结构', test_file_structure()))
    results.append(('数据集准备', test_dataset_preparation()))
    results.append(('模型加载', test_model_loading()))
    results.append(('检测功能', test_detection()))
    results.append(('训练初始化', test_training_init()))
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过! 系统准备就绪。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
