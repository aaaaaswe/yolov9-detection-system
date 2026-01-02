#!/usr/bin/env python3
"""
数据集准备脚本
用于准备YOLO格式的数据集
"""

import os
import shutil
import argparse
import random
from pathlib import Path
import yaml
from tqdm import tqdm


def create_yolo_dataset_structure(dataset_path, classes):
    """
    创建YOLO格式的数据集目录结构
    
    Args:
        dataset_path: 数据集根目录
        classes: 类别列表
    """
    print(f"创建YOLO数据集结构在: {dataset_path}")
    
    # 创建必要的目录
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dataset_path, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, 'labels', split), exist_ok=True)
    
    # 创建data.yaml配置文件
    data_yaml = {
        'path': os.path.abspath(dataset_path),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    
    print(f"数据集配置文件已创建: {yaml_path}")
    return yaml_path


def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    将数据集划分为训练集、验证集和测试集
    
    Args:
        image_dir: 原始图像目录
        label_dir: 原始标签目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    print(f"划分数据集...")
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(image_dir).glob(ext))
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 随机打乱
    random.shuffle(image_files)
    
    # 计算划分数量
    total = len(image_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    
    print(f"训练集: {train_count}, 验证集: {val_count}, 测试集: {test_count}")
    
    # 划分数据集
    splits = {
        'train': image_files[:train_count],
        'val': image_files[train_count:train_count + val_count],
        'test': image_files[train_count + val_count:]
    }
    
    # 复制文件
    for split_name, files in splits.items():
        print(f"\n处理 {split_name} 集...")
        for img_path in tqdm(files):
            # 复制图像
            shutil.copy2(
                img_path,
                os.path.join(output_dir, 'images', split_name, img_path.name)
            )
            
            # 复制对应的标签文件（如果有）
            label_name = img_path.stem + '.txt'
            label_path = os.path.join(label_dir, label_name)
            
            if os.path.exists(label_path):
                shutil.copy2(
                    label_path,
                    os.path.join(output_dir, 'labels', split_name, label_name)
                )
    
    print("\n数据集划分完成!")


def create_sample_labels(image_dir, output_dir, classes, num_samples=10):
    """
    创建示例标签文件（用于测试）
    
    Args:
        image_dir: 图像目录
        output_dir: 输出目录
        classes: 类别列表
        num_samples: 创建示例标签的数量
    """
    print(f"创建示例标签文件...")
    
    image_files = list(Path(image_dir).glob('*.jpg'))[:num_samples]
    
    for img_path in tqdm(image_files):
        # 生成随机标注
        label_name = img_path.stem + '.txt'
        label_path = os.path.join(output_dir, label_name)
        
        with open(label_path, 'w') as f:
            # 随机添加1-3个标注
            num_boxes = random.randint(1, 3)
            for _ in range(num_boxes):
                class_id = random.randint(0, len(classes) - 1)
                x_center = random.uniform(0.1, 0.9)
                y_center = random.uniform(0.1, 0.9)
                width = random.uniform(0.1, 0.3)
                height = random.uniform(0.1, 0.3)
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"已创建 {len(image_files)} 个示例标签文件")


def visualize_dataset(dataset_path, output_dir='visualization'):
    """
    可视化数据集标注
    
    Args:
        dataset_path: 数据集路径
        output_dir: 输出目录
    """
    import cv2
    
    print(f"可视化数据集...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取data.yaml获取类别
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    classes = data_config['names']
    
    # 可视化训练集前10张
    train_images = list(Path(os.path.join(dataset_path, 'images', 'train')).glob('*.jpg'))[:10]
    
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
              for _ in range(len(classes))]
    
    for img_path in tqdm(train_images):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        # 读取标签
        label_path = os.path.join(dataset_path, 'labels', 'train', img_path.stem + '.txt')
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # 转换为像素坐标
                        x = int((x_center - width / 2) * w)
                        y = int((y_center - height / 2) * h)
                        x_max = int((x_center + width / 2) * w)
                        y_max = int((y_center + height / 2) * h)
                        
                        # 绘制边界框
                        color = colors[class_id]
                        cv2.rectangle(img, (x, y), (x_max, y_max), color, 2)
                        
                        # 绘制类别标签
                        label = classes[class_id]
                        cv2.putText(img, label, (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 保存可视化结果
        output_path = os.path.join(output_dir, img_path.name)
        cv2.imwrite(output_path, img)
    
    print(f"可视化结果已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='YOLO数据集准备工具')
    parser.add_argument('--mode', type=str, default='create', 
                       choices=['create', 'split', 'visualize'],
                       help='操作模式: create(创建结构), split(划分数据集), visualize(可视化)')
    parser.add_argument('--dataset_path', type=str, default='data/custom_dataset',
                       help='数据集路径')
    parser.add_argument('--classes', type=str, nargs='+', 
                       default=['person', 'car', 'dog', 'cat'],
                       help='类别名称列表')
    parser.add_argument('--image_dir', type=str, help='原始图像目录')
    parser.add_argument('--label_dir', type=str, help='原始标签目录')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例')
    
    args = parser.parse_args()
    
    if args.mode == 'create':
        create_yolo_dataset_structure(args.dataset_path, args.classes)
    elif args.mode == 'split':
        if not args.image_dir or not args.label_dir:
            print("错误: split模式需要指定 --image_dir 和 --label_dir")
            return
        split_dataset(args.image_dir, args.label_dir, args.dataset_path,
                    args.train_ratio, args.val_ratio, args.test_ratio)
    elif args.mode == 'visualize':
        visualize_dataset(args.dataset_path)


if __name__ == '__main__':
    main()
