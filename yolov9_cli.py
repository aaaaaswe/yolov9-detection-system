#!/usr/bin/env python3
"""
YOLOv9 完整命令行工具
整合训练、检测、数据准备等功能
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


class YOLOv9CLI:
    """YOLOv9 命令行工具"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        
    def prepare_dataset(self, args):
        """准备数据集"""
        print("=" * 80)
        print("YOLOv9 数据集准备工具")
        print("=" * 80)
        
        cmd = [
            sys.executable, 'prepare_dataset.py',
            '--mode', args.mode,
            '--dataset_path', args.dataset_path
        ]
        
        if args.classes:
            cmd.extend(['--classes'] + args.classes)
        if args.image_dir:
            cmd.extend(['--image_dir', args.image_dir])
        if args.label_dir:
            cmd.extend(['--label_dir', args.label_dir])
        if args.train_ratio:
            cmd.extend(['--train_ratio', str(args.train_ratio)])
        if args.val_ratio:
            cmd.extend(['--val_ratio', str(args.val_ratio)])
        if args.test_ratio:
            cmd.extend(['--test_ratio', str(args.test_ratio)])
        
        subprocess.run(cmd, check=True)
    
    def train(self, args):
        """训练模型"""
        print("=" * 80)
        print("YOLOv9 训练工具")
        print("=" * 80)
        
        cmd = [
            sys.executable, 'train.py',
            '--mode', 'train'
        ]
        
        if args.data:
            cmd.extend(['--data', args.data])
        if args.model_size:
            cmd.extend(['--model_size', args.model_size])
        if args.weights:
            cmd.extend(['--weights', args.weights])
        if args.epochs:
            cmd.extend(['--epochs', str(args.epochs)])
        if args.batch:
            cmd.extend(['--batch', str(args.batch)])
        if args.imgsz:
            cmd.extend(['--imgsz', str(args.imgsz)])
        if args.optimizer:
            cmd.extend(['--optimizer', args.optimizer])
        if args.lr:
            cmd.extend(['--lr', str(args.lr)])
        if args.patience:
            cmd.extend(['--patience', str(args.patience)])
        if args.device:
            cmd.extend(['--device', args.device])
        if args.project:
            cmd.extend(['--project', args.project])
        if args.name:
            cmd.extend(['--name', args.name])
        
        subprocess.run(cmd, check=True)
    
    def detect(self, args):
        """检测目标"""
        print("=" * 80)
        print("YOLOv9 检测工具")
        print("=" * 80)
        
        cmd = [
            sys.executable, 'detect.py'
        ]
        
        if args.source:
            cmd.extend(['--source', args.source])
        if args.weights:
            cmd.extend(['--weights', args.weights])
        if args.conf:
            cmd.extend(['--conf', str(args.conf)])
        if args.iou:
            cmd.extend(['--iou', str(args.iou)])
        if args.max_det:
            cmd.extend(['--max_det', str(args.max_det)])
        if args.device:
            cmd.extend(['--device', args.device])
        if args.output:
            cmd.extend(['--output', args.output])
        if args.hide:
            cmd.append('--hide')
        if args.save:
            cmd.append('--save')
        
        subprocess.run(cmd, check=True)
    
    def export(self, args):
        """导出模型"""
        print("=" * 80)
        print("YOLOv9 模型导出工具")
        print("=" * 80)
        
        cmd = [
            sys.executable, 'train.py',
            '--mode', 'export',
            '--weights', args.weights,
            '--format', args.format
        ]
        
        subprocess.run(cmd, check=True)
    
    def interactive_mode(self):
        """交互模式"""
        print("=" * 80)
        print("YOLOv9 交互式命令行工具")
        print("=" * 80)
        print()
        
        while True:
            print("\n请选择操作:")
            print("1. 准备数据集")
            print("2. 训练模型")
            print("3. 检测目标")
            print("4. 导出模型")
            print("5. 查看帮助")
            print("0. 退出")
            print()
            
            choice = input("请输入选项 (0-5): ").strip()
            
            if choice == '0':
                print("再见!")
                break
            elif choice == '1':
                self.interactive_prepare_dataset()
            elif choice == '2':
                self.interactive_train()
            elif choice == '3':
                self.interactive_detect()
            elif choice == '4':
                self.interactive_export()
            elif choice == '5':
                self.show_help()
            else:
                print("无效选项，请重新选择!")
    
    def interactive_prepare_dataset(self):
        """交互式数据集准备"""
        print("\n--- 数据集准备 ---")
        
        mode = input("操作模式 (create/split/visualize) [create]: ").strip() or 'create'
        dataset_path = input("数据集路径 [data/custom_dataset]: ").strip() or 'data/custom_dataset'
        
        cmd = [sys.executable, 'prepare_dataset.py', '--mode', mode, '--dataset_path', dataset_path]
        
        if mode == 'create':
            classes = input("类别名称 (空格分隔, 默认: person car dog): ").strip()
            if classes:
                cmd.extend(['--classes'] + classes.split())
        elif mode == 'split':
            image_dir = input("原始图像目录: ").strip()
            label_dir = input("原始标签目录: ").strip()
            cmd.extend(['--image_dir', image_dir, '--label_dir', label_dir])
        
        subprocess.run(cmd)
    
    def interactive_train(self):
        """交互式训练"""
        print("\n--- 训练模型 ---")
        
        data = input("数据集配置文件 [data/custom_dataset/data.yaml]: ").strip() or 'data/custom_dataset/data.yaml'
        model_size = input("模型大小 (n/s/m/l/x) [y]: ").strip() or 'y'
        epochs = input("训练轮数 [100]: ").strip() or '100'
        batch = input("批次大小 [16]: ").strip() or '16'
        
        cmd = [
            sys.executable, 'train.py',
            '--data', data,
            '--model_size', model_size,
            '--epochs', epochs,
            '--batch', batch
        ]
        
        subprocess.run(cmd)
    
    def interactive_detect(self):
        """交互式检测"""
        print("\n--- 检测目标 ---")
        
        source = input("输入源 (摄像头索引/视频/图片路径) [0]: ").strip() or '0'
        weights = input("模型权重 [yolov8s.pt]: ").strip() or 'yolov8s.pt'
        conf = input("置信度阈值 [0.25]: ").strip() or '0.25'
        
        cmd = [
            sys.executable, 'detect.py',
            '--source', source,
            '--weights', weights,
            '--conf', conf
        ]
        
        subprocess.run(cmd)
    
    def interactive_export(self):
        """交互式导出"""
        print("\n--- 导出模型 ---")
        
        weights = input("模型权重路径: ").strip()
        format_type = input("导出格式 [onnx]: ").strip() or 'onnx'
        
        cmd = [
            sys.executable, 'train.py',
            '--mode', 'export',
            '--weights', weights,
            '--format', format_type
        ]
        
        subprocess.run(cmd)
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
        ======== YOLOv9 使用说明 ========
        
        1. 数据集准备:
           - create: 创建YOLO格式的数据集目录结构
           - split: 划分数据集为训练集、验证集、测试集
           - visualize: 可视化数据集标注
        
        2. 训练模型:
           - 支持从头训练和微调
           - 支持断点续训
           - 自动保存最佳模型
        
        3. 检测目标:
           - 支持摄像头实时检测
           - 支持视频文件检测
           - 支持图片和文件夹批量检测
        
        4. 导出模型:
           - 支持导出为 ONNX, TorchScript 等格式
           - 便于部署到其他平台
        
        ================
        """
        print(help_text)


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv9 完整命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互模式
  python yolov9_cli.py
  
  # 准备数据集
  python yolov9_cli.py prepare --mode create --dataset_path data/my_dataset --classes person car
  
  # 训练模型
  python yolov9_cli.py train --data data/my_dataset/data.yaml --model_size y --epochs 100
  
  # 摄像头检测
  python yolov9_cli.py detect --source 0 --weights yolov9y.pt
  
  # 视频检测
  python yolov9_cli.py detect --source video.mp4 --weights yolov9y.pt --output result.mp4
  
  # 图片检测
  python yolov9_cli.py detect --source image.jpg --weights yolov9y.pt --output result.jpg
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 数据集准备子命令
    prepare_parser = subparsers.add_parser('prepare', help='准备数据集')
    prepare_parser.add_argument('--mode', default='create', 
                                choices=['create', 'split', 'visualize'])
    prepare_parser.add_argument('--dataset_path', default='data/custom_dataset')
    prepare_parser.add_argument('--classes', nargs='+')
    prepare_parser.add_argument('--image_dir')
    prepare_parser.add_argument('--label_dir')
    prepare_parser.add_argument('--train_ratio', type=float, default=0.7)
    prepare_parser.add_argument('--val_ratio', type=float, default=0.2)
    prepare_parser.add_argument('--test_ratio', type=float, default=0.1)
    
    # 训练子命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data', default='data/custom_dataset/data.yaml')
    train_parser.add_argument('--model_size', default='y', choices=['n', 's', 'm', 'l', 'x'])
    train_parser.add_argument('--weights')
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--batch', type=int, default=16)
    train_parser.add_argument('--imgsz', type=int, default=640)
    train_parser.add_argument('--optimizer', default='auto')
    train_parser.add_argument('--lr', type=float, default=0.01)
    train_parser.add_argument('--patience', type=int, default=50)
    train_parser.add_argument('--device')
    train_parser.add_argument('--project')
    train_parser.add_argument('--name')
    
    # 检测子命令
    detect_parser = subparsers.add_parser('detect', help='检测目标')
    detect_parser.add_argument('--source')
    detect_parser.add_argument('--weights', default='yolov8s.pt')
    detect_parser.add_argument('--conf', type=float, default=0.25)
    detect_parser.add_argument('--iou', type=float, default=0.45)
    detect_parser.add_argument('--max_det', type=int, default=300)
    detect_parser.add_argument('--device')
    detect_parser.add_argument('--output')
    detect_parser.add_argument('--hide', action='store_true')
    detect_parser.add_argument('--save', action='store_true')
    
    # 导出子命令
    export_parser = subparsers.add_parser('export', help='导出模型')
    export_parser.add_argument('--weights', required=True)
    export_parser.add_argument('--format', default='onnx')
    
    # 无子命令时进入交互模式
    args = parser.parse_args()
    
    cli = YOLOv9CLI()
    
    if not args.command:
        # 交互模式
        cli.interactive_mode()
    elif args.command == 'prepare':
        cli.prepare_dataset(args)
    elif args.command == 'train':
        cli.train(args)
    elif args.command == 'detect':
        cli.detect(args)
    elif args.command == 'export':
        cli.export(args)


if __name__ == '__main__':
    main()
